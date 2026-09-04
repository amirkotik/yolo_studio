import os
import shutil
import asyncio
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, WebSocket, WebSocketDisconnect, Query, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from backend.config import WORKSPACE_DIR, get_project_dir, get_raw_images_dir
from backend.models import (
    ProjectCreate, ClassItem, BoundingBox, AnnotationSaveRequest,
    AutoAnnotateRequest, BatchDeleteRequest, GenerateDatasetRequest, TrainRequest, ExportRequest
)
from backend.projects import list_projects, create_project, delete_project, get_project_info
from backend.files import list_images, save_uploaded_file, delete_image_file, delete_image_files_batch
from backend.annotations import get_classes, save_classes, get_annotations, get_image_annotations, save_image_annotations
from backend.dataset import generate_yolo_dataset
from backend.train import start_training, stop_training, get_training_status, register_websocket, unregister_websocket, broadcast_log_messages
from backend.export import list_trained_models, export_model
from backend.auto_annotate import run_auto_annotate

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: launch background log broadcaster
    broadcaster_task = asyncio.create_task(broadcast_log_messages())
    yield
    # Shutdown
    broadcaster_task.cancel()

app = FastAPI(title="TeachYOLO - Agentic AI YOLO Studio", version="1.0.0", lifespan=lifespan)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- REST API Endpoints ---

@app.get("/api/projects", summary="List all projects")
async def api_list_projects():
    try:
        return list_projects()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/projects", summary="Create a new project")
async def api_create_project(req: ProjectCreate):
    try:
        return create_project(req.name, req.description)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/projects/import_zip", summary="Import a project from a ZIP archive")
async def api_import_project_zip(file: UploadFile = File(...)):
    import zipfile, tempfile
    try:
        base_name = os.path.splitext(file.filename)[0]
        project_name = "".join(c for c in base_name if c.isalnum() or c in ("_", "-")).strip()
        if not project_name:
            project_name = "imported_project"
            
        target_dir = get_project_dir(project_name)
        if os.path.exists(target_dir):
            counter = 1
            while os.path.exists(get_project_dir(f"{project_name}_{counter}")):
                counter += 1
            project_name = f"{project_name}_{counter}"
            target_dir = get_project_dir(project_name)
            
        os.makedirs(target_dir, exist_ok=True)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
            
        with zipfile.ZipFile(tmp_path, 'r') as zf:
            zf.extractall(target_dir)
        os.unlink(tmp_path)
        
        os.makedirs(os.path.join(target_dir, "raw_images"), exist_ok=True)
        if not os.path.exists(os.path.join(target_dir, "classes.json")):
            with open(os.path.join(target_dir, "classes.json"), "w", encoding="utf-8") as f:
                f.write("[]")
        if not os.path.exists(os.path.join(target_dir, "annotations.json")):
            with open(os.path.join(target_dir, "annotations.json"), "w", encoding="utf-8") as f:
                f.write("{}")
                
        return get_project_info(project_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to import project: {str(e)}")

@app.get("/api/projects/{project}/export_zip", summary="Export project as ZIP archive")
async def api_export_project_zip(project: str):
    proj_dir = get_project_dir(project)
    if not os.path.exists(proj_dir):
        raise HTTPException(status_code=404, detail=f"Project '{project}' not found.")
    
    zip_path = f"{proj_dir}.zip"
    if os.path.exists(zip_path):
        try:
            os.unlink(zip_path)
        except Exception:
            pass
    shutil.make_archive(proj_dir, 'zip', proj_dir)
    return FileResponse(zip_path, filename=f"{project}.zip")

@app.get("/api/projects/{project}", summary="Get project info")
async def api_get_project(project: str):
    info = get_project_info(project)
    if not info:
        raise HTTPException(status_code=404, detail=f"Project '{project}' not found.")
    return info

@app.delete("/api/projects/{project}", summary="Delete a project")
async def api_delete_project(project: str):
    success = delete_project(project)
    if not success:
        raise HTTPException(status_code=404, detail=f"Project '{project}' not found.")
    return {"status": "success", "message": f"Project '{project}' deleted."}

@app.post("/api/{project}/upload", summary="Upload media files (photos/videos)")
async def api_upload_file(
    project: str,
    file: UploadFile = File(...),
    step_mode: str = Form("nth_frame"),
    step_value: int = Form(5)
):
    if not os.path.exists(get_project_dir(project)):
        raise HTTPException(status_code=404, detail=f"Project '{project}' not found.")
    return await save_uploaded_file(project, file, step_mode, step_value)

@app.get("/api/{project}/images", summary="List images in project")
async def api_list_images(project: str):
    if not os.path.exists(get_project_dir(project)):
        raise HTTPException(status_code=404, detail=f"Project '{project}' not found.")
    return list_images(project)

@app.delete("/api/{project}/images/{filename}", summary="Delete an image")
async def api_delete_image(project: str, filename: str):
    if not delete_image_file(project, filename):
        raise HTTPException(status_code=404, detail="Image not found.")
    return {"status": "success", "message": f"Deleted {filename}"}

@app.post("/api/{project}/images/batch_delete", summary="Batch delete images")
async def api_delete_images_batch(project: str, req: BatchDeleteRequest):
    return delete_image_files_batch(project, req.filenames)

@app.post("/api/{project}/images/delete_unannotated", summary="Delete all unannotated images")
async def api_delete_unannotated_images(project: str):
    images = list_images(project)
    unannotated = [img["filename"] for img in images if not img.get("annotated")]
    if not unannotated:
        return {"status": "success", "deleted_count": 0, "message": "Все кадры уже размечены!"}
    res = delete_image_files_batch(project, unannotated)
    res["message"] = f"Удалено неразмеченных кадров: {res['deleted_count']}"
    return res

@app.get("/api/{project}/raw/{filename}", summary="Serve raw image file")
async def api_serve_raw_image(project: str, filename: str):
    raw_dir = get_raw_images_dir(project)
    fpath = os.path.join(raw_dir, filename)
    if not os.path.exists(fpath):
        raise HTTPException(status_code=404, detail="Image file not found.")
    return FileResponse(fpath)

@app.get("/api/{project}/classes", summary="Get classes list")
async def api_get_classes(project: str):
    return get_classes(project)

@app.post("/api/{project}/classes", summary="Save classes list")
async def api_save_classes(project: str, classes: List[ClassItem]):
    return save_classes(project, classes)

@app.get("/api/{project}/annotations", summary="Get all annotations")
async def api_get_annotations(project: str):
    return get_annotations(project)

@app.post("/api/{project}/annotations", summary="Save annotations for an image")
async def api_save_annotations(project: str, req: AnnotationSaveRequest):
    return save_image_annotations(project, req.filename, req.boxes)

@app.post("/api/{project}/auto_annotate", summary="Run YOLOE auto-annotation")
async def api_auto_annotate(project: str, req: AutoAnnotateRequest):
    try:
        res = await asyncio.to_thread(
            run_auto_annotate,
            project,
            req.mode,
            req.model_name,
            req.conf_threshold,
            req.text_prompts or [],
            req.reference_image or "",
            req.overwrite
        )
        return res
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Auto-annotation failed: {str(e)}")

@app.post("/api/{project}/generate", summary="Assemble YOLO dataset")
async def api_generate_dataset(project: str, req: GenerateDatasetRequest):
    try:
        return generate_yolo_dataset(project, req.train_ratio)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Dataset generation failed: {str(e)}")

@app.post("/api/{project}/train", summary="Start YOLO training")
async def api_start_training(project: str, req: TrainRequest):
    try:
        return start_training(project, req)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start training: {str(e)}")

@app.post("/api/{project}/train/stop", summary="Stop/cancel active training")
async def api_stop_training(project: str):
    try:
        return stop_training(project)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/{project}/train/status", summary="Check training status")
async def api_train_status(project: str):
    return get_training_status()

@app.get("/api/{project}/models", summary="List trained models and runs")
async def api_list_models(project: str):
    return list_trained_models(project)

@app.post("/api/{project}/export", summary="Export model to ONNX/TensorRT/etc.")
async def api_export_model(project: str, req: ExportRequest):
    try:
        return export_model(project, req)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/{project}/download", summary="Download file")
async def api_download_file(project: str, file_path: str = Query(...)):
    proj_dir = os.path.abspath(get_project_dir(project))
    abs_path = os.path.abspath(file_path)
    
    # Security check: ensure path is within workspace or temporary exports
    if not os.path.exists(abs_path):
        raise HTTPException(status_code=404, detail="File not found.")
        
    if os.path.isfile(abs_path):
        return FileResponse(abs_path, filename=os.path.basename(abs_path))
    elif os.path.isdir(abs_path):
        # Zip directory on the fly
        shutil.make_archive(abs_path, 'zip', abs_path)
        zip_path = f"{abs_path}.zip"
        return FileResponse(zip_path, filename=f"{os.path.basename(abs_path)}.zip")
    else:
        raise HTTPException(status_code=400, detail="Invalid path for download.")

# --- WebSockets ---

@app.websocket("/ws/logs")
async def websocket_logs(websocket: WebSocket):
    await websocket.accept()
    register_websocket(websocket)
    try:
        # Send initial status
        status = get_training_status()
        await websocket.send_json({"type": "status", **status})
        while True:
            # Keep alive / handle client messages if any
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        unregister_websocket(websocket)
    except Exception:
        unregister_websocket(websocket)

# --- Static Frontend Serving ---
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
