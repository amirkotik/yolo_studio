import os
from typing import Dict, Any, List, Optional
from fastapi import HTTPException
from .config import get_runs_dir
from .models import ExportRequest

def list_trained_models(project_name: str) -> List[Dict[str, Any]]:
    runs_dir = get_runs_dir(project_name)
    if not os.path.exists(runs_dir):
        return []
        
    models = []
    for r in sorted(os.listdir(runs_dir)):
        r_path = os.path.join(runs_dir, r)
        if os.path.isdir(r_path):
            weights_dir = os.path.join(r_path, "weights")
            if os.path.exists(weights_dir):
                for w in sorted(os.listdir(weights_dir)):
                    w_path = os.path.join(weights_dir, w)
                    if os.path.isfile(w_path):
                        size_mb = round(os.path.getsize(w_path) / (1024 * 1024), 2)
                        models.append({
                            "run_name": r,
                            "filename": w,
                            "path": w_path,
                            "size_mb": size_mb,
                            "is_best": w == "best.pt",
                            "download_url": f"/api/{project_name}/download?file_path={w_path}"
                        })
    return models

def export_model(project_name: str, req: ExportRequest) -> Dict[str, Any]:
    weights_path = req.weights_path
    
    # If not specified, find the latest best.pt
    if not weights_path:
        runs_dir = get_runs_dir(project_name)
        if not os.path.exists(runs_dir):
            raise ValueError("No training runs found for this project.")
            
        # Find best.pt in latest run
        latest_best = None
        for r in sorted(os.listdir(runs_dir), reverse=True):
            bp = os.path.join(runs_dir, r, "weights", "best.pt")
            if os.path.exists(bp):
                latest_best = bp
                break
        if not latest_best:
            raise ValueError("Could not find 'best.pt' in any training runs. Please train a model first or specify weights_path.")
        weights_path = latest_best
        
    if not os.path.exists(weights_path):
        raise ValueError(f"Weights file not found at: {weights_path}")
        
    try:
        from ultralytics import YOLO
        import torch
        
        print(f"Loading weights for export: {weights_path}")
        model = YOLO(weights_path)
        
        print(f"Exporting model to format: {req.format}")
        exported_path = model.export(format=req.format)
        
        # Free VRAM
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        if not exported_path or not os.path.exists(str(exported_path)):
            # Determine expected extension if export didn't return string
            ext_map = {
                "onnx": ".onnx",
                "engine": ".engine",
                "torchscript": ".torchscript",
                "openvino": "_openvino_model",
                "coreml": ".mlpackage",
                "pb": ".pb"
            }
            expected_ext = ext_map.get(req.format, f".{req.format}")
            base_p = os.path.splitext(weights_path)[0]
            exported_path = f"{base_p}{expected_ext}"
            
        size_mb = 0
        if os.path.exists(str(exported_path)):
            if os.path.isfile(str(exported_path)):
                size_mb = round(os.path.getsize(str(exported_path)) / (1024 * 1024), 2)
            else:
                # Directory (e.g., openvino / coreml)
                total_size = sum(os.path.getsize(os.path.join(dirpath, filename)) for dirpath, _, filenames in os.walk(str(exported_path)) for filename in filenames)
                size_mb = round(total_size / (1024 * 1024), 2)
                
        return {
            "status": "success",
            "format": req.format,
            "weights_used": weights_path,
            "exported_path": str(exported_path),
            "size_mb": size_mb,
            "download_url": f"/api/{project_name}/download?file_path={str(exported_path)}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model export failed: {str(e)}")
