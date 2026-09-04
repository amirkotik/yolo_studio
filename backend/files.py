import os
import cv2
import json
from PIL import Image
from typing import List, Dict, Any, Tuple
from fastapi import HTTPException, UploadFile
from .config import get_raw_images_dir, get_annotations_file

VALID_IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
VALID_VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}

def list_images(project_name: str) -> List[Dict[str, Any]]:
    raw_dir = get_raw_images_dir(project_name)
    if not os.path.exists(raw_dir):
        return []
        
    annotations_file = get_annotations_file(project_name)
    annotations = {}
    if os.path.exists(annotations_file):
        try:
            with open(annotations_file, 'r', encoding='utf-8') as f:
                annotations = json.load(f)
        except Exception:
            pass
            
    images = []
    for fname in sorted(os.listdir(raw_dir)):
        ext = os.path.splitext(fname.lower())[1]
        if ext in VALID_IMAGE_EXTS:
            fpath = os.path.join(raw_dir, fname)
            width, height = 0, 0
            try:
                with Image.open(fpath) as img:
                    width, height = img.size
            except Exception:
                pass
                
            boxes = annotations.get(fname, [])
            images.append({
                "filename": fname,
                "width": width,
                "height": height,
                "annotated": len(boxes) > 0,
                "boxes_count": len(boxes),
                "url": f"/api/{project_name}/raw/{fname}"
            })
    return images

async def save_uploaded_file(project_name: str, file: UploadFile, step_mode: str = "nth_frame", step_value: int = 5) -> Dict[str, Any]:
    raw_dir = get_raw_images_dir(project_name)
    if not os.path.exists(raw_dir):
        os.makedirs(raw_dir, exist_ok=True)
        
    filename = file.filename or "upload.tmp"
    ext = os.path.splitext(filename.lower())[1]
    
    if ext not in VALID_IMAGE_EXTS and ext not in VALID_VIDEO_EXTS:
        raise HTTPException(status_code=400, detail=f"Unsupported file format: {ext}. Supported formats: {VALID_IMAGE_EXTS | VALID_VIDEO_EXTS}")
        
    # Save temporary or direct file
    dest_path = os.path.join(raw_dir, filename)
    
    # Handle filename collision
    base, extension = os.path.splitext(filename)
    counter = 1
    while os.path.exists(dest_path):
        filename = f"{base}_{counter}{extension}"
        dest_path = os.path.join(raw_dir, filename)
        counter += 1
        
    try:
        content = await file.read()
        with open(dest_path, "wb") as f:
            f.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {str(e)}")
        
    if ext in VALID_IMAGE_EXTS:
        # Validate image
        try:
            with Image.open(dest_path) as img:
                img.verify()
        except Exception as e:
            if os.path.exists(dest_path):
                os.remove(dest_path)
            raise HTTPException(status_code=400, detail=f"Corrupt or invalid image file: {str(e)}")
            
        return {"status": "success", "type": "image", "saved_files": [filename], "count": 1}
        
    elif ext in VALID_VIDEO_EXTS:
        # Video slicing using OpenCV
        saved_frames = []
        try:
            cap = cv2.VideoCapture(dest_path)
            if not cap.isOpened():
                raise ValueError("OpenCV could not open video stream. Video file might be corrupt or codec unsupported.")
                
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0 or fps != fps: # NaN check
                fps = 30.0
                
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Calculate frame interval
            if step_mode == "fps" or step_mode == "sec":
                # step_value is e.g. 1 frame per second -> interval is fps / step_value
                interval = max(1, int(round(fps / max(1, step_value))))
            else:
                # nth_frame -> save every step_value frames
                interval = max(1, int(step_value))
                
            frame_idx = 0
            saved_count = 0
            video_base = os.path.splitext(filename)[0]
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                if frame_idx % interval == 0:
                    frame_filename = f"{video_base}_frame_{frame_idx:06d}.jpg"
                    frame_path = os.path.join(raw_dir, frame_filename)
                    
                    # Prevent collision
                    f_counter = 1
                    while os.path.exists(frame_path):
                        frame_filename = f"{video_base}_frame_{frame_idx:06d}_{f_counter}.jpg"
                        frame_path = os.path.join(raw_dir, frame_filename)
                        f_counter += 1
                        
                    cv2.imwrite(frame_path, frame)
                    saved_frames.append(frame_filename)
                    saved_count += 1
                    
                frame_idx += 1
                
            cap.release()
            
            # Remove original video file after slicing
            if os.path.exists(dest_path):
                os.remove(dest_path)
                
            if saved_count == 0:
                raise ValueError("No frames could be extracted from the video.")
                
            return {"status": "success", "type": "video", "saved_files": saved_frames, "count": saved_count}
            
        except Exception as e:
            # Clean up video file and any extracted frames on failure
            if os.path.exists(dest_path):
                try:
                    os.remove(dest_path)
                except Exception:
                    pass
            for f in saved_frames:
                fpath = os.path.join(raw_dir, f)
                if os.path.exists(fpath):
                    try:
                        os.remove(fpath)
                    except Exception:
                        pass
            raise HTTPException(status_code=400, detail=f"Video slicing error: {str(e)}")

def delete_image_file(project_name: str, filename: str) -> bool:
    raw_dir = get_raw_images_dir(project_name)
    fpath = os.path.join(raw_dir, filename)
    if os.path.exists(fpath):
        os.remove(fpath)
        
        # Remove from annotations
        annotations_file = get_annotations_file(project_name)
        if os.path.exists(annotations_file):
            try:
                with open(annotations_file, 'r', encoding='utf-8') as f:
                    annotations = json.load(f)
                if filename in annotations:
                    del annotations[filename]
                    with open(annotations_file, 'w', encoding='utf-8') as f:
                        json.dump(annotations, f, indent=2, ensure_ascii=False)
            except Exception:
                pass
        return True
    return False

def delete_image_files_batch(project_name: str, filenames: List[str]) -> Dict[str, Any]:
    raw_dir = get_raw_images_dir(project_name)
    deleted_count = 0
    
    for filename in filenames:
        fpath = os.path.join(raw_dir, filename)
        if os.path.exists(fpath):
            try:
                os.remove(fpath)
                deleted_count += 1
            except Exception:
                pass
                
    annotations_file = get_annotations_file(project_name)
    if os.path.exists(annotations_file):
        try:
            with open(annotations_file, 'r', encoding='utf-8') as f:
                annotations = json.load(f)
            modified = False
            for filename in filenames:
                if filename in annotations:
                    del annotations[filename]
                    modified = True
            if modified:
                with open(annotations_file, 'w', encoding='utf-8') as f:
                    json.dump(annotations, f, indent=2, ensure_ascii=False)
        except Exception:
            pass
            
    return {"status": "success", "deleted_count": deleted_count}
