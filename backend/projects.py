import os
import json
import shutil
from typing import List, Dict, Any
from .config import (
    WORKSPACE_DIR, get_project_dir, get_raw_images_dir,
    get_classes_file, get_annotations_file, get_dataset_dir, get_runs_dir
)

def list_projects() -> List[Dict[str, Any]]:
    if not os.path.exists(WORKSPACE_DIR):
        return []
    
    projects = []
    for entry in sorted(os.listdir(WORKSPACE_DIR)):
        proj_path = os.path.join(WORKSPACE_DIR, entry)
        if os.path.isdir(proj_path):
            info = get_project_info(entry)
            if info:
                projects.append(info)
    return projects

def get_project_info(project_name: str) -> Dict[str, Any]:
    proj_dir = get_project_dir(project_name)
    if not os.path.exists(proj_dir):
        return {}
    
    raw_images_dir = get_raw_images_dir(project_name)
    classes_file = get_classes_file(project_name)
    annotations_file = get_annotations_file(project_name)
    dataset_dir = get_dataset_dir(project_name)
    runs_dir = get_runs_dir(project_name)
    
    # Count images
    images_count = 0
    if os.path.exists(raw_images_dir):
        valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        images_count = len([
            f for f in os.listdir(raw_images_dir)
            if os.path.splitext(f.lower())[1] in valid_exts
        ])
        
    # Count classes
    classes_count = 0
    if os.path.exists(classes_file):
        try:
            with open(classes_file, 'r', encoding='utf-8') as f:
                classes = json.load(f)
                classes_count = len(classes)
        except Exception:
            pass
            
    # Count annotated images
    annotated_count = 0
    total_boxes = 0
    if os.path.exists(annotations_file):
        try:
            with open(annotations_file, 'r', encoding='utf-8') as f:
                annotations = json.load(f)
                for img, boxes in annotations.items():
                    if boxes:
                        annotated_count += 1
                        total_boxes += len(boxes)
        except Exception:
            pass
            
    has_dataset = os.path.exists(os.path.join(dataset_dir, "dataset.yaml"))
    
    runs_list = []
    if os.path.exists(runs_dir):
        for r in sorted(os.listdir(runs_dir)):
            r_path = os.path.join(runs_dir, r)
            if os.path.isdir(r_path):
                weights_dir = os.path.join(r_path, "weights")
                best_pt = os.path.join(weights_dir, "best.pt")
                runs_list.append({
                    "name": r,
                    "has_best_model": os.path.exists(best_pt),
                    "path": r_path
                })
                
    return {
        "name": project_name,
        "images_count": images_count,
        "classes_count": classes_count,
        "annotated_images_count": annotated_count,
        "total_boxes_count": total_boxes,
        "has_dataset": has_dataset,
        "runs_count": len(runs_list),
        "runs": runs_list
    }

def create_project(project_name: str, description: str = "") -> Dict[str, Any]:
    proj_dir = get_project_dir(project_name)
    if os.path.exists(proj_dir):
        raise ValueError(f"Project '{project_name}' already exists.")
        
    os.makedirs(get_raw_images_dir(project_name), exist_ok=True)
    os.makedirs(get_dataset_dir(project_name), exist_ok=True)
    os.makedirs(get_runs_dir(project_name), exist_ok=True)
    
    # Initialize default classes.json
    default_classes = [{"id": 0, "name": "object", "color": "#3B82F6"}]
    with open(get_classes_file(project_name), 'w', encoding='utf-8') as f:
        json.dump(default_classes, f, indent=2, ensure_ascii=False)
        
    # Initialize default annotations.json
    with open(get_annotations_file(project_name), 'w', encoding='utf-8') as f:
        json.dump({}, f, indent=2, ensure_ascii=False)
        
    return get_project_info(project_name)

def delete_project(project_name: str) -> bool:
    proj_dir = get_project_dir(project_name)
    if os.path.exists(proj_dir):
        shutil.rmtree(proj_dir)
        return True
    return False
