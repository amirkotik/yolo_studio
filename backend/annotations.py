import os
import json
from typing import List, Dict, Any
from .config import get_classes_file, get_annotations_file
from .models import ClassItem, BoundingBox

def get_classes(project_name: str) -> List[Dict[str, Any]]:
    classes_file = get_classes_file(project_name)
    if not os.path.exists(classes_file):
        return []
    try:
        with open(classes_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return []

def save_classes(project_name: str, classes: List[ClassItem]) -> List[Dict[str, Any]]:
    classes_file = get_classes_file(project_name)
    data = [c.model_dump() for c in classes]
    with open(classes_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return data

def get_annotations(project_name: str) -> Dict[str, Any]:
    annotations_file = get_annotations_file(project_name)
    if not os.path.exists(annotations_file):
        return {}
    try:
        with open(annotations_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}

def get_image_annotations(project_name: str, filename: str) -> List[Dict[str, Any]]:
    annotations = get_annotations(project_name)
    return annotations.get(filename, [])

def save_image_annotations(project_name: str, filename: str, boxes: List[BoundingBox]) -> Dict[str, Any]:
    annotations_file = get_annotations_file(project_name)
    annotations = get_annotations(project_name)
    
    # Update entry for this filename
    annotations[filename] = [b.model_dump() for b in boxes]
    
    with open(annotations_file, 'w', encoding='utf-8') as f:
        json.dump(annotations, f, indent=2, ensure_ascii=False)
        
    return {"status": "success", "filename": filename, "boxes_count": len(boxes)}
