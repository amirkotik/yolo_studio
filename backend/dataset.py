import os
import shutil
import random
import yaml
from PIL import Image
from typing import Dict, Any, List
from .config import get_raw_images_dir, get_dataset_dir, get_classes_file, get_annotations_file
from .annotations import get_classes, get_annotations

VALID_IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

def generate_yolo_dataset(project_name: str, train_ratio: float = 0.8) -> Dict[str, Any]:
    raw_dir = get_raw_images_dir(project_name)
    dataset_dir = get_dataset_dir(project_name)
    
    if not os.path.exists(raw_dir):
        raise ValueError("No raw images found for this project.")
        
    # Get all valid images
    images = [f for f in sorted(os.listdir(raw_dir)) if os.path.splitext(f.lower())[1] in VALID_IMAGE_EXTS]
    if not images:
        raise ValueError("No valid image files found in raw_images directory.")
        
    # Idempotency: recreate dataset directory from scratch
    if os.path.exists(dataset_dir):
        shutil.rmtree(dataset_dir)
        
    images_train_dir = os.path.join(dataset_dir, "images", "train")
    images_val_dir = os.path.join(dataset_dir, "images", "val")
    labels_train_dir = os.path.join(dataset_dir, "labels", "train")
    labels_val_dir = os.path.join(dataset_dir, "labels", "val")
    
    os.makedirs(images_train_dir, exist_ok=True)
    os.makedirs(images_val_dir, exist_ok=True)
    os.makedirs(labels_train_dir, exist_ok=True)
    os.makedirs(labels_val_dir, exist_ok=True)
    
    # Shuffle images deterministically for reproducible split
    shuffled_images = list(images)
    random.seed(42)
    random.shuffle(shuffled_images)
    
    train_count = int(round(len(shuffled_images) * train_ratio))
    # Ensure at least 1 train and 1 val if we have >= 2 images
    if len(shuffled_images) >= 2:
        train_count = max(1, min(len(shuffled_images) - 1, train_count))
    else:
        train_count = len(shuffled_images)
        
    train_images = set(shuffled_images[:train_count])
    
    annotations = get_annotations(project_name)
    classes = get_classes(project_name)
    
    # Prepare sequential class mapping (YOLO requires 0..N-1 indices)
    sorted_classes = sorted(classes, key=lambda x: x["id"])
    names_dict = {}
    id_to_seq = {}
    for idx, c in enumerate(sorted_classes):
        names_dict[idx] = c["name"]
        id_to_seq[c["id"]] = idx
    if not names_dict:
        names_dict = {0: "object"}
        id_to_seq = {0: 0}
        
    stats = {
        "total_images": len(images),
        "train_images": 0,
        "val_images": 0,
        "train_labels": 0,
        "val_labels": 0,
        "background_images": 0
    }
    
    for fname in images:
        src_path = os.path.join(raw_dir, fname)
        is_train = fname in train_images
        
        target_img_dir = images_train_dir if is_train else images_val_dir
        target_lbl_dir = labels_train_dir if is_train else labels_val_dir
        
        # Copy image
        dest_img_path = os.path.join(target_img_dir, fname)
        shutil.copy2(src_path, dest_img_path)
        
        if is_train:
            stats["train_images"] += 1
        else:
            stats["val_images"] += 1
            
        # Get image dimensions
        width, height = 0, 0
        try:
            with Image.open(src_path) as img:
                width, height = img.size
        except Exception:
            continue
            
        if width <= 0 or height <= 0:
            continue
            
        # Process annotations
        boxes = annotations.get(fname, [])
        if not boxes:
            # Background image: do NOT create a .txt file
            stats["background_images"] += 1
            continue
            
        # Normalization to YOLO format
        label_lines = []
        for b in boxes:
            raw_cid = b.get("class_id", 0)
            cid = id_to_seq.get(raw_cid, 0)
            xmin = float(b.get("x_min", 0.0))
            ymin = float(b.get("y_min", 0.0))
            box_w = float(b.get("width", 0.0))
            box_h = float(b.get("height", 0.0))
            
            x_center = (xmin + box_w / 2.0) / width
            y_center = (ymin + box_h / 2.0) / height
            norm_w = box_w / width
            norm_h = box_h / height
            
            # Clamp values to [0.0, 1.0]
            x_center = max(0.0, min(1.0, x_center))
            y_center = max(0.0, min(1.0, y_center))
            norm_w = max(0.0, min(1.0, norm_w))
            norm_h = max(0.0, min(1.0, norm_h))
            
            label_lines.append(f"{cid} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")
            
        if label_lines:
            base_name = os.path.splitext(fname)[0]
            label_path = os.path.join(target_lbl_dir, f"{base_name}.txt")
            with open(label_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(label_lines) + "\n")
                
            if is_train:
                stats["train_labels"] += 1
            else:
                stats["val_labels"] += 1
        else:
            stats["background_images"] += 1
            
    # Create dataset.yaml with absolute paths
    abs_dataset_dir = os.path.abspath(dataset_dir)
    yaml_content = {
        "path": abs_dataset_dir,
        "train": os.path.join(abs_dataset_dir, "images", "train"),
        "val": os.path.join(abs_dataset_dir, "images", "val"),
        "names": names_dict
    }
    
    yaml_path = os.path.join(dataset_dir, "dataset.yaml")
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(yaml_content, f, default_flow_style=False, allow_unicode=True)
        
    stats["yaml_path"] = yaml_path
    return stats
