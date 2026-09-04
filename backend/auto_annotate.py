import os
import cv2
import torch
import numpy as np
from PIL import Image
from typing import List
from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEVPDetectPredictor

from backend.config import get_raw_images_dir
from backend.files import list_images
from backend.annotations import get_classes, save_classes, get_annotations, get_image_annotations, save_image_annotations
from backend.models import ClassItem, BoundingBox

def add_class_if_missing(project_name: str, class_name: str) -> ClassItem:
    classes = get_classes(project_name)
    existing = next((c for c in classes if c["name"].lower() == class_name.lower()), None)
    if existing:
        return ClassItem(**existing)
    
    max_id = max([c["id"] for c in classes], default=-1)
    new_id = max_id + 1
    colors = ['#3B82F6', '#EF4444', '#10B981', '#F59E0B', '#8B5CF6', '#EC4899', '#06B6D4', '#14B8A6']
    color = colors[new_id % len(colors)]
    
    new_cls = ClassItem(id=new_id, name=class_name, color=color)
    class_items = [ClassItem(**c) for c in classes] + [new_cls]
    save_classes(project_name, class_items)
    return new_cls

def run_auto_annotate(
    project_name: str,
    mode: str,
    model_name: str,
    conf_threshold: float,
    text_prompts: list,
    reference_image: str,
    overwrite: bool
) -> dict:
    raw_dir = get_raw_images_dir(project_name)
    image_files = [item["filename"] for item in list_images(project_name)]
    classes = get_classes(project_name)
    annotations = get_annotations(project_name)
    
    if not image_files:
        raise ValueError("В проекте нет изображений для разметки.")

    # Determine device
    if torch.cuda.is_available():
        device = 0
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"[AutoAnnotate] Загрузка модели YOLOE '{model_name}' на устройстве {device}...")
    model = YOLOE(model_name)

    mapped_class_ids = []
    class_names_list = []

    if mode == "text":
        prompts_to_use = text_prompts if text_prompts else [c["name"] for c in classes]
        if not prompts_to_use:
            raise ValueError("Укажите хотя бы один класс или слово для текстовой разметки.")
        
        for text in prompts_to_use:
            cls_item = add_class_if_missing(project_name, text.strip())
            mapped_class_ids.append(cls_item.id)
            class_names_list.append(cls_item.name)
        
        print(f"[AutoAnnotate] Режим Text Prompting: {class_names_list}")
        model.set_classes(class_names_list)

    elif mode == "visual":
        if not reference_image:
            raise ValueError("Выберите эталонный кадр (референс) для визуальной разметки.")
        
        ref_img_path = os.path.join(raw_dir, reference_image)
        if not os.path.exists(ref_img_path):
            raise ValueError(f"Файл референса {reference_image} не найден.")
        
        ref_boxes = get_image_annotations(project_name, reference_image)
        if not ref_boxes:
            raise ValueError(f"На эталонном кадре '{reference_image}' нет размеченных рамок! Разметьте хотя бы 1 объект.")
        
        unique_class_ids = []
        for b in ref_boxes:
            if b["class_id"] not in unique_class_ids:
                unique_class_ids.append(b["class_id"])
        
        for cid in unique_class_ids:
            cls_obj = next((c for c in classes if c["id"] == cid), None)
            name = cls_obj["name"] if cls_obj else f"class_{cid}"
            mapped_class_ids.append(cid)
            class_names_list.append(name)
        
        bboxes_list = []
        cls_indices_list = []
        for b in ref_boxes:
            x1 = float(b["x_min"])
            y1 = float(b["y_min"])
            x2 = float(b["x_min"] + b["width"])
            y2 = float(b["y_min"] + b["height"])
            idx = mapped_class_ids.index(b["class_id"])
            bboxes_list.append([x1, y1, x2, y2])
            cls_indices_list.append(idx)
        
        ref_img_data = cv2.imread(ref_img_path)
        if ref_img_data is None:
            ref_img_data = np.array(Image.open(ref_img_path).convert("RGB"))[:, :, ::-1]

        prompts = {"bboxes": bboxes_list, "cls": cls_indices_list}
        print(f"[AutoAnnotate] Режим Visual Prompting: извлечение VPE из {reference_image} ({len(ref_boxes)} рамок)...")
        
        predictor = YOLOEVPDetectPredictor(overrides={
            "task": model.model.task,
            "mode": "predict",
            "save": False,
            "verbose": False,
            "batch": 1,
            "device": device,
            "imgsz": 640
        })
        predictor.setup_model(model=model.model)
        predictor.set_prompts(prompts)
        vpe = predictor.get_vpe(ref_img_data)
        model.model.set_classes(class_names_list, vpe)
    else:
        raise ValueError(f"Неизвестный режим авторазметки: {mode}")

    annotated_count = 0
    total_boxes = 0

    print(f"[AutoAnnotate] Старт прохода по {len(image_files)} кадрам (порог {conf_threshold:.2f}, overwrite={overwrite})...")
    for img_name in image_files:
        if mode == "visual" and img_name == reference_image:
            continue
        
        existing_boxes = get_image_annotations(project_name, img_name)
        if not overwrite and existing_boxes:
            continue
        
        img_path = os.path.join(raw_dir, img_name)
        if not os.path.exists(img_path):
            continue
        
        results = model.predict(img_path, conf=conf_threshold, verbose=False)
        new_boxes = []
        if len(results) > 0 and results[0].boxes is not None:
            for box in results[0].boxes:
                conf = float(box.conf[0].item())
                if conf < conf_threshold:
                    continue
                xyxy = box.xyxy[0].tolist()
                cls_idx = int(box.cls[0].item())
                if cls_idx < len(mapped_class_ids):
                    class_id = mapped_class_ids[cls_idx]
                elif len(mapped_class_ids) > 0:
                    class_id = mapped_class_ids[0]
                else:
                    continue
                
                x_min = xyxy[0]
                y_min = xyxy[1]
                w = xyxy[2] - xyxy[0]
                h = xyxy[3] - xyxy[1]
                if w <= 1 or h <= 1:
                    continue
                
                new_boxes.append(BoundingBox(
                    class_id=class_id,
                    x_min=round(x_min, 1),
                    y_min=round(y_min, 1),
                    width=round(w, 1),
                    height=round(h, 1)
                ))
        
        save_image_annotations(project_name, img_name, new_boxes)
        annotations[img_name] = [b.model_dump() for b in new_boxes]
        
        annotated_count += 1
        total_boxes += len(new_boxes)

    print(f"[AutoAnnotate] Готово! Размечено кадров: {annotated_count}, добавлено рамок: {total_boxes}")
    return {
        "status": "success",
        "annotated_count": annotated_count,
        "total_boxes": total_boxes,
        "classes_used": class_names_list
    }
