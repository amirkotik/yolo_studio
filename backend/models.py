from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any

class ProjectCreate(BaseModel):
    name: str = Field(..., description="Unique name of the project")
    description: Optional[str] = ""

class ClassItem(BaseModel):
    id: int
    name: str
    color: Optional[str] = "#3B82F6"

class BoundingBox(BaseModel):
    class_id: int
    x_min: float
    y_min: float
    width: float
    height: float

class AnnotationSaveRequest(BaseModel):
    filename: str
    boxes: List[BoundingBox]
    image_width: Optional[int] = None
    image_height: Optional[int] = None

class AutoAnnotateRequest(BaseModel):
    mode: str = Field("text", description="Auto-annotation mode: 'text' or 'visual'")
    model_name: str = Field("yoloe-26n-seg.pt", description="YOLOE weight file")
    conf_threshold: float = Field(0.25, ge=0.01, le=0.95)
    text_prompts: Optional[List[str]] = []
    reference_image: Optional[str] = ""
    overwrite: bool = Field(False, description="Whether to overwrite existing boxes")

class BatchDeleteRequest(BaseModel):
    filenames: List[str] = Field(..., description="List of filenames to delete")

class GenerateDatasetRequest(BaseModel):
    train_ratio: float = Field(0.8, ge=0.1, le=0.95, description="Proportion of dataset for training (e.g. 0.8 for 80%)")

class TrainRequest(BaseModel):
    model_name: str = Field("yolo26n.pt", description="Base model weight file (e.g. yolo26n.pt, yolo26s.pt)")
    epochs: int = Field(10, ge=1, le=1000)
    batch_size: int = Field(16, ge=1, le=256)
    imgsz: int = Field(640, ge=32, le=2048)
    device: str = Field("cpu", description="Device to train on: 'cpu', '0', '0,1', etc.")
    run_name: Optional[str] = "train_run"

class ExportRequest(BaseModel):
    weights_path: Optional[str] = None
    format: str = Field("onnx", description="Export format: onnx, engine (tensorrt), torchscript, openvino, etc.")

class VideoSliceParams(BaseModel):
    step_mode: str = Field("nth_frame", description="Mode: 'nth_frame' or 'fps'")
    step_value: int = Field(5, ge=1, description="N value for every Nth frame or N frames per second")
