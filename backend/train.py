import os
import sys
import time
import json
import traceback
import multiprocessing
from datetime import datetime
from typing import Dict, Any, Optional, List
from .config import get_dataset_dir, get_runs_dir
from .models import TrainRequest

# Global state for training manager
_active_process: Optional[multiprocessing.Process] = None
_active_project: Optional[str] = None
_log_queue: Optional[multiprocessing.Queue] = None
_connected_websockets: List[Any] = []

class QueueWriter:
    def __init__(self, queue, project_name, stream_name="stdout"):
        self.queue = queue
        self.project_name = project_name
        self.stream_name = stream_name
        self.buffer = ""

    def write(self, text):
        if not text:
            return
        self.buffer += text
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            line = line.strip("\r")
            if line:
                try:
                    self.queue.put({
                        "type": "log",
                        "stream": self.stream_name,
                        "project": self.project_name,
                        "message": line,
                        "timestamp": datetime.now().strftime("%H:%M:%S")
                    })
                except Exception:
                    pass

    def flush(self):
        if self.buffer:
            line = self.buffer.strip("\r\n")
            if line:
                try:
                    self.queue.put({
                        "type": "log",
                        "stream": self.stream_name,
                        "project": self.project_name,
                        "message": line,
                        "timestamp": datetime.now().strftime("%H:%M:%S")
                    })
                except Exception:
                    pass
            self.buffer = ""

def _run_training_process(
    project_name: str,
    dataset_yaml_path: str,
    runs_dir: str,
    model_name: str,
    epochs: int,
    batch_size: int,
    imgsz: int,
    device: str,
    run_name: str,
    log_queue: multiprocessing.Queue
):
    # Redirect stdout and stderr to queue
    stdout_writer = QueueWriter(log_queue, project_name, "stdout")
    stderr_writer = QueueWriter(log_queue, project_name, "stderr")
    sys.stdout = stdout_writer
    sys.stderr = stderr_writer

    try:
        log_queue.put({
            "type": "log",
            "project": project_name,
            "message": f"Запуск процесса обучения YOLO для проекта '{project_name}'...",
            "timestamp": datetime.now().strftime("%H:%M:%S")
        })
        log_queue.put({
            "type": "status",
            "project": project_name,
            "status": "running",
            "model": model_name,
            "epochs": epochs
        })

        from ultralytics import YOLO
        import torch

        # Load model
        print(f"Загрузка базовой архитектуры весов: {model_name}")
        model = YOLO(model_name)

        # Attach callbacks for UI progress
        def on_train_epoch_end(trainer):
            try:
                epoch = trainer.epoch + 1
                total_epochs = trainer.epochs
                # Parse loss or metrics
                metrics = {}
                if hasattr(trainer, "metrics") and trainer.metrics:
                    for k, v in trainer.metrics.items():
                        if isinstance(v, (int, float)):
                            metrics[k] = round(float(v), 5)
                log_queue.put({
                    "type": "progress",
                    "project": project_name,
                    "epoch": epoch,
                    "total_epochs": total_epochs,
                    "metrics": metrics,
                    "timestamp": datetime.now().strftime("%H:%M:%S")
                })
            except Exception as e:
                print(f"Callback error: {e}")

        def on_train_end(trainer):
            try:
                best_path = str(trainer.best) if hasattr(trainer, "best") else ""
                log_queue.put({
                    "type": "finish",
                    "project": project_name,
                    "best_model": best_path,
                    "message": "Обучение нейросети успешно завершено!",
                    "timestamp": datetime.now().strftime("%H:%M:%S")
                })
            except Exception:
                pass

        model.add_callback("on_train_epoch_end", on_train_epoch_end)
        model.add_callback("on_train_end", on_train_end)

        # Execute training (handle auto device selection for Mac MPS / CUDA / CPU)
        train_device = None if device in ("auto", "") else device
        print(f"Запуск вычислений model.train(data='{dataset_yaml_path}', epochs={epochs}, batch={batch_size}, imgsz={imgsz}, device='{train_device}')")
        model.train(
            data=dataset_yaml_path,
            project=runs_dir,
            name=run_name,
            epochs=epochs,
            batch=batch_size,
            imgsz=imgsz,
            device=train_device,
            exist_ok=True,
            verbose=True
        )

        stdout_writer.flush()
        stderr_writer.flush()

        # Free VRAM explicitly in child process
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except Exception as e:
        err_msg = traceback.format_exc()
        try:
            log_queue.put({
                "type": "error",
                "project": project_name,
                "message": f"Ошибка обучения: {str(e)}",
                "details": err_msg,
                "timestamp": datetime.now().strftime("%H:%M:%S")
            })
            log_queue.put({
                "type": "status",
                "project": project_name,
                "status": "error"
            })
        except Exception:
            pass
    finally:
        stdout_writer.flush()
        stderr_writer.flush()

def start_training(project_name: str, req: TrainRequest) -> Dict[str, Any]:
    global _active_process, _active_project, _log_queue

    # Check if a process is already running
    if _active_process is not None and _active_process.is_alive():
        raise RuntimeError(f"Training is already running for project '{_active_project}'. Stop it before starting a new run.")

    # Automatically generate/update dataset to ensure sequential indices (0..N-1) and no stale YAML
    try:
        from .dataset import generate_yolo_dataset
        generate_yolo_dataset(project_name)
    except Exception as e:
        print(f"Warning: automatic dataset generation failed: {e}")

    dataset_dir = get_dataset_dir(project_name)
    dataset_yaml = os.path.join(dataset_dir, "dataset.yaml")
    if not os.path.exists(dataset_yaml):
        raise ValueError(f"Dataset YAML not found at {dataset_yaml}. Please generate dataset first.")

    runs_dir = get_runs_dir(project_name)
    os.makedirs(runs_dir, exist_ok=True)

    ctx = multiprocessing.get_context("spawn")
    if _log_queue is None:
        _log_queue = ctx.Queue()

    _active_project = project_name
    _active_process = ctx.Process(
        target=_run_training_process,
        args=(
            project_name,
            dataset_yaml,
            runs_dir,
            req.model_name,
            req.epochs,
            req.batch_size,
            req.imgsz,
            req.device,
            req.run_name or "train_run",
            _log_queue
        )
    )
    _active_process.start()

    return {
        "status": "started",
        "project": project_name,
        "pid": _active_process.pid,
        "model": req.model_name,
        "epochs": req.epochs
    }

def stop_training(project_name: Optional[str] = None) -> Dict[str, Any]:
    global _active_process, _active_project, _log_queue

    if _active_process is None or not _active_process.is_alive():
        return {"status": "stopped", "message": "Процесс обучения в данный момент не запущен."}

    if project_name and _active_project and project_name != _active_project:
        raise ValueError(f"Active training is for project '{_active_project}', not '{project_name}'.")

    pid = _active_process.pid
    try:
        _active_process.terminate()
        _active_process.join(timeout=2)
        if _active_process.is_alive():
            _active_process.kill()
            _active_process.join(timeout=1)
    except Exception as e:
        print(f"Error terminating process: {e}")

    if _log_queue:
        try:
            _log_queue.put({
                "type": "status",
                "project": _active_project,
                "status": "stopped",
                "message": "Процесс обучения принудительно завершен пользователем.",
                "timestamp": datetime.now().strftime("%H:%M:%S")
            })
        except Exception:
            pass

    # Free VRAM in parent process if needed
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    _active_process = None
    _active_project = None

    return {"status": "success", "message": f"Terminated training process (PID {pid}).", "pid": pid}

def get_training_status() -> Dict[str, Any]:
    global _active_process, _active_project
    is_running = _active_process is not None and _active_process.is_alive()
    return {
        "is_running": is_running,
        "project": _active_project if is_running else None,
        "pid": _active_process.pid if is_running and _active_process else None
    }

def register_websocket(ws):
    if ws not in _connected_websockets:
        _connected_websockets.append(ws)

def unregister_websocket(ws):
    if ws in _connected_websockets:
        _connected_websockets.remove(ws)

async def broadcast_log_messages():
    """Background task running in asyncio loop to pump messages from multiprocessing queue to websockets."""
    global _log_queue, _connected_websockets
    import asyncio
    while True:
        if _log_queue is not None:
            while not _log_queue.empty():
                try:
                    msg = _log_queue.get_nowait()
                    if _connected_websockets:
                        # Broadcast to all connected clients
                        dead_ws = []
                        for ws in _connected_websockets:
                            try:
                                await ws.send_json(msg)
                            except Exception:
                                dead_ws.append(ws)
                        for d in dead_ws:
                            unregister_websocket(d)
                except Exception:
                    break
        await asyncio.sleep(0.1)
