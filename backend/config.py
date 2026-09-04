import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKSPACE_DIR = os.path.join(BASE_DIR, "workspace")

# Ensure workspace directory exists
os.makedirs(WORKSPACE_DIR, exist_ok=True)

def get_project_dir(project_name: str) -> str:
    return os.path.join(WORKSPACE_DIR, project_name)

def get_raw_images_dir(project_name: str) -> str:
    return os.path.join(get_project_dir(project_name), "raw_images")

def get_classes_file(project_name: str) -> str:
    return os.path.join(get_project_dir(project_name), "classes.json")

def get_annotations_file(project_name: str) -> str:
    return os.path.join(get_project_dir(project_name), "annotations.json")

def get_dataset_dir(project_name: str) -> str:
    return os.path.join(get_project_dir(project_name), "dataset")

def get_runs_dir(project_name: str) -> str:
    return os.path.join(get_project_dir(project_name), "runs")
