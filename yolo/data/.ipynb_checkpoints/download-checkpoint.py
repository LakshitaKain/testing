import os
from roboflow import Roboflow

api_key = os.getenv("ROBOFLOW_API")

if not api_key:
    raise ValueError("ROBOFLOW_API key not found. Please set the environment variable.")

rf = Roboflow(api_key=api_key)
project = rf.workspace("lakshita96-tgk5f").project("menu-text-box-noslb")
version = project.version(2)

print("Downloading dataset...")
dataset = version.download("yolov8")
print("Dataset downloaded successfully!")
