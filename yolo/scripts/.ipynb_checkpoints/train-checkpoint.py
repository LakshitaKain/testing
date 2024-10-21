import os
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

model_save_path = os.path.join('..', 'models', 'best.pt')

data = "Users/lakshita/Desktop/pipeline/yolo/data/Menu-Text-Box-2/data.yaml"

model.train(
    data=data,  
    epochs=50,  
    imgsz=640,  
    batch=16,   
    name='yolo_menusections',
    save = True
)
model.export(format='pt', path=model_save_path)  
