from fastapi import FastAPI, HTTPException, APIRouter
from pydantic import BaseModel
from ultralytics import YOLO
import requests
from PIL import Image
from io import BytesIO
import numpy as np
from typing import List
import torch

 
model = YOLO()


app = FastAPI()
router = APIRouter()

class ImageUrl(BaseModel):
    image_url: str

@router.post("/yolo_segment/")
def process_image(image: ImageUrl):
    try:
        #image_url = image.image_url
        
        #response = requests.get(image_url)
        #response.raise_for_status()  
        #img = Image.open(BytesIO(response.content))

        #results = model(img) 
        results = model(image)
        
        predictions = []
        for result in results:
            boxes = result.boxes 
            xywh = boxes.xywh.cpu().numpy()  
            conf = boxes.conf.cpu().numpy()  
            cls = boxes.cls.cpu().numpy() 
            names = result.names  

            for i in range(len(xywh)):
                prediction = {
                    'x': float(xywh[i][0]), 
                    'y': float(xywh[i][1]),  
                    'width': float(xywh[i][2]),  
                    'height': float(xywh[i][3]),
                    'confidence': float(conf[i]),  
                    'class_name': names[int(cls[i])],  
                    'segment_id': i 
                }
                predictions.append(prediction)

        return {"predictions": predictions}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during inference: {e}")

app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9881)