from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ultralytics import YOLO
import requests
from PIL import Image
from io import BytesIO
import traceback

# Load the YOLO model
model = YOLO("../models/best.pt")

app = FastAPI()

class ImageUrl(BaseModel):
    image_url: str

@app.post("/yolo_segment/")
def process_image(image: ImageUrl):
    try:
        print(f"Processing image from URL: {image.image_url}")
        
        response = requests.get(image.image_url)
        response.raise_for_status() 
        
        img = Image.open(BytesIO(response.content))
        print(f"Image successfully downloaded and opened. Image size: {img.size}")

        results = model(img)
        print(f"Model inference completed.")

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

    except requests.exceptions.RequestException as req_err:
        print(f"Request error: {req_err}")
        raise HTTPException(status_code=400, detail=f"Failed to download image: {req_err}")

    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Error during inference: {error_trace}")
        raise HTTPException(status_code=500, detail=f"Error during inference: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9881)
