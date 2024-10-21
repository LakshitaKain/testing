from ultralytics import YOLO

model = YOLO("../models/best.pt")

#Desktop/pipeline/yolo/models/best.pt

def yolo_segment(image):
    try:
        results = model(image)
        
        predictions = []  

        for result in results:
            boxes = result.boxes  
            xywh = boxes.xyxy.cpu().numpy()  
            conf = boxes.conf.cpu().numpy()  
            cls = boxes.cls.cpu().numpy()  
            names = result.names  
            
            result.show()

            for i in range(len(xywh)):
                prediction = {
                    'x': xywh[i][0],  
                    'y': xywh[i][1], 
                    'width': xywh[i][2],
                    'height': xywh[i][3], 
                    'confidence': float(conf[i]),  
                    'class': names[int(cls[i])],  
                    'segment_id': i  
                }
                predictions.append(prediction)

        return predictions 
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during inference: {e}")

#Example usage      
# img = "https://onemenus.s3.wasabisys.com/deskew/de-skewed-image_94be9b0c81a10c756fcafdc18894383d.jpg"
# results = yolo_segment(img)