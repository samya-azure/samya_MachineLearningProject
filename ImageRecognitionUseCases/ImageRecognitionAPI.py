
# app/main.py

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import shutil
import os

app = FastAPI()

# Load trained YOLO model
model = YOLO("yolov8n.pt")  # Replace with your actual trained model path

@app.get("/")
def read_root():
    return {"message": "Welcome to Animal Detection API!"}

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    # Save uploaded file temporarily
    file_path = f"temp_{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Run prediction
    results = model.predict(source=file_path, conf=0.25)

    # Count detected objects
    detected_counts = {}
    for r in results:
        for cls_id in r.boxes.cls:
            cls = int(cls_id)
            name = model.names[cls]
            detected_counts[name] = detected_counts.get(name, 0) + 1

    # Clean up temporary file
    os.remove(file_path)

    return JSONResponse(content={"detected_animals": detected_counts})
