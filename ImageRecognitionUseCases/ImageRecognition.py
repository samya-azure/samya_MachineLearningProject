
# yolo_custom_train_and_predict.py

from ultralytics import YOLO
import os

# Step 1: Path to your dataset YAML file
dataset_path = "./DataSets/custom_yolo_dataset/data.yaml"

# Step 2: Load YOLOv8 model (nano version - fastest for training)
model = YOLO("yolov8n.pt")  # You can use yolov8s.pt for slightly better accuracy

# Step 3: Train the model
model.train(
    data=dataset_path,
    epochs=50,
    imgsz=640,
    batch=4,          # Adjust batch size based on your system
    name="custom_model"
)

# Step 4: Load the trained model
trained_model = YOLO("runs/detect/custom_model/weights/best.pt")

# Step 5: Predict on your image
results = trained_model.predict("./DataSets/custom_yolo_dataset/images/train/AnimalsImage.png", save=True, conf=0.25)

# Optional: Show predictions
results[0].show()

# Optional: Print how many of each class was detected
for r in results:
    class_counts = {}
    for cls in r.boxes.cls:
        class_idx = int(cls)
        class_counts[class_idx] = class_counts.get(class_idx, 0) + 1

    print("\nDetected Animals:")
    for cls_idx, count in class_counts.items():
        name = trained_model.names[cls_idx]
        print(f"{name}: {count}")
