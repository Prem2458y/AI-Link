from ultralytics import YOLO

# Load a YOLOv8n model (can change to 'yolov8s', 'yolov8m', 'yolov8l' based on speed/accuracy need)
model = YOLO('yolov8n.pt')  # 'n' = nano; fastest, smallest

# Train the model
model.train(
    data='data.yaml',
    epochs=15,
    imgsz=640,
    batch=4,
)
