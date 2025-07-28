from ultralytics import YOLO
import cv2
import os

model_path = r"D:\Projects\Landmine Detection\runs\detect\train2\weights\best.pt"
image_path = r"D:\Projects\Landmine Detection\train\images\0082_jpg.rf.fd75e047d8866d3aec64246ee2e86afa.jpg"

assert os.path.exists(model_path), f"Model not found: {model_path}"
assert os.path.exists(image_path), f"Image not found: {image_path}"

model = YOLO(model_path)

results = model(image_path, save=True)  
annotated_frame = results[0].plot()
cv2.imshow("Prediction", annotated_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
