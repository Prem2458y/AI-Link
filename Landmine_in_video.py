from ultralytics import YOLO
import cv2
import numpy as np
import os

#path
model_path   = r"D:\Projects\Landmine Detection\runs\detect\train2\weights\best.pt"
INPUT_SOURCE = r"D:\Projects\Landmine Detection\video.mp4"  
conf_thres   = 0.25
target_size  = 640

USE_DENOISE = True
USE_CLAHE   = True
USE_GAMMA   = True
GAMMA_VALUE = None 


def auto_gamma(img_bgr):
    mean = img_bgr.mean()
    gamma = np.interp(mean, [50, 200], [1.6, 0.8])
    return float(np.clip(gamma, 0.6, 2.2))

def apply_gamma(img_bgr, gamma):
    inv = 1.0 / max(gamma, 1e-6)
    table = (np.linspace(0, 1, 256) ** inv * 255).astype(np.uint8)
    return cv2.LUT(img_bgr, table)

def clahe_bgr(img_bgr, clip_limit=2.0, tile_grid_size=(8, 8)):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l2 = clahe.apply(l)
    lab2 = cv2.merge([l2, a, b])
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

def denoise_bgr(img_bgr):
    return cv2.bilateralFilter(img_bgr, d=5, sigmaColor=50, sigmaSpace=50)

def letterbox(img, new_size=640, color=(114, 114, 114)):
    h, w = img.shape[:2]
    scale = min(new_size / h, new_size / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top = (new_size - nh) // 2
    bottom = new_size - nh - top
    left = (new_size - nw) // 2
    right = new_size - nw - left
    out = cv2.copyMakeBorder(resized, top, bottom, left, right,
                             borderType=cv2.BORDER_CONSTANT, value=color)
    return out, scale, (left, top)

def preprocess(img_bgr, target=640, do_letterbox=False):
    if USE_DENOISE:
        img_bgr = denoise_bgr(img_bgr)
    if USE_CLAHE:
        img_bgr = clahe_bgr(img_bgr)
    if USE_GAMMA:
        gamma = auto_gamma(img_bgr) if GAMMA_VALUE is None else float(GAMMA_VALUE)
        img_bgr = apply_gamma(img_bgr, gamma)
    if do_letterbox:
        img_bgr, _, _ = letterbox(img_bgr, new_size=target)

    return img_bgr



IMG_EXTS={".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
VID_EXTS={".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v", ".webm"}

def is_image(path):
    return os.path.isfile(path) and os.path.splitext(path)[1].lower() in IMG_EXTS

def is_video(path):
    return os.path.isfile(path) and os.path.splitext(path)[1].lower() in VID_EXTS

#load model
assert os.path.exists(model_path), f"Model not found: {model_path}"
model = YOLO(model_path)
save_root = os.path.join("runs", "detect", "predict")
os.makedirs(save_root, exist_ok=True)


if (isinstance(INPUT_SOURCE, int) and INPUT_SOURCE == 0) or (isinstance(INPUT_SOURCE, str) and INPUT_SOURCE == "0"):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (device 0).")
    out_path = os.path.join(save_root, "webcam_output.mp4")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    print("Press 'q' to stop...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        pre = preprocess(frame, target=target_size)
        res = model.predict(source=pre, imgsz=target_size, conf=conf_thres, verbose=False)
        annotated = res[0].plot()
        writer.write(annotated)
        cv2.imshow("Webcam (q to quit)", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release(); writer.release(); cv2.destroyAllWindows()
    print(f"Annotated webcam video saved to: {out_path}")

#for image 
elif isinstance(INPUT_SOURCE, str) and is_image(INPUT_SOURCE):
    assert os.path.exists(INPUT_SOURCE), f"Image not found: {INPUT_SOURCE}"
    img = cv2.imread(INPUT_SOURCE)
    if img is None:
        raise RuntimeError(f"Failed to read image: {INPUT_SOURCE}")
    pre = preprocess(img, target=target_size)
    results = model.predict(source=pre, imgsz=target_size, conf=conf_thres, save=True, verbose=False)
    annotated = results[0].plot()
    cv2.imshow("Prediction (Image)", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("Annotated image saved under runs/detect/predict/")

#for video 
elif isinstance(INPUT_SOURCE, str) and is_video(INPUT_SOURCE):
    assert os.path.exists(INPUT_SOURCE), f"Video not found: {INPUT_SOURCE}"
    cap = cv2.VideoCapture(INPUT_SOURCE)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {INPUT_SOURCE}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    base = os.path.basename(INPUT_SOURCE)
    out_path = os.path.join(save_root, base)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    print("Processing video... Press 'q' to stop preview early.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        pre = preprocess(frame, target=target_size)
        res = model.predict(source=pre, imgsz=target_size, conf=conf_thres, verbose=False)
        annotated = res[0].plot()
        writer.write(annotated)

        cv2.imshow("Prediction (Video)", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release(); writer.release(); cv2.destroyAllWindows()
    print(f"Annotated video saved to: {out_path}")

else:
    raise ValueError("INPUT_SOURCE must be an existing image path, video path, or 0 for webcam.")
