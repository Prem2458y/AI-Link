from ultralytics import YOLO
import os
import numpy as np

# ==== paths ====
model_path = r"D:\Projects\Landmine Detection\runs\detect\train2\weights\best.pt"
data_yaml  = r"D:\Projects\Landmine Detection\data.yaml"

assert os.path.exists(model_path), f"Model not found: {model_path}"
assert os.path.exists(data_yaml),  f"data.yaml not found: {data_yaml}"

# ==== load model ====
model = YOLO(model_path)

# ==== try test split, else fall back to val ====
def eval_split(split_name):
    print(f"\nEvaluating split: {split_name}")
    metrics = model.val(
        data=data_yaml,
        split=split_name,   # 'test' or 'val'
        imgsz=640,
        batch=4,
        device='cpu',
        conf=0.001,
        iou=0.6,
        plots=False,
        save_json=False,
        verbose=False
    )
    return metrics

try:
    metrics = eval_split("test")
except Exception as e:
    print(f"Couldn't evaluate 'test' split ({e}). Falling back to 'val'...")
    metrics = eval_split("val")

# ---- mAPs (already scalars) ----
print(f"mAP@0.5:0.95 : {metrics.box.map:.4f}")
print(f"mAP@0.5      : {metrics.box.map50:.4f}")
print(f"mAP@0.75     : {metrics.box.map75:.4f}")

# ---- Precision/Recall/F1 are arrays -> average them ----
# Some Ultralytics versions expose mp/mr/mf1; if not, we compute nanmean.
mp = getattr(metrics.box, "mp", float(np.nanmean(metrics.box.p)))
mr = getattr(metrics.box, "mr", float(np.nanmean(metrics.box.r)))
mf1 = getattr(metrics.box, "mf1", float(np.nanmean(getattr(metrics.box, "f1", np.array([])))) if hasattr(metrics.box, "f1") else None)

print(f"Precision (mean) : {mp:.4f}")
print(f"Recall (mean)    : {mr:.4f}")
if mf1 is not None and not np.isnan(mf1):
    print(f"F1 (mean)        : {mf1:.4f}")

