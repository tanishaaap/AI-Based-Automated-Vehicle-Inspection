"""
dent_model/predict_dent.py

Callable wrapper around the dent YOLO model. Returns structured data with
real per-photo ArUco calibration (reusing panel_gap_model's calibration
engine) instead of a fixed placeholder ratio.
"""

import os
import sys
import cv2
import torch
from ultralytics import YOLO

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "panel_gap_model"))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from panel_gap_inspector import detect_aruco_scale  # noqa: E402
from calibration_config import MARKER_REAL_SIZE_MM   # noqa: E402

MODEL_PATH = os.path.join(os.path.dirname(__file__), "weights", "best.pt")
CONFIDENCE = 0.15
IMG_SIZE = 768

# Fallback ratio, used ONLY if no ArUco marker is found in the photo.
FALLBACK_PIXEL_TO_MM = 0.05

_model_cache = None


def _get_model():
    global _model_cache
    if _model_cache is None:
        _model_cache = YOLO(MODEL_PATH)
    return _model_cache


def detect_dents(image_path: str, conf: float = CONFIDENCE) -> list[dict]:
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    # Step 1+2 of the conversion: find marker pixel width, compute mm/pixel.
    mm_per_pixel, _ = detect_aruco_scale(image, marker_real_size_mm=MARKER_REAL_SIZE_MM)
    calibration_valid = mm_per_pixel is not None
    if not calibration_valid:
        mm_per_pixel = FALLBACK_PIXEL_TO_MM

    device = 0 if torch.cuda.is_available() else "cpu"
    model = _get_model()
    results = model.predict(source=image_path, imgsz=IMG_SIZE, conf=conf, iou=0.5, device=device, save=False)

    dents = []
    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            width_px = float(x2 - x1)
            height_px = float(y2 - y1)
            confidence = float(box.conf[0])

            # Step 3a: 1D conversion -- multiply by mm_per_pixel ONCE.
            width_mm = round(width_px * mm_per_pixel, 1)
            height_mm = round(height_px * mm_per_pixel, 1)

            dent = {
                "confidence": confidence,
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "width_mm": width_mm,
                "height_mm": height_mm,
                "diameter_mm": max(width_mm, height_mm),
                "calibration_valid": calibration_valid,
                "paint_damage": False,
            }
            if not calibration_valid:
                dent["note"] = "No ArUco marker detected -- using fallback approximation, not a calibrated measurement."
            dents.append(dent)
    return dents


if __name__ == "__main__":
    import json
    test_image = r"C:\path\to\your\test\image.jpg"
    print(json.dumps(detect_dents(test_image), indent=2))