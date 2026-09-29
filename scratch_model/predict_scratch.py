"""
scratch_model/predict_scratch.py

Callable wrapper around the scratch segmentation model. Measures pixel AREA
from the segmentation mask (this model does not measure length), converted
to physical mm^2 using the shared ArUco calibration engine. Also returns a
bounding box (for drawing) and the mask polygon (for accurate outline drawing).
"""

import os
import sys
import cv2
import numpy as np
import torch
from ultralytics import YOLO

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "panel_gap_model"))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from panel_gap_inspector import detect_aruco_scale  # noqa: E402
from calibration_config import MARKER_REAL_SIZE_MM   # noqa: E402

MODEL_PATH = os.path.join(os.path.dirname(__file__), "weights", "best.pt")
IMGSZ = 640
CONFIDENCE = 0.25

_model_cache = None


def _get_model():
    global _model_cache
    if _model_cache is None:
        _model_cache = YOLO(MODEL_PATH)
    return _model_cache


def detect_scratches(image_path: str, conf: float = CONFIDENCE, imgsz: int = IMGSZ) -> list[dict]:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at: {image_path}")

    image = cv2.imread(image_path)
    mm_per_pixel, _ = detect_aruco_scale(image, marker_real_size_mm=MARKER_REAL_SIZE_MM)
    calibration_valid = mm_per_pixel is not None

    device = 0 if torch.cuda.is_available() else "cpu"
    model = _get_model()
    results = model.predict(source=image_path, conf=conf, imgsz=imgsz, save=False, device=device)

    scratches = []
    for result in results:
        boxes = result.boxes
        masks = result.masks

        if masks is None or len(masks) == 0:
            continue

        for i, polygon in enumerate(masks.xy):
            cls_id = int(boxes[i].cls[0])
            class_name = model.names[cls_id]
            confidence = float(boxes[i].conf[0])
            pixel_area = float(cv2.contourArea(polygon.astype(np.float32)))

            x1, y1, x2, y2 = boxes[i].xyxy[0].cpu().numpy()

            scratch = {
                "class": class_name,
                "confidence": confidence,
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "polygon": polygon.astype(int).tolist(),  # for accurate outline drawing
                "pixel_area_px2": round(pixel_area, 1),
                "calibration_valid": calibration_valid,
            }
            if calibration_valid:
                # Step 3b: AREA conversion -- mm_per_pixel SQUARED, not once.
                scratch["physical_area_mm2"] = round(pixel_area * (mm_per_pixel ** 2), 2)
            else:
                scratch["physical_area_mm2"] = None
                scratch["note"] = "No ArUco marker detected -- physical area unavailable without calibration."
            scratches.append(scratch)

    return scratches


if __name__ == "__main__":
    import json
    test_image = "seg test images/img1.jpeg"
    print(json.dumps(detect_scratches(test_image), indent=2))