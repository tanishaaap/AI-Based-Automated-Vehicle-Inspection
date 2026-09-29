import os
import torch
from ultralytics import YOLO

MODEL_PATH = "runs/segment/car_scratches_seg_v1-2/weights/best.pt"
IMGSZ = 640


def analyze_scratches_seg(image_path, conf=0.25, imgsz=IMGSZ):
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model file not found at: {MODEL_PATH}")
        return
    if not os.path.exists(image_path):
        print(f"[ERROR] Image not found at: {image_path}")
        return

    device = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO(MODEL_PATH)
    results = model.predict(source=image_path, conf=conf, imgsz=imgsz, save=True, device=device)

    for result in results:
        boxes = result.boxes
        masks = result.masks  # None if no detections

        if masks is None or len(masks) == 0:
            print("[INFO] No scratches detected.")
            continue

        print(f"\n[INFO] Detected {len(masks)} scratch(es):")
        for i, polygon in enumerate(masks.xy):
            cls_id = int(boxes[i].cls[0])
            class_name = model.names[cls_id]
            confidence = float(boxes[i].conf[0])

            # Pixel area from the actual mask polygon -- real pixel measurement,
            # not a box approximation. This is the segmentation payoff.
            import cv2
            import numpy as np
            pixel_area = cv2.contourArea(polygon.astype(np.float32))

            print(
                f"  -> #{i + 1}: [{class_name.upper()}] conf={confidence:.2f} "
                f"pixel_area={pixel_area:.1f} px^2"
            )
            # NOTE: physical_area_mm2 is intentionally NOT calculated here.
            # Converting pixel_area to real-world mm2 requires a per-image
            # pixel-to-mm calibration (e.g. an ArUco marker in frame) which
            # is a separate, not-yet-built module. Reporting a physical area
            # without that calibration would be exactly the fake measurement
            # your original project brief said never to do.


if __name__ == "__main__":
    test_image_path = "seg test images/img1.jpeg"
    analyze_scratches_seg(test_image_path)