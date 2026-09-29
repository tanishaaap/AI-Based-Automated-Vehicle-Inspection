from ultralytics import YOLO
import cv2
import os


# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = r"C:\Users\Tanisha\Downloads\Paint_defect\Paint_defect\dent_model\runs\detect\runs\dent_detection\yolov8m_dent_768\weights\best.pt"

IMAGE_PATH = r"C:\Users\Tanisha\Downloads\Paint_defect\Paint_defect\dent_model\real images\dentimg7.png"

# Detection confidence threshold
CONFIDENCE = 0.15

# Pixel-to-mm calibration
# Example only: 1 pixel = 0.05 mm
PIXEL_TO_MM = 0.05


# ============================================================
# LOAD MODEL
# ============================================================

model = YOLO(MODEL_PATH)


# ============================================================
# RUN DETECTION
# ============================================================

results = model.predict(
    source=IMAGE_PATH,
    imgsz=768,
    conf=CONFIDENCE,
    iou=0.5,
    device=0,
    save=False
)


# ============================================================
# PROCESS DETECTIONS
# ============================================================

image = cv2.imread(IMAGE_PATH)

if image is None:
    print("ERROR: Could not load image.")
    exit()


detections_found = False


for result in results:

    boxes = result.boxes

    if boxes is None:
        continue

    for box in boxes:

        detections_found = True

        # --------------------------------------------
        # Bounding box coordinates
        # --------------------------------------------

        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)


        # --------------------------------------------
        # Confidence
        # --------------------------------------------

        confidence = float(box.conf[0])


        # --------------------------------------------
        # Bounding box dimensions in pixels
        # --------------------------------------------

        width_pixels = x2 - x1
        height_pixels = y2 - y1


        # --------------------------------------------
        # Convert pixels → millimetres
        # --------------------------------------------

        width_mm = width_pixels * PIXEL_TO_MM
        height_mm = height_pixels * PIXEL_TO_MM


        # --------------------------------------------
        # Centre of dent
        # --------------------------------------------

        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2


        # --------------------------------------------
        # Print results
        # --------------------------------------------

        print("\n========================================")
        print("          DENT DETECTED")
        print("========================================")

        print(f"Confidence          : {confidence * 100:.2f}%")

        print(
            f"Bounding box        : "
            f"{width_pixels} × {height_pixels} pixels"
        )

        print(
            f"Estimated dimensions: "
            f"{width_mm:.2f} mm × {height_mm:.2f} mm"
        )

        print(
            f"Location (pixels)   : "
            f"({center_x}, {center_y})"
        )

        print("========================================")


        # ====================================================
        # DRAW BOUNDING BOX
        # ====================================================

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (255, 0, 0),
            2
        )


        # ====================================================
        # LABEL
        # ====================================================

        label = f"DENT {confidence:.2f}"

        cv2.putText(
            image,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2
        )


        # ====================================================
        # DIMENSION TEXT
        # ====================================================

        dimension_text = (
            f"{width_mm:.1f} x {height_mm:.1f} mm"
        )

        cv2.putText(
            image,
            dimension_text,
            (x1, y2 + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2
        )


        # ====================================================
        # CENTRE POINT
        # ====================================================

        cv2.circle(
            image,
            (center_x, center_y),
            5,
            (255, 0, 0),
            -1
        )


# ============================================================
# SAVE RESULT
# ============================================================

output_folder = r"C:\Users\Tanisha\Downloads\Paint_defect\Paint_defect\dent_model\dent_dimension_results"

os.makedirs(output_folder, exist_ok=True)


output_path = os.path.join(
    output_folder,
    "dent_dimension_result.jpg"
)


cv2.imwrite(output_path, image)


# ============================================================
# FINAL MESSAGE
# ============================================================

if detections_found:

    print("\nResult image saved to:")
    print(output_path)

else:

    print("\nNo dent detected.")