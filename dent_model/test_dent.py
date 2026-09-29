from ultralytics import YOLO


def main():

    # Load the BEST trained model
    model = YOLO(
        r"C:\Users\Tanisha\Downloads\Paint_defect\Paint_defect\dent_model\runs\detect\runs\dent_detection\yolov8m_dent_768\weights\best.pt"
    )

    # ==========================================
    # 1. EVALUATE ON FINAL TEST SET
    # ==========================================

    metrics = model.val(
        data=r"C:\Users\Tanisha\Downloads\Paint_defect\Paint_defect\dent_model\minor-dent-detection-dataset\data.yaml",
        split="test",
        imgsz=768,
        batch=4,
        device=0,
        workers=0,
        plots=True
    )

    # ==========================================
    # 2. RUN PREDICTIONS ON TEST IMAGES
    # ==========================================

    results = model.predict(
        source=r"C:\Users\Tanisha\Downloads\Paint_defect\Paint_defect\dent_model\minor-dent-detection-dataset\test\images",
        imgsz=768,
        conf=0.15,
        iou=0.5,
        device=0,
        workers=0,
        save=True,
        save_txt=True,
        save_conf=True
    )

    # ==========================================
    # 3. DISPLAY FINAL TEST RESULTS
    # ==========================================

    print("\n========== FINAL TEST RESULTS ==========")

    print(f"Precision       : {metrics.box.mp:.4f}")
    print(f"Recall          : {metrics.box.mr:.4f}")
    print(f"mAP@0.50        : {metrics.box.map50:.4f}")
    print(f"mAP@0.50:0.95   : {metrics.box.map:.4f}")

    print("========================================")


if __name__ == "__main__":
    main()