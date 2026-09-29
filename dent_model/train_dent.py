from ultralytics import YOLO
from ultralytics import YOLO

# Load the last checkpoint
model = YOLO(
    r"C:\Users\Tanisha\Downloads\Paint_defect\Paint_defect\dent_model\runs\detect\runs\dent_detection\yolov8m_dent_768\weights\last.pt"
)

# Resume training
model.train(
    resume=True
)
# # Load pretrained YOLOv8m
# model = YOLO("yolov8m.pt")

# results = model.train(
#     data=r"C:\Users\Tanisha\Downloads\Paint_defect\Paint_defect\dent_model\minor-dent-detection-dataset\data.yaml",

#     # Training
#     epochs=150,
#     imgsz=768,
#     batch=4,

#     # Hardware
#     device=0,

#     workers=0,

#     # Optimization
#     optimizer="AdamW",
#     lr0=0.001,
#     lrf=0.01,
#     weight_decay=0.0005,

#     # Augmentation
#     hsv_h=0.015,
#     hsv_s=0.5,
#     hsv_v=0.4,

#     degrees=5,
#     translate=0.1,
#     scale=0.4,
#     shear=2.0,
#     perspective=0.0005,

#     fliplr=0.5,

#     # Mosaic / MixUp
#     mosaic=0.5,
#     mixup=0.05,

#     # Training stability
#     patience=30,
#     cos_lr=True,

#     # Save
#     project="runs/dent_detection",
#     name="yolov8m_dent_768",
#     exist_ok=True,

#     plots=True,
#     save=True
# )
