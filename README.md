# AI-Based Automated Vehicle Inspection System

An AI-based vehicle inspection system that uses computer vision, deep learning, image processing, and RAG-based reporting to detect and quantify vehicle defects from images.

## Project Overview

The system is designed to assist in automated inspection of vehicle body defects such as scratches, dents, and panel-gap irregularities.

The system combines deep learning-based defect detection with computer vision-based measurement techniques to generate structured inspection results.

## Key Features

- Vehicle body defect detection using YOLO-based deep learning models
- Scratch detection and segmentation for pixel-level localization
- Dent detection using YOLOv8
- Scratch dimension estimation using segmentation masks
- Panel-gap measurement using OpenCV and ArUco marker calibration
- Pixel-to-real-world measurement conversion
- Defect visualization and automated reporting
- RAG + LLM-based inspection knowledge and report generation

## System Architecture

```text
Vehicle Images
      |
      v
Image Preprocessing
      |
      +--------------------+
      |                    |
      v                    v
Scratch Detection       Dent Detection
& Segmentation          (YOLOv8)
      |                    |
      v                    v
Scratch Measurement    Dent Measurement
      |                    |
      +---------+----------+
                |
                v
       Panel Gap Measurement
       (OpenCV + ArUco)
                |
                v
       Defect Quantification
                |
                v
       RAG + LLM Reporting
                |
                v
        Final Inspection Report
````

## Models and Techniques

### Scratch Detection and Segmentation

A YOLO-based segmentation approach is used to identify scratches and generate pixel-level masks.

The segmentation masks can be used for:

* Scratch localization
* Scratch area estimation
* Scratch length estimation
* Defect visualization

### Dent Detection

YOLOv8m is used to detect minor dents in vehicle body panels.

The model is evaluated using:

* Precision
* Recall
* mAP@50
* mAP@50:95

Hard-negative images containing reflections, shadows, scratches, and other visually confusing regions are also considered to reduce false-positive dent detections.

### Panel Gap Measurement

The panel-gap module uses:

* OpenCV
* NumPy
* ArUco marker calibration
* Morphological filtering
* Seam extraction
* Boundary profiling
* Geometric analysis

A known-size ArUco marker is used to obtain the pixel-to-real-world scale for physical gap measurement.

## RAG + LLM Module

The project also includes a Retrieval-Augmented Generation (RAG) and Large Language Model module.

The module is intended to provide domain-specific information related to:

* Panel-gap standards
* Scratch severity
* Dent severity and measurement
* Inspection interpretation

The retrieved information is used to generate structured and explainable inspection results.

## Technology Stack

**Programming:** Python

**Machine Learning / Deep Learning:** YOLO, Ultralytics

**Computer Vision:** OpenCV, ArUco

**Data Processing:** NumPy, Pandas

**Annotation / Dataset Management:** Roboflow

**RAG / LLM:** ChromaDB, Sentence Transformers, LangChain, Ollama

**Development:** Git, GitHub

## Project Structure

```text
AI-Based-Automated-Vehicle-Inspection/
│
├── dent_model/
├── scratch_model/
├── panel_gap_model/
├── rag_llm/
├── frontend/
├── outputs/
│
├── auth.py
├── calibration_config.py
├── main.py
├── pdf_report.py
├── visualization.py
├── .gitignore
└── README.md
```

## Workflow

1. Capture or provide vehicle images.
2. Preprocess the input images.
3. Detect scratches and dents using trained deep learning models.
4. Segment scratches for detailed localization.
5. Estimate defect dimensions using calibrated image measurements.
6. Measure panel gaps using ArUco-based scale calibration and OpenCV image processing.
7. Combine the detected defects and measurements.
8. Retrieve relevant inspection knowledge using the RAG module.
9. Generate a structured vehicle inspection report.

## Applications

* Used-car inspection
* Vehicle quality assessment
* Automotive manufacturing inspection
* Body-panel defect analysis
* Automated vehicle condition reporting

## Future Improvements

* Larger and more diverse real-world datasets
* Improved scratch segmentation performance
* More robust dent detection under different lighting conditions
* 3D/depth-based dent measurement
* Improved physical measurement calibration
* Mobile/web deployment for real-time vehicle inspection

## Author

**Tanisha Prajapati**

B.E. Electronics and Telecommunication Engineering
Vidyalankar Institute of Technology

Specialization: Data Analytics & Machine Learning
