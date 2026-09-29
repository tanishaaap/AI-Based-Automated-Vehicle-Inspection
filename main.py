"""
main.py -- Orchestrator for the Automated Vehicle Quality Inspection pipeline.

Wires together: panel_gap_model, dent_model, scratch_model, rag_llm, and
the annotated-composite visualization.

Run from the project root:
    python main.py --image path/to/photo.jpg --location front_left_door
"""

import os
import sys
import argparse
import json
import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "panel_gap_model"))
sys.path.append(os.path.join(BASE_DIR, "dent_model"))
sys.path.append(os.path.join(BASE_DIR, "scratch_model"))
sys.path.append(os.path.join(BASE_DIR, "rag_llm"))

from panel_gap_inspector import measure_panel_gap   # noqa: E402
from calibration_config import MARKER_REAL_SIZE_MM  # noqa: E402
from predict_dent import detect_dents               # noqa: E402
from predict_scratch import detect_scratches        # noqa: E402
from report_generator import generate_report        # noqa: E402
from visualization import create_annotated_composite  # noqa: E402


def run_inspection(image_path: str, panel_location: str, vehicle_info: dict = None,
                    dent_conf: float = None, scratch_conf: float = None):
    """Runs all three CV models on one image. Returns (inspection_json, panel_gap_annotated_image)."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    gap_mm, panel_gap_annotated = measure_panel_gap(image, marker_real_size_mm=MARKER_REAL_SIZE_MM)
    panel_gaps = []
    if gap_mm is not None:
        panel_gaps.append({
            "location": panel_location,
            "measurement_mm": round(gap_mm, 2),
            "calibration_valid": True,
        })

    dents = detect_dents(image_path, conf=dent_conf) if dent_conf is not None else detect_dents(image_path)
    scratches = detect_scratches(image_path, conf=scratch_conf) if scratch_conf is not None else detect_scratches(image_path)

    inspection_json = {
        "image_name": os.path.basename(image_path),
        "vehicle": vehicle_info or {},
        "panel_gaps": panel_gaps,
        "dents": dents,
        "scratches": scratches,
        "inspection_status": "requires_review",
    }
    return inspection_json, panel_gap_annotated


def run_full_pipeline(image_path: str, panel_location: str, vehicle_info: dict = None,
                       dent_conf: float = None, scratch_conf: float = None):
    """Runs CV inspection + annotated visualization + RAG/LLM summary.
    Returns (inspection_json, summary_text, annotated_composite_path)."""
    inspection_json, panel_gap_annotated = run_inspection(
        image_path, panel_location, vehicle_info, dent_conf=dent_conf, scratch_conf=scratch_conf
    )
    summary = generate_report(inspection_json)

    reports_dir = os.path.join(BASE_DIR, "outputs", "reports")
    os.makedirs(reports_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    composite_path = os.path.join(reports_dir, f"{base_name}_annotated.jpg")
    create_annotated_composite(image_path, inspection_json, panel_gap_annotated, composite_path)

    return inspection_json, summary, composite_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the full vehicle inspection pipeline on one image.")
    parser.add_argument("--image", required=True, help="Path to the input image.")
    parser.add_argument("--location", default="front_left_door",
                         help="Which panel/seam this photo is of, e.g. front_left_door, hood, trunk.")
    args = parser.parse_args()

    inspection_json, summary, composite_path = run_full_pipeline(args.image, args.location)

    reports_dir = os.path.join(BASE_DIR, "outputs", "reports")
    out_name = os.path.splitext(os.path.basename(args.image))[0] + "_report.json"
    report_path = os.path.join(reports_dir, out_name)
    with open(report_path, "w") as f:
        json.dump({"inspection": inspection_json, "summary": summary}, f, indent=2)

    print(summary)
    print(f"\nAnnotated image saved to: {composite_path}")
    print(f"Saved full report to: {report_path}")