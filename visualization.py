"""
visualization.py

Builds the trust-building annotated output: one image split into three
panels (Panel Gap / Dents / Scratches), each showing the original photo
with that defect type's detections drawn on -- box, label, confidence,
and physical dimension. Composited side by side for a single glance.
"""

import os
import cv2
import numpy as np

COLOR_DENT = (255, 100, 0)       # blue-ish (BGR)
COLOR_SCRATCH = (0, 140, 255)    # orange (BGR)
COLOR_PANEL_GAP = (0, 200, 0)    # green (BGR)
COLOR_HEADER_BG = (30, 30, 30)


def _draw_label(image, text, x, y, color):
    """Draws text with a filled background rectangle behind it for readability
    over any part of the photo (light or dark)."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 1
    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
    y = max(y, text_h + 6)
    cv2.rectangle(image, (x, y - text_h - 6), (x + text_w + 6, y + baseline), color, -1)
    cv2.putText(image, text, (x + 3, y - 3), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def _add_header(image, text, color):
    """Adds a colored title bar across the top of a panel so it's obvious
    which defect type this panel represents, even before reading labels."""
    h, w = image.shape[:2]
    bar_h = 40
    canvas = np.full((h + bar_h, w, 3), 255, dtype=np.uint8)
    canvas[bar_h:, :] = image
    cv2.rectangle(canvas, (0, 0), (w, bar_h), COLOR_HEADER_BG, -1)
    cv2.putText(canvas, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)
    return canvas


def draw_dents(image, dents: list[dict]):
    panel = image.copy()
    for dent in dents:
        x1, y1, x2, y2 = dent["bbox"]
        cv2.rectangle(panel, (x1, y1), (x2, y2), COLOR_DENT, 2)
        cal_flag = "" if dent.get("calibration_valid") else " ~approx"
        label = f"Dent {dent['diameter_mm']}mm{cal_flag} ({dent['confidence']*100:.0f}%)"
        _draw_label(panel, label, x1, y1, COLOR_DENT)
    return _add_header(panel, f"DENTS ({len(dents)} found)", COLOR_DENT)


def draw_scratches(image, scratches: list[dict]):
    panel = image.copy()
    for scratch in scratches:
        # Draw the actual segmentation outline -- more accurate than a
        # rectangle since the model already gives a precise mask shape.
        polygon = np.array(scratch["polygon"], dtype=np.int32)
        cv2.polylines(panel, [polygon], isClosed=True, color=COLOR_SCRATCH, thickness=2)

        area_text = (
            f"{scratch['physical_area_mm2']}mm^2" if scratch.get("physical_area_mm2") is not None
            else f"{scratch['pixel_area_px2']}px^2 (uncal.)"
        )
        x1, y1, _, _ = scratch["bbox"]
        label = f"Scratch {area_text} ({scratch['confidence']*100:.0f}%)"
        _draw_label(panel, label, x1, y1, COLOR_SCRATCH)
    return _add_header(panel, f"SCRATCHES ({len(scratches)} found)", COLOR_SCRATCH)


def draw_panel_gaps(image, panel_gaps: list[dict], panel_gap_annotated=None):
    """panel_gap_inspector.py's own measure_panel_gap() already draws the
    marker, seam edges, and dimension line -- reuse that annotated image
    directly rather than re-drawing from scratch."""
    if panel_gap_annotated is not None:
        panel = panel_gap_annotated.copy()
    else:
        panel = image.copy()

    if not panel_gaps:
        _draw_label(panel, "No gap measured in this photo", 20, 40, COLOR_PANEL_GAP)

    count_text = f"PANEL GAP ({len(panel_gaps)} measured)" if panel_gaps else "PANEL GAP (none detected)"
    return _add_header(panel, count_text, COLOR_PANEL_GAP)


def create_annotated_composite(
    image_path: str,
    inspection_json: dict,
    panel_gap_annotated=None,
    output_path: str = None,
) -> str:
    """Builds the 3-panel composite: Panel Gap | Dents | Scratches, side by side.
    Returns the path to the saved composite image."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    panel_1 = draw_panel_gaps(image, inspection_json.get("panel_gaps", []), panel_gap_annotated)
    panel_2 = draw_dents(image, inspection_json.get("dents", []))
    panel_3 = draw_scratches(image, inspection_json.get("scratches", []))

    # Resize all three panels to the same height before concatenating.
    target_h = min(p.shape[0] for p in (panel_1, panel_2, panel_3))
    def resize_to_height(p, h):
        scale = h / p.shape[0]
        return cv2.resize(p, (int(p.shape[1] * scale), h))

    panel_1 = resize_to_height(panel_1, target_h)
    panel_2 = resize_to_height(panel_2, target_h)
    panel_3 = resize_to_height(panel_3, target_h)

    separator = np.full((target_h, 6, 3), 255, dtype=np.uint8)
    composite = np.hstack([panel_1, separator, panel_2, separator, panel_3])

    if output_path is None:
        base = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(os.path.dirname(image_path), f"{base}_annotated.jpg")

    cv2.imwrite(output_path, composite)
    return output_path


