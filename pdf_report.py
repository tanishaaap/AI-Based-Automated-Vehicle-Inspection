"""
pdf_report.py -- Generates a downloadable PDF inspection report.
Requires: pip install fpdf2
"""

import os
from fpdf import FPDF
from PIL import Image as PILImage


def _safe_text(text: str) -> str:
    """fpdf2's core Helvetica font only supports Latin-1. LLM-generated text
    often contains smart quotes, em-dashes, bullets, etc. that aren't in that
    charset and would crash multi_cell(). Replace the common offenders, then
    fall back to dropping anything else rather than crashing the whole report."""
    if not text:
        return ""
    replacements = {
        "\u2018": "'", "\u2019": "'",   # smart single quotes
        "\u201c": '"', "\u201d": '"',   # smart double quotes
        "\u2013": "-", "\u2014": "-",   # en/em dash
        "\u2026": "...",                 # ellipsis
        "\u2022": "-",                   # bullet
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    return text.encode("latin-1", errors="replace").decode("latin-1")


def generate_pdf_bytes(inspection_json: dict, summary_text: str, composite_image_path: str,
                        vehicle_info: dict) -> bytes:
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, _safe_text("Vehicle Inspection Report"), ln=True)

    pdf.set_font("Helvetica", "", 11)
    vehicle_line = f"{vehicle_info.get('year', '')} {vehicle_info.get('make', '')} {vehicle_info.get('model', '')} ({vehicle_info.get('color', '')})"
    pdf.cell(0, 8, _safe_text(vehicle_line.strip()), ln=True)
    pdf.ln(2)

    if composite_image_path and os.path.exists(composite_image_path):
        # Compute the image's actual rendered height so we can explicitly move
        # the cursor past it afterward -- fpdf2's image() does not reliably
        # advance x/y on its own, which is what caused the crash.
        with PILImage.open(composite_image_path) as im:
            img_w_px, img_h_px = im.size
        display_w = 180
        display_h = display_w * (img_h_px / img_w_px)

        pdf.image(composite_image_path, x=pdf.l_margin, y=pdf.get_y(), w=display_w)
        pdf.set_xy(pdf.l_margin, pdf.get_y() + display_h + 6)

    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, _safe_text("Detected Findings"), ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_x(pdf.l_margin)

    for gap in inspection_json.get("panel_gaps", []):
        cls = gap.get("classification", {})
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(0, 6, _safe_text(
            f"Panel Gap ({gap['location']}): {gap['measurement_mm']}mm -- {cls.get('status', 'unknown')}"
        ))
    for dent in inspection_json.get("dents", []):
        cls = dent.get("classification", {})
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(0, 6, _safe_text(
            f"Dent: {dent['diameter_mm']}mm, confidence {dent['confidence']:.0%} -- {cls.get('status', 'unknown')}"
        ))
    for scratch in inspection_json.get("scratches", []):
        area = scratch.get("physical_area_mm2")
        area_text = f"{area}mm^2" if area is not None else f"{scratch['pixel_area_px2']}px^2 (uncalibrated)"
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(0, 6, _safe_text(
            f"Scratch: {area_text}, confidence {scratch['confidence']:.0%}"
        ))

    pdf.ln(2)
    pdf.set_x(pdf.l_margin)
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, _safe_text("Inspection Summary"), ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(0, 6, _safe_text(summary_text))

    output = pdf.output(dest="S")
    if isinstance(output, str):
        return output.encode("latin-1")
    return bytes(output)