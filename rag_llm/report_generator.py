import copy
import requests
import json
from retriever import retrieve
from panel_gap_classifier import classify_panel_gap
from dent_classifier import classify_dent


def _sanitize_for_llm(inspection_json: dict) -> dict:
    """Strip fields that exist purely for drawing/storage (raw polygon
    coordinates, bounding boxes) before this gets serialized into the LLM
    prompt. A 3B model has no use for dozens of raw pixel coordinates --
    it just confuses it into inventing nonsense ("pixels out of calibration")
    instead of explaining the actual measurements. The full data (polygon,
    bbox) is still returned to the caller untouched for visualization/storage;
    this sanitized copy is used ONLY for what the LLM sees."""
    sanitized = copy.deepcopy(inspection_json)
    for scratch in sanitized.get("scratches", []):
        scratch.pop("polygon", None)
        scratch.pop("bbox", None)
    for dent in sanitized.get("dents", []):
        dent.pop("bbox", None)
    return sanitized


def generate_report(inspection_json: dict, ollama_model="llama3.2:3b"):
    # Classification mutates the REAL inspection_json in place -- callers
    # (main.py, the frontend) need these classification fields for display,
    # the PDF, and history storage.
    for gap in inspection_json.get("panel_gaps", []):
        gap["classification"] = classify_panel_gap(gap["location"], gap["measurement_mm"])

    for dent in inspection_json.get("dents", []):
        dent["classification"] = classify_dent(
            dent.get("diameter_mm"),
            has_paint_damage=dent.get("paint_damage", False)
        )

    # Build the LLM's view from a SANITIZED copy -- drawing-only fields never
    # reach the prompt.
    llm_view = _sanitize_for_llm(inspection_json)

    query = f"Explain this vehicle inspection result: {json.dumps(llm_view)}"
    context_chunks = retrieve(query)
    context = "\n\n".join(context_chunks)

    prompt = f"""You are a vehicle inspection assistant. Every "classification" field below has
ALREADY been computed correctly in code -- do not recalculate or second-guess it, just explain it
in plain language for the customer. Use ONLY the reference context for any general explanation.
Do not invent numeric tolerances that aren't in the context. If the context doesn't cover
something, say so. Do not use placeholder text like "[insert item]" -- use the actual vehicle
details and numbers given below.

REFERENCE CONTEXT:
{context}

INSPECTION DATA (classification fields are already correct, trust them as-is):
{json.dumps(llm_view, indent=2)}

Write a short, clear inspection summary for the customer."""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": ollama_model, "prompt": prompt, "stream": False}
    )
    return response.json()["response"]


if __name__ == "__main__":
    mock_inspection = {
        "panel_gaps": [{"location": "front_left_door", "measurement_mm": 4.2, "calibration_valid": True}],
        "scratches": [{"class": "scratch", "confidence": 0.78, "pixel_area_px2": 1637.5, "physical_area_mm2": 57.51}],
        "dents": [{"diameter_mm": 65.0, "confidence": 0.81, "paint_damage": False}],
        "inspection_status": "requires_review"
    }
    print(generate_report(mock_inspection))