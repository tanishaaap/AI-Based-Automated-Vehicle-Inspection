"""
Deterministic classification for dents, mirroring knowledge/sources/dent_dim_stds.txt.
Same reasoning as panel_gap_classifier.py: let code decide the threshold, let the
LLM only explain it in plain language.
"""

PDR_DIAMETER_THRESHOLD_MM = 50.0


def classify_dent(diameter_mm: float, has_paint_damage: bool = False) -> dict:
    """
    diameter_mm: the larger of width/height from the bounding box, converted to mm.
    has_paint_damage: pass True if paint chipping/cracking/bare metal was also detected —
        per the doc, this alone pushes a dent to "severe" regardless of size.
    """
    if diameter_mm is None:
        return {"status": "unknown", "reason": "No diameter measurement available."}

    if has_paint_damage:
        return {
            "status": "severe",
            "reason": f"Paint integrity compromised (chipping/cracking/bare metal), "
                      f"which requires body shop repair regardless of the {diameter_mm}mm size."
        }

    if diameter_mm < PDR_DIAMETER_THRESHOLD_MM:
        return {
            "status": "minor",
            "reason": f"{diameter_mm}mm is below the {PDR_DIAMETER_THRESHOLD_MM}mm threshold — "
                      f"a Paintless Dent Repair (PDR) candidate."
        }
    else:
        return {
            "status": "severe",
            "reason": f"{diameter_mm}mm exceeds the {PDR_DIAMETER_THRESHOLD_MM}mm threshold — "
                      f"likely requires body filler and repainting, or panel replacement."
        }


if __name__ == "__main__":
    print(classify_dent(35.0))   # expect: minor
    print(classify_dent(65.0))   # expect: severe
    print(classify_dent(20.0, has_paint_damage=True))  # expect: severe (paint override)