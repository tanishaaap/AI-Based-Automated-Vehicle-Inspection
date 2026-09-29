"""
Deterministic classification for panel gaps, mirroring the tolerance table
in knowledge/sources/panel_gap_stdquality.txt.

WHY THIS EXISTS: small local LLMs (e.g. llama3.2:3b) are unreliable at doing
numeric range comparisons inside free-text generation. Testing showed it
called a 4.2mm gap "above the 4.0-5.0mm range" when 4.2 is clearly inside
that range. The fix is to never ask the LLM to decide IF something is out
of tolerance -- only to explain a decision that Python already made.
"""

# Zone tolerance table -- keep this in sync with panel_gap_stdquality.txt Section 2.
# (single source of truth would ideally be the doc itself, parsed once; hardcoding
# here for now since the table is small and stable)
PANEL_GAP_ZONES = {
    "hood_to_fender":      {"nominal_min": 3.5, "nominal_max": 4.5, "max_deviation": 0.5},
    "door_to_door":        {"nominal_min": 4.0, "nominal_max": 5.0, "max_deviation": 0.6},
    "door_to_fender":      {"nominal_min": 3.5, "nominal_max": 4.5, "max_deviation": 0.5},
    "tailgate_to_quarter": {"nominal_min": 3.0, "nominal_max": 5.0, "max_deviation": 0.8},
    "fascia_to_fender":    {"nominal_min": 1.5, "nominal_max": 3.0, "max_deviation": None},
}

# Map raw location strings (as they'll appear in your inspection JSON) to a zone key.
# Extend this as your panel_gap.py module reports more specific locations.
LOCATION_TO_ZONE = {
    "front_left_door": "door_to_fender",
    "front_right_door": "door_to_fender",
    "rear_left_door": "door_to_door",
    "rear_right_door": "door_to_door",
    "hood": "hood_to_fender",
    "trunk": "tailgate_to_quarter",
    "bumper": "fascia_to_fender",
}


def classify_panel_gap(location: str, measurement_mm: float) -> dict:
    """Returns a status classification computed in code, not left to the LLM."""
    zone_key = LOCATION_TO_ZONE.get(location)
    if zone_key is None:
        return {
            "zone": None,
            "status": "unknown_zone",
            "reason": f"No tolerance mapping defined for location '{location}'. "
                      f"Cannot classify without a known zone."
        }

    zone = PANEL_GAP_ZONES[zone_key]
    nominal_min, nominal_max = zone["nominal_min"], zone["nominal_max"]

    if nominal_min <= measurement_mm <= nominal_max:
        status = "within_tolerance"
        reason = f"{measurement_mm}mm falls within the {zone_key.replace('_', ' ')} nominal range of {nominal_min}-{nominal_max}mm."
    elif measurement_mm < nominal_min:
        deviation = nominal_min - measurement_mm
        status = "major_deviation" if deviation > 1.5 else "minor_deviation"
        reason = f"{measurement_mm}mm is {deviation:.1f}mm below the nominal {nominal_min}-{nominal_max}mm range."
    else:
        deviation = measurement_mm - nominal_max
        status = "major_deviation" if deviation > 1.5 else "minor_deviation"
        reason = f"{measurement_mm}mm is {deviation:.1f}mm above the nominal {nominal_min}-{nominal_max}mm range."

    return {"zone": zone_key, "status": status, "reason": reason}


if __name__ == "__main__":
    # Sanity check against the exact case that exposed the bug
    result = classify_panel_gap("front_left_door", 4.2)
    print(result)
    # Expected: status == "within_tolerance", since door_to_fender range is 3.5-4.5mm