"""
calibration_config.py

Single source of truth for the physical ArUco marker size used across
panel_gap_model, dent_model, and scratch_model. Change it in exactly one
place if you ever print a different marker.
"""

MARKER_REAL_SIZE_MM = 20.0  # confirmed from your printed marker photos


# ---------------------------------------------------------------------------
# THE ACTUAL CONVERSION MATH (this is what "pixel to physical" means here)
# ---------------------------------------------------------------------------
#
# Step 1 -- find the marker's pixel width in the photo (detect_aruco_scale()
#           in panel_gap_inspector.py does this via cv2.aruco detection):
#
#     marker_pixel_width = distance between two known marker corners, in pixels
#
# Step 2 -- compute the scale factor (mm per pixel), using the REAL marker
#           size as the known reference:
#
#     mm_per_pixel = MARKER_REAL_SIZE_MM / marker_pixel_width
#
#     Example: if the 20mm marker measures 96 pixels wide in the photo,
#     mm_per_pixel = 20.0 / 96 = 0.2083 mm/pixel
#
# Step 3a -- DENT: convert a 1D pixel length (bbox width/height) to mm by
#            multiplying by mm_per_pixel ONCE:
#
#     width_mm  = width_pixels  * mm_per_pixel
#     height_mm = height_pixels * mm_per_pixel
#
# Step 3b -- SCRATCH: convert a 2D pixel AREA to physical area (mm^2) by
#            multiplying by mm_per_pixel SQUARED -- area scales with the
#            square of a linear measurement, not linearly:
#
#     physical_area_mm2 = pixel_area_px2 * (mm_per_pixel ** 2)
#
#     This is the single most common mistake in this kind of conversion --
#     using mm_per_pixel once on an area value undercounts it badly.
#     Example: if mm_per_pixel = 0.2, a 100px^2 scratch is NOT 20mm^2,
#     it's 100 * (0.2 ** 2) = 4mm^2.
# ---------------------------------------------------------------------------
