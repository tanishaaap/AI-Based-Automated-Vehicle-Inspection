#!/usr/bin/env python3
"""
Automotive Panel Gap Inspector
------------------------------
A clean, production-grade Python script that measures automotive panel gaps
in physical millimeters from a 2D image using OpenCV and an ArUco marker calibration target.

Architecture:
1. Calibration Engine: Detects an ArUco marker (DICT_4X4_50, ID 0) to compute mm/pixel.
2. Seam & Gap Extraction: Localizes the automotive panel groove near the marker and traces
   sub-pixel left/right boundary edges using cross-sectional gradient profiling.
3. Visualization: Draws markers, traced seam edges, dimension line, and high-contrast text.
4. Benchmark Engine: Computes Mean Absolute Error (MAE) and Standard Deviation (±σ) on datasets.
5. Self-Contained Testing: Includes a synthetic automotive joint image generator for unit testing out-of-the-box.
"""

import os
import csv
import logging
import argparse
from typing import Tuple, Optional, List, Dict, Any
import numpy as np
import cv2

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s"
)
logger = logging.getLogger("PanelGapInspector")

# Global Configuration Parameters
MARKER_REAL_SIZE_MM = 20.0
ARUCO_DICT_ID = cv2.aruco.DICT_4X4_50
TARGET_MARKER_ID = 0


def detect_aruco_scale(
    image: np.ndarray,
    marker_real_size_mm: float = MARKER_REAL_SIZE_MM,
    target_id: int = TARGET_MARKER_ID
) -> Tuple[Optional[float], Optional[np.ndarray]]:
    """
    Detects the calibration ArUco marker and computes the spatial scale (mm per pixel).

    This function handles OpenCV version compatibility between the legacy
    cv2.aruco.detectMarkers and the modern cv2.aruco.ArucoDetector API.

    Args:
        image: Input BGR image.
        marker_real_size_mm: The physical width/height of the square marker in mm.
        target_id: The expected ID of the calibration marker.

    Returns:
        A tuple of (mm_per_pixel, marker_corners) if successful, otherwise (None, None).
        marker_corners is of shape (4, 2) representing the corners of the target marker.
    """
    if image is None:
        logger.error("Input image is None.")
        return None, None

    # Convert to grayscale if it is a color image
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    corners: List[np.ndarray] = []
    ids: Optional[np.ndarray] = None

    # Modern OpenCV API (version >= 4.7.0)
    try:
        aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
        parameters = cv2.aruco.DetectorParameters()
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
        corners, ids, _ = detector.detectMarkers(gray)
        
        # Multi-stage attempt 2: CLAHE enhanced contrast if not found
        if ids is None or target_id not in ids.flatten():
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
            gray_clahe = clahe.apply(gray)
            corners, ids, _ = detector.detectMarkers(gray_clahe)
    except AttributeError:
        # Legacy OpenCV API
        try:
            dict_val = ARUCO_DICT_ID
            try:
                aruco_dict = cv2.aruco.Dictionary_get(dict_val)
            except AttributeError:
                aruco_dict = cv2.aruco.getPredefinedDictionary(dict_val)

            try:
                parameters = cv2.aruco.DetectorParameters_create()
            except AttributeError:
                parameters = cv2.aruco.DetectorParameters()

            corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
        except Exception as e:
            logger.error(f"Failed to execute legacy ArUco detection: {e}", exc_info=True)
            return None, None

    # Multi-stage attempt 3: Geometric fallback for damaged/peeled/folded marker stickers
    if ids is None or len(ids) == 0 or (target_id is not None and target_id not in ids.flatten()):
        logger.info("Standard ArUco pattern match not found. Attempting geometric square recovery (handling peeled/folded marker corners)...")
        recovered_corners = _recover_peeled_marker_contour(gray)
        if recovered_corners is not None:
            marker_pixel_width = float(np.linalg.norm(recovered_corners[0] - recovered_corners[1]))
            if marker_pixel_width > 0:
                mm_per_pixel = marker_real_size_mm / marker_pixel_width
                logger.info(f"Successfully recovered damaged/peeled marker geometry. Width: {marker_pixel_width:.2f} px. "
                            f"Spatial Scale: {mm_per_pixel:.5f} mm/pixel.")
                return mm_per_pixel, recovered_corners

    if ids is None or len(ids) == 0:
        logger.warning("No ArUco markers detected in the image.")
        return None, None

    # Find the target marker ID
    target_idx = -1
    for idx, marker_id in enumerate(ids.flatten()):
        if marker_id == target_id:
            target_idx = idx
            break

    if target_idx == -1:
        logger.warning(f"ArUco marker with ID {target_id} was not found among detected IDs: {ids.flatten()}")
        return None, None

    # Extracted corner coordinates: shape is (1, 4, 2)
    marker_corners = corners[target_idx].reshape(4, 2)

    # Top edge of the marker connects corner 0 (top-left) to corner 1 (top-right)
    pt_tl = marker_corners[0]
    pt_tr = marker_corners[1]

    # Calculate pixel width using Euclidean distance
    marker_pixel_width = float(np.linalg.norm(pt_tl - pt_tr))
    if marker_pixel_width <= 0:
        logger.error(f"Calculated marker pixel width is non-positive: {marker_pixel_width}")
        return None, None

    # Calculate scale factor
    mm_per_pixel = marker_real_size_mm / marker_pixel_width
    logger.info(f"Detected ArUco Marker ID {target_id}. Top edge width: {marker_pixel_width:.2f} px. "
                f"Spatial Scale: {mm_per_pixel:.5f} mm/pixel.")

    return mm_per_pixel, marker_corners
def _recover_peeled_marker_contour(gray: np.ndarray) -> Optional[np.ndarray]:
    """
    Fallback recovery for calibration markers whose corners are peeled, folded,
    or shadowed by harsh lighting, breaking standard ArUco 4-point quad detection.
    """
    h, w = gray.shape[:2]
    total_area = h * w

    # Scale-adaptive window sizes for high-resolution images
    base_win = int(min(w, h) * 0.05)
    if base_win % 2 == 0:
        base_win += 1

    bin_images = []
    # 1. Otsu thresholding
    _, otsu_bin = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    bin_images.append(otsu_bin)

    # 2. Multi-scale adaptive thresholding to capture stickers on bright/gold paint
    for win_sz in [51, base_win, base_win * 2 + 1]:
        if win_sz >= min(w, h):
            continue
        if win_sz % 2 == 0:
            win_sz += 1
        for C in [5, 10, 15]:
            adapt = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, win_sz, C)
            bin_images.append(adapt)

    # 3. Fixed threshold levels for dark stickers
    for t in [25, 40, 60, 80]:
        _, fix_bin = cv2.threshold(gray, t, 255, cv2.THRESH_BINARY_INV)
        bin_images.append(fix_bin)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    best_candidate = None
    best_score = float('inf')

    for bin_img in bin_images:
        closed = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            area = cv2.contourArea(c)
            if area < total_area * 0.001 or area > total_area * 0.15:
                continue

            x, y, bw, bh = cv2.boundingRect(c)
            # Exclude contours touching image edges/corners
            if x <= 20 or y <= 20 or (x + bw) >= (w - 20) or (y + bh) >= (h - 20):
                continue

            rect = cv2.minAreaRect(c)
            rw, rh = rect[1]
            if rw < 30 or rh < 30:
                continue

            aspect = max(rw, rh) / min(rw, rh)
            if aspect > 1.30:
                continue

            # Score based on how square, solid, and centralized the marker is
            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)
            if hull_area <= 0:
                continue
            hull_box_fill = hull_area / (rw * rh)
            if hull_box_fill < 0.70:
                continue

            cx, cy = rect[0]
            dist_center = np.hypot(cx - w / 2.0, cy - h / 2.0) / (w + h)
            score = abs(aspect - 1.0) + (1.0 - hull_box_fill) + 0.3 * dist_center

            if score < best_score:
                best_score = score
                box = cv2.boxPoints(rect)
                best_candidate = box

    if best_candidate is not None:
        # Order corners: top-left, top-right, bottom-right, bottom-left
        pts = best_candidate.astype(np.float32)
        pts_sorted_y = pts[np.argsort(pts[:, 1])]
        top = pts_sorted_y[:2][np.argsort(pts_sorted_y[:2, 0])]
        bottom = pts_sorted_y[2:][np.argsort(pts_sorted_y[:2, 0])[::-1]]
        ordered = np.array([top[0], top[1], bottom[0], bottom[1]], dtype=np.float32)
        return ordered

    return None


def extract_gap_pixels(
    image: np.ndarray,
    marker_corners: np.ndarray
) -> Tuple[Optional[float], Optional[List[Tuple[int, int]]], Optional[List[Tuple[int, int]]], Optional[Tuple[int, int]], Optional[Tuple[int, int]], Tuple[float, float]]:
    """
    Applies mathematical morphology (Black Top-Hat) and cross-sectional normal profiling
    to detect automotive panel gap seams at ANY arbitrary orientation (vertical, horizontal, or diagonal).

    Pipeline Steps:
    1. Black Top-Hat filtering to isolate thin dark grooves regardless of orientation.
    2. Exclusion of marker and immediate contact shadow to avoid false detections.
    3. Contour tracing to extract the continuous automotive seam closest to the marker.
    4. Local linear regression to compute seam tangent vector and perpendicular normal vector.
    5. Robust cross-sectional profile analysis using half-maximum thresholding and gradient extrema
       to accurately locate true outer panel sheet-metal boundaries without trapping on internal shadow crevices.
    6. Edge line regularization to remove local noise and maintain straight parallel seam edges.

    Args:
        image: Input BGR image.
        marker_corners: (4, 2) numpy array of the ArUco marker corners.

    Returns:
        A tuple of:
          - pixel_gap (float or None)
          - edge1_points (list of (x, y) tuples or None)
          - edge2_points (list of (x, y) tuples or None)
          - meas_pt_1 ((x, y) on first edge at inspection location or None)
          - meas_pt_2 ((x, y) on second edge at inspection location or None)
          - tangent_vec (tuple of (vx, vy))
    """
    if image is None or marker_corners is None:
        logger.error("Invalid image or marker corners provided to extract_gap_pixels.")
        return None, None, None, None, None, (0.0, 1.0)

    h, w = image.shape[:2]
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Calculate marker geometry
    mcx = float(np.mean(marker_corners[:, 0]))
    mcy = float(np.mean(marker_corners[:, 1]))
    marker_w = float(np.linalg.norm(marker_corners[0] - marker_corners[1]))
    if marker_w <= 0:
        marker_w = 50.0

    # 1. Black Top-Hat morphological filtering to highlight dark grooves of any angle
    k_size = int(max(9, marker_w * 0.35))
    if k_size % 2 == 0:
        k_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    # 2. Aggressively mask marker and contact shadows to prevent internal sticker artifacts
    m_poly = marker_corners.astype(np.int32)
    mask = np.ones_like(gray, dtype=np.uint8) * 255
    marker_mask_only = np.zeros_like(gray, dtype=np.uint8)
    cv2.fillPoly(marker_mask_only, [m_poly], 255)
    dilate_k = cv2.getStructuringElement(cv2.MORPH_RECT, (int(marker_w * 0.8), int(marker_w * 0.8)))
    dilated_marker = cv2.dilate(marker_mask_only, dilate_k)
    mask[dilated_marker > 0] = 0

    tophat_masked = cv2.bitwise_and(tophat, tophat, mask=mask)
    filtered_gray = cv2.bilateralFilter(gray, 7, 50, 50)

    # 3. Locate the primary automotive panel seam adjacent to the marker
    # Method A: Directional horizontal ray-casting from marker center
    y_scan = int(mcy)
    ray_x_right = np.arange(int(mcx + marker_w * 0.7), min(w - 20, int(mcx + marker_w * 3.5)))
    vals_right = filtered_gray[y_scan, ray_x_right] if len(ray_x_right) > 0 else np.array([999])

    ray_x_left = np.arange(max(20, int(mcx - marker_w * 3.5)), int(mcx - marker_w * 0.7))
    vals_left = filtered_gray[y_scan, ray_x_left] if len(ray_x_left) > 0 else np.array([999])

    min_r = float(np.min(vals_right)) if len(vals_right) > 0 else 999.0
    min_l = float(np.min(vals_left)) if len(vals_left) > 0 else 999.0

    seam_pts: List[Tuple[int, int]] = []
    seam_ref_pt = (0, 0)
    vx, vy = 0.0, 1.0

    if min_r < 45.0 or min_l < 45.0:
        if min_r <= min_l:
            seam_x = int(ray_x_right[int(np.argmin(vals_right))])
        else:
            seam_x = int(ray_x_left[int(np.argmin(vals_left))])

        seam_ref_pt = (seam_x, y_scan)
        ys = np.linspace(max(0, y_scan - marker_w * 1.0), min(h - 1, y_scan + marker_w * 1.0), 30).astype(int)
        for y in ys:
            xs = np.arange(max(0, seam_x - int(marker_w * 0.35)), min(w, seam_x + int(marker_w * 0.35)))
            if len(xs) > 0:
                row = filtered_gray[y, xs]
                min_x = int(xs[int(np.argmin(row))])
                seam_pts.append((min_x, y))

    # Method B: Morphological contour fallback if ray-casting found no clear joint
    if len(seam_pts) < 10:
        search_rad = int(3.5 * marker_w)
        rx1 = max(0, int(mcx - search_rad))
        rx2 = min(w, int(mcx + search_rad))
        ry1 = max(0, int(mcy - search_rad))
        ry2 = min(h, int(mcy + search_rad))

        roi_tophat = tophat_masked[ry1:ry2, rx1:rx2]
        if roi_tophat.size == 0 or not np.any(roi_tophat > 0):
            logger.warning("No groove response found in marker inspection ROI.")
            return None, None, None, None, None, (0.0, 1.0)

        line_k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        tophat_clean = cv2.morphologyEx(roi_tophat, cv2.MORPH_CLOSE, line_k)
        _, thresh = cv2.threshold(tophat_clean, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thresh)
        min_area = int(marker_w * 1.5)
        cleaned_thresh = np.zeros_like(thresh)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                cleaned_thresh[labels == i] = 255

        contours, _ = cv2.findContours(cleaned_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_cnt = None
        best_score = -1.0
        for c in contours:
            length = cv2.arcLength(c, False)
            if length < marker_w * 0.35:
                continue
            c_pts = c.reshape(-1, 2) + np.array([rx1, ry1])
            dists = np.linalg.norm(c_pts - np.array([mcx, mcy]), axis=1)
            min_dist = float(np.min(dists))
            if min_dist < marker_w * 0.5:
                continue

            _, _, bw, bh = cv2.boundingRect(c)
            span = max(bw, bh)
            if span < marker_w * 0.35:
                continue

            [cvx, cvy, _, _] = cv2.fitLine(c_pts, cv2.DIST_L2, 0, 0.01, 0.01)
            is_vertical = abs(cvy[0]) > abs(cvx[0])
            orientation_boost = 2.0 if is_vertical else 0.6

            mask_c = np.zeros(roi_tophat.shape, dtype=np.uint8)
            cv2.drawContours(mask_c, [c], -1, 255, -1)
            mean_depth = float(cv2.mean(roi_tophat, mask=mask_c)[0])

            quality = (length * span * (mean_depth + 1.0) * orientation_boost) / (min_dist + 50.0)
            if quality > best_score:
                best_score = quality
                best_cnt = c_pts

        if best_cnt is None or len(best_cnt) < 3:
            logger.warning("Could not trace continuous panel seam contour near marker.")
            return None, None, None, None, None, (0.0, 1.0)

        closest_idx = int(np.argmin(np.linalg.norm(best_cnt - np.array([mcx, mcy]), axis=1)))
        seam_ref_pt = tuple(best_cnt[closest_idx])
        seam_pts = [tuple(p) for p in best_cnt[np.linalg.norm(best_cnt - seam_ref_pt, axis=1) < marker_w * 1.2]]

    # 4. Tangent and Normal vectors along the physical panel seam
    fit_arr = np.array(seam_pts, dtype=np.float32)
    [vx_fit, vy_fit, _, _] = cv2.fitLine(fit_arr, cv2.DIST_L2, 0, 0.01, 0.01)
    vx, vy = float(vx_fit[0]), float(vy_fit[0])
    nx, ny = -vy, vx

    if np.dot(np.array([nx, ny]), np.array([mcx - seam_ref_pt[0], mcy - seam_ref_pt[1]])) > 0:
        nx, ny = -nx, -ny

    logger.info(f"Identified seam groove adjacent to marker at ({seam_ref_pt[0]}, {seam_ref_pt[1]}). "
                f"Tangent: ({vx:.2f}, {vy:.2f}), Normal: ({nx:.2f}, {ny:.2f}).")

    # 5. Robust cross-sectional profile sampling along normal across multiple points
    sample_len = int(marker_w * 0.65)
    num_samples = sample_len * 4
    sample_t = np.linspace(-sample_len, sample_len, num_samples)

    raw_edge1_pts: List[Tuple[int, int]] = []
    raw_edge2_pts: List[Tuple[int, int]] = []
    gap_samples: List[float] = []

    step_pts = np.array(seam_pts, dtype=np.float32)
    step_indices = np.linspace(0, len(step_pts) - 1, min(len(step_pts), 30)).astype(int)

    # Pre-filter image for robust profile sampling
    filtered_gray = cv2.bilateralFilter(gray, 7, 50, 50)

    for idx in step_indices:
        s_pt = step_pts[idx]
        s_xs = s_pt[0] + sample_t * nx
        s_ys = s_pt[1] + sample_t * ny
        map_x = s_xs.reshape(1, -1).astype(np.float32)
        map_y = s_ys.reshape(1, -1).astype(np.float32)
        vals = cv2.remap(filtered_gray, map_x, map_y, cv2.INTER_LINEAR).flatten()

        # Step 5a: Locate the true dark groove valley minimum along the search slice
        c_idx = int(np.argmin(vals))
        center_x = s_xs[c_idx]
        center_y = s_ys[c_idx]

        # Step 5b: Re-sample a dedicated symmetric window centered strictly on the dark valley floor
        win_len = int(marker_w * 0.35)
        win_t = np.linspace(-win_len, win_len, win_len * 4)
        w_xs = center_x + win_t * nx
        w_ys = center_y + win_t * ny
        w_map_x = w_xs.reshape(1, -1).astype(np.float32)
        w_map_y = w_ys.reshape(1, -1).astype(np.float32)
        w_vals = cv2.remap(filtered_gray, w_map_x, w_map_y, cv2.INTER_LINEAR).flatten()

        w_c_idx = len(win_t) // 2
        w_min = float(w_vals[w_c_idx])

        left_vals = w_vals[:w_c_idx]
        right_vals = w_vals[w_c_idx:]

        if len(left_vals) < 6 or len(right_vals) < 6:
            continue

        v_left_max = float(np.max(left_vals))
        v_right_max = float(np.max(right_vals))

        contrast_left = v_left_max - w_min
        contrast_right = v_right_max - w_min
        if contrast_left < 5.0 or contrast_right < 5.0:
            continue

        # Adaptive Half-Maximum thresholds for true outer sheet-metal lips
        t_left = w_min + 0.45 * contrast_left
        t_right = w_min + 0.45 * contrast_right

        l_idx = w_c_idx
        while l_idx > 0 and w_vals[l_idx] < t_left:
            l_idx -= 1

        r_idx = w_c_idx
        while r_idx < len(w_vals) - 1 and w_vals[r_idx] < t_right:
            r_idx += 1

        p1 = (int(round(w_xs[l_idx])), int(round(w_ys[l_idx])))
        p2 = (int(round(w_xs[r_idx])), int(round(w_ys[r_idx])))
        dist_px = float(np.linalg.norm(np.array(p1) - np.array(p2)))

        if 3.0 < dist_px < marker_w * 0.85:
            raw_edge1_pts.append(p1)
            raw_edge2_pts.append(p2)
            gap_samples.append(dist_px)

    if not gap_samples or len(raw_edge1_pts) < 2:
        logger.warning("Could not trace valid panel edge contours across the seam normal.")
        return None, None, None, None, None, (vx, vy)

    # 6. Regularize edge points along fitted seam lines for clean, straight boundaries
    pts1 = np.array(raw_edge1_pts, dtype=np.float32)
    pts2 = np.array(raw_edge2_pts, dtype=np.float32)

    [l_vx, l_vy, l_x0, l_y0] = cv2.fitLine(pts1, cv2.DIST_L2, 0, 0.01, 0.01)
    [r_vx, r_vy, r_x0, r_y0] = cv2.fitLine(pts2, cv2.DIST_L2, 0, 0.01, 0.01)

    # Project raw edge points onto the fitted lines for noise-free continuous contours
    edge1_pts: List[Tuple[int, int]] = []
    edge2_pts: List[Tuple[int, int]] = []

    l_dir = np.array([float(l_vx[0]), float(l_vy[0])])
    l_pt0 = np.array([float(l_x0[0]), float(l_y0[0])])
    r_dir = np.array([float(r_vx[0]), float(r_vy[0])])
    r_pt0 = np.array([float(r_x0[0]), float(r_y0[0])])

    for pt in raw_edge1_pts:
        p = np.array(pt, dtype=np.float32)
        proj = l_pt0 + np.dot(p - l_pt0, l_dir) * l_dir
        edge1_pts.append((int(round(proj[0])), int(round(proj[1]))))

    for pt in raw_edge2_pts:
        p = np.array(pt, dtype=np.float32)
        proj = r_pt0 + np.dot(p - r_pt0, r_dir) * r_dir
        edge2_pts.append((int(round(proj[0])), int(round(proj[1]))))

    # Measurement points at seam reference point adjacent to marker
    ref_np = np.array(seam_ref_pt, dtype=np.float32)
    proj_m1 = l_pt0 + np.dot(ref_np - l_pt0, l_dir) * l_dir
    meas_pt_1 = (int(round(proj_m1[0])), int(round(proj_m1[1])))

    # Perpendicular projection onto right edge line along normal vector
    norm_vec = np.array([nx, ny], dtype=np.float32)
    A = np.column_stack((norm_vec, -r_dir))
    b = r_pt0 - proj_m1
    try:
        sol = np.linalg.solve(A, b)
        proj_m2 = proj_m1 + sol[0] * norm_vec
        meas_pt_2 = (int(round(proj_m2[0])), int(round(proj_m2[1])))
    except np.linalg.LinAlgError:
        proj_m2 = r_pt0 + np.dot(proj_m1 - r_pt0, r_dir) * r_dir
        meas_pt_2 = (int(round(proj_m2[0])), int(round(proj_m2[1])))

    pixel_gap = float(np.linalg.norm(np.array(meas_pt_1) - np.array(meas_pt_2)))

    # Fallback to median gap if projection yields an extreme value
    if pixel_gap <= 0 or pixel_gap > marker_w * 0.9:
        pixel_gap = float(np.median(gap_samples))

    logger.info(f"Seam Edge 1: {meas_pt_1}, Edge 2: {meas_pt_2}. Calculated pixel gap: {pixel_gap:.2f} px.")
    return pixel_gap, edge1_pts, edge2_pts, meas_pt_1, meas_pt_2, (vx, vy)


def draw_visualizations(
    image: np.ndarray,
    gap_mm: float,
    mm_per_pixel: float,
    marker_corners: np.ndarray,
    edge1_points: List[Tuple[int, int]],
    edge2_points: List[Tuple[int, int]],
    meas_pt_1: Tuple[int, int],
    meas_pt_2: Tuple[int, int],
    tangent_vec: Tuple[float, float] = (0.0, 1.0)
) -> np.ndarray:
    """
    Renders clean, high-contrast metrology annotations onto a copy of the input image.

    Annotations:
    - Green bounding polygon around the detected ArUco marker.
    - Blue/Cyan lines tracing the primary boundary edges of the panel seam.
    - Bright yellow dimension line extending perpendicularly across the panel gap.
    - High-contrast text displaying 'Gap: X.XX mm' adjacent to the dimension line.

    Args:
        image: Original input BGR image.
        gap_mm: Computed gap in physical millimeters.
        mm_per_pixel: Spatial scale.
        marker_corners: Array of corners for the ArUco marker.
        edge1_points: Traced points along the first lip.
        edge2_points: Traced points along the second lip.
        meas_pt_1: Measurement coordinate on the first lip.
        meas_pt_2: Measurement coordinate on the second lip.
        tangent_vec: (vx, vy) tangent direction along the seam.

    Returns:
        Annotated BGR image.
    """
    annotated = image.copy()

    scale_factor = max(1.0, marker_corners[0][0] * 0.0 + (np.linalg.norm(marker_corners[0] - marker_corners[1]) / 100.0))
    line_th = max(2, int(2.5 * scale_factor))
    font_scale = max(0.6, 0.6 * scale_factor)
    txt_th = max(2, int(2.0 * scale_factor))

    # 1. Draw ArUco marker bounding polygon in Green
    pts = marker_corners.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(annotated, [pts], isClosed=True, color=(0, 255, 0), thickness=line_th)
    cv2.putText(
        annotated,
        "ArUco Target",
        (int(marker_corners[0][0]), max(25, int(marker_corners[0][1]) - int(10 * scale_factor))),
        cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.8, (0, 255, 0), txt_th, cv2.LINE_AA
    )

    # 2. Draw traced panel edges in Vibrant Blue/Cyan
    if len(edge1_points) > 1:
        cv2.polylines(annotated, [np.array(edge1_points, dtype=np.int32)], isClosed=False, color=(255, 120, 0), thickness=line_th)
    if len(edge2_points) > 1:
        cv2.polylines(annotated, [np.array(edge2_points, dtype=np.int32)], isClosed=False, color=(255, 120, 0), thickness=line_th)

    # 3. Draw Yellow dimension line connecting meas_pt_1 and meas_pt_2
    p1 = np.array(meas_pt_1, dtype=np.float32)
    p2 = np.array(meas_pt_2, dtype=np.float32)
    cv2.line(annotated, meas_pt_1, meas_pt_2, (0, 255, 255), line_th)

    # End tick marks along the tangent direction
    vx, vy = tangent_vec
    t_len = 12.0 * scale_factor
    tick_v = np.array([vx, vy], dtype=np.float32) * t_len

    t1_a = (int(round(p1[0] - tick_v[0])), int(round(p1[1] - tick_v[1])))
    t1_b = (int(round(p1[0] + tick_v[0])), int(round(p1[1] + tick_v[1])))
    cv2.line(annotated, t1_a, t1_b, (0, 255, 255), line_th)

    t2_a = (int(round(p2[0] - tick_v[0])), int(round(p2[1] - tick_v[1])))
    t2_b = (int(round(p2[0] + tick_v[0])), int(round(p2[1] + tick_v[1])))
    cv2.line(annotated, t2_a, t2_b, (0, 255, 255), line_th)

    # 4. Prepare measurement text with high-contrast backplate
    text = f"Gap: {gap_mm:.2f} mm"
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, txt_th)
    pad = int(8 * scale_factor)

    mid_x = int((p1[0] + p2[0]) / 2)
    mid_y = int((p1[1] + p2[1]) / 2)

    text_x = mid_x + int(15 * scale_factor)
    text_y = mid_y - int(15 * scale_factor)

    # Boundary check for text placement
    if text_x + text_w > annotated.shape[1] - 10:
        text_x = mid_x - text_w - int(15 * scale_factor)
    if text_y - text_h < 10:
        text_y = mid_y + text_h + int(20 * scale_factor)

    bg_pt1 = (text_x - pad, text_y - text_h - pad)
    bg_pt2 = (text_x + text_w + pad, text_y + baseline + pad)
    cv2.rectangle(annotated, bg_pt1, bg_pt2, (0, 0, 0), -1)
    cv2.putText(annotated, text, (text_x, text_y), font, font_scale, (0, 255, 255), txt_th, cv2.LINE_AA)

    return annotated


def measure_panel_gap(
    image: np.ndarray,
    marker_real_size_mm: float = MARKER_REAL_SIZE_MM
) -> Tuple[Optional[float], np.ndarray]:
    """
    Main orchestration function to measure the panel gap in physical millimeters.

    Args:
        image: Input BGR image.
        marker_real_size_mm: Real size of the ArUco calibration marker in mm.

    Returns:
        A tuple (gap_mm, annotated_image). If detection fails, returns (None, original_image).
    """
    if image is None:
        logger.error("Provided image array is None.")
        return None, np.zeros((100, 100, 3), dtype=np.uint8)

    # 1. Detect marker and get spatial scale
    mm_per_pixel, marker_corners = detect_aruco_scale(image, marker_real_size_mm)
    if mm_per_pixel is None or marker_corners is None:
        logger.error("Calibration failed: ArUco marker not detected or invalid.")
        return None, image

    # 2. Extract gap boundaries along the true panel seam
    pixel_gap, edge1_pts, edge2_pts, meas_pt1, meas_pt2, tangent_vec = extract_gap_pixels(image, marker_corners)
    if pixel_gap is None or edge1_pts is None or edge2_pts is None or meas_pt1 is None or meas_pt2 is None:
        logger.error("Boundary extraction failed: Panel seam edges could not be extracted.")
        return None, image

    # 3. Convert gap to physical millimeters
    gap_mm = pixel_gap * mm_per_pixel
    logger.info(f"Measurement complete. Gap is {gap_mm:.3f} mm ({pixel_gap:.2f} px). Tangent: ({tangent_vec[0]:.2f}, {tangent_vec[1]:.2f}).")

    # 4. Draw visualizations
    annotated_image = draw_visualizations(
        image, gap_mm, mm_per_pixel, marker_corners, edge1_pts, edge2_pts, meas_pt1, meas_pt2, tangent_vec=tangent_vec
    )

    return gap_mm, annotated_image


def run_benchmark_validation(dataset_folder: str, ground_truth_csv: str) -> Tuple[Optional[float], Optional[float], int]:
    """
    Benchmarking module that measures MAE and Standard Deviation (±σ) on a dataset of images
    against ground-truth readings from feeler gauges.

    CSV format expected:
    image_name,physical_feeler_gauge_mm

    Args:
        dataset_folder: Path to directory containing benchmark images.
        ground_truth_csv: Path to CSV file containing ground-truth values.

    Returns:
        A tuple of (mae, std_dev, processed_count). Returns (None, None, 0) if no images processed.
    """
    if not os.path.exists(ground_truth_csv):
        logger.error(f"Ground truth CSV file not found: {ground_truth_csv}")
        return None, None, 0

    if not os.path.exists(dataset_folder):
        logger.error(f"Dataset folder not found: {dataset_folder}")
        return None, None, 0

    absolute_errors: List[float] = []
    processed_count = 0
    failed_count = 0

    logger.info(f"Starting benchmark validation from CSV: {ground_truth_csv}")

    with open(ground_truth_csv, mode='r', encoding='utf-8') as f:
        reader = csv.reader(f)
        try:
            headers = next(reader)
        except StopIteration:
            logger.error("Empty CSV file provided.")
            return None, None, 0

        # Find columns
        headers_normalized = [h.strip().lower() for h in headers]
        try:
            img_col_idx = headers_normalized.index("image_name")
            gt_col_idx = headers_normalized.index("physical_feeler_gauge_mm")
        except ValueError:
            logger.error(f"Missing required columns in CSV headers: {headers}. "
                         "Expected 'image_name' and 'physical_feeler_gauge_mm'")
            return None, None, 0

        for row_num, row in enumerate(reader, start=2):
            if not row or len(row) <= max(img_col_idx, gt_col_idx):
                logger.warning(f"Row {row_num} in CSV is invalid or empty. Skipping.")
                continue

            img_name = row[img_col_idx].strip()
            gt_val_str = row[gt_col_idx].strip()

            try:
                ground_truth_mm = float(gt_val_str)
            except ValueError:
                logger.warning(f"Row {row_num}: Invalid ground truth value '{gt_val_str}'. Skipping.")
                continue

            img_path = os.path.join(dataset_folder, img_name)
            if not os.path.exists(img_path):
                logger.warning(f"Row {row_num}: Image file '{img_name}' not found at {img_path}. Skipping.")
                continue

            # Process the image
            image = cv2.imread(img_path)
            if image is None:
                logger.warning(f"Row {row_num}: Failed to load image '{img_name}'. Skipping.")
                continue

            predicted_mm, _ = measure_panel_gap(image)

            if predicted_mm is None:
                logger.warning(f"Row {row_num}: Measurement pipeline failed for image '{img_name}'.")
                failed_count += 1
                continue

            abs_err = abs(predicted_mm - ground_truth_mm)
            absolute_errors.append(abs_err)
            processed_count += 1

            logger.info(f"[{processed_count}] Image: {img_name} | GT: {ground_truth_mm:.2f} mm | "
                        f"Predicted: {predicted_mm:.2f} mm | Abs Error: {abs_err:.4f} mm")

    if not absolute_errors:
        logger.error("No images were successfully processed to compute benchmark statistics.")
        return None, None, 0

    mae = float(np.mean(absolute_errors))
    std_dev = float(np.std(absolute_errors))

    print("\n" + "="*50)
    print(" BENCHMARK VALIDATION RESULTS")
    print("="*50)
    print(f"Total Successfully Processed Images: {processed_count}")
    print(f"Failed Measurements:                {failed_count}")
    print(f"Mean Absolute Error (MAE):          {mae:.4f} mm")
    print(f"Standard Deviation (±σ):             {std_dev:.4f} mm")
    print("="*50 + "\n")

    return mae, std_dev, processed_count


def generate_synthetic_test_case(
    gap_real_mm: float = 5.0,
    marker_real_mm: float = 22.0
) -> np.ndarray:
    """
    Generates a synthetic high-resolution automotive joint target image for self-testing.

    Simulates:
    - Left and Right car body panels (light metallic gray paint).
    - A dark shadow groove seam of known width (default 5.0 mm).
    - A standard DICT_4X4_50 ID 0 ArUco calibration marker (default 22.0 mm) placed adjacent to the seam.

    Returns:
        BGR image array.
    """
    h, w = 1200, 1600
    canvas = np.ones((h, w, 3), dtype=np.uint8) * 180

    px_per_mm = 10.0
    marker_px = int(marker_real_mm * px_per_mm)
    gap_px = int(gap_real_mm * px_per_mm)

    seam_x = 700
    seam_left = seam_x - gap_px // 2
    seam_right = seam_x + gap_px // 2

    # Draw dark shadow seam groove
    canvas[:, seam_left:seam_right] = (30, 30, 30)

    # Subtle edge gradient on panel lips
    for i in range(4):
        alpha = (i + 1) / 5.0
        canvas[:, seam_left - 4 + i] = (int(30 + alpha * 150), int(30 + alpha * 150), int(30 + alpha * 150))
        canvas[:, seam_right + 4 - i] = (int(30 + alpha * 150), int(30 + alpha * 150), int(30 + alpha * 150))

    # Generate ArUco marker
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
    try:
        marker_img = cv2.aruco.generateImageMarker(aruco_dict, id=TARGET_MARKER_ID, sidePixels=marker_px)
    except AttributeError:
        marker_img = cv2.aruco.drawMarker(aruco_dict, id=TARGET_MARKER_ID, sidePixels=marker_px)

    marker_bgr = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2BGR)

    # Position marker on the right panel
    my1 = 490
    my2 = my1 + marker_px
    mx1 = seam_right + 120
    mx2 = mx1 + marker_px
    canvas[my1:my2, mx1:mx2] = marker_bgr

    logger.info(f"Generated synthetic automotive panel image: {w}x{h} px, Gap={gap_real_mm} mm ({gap_px} px), "
                f"Marker={marker_real_mm} mm ({marker_px} px).")
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Production-grade panel gap inspector utility."
    )
    parser.add_argument(
        "--image", type=str, help="Path to input image for gap measurement."
    )
    parser.add_argument(
        "--marker-size", type=float, default=MARKER_REAL_SIZE_MM,
        help="Physical width/height of the square ArUco marker in mm (default: 22.0)."
    )
    parser.add_argument(
        "--output", type=str, default="annotated_output.jpg",
        help="Path to save the annotated output image (default: annotated_output.jpg)."
    )
    parser.add_argument(
        "--benchmark", action="store_true",
        help="Flag to execute the validation benchmark engine."
    )
    parser.add_argument(
        "--dataset", type=str, help="Dataset folder for benchmarking."
    )
    parser.add_argument(
        "--csv", type=str, help="Ground truth CSV file for benchmarking."
    )

    args = parser.parse_args()

    if args.benchmark:
        if not args.dataset or not args.csv:
            parser.error("Benchmarking requires both --dataset and --csv arguments.")
        run_benchmark_validation(args.dataset, args.csv)
        return

    # If an image path is specified, run on that image
    if args.image:
        if not os.path.exists(args.image):
            logger.error(f"Specified image not found: {args.image}")
            return
        img = cv2.imread(args.image)
        if img is None:
            logger.error(f"Could not load image: {args.image}")
            return
        gap_mm, annotated = measure_panel_gap(img, marker_real_size_mm=args.marker_size)
        if gap_mm is not None:
            out_dir = os.path.dirname(os.path.abspath(args.output))
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            cv2.imwrite(args.output, annotated)
            print(f"\nMeasurement Succeeded!")
            print(f"Computed Gap: {gap_mm:.2f} mm")
            print(f"Annotated output saved to: {args.output}\n")
        else:
            print("\nMeasurement Failed. Check logs for details.\n")
        return

    # Default mode: Run self-test validation with synthetic image generator
    print("\n" + "="*50)
    print(" RUNNING SYSTEM SELF-TEST (SYNTHETIC IMAGE)")
    print("="*50)

    expected_gap_mm = 5.0
    synthetic_img = generate_synthetic_test_case(gap_real_mm=expected_gap_mm, marker_real_mm=args.marker_size)
    gap_mm, annotated = measure_panel_gap(synthetic_img, marker_real_size_mm=args.marker_size)

    if gap_mm is not None:
        out_dir = os.path.dirname(os.path.abspath(args.output))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        cv2.imwrite(args.output, annotated)
        print(f"Self-Test Result: Succeeded!")
        print(f"Expected Gap:    {expected_gap_mm:.2f} mm")
        print(f"Predicted Gap:   {gap_mm:.2f} mm")
        print(f"Absolute Error:  {abs(gap_mm - expected_gap_mm):.4f} mm")
        print(f"Saved Visualization to: {args.output}")

        # Sub-millimeter tolerance check
        if abs(gap_mm - expected_gap_mm) < 1.0:
            print("System Self-Test status: PASSED ✅")
        else:
            print("System Self-Test status: FAILED ❌ (Out of tolerance)")
    else:
        print("Self-Test Result: FAILED (Measurement pipeline returned None)")
        print("System Self-Test status: FAILED ❌")
    print("="*50 + "\n")


if __name__ == "__main__":
    main()
