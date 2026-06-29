"""
cross_section_colorizer.py
==========================
OpenCV-based cross-section extraction and coloring module.

Two main capabilities:
  1. PDF EXTRACTION — Extracts individual cross-section images from
     23-series engineering PDF pages (multiple graphs per page).
  2. COLORING — Colors the cross-section regions on each image:
       - RED   = CUT  (existing ground above proposed grade — excavation)
       - GREEN = FILL (proposed grade above existing ground — embankment)
       - BLUE  = solid proposed-grade line (highlighted)

The coloring is purely visual assistance for the LLM.
The LLM + Shoelace pipeline still handles area calculation.

Two-pass coloring approach:
  PASS 1 — Per-column gap profiling (detects bridge abutments)
  PASS 2 — Production coloring with bridge-aware fill
"""

import cv2
import numpy as np
import os
import re
import shutil
import tempfile

import fitz  # PyMuPDF


# ═══════════════════════════════════════════════════════════════════════════════
# PDF → INDIVIDUAL CROSS-SECTION EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

def extract_cross_sections_from_pdf(pdf_bytes: bytes) -> list[dict]:
    """
    Extract individual cross-section images from a 23-series engineering PDF.

    Filters pages by drawing number (23-series) and "cross section" label,
    then crops each station's cross-section at high resolution.

    Args:
        pdf_bytes : Raw PDF file bytes.

    Returns:
        list[dict] : Each dict contains:
            - "img_bytes"  : PNG bytes of the extracted cross-section
            - "station"    : Station string (e.g. "23+50") or filename fallback
            - "page_num"   : Original PDF page number (1-indexed)
            - "index"      : Cross-section index within the page (1-indexed)
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    extracted = []

    for i in range(len(doc)):
        page = doc[i]
        rect = page.rect

        # Filter for 23-series cross sections
        br_rect = fitz.Rect(
            rect.width * 0.65, rect.height * 0.75,
            rect.width, rect.height
        )
        corner_text = page.get_text("text", clip=br_rect).strip()

        is_23_series = False
        for line in corner_text.split('\n'):
            clean_line = re.sub(
                r'(?i)DRAWING|NO\.?|DRG|[:\s]', '', line
            ).strip()
            if (
                (clean_line.startswith("23-") or clean_line.startswith("23"))
                and "+" not in clean_line
            ):
                is_23_series = True
                break

        if not (is_23_series
                and re.search(r'(?i)cross[- \s]*section', corner_text)):
            continue

        # Find station labels on the right strip
        right_strip = fitz.Rect(rect.width * 0.80, 0, rect.width, rect.height)
        station_labels = page.search_for("+", clip=right_strip)
        station_labels.sort(key=lambda x: x.y0)

        last_bottom_cut = 35

        for j, label in enumerate(station_labels):
            y_top = last_bottom_cut
            current_bottom_target = label.y1 + 48

            if j + 1 < len(station_labels):
                y_bottom = min(
                    current_bottom_target,
                    station_labels[j + 1].y0 - 20,
                )
            else:
                y_bottom = min(current_bottom_target, rect.height * 0.88)

            last_bottom_cut = y_bottom
            crop_rect = fitz.Rect(0, y_top, rect.width, y_bottom)

            # Build station name from text near the label
            text_area = fitz.Rect(
                rect.width * 0.80, label.y0 - 30,
                rect.width, label.y1 + 30,
            )
            sta_val = page.get_text("text", clip=text_area).strip()
            clean_name = re.sub(r'[^0-9+]', '', sta_val)
            if not clean_name:
                clean_name = f"sta_{j + 1}"

            # High-res render (3× zoom)
            pix = page.get_pixmap(matrix=fitz.Matrix(3, 3), clip=crop_rect)
            img_bytes = pix.tobytes("png")

            extracted.append({
                "img_bytes": img_bytes,
                "station": clean_name,
                "page_num": i + 1,
                "index": j + 1,
            })

    doc.close()
    return extracted


# ═══════════════════════════════════════════════════════════════════════════════
# GRID REMOVAL + SOLID/DOTTED MASK SEPARATION
# ═══════════════════════════════════════════════════════════════════════════════

def _build_masks(img_gray):
    """
    Separate the image into solid-line and dotted-line masks after
    removing background grid lines while preserving diagram overlaps.
    """
    h_img, w_img = img_gray.shape

    # Grid removal
    _, bw = cv2.threshold(img_gray, 235, 255, cv2.THRESH_BINARY_INV)
    ver_k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 150))
    hor_k = cv2.getStructuringElement(cv2.MORPH_RECT, (150, 1))
    grid_mask = cv2.add(
        cv2.morphologyEx(bw, cv2.MORPH_OPEN, ver_k),
        cv2.morphologyEx(bw, cv2.MORPH_OPEN, hor_k)
    )

    # Bridge-safe eraser: protect pixels where diagram overlaps the grid
    diagram_only  = cv2.subtract(bw, grid_mask)
    bridge_shield = cv2.dilate(diagram_only, np.ones((3, 3), np.uint8), iterations=1)
    eraser        = cv2.subtract(grid_mask, bridge_shield)
    clean_bw      = cv2.subtract(bw, eraser)

    # Connected component separation
    nlabels, labels, stats, _ = cv2.connectedComponentsWithStats(clean_bw, 8, cv2.CV_32S)

    border_w = 3
    border_h = 3
    scale_line_y_threshold = int(h_img * 0.88)

    solid_mask  = np.zeros_like(clean_bw)
    dotted_mask = np.zeros_like(clean_bw)

    for i in range(1, nlabels):
        x, y, w, h, area = stats[i]

        if x < border_w or (x + w) > (w_img - border_w):
            continue
        if y < border_h or (y + h) > (h_img - border_h):
            continue
        if y > scale_line_y_threshold:
            continue
        if h >= 16 and w <= 45:
            continue

        aspect_ratio = w / h if h > 0 else 0
        diag_len     = np.sqrt(w ** 2 + h ** 2)

        if 3 <= w <= 45 and 2 <= h <= 12 and aspect_ratio > 1.0:
            dotted_mask[labels == i] = 255
        elif diag_len > 55 or w > 50 or h > 50:
            solid_mask[labels == i] = 255

    return solid_mask, dotted_mask, eraser, scale_line_y_threshold


# ═══════════════════════════════════════════════════════════════════════════════
# PASS 1 — Per-column gap profiling
# ═══════════════════════════════════════════════════════════════════════════════

def _analyze_page_gaps(img_gray, masks=None):
    """
    PASS 1: Profile the vertical gap between the solid and dotted lines
    at each column. Large gaps indicate bridge abutments that should not
    be colored through.

    Returns:
        col_gap_profile : np.array of gap sizes per column
        masks           : cached (solid_mask, dotted_mask, eraser,
                          scale_line_y_threshold) for reuse in PASS 2
    """
    h_img, w_img = img_gray.shape
    if masks is None:
        masks = _build_masks(img_gray)
    solid_mask, dotted_mask, _, _ = masks

    heal_kernel     = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
    solid_boundary  = cv2.morphologyEx(solid_mask,  cv2.MORPH_CLOSE, heal_kernel)
    dotted_boundary = cv2.morphologyEx(dotted_mask, cv2.MORPH_CLOSE, heal_kernel)

    col_gap_profile = np.zeros(w_img, dtype=np.float32)

    # Vectorized: find columns where both solid and dotted exist
    solid_any  = (solid_boundary[:, 20:w_img - 20] == 255)
    dotted_any = (dotted_boundary[:, 20:w_img - 20] == 255)

    solid_col_has  = solid_any.any(axis=0)
    dotted_col_has = dotted_any.any(axis=0)
    both_have      = solid_col_has & dotted_col_has

    if np.any(both_have):
        cols_with_both = np.where(both_have)[0]
        for ci in cols_with_both:
            c = ci + 20  # offset back to original column index
            solid_rows  = np.where(solid_boundary[:, c] == 255)[0]
            dotted_rows = np.where(dotted_boundary[:, c] == 255)[0]
            top_solid_y     = solid_rows[0]
            bottom_dotted_y = dotted_rows[-1]
            if bottom_dotted_y > top_solid_y:
                col_gap_profile[c] = float(abs(top_solid_y - bottom_dotted_y))

    return col_gap_profile, masks


# ═══════════════════════════════════════════════════════════════════════════════
# PASS 2 — Production coloring with bridge-aware fill
# ═══════════════════════════════════════════════════════════════════════════════

def _process_image(img_gray, col_gap_profile, masks=None):
    """
    PASS 2: Color the cross-section regions on the image.

    Returns a BGR color image with:
      - Red regions   (CUT)
      - Green regions (FILL)
      - Blue solid line (proposed grade)
      - Black text preserved
    """
    h_img, w_img = img_gray.shape
    if masks is None:
        masks = _build_masks(img_gray)
    solid_mask, dotted_mask, eraser, scale_line_y_threshold = masks

    # Base output: grayscale → BGR, erased grid → white
    color_output = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    color_output[eraser > 0] = [255, 255, 255]

    heal_kernel     = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    solid_boundary  = cv2.morphologyEx(solid_mask,  cv2.MORPH_CLOSE, heal_kernel)
    dotted_boundary = cv2.morphologyEx(dotted_mask, cv2.MORPH_CLOSE, heal_kernel)

    red_overlay   = np.zeros_like(solid_mask)
    green_overlay = np.zeros_like(solid_mask)

    # Bridge gate: only enter fill logic where solid+dotted regions touch
    touch_kernel   = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated_solid  = cv2.dilate(solid_boundary,  touch_kernel, iterations=1)
    dilated_dotted = cv2.dilate(dotted_boundary, touch_kernel, iterations=1)
    intersection   = cv2.bitwise_and(dilated_solid, dilated_dotted)

    BRIDGE_GAP_PX  = 80
    look_window    = 25
    MAX_INTERP_GAP = 120

    if np.any(intersection > 0):
        col_has_fill    = np.zeros(w_img, dtype=bool)
        fill_top_y      = np.zeros(w_img, dtype=int)
        fill_bot_y      = np.zeros(w_img, dtype=int)
        fill_color_type = np.zeros(w_img, dtype=int)
        fill_guard_y    = np.full(w_img, -1, dtype=int)

        for c in range(w_img):
            solid_rows = np.where(solid_boundary[:, c] == 255)[0]

            if len(solid_rows) == 0:
                c_l = max(0, c - look_window)
                c_r = min(w_img, c + look_window + 1)
                nb  = np.where(solid_boundary[:, c_l:c_r] == 255)
                if len(nb[0]) == 0:
                    continue
                st = int(np.min(nb[0]))
                sb = int(np.max(nb[0]))
            else:
                st = int(np.min(solid_rows))
                sb = int(np.max(solid_rows))

            # Red: only search directly in this column (no neighbour search)
            dotted_above = np.where(dotted_boundary[:st, c] == 255)[0]

            # Green: allow neighbour search
            dotted_below = np.where(dotted_boundary[sb:, c] == 255)[0]
            if len(dotted_below) == 0:
                c_l  = max(0, c - look_window)
                c_r  = min(w_img, c + look_window + 1)
                nb_b = np.where(dotted_boundary[sb:, c_l:c_r] == 255)
                if len(nb_b[0]) > 0:
                    dotted_below = nb_b[0]

            is_bridge_col = col_gap_profile[c] > BRIDGE_GAP_PX if c < len(col_gap_profile) else False

            if len(dotted_above) > 0 and not is_bridge_col:
                boundary_y = int(np.max(dotted_above))
                if boundary_y < st:
                    col_has_fill[c]    = True
                    fill_top_y[c]      = boundary_y
                    fill_bot_y[c]      = st
                    fill_color_type[c] = 1

            elif len(dotted_below) > 0:
                boundary_y = int(np.min(dotted_below)) + sb
                if sb < boundary_y:
                    col_has_fill[c]    = True
                    fill_top_y[c]      = sb
                    fill_bot_y[c]      = boundary_y
                    fill_color_type[c] = 2
                    if is_bridge_col:
                        fill_guard_y[c] = st

        valid_indices = np.where(col_has_fill)[0]

        if len(valid_indices) > 1:
            start_idx = int(np.min(valid_indices))
            end_idx   = int(np.max(valid_indices))

            for c in range(start_idx, end_idx + 1):
                if col_has_fill[c]:
                    t_y    = fill_top_y[c]
                    b_y    = fill_bot_y[c]
                    c_type = fill_color_type[c]
                    guard  = fill_guard_y[c]
                else:
                    left_v  = valid_indices[valid_indices < c]
                    right_v = valid_indices[valid_indices > c]
                    if len(left_v) == 0 or len(right_v) == 0:
                        continue
                    l = int(left_v[-1])
                    r = int(right_v[0])
                    if (r - l) > MAX_INTERP_GAP:
                        continue

                    wt     = (c - l) / (r - l)
                    t_y    = int(fill_top_y[l] + wt * (fill_top_y[r] - fill_top_y[l]))
                    b_y    = int(fill_bot_y[l] + wt * (fill_bot_y[r] - fill_bot_y[l]))
                    c_type = fill_color_type[l] if wt <= 0.5 else fill_color_type[r]

                    g_l   = fill_guard_y[l] if fill_guard_y[l] >= 0 else t_y
                    g_r   = fill_guard_y[r] if fill_guard_y[r] >= 0 else t_y
                    guard = int(g_l + wt * (g_r - g_l)) if (fill_guard_y[l] >= 0 or fill_guard_y[r] >= 0) else -1

                if guard >= 0:
                    t_y = max(t_y, guard)

                if t_y < b_y:
                    if c_type == 1:
                        red_overlay[t_y:b_y, c] = 255
                    elif c_type == 2:
                        green_overlay[t_y:b_y, c] = 255

            # Horizontal row-gap stitching (vectorized via morphological close)
            MAX_H_GAP = 60
            h_stitch_kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT, (MAX_H_GAP, 1)
            )
            for overlay in [green_overlay, red_overlay]:
                region = overlay[:scale_line_y_threshold, start_idx:end_idx + 1]
                closed = cv2.morphologyEx(region, cv2.MORPH_CLOSE, h_stitch_kernel)
                overlay[:scale_line_y_threshold, start_idx:end_idx + 1] = closed

    # Smoothing
    hor_close = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
    ver_close = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))

    red_clean   = cv2.morphologyEx(red_overlay,   cv2.MORPH_CLOSE, hor_close)
    red_clean   = cv2.morphologyEx(red_clean,     cv2.MORPH_CLOSE, ver_close)
    green_clean = cv2.morphologyEx(green_overlay, cv2.MORPH_CLOSE, hor_close)
    green_clean = cv2.morphologyEx(green_clean,   cv2.MORPH_CLOSE, ver_close)

    # Apply colors
    color_output[red_clean   == 255] = [0, 0, 255]    # Red  = CUT
    color_output[green_clean == 255] = [0, 210, 0]    # Green = FILL
    color_output[solid_mask  == 255] = [255, 0, 0]    # Blue = solid line

    # Preserve text
    text_mask = (img_gray < 80) & (eraser == 0)
    color_output[text_mask] = [0, 0, 0]

    return color_output


# ═══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════════

def colorize_cross_section(img_bytes: bytes) -> bytes:
    """
    Colorize a single cross-section image.

    Takes raw PNG image bytes, runs the two-pass coloring pipeline,
    and returns colored PNG bytes. Used for direct image uploads.
    """
    np_arr   = np.frombuffer(img_bytes, np.uint8)
    img_gray = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)

    if img_gray is None:
        return img_bytes

    # PASS 1: gap profiling (returns cached masks)
    col_gap_profile, masks = _analyze_page_gaps(img_gray)

    # PASS 2: coloring (reuses cached masks — no redundant _build_masks)
    colored_bgr = _process_image(img_gray, col_gap_profile, masks=masks)

    _, encoded = cv2.imencode('.png', colored_bgr)
    return encoded.tobytes()


def extract_and_colorize_pdf(pdf_bytes: bytes) -> list[dict]:
    """
    Full pipeline for PDF input:
      1. Extract individual cross-section images from the PDF
      2. Run two-pass coloring on each extracted image

    Args:
        pdf_bytes : Raw PDF file bytes.

    Returns:
        list[dict] : Each dict contains:
            - "raw_bytes"     : Original extracted PNG bytes (before coloring)
            - "colored_bytes" : Colored PNG bytes
            - "station"       : Station string (e.g. "23+50")
            - "page_num"      : Original PDF page number (1-indexed)
            - "index"         : Cross-section index within page (1-indexed)
    """
    # Step 1: Extract individual cross-section images
    extracted = extract_cross_sections_from_pdf(pdf_bytes)

    if not extracted:
        return []

    # Step 2: Read all images and run both passes with mask caching
    results = []
    for item in extracted:
        raw_bytes = item["img_bytes"]

        np_arr   = np.frombuffer(raw_bytes, np.uint8)
        img_gray = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)

        if img_gray is None:
            results.append({
                "raw_bytes": raw_bytes,
                "colored_bytes": raw_bytes,
                "station": item["station"],
                "page_num": item["page_num"],
                "index": item["index"],
            })
            continue

        try:
            # PASS 1: gap profiling (caches masks)
            col_gap_profile, masks = _analyze_page_gaps(img_gray)

            # PASS 2: coloring (reuses cached masks)
            colored_bgr = _process_image(img_gray, col_gap_profile, masks=masks)

            _, encoded = cv2.imencode('.png', colored_bgr)
            colored_bytes = encoded.tobytes()
        except Exception:
            colored_bytes = raw_bytes

        results.append({
            "raw_bytes": raw_bytes,
            "colored_bytes": colored_bytes,
            "station": item["station"],
            "page_num": item["page_num"],
            "index": item["index"],
        })

    return results
