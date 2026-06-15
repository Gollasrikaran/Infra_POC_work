import fitz  # PyMuPDF
import cv2
import numpy as np
import os
import re


def extract_scale_report(pdf_path):
    """
    Scans a highway PDF for 19-series cross-section pages and pulls out
    horizontal/vertical scale info from each matching page.
    Returns a list of dicts like: [{"page": 42, "h_scale": "1\" = 10'", ...}]
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(pdf_path)
    results = []

    for i in range(len(doc)):
        page = doc[i]
        rect = page.rect

        # Read text from the bottom-right corner (where drawing numbers live)
        corner = fitz.Rect(rect.width * 0.70, rect.height * 0.80, rect.width, rect.height)
        corner_text = page.get_text("text", clip=corner).strip()

        # Check if drawing number starts with "19" (skip station numbers with "+")
        is_19 = False
        for line in corner_text.split('\n'):
            cleaned = re.sub(r'(?i)DRAWING|NO\.?|DRG|[:\s]', '', line).strip()
            if (cleaned.startswith("19-") or cleaned.startswith("19")) and "+" not in cleaned:
                is_19 = True
                break

        if not is_19:
            continue

        # Must also mention "cross section"
        if not re.search(r'(?i)cross[- \s]*section', corner_text):
            continue

        # Pull scale values from the bottom 30% of the page
        scale_area = fitz.Rect(0, rect.height * 0.70, rect.width, rect.height)
        text = page.get_text("text", clip=scale_area)

        h_match = re.search(r'(?i)HORIZONTAL[:\s]*1"\s*=\s*(\d+\'?)', text)
        v_match = re.search(r'(?i)VERTICAL[:\s]*1"\s*=\s*(\d+\'?)', text)

        results.append({
            "page": i + 1,
            "h_scale": f'1" = {h_match.group(1)}' if h_match else '1" = 10\' (default)',
            "v_scale": f'1" = {v_match.group(1)}' if v_match else '1" = 10\' (default)',
        })

    doc.close()
    return results


def extract_cross_section_images(pdf_path, output_dir, zoom=4.0, progress_callback=None):
    """
    Goes through a highway PDF, finds 19-series cross-section pages, and
    crops out individual station images at high resolution.
    Returns a list of dicts: [{"path": "...", "page": 42, "station": "125+00"}, ...]
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    os.makedirs(output_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    total = len(doc)
    extracted = []

    for i in range(total):
        page = doc[i]
        rect = page.rect

        # Same 19-series + cross-section filter
        corner = fitz.Rect(rect.width * 0.65, rect.height * 0.75, rect.width, rect.height)
        corner_text = page.get_text("text", clip=corner).strip()

        is_19 = False
        for line in corner_text.split('\n'):
            cleaned = re.sub(r'(?i)DRAWING|NO\.?|DRG|[:\s]', '', line).strip()
            if (cleaned.startswith("19-") or cleaned.startswith("19")) and "+" not in cleaned:
                is_19 = True
                break

        if not (is_19 and re.search(r'(?i)cross[- \s]*section', corner_text)):
            if progress_callback:
                progress_callback(i + 1, total)
            continue

        # Find station labels by searching for "+" in the right margin
        right_strip = fitz.Rect(rect.width * 0.80, 0, rect.width, rect.height)
        stations = page.search_for("+", clip=right_strip)
        stations.sort(key=lambda s: s.y0)

        last_cut = 35
        for j, label in enumerate(stations):
            y_top = last_cut
            y_bot = label.y1 + 48

            # Don't overlap with the next station
            if j + 1 < len(stations):
                y_bot = min(y_bot, stations[j + 1].y0 - 20)
            else:
                y_bot = min(y_bot, rect.height * 0.88)

            last_cut = y_bot
            crop = fitz.Rect(0, y_top, rect.width, y_bot)

            # Get station name from text near the label
            text_area = fitz.Rect(rect.width * 0.80, label.y0 - 30, rect.width, label.y1 + 30)
            sta_text = page.get_text("text", clip=text_area).strip()
            sta_name = re.sub(r'[^0-9+]', '', sta_text) or f"sta_{j + 1}"

            # Render at high resolution and save
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=crop)
            filename = f"Page{i + 1}_Sta_{sta_name}.png"
            save_path = os.path.join(output_dir, filename)
            pix.save(save_path)

            extracted.append({"path": save_path, "page": i + 1, "station": sta_name})

        if progress_callback:
            progress_callback(i + 1, total)

    doc.close()
    return extracted


def fill_earthwork_zones(img_gray, debug_dir=None):
    """
    Takes a grayscale cross-section image and identifies Cut vs Fill zones
    using precision grid-removal and column-scan fill logic.

    How it works:
      1. Removes background grid lines (those light engineering grids)
      2. Separates the solid design line from the dotted ground line
      3. Scans each column to determine if dotted is above or below solid
      4. Fills the area between them: red = cut (dotted above), green = fill (dotted below)
      5. Applies morphological smoothing for clean continuous regions
      6. Restores the original text and design lines on top

    Args:
        img_gray: Grayscale input image.
        debug_dir: If provided, saves intermediate images to this directory
                   for diagnosing pipeline issues.

    Returns a BGR color image with the overlays applied.
    """
    if debug_dir is not None:
        os.makedirs(debug_dir, exist_ok=True)

    h_img, w_img = img_gray.shape

    # --- STEP 1: GRID REMOVAL ---
    _, bw = cv2.threshold(img_gray, 235, 255, cv2.THRESH_BINARY_INV)

    if debug_dir is not None:
        cv2.imwrite(os.path.join(debug_dir, "1_binary.png"), bw)

    ver_k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 150))
    hor_k = cv2.getStructuringElement(cv2.MORPH_RECT, (150, 1))
    grid_mask = cv2.add(cv2.morphologyEx(bw, cv2.MORPH_OPEN, ver_k),
                        cv2.morphologyEx(bw, cv2.MORPH_OPEN, hor_k))

    if debug_dir is not None:
        cv2.imwrite(os.path.join(debug_dir, "2_grid_detected.png"), grid_mask)

    diagram_only = cv2.subtract(bw, grid_mask)
    bridge_shield = cv2.dilate(diagram_only, np.ones((3, 3), np.uint8), iterations=1)
    eraser = cv2.subtract(grid_mask, bridge_shield)
    clean_bw = cv2.subtract(bw, eraser)

    if debug_dir is not None:
        cv2.imwrite(os.path.join(debug_dir, "3_after_grid_removal.png"), clean_bw)

    # --- STEP 2: PREPARE OUTPUT & SEPARATE MASKS ---
    color_output = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    color_output[eraser > 0] = [255, 255, 255]  # Clean background

    nlabels, labels, stats, _ = cv2.connectedComponentsWithStats(clean_bw, 8, cv2.CV_32S)

    border_w = 3
    border_h = 3
    scale_line_y_threshold = int(h_img * 0.88)

    # Isolated clean masks extracted via connected component logic
    solid_mask = np.zeros_like(clean_bw)
    dotted_mask = np.zeros_like(clean_bw)

    for i in range(1, nlabels):
        x, y, w, h, area = stats[i]

        # Border protection
        if x < border_w or (x + w) > (w_img - border_w) or y < border_h or (y + h) > (h_img - border_h):
            continue

        # Ignore bottom scaling metrics
        if y > scale_line_y_threshold:
            continue

        # Arrow Interceptor
        if h >= 16 and w <= 45:
            continue

        diag_len = np.sqrt(w**2 + h**2)
        aspect_ratio = w / h if h > 0 else 0

        # 1. Dotted line identification (Extracted from Component Detection)
        if 3 <= w <= 45 and 2 <= h <= 12 and aspect_ratio > 1.0:
            dotted_mask[labels == i] = 255

        # 2. Strict solid line identification
        elif diag_len > 55 or w > 50 or h > 50:
            solid_mask[labels == i] = 255

    # --- STEP 3: TEXT GAP HEALING & FILL LOGIC ---
    # We apply horizontal healing elements to close mask splits created by numbers/text tags
    heal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
    solid_boundary = cv2.morphologyEx(solid_mask, cv2.MORPH_CLOSE, heal_kernel)
    dotted_boundary = cv2.morphologyEx(dotted_mask, cv2.MORPH_CLOSE, heal_kernel)

    if debug_dir is not None:
        cv2.imwrite(os.path.join(debug_dir, "4_design_mask.png"), solid_mask)
        cv2.imwrite(os.path.join(debug_dir, "5_dotted_mask.png"), dotted_boundary)

    red_overlay = np.zeros_like(clean_bw)
    green_overlay = np.zeros_like(clean_bw)

    # Scan EVERY column across the entire image canvas width using the healed boundaries
    for c in range(w_img):
        solid_rows = np.where(solid_boundary[:, c] == 255)[0]
        if len(solid_rows) == 0:
            continue

        st = np.min(solid_rows)
        sb = np.max(solid_rows)

        # Look above and below the solid line bounds in this specific column
        dotted_above = np.where(dotted_boundary[:st, c] == 255)[0]
        dotted_below = np.where(dotted_boundary[sb:, c] == 255)[0]

        # STRICT COLORING CONDITION RULE
        if len(dotted_above) > 0:
            # Dotted line is ABOVE -> Color the empty space completely RED (cut)
            boundary_y = np.max(dotted_above)
            red_overlay[boundary_y:st, c] = 255
        elif len(dotted_below) > 0:
            # Solid line is ABOVE (Dotted is below) -> Color the space completely GREEN (fill)
            boundary_y = np.min(dotted_below) + sb
            green_overlay[sb:boundary_y, c] = 255

    # --- STEP 4: GLOBAL CORNER SMOOTHING ENGINE ---
    gap_threshold = 60
    hor_close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (gap_threshold, 1))

    red_overlay_clean = cv2.morphologyEx(red_overlay, cv2.MORPH_CLOSE, hor_close_kernel)
    green_overlay_clean = cv2.morphologyEx(green_overlay, cv2.MORPH_CLOSE, hor_close_kernel)

    # Secondary closures to lock overlay bounds to the exact outer corners
    corner_kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
    smooth_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  # Round edges prevent jagged stair-stepping

    # Execute precise structural edge expansions
    green_overlay_clean = cv2.morphologyEx(green_overlay_clean, cv2.MORPH_CLOSE, corner_kernel_v)
    green_overlay_clean = cv2.dilate(green_overlay_clean, smooth_kernel, iterations=1)

    red_overlay_clean = cv2.morphologyEx(red_overlay_clean, cv2.MORPH_CLOSE, corner_kernel_v)
    red_overlay_clean = cv2.dilate(red_overlay_clean, smooth_kernel, iterations=1)

    # Stamp the polished continuous color fills onto our final canvas image
    color_output[red_overlay_clean == 255] = [0, 0, 255]     # Pure Red (Dotted line above = cut)
    color_output[green_overlay_clean == 255] = [0, 210, 0]   # Pure Green (Solid line above = fill)

    # --- STEP 5: RESTORE ORIGINAL DESIGN LINES (BLUE) & EMBEDDED TEXT ---
    color_output[solid_mask == 255] = [255, 0, 0]             # Pure Blue (design profile overlaid cleanly)

    text_mask = (img_gray < 80) & (eraser == 0)
    color_output[text_mask] = [0, 0, 0]

    if debug_dir is not None:
        cv2.imwrite(os.path.join(debug_dir, "6_final.png"), color_output)

    return color_output