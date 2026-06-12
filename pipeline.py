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


def fill_earthwork_zones(img_gray):
    """
    Takes a grayscale cross-section image and identifies Cut vs Fill zones.

    How it works:
      1. Removes background grid lines (those light engineering grids)
      2. Separates the solid design line from the dotted ground line
      3. Fills the area between them: red = cut, green = fill
      4. Restores the original text and design lines on top

    Returns a BGR color image with the overlays applied.
    """
    h_img, w_img = img_gray.shape

    # --- Grid removal ---
    # Threshold to get all dark marks, then isolate grid lines using morphology
    _, bw = cv2.threshold(img_gray, 235, 255, cv2.THRESH_BINARY_INV)

    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 150))
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (150, 1))
    grid = cv2.add(
        cv2.morphologyEx(bw, cv2.MORPH_OPEN, vert_kernel),
        cv2.morphologyEx(bw, cv2.MORPH_OPEN, horiz_kernel),
    )

    # Protect actual drawing content from being erased along with the grid
    drawing = cv2.subtract(bw, grid)
    shield = cv2.dilate(drawing, np.ones((3, 3), np.uint8), iterations=1)
    eraser = cv2.subtract(grid, shield)
    clean = cv2.subtract(bw, eraser)

    # Start building the color output
    output = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    output[eraser > 0] = [255, 255, 255]  # white out the grid

    # --- Classify line components ---
    # Separate dotted ground line from solid design line based on shape/size
    nlabels, labels, stats, _ = cv2.connectedComponentsWithStats(clean, 8, cv2.CV_32S)

    border = 3
    cutoff_y = int(h_img * 0.88)  # ignore anything below the scale bar

    solid_mask = np.zeros_like(clean)
    dotted_mask = np.zeros_like(clean)

    for i in range(1, nlabels):
        x, y, w, h, area = stats[i]

        # Skip border and scale bar components
        if x < border or (x + w) > (w_img - border):
            continue
        if y < border or (y + h) > (h_img - border):
            continue
        if y > cutoff_y:
            continue
        if h >= 16 and w <= 45:  # skip text-like vertical strokes
            continue

        diag = np.sqrt(w**2 + h**2)
        aspect = w / h if h > 0 else 0

        # Small, wide-ish blobs -> dotted line segments
        if 3 <= w <= 45 and 2 <= h <= 12 and aspect > 1.0:
            dotted_mask[labels == i] = 255
        # Large components -> solid design line
        # But skip compact text labels: short height (< 15px) with moderate width
        # are likely text/numbers, not the design profile
        elif diag > 55 or w > 50 or h > 50:
            if h < 15 and w < 120:
                continue  # skip text labels like "652.71", station numbers, etc.
            solid_mask[labels == i] = 255

    # --- Heal small gaps in the boundary lines ---
    heal_k = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    solid_boundary = cv2.morphologyEx(solid_mask, cv2.MORPH_CLOSE, heal_k)
    dotted_boundary = cv2.morphologyEx(dotted_mask, cv2.MORPH_CLOSE, heal_k)

    # --- Area-based dotted contour filtering ---
    # Remove tiny noise contours while preserving all valid ground profile segments.
    # Unlike keeping only the largest contour, this preserves broken/multi-segment profiles.
    MIN_CONTOUR_AREA = 200
    contours, _ = cv2.findContours(
        dotted_boundary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    filtered_dotted = np.zeros_like(dotted_boundary)
    for cnt in contours:
        if cv2.contourArea(cnt) >= MIN_CONTOUR_AREA:
            cv2.drawContours(filtered_dotted, [cnt], -1, 255, thickness=cv2.FILLED)
    # Keep original boundary pixels within the filtered regions
    dotted_boundary = cv2.bitwise_and(dotted_boundary, filtered_dotted)

    # --- Define corridor as overlap of BOTH profiles ---
    # Colors should only appear where both solid and dotted lines are present.
    solid_cols = np.where(np.any(solid_boundary > 0, axis=0))[0]
    dotted_cols = np.where(np.any(dotted_boundary > 0, axis=0))[0]

    if len(solid_cols) > 0 and len(dotted_cols) > 0:
        margin = 30
        # Corridor = overlap of both profile X-ranges + small margin
        road_x_min = max(np.min(solid_cols), np.min(dotted_cols))
        road_x_max = min(np.max(solid_cols), np.max(dotted_cols))
        road_x_min = max(0, road_x_min - margin)
        road_x_max = min(w_img - 1, road_x_max + margin)

        # If overlap is too narrow, fall back to solid line extent
        if road_x_max - road_x_min < 20:
            road_x_min = max(0, np.min(solid_cols) - margin)
            road_x_max = min(w_img - 1, np.max(solid_cols) + margin)
    elif len(solid_cols) > 0:
        margin = 30
        road_x_min = max(0, np.min(solid_cols) - margin)
        road_x_max = min(w_img - 1, np.max(solid_cols) + margin)
    else:
        road_x_min = 0
        road_x_max = w_img - 1

    # --- Column-by-column: find the fill zone between the two lines ---
    red_fill = np.zeros_like(clean)
    green_fill = np.zeros_like(clean)

    has_fill = np.zeros(w_img, dtype=bool)
    top_y = np.zeros(w_img, dtype=int)
    bot_y = np.zeros(w_img, dtype=int)
    fill_type = np.zeros(w_img, dtype=int)  # 1 = red (cut), 2 = green (fill)

    window = 25  # how far to look for nearby SOLID line pixels
    MAX_VERTICAL_DIFF = 60

    for c in range(road_x_min, road_x_max):
        solid_rows = np.where(solid_boundary[:, c] == 255)[0]

        # If no solid line at this column, check nearby columns
        if len(solid_rows) == 0:
            left = max(0, c - window)
            right = min(w_img, c + window + 1)
            nearby = np.where(solid_boundary[:, left:right] == 255)
            if len(nearby[0]) > 0:
                st, sb = np.min(nearby[0]), np.max(nearby[0])
            else:
                continue
        else:
            st, sb = np.min(solid_rows), np.max(solid_rows)

        # Look for dotted line DIRECTLY at this column only (no neighbor search).
        # This prevents picking up noise dots from adjacent columns.
        dots_above = np.where(dotted_boundary[:st, c] == 255)[0]
        dots_below = np.where(dotted_boundary[sb:, c] == 255)[0]

        # Check BOTH above and below, pick the closer ground profile.
        dist_above = float('inf')
        dist_below = float('inf')
        dot_y_above = None
        dot_y_below = None

        if len(dots_above) > 0:
            dot_y_above = np.max(dots_above)
            if dot_y_above < st:
                gap = st - dot_y_above
                if gap <= MAX_VERTICAL_DIFF:
                    dist_above = gap

        if len(dots_below) > 0:
            dot_y_below = np.min(dots_below) + sb
            if sb < dot_y_below:
                gap = dot_y_below - sb
                if gap <= MAX_VERTICAL_DIFF:
                    dist_below = gap

        # Pick the direction with the closer ground profile.
        # Require a minimum gap of 5px to avoid coloring noise-level differences
        # when the two profiles are nearly coincident.
        MIN_GAP = 5
        if dist_above <= dist_below and dist_above < float('inf') and dist_above >= MIN_GAP:
            has_fill[c] = True
            top_y[c] = dot_y_above
            bot_y[c] = st
            fill_type[c] = 1  # cut (ground above design)
        elif dist_below < float('inf') and dist_below >= MIN_GAP:
            has_fill[c] = True
            top_y[c] = sb
            bot_y[c] = dot_y_below
            fill_type[c] = 2  # fill (ground below design)

    # --- Interpolate across small gaps (same fill type only) ---
    valid = np.where(has_fill)[0]

    if len(valid) > 1:
        max_gap = 40

        for c in range(np.min(valid), np.max(valid) + 1):
            if has_fill[c]:
                ty, by, ct = top_y[c], bot_y[c], fill_type[c]
            else:
                lefts = valid[valid < c]
                rights = valid[valid > c]
                if len(lefts) == 0 or len(rights) == 0:
                    continue

                li, ri = lefts[-1], rights[0]
                if (ri - li) > max_gap:
                    continue

                # Only interpolate if both neighbors have the SAME fill type.
                # This prevents color bleeding across cut/fill transitions.
                if fill_type[li] != fill_type[ri]:
                    continue

                w_frac = (c - li) / (ri - li)
                ty = int(top_y[li] + w_frac * (top_y[ri] - top_y[li]))
                by = int(bot_y[li] + w_frac * (bot_y[ri] - bot_y[li]))
                ct = fill_type[li]

            if ty < by:
                if ct == 1:
                    red_fill[ty:by, c] = 255
                elif ct == 2:
                    green_fill[ty:by, c] = 255

    # --- Horizontal gap filling ---
    if len(valid) > 1:
        x_min, x_max = np.min(valid), np.max(valid)
        max_h_gap = 60

        for r in range(cutoff_y):
            for overlay in [green_fill, red_fill]:
                cols = np.where(overlay[r, x_min:x_max + 1] == 255)[0] + x_min
                if len(cols) > 1:
                    for idx in range(len(cols) - 1):
                        gap = cols[idx + 1] - cols[idx]
                        if 1 < gap < max_h_gap:
                            overlay[r, cols[idx]:cols[idx + 1]] = 255

    # --- Smooth edges ---
    smooth_h = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
    smooth_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))

    red_clean = cv2.morphologyEx(red_fill, cv2.MORPH_CLOSE, smooth_h)
    red_clean = cv2.morphologyEx(red_clean, cv2.MORPH_CLOSE, smooth_v)

    num_labels, labels_r, stats_r, _ = cv2.connectedComponentsWithStats(red_clean, 8)
    filtered_red = np.zeros_like(red_clean)
    for i in range(1, num_labels):
        if stats_r[i, cv2.CC_STAT_AREA] > 300:
            filtered_red[labels_r == i] = 255
    red_clean = filtered_red

    green_clean = cv2.morphologyEx(green_fill, cv2.MORPH_CLOSE, smooth_h)
    green_clean = cv2.morphologyEx(green_clean, cv2.MORPH_CLOSE, smooth_v)

    num_labels, labels_g, stats_g, _ = cv2.connectedComponentsWithStats(green_clean, 8)
    filtered_green = np.zeros_like(green_clean)
    for i in range(1, num_labels):
        if stats_g[i, cv2.CC_STAT_AREA] > 300:
            filtered_green[labels_g == i] = 255
    green_clean = filtered_green

    # --- Mask out text and design lines from fill regions ---
    # Ensure colors never overlap with text, annotations, or the solid design line.
    text_mask = (img_gray < 80).astype(np.uint8) * 255
    protect_mask = cv2.bitwise_or(text_mask, solid_mask)
    protect_dilated = cv2.dilate(protect_mask, np.ones((3, 3), np.uint8), iterations=1)
    red_clean = cv2.subtract(red_clean, protect_dilated)
    green_clean = cv2.subtract(green_clean, protect_dilated)

    # --- Apply colors ---
    output[red_clean == 255] = [0, 0, 255]     # red = cut
    output[green_clean == 255] = [0, 210, 0]   # green = fill

    # Restore the design lines (blue) and original text (black) on top
    output[solid_mask == 255] = [255, 0, 0]
    text = (img_gray < 80) & (eraser == 0)
    output[text] = [0, 0, 0]

    return output
