"""
area_calculator.py
==================
OpenCV-based cross-section area calculator.

Supports TWO image types automatically:
  - COLORED images (blue = existing ground, orange = proposed grade)
  - B&W PDF renders (solid + dashed black lines on white/gray grid)

B&W detection pipeline (ported from working Colab logic):
  1. Detect grid scale FIRST on raw gray image
  2. Remove grid lines via morphological operations
  3. Connected components → classify by shape:
       dotted segments  → proposed grade  (orange in debug)
       long solid lines → existing ground (red in debug)
  4. Build x→y profiles from the two masks
  5. Calculate area using grid scale from step 1

Area formula:
    total_px_area = Σ |y_line1[x] − y_line2[x]|  for all x where both lines exist
    area_sqft     = total_px_area / px_per_sqft
    px_per_sqft   = (grid_px²) / 100              (10ft × 10ft = 100 sqft)
"""

import cv2
from scipy.signal import find_peaks
import numpy as np

# ─── HSV colour ranges (coloured images only) ────────────────────────────────
_BLUE_LOWER   = np.array([100,  80,  50])
_BLUE_UPPER   = np.array([130, 255, 255])
_ORANGE_LOWER = np.array([  8, 100, 100])
_ORANGE_UPPER = np.array([ 28, 255, 255])


# ════════════════════════════════════════════════════════════════════════════
# CROP  –  remove empty white space below the drawing
# ════════════════════════════════════════════════════════════════════════════

def crop_to_drawing(img_bytes: bytes) -> bytes:
    """
    Removes large empty white space below the actual cross-section drawing.
    Strategy: scan rows from the bottom upward; the first row that has
    more than 30% dark pixels is the bottom edge of the drawing.
    """
    nparr = np.frombuffer(img_bytes, np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w  = img.shape[:2]

    _, dark  = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
    row_dark = np.array([dark[y, :].sum() // 255 for y in range(h)])

    bottom = h
    for y in range(h - 1, 0, -1):
        if row_dark[y] > w * 0.3:
            bottom = y + 10
            break

    cropped = img[:bottom, :]
    _, enc  = cv2.imencode('.png', cropped)
    return enc.tobytes()


# ════════════════════════════════════════════════════════════════════════════
# GRID DETECTION  –  runs on raw gray image BEFORE any processing
# ════════════════════════════════════════════════════════════════════════════

def _detect_grid_px(gray: np.ndarray) -> float:
    """
    Measures the pixel width of one grid square from the raw image.
    One grid square = 10ft × 10ft = 100 sq ft (engineering standard).
    """
    grid_mask = cv2.inRange(gray, 155, 220)
    col_sums  = grid_mask.sum(axis=0).astype(float)

    print(f"GRID DEBUG: col_sums max={col_sums.max():.0f}")

    if col_sums.max() > 0:
        peaks, _ = find_peaks(col_sums,
                               height=col_sums.max() * 0.3,
                               distance=5)
        print(f"GRID DEBUG: peaks found={len(peaks)}")
        if len(peaks) >= 4:
            spacings = np.diff(peaks).tolist()
            minor = [s for s in spacings if 10 < s < 100]
            print(f"GRID DEBUG: minor spacings={sorted(minor)}")
            if len(minor) >= 1:
                result = float(np.median(minor))
                print(f"GRID DEBUG: result={result}px")
                return result

    print("GRID DEBUG: fell through to border fallback")
    h, w = gray.shape
    _, dark = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
    left_border = 0
    for x in range(w):
        if dark[:, x].sum() // 255 > h * 0.4:
            left_border = x
            break
    right_border = w - 1
    for x in range(w - 1, 0, -1):
        if dark[:, x].sum() // 255 > h * 0.4:
            right_border = x
            break
    return ((right_border - left_border) / 270.0) * 10.0


# ════════════════════════════════════════════════════════════════════════════
# COLORED IMAGE
# ════════════════════════════════════════════════════════════════════════════

def _is_colored(hsv: np.ndarray) -> bool:
    blue_px   = int(cv2.inRange(hsv, _BLUE_LOWER,   _BLUE_UPPER).sum())   // 255
    orange_px = int(cv2.inRange(hsv, _ORANGE_LOWER, _ORANGE_UPPER).sum()) // 255
    return blue_px > 50 and orange_px > 50


def _profile_from_mask(mask: np.ndarray) -> dict:
    profile = {}
    for x in range(mask.shape[1]):
        ys = np.where(mask[:, x] > 0)[0]
        if len(ys):
            profile[x] = float(ys.mean())
    return profile


def _profiles_colored(hsv: np.ndarray):
    blue_mask   = cv2.inRange(hsv, _BLUE_LOWER,   _BLUE_UPPER)
    orange_mask = cv2.inRange(hsv, _ORANGE_LOWER, _ORANGE_UPPER)
    return _profile_from_mask(blue_mask), _profile_from_mask(orange_mask)


# ════════════════════════════════════════════════════════════════════════════
# B&W IMAGE  –  Colab-style connected components classification
# ════════════════════════════════════════════════════════════════════════════

def _profiles_bw(gray: np.ndarray):
    """
    Ported from working Colab notebook:
      1. Threshold → binary dark pixels
      2. Remove grid lines via morphological kernels
      3. Connected components on clean image
      4. Border protection (8% each side) — ignores scale numbers on edges
      5. Classify each component:
           arrows    → skip       (h >= 16 and w <= 45)
           dotted    → proposed grade  (small segments, orange in debug)
           solid     → existing ground (long components, red in debug)
      6. Build x→y profile dicts from the two masks
    """
    h_img, w_img = gray.shape

    # ── Step 1: threshold ────────────────────────────────────────────────────
    _, bw = cv2.threshold(gray, 235, 255, cv2.THRESH_BINARY_INV)

    # ── Step 2: grid removal ─────────────────────────────────────────────────
    ver_k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 200))
    hor_k = cv2.getStructuringElement(cv2.MORPH_RECT, (200, 1))
    grid_mask = cv2.add(
        cv2.morphologyEx(bw, cv2.MORPH_OPEN, ver_k),
        cv2.morphologyEx(bw, cv2.MORPH_OPEN, hor_k)
    )
    # Protect diagram pixels that touch grid lines from being erased
    diagram_only  = cv2.subtract(bw, grid_mask)
    bridge_shield = cv2.dilate(diagram_only, np.ones((3, 3), np.uint8), iterations=1)
    eraser        = cv2.subtract(grid_mask, bridge_shield)
    clean_bw      = cv2.subtract(bw, eraser)

    # ── Step 3: connected components ─────────────────────────────────────────
    nlabels, labels, stats, _ = cv2.connectedComponentsWithStats(
        clean_bw, 8, cv2.CV_32S
    )

    # ── Step 4: classify into two masks ──────────────────────────────────────
    border_w = int(w_img * 0.08)
    border_h = int(h_img * 0.08)

    existing_mask = np.zeros((h_img, w_img), dtype=np.uint8)   # solid  → profile_a
    proposed_mask = np.zeros((h_img, w_img), dtype=np.uint8)   # dotted → profile_b

    for i in range(1, nlabels):
        x, y, w, h, area = stats[i]

        # Border protection — skip anything touching the 8% border
        if (x < border_w or (x + w) > (w_img - border_w) or
                y < border_h or (y + h) > (h_img - border_h)):
            continue

        diag_len     = np.sqrt(w ** 2 + h ** 2)
        aspect_ratio = w / h if h > 0 else 0

        # Arrow interceptor — tall but narrow → skip
        if h >= 16 and w <= 45:
            continue

        # Dotted / dashed segment → proposed grade
        if 3 <= w <= 45 and 2 <= h <= 12 and aspect_ratio > 1.0:
            proposed_mask[labels == i] = 255
            continue

        # Long solid component → existing ground
        if diag_len > 55 or w > 50 or h > 50:
            existing_mask[labels == i] = 255
            continue

        # Everything else (numbers, symbols) → ignore

    # ── Step 5: build x→y profiles ───────────────────────────────────────────
    profile_a = _profile_from_mask(existing_mask)   # existing ground (red in debug)
    profile_b = _profile_from_mask(proposed_mask)   # proposed grade  (orange in debug)

    # ── Debug prints ─────────────────────────────────────────────────────────
    existing_cols = np.where(existing_mask.sum(axis=0) > 0)[0]
    proposed_cols = np.where(proposed_mask.sum(axis=0) > 0)[0]
    if len(existing_cols) > 0:
        print(f"existing x: {existing_cols[0]} to {existing_cols[-1]}")
    if len(proposed_cols) > 0:
        print(f"proposed x: {proposed_cols[0]} to {proposed_cols[-1]}")
    print(f"BW DEBUG: existing cols={len(profile_a)}  proposed cols={len(profile_b)}")

    return profile_a, profile_b


# ════════════════════════════════════════════════════════════════════════════
# REGION DETECTION
# ════════════════════════════════════════════════════════════════════════════

def _find_regions(profile_a: dict, profile_b: dict) -> list:
    common_x = sorted(set(profile_a.keys()) & set(profile_b.keys()))
    if not common_x:
        return []

    regions = []
    current = None

    for x in common_x:
        ay    = profile_a[x]
        by    = profile_b[x]
        gap   = abs(ay - by)
        rtype = "FILL" if ay < by else "CUT"

        if current is None:
            current = {"type": rtype, "start_x": x, "end_x": x,
                       "pixel_area": gap, "col_count": 1}
        elif rtype == current["type"]:
            current["end_x"]       = x
            current["pixel_area"] += gap
            current["col_count"]  += 1
        else:
            regions.append(current)
            current = {"type": rtype, "start_x": x, "end_x": x,
                       "pixel_area": gap, "col_count": 1}

    if current:
        regions.append(current)

    # Merge tiny transition slivers (< 5 columns) into the previous region
    merged = []
    for r in regions:
        if r["col_count"] < 5 and merged:
            merged[-1]["pixel_area"] += r["pixel_area"]
            merged[-1]["end_x"]       = r["end_x"]
            merged[-1]["col_count"]  += r["col_count"]
        else:
            merged.append(r)

    return merged


# ════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ════════════════════════════════════════════════════════════════════════════

def calculate_area(img_bytes: bytes) -> dict:
    print("CALCULATE_AREA CALLED")
    try:
        nparr = np.frombuffer(img_bytes, np.uint8)
        img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return {"error": "Could not decode image"}

        import inspect
        print("FILE:", inspect.getfile(calculate_area))

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # ── Step 1: grid scale FIRST on raw gray ──────────────────────────────
        grid_px     = _detect_grid_px(gray)
        px_per_sqft = (grid_px ** 2) / 100.0

        # ── Step 2: line detection ────────────────────────────────────────────
        if _is_colored(hsv):
            profile_a, profile_b = _profiles_colored(hsv)
            image_type = "colored"
        else:
            profile_a, profile_b = _profiles_bw(gray)
            image_type = "bw"

        if not profile_a or not profile_b:
            return {
                "error": (
                    f"Could not detect both lines (image_type={image_type}). "
                    "Ensure cross-section curves are clearly visible."
                )
            }

        # ── Step 3: find regions and calculate area ───────────────────────────
        raw_regions = _find_regions(profile_a, profile_b)

        regions_out = []
        for r in raw_regions:
            area_sqft = r["pixel_area"] / px_per_sqft
            regions_out.append({
                "type":       r["type"],
                "start_x_px": r["start_x"],
                "end_x_px":   r["end_x"],
                "width_px":   r["end_x"] - r["start_x"],
                "area_sqft":  round(area_sqft, 2)
            })

        total_cut  = sum(r["area_sqft"] for r in regions_out if r["type"] == "CUT")
        total_fill = sum(r["area_sqft"] for r in regions_out if r["type"] == "FILL")

        # ── Debug: profile ranges and pixel counts ────────────────────────────
        common = set(profile_a.keys()) & set(profile_b.keys())
        print(f"profile_a x range: {min(profile_a.keys())} to {max(profile_a.keys())}, cols={len(profile_a)}")
        print(f"profile_b x range: {min(profile_b.keys())} to {max(profile_b.keys())}, cols={len(profile_b)}")
        print(f"common x range: {min(common)} to {max(common)}, cols={len(common)}")
        print(f"px_per_sqft={px_per_sqft:.4f}  total_cut={total_cut}  total_fill={total_fill}")

        return {
            "image_type":      image_type,
            "grid_px":         round(grid_px, 2),
            "px_per_sqft":     round(px_per_sqft, 4),
            "regions":         regions_out,
            "total_cut_sqft":  round(total_cut,  2),
            "total_fill_sqft": round(total_fill, 2),
            "net_sqft":        round(total_cut - total_fill, 2),
            "error":           None
        }

    except Exception as e:
        return {"error": str(e)}


# ════════════════════════════════════════════════════════════════════════════
# DEBUG VISUALIZATION
# ════════════════════════════════════════════════════════════════════════════

def generate_debug_image(img_bytes: bytes) -> bytes:
    nparr = np.frombuffer(img_bytes, np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv   = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, w  = img.shape[:2]

    vis = img.copy()

    # ── Grid lines overlay ────────────────────────────────────────────────────
    grid_mask = cv2.inRange(gray, 175, 225)
    col_sums  = grid_mask.sum(axis=0).astype(float)
    row_sums  = grid_mask.sum(axis=1).astype(float)

    v_peaks, _ = find_peaks(col_sums, height=col_sums.max() * 0.5, distance=8)
    h_peaks, _ = find_peaks(row_sums, height=row_sums.max() * 0.5, distance=8)

    for x in v_peaks:
        cv2.line(vis, (int(x), 0), (int(x), h), (0, 200, 0), 1)
    for y in h_peaks:
        cv2.line(vis, (0, int(y)), (w, int(y)), (200, 0, 0), 1)

    # ── Highlighted grid square ───────────────────────────────────────────────
    grid_px = _detect_grid_px(gray)
    if len(h_peaks) >= 2 and len(v_peaks) >= 1:
        x1 = int(v_peaks[0])
        x2 = x1 + int(grid_px)
        y1, y2 = int(h_peaks[0]), int(h_peaks[1])
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(vis, f"{int(grid_px)}x{int(grid_px)}px = 100 sqft",
                    (x1 + 2, y1 + (y2 - y1) // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    # ── Line profiles ─────────────────────────────────────────────────────────
    if _is_colored(hsv):
        profile_a, profile_b = _profiles_colored(hsv)
    else:
        profile_a, profile_b = _profiles_bw(gray)

    for x, y in profile_a.items():
        iy = int(round(y))
        if 0 <= iy < h:
            cv2.circle(vis, (x, iy), 1, (0, 0, 255), -1)      # red = existing ground

    for x, y in profile_b.items():
        iy = int(round(y))
        if 0 <= iy < h:
            cv2.circle(vis, (x, iy), 1, (0, 165, 255), -1)    # orange = proposed grade

    # ── Enclosed area — CUT=red fill, FILL=blue fill ─────────────────────────
    overlay  = vis.copy()
    common_x = sorted(set(profile_a.keys()) & set(profile_b.keys()))
    for x in common_x:
        ya    = int(round(profile_a[x]))
        yb    = int(round(profile_b[x]))
        y_top = min(ya, yb)
        y_bot = max(ya, yb)
        if 0 <= y_top < h and 0 <= y_bot < h:
            # CUT: existing below proposed (ya > yb in image coords)
            if ya > yb:
                cv2.line(overlay, (x, y_top), (x, y_bot), (0, 0, 220), 1)    # red
            else:
                cv2.line(overlay, (x, y_top), (x, y_bot), (220, 100, 0), 1)  # blue

    cv2.addWeighted(overlay, 0.6, vis, 0.4, 0, vis)

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_y = 12
    for color, text in [
        ((0, 200, 0),   "Green      = grid lines"),
        ((0, 0, 255),   "Red dots   = existing ground"),
        ((0, 165, 255), "Orange dots= proposed grade"),
        ((0, 0, 220),   "Red fill   = CUT region"),
        ((220, 100, 0), "Blue fill  = FILL region"),
        ((0, 255, 255), "Yellow box = 1 grid square = 100 sqft"),
    ]:
        cv2.putText(vis, text, (5, legend_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, color, 1)
        legend_y += 12

    _, enc = cv2.imencode('.png', vis)
    return enc.tobytes()