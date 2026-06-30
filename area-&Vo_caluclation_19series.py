import cv2
import numpy as np
import os
import zipfile
import shutil
import re
import csv

# --- 1. SETUP PATHS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.join(BASE_DIR, "/Users/sudeep/Desktop/Infra_DMT_POC/perfect_grid_removed.zip")
TMP_EXTRACT_DIR = os.path.join(BASE_DIR, "tmp_stable_extraction")
OUTPUT_DIR = os.path.join(BASE_DIR, "output_merged55")

for folder in [OUTPUT_DIR, TMP_EXTRACT_DIR]:
    if os.path.exists(folder):
        shutil.rmtree(folder)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def extract_page_identifier(filename):
    match = re.search(r'Page\d+|\b\d+\b', filename, re.IGNORECASE)
    if match:
        return match.group(0).lower().replace("page", "").strip()
    return os.path.splitext(filename)[0]


def extract_chainage_feet(filename):
    """
    Extracts chainage in feet from filenames like:
      cleaned_Page47_Sta_15+50630640.png  → station 15+50 → 1550 feet
      cleaned_Page47_Sta_12+00.png        → station 12+00 → 1200 feet

    Pattern: Sta_<major>+<minor> where minor is first 2 digits after +
    """
    match = re.search(r'Sta_?(\d+)\+(\d{2})', filename, re.IGNORECASE)
    if match:
        major = int(match.group(1))   # e.g. 15
        minor = int(match.group(2))   # e.g. 50
        chainage = major * 100 + minor
        return chainage
    return None


# ── Grid removal + solid/dotted mask separation ──────────────────────────────

def build_masks(img_gray):
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


# ── Scale detection ───────────────────────────────────────────────────────────

def detect_scale_pixels_per_unit(img_gray):
    """
    Detects pixels-per-10-feet by reading the horizontal scale bar
    tick spacing at the bottom of the image.
    Falls back to a default if detection fails.
    """
    h, w = img_gray.shape
    scale_region = img_gray[int(h * 0.88):, :]

    _, bw    = cv2.threshold(scale_region, 80, 255, cv2.THRESH_BINARY_INV)
    ver_k    = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 10))
    ticks    = cv2.morphologyEx(bw, cv2.MORPH_OPEN, ver_k)
    tick_cols = np.where(np.sum(ticks, axis=0) > 5)[0]

    if len(tick_cols) < 2:
        print("⚠️  Scale bar not detected, using fallback 50px = 10ft")
        return 50

    tick_positions = []
    cluster_start  = tick_cols[0]
    prev           = tick_cols[0]
    for col in tick_cols[1:]:
        if col - prev > 5:
            tick_positions.append((cluster_start + prev) // 2)
            cluster_start = col
        prev = col
    tick_positions.append((cluster_start + prev) // 2)

    if len(tick_positions) < 2:
        print("⚠️  Too few ticks detected, using fallback 50px = 10ft")
        return 50

    spacings     = np.diff(tick_positions)
    px_per_10ft  = float(np.median(spacings))
    print(f"📏 Scale detected: {px_per_10ft:.1f} px = 10 ft")
    return px_per_10ft


# ── Area calculation ──────────────────────────────────────────────────────────

def calculate_area(red_clean, green_clean, px_per_10ft):
    """
    Converts pixel counts to real-world square feet.

    Scale:  horizontal 1" = 10ft,  vertical 1" = 10ft
    So:     1 pixel = (10 / px_per_10ft) ft in both directions
            1 pixel area = (10 / px_per_10ft)^2 sq ft
    """
    ft_per_px   = 10.0 / px_per_10ft
    sqft_per_px = ft_per_px ** 2

    red_pixels   = int(np.sum(red_clean   > 0))
    green_pixels = int(np.sum(green_clean > 0))

    red_sqft   = red_pixels   * sqft_per_px
    green_sqft = green_pixels * sqft_per_px

    return {
        "red_pixels"  : red_pixels,
        "green_pixels": green_pixels,
        "red_sqft"    : round(red_sqft,   2),
        "green_sqft"  : round(green_sqft, 2),
        "px_per_10ft" : round(px_per_10ft, 2)
    }


# ── PASS 1 — Per-column gap profiling ────────────────────────────────────────

def analyze_page(img_gray, filename, page_str):
    h_img, w_img = img_gray.shape
    solid_mask, dotted_mask, _, _ = build_masks(img_gray)

    heal_kernel    = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
    solid_boundary  = cv2.morphologyEx(solid_mask,  cv2.MORPH_CLOSE, heal_kernel)
    dotted_boundary = cv2.morphologyEx(dotted_mask, cv2.MORPH_CLOSE, heal_kernel)

    col_gap_profile = np.zeros(w_img, dtype=np.float32)
    has_any_valid   = False

    for c in range(20, w_img - 20):
        solid_rows  = np.where(solid_boundary[:, c]  == 255)[0]
        dotted_rows = np.where(dotted_boundary[:, c] == 255)[0]
        if len(solid_rows) == 0 or len(dotted_rows) == 0:
            continue

        top_solid_y     = np.min(solid_rows)
        bottom_dotted_y = np.max(dotted_rows)

        if bottom_dotted_y > top_solid_y:
            col_gap_profile[c] = float(abs(top_solid_y - bottom_dotted_y))
            has_any_valid = True

    if has_any_valid:
        max_gap_col = int(np.argmax(col_gap_profile))
        max_gap     = col_gap_profile[max_gap_col]
        print(f"📄 Page {page_str} ({filename}) -> "
              f"Max gap col: {max_gap_col} | Max gap: {max_gap:.1f} px")
    else:
        print(f"📄 Page {page_str} ({filename}) -> No co-occurrence found.")

    return col_gap_profile, has_any_valid


# ── PASS 2 — Production coloring with bridge-aware fill ──────────────────────

def process_image(img_gray, col_gap_profile):
    h_img, w_img = img_gray.shape
    solid_mask, dotted_mask, eraser, scale_line_y_threshold = build_masks(img_gray)

    color_output = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    color_output[eraser > 0] = [255, 255, 255]

    heal_kernel     = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    solid_boundary  = cv2.morphologyEx(solid_mask,  cv2.MORPH_CLOSE, heal_kernel)
    dotted_boundary = cv2.morphologyEx(dotted_mask, cv2.MORPH_CLOSE, heal_kernel)

    red_overlay   = np.zeros_like(solid_mask)
    green_overlay = np.zeros_like(solid_mask)

    touch_kernel   = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated_solid  = cv2.dilate(solid_boundary,  touch_kernel, iterations=1)
    dilated_dotted = cv2.dilate(dotted_boundary, touch_kernel, iterations=1)
    intersection   = cv2.bitwise_and(dilated_solid, dilated_dotted)

    BRIDGE_GAP_PX  = 80
    look_window    = 25
    MAX_INTERP_GAP = 120

    if not np.any(intersection > 0):
        pass
    else:
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

            # ── Red: only search directly in this column (no neighbour search)
            dotted_above = np.where(dotted_boundary[:st, c] == 255)[0]

            # ── Green: allow neighbour search (works correctly)
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

            # ── PASS C: horizontal row-gap stitching
            MAX_H_GAP = 60
            for row in range(0, scale_line_y_threshold):
                for overlay in [green_overlay, red_overlay]:
                    cols = (np.where(overlay[row, start_idx:end_idx + 1] == 255)[0]
                            + start_idx)
                    if len(cols) > 1:
                        for idx in range(len(cols) - 1):
                            gap = int(cols[idx + 1]) - int(cols[idx])
                            if 1 < gap < MAX_H_GAP:
                                overlay[row, cols[idx]:cols[idx + 1]] = 255

    # ── SMOOTHING
    hor_close = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
    ver_close = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))

    red_clean   = cv2.morphologyEx(red_overlay,   cv2.MORPH_CLOSE, hor_close)
    red_clean   = cv2.morphologyEx(red_clean,     cv2.MORPH_CLOSE, ver_close)
    green_clean = cv2.morphologyEx(green_overlay, cv2.MORPH_CLOSE, hor_close)
    green_clean = cv2.morphologyEx(green_clean,   cv2.MORPH_CLOSE, ver_close)

    color_output[red_clean   == 255] = [0, 0, 255]
    color_output[green_clean == 255] = [0, 210, 0]
    color_output[solid_mask  == 255] = [255, 0, 0]

    text_mask = (img_gray < 80) & (eraser == 0)
    color_output[text_mask] = [0, 0, 0]

    # ── Area calculation
    px_per_10ft = detect_scale_pixels_per_unit(img_gray)
    area_data   = calculate_area(red_clean, green_clean, px_per_10ft)

    return color_output, area_data


# ── MAIN PIPELINE ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not os.path.exists(FILE_PATH):
        print(f"❌ ERROR: Source zip not found at: {FILE_PATH}")
        exit(1)

    print("📦 Unzipping source package...")
    with zipfile.ZipFile(FILE_PATH, 'r') as z:
        z.extractall(TMP_EXTRACT_DIR)

    valid_file_paths = []
    for root, _, filenames in os.walk(TMP_EXTRACT_DIR):
        for file in sorted(filenames):
            if (file.lower().endswith(('.png', '.jpg', '.jpeg'))
                    and "__MACOSX" not in root
                    and not file.startswith('.')):
                valid_file_paths.append(os.path.join(root, file))

    print(f"🔍 Found {len(valid_file_paths)} images.\n")

    # ── PASS 1: per-page column gap profiling
    print("=" * 65)
    print("PASS 1: Column gap profiling")
    print("=" * 65)
    gap_profiles = {}

    for img_path in valid_file_paths:
        img_gray  = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            continue
        base_name = os.path.basename(img_path)
        page_str  = extract_page_identifier(base_name)
        profile, _ = analyze_page(img_gray, base_name, page_str)
        gap_profiles[base_name] = profile

    print("=" * 65 + "\n")

    # ── PASS 2: coloring + area collection
    print("🚀 PASS 2: Production coloring with bridge-aware fill...")
    page_results = []   # list of dicts — one per page

    for idx, img_path in enumerate(valid_file_paths, 1):
        try:
            img_gray  = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img_gray is None:
                continue
            base_name = os.path.basename(img_path)
            profile   = gap_profiles.get(base_name, np.zeros(img_gray.shape[1]))
            result, area_data = process_image(img_gray, profile)
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"precision_{base_name}"), result)

            chainage = extract_chainage_feet(base_name)

            page_results.append({
                "filename"    : base_name,
                "chainage_ft" : chainage,
                "red_sqft"    : area_data["red_sqft"],
                "green_sqft"  : area_data["green_sqft"],
                "px_per_10ft" : area_data["px_per_10ft"],
            })

            if idx % 20 == 0 or idx == len(valid_file_paths):
                print(f"    ↳ {idx}/{len(valid_file_paths)} processed.")
        except Exception as e:
            print(f"⚠️  {os.path.basename(img_path)}: {e}")

    # ── Sort by chainage so consecutive pages are neighbours
    page_results.sort(key=lambda x: (x["chainage_ft"] if x["chainage_ft"] is not None else 0))

    # ── Volume: Average End Area method
    # Volume (cu ft) = ((Area1 + Area2) / 2) × distance_between_stations
    for i in range(len(page_results)):
        r = page_results[i]
        if i == 0 or page_results[i-1]["chainage_ft"] is None or r["chainage_ft"] is None:
            r["distance_ft"] = 0.0
            r["red_cuft"]    = 0.0
            r["green_cuft"]  = 0.0
        else:
            prev = page_results[i - 1]
            dist = abs(r["chainage_ft"] - prev["chainage_ft"])
            r["distance_ft"] = dist
            r["red_cuft"]    = round(((r["red_sqft"]   + prev["red_sqft"])   / 2) * dist, 2)
            r["green_cuft"]  = round(((r["green_sqft"] + prev["green_sqft"]) / 2) * dist, 2)

    # ── Terminal print
    SEP = "=" * 100
    HDR = f"{'FILE':<45} {'STA(ft)':>8} {'RED AREA':>10} {'GRN AREA':>10} {'DIST(ft)':>9} {'RED VOL':>12} {'GRN VOL':>12}"
    SUB = f"{'':45} {'':8} {'(sqft)':>10} {'(sqft)':>10} {'':9} {'(cuft)':>12} {'(cuft)':>12}"

    print("\n" + SEP)
    print(HDR)
    print(SUB)
    print(SEP)

    total_red_sqft   = 0.0
    total_green_sqft = 0.0
    total_red_cuft   = 0.0
    total_green_cuft = 0.0

    for r in page_results:
        sta = str(r["chainage_ft"]) if r["chainage_ft"] is not None else "N/A"
        print(f"{r['filename'][:44]:<45} {sta:>8} "
              f"{r['red_sqft']:>10.2f} {r['green_sqft']:>10.2f} "
              f"{r['distance_ft']:>9.1f} "
              f"{r['red_cuft']:>12.2f} {r['green_cuft']:>12.2f}")
        total_red_sqft   += r["red_sqft"]
        total_green_sqft += r["green_sqft"]
        total_red_cuft   += r["red_cuft"]
        total_green_cuft += r["green_cuft"]

    print(SEP)
    print(f"\n📐 TOTAL RED   AREA   = {total_red_sqft:>12.2f}  sqft  (cut / excavation)")
    print(f"📐 TOTAL GREEN AREA   = {total_green_sqft:>12.2f}  sqft  (fill / embankment)")
    print(f"📦 TOTAL RED   VOLUME = {total_red_cuft:>12.2f}  cuft  (cut / excavation)")
    print(f"📦 TOTAL GREEN VOLUME = {total_green_cuft:>12.2f}  cuft  (fill / embankment)")
    print(SEP)

    # ── Save to text file
    txt_path = os.path.join(OUTPUT_DIR, "area_volume_summary.txt")
    with open(txt_path, "w") as f:
        f.write(SEP + "\n")
        f.write(HDR + "\n")
        f.write(SUB + "\n")
        f.write(SEP + "\n")
        for r in page_results:
            sta = str(r["chainage_ft"]) if r["chainage_ft"] is not None else "N/A"
            f.write(f"{r['filename'][:44]:<45} {sta:>8} "
                    f"{r['red_sqft']:>10.2f} {r['green_sqft']:>10.2f} "
                    f"{r['distance_ft']:>9.1f} "
                    f"{r['red_cuft']:>12.2f} {r['green_cuft']:>12.2f}\n")
        f.write(SEP + "\n")
        f.write(f"\nTOTAL RED   AREA   = {total_red_sqft:.2f} sqft  (cut / excavation)\n")
        f.write(f"TOTAL GREEN AREA   = {total_green_sqft:.2f} sqft  (fill / embankment)\n")
        f.write(f"TOTAL RED   VOLUME = {total_red_cuft:.2f} cuft  (cut / excavation)\n")
        f.write(f"TOTAL GREEN VOLUME = {total_green_cuft:.2f} cuft  (fill / embankment)\n")

    # ── Save to CSV
    csv_path = os.path.join(OUTPUT_DIR, "area_volume_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "filename", "chainage_ft", "distance_ft",
            "red_sqft", "green_sqft",
            "red_cuft", "green_cuft",
            "px_per_10ft"
        ])
        writer.writeheader()
        for r in page_results:
            writer.writerow({
                "filename"    : r["filename"],
                "chainage_ft" : r["chainage_ft"],
                "distance_ft" : r["distance_ft"],
                "red_sqft"    : r["red_sqft"],
                "green_sqft"  : r["green_sqft"],
                "red_cuft"    : r["red_cuft"],
                "green_cuft"  : r["green_cuft"],
                "px_per_10ft" : r["px_per_10ft"],
            })

    if os.path.exists(TMP_EXTRACT_DIR):
        shutil.rmtree(TMP_EXTRACT_DIR)

    print(f"\n📄 Text summary : '{txt_path}'")
    print(f"📊 CSV  summary : '{csv_path}'")
    print(f"🖼️  Images saved : '{OUTPUT_DIR}'")
    print(f"\n✨ Done!")
