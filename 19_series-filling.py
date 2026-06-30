import cv2
import numpy as np
import os
import zipfile
import shutil
import re

# --- 1. SETUP PATHS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.join(BASE_DIR, "/Users/sudeep/Desktop/Infra_DMT_POC/perfect_grid_removed.zip")
TMP_EXTRACT_DIR = os.path.join(BASE_DIR, "tmp_stable_extraction")
OUTPUT_DIR = os.path.join(BASE_DIR, "output_merged4")

for folder in [OUTPUT_DIR, TMP_EXTRACT_DIR]:
    if os.path.exists(folder):
        shutil.rmtree(folder)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def extract_page_identifier(filename):
    match = re.search(r'Page\d+|\b\d+\b', filename, re.IGNORECASE)
    if match:
        return match.group(0).lower().replace("page", "").strip()
    return os.path.splitext(filename)[0]



#  Grid removal + solid/dotted mask separation

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
    diagram_only = cv2.subtract(bw, grid_mask)
    bridge_shield = cv2.dilate(diagram_only, np.ones((3, 3), np.uint8), iterations=1)
    eraser = cv2.subtract(grid_mask, bridge_shield)
    clean_bw = cv2.subtract(bw, eraser)

    # Connected component separation
    nlabels, labels, stats, _ = cv2.connectedComponentsWithStats(clean_bw, 8, cv2.CV_32S)

    border_w = 3
    border_h = 3
    scale_line_y_threshold = int(h_img * 0.88)

    solid_mask = np.zeros_like(clean_bw)
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
        diag_len = np.sqrt(w ** 2 + h ** 2)

        if 3 <= w <= 45 and 2 <= h <= 12 and aspect_ratio > 1.0:
            dotted_mask[labels == i] = 255
        elif diag_len > 55 or w > 50 or h > 50:
            solid_mask[labels == i] = 255

    return solid_mask, dotted_mask, eraser, scale_line_y_threshold



# PASS 1 — Per-column gap profiling
#
# CHANGE from old design: instead of a single page-level wide_gap flag,
# we now return a per-column gap profile so Pass 2 can make column-level
# decisions. A bridge only affects a sub-range of columns — flagging the
# whole page was too aggressive.

def analyze_page(img_gray, filename, page_str):
    """
    Returns:
        col_gap_profile : np.array shape (w_img,), float
            Vertical pixel distance between solid top and dotted bottom
            at each column. 0 if no co-occurrence.
        has_any_valid   : bool — True if at least one column has lines.
    """
    h_img, w_img = img_gray.shape
    solid_mask, dotted_mask, _, _ = build_masks(img_gray)

    heal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
    solid_boundary = cv2.morphologyEx(solid_mask, cv2.MORPH_CLOSE, heal_kernel)
    dotted_boundary = cv2.morphologyEx(dotted_mask, cv2.MORPH_CLOSE, heal_kernel)

    col_gap_profile = np.zeros(w_img, dtype=np.float32)
    has_any_valid = False

    for c in range(20, w_img - 20):
        solid_rows = np.where(solid_boundary[:, c] == 255)[0]
        dotted_rows = np.where(dotted_boundary[:, c] == 255)[0]
        if len(solid_rows) == 0 or len(dotted_rows) == 0:
            continue

        top_solid_y    = np.min(solid_rows)
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



# PASS 2 — Production coloring with bridge-aware fill
#
# Core idea for bridge columns
# ─────────────────────────────
# A bridge looks like this at column c:
#
#   solid_top  ───── (top of bridge deck)
#   solid_bot  ───── (underside of bridge)
#        [air gap — DO NOT fill this]
#   dotted     ─ ─ ─ (existing ground / reference line)
#
# The dotted line is BELOW the solid line, so fill_color_type = 2 (green).
# fill_top_y  = solid_bot   (start just under the bridge underside)
# fill_bot_y  = dotted_y    (stop at the dotted line)
#
# When col_gap_profile[c] is large (bridge span), the column has no
# direct solid pixel (the bridge deck ends), so we interpolate
# fill_top_y and fill_bot_y from the nearest valid neighbours on each
# side — giving a smooth green fill that follows the ground profile
# UNDER the bridge without touching the bridge deck above.
#
# The key guard: we never fill ABOVE solid_top (the bridge deck roof).

def process_image(img_gray, col_gap_profile):
    h_img, w_img = img_gray.shape
    solid_mask, dotted_mask, eraser, scale_line_y_threshold = build_masks(img_gray)

    # Base output: grayscale → BGR, erased grid → white
    color_output = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    color_output[eraser > 0] = [255, 255, 255]

    heal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    solid_boundary = cv2.morphologyEx(solid_mask, cv2.MORPH_CLOSE, heal_kernel)
    dotted_boundary = cv2.morphologyEx(dotted_mask, cv2.MORPH_CLOSE, heal_kernel)

    red_overlay   = np.zeros_like(solid_mask)
    green_overlay = np.zeros_like(solid_mask)

    # Bridge gate: only enter fill logic where solid+dotted regions touch
    touch_kernel   = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated_solid  = cv2.dilate(solid_boundary, touch_kernel, iterations=1)
    dilated_dotted = cv2.dilate(dotted_boundary, touch_kernel, iterations=1)
    intersection   = cv2.bitwise_and(dilated_solid, dilated_dotted)

    # A BRIDGE COLUMN threshold — if gap at this column > BRIDGE_GAP_PX
    # the solid line has risen steeply (bridge deck); treat as bridge zone.
    BRIDGE_GAP_PX  = 80   # px — tune to your image scale
    look_window    = 25
    MAX_INTERP_GAP = 120  # raised: bridge spans can be wide; interpolate across them

    if not np.any(intersection > 0):
        # No intersection anywhere → nothing to color
        pass
    else:
        # ── PASS A: per-column fill acquisition 
        #
        # For every column we record:
        #   col_has_fill   : bool
        #   fill_top_y     : upper boundary of color region
        #   fill_bot_y     : lower boundary of color region
        #   fill_color_type: 1=red (dotted above solid), 2=green (dotted below)
        #   fill_guard_y   : for bridge columns, the y above which we must NOT fill
        #                    (= solid_top, i.e. the bridge deck ceiling)
        #                    -1 means no guard needed.

        col_has_fill    = np.zeros(w_img, dtype=bool)
        fill_top_y      = np.zeros(w_img, dtype=int)
        fill_bot_y      = np.zeros(w_img, dtype=int)
        fill_color_type = np.zeros(w_img, dtype=int)
        fill_guard_y    = np.full(w_img, -1, dtype=int)

        for c in range(w_img):
            solid_rows = np.where(solid_boundary[:, c] == 255)[0]

            if len(solid_rows) == 0:
                # No solid pixel here — look in neighbours
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
            dotted_above = np.where(dotted_boundary[:st, c] == 255)[0]
            dotted_below = np.where(dotted_boundary[sb:, c] == 255)[0]
            # Neighbour search ONLY for dotted_below (green) — not for dotted_above (red)
            # Red must only fill where dotted is directly found above solid in that column
            if len(dotted_below) == 0:
                c_l  = max(0, c - look_window)
                c_r  = min(w_img, c + look_window + 1)
                nb_b = np.where(dotted_boundary[sb:, c_l:c_r] == 255)
                if len(nb_b[0]) > 0:
                    dotted_below = nb_b[0]



            # ── Detect bridge column 
            # A bridge column has the solid line raised high above the
            # dotted line. The fill should go from solid_bottom down to
            # the dotted line, NOT up into the bridge deck.
            is_bridge_col = col_gap_profile[c] > BRIDGE_GAP_PX if c < len(col_gap_profile) else False

            if len(dotted_above) > 0 and not is_bridge_col:
                # Red zone: dotted is above solid top → fill dotted→solid_top
                boundary_y = int(np.max(dotted_above))
                if boundary_y < st:
                    col_has_fill[c]    = True
                    fill_top_y[c]      = boundary_y
                    fill_bot_y[c]      = st
                    fill_color_type[c] = 1

            elif len(dotted_below) > 0:
                # Green zone: dotted is below solid bottom → fill solid_bot→dotted
                # For bridge columns: fill_top_y = solid_bot (underside of bridge)
                # Guard: never fill above solid_top (the bridge deck roof)
                boundary_y = int(np.min(dotted_below)) + sb
                if sb < boundary_y:
                    col_has_fill[c]    = True
                    fill_top_y[c]      = sb          # start at solid underside
                    fill_bot_y[c]      = boundary_y
                    fill_color_type[c] = 2
                    if is_bridge_col:
                        fill_guard_y[c] = st         # ceiling = bridge deck top

        # ── PASS B: interpolate across gaps & draw 
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
                        continue   # too wide even for bridge interpolation

                    wt     = (c - l) / (r - l)
                    t_y    = int(fill_top_y[l]  + wt * (fill_top_y[r]  - fill_top_y[l]))
                    b_y    = int(fill_bot_y[l]  + wt * (fill_bot_y[r]  - fill_bot_y[l]))
                    c_type = fill_color_type[l] if wt <= 0.5 else fill_color_type[r]

                    # Interpolate guard ceiling too (for bridge midspan columns)
                    g_l = fill_guard_y[l] if fill_guard_y[l] >= 0 else t_y
                    g_r = fill_guard_y[r] if fill_guard_y[r] >= 0 else t_y
                    guard = int(g_l + wt * (g_r - g_l)) if (fill_guard_y[l] >= 0 or fill_guard_y[r] >= 0) else -1

                # Clamp fill_top_y so it never goes above the bridge deck roof
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

    # ── SMOOTHING: close-only, no dilation 
    hor_close = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
    ver_close = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))

    red_clean   = cv2.morphologyEx(red_overlay,   cv2.MORPH_CLOSE, hor_close)
    red_clean   = cv2.morphologyEx(red_clean,     cv2.MORPH_CLOSE, ver_close)
    green_clean = cv2.morphologyEx(green_overlay, cv2.MORPH_CLOSE, hor_close)
    green_clean = cv2.morphologyEx(green_clean,   cv2.MORPH_CLOSE, ver_close)

    color_output[red_clean   == 255] = [0, 0, 255]
    color_output[green_clean == 255] = [0, 210, 0]

    # Solid lines on top in blue
    color_output[solid_mask == 255] = [255, 0, 0]

    # Restore original dark text
    text_mask = (img_gray < 80) & (eraser == 0)
    color_output[text_mask] = [0, 0, 0]

    return color_output


# MAIN PIPELINE

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
    gap_profiles = {}   # base_name -> col_gap_profile (np.array)

    for img_path in valid_file_paths:
        img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            continue
        base_name = os.path.basename(img_path)
        page_str  = extract_page_identifier(base_name)
        profile, _ = analyze_page(img_gray, base_name, page_str)
        gap_profiles[base_name] = profile

    print("=" * 65 + "\n")

    # ── PASS 2: coloring ─
    print("🚀 PASS 2: Production coloring with bridge-aware fill...")
    for idx, img_path in enumerate(valid_file_paths, 1):
        try:
            img_gray  = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img_gray is None:
                continue
            base_name = os.path.basename(img_path)
            profile   = gap_profiles.get(base_name, np.zeros(img_gray.shape[1]))
            result    = process_image(img_gray, profile)
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"precision_{base_name}"), result)

            if idx % 20 == 0 or idx == len(valid_file_paths):
                print(f"    ↳ {idx}/{len(valid_file_paths)} processed.")
        except Exception as e:
            print(f"⚠️  {os.path.basename(img_path)}: {e}")

    if os.path.exists(TMP_EXTRACT_DIR):
        shutil.rmtree(TMP_EXTRACT_DIR)

    print(f"\n✨ Done! Outputs saved to: '{OUTPUT_DIR}'")
