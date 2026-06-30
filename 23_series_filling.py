import cv2
import numpy as np
import os
import zipfile
import shutil
import re

# --- 1. SETUP PATHS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.join(BASE_DIR, "/Users/sudeep/Desktop/Infra_DMT_POC/23-series_cleaned.zip")
TMP_EXTRACT_DIR = os.path.join(BASE_DIR, "tmp_stable_extraction")
OUTPUT_DIR = os.path.join(BASE_DIR, "output_mergedFinal")

for folder in [OUTPUT_DIR, TMP_EXTRACT_DIR]:
    if os.path.exists(folder):
        shutil.rmtree(folder)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def extract_page_identifier(filename):
    match = re.search(r'Page\d+|\b\d+\b', filename, re.IGNORECASE)
    if match:
        return match.group(0).lower().replace("page", "").strip()
    return os.path.splitext(filename)[0]


# --- 2. MASK GENERATION ---
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

    # Bridge-safe eraser
    diagram_only = cv2.subtract(bw, grid_mask)
    bridge_shield = cv2.dilate(diagram_only, np.ones((3, 3), np.uint8), iterations=1)
    eraser = cv2.subtract(grid_mask, bridge_shield)
    clean_bw = cv2.subtract(bw, eraser)

    # Connected component separation
    nlabels, labels, stats, _ = cv2.connectedComponentsWithStats(clean_bw, 8, cv2.CV_32S)

    scale_line_y_threshold = int(h_img * 0.88)
    solid_mask = np.zeros_like(clean_bw)
    dotted_mask = np.zeros_like(clean_bw)

    # Set margins to clear out rulers
    left_margin_gate = int(w_img * 0.06)
    right_margin_gate = int(w_img * 0.94)
    top_margin_gate = 10

    for i in range(1, nlabels):
        x, y, w, h, area = stats[i]

        if x < left_margin_gate or (x + w) > right_margin_gate:
            continue
        if y < top_margin_gate or (y + h) > scale_line_y_threshold:
            continue
        if h > 100 and w <= 3:
            continue

        aspect_ratio = w / h if h > 0 else 0
        diag_len = np.sqrt(w ** 2 + h ** 2)

        if 3 <= w <= 45 and 2 <= h <= 12 and aspect_ratio > 1.0:
            dotted_mask[labels == i] = 255
        elif diag_len > 55 or w > 50 or h > 50:
            solid_mask[labels == i] = 255

    return solid_mask, dotted_mask, eraser, scale_line_y_threshold


# --- 3. PASS 1: GAP PROFILING ---
def analyze_page(img_gray, filename, page_str, cached_masks=None):
    h_img, w_img = img_gray.shape
    solid_mask, dotted_mask, eraser, scale_line_y_threshold = cached_masks if cached_masks else build_masks(img_gray)

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
        print(f"📄 Page {page_str} ({filename}) -> Max gap col: {max_gap_col} | Max gap: {max_gap:.1f} px")
    else:
        print(f"📄 Page {page_str} ({filename}) -> No co-occurrence found.")

    return col_gap_profile, has_any_valid, (solid_mask, dotted_mask, eraser, scale_line_y_threshold)


# --- 4. PASS 2: PRODUCTION COLORING (ADAPTIVE FULL INTERPOLATION) ---
def process_image(img_gray, col_gap_profile, cached_masks):
    h_img, w_img = img_gray.shape
    solid_mask, dotted_mask, eraser, scale_line_y_threshold = cached_masks

    color_output = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    color_output[eraser > 0] = [255, 255, 255]

    # Structurally connect lines over missing gaps
    heal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 1))
    solid_boundary = cv2.morphologyEx(solid_mask, cv2.MORPH_CLOSE, heal_kernel)
    dotted_boundary = cv2.morphologyEx(dotted_mask, cv2.MORPH_CLOSE, heal_kernel)

    red_overlay   = np.zeros_like(solid_mask)
    green_overlay = np.zeros_like(solid_mask)

    touch_kernel   = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    dilated_solid  = cv2.dilate(solid_boundary, touch_kernel, iterations=1)
    dilated_dotted = cv2.dilate(dotted_boundary, touch_kernel, iterations=1)
    intersection   = cv2.bitwise_and(dilated_solid, dilated_dotted)

    BRIDGE_GAP_PX  = 80   
    MAX_INTERP_GAP = 95  # Significantly expanded to prevent early cliff cutoffs

    if np.any(intersection > 0):
        col_has_fill    = np.zeros(w_img, dtype=bool)
        fill_top_y      = np.zeros(w_img, dtype=int)
        fill_bot_y      = np.zeros(w_img, dtype=int)
        fill_color_type = np.zeros(w_img, dtype=int)
        fill_guard_y    = np.full(w_img, -1, dtype=int)

        for c in range(w_img):
            solid_rows = np.where(solid_boundary[:, c] == 255)[0]
            
            # SOLUTION FOR ERROR G: Dynamic Bi-directional neighborhood lookup
            if len(solid_rows) == 0:
                c_start = max(0, c - 6)
                c_end = min(w_img, c + 7)
                neighborhood = np.where(solid_boundary[:, c_start:c_end] == 255)[0]
                if len(neighborhood) > 0:
                    solid_rows = neighborhood
                else:
                    continue
            
            st = int(np.min(solid_rows))
            sb = int(np.max(solid_rows))

            if (sb - st) > 110:
                local_dotted = np.where(dotted_boundary[st:sb, max(0, c-4):min(w_img, c+5)] == 255)[0]
                if len(local_dotted) == 0:
                    continue

            dotted_above = np.where(dotted_boundary[:st, c] == 255)[0]
            dotted_below = np.where(dotted_boundary[sb:, c] == 255)[0]

            is_bridge_col = col_gap_profile[c] > BRIDGE_GAP_PX if c < len(col_gap_profile) else False

            # SOLUTION FOR ERROR F: Smart crossover checking
            if len(dotted_above) > 0 and not is_bridge_col:
                boundary_y = int(np.max(dotted_above))
                if boundary_y < st:
                    col_has_fill[c]    = True
                    fill_top_y[c]      = boundary_y
                    fill_bot_y[c]      = st
                    fill_color_type[c] = 1 # Red (Cut)
            
            # Fallback evaluation for zero-crossing transitions
            if not col_has_fill[c] and len(dotted_below) > 0:
                boundary_y = int(np.min(dotted_below)) + sb
                if sb < boundary_y:
                    col_has_fill[c]    = True
                    fill_top_y[c]      = sb          
                    fill_bot_y[c]      = boundary_y
                    fill_color_type[c] = 2 # Green (Fill)
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
                    t_y    = int(fill_top_y[l]  + wt * (fill_top_y[r]  - fill_top_y[l]))
                    b_y    = int(fill_bot_y[l]  + wt * (fill_bot_y[r]  - fill_bot_y[l]))
                    c_type = fill_color_type[l] if wt <= 0.5 else fill_color_type[r]

                    g_l = fill_guard_y[l] if fill_guard_y[l] >= 0 else t_y
                    g_r = fill_guard_y[r] if fill_guard_y[r] >= 0 else t_y
                    guard = int(g_l + wt * (g_r - g_l)) if (fill_guard_y[l] >= 0 or fill_guard_y[r] >= 0) else -1

                if guard >= 0:
                    t_y = max(t_y, guard)

                t_y = max(0, min(t_y, h_img - 1))
                b_y = max(0, min(b_y, h_img - 1))

                if t_y < b_y and (b_y - t_y) < 130:
                    if c_type == 1:
                        red_overlay[t_y:b_y, c] = 255
                    elif c_type == 2:
                        green_overlay[t_y:b_y, c] = 255

            # Robust Horizontal closing morphology to completely tie off slope tails
            MAX_H_GAP = 55 
            h_stitch_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (MAX_H_GAP, 1))
            
            green_overlay[:scale_line_y_threshold, start_idx:end_idx+1] = cv2.morphologyEx(
                green_overlay[:scale_line_y_threshold, start_idx:end_idx+1], cv2.MORPH_CLOSE, h_stitch_kernel
            )
            red_overlay[:scale_line_y_threshold, start_idx:end_idx+1] = cv2.morphologyEx(
                red_overlay[:scale_line_y_threshold, start_idx:end_idx+1], cv2.MORPH_CLOSE, h_stitch_kernel
            )

    # 2D filtering clean phase
    hor_close = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
    ver_close = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))

    red_clean   = cv2.morphologyEx(red_overlay,   cv2.MORPH_CLOSE, hor_close)
    red_clean   = cv2.morphologyEx(red_clean,     cv2.MORPH_CLOSE, ver_close)
    green_clean = cv2.morphologyEx(green_overlay, cv2.MORPH_CLOSE, hor_close)
    green_clean = cv2.morphologyEx(green_clean,   cv2.MORPH_CLOSE, ver_close)

    for mask in [red_clean, green_clean]:
        n_comps, l_comps, s_comps, _ = cv2.connectedComponentsWithStats(mask, 8)
        for i in range(1, n_comps):
            if s_comps[i, cv2.CC_STAT_WIDTH] < 6:
                mask[l_comps == i] = 0

    color_output[red_clean   == 255] = [0, 0, 255]
    color_output[green_clean == 255] = [0, 210, 0]
    color_output[solid_mask == 255] = [255, 0, 0]

    # Overlay black text on top
    text_mask = (img_gray < 110) & (eraser == 0)
    color_output[text_mask] = [0, 0, 0]

    return color_output


# --- 5. MAIN EXECUTION PIPELINE ---
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

    gap_profiles = {}   
    masks_cache = {}    

    print("=" * 65)
    print("PASS 1: Mask Generation & Column Gap Profiling")
    print("=" * 65)

    for img_path in valid_file_paths:
        img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            continue
        base_name = os.path.basename(img_path)
        page_str  = extract_page_identifier(base_name)
        
        initial_masks = build_masks(img_gray)
        profile, _, saved_masks = analyze_page(img_gray, base_name, page_str, cached_masks=initial_masks)
        
        gap_profiles[base_name] = profile
        masks_cache[base_name] = saved_masks

    print("=" * 65 + "\n")

    print("🚀 PASS 2: Production coloring with bridge-aware fill...")
    for idx, img_path in enumerate(valid_file_paths, 1):
        try:
            img_gray  = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img_gray is None:
                continue
            base_name = os.path.basename(img_path)
            profile   = gap_profiles.get(base_name, np.zeros(img_gray.shape[1]))
            cached    = masks_cache.get(base_name)
            
            result    = process_image(img_gray, profile, cached_masks=cached)
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"precision_{base_name}"), result)

            if idx % 20 == 0 or idx == len(valid_file_paths):
                print(f"    ↳ {idx}/{len(valid_file_paths)} processed.")
        except Exception as e:
            print(f"⚠️  {os.path.basename(img_path)}: {e}")

    if os.path.exists(TMP_EXTRACT_DIR):
        shutil.rmtree(TMP_EXTRACT_DIR)

    print(f"\n✨ Done! Outputs saved to: '{OUTPUT_DIR}'")
