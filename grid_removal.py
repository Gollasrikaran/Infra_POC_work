import cv2
import numpy as np
import os
import zipfile
import shutil

# --- 1. SETUP ---
FILE_PATH = "/content/bidx50197-1766424945654 (1)_output.zip"
OUTPUT_DIR = "output_of_grid_removing_30"

if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR)

def process_surgical_clean(img_gray, filename):
    # a. Threshold - Capture the drawing (UNTOUCHED ORIGINAL LOGIC)
    _, bw = cv2.threshold(img_gray, 235, 255, cv2.THRESH_BINARY_INV)

    # b. Detect Background Grid Lines ONLY (UNTOUCHED ORIGINAL LOGIC)
    ver_k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 200))
    hor_k = cv2.getStructuringElement(cv2.MORPH_RECT, (450, 1))

    ver_grid = cv2.morphologyEx(bw, cv2.MORPH_OPEN, ver_k)
    hor_grid = cv2.morphologyEx(bw, cv2.MORPH_OPEN, hor_k)
    grid_mask = cv2.add(ver_grid, hor_grid)

    # c. BRIDGE PROTECTION (UNTOUCHED ORIGINAL LOGIC)
    diagram_only = cv2.subtract(bw, grid_mask)
    bridge_shield = cv2.dilate(diagram_only, np.ones((3,3), np.uint8), iterations=1)

    # d. THE ERASE (UNTOUCHED ORIGINAL LOGIC)
    eraser = cv2.subtract(grid_mask, bridge_shield)

    # --- e. APPLY TO ORIGINAL (EARTHWORK VOLUME GENERATION ENGINE) ---
    # 1. Clear background grid artifacts to pure white using original parameters
    clean_bw_image = img_gray.copy()
    clean_bw_image[eraser > 0] = 255

    # Convert base to 3-channel color (BGR) to enable color flooding
    color_output = cv2.cvtColor(clean_bw_image, cv2.COLOR_GRAY2BGR)

    # 2. Extract separate pixel maps for the two main tracking lines
    clean_drawing_ink = cv2.subtract(bw, eraser)
    nlabels, labels, stats, _ = cv2.connectedComponentsWithStats(clean_drawing_ink, 8, cv2.CV_32S)

    h_img, w_img = img_gray.shape
    border_w = int(w_img * 0.08)
    border_h = int(h_img * 0.08)

    # Initialize tracking arrays to map the profiles column-by-column
    y_ground = np.zeros(w_img, dtype=np.int32)
    y_design = np.zeros(w_img, dtype=np.int32)

    # 3. Classify components and find profile paths using pixel statistics
    for i in range(1, nlabels):
        x, y, w, h, area = stats[i]

        if x < border_w or (x + w) > (w_img - border_w) or y < border_h or (y + h) > (h_img - border_h):
            continue

        diag_len = np.sqrt(w**2 + h**2)
        aspect_ratio = w / h if h > 0 else 0

        # PROFILE A: Track the Dotted Ground Line (Dense small dot configurations)
        if 3 <= w <= 20 and 2 <= h <= 12 and aspect_ratio <= 1.5:
            for col in range(x, x + w):
                rows = np.where(labels[:, col] == i)[0]
                if len(rows) > 0:
                    y_ground[col] = int(np.mean(rows)) # Assign vertical position
            continue

        # PROFILE B: Track the Solid Road Template Box
        elif diag_len > 55 or w > 50 or h > 50:
            for col in range(x, x + w):
                rows = np.where(labels[:, col] == i)[0]
                if len(rows) > 0:
                    y_design[col] = int(np.max(rows)) # Lock to the lowest edge line
            continue

    # 4. Math Cross-Over Scan: Flood earthwork zones and mark intersections
    last_state = 0 # 1 if ground was above (Cut), -1 if design was above (Fill)

    for col in range(border_w, w_img - border_w):
        yg = y_ground[col]
        yd = y_design[col]

        # Process only when both lines exist concurrently in the horizontal column space
        if yg > 0 and yd > 0:
            current_state = 1 if yg < yd else -1 # (In pixel space, lower row coordinate value means higher position)

            # Draw the relative Earthwork fill gaps column-by-column
            if yg < yd:
                # Ground is higher up than Design template -> CUT ZONE (RED)
                cv2.line(color_output, (col, yg), (col, yd), (0, 0, 255), 1)
            else:
                # Design template is higher up than Ground -> FILL ZONE (GREEN)
                cv2.line(color_output, (col, yd), (col, yg), (0, 255, 0), 1)

            # Identify intersection crossover points (State Transition Boundary)
            if last_state != 0 and current_state != last_state:
                # Place a clear vertical demarcation line at the intersection boundary (BLUE)
                cv2.line(color_output, (col, min(yg, yd) - 5), (col, max(yg, yd) + 5), (255, 0, 0), 2)

            last_state = current_state

    # f. SAVE (UNTOUCHED ORIGINAL LOGIC)
    save_path = os.path.join(OUTPUT_DIR, f"perfect_{filename}")
    cv2.imwrite(save_path, color_output)

# --- 3. RUN (UNTOUCHED ORIGINAL LOGIC) ---
if not os.path.exists(FILE_PATH):
    print(f"❌ ZIP file not found.")
else:
    with zipfile.ZipFile(FILE_PATH, 'r') as z:
        items = [i for i in z.namelist() if i.lower().endswith(('.png', '.jpg')) and "__MACOSX" not in i]
        for item in items:
            with z.open(item) as f:
                img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    process_surgical_clean(img, os.path.basename(item))

print(f"\n✨ FINISHED! Earthwork zones completely generated. Cut sections filled with Red, Fill sections filled with Green, and intersection boundaries marked in Blue.")
