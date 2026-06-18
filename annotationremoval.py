!pip install opencv-python-headless -q

import cv2
import os
import zipfile
import shutil
import numpy as np
from google.colab.patches import cv2_imshow

# =========================================================
# PATHS
# =========================================================
ZIP_PATH = "/content/grid_removing (1) (1).zip"

TEMP_DIR = "temp_profile_input"
OUTPUT_DIR = "profile_only_images_output"

for folder in [TEMP_DIR, OUTPUT_DIR]:
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

with zipfile.ZipFile(ZIP_PATH, "r") as z:
    z.extractall(TEMP_DIR)

# =========================================================
# PROFILE ISOLATION LOGIC
# =========================================================
def isolate_profile_lines(img):

    original = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape

    # binary threshold
    _, bw = cv2.threshold(
        gray,
        210,
        255,
        cv2.THRESH_BINARY_INV
    )

    work = bw.copy()

    # remove scale / annotation zones
    work[0:int(h * 0.15), :] = 0
    work[int(h * 0.84):h, :] = 0
    work[:, 0:int(w * 0.08)] = 0
    work[:, int(w * 0.92):w] = 0

    # =====================================================
    # DETECT SOLID / SLOPED PROFILE LINES
    # =====================================================
    edges = cv2.Canny(gray, 40, 140)

    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=25,
        minLineLength=70,
        maxLineGap=45
    )

    solid_mask = np.zeros_like(bw)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]

            length = np.hypot(x2 - x1, y2 - y1)
            angle = abs(
                np.degrees(
                    np.arctan2(y2 - y1, x2 - x1)
                )
            )

            # region filters
            if y1 < h * 0.15 or y2 < h * 0.15:
                continue

            if y1 > h * 0.84 or y2 > h * 0.84:
                continue

            if x1 < w * 0.08 or x2 < w * 0.08:
                continue

            if x1 > w * 0.92 or x2 > w * 0.92:
                continue

            # keep only horizontal/sloped profile lines
            if length > 70 and angle < 75:
                cv2.line(
                    solid_mask,
                    (x1, y1),
                    (x2, y2),
                    255,
                    2
                )

    # =====================================================
    # DETECT DOTTED EXISTING GROUND
    # =====================================================
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        work,
        8
    )

    dotted_mask = np.zeros_like(bw)

    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        ww = stats[i, cv2.CC_STAT_WIDTH]
        hh = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        if y < h * 0.15:
            continue

        if y > h * 0.84:
            continue

        if x < w * 0.08 or x > w * 0.92:
            continue

        aspect = ww / max(hh, 1)

        # dotted profile dash rule
        if (
            3 <= ww <= 45
            and 2 <= hh <= 14
            and aspect > 1.2
            and area <= 350
        ):
            dotted_mask[labels == i] = 255

    dotted_mask = cv2.morphologyEx(
        dotted_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (28, 1))
    )

    # =====================================================
    # COMBINE PROFILE MASK
    # =====================================================
    profile_mask = cv2.bitwise_or(
        solid_mask,
        dotted_mask
    )

    # clean unwanted regions again
    profile_mask[0:int(h * 0.15), :] = 0
    profile_mask[int(h * 0.84):h, :] = 0
    profile_mask[:, 0:int(w * 0.08)] = 0
    profile_mask[:, int(w * 0.92):w] = 0

    # reconnect profile segments
    profile_mask = cv2.morphologyEx(
        profile_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (20, 2))
    )

    profile_mask = cv2.dilate(
        profile_mask,
        np.ones((2, 2), np.uint8),
        iterations=1
    )

    # =====================================================
    # REMOVE SMALL ARTIFACTS
    # =====================================================
    n2, lab2, st2, _ = cv2.connectedComponentsWithStats(
        profile_mask,
        8
    )

    final_mask = np.zeros_like(profile_mask)

    for i in range(1, n2):
        x = st2[i, cv2.CC_STAT_LEFT]
        y = st2[i, cv2.CC_STAT_TOP]
        ww = st2[i, cv2.CC_STAT_WIDTH]
        hh = st2[i, cv2.CC_STAT_HEIGHT]
        area = st2[i, cv2.CC_STAT_AREA]

        # remove tiny artifacts
        if area < 40:
            continue

        # remove vertical annotation-like blobs
        if hh > 90 and ww < 18:
            continue

        # keep long profile-like components
        if ww > 20:
            final_mask[lab2 == i] = 255

    # =====================================================
    # FINAL OUTPUT IMAGE
    # =====================================================
    output = np.full_like(original, 255)
    output[final_mask > 0] = [0, 0, 0]

    return output

# =========================================================
# PROCESS IMAGES
# =========================================================
image_files = []

for root, dirs, files in os.walk(TEMP_DIR):
    for f in files:
        if f.lower().endswith((".png", ".jpg", ".jpeg")):
            image_files.append(os.path.join(root, f))

image_files = sorted(image_files)

print("Total images:", len(image_files))

for idx, path in enumerate(image_files, start=1):
    img = cv2.imread(path)

    if img is None:
        continue

    result = isolate_profile_lines(img)

    out_img_path = os.path.join(
        OUTPUT_DIR,
        f"profile_only_page_{idx}.png"
    )

    cv2.imwrite(out_img_path, result)

    print("Saved image:", out_img_path)

# =========================================================
# ZIP ONLY PNG OUTPUTS
# =========================================================
zip_name = "profile_only_images_output.zip"

with zipfile.ZipFile(zip_name, "w") as zipf:
    for f in os.listdir(OUTPUT_DIR):
        if f.endswith(".png"):
            zipf.write(
                os.path.join(OUTPUT_DIR, f),
                f
            )

print("DONE:", zip_name)

# =========================================================
# SHOW SAMPLE PNG
# =========================================================
png_files = [
    f for f in os.listdir(OUTPUT_DIR)
    if f.endswith(".png")
]

if len(png_files) > 0:
    sample_path = os.path.join(OUTPUT_DIR, png_files[0])
    sample = cv2.imread(sample_path)
    cv2_imshow(sample)
else:
    print("No PNG images found.")
