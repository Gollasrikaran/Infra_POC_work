import os
import zipfile
import cv2
import re
import pytesseract

# --- CONFIGURATION ---
zip_path = "/content/perfect_grid_removed.zip"
extract_to_dir = "/content/unzipped_stations"

# Unzip the archive file
if os.path.exists(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to_dir)
else:
    print(f"❌ ERROR: Zip file not found at '{zip_path}'")
    exit()

# Gather and sort all images numerically/alphabetically
supported_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
image_files = sorted([f for f in os.listdir(extract_to_dir) if f.lower().endswith(supported_extensions)])

# --- CLEAN STRUCTURED TABLE HEADER ---
print(f"{'Page / Index':<15} | {'Station Number'}")
print("-" * 35)

for index, img_name in enumerate(image_files, start=1):
    img_path = os.path.join(extract_to_dir, img_name)

    img = cv2.imread(img_path)
    if img is None:
        continue

    height, width, _ = img.shape
    station_number = None

    # --- TWO-STRATEGY ROBUST SEARCH SYSTEM ---
    # Strategy A: Use your 50% Right Width Crop
    # Strategy B: Fallback to Full Width (100%) minus 50px left margin if A misses
    strategies = [
        ("50% Right Crop", img[0:height, int(width * 0.50):width]),
        ("Full Width Scan", img[0:height, min(50, width):width])
    ]

    for strat_name, roi in strategies:
        if station_number:
            break

        # --- MULTI-STAGE ENHANCEMENT LOOP ---
        for stage in range(3):
            if station_number:
                break
                
            if stage == 0:
                processed_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            elif stage == 1:
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                _, processed_roi = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            elif stage == 2:
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                processed_roi = cv2.resize(gray, (0,0), fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

            # Test both PSM configs (6 for layout, 11 for scattered text elements)
            for psm_mode in ['--psm 6', '--psm 11']:
                extracted_text = pytesseract.image_to_string(processed_roi, config=psm_mode)

                # Clean common text typos
                clean_text = re.sub(r'\s+', '', extracted_text)
                clean_text = clean_text.upper().replace('O', '0')

                # Match structural pattern: XX+XX
                match = re.search(r'\d{2}\+\d{2}', clean_text)
                if match:
                    station_number = match.group(0)
                    break

    # Final check if absolutely all strategies failed
    if not station_number:
        station_number = "NOT FOUND"

    # Print the exact formatted line item row
    print(f"{index:<15} | {station_number}")

print("-" * 35)