import cv2
import numpy as np
import pytesseract
import re

def extract_station_from_image(img_bytes: bytes) -> str:
    # 1. Decode bytes directly to OpenCV image
    np_arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
    if img is None:
        return "Unknown"

    height, width, _ = img.shape
    station_number = None

    # 2. Apply your Colab strategies
    strategies = [
        ("50% Right Crop", img[0:height, int(width * 0.50):width]),
        ("Full Width Scan", img[0:height, min(50, width):width])
    ]

    for strat_name, roi in strategies:
        if station_number:
            break
            
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

            for psm_mode in ['--psm 6', '--psm 11']:
                extracted_text = pytesseract.image_to_string(processed_roi, config=psm_mode)
                clean_text = re.sub(r'\s+', '', extracted_text).upper().replace('O', '0')
                match = re.search(r'\d{1,4}\+\d{2}', clean_text)
                
                if match:
                    station_number = match.group(0)
                    break

    return station_number if station_number else "Unknown"
