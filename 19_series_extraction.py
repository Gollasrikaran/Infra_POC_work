

import fitz
import os
import re

# File path
file_to_process ="/Users/sudeep/Desktop/Infra_DMT_POC/highway_planning1.pdf"

def generate_simplified_scale_report(pdf_path):
    if not os.path.exists(pdf_path):
        print(f"❌ ERROR: File '{pdf_path}' not found.")
        return

    doc = fitz.open(pdf_path)
    total_pages = len(doc)

    # Table Header (3 Columns as requested)
    print(f"{'Page #':<10} | {'Horizontal Scale':<25} | {'Vertical Scale'}")
    print("-" * 65)

    for i in range(total_pages):
        page = doc[i]
        rect = page.rect

        # 1. Focus on the Bottom-Right Corner for the hidden filter
        br_rect = fitz.Rect(rect.width * 0.70, rect.height * 0.80, rect.width, rect.height)
        corner_text = page.get_text("text", clip=br_rect).strip()

        # --- HIDDEN FILTER 1: Drawing Number (19 or 19-) ---
        # We search for it but we don't print it in the final table
        is_drawing_19 = False
        lines = corner_text.split('\n')
        for line in lines:
            clean_line = re.sub(r'(?i)DRAWING|NO\.?|DRG|[:\s]', '', line).strip()
            # Must start with 19 and NOT be a station number (no + sign)
            if (clean_line.startswith("19-") or clean_line.startswith("19")) and "+" not in clean_line:
                is_drawing_19 = True
                break

        if not is_drawing_19:
            continue

        # --- HIDDEN FILTER 2: Cross Section Check ---
        if re.search(r'(?i)cross[- \s]*section', corner_text):

            # --- SCALE EXTRACTION ---
            scale_area = fitz.Rect(0, rect.height * 0.70, rect.width, rect.height)
            page_text = page.get_text("text", clip=scale_area)

            h_match = re.search(r'(?i)HORIZONTAL[:\s]*1"\s*=\s*(\d+\'?)', page_text)
            v_match = re.search(r'(?i)VERTICAL[:\s]*1"\s*=\s*(\d+\'?)', page_text)

            h_scale = f"1\" = {h_match.group(1)}" if h_match else "1\" = 10' "
            v_scale = f"1\" = {v_match.group(1)}" if v_match else "1\" = 10' "

            # Print the final 3-column row
            print(f"{i + 1:<10} | {h_scale:<25} | {v_scale}")

    print("-" * 65)
    print("✅ Scale Report Complete.")

# Run the simplified report
generate_simplified_scale_report(file_to_process)