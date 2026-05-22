
import fitz
import os
import re
import shutil

# --- CONFIGURATION ---
# Just update this path. The code will handle the folder naming automatically!
file_to_process = "/Users/sudeep/Desktop/Infra_DMT_POC/highway_planning1.pdf"

def extract_to_unique_folder(pdf_path):
    if not os.path.exists(pdf_path):
        print(f"❌ ERROR: File '{pdf_path}' not found.")
        return

    # --- THE CHANGE: UNIQUE FOLDER LOGIC ---
    # This takes the PDF name (e.g., 'Drawing1.pdf') and uses it as the folder name.
    pdf_name = os.path.basename(pdf_path).replace(".pdf", "")
    output_dir = os.path.join("/content", pdf_name + "_output")

    # If the folder already exists for THIS specific PDF, we clear it.
    # But it will NOT touch folders from other PDFs.
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    doc = fitz.open(pdf_path)
    print(f"🚀 Processing: {pdf_name}")
    print(f"📂 Saving to: {output_dir}")

    for i in range(len(doc)):
        page = doc[i]
        rect = page.rect

        # Filter for 19-series Cross Sections
        br_rect = fitz.Rect(rect.width * 0.65, rect.height * 0.75, rect.width, rect.height)
        corner_text = page.get_text("text", clip=br_rect).strip()

        is_19_series = False
        for line in corner_text.split('\n'):
            clean_line = re.sub(r'(?i)DRAWING|NO\.?|DRG|[:\s]', '', line).strip()
            if (clean_line.startswith("19-") or clean_line.startswith("19")) and "+" not in clean_line:
                is_19_series = True
                break

        if is_19_series and re.search(r'(?i)cross[- \s]*section', corner_text):
            right_strip = fitz.Rect(rect.width * 0.80, 0, rect.width, rect.height)
            station_labels = page.search_for("+", clip=right_strip)
            station_labels.sort(key=lambda x: x.y0)

            last_bottom_cut = 35

            for j, label in enumerate(station_labels):
                y_top = last_bottom_cut
                current_bottom_target = label.y1 + 48

                if j + 1 < len(station_labels):
                    y_bottom = min(current_bottom_target, station_labels[j+1].y0 - 20)
                else:
                    y_bottom = min(current_bottom_target, rect.height * 0.88)

                last_bottom_cut = y_bottom

                crop_rect = fitz.Rect(0, y_top, rect.width, y_bottom)

                # Filename logic
                text_area = fitz.Rect(rect.width * 0.80, label.y0 - 30, rect.width, label.y1 + 30)
                sta_val = page.get_text("text", clip=text_area).strip()
                clean_name = re.sub(r'[^0-9+]', '', sta_val)
                if not clean_name: clean_name = f"sta_{j+1}"

                # High-Res Render
                pix = page.get_pixmap(matrix=fitz.Matrix(4, 4), clip=crop_rect)
                filename = f"Page{i+1}_Sta_{clean_name}.png"
                pix.save(os.path.join(output_dir, filename))

    print("-" * 30)
    print(f"✅ SUCCESS! Results saved in separate folder: {output_dir}")

# Run the process
extract_to_unique_folder(file_to_process)