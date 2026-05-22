import fitz  # PyMuPDF
import cv2
import numpy as np
from PIL import Image
import os

def process_scanline_intersection(input_pdf_path, output_pdf_path):
    print(f"Opening Scanline Engine for: {input_pdf_path}")
    doc = fitz.open(input_pdf_path)
    total_pages = len(doc)
    
    processed_pages = []
    
    for page_idx in range(total_pages):
        print(f"Scanning sheet {page_idx + 1}/{total_pages}...")
        page = doc.load_page(page_idx)
        
        # Render the PDF page image at high-resolution (300 DPI)
        zoom = 300 / 72
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        
        # Convert to OpenCV BGR matrix
        img_data = np.frombuffer(pix.samples, dtype=np.uint8)
        img_data = img_data.reshape((pix.h, pix.w, pix.n))
        img = cv2.cvtColor(img_data, cv2.COLOR_RGBA2BGR) if pix.n == 4 else cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)
        
        height, width, _ = img.shape
        output_canvas = img.copy()
        
        # 1. Convert to grayscale and threshold to isolate black infrastructure lines
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Target dark lines (both solid design lines and dashed ground lines)
        _, binary_lines = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)
        
        # Remove vertical grid lines so they don't trick our horizontal scanner
        vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
        vert_lines = cv2.morphologyEx(binary_lines, cv2.MORPH_OPEN, vert_kernel)
        clean_lines = cv2.subtract(binary_lines, vert_lines)
        
        # 2. Line-by-Line Horizontal Scanline Processing
        # We scan the vertical region where engineering profiles typically exist (skip top/bottom margins)
        start_row = int(height * 0.2)
        end_row = int(height * 0.8)
        
        for y in range(start_row, end_row):
            # Find all pixel indexes where a line exists in this specific row
            row_pixels = clean_lines[y, :]
            line_intersections = np.where(row_pixels > 0)[0]
            
            # We need at least two line boundaries (Solid and Dotted) crossing this Y row to fill between them
            if len(line_intersections) >= 2:
                # Group adjacent pixels to find distinct line crossings
                distinct_crossings = []
                if len(line_intersections) > 0:
                    distinct_crossings.append(line_intersections[0])
                    for i in range(1, len(line_intersections)):
                        # If the gap between pixels is greater than 5, it's a separate line profile
                        if line_intersections[i] - line_intersections[i-1] > 5:
                            distinct_crossings.append(line_intersections[i])
                
                # Draw filling lines line-by-line between pairs of profiles
                for i in range(0, len(distinct_crossings) - 1, 2):
                    x_start = distinct_crossings[i]
                    x_end = distinct_crossings[i+1]
                    
                    # Prevent accidental full-screen lines caused by border margins
                    if (x_end - x_start) < (width * 0.4):
                        # Determine color logic based on original image characteristics
                        # Check the midpoint pixel color in the original image to determine if it is a Cut or Fill zone
                        mid_x = int((x_start + x_end) / 2)
                        original_pixel = img[y, mid_x]
                        
                        # Check if original CAD pixel had a tint of green or red hatching
                        # original_pixel is in BGR format
                        b, g, r = int(original_pixel[0]), int(original_pixel[1]), int(original_pixel[2])
                        
                        if g > r and g > b:
                            color = (144, 238, 144)  # Fill (Light Green line)
                        elif r > g and r > b:
                            color = (180, 180, 255)  # Cut (Light Red line)
                        else:
                            # Fallback if scanning neutral zones between profiles
                            continue
                            
                        # Draw a 1-pixel thick horizontal stroke line filling the intersection part perfectly
                        cv2.line(output_canvas, (x_start, y), (x_end, y), color, 1)
        
        # Blend the filled color layer gently with the original line definitions
        # This keeps the original structural black text and profile lines perfectly visible on top
        cv2.addWeighted(output_canvas, 0.6, img, 0.4, 0, output_canvas)
        
        # Convert final canvas to PIL image sequence
        rgb_final = cv2.cvtColor(output_canvas, cv2.COLOR_BGR2RGB)
        processed_pages.append(Image.fromarray(rgb_final))
        
    # 3. Compile memory frames back into a high-res structural PDF
    if processed_pages:
        print("\nCompiling line-by-line scan results into final PDF document...")
        processed_pages[0].save(
            output_pdf_path,
            "PDF",
            resolution=300.0,
            save_all=True,
            append_images=processed_pages[1:]
        )
        print(f"Success! Perfect scanline output generated at: {output_pdf_path}")
    else:
        print("Error: Processing pipeline failed.")

# --- Execution Parameters ---
input_file = "/Users/sudeep/Desktop/Infra_DMT_POC/19series.pdf"
output_file = "scanline_intersection_output.pdf"

if os.path.exists(input_file):
    process_scanline_intersection(input_file, output_file)
else:
    print(f"File '{input_file}' not found. Please verify the file name.")