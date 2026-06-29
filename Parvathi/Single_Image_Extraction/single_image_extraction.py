"""
Module: Single Image Extraction

Purpose:
    Extract individual cross-section images from 19-series engineering
    drawing PDFs and save them as high-resolution PNG images.

Dependencies:
    pip install pymupdf
"""

import os
import re
import shutil
import fitz


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

PDF_FILE = "SR 136 AT LOOKOUT CREEK - STAGE 2.pdf"
OUTPUT_BASE = "."


# -------------------------------------------------------------------
# Create Output Folder
# -------------------------------------------------------------------

def create_output_folder(pdf_path):
    """
    Creates a unique output folder based on the PDF name.
    """

    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    output_dir = os.path.join(OUTPUT_BASE, f"{pdf_name}_output")

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    os.makedirs(output_dir)

    return output_dir


# -------------------------------------------------------------------
# Check 19-Series Cross Section
# -------------------------------------------------------------------

def is_valid_cross_section(corner_text):
    """
    Returns True if the page belongs to a 19-series
    Cross Section drawing.
    """

    is_19 = False

    for line in corner_text.splitlines():

        clean_line = re.sub(
            r'(?i)DRAWING|NO\.?|DRG|[:\s]',
            '',
            line
        ).strip()

        if (
            (clean_line.startswith("19") or clean_line.startswith("19-"))
            and "+" not in clean_line
        ):
            is_19 = True
            break

    return (
        is_19 and
        re.search(r'(?i)cross[-\s]*sections?', corner_text)
    )


# -------------------------------------------------------------------
# Save Individual Images
# -------------------------------------------------------------------

def save_cross_sections(page, page_number, output_dir):

    rect = page.rect

    right_strip = fitz.Rect(
        rect.width * 0.80,
        0,
        rect.width,
        rect.height
    )

    station_labels = page.search_for("+", clip=right_strip)
    station_labels.sort(key=lambda item: item.y0)

    last_bottom = 35

    for index, label in enumerate(station_labels):

        top = last_bottom
        current_bottom = label.y1 + 48

        if index + 1 < len(station_labels):
            bottom = min(
                current_bottom,
                station_labels[index + 1].y0 - 20
            )
        else:
            bottom = min(
                current_bottom,
                rect.height * 0.88
            )

        last_bottom = bottom

        crop = fitz.Rect(
            0,
            top,
            rect.width,
            bottom
        )

        text_area = fitz.Rect(
            rect.width * 0.80,
            label.y0 - 30,
            rect.width,
            label.y1 + 30
        )

        station = page.get_text(
            "text",
            clip=text_area
        ).strip()

        station = re.sub(
            r'[^0-9+]',
            '',
            station
        )

        if not station:
            station = f"sta_{index+1}"

        image = page.get_pixmap(
            matrix=fitz.Matrix(4, 4),
            clip=crop
        )

        filename = f"Page{page_number}_Sta_{station}.png"

        image.save(
            os.path.join(output_dir, filename)
        )


# -------------------------------------------------------------------
# Main Extraction Function
# -------------------------------------------------------------------

def extract_single_images(pdf_path):

    if not os.path.exists(pdf_path):
        print(f"ERROR: {pdf_path} not found.")
        return

    output_dir = create_output_folder(pdf_path)

    document = fitz.open(pdf_path)

    print(f"Processing : {os.path.basename(pdf_path)}")
    print(f"Output     : {output_dir}")

    for page_number in range(len(document)):

        page = document[page_number]
        rect = page.rect

        corner_area = fitz.Rect(
            rect.width * 0.65,
            rect.height * 0.75,
            rect.width,
            rect.height
        )

        corner_text = page.get_text(
            "text",
            clip=corner_area
        ).strip()

        if is_valid_cross_section(corner_text):
            save_cross_sections(
                page,
                page_number + 1,
                output_dir
            )

    print("-" * 60)
    print("Image Extraction Completed Successfully.")


# -------------------------------------------------------------------
# Program Entry
# -------------------------------------------------------------------

if __name__ == "__main__":
    extract_single_images(PDF_FILE)