"""
Module: 19-Series Extraction with Scale Detection

Purpose:
    Extract horizontal and vertical scale information from
    Cross Section drawings whose drawing number starts with 19.

Dependencies:
    pip install pymupdf
"""

import os
import re
import fitz


# -------------------------------------------------------------------
# Input PDF
# -------------------------------------------------------------------
PDF_FILE = "SR 136 AT LOOKOUT CREEK - STAGE 2.pdf"


# -------------------------------------------------------------------
# Function: Check whether the drawing belongs to the 19-Series
# -------------------------------------------------------------------
def is_19_series(corner_text):
    """
    Returns True if the drawing number starts with 19
    and is not a station number.
    """

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
            return True

    return False


# -------------------------------------------------------------------
# Function: Extract Horizontal & Vertical Scale
# -------------------------------------------------------------------
def extract_scale(page, page_rect):
    """
    Extracts Horizontal and Vertical Scale from the page.
    """

    scale_area = fitz.Rect(
        0,
        page_rect.height * 0.70,
        page_rect.width,
        page_rect.height
    )

    page_text = page.get_text("text", clip=scale_area)

    horizontal = re.search(
        r'(?i)HORIZONTAL[:\s]*1"\s*=\s*(\d+\'?)',
        page_text
    )

    vertical = re.search(
        r'(?i)VERTICAL[:\s]*1"\s*=\s*(\d+\'?)',
        page_text
    )

    horizontal_scale = (
        f'1" = {horizontal.group(1)}'
        if horizontal
        else '1" = 10\''
    )

    vertical_scale = (
        f'1" = {vertical.group(1)}'
        if vertical
        else '1" = 10\''
    )

    return horizontal_scale, vertical_scale


# -------------------------------------------------------------------
# Main Function
# -------------------------------------------------------------------
def generate_scale_report(pdf_path):

    if not os.path.exists(pdf_path):
        print(f"ERROR: File not found -> {pdf_path}")
        return

    document = fitz.open(pdf_path)

    print(f"{'Page':<8} | {'Horizontal Scale':<22} | Vertical Scale")
    print("-" * 65)

    for page_number in range(len(document)):

        page = document[page_number]
        rect = page.rect

        # Bottom-right area
        corner_area = fitz.Rect(
            rect.width * 0.70,
            rect.height * 0.80,
            rect.width,
            rect.height
        )

        corner_text = page.get_text(
            "text",
            clip=corner_area
        ).strip()

        # Filter 1 : 19-Series
        if not is_19_series(corner_text):
            continue

        # Filter 2 : Cross Section
        if not re.search(r'(?i)cross[-\s]*section', corner_text):
            continue

        horizontal, vertical = extract_scale(page, rect)

        print(
            f"{page_number + 1:<8} | "
            f"{horizontal:<22} | "
            f"{vertical}"
        )

    print("-" * 65)
    print("Scale Report Generated Successfully.")


# -------------------------------------------------------------------
# Program Entry
# -------------------------------------------------------------------
if __name__ == "__main__":
    generate_scale_report(PDF_FILE)