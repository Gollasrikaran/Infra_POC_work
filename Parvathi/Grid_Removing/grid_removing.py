"""
Module: Grid Removing

Purpose:
    Remove unwanted horizontal and vertical grid lines from engineering
    drawing images while preserving the original diagram.

Dependencies:
    pip install opencv-python numpy
"""

import os
import shutil
import zipfile
import cv2
import numpy as np


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

INPUT_ZIP = "PLANS_CALL_022_2_output.zip"
TEMP_FOLDER = "temp_extraction"
OUTPUT_FOLDER = "grid_removed_output"
OUTPUT_ZIP = "grid_removed_pages.zip"


# -------------------------------------------------------------------
# Workspace Preparation
# -------------------------------------------------------------------

def prepare_workspace():
    """Create a clean working environment."""

    for folder in [TEMP_FOLDER, OUTPUT_FOLDER]:
        if os.path.exists(folder):
            shutil.rmtree(folder)

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    if os.path.exists(OUTPUT_ZIP):
        os.remove(OUTPUT_ZIP)


# -------------------------------------------------------------------
# Grid Removal Algorithm
# -------------------------------------------------------------------

def remove_grid_lines(gray_image):
    """
    Removes faint horizontal and vertical grid lines while
    preserving important diagram structures.
    """

    _, binary = cv2.threshold(
        gray_image,
        242,
        255,
        cv2.THRESH_BINARY_INV
    )

    height, width = gray_image.shape

    grid_mask = np.zeros_like(gray_image)

    # Detect horizontal grid lines
    horizontal_projection = np.sum(binary > 0, axis=1)
    grid_rows = np.where(horizontal_projection > width * 0.40)[0]

    for row in grid_rows:
        grid_mask[row, :] = 255

    # Detect vertical grid lines
    vertical_projection = np.sum(binary > 0, axis=0)
    grid_columns = np.where(vertical_projection > height * 0.35)[0]

    for column in grid_columns:
        grid_mask[:, column] = 255

    # Protect diagram structures
    diagram = cv2.subtract(binary, grid_mask)

    protection = cv2.dilate(
        diagram,
        np.ones((3, 3), np.uint8),
        iterations=1
    )

    eraser = cv2.subtract(grid_mask, protection)

    cleaned = gray_image.copy()
    cleaned[eraser > 0] = 255

    return cleaned


# -------------------------------------------------------------------
# Image Processing
# -------------------------------------------------------------------

def process_images():

    if not os.path.exists(INPUT_ZIP):
        print(f"ERROR: {INPUT_ZIP} not found.")
        return

    print("Extracting images...")

    with zipfile.ZipFile(INPUT_ZIP, "r") as archive:
        archive.extractall(TEMP_FOLDER)

    image_files = []

    for root, _, files in os.walk(TEMP_FOLDER):

        for file in files:

            if (
                file.lower().endswith(
                    (".png", ".jpg", ".jpeg")
                )
                and "__MACOSX" not in root
                and not file.startswith(".")
            ):
                image_files.append(
                    os.path.join(root, file)
                )

    print(f"Found {len(image_files)} images.")

    with zipfile.ZipFile(
        OUTPUT_ZIP,
        "w",
        zipfile.ZIP_DEFLATED
    ) as output_zip:

        for image_path in image_files:

            try:

                gray = cv2.imread(
                    image_path,
                    cv2.IMREAD_GRAYSCALE
                )

                if gray is None:
                    continue

                cleaned = remove_grid_lines(gray)

                filename = (
                    "cleaned_" +
                    os.path.basename(image_path)
                )

                save_path = os.path.join(
                    OUTPUT_FOLDER,
                    filename
                )

                cv2.imwrite(save_path, cleaned)

                _, buffer = cv2.imencode(
                    ".png",
                    cleaned
                )

                output_zip.writestr(
                    filename,
                    buffer.tobytes()
                )

                print(f"Processed: {filename}")

            except Exception as error:
                print(f"Failed: {image_path}")
                print(error)

    shutil.rmtree(TEMP_FOLDER)

    print("\nGrid Removal Completed Successfully.")


# -------------------------------------------------------------------
# Program Entry
# -------------------------------------------------------------------

if __name__ == "__main__":

    prepare_workspace()
    process_images()