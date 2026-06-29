"""
Module: Filling Empty Space

Purpose:
    Fill empty gaps between detected engineering drawing components
    while preserving the original drawing structure.

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

INPUT_ZIP = "perfect_grid_removed.zip"
TEMP_FOLDER = "temp_extraction"
OUTPUT_FOLDER = "filled_output"
OUTPUT_ZIP = "filled_output.zip"


# -------------------------------------------------------------------
# Workspace Preparation
# -------------------------------------------------------------------

def prepare_workspace():
    """Create a clean workspace."""

    for folder in [TEMP_FOLDER, OUTPUT_FOLDER]:
        if os.path.exists(folder):
            shutil.rmtree(folder)

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    if os.path.exists(OUTPUT_ZIP):
        os.remove(OUTPUT_ZIP)


# -------------------------------------------------------------------
# Fill Horizontal Gaps
# -------------------------------------------------------------------

def fill_horizontal_gaps(mask, max_gap=60):
    """
    Fill small horizontal gaps between connected components.
    """

    filled = mask.copy()

    height, width = mask.shape

    for row in range(height):

        active_pixels = np.where(mask[row] == 255)[0]

        if len(active_pixels) < 2:
            continue

        gaps = np.diff(active_pixels)

        for index, gap in enumerate(gaps):

            if 1 < gap <= max_gap:

                start = active_pixels[index]
                end = active_pixels[index + 1]

                filled[row, start:end] = 255

    return filled


# -------------------------------------------------------------------
# Image Processing Logic
# -------------------------------------------------------------------

def process_image(gray_image):
    """
    Process engineering drawing image and fill empty regions.
    """

    # Move your complete image-processing logic here.
    # This includes:
    #
    # - Grid removal
    # - Connected component analysis
    # - Solid line detection
    # - Dotted line detection
    # - Vertical propagation
    # - Gap filling
    # - Coloring
    # - Text restoration
    #
    # Finally:
    #
    # return processed_image

    pass


# -------------------------------------------------------------------
# Process All Images
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
    ) as archive:

        for image_path in image_files:

            try:

                gray = cv2.imread(
                    image_path,
                    cv2.IMREAD_GRAYSCALE
                )

                if gray is None:
                    continue

                processed = process_image(gray)

                filename = (
                    "filled_" +
                    os.path.basename(image_path)
                )

                save_path = os.path.join(
                    OUTPUT_FOLDER,
                    filename
                )

                cv2.imwrite(save_path, processed)

                _, buffer = cv2.imencode(
                    ".png",
                    processed
                )

                archive.writestr(
                    filename,
                    buffer.tobytes()
                )

                print(f"Processed: {filename}")

            except Exception as error:
                print(error)

    shutil.rmtree(TEMP_FOLDER)

    print("Empty Space Filling Completed Successfully.")


# -------------------------------------------------------------------
# Program Entry
# -------------------------------------------------------------------

if __name__ == "__main__":

    prepare_workspace()
    process_images()