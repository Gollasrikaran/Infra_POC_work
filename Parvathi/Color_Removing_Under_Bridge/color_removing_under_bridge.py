"""
Module: Color Removing Under Bridge

Purpose:
    Remove unwanted color beneath bridge structures while preserving
    engineering drawing components and embedded text.

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
OUTPUT_FOLDER = "bridge_color_removed_output"


# -------------------------------------------------------------------
# Workspace Preparation
# -------------------------------------------------------------------

def prepare_workspace():
    """Create a clean working environment."""

    for folder in [TEMP_FOLDER, OUTPUT_FOLDER]:
        if os.path.exists(folder):
            shutil.rmtree(folder)

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# -------------------------------------------------------------------
# Grid Removal
# -------------------------------------------------------------------

def remove_grid(gray_image):
    """
    Remove background grid while preserving drawing structures.
    """

    # Move your STEP 1 code here.

    pass


# -------------------------------------------------------------------
# Detect Drawing Components
# -------------------------------------------------------------------

def detect_components(clean_image):
    """
    Detect solid lines and dotted lines using connected components.
    """

    # Move your STEP 2 code here.

    pass


# -------------------------------------------------------------------
# Remove Color Under Bridge
# -------------------------------------------------------------------

def remove_bridge_color(gray_image):
    """
    Apply bridge intersection validation and remove unwanted
    colors beneath bridge regions.
    """

    # Move your STEP 3
    # STEP 4
    # STEP 5

    # Return final processed image

    pass


# -------------------------------------------------------------------
# Process Images
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

    for image_path in image_files:

        try:

            gray = cv2.imread(
                image_path,
                cv2.IMREAD_GRAYSCALE
            )

            if gray is None:
                continue

            processed = remove_bridge_color(gray)

            filename = (
                "bridge_" +
                os.path.basename(image_path)
            )

            cv2.imwrite(
                os.path.join(
                    OUTPUT_FOLDER,
                    filename
                ),
                processed
            )

            print(f"Processed: {filename}")

        except Exception as error:

            print(error)

    shutil.rmtree(TEMP_FOLDER)

    print("\nBridge Color Removal Completed Successfully.")


# -------------------------------------------------------------------
# Program Entry
# -------------------------------------------------------------------

if __name__ == "__main__":

    prepare_workspace()
    process_images()