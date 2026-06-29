"""
grid_removal.py  (legacy compatibility shim)
=============================================
This file is preserved for reference but is NO LONGER the active implementation.

The grid-removal logic has been refactored into the modular pipeline:

    stage1_image_prep.py   ← projection-based grid removal (improved algorithm)
    stage1_pdf_extraction.py ← PDF → station image extraction
    stage2_component_analysis.py ← connected component analysis
    pipeline.py            ← full pipeline orchestrator

To run the complete pipeline:
    python pipeline.py

To run only grid removal on already-extracted images:
    python stage1_image_prep.py

Original Colab code is preserved below as a reference comment.
"""

# ── Original code preserved as reference ─────────────────────────────────────
#
# import cv2
# import numpy as np
# import os
# import zipfile
# import shutil
#
# FILE_PATH = "/content/bidx50197-1766424945654 (1)_output.zip"
# OUTPUT_DIR = "output_of_grid_removing_30"
# ...
# (see git history for original implementation)

print("grid_removal.py is a legacy shim. Please use pipeline.py instead.")
print("Run:  python pipeline.py")
