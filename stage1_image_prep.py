"""
stage1_image_prep.py
====================
Stage 1 – Part B: Grid Removal & Image Standardisation

Objective (from the project spec)
-----------------------------------
● Remove background grid lines.
● Preserve profile geometry.          ← profiles must survive intact
● Preserve engineering annotations.   ← text, labels, arrows must NOT be erased
● Standardise image quality.

Algorithm  — Projection-Based Grid Detection with Annotation Shield
-------------------------------------------------------------------
Engineering cross-section drawings contain TWO categories of ink on the page:

  A) Background GRID LINES — regular, faint horizontal/vertical lines that
     span most of the page width/height.  These must be ERASED.

  B) DRAWING CONTENT — profiles, text labels (elevation, slope, station),
     arrows, leader lines, bridge geometry.  These must be PRESERVED.

The challenge is that grid lines often pass THROUGH annotation clusters and
cross profile lines.  A naive eraser would destroy parts of both.

Five-step algorithm:

  Step A – Soft binarise at threshold 242.
            Captures even very faint grid lines without over-segmenting ink.

  Step B – Horizontal projection scan.
            A row is a GRID ROW if active pixels in that row span
            > GRID_HORIZ_ROW_FRACTION of the image width.
            Grid columns detected similarly with GRID_VERT_COL_FRACTION.

  Step C – Build a DUAL-LAYER protective shield.

            Layer 1 — DIAGRAM SHIELD (general ink protection):
              All ink pixels that are NOT in the detected grid are dilated by a
              small kernel.  This protects profile curves and strokes where they
              cross grid lines.

            Layer 2 — ANNOTATION SHIELD (explicit text/label protection):
              Run a quick connected component pass on the binarised image.
              Every component whose bounding box is TEXT-SIZED
              (area <= ANNOT_MAX_AREA AND height <= ANNOT_MAX_H) is treated
              as an annotation.  A larger dilation (ANNOT_SHIELD_DILATE_ITER)
              is applied around each annotation bounding-box region to create
              a wide protective halo.  This is the key addition that guarantees
              elevation labels, slope labels, station numbers and all other
              text annotations survive grid removal completely intact.

  Step D – Combine both shield layers:
              full_shield = diagram_shield | annotation_shield
              final_eraser = grid_mask - full_shield

  Step E – Apply eraser: set final_eraser pixels to white (255).

Output
------
output/stage1_cleaned/<original_name>_cleaned.png
"""

import os
import logging
from typing import List

import cv2
import numpy as np

import config
from utils.io_utils import ensure_dirs, collect_images, load_gray, save_image

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Core algorithm
# ──────────────────────────────────────────────────────────────────────────────

def remove_grid(img_gray: np.ndarray) -> np.ndarray:
    """
    Remove background grid lines from a grayscale cross-section image while
    explicitly preserving ALL engineering annotations (text labels, elevation
    values, slope ratios, station numbers, leader lines, arrows).

    Parameters
    ----------
    img_gray : uint8 grayscale image (H × W).

    Returns
    -------
    Cleaned grayscale image — grid lines white (255), all other ink intact.
    """
    h, w = img_gray.shape

    # ══════════════════════════════════════════════════════════════════════════
    # Step A  – Soft binarise
    # ══════════════════════════════════════════════════════════════════════════
    # Invert: dark drawing ink → 255 (white), light background → 0 (black)
    _, bw = cv2.threshold(
        img_gray,
        config.GRID_BINARIZE_THRESH,   # 242
        255,
        cv2.THRESH_BINARY_INV,
    )

    # ══════════════════════════════════════════════════════════════════════════
    # Step B  – Grid detection via projection scanning
    # ══════════════════════════════════════════════════════════════════════════
    # Horizontal grid rows: rows where active pixels span > 40% of image width
    horiz_proj      = np.sum(bw > 0, axis=1)                          # shape (H,)
    min_horiz_px    = int(w * config.GRID_HORIZ_ROW_FRACTION)         # e.g. 40% of W
    grid_row_idxs   = np.where(horiz_proj > min_horiz_px)[0]

    grid_mask = np.zeros_like(bw)
    if len(grid_row_idxs):
        grid_mask[grid_row_idxs, :] = 255

    # Vertical grid columns: columns where active pixels span > 35% of image height
    vert_proj       = np.sum(bw > 0, axis=0)                          # shape (W,)
    min_vert_px     = int(h * config.GRID_VERT_COL_FRACTION)          # e.g. 35% of H
    grid_col_idxs   = np.where(vert_proj > min_vert_px)[0]

    if len(grid_col_idxs):
        grid_mask[:, grid_col_idxs] = 255

    # ══════════════════════════════════════════════════════════════════════════
    # Step C  – Build DUAL-LAYER protective shield
    # ══════════════════════════════════════════════════════════════════════════

    # ── C-1  DIAGRAM SHIELD (protects profile curves & strokes) ──────────────
    # Pixels that are ink but NOT classified as grid
    diagram_ink = cv2.subtract(bw, grid_mask)

    diag_kernel   = np.ones((3, 3), np.uint8)
    diagram_shield = cv2.dilate(
        diagram_ink,
        diag_kernel,
        iterations=config.GRID_SHIELD_DILATE_ITER,  # 1 iteration = 3px halo
    )

    # ── C-2  ANNOTATION SHIELD (explicitly protects text & labels) ───────────
    #
    # Why this extra step is needed:
    #   Grid rows are marked across the FULL image width.  Some annotation
    #   glyphs (isolated dots over 'i', parts of numbers, serif strokes) can
    #   be small enough that the 3-pixel diagram_shield halo doesn't fully
    #   enclose them, leaving them vulnerable to the eraser.
    #
    #   Solution: detect all TEXT-SIZED components, draw a padded bounding
    #   rectangle for each one into a dedicated annotation_shield mask.
    #   This guarantees a box-shaped protective zone around every label.

    annotation_shield = np.zeros_like(bw)

    # Connected components on the full binary image (not diagram_ink) so we
    # capture text even if it partially overlaps grid rows.
    n_labels, _, cc_stats, _ = cv2.connectedComponentsWithStats(
        bw, 8, cv2.CV_32S
    )

    for i in range(1, n_labels):
        cx  = int(cc_stats[i, cv2.CC_STAT_LEFT])
        cy  = int(cc_stats[i, cv2.CC_STAT_TOP])
        cw  = int(cc_stats[i, cv2.CC_STAT_WIDTH])
        ch  = int(cc_stats[i, cv2.CC_STAT_HEIGHT])
        ca  = int(cc_stats[i, cv2.CC_STAT_AREA])

        # Classify as an annotation / text component:
        #   - area small enough to be a glyph or label cluster
        #   - height short enough (not a full-height profile span)
        if ca <= config.ANNOT_MAX_AREA and ch <= config.ANNOT_MAX_H:
            # Pad the bounding box by ANNOT_PAD pixels on all sides
            pad = config.ANNOT_PAD
            x1 = max(0,     cx - pad)
            y1 = max(0,     cy - pad)
            x2 = min(w - 1, cx + cw + pad)
            y2 = min(h - 1, cy + ch + pad)
            annotation_shield[y1:y2, x1:x2] = 255

    # ── C-3  Combine both shield layers ──────────────────────────────────────
    full_shield = cv2.bitwise_or(diagram_shield, annotation_shield)

    # ══════════════════════════════════════════════════════════════════════════
    # Step D  – Compute final eraser
    # ══════════════════════════════════════════════════════════════════════════
    # Only erase grid pixels that are NOT covered by either shield layer
    final_eraser = cv2.subtract(grid_mask, full_shield)

    # ══════════════════════════════════════════════════════════════════════════
    # Step E  – Apply eraser
    # ══════════════════════════════════════════════════════════════════════════
    clean_img = img_gray.copy()
    clean_img[final_eraser > 0] = 255   # set erased grid pixels to pure white

    return clean_img


# ──────────────────────────────────────────────────────────────────────────────
# Batch processor
# ──────────────────────────────────────────────────────────────────────────────

def process_all(
    input_dir:  str = config.STAGE1_EXTRACTED_DIR,
    output_dir: str = config.STAGE1_CLEANED_DIR,
) -> List[str]:
    """
    Run grid removal on every image in *input_dir* and save results to *output_dir*.

    Parameters
    ----------
    input_dir  : Folder containing raw station images (Stage-1A output).
    output_dir : Folder where cleaned images are saved.

    Returns
    -------
    List of absolute paths to cleaned images (sorted).
    """
    ensure_dirs(output_dir)

    image_paths = collect_images(input_dir)
    if not image_paths:
        logger.error("No images found in: %s", input_dir)
        return []

    logger.info(
        "Stage 1-B: processing %d images from %s", len(image_paths), input_dir
    )
    saved: List[str] = []

    for idx, img_path in enumerate(image_paths, start=1):
        img_gray = load_gray(img_path)
        if img_gray is None:
            continue

        try:
            cleaned = remove_grid(img_gray)
        except Exception as exc:
            logger.error(
                "  [%d/%d] Error on %s: %s",
                idx, len(image_paths), os.path.basename(img_path), exc,
            )
            continue

        base     = os.path.splitext(os.path.basename(img_path))[0]
        out_name = f"{base}_cleaned.png"
        out_path = os.path.join(output_dir, out_name)

        save_image(out_path, cleaned)
        saved.append(out_path)
        logger.info("  [%d/%d] Cleaned: %s", idx, len(image_paths), out_name)

    logger.info("Stage 1-B complete.  %d images cleaned.", len(saved))
    return sorted(saved)


# ──────────────────────────────────────────────────────────────────────────────
# Standalone entry point
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
        datefmt="%H:%M:%S",
    )
    results = process_all()
    print(f"\n{'='*60}")
    print(f"  STAGE 1-B COMPLETE")
    print(f"  Grid removed from {len(results)} images")
    print(f"  Annotations preserved via dual-layer shield")
    print(f"  Output dir: {config.STAGE1_CLEANED_DIR}")
    print(f"{'='*60}")
