"""
stage2_component_analysis.py
=============================
Stage 2 – Existing Ground & Proposed Grade Identification
via Connected Component Analysis

Objective
---------
Analyse every cleaned cross-section image (output of stage1_image_prep.py) and
produce:
  1. A colour-annotated debug image with every connected component boxed and
     labelled by its classification.
  2. A CSV file containing the full geometric statistics of every component.

Why This Stage Exists
---------------------
Earthwork calculations are not performed on pixels — they are performed on the
geometry of two engineering profiles:

    Existing Ground   (typically a dashed line in these drawings)
    Proposed Grade    (typically a solid, continuous line)

Before any cut/fill calculation can occur the system must determine WHICH
connected components in the cleaned image represent these two profiles.

This stage performs:
  - Connected component extraction using cv2.connectedComponentsWithStats()
  - Rule-based classification of every component into one of:
      NOISE             – too small to matter
      TEXT              – numeric/text annotation blobs
      ARROW             – small diagonal short elements (leaders, arrows)
      BRIDGE_DECK       – wide & tall solid rectangle regions
      PROFILE_CANDIDATE – long, wide-spanning elements (likely a profile line)
      ANNOTATION        – everything else (slope labels, leader lines, etc.)

Classification Rules
--------------------
Rules are driven entirely by config.py — no magic numbers here.

Output
------
output/stage2/<name>_annotated.png   — debug image with colour-coded bounding boxes
output/stage2/<name>_components.csv  — full component table

Component Table Columns
-----------------------
id, x, y, w, h, area, aspect_ratio, diagonal, centroid_x, centroid_y,
img_width, img_height, label
"""

import os
import logging
from typing import List, Dict, Tuple

import cv2
import numpy as np
import pandas as pd

import config
from utils.io_utils import ensure_dirs, collect_images, load_gray, save_image

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Classification logic
# ──────────────────────────────────────────────────────────────────────────────

def classify_component(
    x: int, y: int, w: int, h: int, area: int,
    img_w: int, img_h: int,
) -> str:
    """
    Assign a semantic label to a single connected component.

    Parameters
    ----------
    x, y, w, h : Bounding box (top-left corner + dimensions) in pixels.
    area        : Component pixel count.
    img_w, img_h: Image dimensions for relative-size calculations.

    Returns
    -------
    One of: NOISE | TEXT | ARROW | BRIDGE_DECK | PROFILE_CANDIDATE | ANNOTATION
    """
    aspect_ratio = w / h if h > 0 else 0.0
    diagonal     = float(np.sqrt(w ** 2 + h ** 2))

    # ── NOISE ────────────────────────────────────────────────────────────────
    if area < config.CC_NOISE_MAX_AREA:
        return "NOISE"

    # ── BRIDGE_DECK ──────────────────────────────────────────────────────────
    # Wide AND tall rectangular solid regions (platform slabs, bridge decks)
    if w >= config.CC_BRIDGE_MIN_WIDTH and h >= config.CC_BRIDGE_MIN_HEIGHT:
        return "BRIDGE_DECK"

    # ── TEXT ─────────────────────────────────────────────────────────────────
    # Small area + not taller than text_max_height  →  annotation text blobs
    if (
        area <= config.CC_TEXT_MAX_AREA
        and h <= config.CC_TEXT_MAX_HEIGHT
        and diagonal < config.CC_ARROW_MAX_DIAG
    ):
        return "TEXT"

    # ── ARROW / LEADER LINE ──────────────────────────────────────────────────
    # Short diagonal elements — slope tick marks, arrowheads
    if diagonal < config.CC_ARROW_MAX_DIAG and area < config.CC_TEXT_MAX_AREA:
        return "ARROW"

    # ── PROFILE_CANDIDATE ────────────────────────────────────────────────────
    # Long elements that span a significant fraction of the image width.
    # These are the most likely candidates for Existing Ground or Proposed Grade.
    if (
        diagonal >= config.CC_PROFILE_MIN_DIAG
        and w >= img_w * config.CC_PROFILE_MIN_WIDTH_FRAC
    ):
        return "PROFILE_CANDIDATE"

    # ── ANNOTATION ───────────────────────────────────────────────────────────
    # Anything that doesn't fit the above — slope labels, small dimension lines
    return "ANNOTATION"


# ──────────────────────────────────────────────────────────────────────────────
# Per-image analysis
# ──────────────────────────────────────────────────────────────────────────────

def analyse_image(img_gray: np.ndarray) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Run connected component analysis on a cleaned grayscale image.

    Parameters
    ----------
    img_gray : Cleaned grayscale image from Stage 1-B.

    Returns
    -------
    annotated_bgr : BGR debug image with colour-coded bounding boxes.
    df            : DataFrame containing one row per component.
    """
    img_h, img_w = img_gray.shape

    # ── Binarise ─────────────────────────────────────────────────────────────
    # Invert: dark drawing ink → white (255), light background → black (0)
    _, bw = cv2.threshold(
        img_gray,
        config.CC_BINARIZE_THRESH,
        255,
        cv2.THRESH_BINARY_INV,
    )

    # ── Connected components ──────────────────────────────────────────────────
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        bw, config.CC_CONNECTIVITY, cv2.CV_32S
    )

    # ── Build output canvas ───────────────────────────────────────────────────
    annotated = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

    # ── Collect records ───────────────────────────────────────────────────────
    records: List[Dict] = []

    for i in range(1, nlabels):   # skip label 0 (background)
        x, y, w, h, area = (
            int(stats[i, cv2.CC_STAT_LEFT]),
            int(stats[i, cv2.CC_STAT_TOP]),
            int(stats[i, cv2.CC_STAT_WIDTH]),
            int(stats[i, cv2.CC_STAT_HEIGHT]),
            int(stats[i, cv2.CC_STAT_AREA]),
        )
        cx, cy = float(centroids[i, 0]), float(centroids[i, 1])
        aspect_ratio = round(w / h, 4) if h > 0 else 0.0
        diagonal     = round(float(np.sqrt(w ** 2 + h ** 2)), 2)

        label = classify_component(x, y, w, h, area, img_w, img_h)

        records.append({
            "id":           i,
            "x":            x,
            "y":            y,
            "w":            w,
            "h":            h,
            "area":         area,
            "aspect_ratio": aspect_ratio,
            "diagonal":     diagonal,
            "centroid_x":   round(cx, 2),
            "centroid_y":   round(cy, 2),
            "img_width":    img_w,
            "img_height":   img_h,
            "label":        label,
        })

        # ── Draw bounding box on annotated image ─────────────────────────────
        color = config.CC_COLOR_MAP.get(label, (128, 128, 128))
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)

        # Label text — only for non-noise to keep the image readable
        if label != "NOISE":
            # Abbreviate label for compactness
            abbr = {
                "TEXT":              "TXT",
                "ARROW":             "ARR",
                "BRIDGE_DECK":       "BRG",
                "PROFILE_CANDIDATE": "PRF",
                "ANNOTATION":        "ANN",
            }.get(label, label[:3])

            cv2.putText(
                annotated,
                abbr,
                (x, max(0, y - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )

    df = pd.DataFrame(records)
    return annotated, df


# ──────────────────────────────────────────────────────────────────────────────
# Batch processor
# ──────────────────────────────────────────────────────────────────────────────

def process_all(
    input_dir:  str = config.STAGE1_CLEANED_DIR,
    output_dir: str = config.STAGE2_DIR,
) -> List[str]:
    """
    Run Stage-2 component analysis on every image in *input_dir*.

    Parameters
    ----------
    input_dir  : Folder of cleaned images (Stage-1B output).
    output_dir : Destination for annotated PNGs and CSVs.

    Returns
    -------
    List of paths to saved annotated images.
    """
    ensure_dirs(output_dir)

    image_paths = collect_images(input_dir)
    if not image_paths:
        logger.error("No images found in: %s", input_dir)
        return []

    logger.info(
        "Stage 2: analysing %d images from %s", len(image_paths), input_dir
    )

    saved_annotated: List[str] = []
    all_dfs: List[pd.DataFrame] = []

    for idx, img_path in enumerate(image_paths, start=1):
        img_gray = load_gray(img_path)
        if img_gray is None:
            continue

        try:
            annotated, df = analyse_image(img_gray)
        except Exception as exc:
            logger.error(
                "  [%d/%d] Error on %s: %s",
                idx, len(image_paths), os.path.basename(img_path), exc,
            )
            continue

        base = os.path.splitext(os.path.basename(img_path))[0]

        # ── Save annotated image ─────────────────────────────────────────────
        ann_path = os.path.join(output_dir, f"{base}_annotated.png")
        save_image(ann_path, annotated)
        saved_annotated.append(ann_path)

        # ── Save per-image CSV ───────────────────────────────────────────────
        csv_path = os.path.join(output_dir, f"{base}_components.csv")
        df.to_csv(csv_path, index=False)

        # ── Console summary ──────────────────────────────────────────────────
        if not df.empty:
            counts = df["label"].value_counts().to_dict()
            n_profiles = counts.get("PROFILE_CANDIDATE", 0)
            logger.info(
                "  [%d/%d] %s → %d components  |  PROFILE_CANDIDATES: %d  |  %s",
                idx, len(image_paths), os.path.basename(img_path),
                len(df), n_profiles, counts,
            )
        else:
            logger.warning("  [%d/%d] %s → 0 components found", idx, len(image_paths), os.path.basename(img_path))

        # ── Accumulate for global CSV ─────────────────────────────────────────
        df["source_image"] = os.path.basename(img_path)
        all_dfs.append(df)

    # ── Save global summary CSV ───────────────────────────────────────────────
    if all_dfs:
        global_df = pd.concat(all_dfs, ignore_index=True)
        global_csv = os.path.join(output_dir, "_ALL_components.csv")
        global_df.to_csv(global_csv, index=False)
        logger.info("Global component table saved: %s", global_csv)

    logger.info("Stage 2 complete. %d annotated images saved.", len(saved_annotated))
    return sorted(saved_annotated)


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
    print(f"  STAGE 2 COMPLETE")
    print(f"  {len(results)} annotated images saved")
    print(f"  Output dir: {config.STAGE2_DIR}")
    print(f"  Global CSV: {os.path.join(config.STAGE2_DIR, '_ALL_components.csv')}")
    print(f"{'='*60}")
