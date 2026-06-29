"""
stage2_profile_coloring.py
===========================
Stage 2 – Profile Zone Visualization  (GREEN / RED fill output)

Objective
---------
Produce the primary engineering visual:

    WHITE  background
    BLACK  profile lines + all annotations
    GREEN  zone where Proposed Grade is ABOVE Existing Ground  → FILL
    RED    zone where Proposed Grade is BELOW Existing Ground  → CUT

Algorithm  — Curve-Tracing + Interpolation  (v2 – fixed)
----------------------------------------------------------
The v1 approach used a raw column-by-column pixel scan which produced:
  ✗  Tiny isolated patches (only colored where BOTH profiles had pixels)
  ✗  Large false red/green rectangles (border/scale elements misclassified)
  ✗  Gaps across dash segments of the Existing Ground line

This version uses curve tracing + scipy linear interpolation:

  Step 1  Binarise cleaned image (grid already removed by Stage 1-B).

  Step 2  Strict exclusion zones:
            Left  : exclude leftmost  LEFT_FRAC  of image width
            Right : exclude rightmost (1 - RIGHT_FRAC) of image width
                    ← THIS eliminates scale bars, station labels, frame borders
            Top   : exclude topmost   TOP_FRAC   of image height
            Bottom: exclude below SCALE_STRIP_FRACTION (scale bar strip)

  Step 3  Connected component analysis inside the safe working zone.
          Classify each component as SOLID or DOTTED using geometry rules.
          Build solid_mask and dotted_mask.

  Step 4  Curve tracing — extract one Y value per X column:
            y_solid[x]  = mean row of solid pixels in column x  (Proposed Grade)
            y_dotted[x] = mean row of dotted pixels in column x (Existing Ground)
          NaN where no pixels found.

  Step 5  Scipy linear interpolation fills ALL gaps in both curves,
          producing two continuous arrays across the full working width.
          This is the key fix — no more fragmentation or missing sections.

  Step 6  Zone fill — for each column in the working zone:
            y_solid[x] < y_dotted[x]  → solid ABOVE dotted → FILL → GREEN
            y_solid[x] > y_dotted[x]  → solid BELOW dotted → CUT  → RED
          Paint directly column-by-column between the two interpolated Y values.

  Step 7  Restore ALL original black ink on top:
          img_gray < INK_THRESH  → force to black (profiles + annotations).

Output
------
output/stage2_colored/<name>_colored.png
"""

import os
import logging
from typing import List

import cv2
import numpy as np
from scipy.interpolate import interp1d

import config
from utils.io_utils import ensure_dirs, collect_images, load_gray, save_image

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _build_profile_masks(
    bw:      np.ndarray,
    labels:  np.ndarray,
    stats:   np.ndarray,
    nlabels: int,
    img_h:   int,
    img_w:   int,
    x_left:  int,
    x_right: int,
    y_top:   int,
    y_bot:   int,
) -> tuple:
    """
    Classify connected components into solid_mask (Proposed Grade) and
    dotted_mask (Existing Ground) inside the defined working rectangle.

    Components that touch or extend outside the working zone are discarded.
    This is the primary safeguard against border, scale-bar, and annotation
    elements being mistaken for engineering profiles.

    Parameters
    ----------
    bw           : Binary image (ink=255, background=0).
    labels       : CC label map from connectedComponentsWithStats.
    stats        : CC stats array.
    nlabels      : Number of labels.
    img_h, img_w : Image dimensions.
    x_left       : Left edge of working zone (px).
    x_right      : Right edge of working zone (px).
    y_top        : Top edge of working zone (px).
    y_bot        : Bottom edge of working zone (px).

    Returns
    -------
    (solid_mask, dotted_mask)  — uint8 binary arrays, same shape as bw.
    """
    solid_mask  = np.zeros_like(bw)
    dotted_mask = np.zeros_like(bw)

    for i in range(1, nlabels):
        cx = int(stats[i, cv2.CC_STAT_LEFT])
        cy = int(stats[i, cv2.CC_STAT_TOP])
        cw = int(stats[i, cv2.CC_STAT_WIDTH])
        ch = int(stats[i, cv2.CC_STAT_HEIGHT])

        # ── Strict zone containment check ─────────────────────────────────────
        # Discard any component whose bounding box goes OUTSIDE the working zone.
        # This is the fix for the large false rectangles caused by border elements.
        if cx < x_left:
            continue
        if (cx + cw) > x_right:
            continue
        if cy < y_top:
            continue
        if (cy + ch) > y_bot:
            continue

        # ── Text-glyph exclusion ──────────────────────────────────────────────
        # Tall-narrow components are numeric/text glyphs, not profile segments.
        if ch >= config.SOLID_TEXT_EXCL_H and cw <= config.SOLID_TEXT_EXCL_W:
            continue

        diag         = float(np.sqrt(cw ** 2 + ch ** 2))
        aspect_ratio = cw / ch if ch > 0 else 0.0

        # ── Existing Ground (dashed segments) ─────────────────────────────────
        if (
            config.DOT_W_MIN <= cw <= config.DOT_W_MAX
            and config.DOT_H_MIN <= ch <= config.DOT_H_MAX
            and aspect_ratio > config.DOT_MIN_ASPECT
        ):
            dotted_mask[labels == i] = 255
            continue

        # ── Proposed Grade (solid segments) ───────────────────────────────────
        if diag > config.SOLID_MIN_DIAG or cw > config.SOLID_MIN_W or ch > config.SOLID_MIN_H:
            solid_mask[labels == i] = 255

    return solid_mask, dotted_mask


def _trace_curve(mask: np.ndarray, x_left: int, x_right: int) -> np.ndarray:
    """
    Trace the Y position of a profile line at every X column.

    For each column in [x_left, x_right), compute the mean row index of all
    active pixels.  Returns a float array of length mask.shape[1] with NaN
    where the column had no active pixels.

    Parameters
    ----------
    mask   : Binary mask for one profile (solid or dotted).
    x_left : First column to scan.
    x_right: Last column to scan (exclusive).

    Returns
    -------
    y_curve : float64 array, shape (W,), NaN where no pixels found.
    """
    img_w  = mask.shape[1]
    y_curve = np.full(img_w, np.nan, dtype=np.float64)

    for col in range(x_left, x_right):
        rows = np.where(mask[:, col] == 255)[0]
        if len(rows) > 0:
            y_curve[col] = float(np.mean(rows))

    return y_curve


def _interpolate_curve(y_raw: np.ndarray, x_left: int, x_right: int) -> np.ndarray:
    """
    Fill NaN gaps in a Y-position curve using scipy linear interpolation.

    Interpolation is only performed within [x_left, x_right).
    Extrapolation beyond known data is filled with NaN (not extrapolated).

    Parameters
    ----------
    y_raw   : Raw Y-position array (with NaN gaps).
    x_left  : Start of working zone.
    x_right : End of working zone.

    Returns
    -------
    y_interp : Float array with gaps filled.  NaN outside known data range.
    """
    xs          = np.arange(len(y_raw))
    valid_idx   = np.where(~np.isnan(y_raw))[0]

    # Need at least 2 known points to interpolate
    if len(valid_idx) < 2:
        return y_raw.copy()

    # Clamp valid range to working zone
    valid_in_zone = valid_idx[(valid_idx >= x_left) & (valid_idx < x_right)]
    if len(valid_in_zone) < 2:
        return y_raw.copy()

    f = interp1d(
        valid_in_zone,
        y_raw[valid_in_zone],
        kind="linear",
        bounds_error=False,
        fill_value=np.nan,   # do NOT extrapolate beyond known data
    )

    y_interp = y_raw.copy()
    y_interp[x_left:x_right] = f(xs[x_left:x_right])
    return y_interp


# ──────────────────────────────────────────────────────────────────────────────
# Core algorithm
# ──────────────────────────────────────────────────────────────────────────────

def colorize_profiles(img_gray: np.ndarray) -> np.ndarray:
    """
    Produce the GREEN/RED zone-fill visualization for one cleaned image.

    Parameters
    ----------
    img_gray : Cleaned grayscale image from Stage 1-B.

    Returns
    -------
    BGR colour image  (WHITE bg, GREEN fill, RED cut, BLACK ink).
    """
    img_h, img_w = img_gray.shape

    # ══════════════════════════════════════════════════════════════════════════
    # Step 1  – Binarise
    # ══════════════════════════════════════════════════════════════════════════
    _, bw = cv2.threshold(img_gray, 235, 255, cv2.THRESH_BINARY_INV)

    # ══════════════════════════════════════════════════════════════════════════
    # Step 2  – Define strict working zone (exclude borders, scale bars, etc.)
    # ══════════════════════════════════════════════════════════════════════════
    x_left  = int(img_w * config.COLORING_LEFT_FRAC)    # e.g. leftmost 3%
    x_right = int(img_w * config.COLORING_RIGHT_FRAC)   # e.g. rightmost 13% excluded
    y_top   = int(img_h * config.COLORING_TOP_FRAC)     # e.g. topmost 5%
    y_bot   = int(img_h * config.SCALE_STRIP_FRACTION)  # e.g. bottom 12% excluded

    # ══════════════════════════════════════════════════════════════════════════
    # Step 3  – CC analysis & profile mask separation
    # ══════════════════════════════════════════════════════════════════════════
    nlabels, labels, stats, _ = cv2.connectedComponentsWithStats(
        bw, 8, cv2.CV_32S
    )

    solid_mask, dotted_mask = _build_profile_masks(
        bw, labels, stats, nlabels,
        img_h, img_w,
        x_left, x_right, y_top, y_bot,
    )

    # ══════════════════════════════════════════════════════════════════════════
    # Step 4  – Curve tracing: one Y value per column for each profile
    # ══════════════════════════════════════════════════════════════════════════
    y_solid_raw  = _trace_curve(solid_mask,  x_left, x_right)
    y_dotted_raw = _trace_curve(dotted_mask, x_left, x_right)

    # ══════════════════════════════════════════════════════════════════════════
    # Step 5  – Interpolation: fill all gaps → continuous curves
    # ══════════════════════════════════════════════════════════════════════════
    y_solid  = _interpolate_curve(y_solid_raw,  x_left, x_right)
    y_dotted = _interpolate_curve(y_dotted_raw, x_left, x_right)

    # Sanity-check: if either curve is entirely NaN, return original (no profiles found)
    n_solid_valid  = np.sum(~np.isnan(y_solid[x_left:x_right]))
    n_dotted_valid = np.sum(~np.isnan(y_dotted[x_left:x_right]))

    if n_solid_valid < 10 or n_dotted_valid < 10:
        logger.warning(
            "Insufficient profile data: solid=%d, dotted=%d valid cols. "
            "Returning original image.",
            n_solid_valid, n_dotted_valid,
        )
        return cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

    # ══════════════════════════════════════════════════════════════════════════
    # Step 6  – Zone fill  (column-by-column between the two interpolated curves)
    # ══════════════════════════════════════════════════════════════════════════
    canvas = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

    for col in range(x_left, x_right):
        ys = y_solid[col]
        yd = y_dotted[col]

        # Skip columns where either curve has no data (outside known range)
        if np.isnan(ys) or np.isnan(yd):
            continue

        ys_i = int(round(ys))
        yd_i = int(round(yd))

        # No zone if the two curves coincide
        if ys_i == yd_i:
            continue

        # Clamp to image bounds
        top_y = max(0,       min(ys_i, yd_i))
        bot_y = min(img_h-1, max(ys_i, yd_i))

        if bot_y <= top_y:
            continue

        if ys_i < yd_i:
            # Solid (Proposed Grade) is ABOVE Dotted (Existing Ground)
            # → We need to FILL up to reach Proposed Grade  →  GREEN
            canvas[top_y:bot_y, col] = list(config.COLOR_FILL)
        else:
            # Solid (Proposed Grade) is BELOW Dotted (Existing Ground)
            # → We need to CUT down to reach Proposed Grade  →  RED
            canvas[top_y:bot_y, col] = list(config.COLOR_CUT)

    # ══════════════════════════════════════════════════════════════════════════
    # Step 7  – Restore ALL original black ink on top
    #           (profile lines + every annotation — elevation labels, slope
    #            ratios, station numbers, leader lines, arrows, etc.)
    # ══════════════════════════════════════════════════════════════════════════
    ink_mask = img_gray < config.INK_RESTORE_THRESH
    canvas[ink_mask] = list(config.COLOR_INK)

    return canvas


# ──────────────────────────────────────────────────────────────────────────────
# Batch processor
# ──────────────────────────────────────────────────────────────────────────────

def process_all(
    input_dir:  str = config.STAGE1_CLEANED_DIR,
    output_dir: str = config.STAGE2_COLORED_DIR,
) -> List[str]:
    """
    Colorize all cleaned station images.

    Parameters
    ----------
    input_dir  : Cleaned images from Stage 1-B.
    output_dir : Destination for GREEN/RED zone-fill PNGs.

    Returns
    -------
    List of absolute paths to saved coloured images (sorted).
    """
    ensure_dirs(output_dir)

    image_paths = collect_images(input_dir)
    if not image_paths:
        logger.error("No images found in: %s", input_dir)
        return []

    logger.info(
        "Stage 2 Coloring: processing %d images from %s",
        len(image_paths), input_dir,
    )
    saved: List[str] = []

    for idx, img_path in enumerate(image_paths, start=1):
        img_gray = load_gray(img_path)
        if img_gray is None:
            continue

        try:
            colored = colorize_profiles(img_gray)
        except Exception as exc:
            logger.error(
                "  [%d/%d] Error on %s: %s",
                idx, len(image_paths), os.path.basename(img_path), exc,
            )
            continue

        base     = os.path.splitext(os.path.basename(img_path))[0]
        out_name = f"{base}_colored.png"
        out_path = os.path.join(output_dir, out_name)

        save_image(out_path, colored)
        saved.append(out_path)
        logger.info("  [%d/%d] Colored: %s", idx, len(image_paths), out_name)

    logger.info(
        "Stage 2 Coloring complete.  %d images saved to %s",
        len(saved), output_dir,
    )
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
    print(f"  STAGE 2 PROFILE COLORING COMPLETE")
    print(f"  {len(results)} colored images saved")
    print(f"  GREEN = Fill zone  |  RED = Cut zone")
    print(f"  Output dir: {config.STAGE2_COLORED_DIR}")
    print(f"{'='*60}")
