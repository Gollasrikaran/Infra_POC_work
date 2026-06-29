"""
stage1_pdf_extraction.py
========================
Stage 1 – Part A: PDF → Individual Station Cross-Section Images

Responsibilities
----------------
1.  Open the engineering PDF.
2.  Scan every page and keep only pages belonging to the target drawing series
    (e.g. 19-series, 23-series) that are also labelled as Cross-Sections.
3.  On each qualifying page, locate every station label ("+" signs) and crop
    one individual cross-section image per station.
4.  Save all cropped images to  output/stage1_extracted/

Algorithm (derived from and improving on the original Colab snippets)
----------------------------------------------------------------------
- Drawing-number detection: bottom-right corner text, cleaned of "DRAWING /
  NO. / DRG" noise.  Must start with one of TARGET_SERIES values and must NOT
  contain "+" (which would make it a station number, not a drawing number).
- Station detection: search for "+" characters inside the right 20% of the
  page, sorted top-to-bottom.
- Cropping: cut from just above one "+" label to just above the next.
- Render: PyMuPDF Matrix(scale, scale) for high-resolution output.

No validation reports are generated — manual inspection of the output folder
is sufficient.
"""

import os
import re
import logging
from typing import List, Tuple

import fitz  # PyMuPDF

import config
from utils.io_utils import ensure_dirs

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _is_target_series(corner_text: str, series_list: List[str]) -> bool:
    """
    Return True if any line in *corner_text* is a drawing number that
    starts with one of the target series identifiers.

    Rules
    -----
    - Strip noise words: DRAWING, NO., DRG, colons, whitespace.
    - The cleaned token must START WITH the series string.
    - It must NOT contain "+" (station numbers look like "10+00").
    """
    for line in corner_text.split("\n"):
        clean = re.sub(r"(?i)DRAWING|NO\.?|DRG|[:\s]", "", line).strip()
        if not clean or "+" in clean:
            continue
        for series in series_list:
            # Matches "19", "19-001", "23", "23-004", etc.
            if clean.startswith(series + "-") or clean == series or re.match(rf"^{re.escape(series)}\d", clean):
                return True
    return False


def _is_cross_section(text: str) -> bool:
    """Return True if the text contains a 'CROSS SECTION' (or 'CROSS-SECTION') label."""
    return bool(re.search(r"(?i)cross[- \s]*section", text))


def _clean_station_name(raw: str) -> str:
    """Keep only digits and '+' from a station label string."""
    clean = re.sub(r"[^0-9+]", "", raw)
    return clean if clean else "unk"


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def extract_stations_from_pdf(
    pdf_path: str = config.PDF_PATH,
    output_dir: str = config.STAGE1_EXTRACTED_DIR,
    series: List[str] = None,
) -> List[str]:
    """
    Extract individual station cross-section images from *pdf_path*.

    Parameters
    ----------
    pdf_path   : Path to the engineering PDF.
    output_dir : Folder where cropped station PNGs are saved.
    series     : List of drawing-series prefixes to keep (default: config.TARGET_SERIES).

    Returns
    -------
    List of absolute paths to every saved station image (sorted).
    """
    if series is None:
        series = config.TARGET_SERIES

    ensure_dirs(output_dir)

    if not os.path.exists(pdf_path):
        logger.error("PDF not found: %s", pdf_path)
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    logger.info("Opened PDF: %s  (%d pages)", pdf_path, total_pages)

    saved_paths: List[str] = []
    pages_matched = 0

    for page_idx in range(total_pages):
        page = doc[page_idx]
        rect = page.rect
        pw, ph = rect.width, rect.height

        # ── 1. Title-block filter ────────────────────────────────────────────
        title_rect = fitz.Rect(
            pw * config.TITLE_BLOCK_LEFT,
            ph * config.TITLE_BLOCK_TOP,
            pw,
            ph,
        )
        corner_text = page.get_text("text", clip=title_rect).strip()

        if not _is_target_series(corner_text, series):
            continue
        if not _is_cross_section(corner_text):
            continue

        pages_matched += 1
        logger.info(
            "  Page %d matches series %s (cross-section)", page_idx + 1, series
        )

        # ── 2. Determine which series this page belongs to ───────────────────
        matched_series = "XX"
        for line in corner_text.split("\n"):
            clean = re.sub(r"(?i)DRAWING|NO\.?|DRG|[:\s]", "", line).strip()
            if "+" in clean:
                continue
            for s in series:
                if clean.startswith(s):
                    matched_series = s
                    break

        # ── 3. Find station labels ("+") on the right strip of the page ──────
        plus_strip = fitz.Rect(pw * config.STATION_PLUS_SEARCH_LEFT, 0, pw, ph)
        station_hits = page.search_for("+", clip=plus_strip)
        station_hits.sort(key=lambda r: r.y0)

        if not station_hits:
            logger.warning("  Page %d: no station labels found, skipping.", page_idx + 1)
            continue

        logger.info("  Page %d: found %d stations", page_idx + 1, len(station_hits))

        # ── 4. Crop one image per station ────────────────────────────────────
        last_bottom = config.CROP_TOP_MARGIN_PX

        for j, hit in enumerate(station_hits):
            y_top = last_bottom
            y_bottom_target = hit.y1 + config.CROP_BOTTOM_EXTRA

            if j + 1 < len(station_hits):
                y_bottom = min(y_bottom_target, station_hits[j + 1].y0 - config.CROP_BETWEEN_GAP)
            else:
                y_bottom = min(y_bottom_target, ph * config.CROP_PAGE_BOTTOM_FRACTION)

            last_bottom = y_bottom

            # Sanity-check: skip degenerate crops
            if y_bottom - y_top < 30:
                logger.debug("  Skipping degenerate crop at station index %d (height < 30px)", j)
                continue

            crop_rect = fitz.Rect(0, y_top, pw, y_bottom)

            # Read the station name from a tight region around the "+" label
            label_rect = fitz.Rect(
                pw * config.STATION_PLUS_SEARCH_LEFT,
                hit.y0 - 30,
                pw,
                hit.y1 + 30,
            )
            sta_text = page.get_text("text", clip=label_rect).strip()
            sta_name = _clean_station_name(sta_text)

            # ── Render at high resolution ────────────────────────────────────
            mat = fitz.Matrix(config.PDF_RENDER_SCALE, config.PDF_RENDER_SCALE)
            pix = page.get_pixmap(matrix=mat, clip=crop_rect)

            filename = f"Series{matched_series}_Page{page_idx + 1}_Sta_{sta_name}.png"
            out_path = os.path.join(output_dir, filename)
            pix.save(out_path)
            saved_paths.append(out_path)
            logger.info("    Saved: %s", filename)

    doc.close()

    logger.info(
        "Extraction complete. Pages matched: %d | Stations saved: %d",
        pages_matched,
        len(saved_paths),
    )

    if pages_matched == 0:
        logger.warning(
            "No pages matched series %s. Check title-block layout and TARGET_SERIES in config.py.",
            series,
        )

    return sorted(saved_paths)


# ──────────────────────────────────────────────────────────────────────────────
# Standalone entry point
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
        datefmt="%H:%M:%S",
    )
    paths = extract_stations_from_pdf()
    print(f"\n{'='*60}")
    print(f"  STAGE 1-A COMPLETE")
    print(f"  Extracted {len(paths)} station images")
    print(f"  Output dir: {config.STAGE1_EXTRACTED_DIR}")
    print(f"{'='*60}")
