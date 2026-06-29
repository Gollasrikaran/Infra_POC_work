"""
pipeline.py
===========
Approach-1 Earthwork CV Pipeline — Orchestrator

Runs the complete pipeline in sequence:

    Stage 1-A  PDF → Individual station cross-section images
    Stage 1-B  Grid removal & image standardisation
    Stage 2    Connected component analysis & profile candidate identification

Usage
-----
    # Run the full pipeline (default: process the PDF defined in config.py)
    python pipeline.py

    # Run only specific stages
    python pipeline.py --stages 1a 1b
    python pipeline.py --stages 2

    # Use a custom PDF path
    python pipeline.py --pdf "C:/path/to/your/drawing.pdf"

Output Structure
----------------
output/
├── stage1_extracted/    Raw cropped station images (from Stage 1-A)
├── stage1_cleaned/      Grid-removed station images (from Stage 1-B)
└── stage2/              Annotated PNGs + CSVs (from Stage 2)
"""

import argparse
import logging
import os
import sys
import time

import config

# ──────────────────────────────────────────────────────────────────────────────
# Logging setup
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            os.path.join(config.OUTPUT_ROOT, "pipeline.log"),
            mode="a",
            encoding="utf-8",
        ),
    ],
)
logger = logging.getLogger("pipeline")


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _banner(title: str) -> None:
    bar = "=" * 62
    logger.info(bar)
    logger.info("  %s", title)
    logger.info(bar)


def _elapsed(start: float) -> str:
    s = time.time() - start
    return f"{s:.1f}s"


# ──────────────────────────────────────────────────────────────────────────────
# Stage runners
# ──────────────────────────────────────────────────────────────────────────────

def run_stage_1a(pdf_path: str) -> int:
    """Stage 1-A: Extract station images from PDF."""
    from stage1_pdf_extraction import extract_stations_from_pdf
    _banner("STAGE 1-A  –  PDF Extraction")
    t = time.time()
    paths = extract_stations_from_pdf(pdf_path=pdf_path)
    logger.info("Stage 1-A done in %s  |  %d images extracted", _elapsed(t), len(paths))
    return len(paths)


def run_stage_1b() -> int:
    """Stage 1-B: Grid removal on extracted station images."""
    from stage1_image_prep import process_all
    _banner("STAGE 1-B  –  Grid Removal")
    t = time.time()
    paths = process_all()
    logger.info("Stage 1-B done in %s  |  %d images cleaned", _elapsed(t), len(paths))
    return len(paths)


def run_stage_2() -> int:
    """Stage 2: Connected component analysis (debug bounding-box view)."""
    from stage2_component_analysis import process_all
    _banner("STAGE 2  –  Component Analysis (Debug View)")
    t = time.time()
    paths = process_all()
    logger.info("Stage 2 done in %s  |  %d annotated images", _elapsed(t), len(paths))
    return len(paths)


def run_stage_2c() -> int:
    """Stage 2C: Profile coloring — GREEN/RED zone fill (primary visual output)."""
    from stage2_profile_coloring import process_all
    _banner("STAGE 2C  –  Profile Zone Coloring  (GREEN=Fill / RED=Cut)")
    t = time.time()
    paths = process_all()
    logger.info("Stage 2C done in %s  |  %d colored images", _elapsed(t), len(paths))
    return len(paths)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Approach-1 Earthwork CV Pipeline"
    )
    parser.add_argument(
        "--pdf",
        default=config.PDF_PATH,
        help=f"Path to the engineering PDF (default: {config.PDF_PATH})",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        default=["1a", "1b", "2", "2c"],
        choices=["1a", "1b", "2", "2c"],
        help="Which stages to run (default: 1a 1b 2 2c)",
    )
    args = parser.parse_args()

    # Ensure top-level output dir exists (log file needs it)
    os.makedirs(config.OUTPUT_ROOT, exist_ok=True)

    _banner("APPROACH-1 EARTHWORK CV PIPELINE")
    logger.info("PDF        : %s", args.pdf)
    logger.info("Series     : %s", config.TARGET_SERIES)
    logger.info("Stages     : %s", args.stages)
    logger.info("Output root: %s", config.OUTPUT_ROOT)

    pipeline_start = time.time()

    if "1a" in args.stages:
        run_stage_1a(args.pdf)

    if "1b" in args.stages:
        run_stage_1b()

    if "2" in args.stages:
        run_stage_2()

    if "2c" in args.stages:
        run_stage_2c()

    _banner("PIPELINE COMPLETE")
    logger.info("Total wall-clock time: %s", _elapsed(pipeline_start))
    logger.info("")
    logger.info("  Stage 1-A  →  %s", config.STAGE1_EXTRACTED_DIR)
    logger.info("  Stage 1-B  →  %s", config.STAGE1_CLEANED_DIR)
    logger.info("  Stage 2    →  %s  (debug bounding boxes)", config.STAGE2_DIR)
    logger.info("  Stage 2C   →  %s  (GREEN=Fill / RED=Cut)", config.STAGE2_COLORED_DIR)
    logger.info("  Log file   →  %s", os.path.join(config.OUTPUT_ROOT, "pipeline.log"))


if __name__ == "__main__":
    main()
