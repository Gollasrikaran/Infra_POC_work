"""
config.py
=========
Central configuration for the Approach-1 Earthwork CV Pipeline.
All tunable parameters live here — no magic numbers in stage files.
"""

import os
from dataclasses import dataclass, field
from typing import List

# ── Resolve project root (this file lives in Approch-1/) ────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ──────────────────────────────────────────────────────────────────────────────
# Input
# ──────────────────────────────────────────────────────────────────────────────
PDF_PATH = os.path.join(PROJECT_ROOT, "bidx50197-1766424945654 (1).pdf")

# Drawing series to extract from the PDF.
# Add / remove series numbers here as needed.
TARGET_SERIES: List[str] = ["19", "23"]

# ──────────────────────────────────────────────────────────────────────────────
# Output directories
# ──────────────────────────────────────────────────────────────────────────────
OUTPUT_ROOT            = os.path.join(PROJECT_ROOT, "output")
STAGE1_EXTRACTED_DIR   = os.path.join(OUTPUT_ROOT, "stage1_extracted")   # raw cropped stations
STAGE1_CLEANED_DIR     = os.path.join(OUTPUT_ROOT, "stage1_cleaned")     # grid-removed stations
STAGE2_DIR             = os.path.join(OUTPUT_ROOT, "stage2")             # component analysis (CSV + debug boxes)
STAGE2_COLORED_DIR     = os.path.join(OUTPUT_ROOT, "stage2_colored")     # GREEN/RED zone fill visualization

# ──────────────────────────────────────────────────────────────────────────────
# PDF → Image render resolution
# ──────────────────────────────────────────────────────────────────────────────
PDF_RENDER_SCALE: float = 4.0          # PyMuPDF Matrix scale factor (4× ≈ 300 DPI)

# ──────────────────────────────────────────────────────────────────────────────
# PDF page-filter regions (as fraction of page width/height)
# ──────────────────────────────────────────────────────────────────────────────
TITLE_BLOCK_LEFT:   float = 0.65       # left edge of title-block search region
TITLE_BLOCK_TOP:    float = 0.75       # top edge of title-block search region
SCALE_AREA_TOP:     float = 0.70       # top edge of scale-label search region

# ──────────────────────────────────────────────────────────────────────────────
# Station-crop parameters
# ──────────────────────────────────────────────────────────────────────────────
STATION_PLUS_SEARCH_LEFT: float = 0.80  # left edge of "+" label search strip
CROP_TOP_MARGIN_PX:  int  = 35          # pixels above first station to start crop
CROP_BOTTOM_EXTRA:   int  = 48          # pixels below label bottom to extend crop
CROP_BETWEEN_GAP:    int  = 20          # gap guard between adjacent crops
CROP_PAGE_BOTTOM_FRACTION: float = 0.88 # never crop below this page fraction

# ──────────────────────────────────────────────────────────────────────────────
# Stage 1 – Grid removal
# ──────────────────────────────────────────────────────────────────────────────
# Projection-based approach (Code-3 method — most robust for faint scan grids)
GRID_BINARIZE_THRESH:       int   = 242   # pixels brighter than this → background
GRID_HORIZ_ROW_FRACTION:    float = 0.40  # ≥40% of row width active → horizontal grid line
GRID_VERT_COL_FRACTION:     float = 0.35  # ≥35% of col height active → vertical grid line
GRID_SHIELD_DILATE_ITER:    int   = 1     # dilation iterations for DIAGRAM shield (profile curves)

# Annotation preservation — Layer 2 of the dual-layer shield
# A connected component is treated as an annotation (text/label) when:
#   area  <=  ANNOT_MAX_AREA   AND   height  <=  ANNOT_MAX_H
# A padded bounding-box of size ANNOT_PAD pixels is then written into the
# annotation_shield mask, guaranteeing no grid eraser touches any label region.
#
# At 4× render scale (≈ 300 DPI):
#   A typical elevation label "123.45" is roughly 200 px wide × 35 px tall.
#   A single digit glyph is roughly 25 px wide × 40 px tall.
#   A slope label "1.5:1" cluster spans ≈ 120 × 40 px.
ANNOT_MAX_AREA:     int = 18_000   # max pixel area for a text/annotation component
ANNOT_MAX_H:        int = 120      # max bounding-box height for a text component (px)
ANNOT_PAD:          int = 8        # padding (px) added around each annotation bbox

# ──────────────────────────────────────────────────────────────────────────────
# Stage 2 – Connected Component Analysis  (debug bounding-box output)
# ──────────────────────────────────────────────────────────────────────────────
CC_BINARIZE_THRESH:     int   = 235       # threshold for binary CC input image
CC_CONNECTIVITY:        int   = 8         # 4 or 8 connectivity

# Classification thresholds
CC_NOISE_MAX_AREA:      int   = 20        # components smaller than this → NOISE
CC_TEXT_MAX_AREA:       int   = 2500      # area ceiling for TEXT label
CC_TEXT_MIN_ASPECT:     float = 0.3       # min w/h for a TEXT blob (wide short)
CC_TEXT_MAX_HEIGHT:     int   = 45        # text components shorter than this
CC_ARROW_MAX_DIAG:      float = 60.0      # diagonal ceiling for ARROW/LEADER
CC_BRIDGE_MIN_WIDTH:    int   = 120       # minimum width for BRIDGE_DECK
CC_BRIDGE_MIN_HEIGHT:   int   = 80        # minimum height for BRIDGE_DECK
CC_PROFILE_MIN_DIAG:    float = 80.0      # minimum diagonal for a PROFILE_CANDIDATE
CC_PROFILE_MIN_WIDTH_FRAC: float = 0.12   # component must span ≥12% of image width

# Bounding-box colors for Stage-2 annotated debug image (BGR)
CC_COLOR_MAP = {
    "NOISE":             (200, 200, 200),   # light grey
    "TEXT":              (255, 165,   0),   # orange
    "ARROW":             (255, 255,   0),   # yellow
    "BRIDGE_DECK":       (255,   0, 255),   # magenta
    "PROFILE_CANDIDATE": (  0, 255,   0),   # bright green
    "ANNOTATION":        (  0, 165, 255),   # orange-blue
}

# ──────────────────────────────────────────────────────────────────────────────
# Stage 2 – Profile Coloring  (GREEN / RED zone fill — the main visual output)
# ──────────────────────────────────────────────────────────────────────────────
# Connected component rules for separating the two engineering profiles:
#
#   PROPOSED GRADE  (solid continuous line)
#     → large components: diagonal > SOLID_MIN_DIAG  OR  w > SOLID_MIN_W  OR  h > SOLID_MIN_H
#     → explicitly exclude text-height components
#
#   EXISTING GROUND  (dashed/dotted line)
#     → small dash segments: DOT_W_MIN ≤ w ≤ DOT_W_MAX
#                            DOT_H_MIN ≤ h ≤ DOT_H_MAX
#                            aspect_ratio > DOT_MIN_ASPECT

# ── Working zone (strict exclusion of borders, scale bars, title block) ───────
# All fractions are relative to the image width or height.
# Components whose bounding box falls OUTSIDE this zone are completely discarded.
# This is the primary fix for false large rectangles from border/scale elements.
#
#   COLORING_LEFT_FRAC  : exclude the leftmost X%  (left border & elevation labels)
#   COLORING_RIGHT_FRAC : only process up to this X fraction  (exclude scale/station right strip)
#   COLORING_TOP_FRAC   : exclude the topmost Y%   (top frame border)
#   SCALE_STRIP_FRACTION: exclude below this Y%    (bottom scale bar strip)
COLORING_LEFT_FRAC:     float = 0.03   # exclude left 3%
COLORING_RIGHT_FRAC:    float = 0.87   # process only up to 87% width (exclude right 13%)
COLORING_TOP_FRAC:      float = 0.05   # exclude top 5%
SCALE_STRIP_FRACTION:   float = 0.88   # exclude bottom 12%  (scale bar)

# ── Proposed Grade (solid line) detection ────────────────────────────────────
SOLID_MIN_DIAG:         float = 55.0   # minimum diagonal for a solid-line segment
SOLID_MIN_W:            int   = 50     # OR minimum width
SOLID_MIN_H:            int   = 50     # OR minimum height
SOLID_TEXT_EXCL_H:      int   = 16     # exclude if h >= this AND w <= SOLID_TEXT_EXCL_W
SOLID_TEXT_EXCL_W:      int   = 45     # (catches tall-narrow text glyphs)

# ── Existing Ground (dashed line) detection ───────────────────────────────────
DOT_W_MIN:              int   = 3      # minimum dash width
DOT_W_MAX:              int   = 45     # maximum dash width
DOT_H_MIN:              int   = 2      # minimum dash height
DOT_H_MAX:              int   = 12     # maximum dash height
DOT_MIN_ASPECT:         float = 1.0    # dash must be wider than it is tall

# ── Legacy gap-fill (kept for reference, not used in v2 interpolation algo) ──
BORDER_MARGIN:          int   = 3
GAP_FILL_MAX:           int   = 60

# ── Ink restoration ───────────────────────────────────────────────────────────
# Any pixel darker than this in the cleaned grayscale image is restored to
# pure black on the output canvas.  This brings back both profile lines AND
# every annotation label (elevation values, slope ratios, station numbers, etc.)
INK_RESTORE_THRESH:     int   = 80

# ── Output colors (BGR) ───────────────────────────────────────────────────────
COLOR_FILL:             tuple = (0, 210,   0)   # GREEN — fill zone
COLOR_CUT:              tuple = (0,   0, 255)   # RED   — cut zone
COLOR_INK:              tuple = (0,   0,   0)   # BLACK — restored ink
