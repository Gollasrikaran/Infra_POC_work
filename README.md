# 🛣️ Highway Cross-Section Analyzer
### Approach 1 — Traditional Computer Vision Based Earthwork Calculation

**Module:** CV-Based Cross-Section Profile Extraction & Earthwork Estimation

---

An AI-powered and Computer Vision-assisted pipeline designed to analyze highway earthwork cross-section engineering drawings, identify **Existing Ground (EG)** and **Proposed Grade (PG)** profiles, classify **Cut / Fill** regions, and generate color-filled visual overlays — directly from raw engineering PDFs.

---

## 🎯 Purpose of the Module

In highway construction, engineers analyze transverse cross-sections to estimate the volume of earthwork (Cut & Fill) required along a road alignment. This module automates the visual inspection and verification process by:

- **PDF Ingestion & Station Extraction:** Parsing multi-series engineering PDFs (Series 19, Series 23) using PyMuPDF, identifying cross-section pages by title-block drawing numbers, and cropping one high-DPI image per station label — fully automated.

- **Grid Removal with Annotation Preservation:** Detecting and erasing background engineering grid lines using projection-based scanning while protecting all annotations (elevation labels, slope ratios, station numbers, leader lines, arrows) via a dual-layer pixel shield algorithm.

- **Profile Segmentation:** Distinguishing between the natural terrain (**Existing Ground** — dashed/dotted lines) and the design template (**Proposed Grade** — solid continuous lines) using connected component geometry analysis and curve-tracing.

- **Region Classification:** Segmenting the cross-section into:
  - 🟢 **FILL** — Proposed Grade lies above the Existing Ground (engineer must add fill material)
  - 🔴 **CUT** — Proposed Grade lies below the Existing Ground (engineer must remove earth)

- **Visual Validation Overlay:** Dynamically rendering a GREEN/RED color mask directly on the cleaned cross-section drawing to highlight earthwork zones for engineering review — with all original annotations and profile lines restored in black on top.

- **Local Pixel Quantification:** Utilizing OpenCV-based connected components, projection scanning, and scipy interpolation as the core algorithmic engine for profile coordinate extraction and local area measurement — the foundation for volumetric earthwork calculations.

---

## 👥 Team Contributions

| Member | Role | Contributions |
|---|---|---|
| **Muni Kumar Netlapalli** | Designed and implemented the full Approach 1 pipeline end-to-end — PDF extraction engine, projection-based grid removal with dual-layer annotation shield, CC-based profile analysis, curve-tracing with scipy interpolation, GREEN/RED zone fill visualization, pipeline orchestrator, centralised configuration, and full feasibility study |

---

## 📅 Project Status

### ✅ Completed

- **Stage 1-A — PDF Extraction (`stage1_pdf_extraction.py`)**
  - Parses all pages of the engineering PDF using PyMuPDF (`fitz`)
  - Identifies pages belonging to Series 19 and Series 23 by scanning the bottom-right title block text
  - Filters strictly to **Cross-Section** pages only (excludes plan views, profile sheets, details)
  - Locates station labels (`+` signs) in the right margin of each matching page
  - Crops one individual PNG per station at **4× render scale (≈ 300 DPI)**
  - Output: `output/stage1_extracted/Series19_Page12_Sta_10+00.png` etc.

- **Stage 1-B — Grid Removal (`stage1_image_prep.py`)**
  - Projection-based horizontal/vertical grid detection (row scan > 40% width threshold, column scan > 35% height threshold)
  - **Dual-Layer Annotation Shield:**
    - Layer 1 (Diagram Shield): dilates all non-grid ink to protect profile curves at grid intersections
    - Layer 2 (Annotation Shield): CC analysis identifies text-sized components; padded bounding boxes written into a protection mask for every elevation label, slope ratio, station number, arrow
  - Applies eraser only to grid pixels not covered by either shield — annotations survive 100% intact
  - Output: `output/stage1_cleaned/`

- **Stage 2 — Connected Component Analysis (`stage2_component_analysis.py`)**
  - `cv2.connectedComponentsWithStats()` with 8-connectivity on all cleaned images
  - Six-class rule-based classification: `PROFILE_CANDIDATE`, `BRIDGE_DECK`, `TEXT`, `ARROW`, `ANNOTATION`, `NOISE`
  - Colour-coded bounding-box debug PNG per image
  - Per-image component CSV + global `_ALL_components.csv` across all images
  - Output: `output/stage2/`

- **Stage 2C — Profile Zone Coloring — v2 (`stage2_profile_coloring.py`)**
  - Strict working-zone exclusion (left 3%, right 13%, top 5%, bottom 12%) to eliminate border/scale contamination
  - CC-based mask separation: `solid_mask` (Proposed Grade) and `dotted_mask` (Existing Ground)
  - Column-by-column curve tracing: `y_solid[x]` and `y_dotted[x]` as mean pixel row per column
  - `scipy.interpolate.interp1d` fills ALL NaN gaps → continuous curves across full image width
  - Column-by-column GREEN/RED zone paint between the two interpolated curves
  - Original black ink restored on top (profiles + all annotations)
  - Output: `output/stage2_colored/`

- **Pipeline Orchestrator (`pipeline.py`)**
  - Sequential or selective stage execution via `--stages` CLI argument
  - Timestamped log to `output/pipeline.log`
  - Custom PDF path via `--pdf` argument

- **Centralised Configuration (`config.py`)**
  - 30+ tunable parameters in one file — no magic numbers in any stage
  - Covers render scale, grid detection, annotation shield, CC classification rules, zone exclusion fractions, output colours

- **Utilities (`utils/io_utils.py`)**
  - `ensure_dirs()`, `collect_images()`, `load_gray()`, `load_bgr()`, `save_image()`

- **Feasibility Analysis**
  - Root-cause analysis identifying 5 specific failure points in pure rule-based CV
  - Technical evaluation of 4 alternative approaches (Hough, Template Matching, Manual Seed + BFS, U-Net)
  - Recommended two-phase hybrid strategy documented

---

### 🚧 In Progress

- **Manual Seed + BFS Traversal (`stage2_seed_traversal.py`) — Designed, not yet implemented**

  The rule-based CC geometry approach (size, aspect ratio, diagonal) cannot reliably distinguish profile line segments from visually similar annotation elements (bridge edges, slope bars). The next stage replaces classification with traversal:
  - Engineer clicks **one pixel** on each profile line per image (Tkinter window)
  - **BFS flood fill** traces the full connected profile from that confirmed seed pixel
  - Completely eliminates the classification problem — you never classify, you just follow the ink
  - Expected reliability: **100%** | Effort: ~5 seconds per image

- **U-Net Semantic Segmentation — Architecture decided, training not started**

  Phase 2 after BFS:
  - BFS outputs auto-generate the pixel-level training labels (no manual annotation needed)
  - U-Net trained on 50 cross-section images → per-pixel class map (Background / EG / PG)
  - Expected inference: < 0.1 sec/image with GPU
  - Fully automated — zero human input per image once trained

- **Adaptive Border & Margin Alignment**

  Automatic detection of the drawing content boundary to precisely align the working zone with the physical drawing area, excluding title blocks and margin content dynamically rather than using fixed fractions.

- **Hybrid Pipeline Integration**

  Merging the CV-based grid removal and profile tracing with LLM-based semantic classification to handle edge cases (bridge deck identification, annotation disambiguation) that pure geometry rules cannot solve reliably.

---

## 🗂️ Project Structure

```
Approch-1/
│
├── pipeline.py                    ← Orchestrator — run all stages or selective stages
├── config.py                      ← All tunable parameters (single source of truth)
│
├── stage1_pdf_extraction.py       ← Stage 1-A : PDF → cropped station PNGs
├── stage1_image_prep.py           ← Stage 1-B : Grid removal + dual-layer annotation shield
│
├── stage2_component_analysis.py   ← Stage 2   : CC analysis debug view (CSV + bounding boxes)
├── stage2_profile_coloring.py     ← Stage 2C  : GREEN/RED zone fill (★ main visual output)
│
├── utils/
│   ├── __init__.py
│   └── io_utils.py                ← File I/O helpers (load, save, collect images)
│
├── requirements.txt               ← Python package dependencies
├── README.md                      ← This file
├── grid_removal.py                ← Legacy shim (superseded — do not use directly)
│
└── output/                        ← Auto-created at runtime (gitignored)
    ├── stage1_extracted/          ← Raw station crops — one PNG per station
    ├── stage1_cleaned/            ← Grid-removed images, annotations intact
    ├── stage2/                    ← Debug: annotated PNGs + component CSVs
    ├── stage2_colored/            ← ★ Main output: GREEN=Fill / RED=Cut zone images
    └── pipeline.log               ← Timestamped execution log
```

---

## ⚙️ Dependencies & Setup Instructions

### System Requirements

- **Python:** 3.9 or higher
- **OS:** Windows 10/11 (primary dev), macOS, Linux compatible

### Python Dependencies

```
PyMuPDF>=1.23.0        # PDF parsing and high-DPI image rendering
opencv-python>=4.8.0   # Connected components, morphological ops, binarization
numpy>=1.24.0          # Array operations, projection scanning, pixel masks
scipy>=1.11.0          # Linear interpolation (interp1d) for curve gap-filling
pandas>=2.0.0          # Component statistics CSV export
matplotlib>=3.7.0      # Debug visualization utilities
Pillow>=10.0.0         # Image manipulation and overlay blending
```

### Installation & Run Steps

**Step 1 — Clone the repository and checkout the branch**

```bash
git clone <repository-url>
cd Approch-1
```

**Step 2 — Create and activate a virtual environment**

```powershell
# Windows
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# macOS / Linux
python -m venv .venv
source .venv/bin/activate
```

**Step 3 — Install dependencies**

```bash
pip install -r requirements.txt
```

**Step 4 — Place the engineering PDF**

Place the PDF in the project root:
```
Approch-1/
└── bidx50197-1766424945654 (1).pdf   ← place here
```

The path is configured in `config.py`:
```python
PDF_PATH = os.path.join(PROJECT_ROOT, "bidx50197-1766424945654 (1).pdf")
```

To use a different PDF at runtime:
```bash
python pipeline.py --pdf "C:\path\to\your\drawing.pdf"
```

---

## ▶️ How to Run

### Run the Full Pipeline

```bash
python pipeline.py
```

### Run Stage by Stage (Recommended for First Use)

```bash
# Stage 1-A: Extract station images from PDF
python stage1_pdf_extraction.py
# → output/stage1_extracted/   (one PNG per station)

# Stage 1-B: Remove grid lines
python stage1_image_prep.py
# → output/stage1_cleaned/     (grid gone, annotations preserved)

# Stage 2: Component analysis debug view
python stage2_component_analysis.py
# → output/stage2/             (bounding boxes + CSVs)

# Stage 2C: GREEN/RED zone fill  ★ Main output
python stage2_profile_coloring.py
# → output/stage2_colored/     (GREEN=Fill, RED=Cut)
```

### Run Specific Stages Only

```bash
python pipeline.py --stages 1a 1b        # Extract + grid removal only
python pipeline.py --stages 2c           # Coloring only (Stage 1 already done)
python pipeline.py --stages 1a 1b 2c     # Full pipeline, skip debug CC view
```

---

## 📊 Feasibility Report — Rule-Based CV vs. Hybrid AI Approach

### Background

The initial design goal relied on geometry-based Connected Component Analysis (CC) to classify every ink blob in a cleaned drawing as either a profile line or an annotation, then separate the two engineering profiles (Existing Ground and Proposed Grade) for zone filling. During implementation and evaluation, a technical feasibility assessment was conducted comparing this approach against alternative methods.

---

### Key Findings

#### Finding 1 — Geometry Rules Cannot Reliably Separate Profiles from Annotations (Low Feasibility)

**Problem:** CC geometry rules (size, aspect ratio, diagonal length) operate on visual measurements only. Multiple object types in engineering drawings share identical geometric properties:

| Object | Width | Height | Diagonal | Misclassified As |
|---|---|---|---|---|
| Proposed Grade segment | 80–400px | 3–8px | 80–400px | ✅ SOLID (correct) |
| Bridge deck edge | 200–600px | 2–5px | 200–600px | ❌ SOLID (false positive) |
| Existing Ground dash | 10–40px | 2–5px | 10–40px | ✅ DOTTED (correct) |
| Slope annotation bar | 10–45px | 2–4px | 10–45px | ❌ DOTTED (false positive) |

**Impact:** False positives contaminate the profile masks. Interpolation then connects wrong anchor points, producing fill zones that extend into incorrect regions of the drawing.

---

#### Finding 2 — Grid Removal Can Weaken Profile Data (Medium Feasibility)

**Problem:** The projection scanner marks any row where >40% of pixels are active as a grid row. When the Existing Ground dashed profile is dense, it can activate 40–50% of a row by itself — causing parts of the profile to be misidentified as grid and partially erased.

**Impact:** Profile lines have additional gaps after grid removal, increasing fragmentation and making CC classification even harder.


