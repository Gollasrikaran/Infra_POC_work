# XDOT Contractor — Road Quantity Analyzer (Approach 2)

## Module Purpose

This module implements **Approach 2: Direct PDF Vector Extraction** for highway cross-section analysis.

The system processes engineering PDF drawings and automatically:

* Detects cross-section pages
* Extracts vector geometry
* Identifies Existing Ground (EG) and Proposed Grade (PG) profiles
* Converts PDF coordinates into engineering coordinates
* Computes cut/fill areas
* Estimates earthwork volumes using Average End Area method

---

## Team Contributions

### Ashrith Bejjarapu

Worked on:

* Approach 2 vector extraction pipeline
* Profile identification logic
* Cross-verification overlay generation
* Diagnostic enhancements for failed stations
* Existing Ground recovery enhancement analysis
* Validation and debugging workflow

---

## Completed Work

### 1. PDF Classification

Implemented page classification:

* Vector pages
* Hybrid pages
* Raster pages
* Cross-section page detection

### 2. Geometry Extraction

Extracted vector paths from PDF using PyMuPDF:

* lines
* curves
* rectangles
* dashed segments

### 3. Profile Detection

Implemented profile identification pipeline:

* Grid filtering
* Path merging
* Noise filtering
* Candidate scoring
* EG / PG classification

### 4. Coordinate Transformation

Converted PDF coordinates into engineering units:

* Horizontal offset
* Elevation

### 5. Earthwork Computation

Implemented:

* Cut area calculation
* Fill area calculation
* Volume estimation using Average End Area

### 6. Visualization

Generated overlays for manual validation:

* Existing Ground
* Proposed Grade
* Diagnostic plots

---

## Current Work In Progress

### Enhancement 1 — Recover Missing Existing Ground Line

Problem:
3 stations fail because the dashed Existing Ground line is fragmented and gets dropped before scoring.

Current investigation:

* Merge tolerance tuning
* Coverage threshold tuning
* Overlay-based failure tracing

Potential fixes:

* Increase merge tolerance for dashed segments
* Reduce minimum coverage threshold
* Improve dashed line recovery logic

---

## Dependencies

Install requirements:

```bash
pip install -r requirements.txt
```

Main libraries:

* PyMuPDF
* OpenCV
* NumPy
* SciPy
* Matplotlib
* Streamlit
* Pandas

---

## Setup Instructions

Run Streamlit app:

```bash
streamlit run app.py
```

Run verification tool:

```bash
python cross_verify.py <pdf_path>
```

---

## Feasibility Report

### Approach 2 Feasibility

Approach 2 is feasible because it directly extracts vector geometry from engineering PDFs instead of relying on image processing.

Advantages:

* Higher accuracy
* No OCR dependency
* Better scalability
* More reliable profile extraction

Current Progress:

* Successfully processed 8/11 test stations (~73%)

Remaining Challenges:

* Recover failed EG detection for 3 stations
* Improve shading accuracy
* Increase robustness across drawing styles
