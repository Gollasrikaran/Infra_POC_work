# 📖 Infra_DMT_POC — Complete Code Explanation

> **Project Goal:** Automatically detect **Cut** (excavation) and **Fill** (embankment) zones in highway cross-section engineering drawings from PDF plans.  
> **Part of:** XDOT Contractor — Phase 1: RPA-Driven Automated Estimating.

---

## 📁 Project Structure

```
Infra_DMT_POC/
├── pipeline.py          ← Core processing logic (PDF parsing + image analysis)
├── app.py               ← Streamlit web UI (the user-facing app)
├── requirements.txt     ← Python dependencies
├── 19series.pdf         ← Sample PDF (cross-section drawings)
├── highway_planning1.pdf← Another sample PDF
└── venv/                ← Python virtual environment
```

---

## File 1: `requirements.txt`

### Why It Exists

This file tells Python's package manager (`pip`) which external libraries the project needs. Without it, anyone cloning the repo would have to guess what to install.

### Line-by-Line

```text
PyMuPDF==1.24.2
```
- **What:** Installs `PyMuPDF` (imported as `fitz` in code) at exactly version 1.24.2.  
- **Why:** This library reads and renders PDF files. It can extract text from specific regions of a page and render pages as high-resolution images. It's the foundation for reading the highway planning PDFs.

```text
opencv-python==4.9.0.80
```
- **What:** Installs OpenCV (imported as `cv2`) at exactly version 4.9.0.80.  
- **Why:** The main image processing library. Used for thresholding (separating ink from background), morphological operations (finding/removing grid lines), connected component analysis (classifying line types), and drawing the colored overlays.

```text
numpy==1.26.4
```
- **What:** Installs NumPy at exactly version 1.26.4.  
- **Why:** NumPy provides fast array/matrix operations. Since images are stored as NumPy arrays, nearly all pixel-level manipulation (finding where lines are, filling zones, etc.) relies on NumPy.

```text
streamlit
```
- **What:** Installs the latest version of Streamlit (no version pinned).  
- **Why:** Streamlit is the web UI framework. It turns `app.py` into an interactive web application with file uploaders, buttons, progress bars, image displays, etc. — all without writing HTML/JS manually.

```text
Pillow
```
- **What:** Installs Pillow (imported as `PIL`).  
- **Why:** Python's standard image library. Used to open images for display in Streamlit (`Image.open()`). Streamlit uses Pillow internally to render images.

```text
pandas
```
- **What:** Installs pandas.  
- **Why:** Used to display the scale report as a formatted table in the Streamlit sidebar (`pd.DataFrame` → `st.dataframe`).

---

## File 2: `pipeline.py`

### Why It Exists

This is the **brain** of the application. It contains all the actual engineering logic — reading PDFs, extracting images, removing grid lines, and detecting Cut/Fill zones. It's separated from the UI (`app.py`) so the logic can be reused, tested, or called from anywhere without needing Streamlit.

### Line-by-Line

---

#### Lines 1–5: Imports

```python
import fitz  # PyMuPDF
```
- **What:** Imports PyMuPDF under the name `fitz`.  
- **Why:** Needed to open PDFs, read text from specific page regions, search for text, and render pages as images. The library is called `fitz` because it's based on the MuPDF rendering engine created by Artifex.

```python
import cv2
```
- **What:** Imports OpenCV.  
- **Why:** All image processing operations: thresholding, morphological transforms (opening, closing, dilation), connected component analysis, and pixel manipulation.

```python
import numpy as np
```
- **What:** Imports NumPy.  
- **Why:** Images are NumPy arrays. Every pixel-level operation (finding where lines are, filling regions, interpolation) uses NumPy's fast array indexing.

```python
import os
```
- **What:** Imports the OS module.  
- **Why:** File path operations — checking if files exist (`os.path.exists`), joining paths (`os.path.join`), creating directories (`os.makedirs`).

```python
import re
```
- **What:** Imports Python's regular expression module.  
- **Why:** Used to parse drawing numbers and scale text from PDFs. For example, extracting `10'` from `HORIZONTAL: 1" = 10'`.

---

#### Lines 8–57: `extract_scale_report(pdf_path)`

**Purpose:** Scans every page of a highway PDF to find pages that contain 19-series cross-section drawings, then extracts the horizontal and vertical scale information from those pages.

**Why "19-series"?** In U.S. highway plan sets, sheets numbered in the 19-series specifically contain cross-section views — the side-view cuts of the road that show the existing ground vs. the proposed design.

```python
def extract_scale_report(pdf_path):
```
- Defines the function. Takes one argument: the file path to a PDF.

```python
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
```
- **Safety check.** If the file doesn't exist, raise a clear error immediately rather than crashing deep inside PyMuPDF with a confusing message.

```python
    doc = fitz.open(pdf_path)
    results = []
```
- Opens the PDF file. `doc` is now a PyMuPDF Document object containing all pages.
- `results` will collect the scale info for each matching page.

```python
    for i in range(len(doc)):
        page = doc[i]
        rect = page.rect
```
- Loops through every page in the PDF (0-indexed).  
- `page` = the current page object.  
- `rect` = the page's bounding rectangle (gives us width and height in PDF points).

```python
        corner = fitz.Rect(rect.width * 0.70, rect.height * 0.80, rect.width, rect.height)
        corner_text = page.get_text("text", clip=corner).strip()
```
- **Why the bottom-right corner?** Highway plan sheets always put the drawing number and title in a title block in the bottom-right area.  
- Creates a rectangle covering the bottom-right 30% width × bottom 20% height.  
- Extracts all text within that rectangle.

```python
        is_19 = False
        for line in corner_text.split('\n'):
            cleaned = re.sub(r'(?i)DRAWING|NO\.?|DRG|[:\s]', '', line).strip()
            if (cleaned.startswith("19-") or cleaned.startswith("19")) and "+" not in cleaned:
                is_19 = True
                break
```
- **Goal:** Determine if this page has a drawing number starting with "19".  
- Splits the corner text into individual lines and checks each one.  
- `re.sub(...)` removes common label words like "DRAWING", "NO.", "DRG", colons, and spaces — leaving just the raw drawing number (e.g., `19-004`).  
- `cleaned.startswith("19")` — checks if it's a 19-series sheet.  
- `"+" not in cleaned` — excludes station numbers like `125+00` which also contain "19" sometimes. Station numbers always have a `+` sign.

```python
        if not is_19:
            continue
```
- Skip this page if it's not a 19-series drawing.

```python
        if not re.search(r'(?i)cross[- \s]*section', corner_text):
            continue
```
- **Second filter:** Even among 19-series sheets, we only want pages that say "CROSS SECTION" (or "cross-section", "cross section").  
- `(?i)` makes it case-insensitive. `[- \s]*` allows for hyphens, spaces, or nothing between "cross" and "section".

```python
        scale_area = fitz.Rect(0, rect.height * 0.70, rect.width, rect.height)
        text = page.get_text("text", clip=scale_area)
```
- Now that we've confirmed this is a 19-series cross-section page, extract text from the bottom 30% of the entire page width — this is where scale legends typically appear.

```python
        h_match = re.search(r'(?i)HORIZONTAL[:\s]*1"\s*=\s*(\d+\'?)', text)
        v_match = re.search(r'(?i)VERTICAL[:\s]*1"\s*=\s*(\d+\'?)', text)
```
- **Regex breakdown:** Looks for text like `HORIZONTAL: 1" = 10'`  
  - `(?i)` — case insensitive  
  - `HORIZONTAL[:\s]*` — the word "HORIZONTAL" followed by optional colons/spaces  
  - `1"\s*=\s*` — literally `1" =` with flexible spacing  
  - `(\d+\'?)` — captures the number (e.g., `10'`), the `'` foot mark is optional  
- Same pattern for VERTICAL scale.

```python
        results.append({
            "page": i + 1,
            "h_scale": f'1" = {h_match.group(1)}' if h_match else '1" = 10\' (default)',
            "v_scale": f'1" = {v_match.group(1)}' if v_match else '1" = 10\' (default)',
        })
```
- Adds this page's info to the results list.  
- `i + 1` converts from 0-indexed to 1-indexed (human-friendly page numbers).  
- If the regex didn't match (scale text is missing or formatted differently), uses a sensible default of `1" = 10'`.

```python
    doc.close()
    return results
```
- Closes the PDF (releases memory/file handles).  
- Returns the list of scale reports.

---

#### Lines 60–130: `extract_cross_section_images(pdf_path, output_dir, zoom=4.0, progress_callback=None)`

**Purpose:** Goes through the PDF, finds 19-series cross-section pages (same filtering logic), then splits each page into individual station-level images and saves them as high-resolution PNGs.

**Why split by station?** Each cross-section page typically shows 3–6 different stations (locations along the highway) stacked vertically. Processing works better on individual stations.

```python
def extract_cross_section_images(pdf_path, output_dir, zoom=4.0, progress_callback=None):
```
- `pdf_path` — path to the PDF file.  
- `output_dir` — folder where extracted PNGs will be saved.  
- `zoom=4.0` — render at 4× resolution (turns a 72 DPI PDF into a 288 DPI image for better detail).  
- `progress_callback` — optional function that receives `(current_page, total_pages)` to update a progress bar.

```python
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
```
- Same safety check as before.

```python
    os.makedirs(output_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    total = len(doc)
    extracted = []
```
- Creates the output directory if it doesn't exist.  
- Opens the PDF.  
- `total` stores the page count (for progress tracking).  
- `extracted` will collect info about each saved image.

```python
    for i in range(total):
        page = doc[i]
        rect = page.rect
```
- Loop through every page.

```python
        corner = fitz.Rect(rect.width * 0.65, rect.height * 0.75, rect.width, rect.height)
        corner_text = page.get_text("text", clip=corner).strip()
```
- Same corner text extraction, but with a slightly larger area (65% width, 75% height) — gives a bit more room to find drawing numbers.

```python
        is_19 = False
        for line in corner_text.split('\n'):
            cleaned = re.sub(r'(?i)DRAWING|NO\.?|DRG|[:\s]', '', line).strip()
            if (cleaned.startswith("19-") or cleaned.startswith("19")) and "+" not in cleaned:
                is_19 = True
                break
```
- Same 19-series detection logic as `extract_scale_report`.

```python
        if not (is_19 and re.search(r'(?i)cross[- \s]*section', corner_text)):
            if progress_callback:
                progress_callback(i + 1, total)
            continue
```
- If this page isn't a 19-series cross-section, skip it. But still fire the progress callback so the progress bar moves.

```python
        right_strip = fitz.Rect(rect.width * 0.80, 0, rect.width, rect.height)
        stations = page.search_for("+", clip=right_strip)
        stations.sort(key=lambda s: s.y0)
```
- **Finding station labels:** Station numbers look like `125+00`. The `+` is always present.  
- Searches for all `+` symbols in the right 20% of the page (where station labels are printed).  
- Sorts them top-to-bottom by their Y position on the page.

```python
        last_cut = 35
```
- `last_cut` tracks where the bottom of the previous crop ended.  
- Starts at 35 points from the top (skips the page header/border area).

```python
        for j, label in enumerate(stations):
            y_top = last_cut
            y_bot = label.y1 + 48
```
- For each station label found:
  - `y_top` = bottom of the previous crop (so crops don't overlap).
  - `y_bot` = 48 points below the station label (gives some padding below).

```python
            if j + 1 < len(stations):
                y_bot = min(y_bot, stations[j + 1].y0 - 20)
            else:
                y_bot = min(y_bot, rect.height * 0.88)
```
- **Prevent overlap:** If there's a next station, don't extend past 20 points above it.  
- For the last station, don't extend past 88% of the page height (avoids the title block).

```python
            last_cut = y_bot
            crop = fitz.Rect(0, y_top, rect.width, y_bot)
```
- Update `last_cut` for the next iteration.  
- Define the crop rectangle: full width, from `y_top` to `y_bot`.

```python
            text_area = fitz.Rect(rect.width * 0.80, label.y0 - 30, rect.width, label.y1 + 30)
            sta_text = page.get_text("text", clip=text_area).strip()
            sta_name = re.sub(r'[^0-9+]', '', sta_text) or f"sta_{j + 1}"
```
- Reads the text around the `+` label to get the full station name (e.g., `125+00`).  
- `re.sub(r'[^0-9+]', '', ...)` strips everything except digits and `+` — removes "STA", spaces, etc.  
- Falls back to `sta_1`, `sta_2` if no valid name is found.

```python
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=crop)
            filename = f"Page{i + 1}_Sta_{sta_name}.png"
            save_path = os.path.join(output_dir, filename)
            pix.save(save_path)
```
- **Renders the cropped region** at 4× zoom as a pixel map (raster image).  
- `fitz.Matrix(zoom, zoom)` scales both X and Y by the zoom factor.  
- Saves as PNG with a descriptive filename like `Page42_Sta_125+00.png`.

```python
            extracted.append({"path": save_path, "page": i + 1, "station": sta_name})
```
- Records the saved image's path, page number, and station name.

```python
        if progress_callback:
            progress_callback(i + 1, total)
```
- Updates the progress bar after each page.

```python
    doc.close()
    return extracted
```
- Closes the PDF and returns the list of all extracted images.

---

#### Lines 133–325: `fill_earthwork_zones(img_gray)`

**Purpose:** This is the most complex and important function. It takes a single grayscale cross-section image and:

1. Removes the background engineering grid
2. Separates the **solid design line** (proposed road shape) from the **dotted ground line** (existing terrain)
3. Fills the area between them with colors:
   - 🔴 **Red = Cut** (ground is above design → material must be excavated)
   - 🟢 **Green = Fill** (design is above ground → material must be added)

Returns a color (BGR) image with the overlays.

```python
def fill_earthwork_zones(img_gray):
```
- Takes a grayscale image (single-channel NumPy array, values 0–255).

```python
    h_img, w_img = img_gray.shape
```
- Gets the image dimensions. `h_img` = height (rows), `w_img` = width (columns).

---

##### Grid Removal (Lines 148–166)

Cross-section drawings have a light gray engineering grid in the background. This grid interferes with line detection, so it must be removed first.

```python
    _, bw = cv2.threshold(img_gray, 235, 255, cv2.THRESH_BINARY_INV)
```
- **Binary threshold at 235:** Any pixel darker than 235 (ink, lines, text) becomes white (255). Anything lighter (background, faint grid) becomes black (0).  
- `THRESH_BINARY_INV` inverts so that **marks = white, background = black** — this is the convention for morphological operations.

```python
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 150))
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (150, 1))
    grid = cv2.add(
        cv2.morphologyEx(bw, cv2.MORPH_OPEN, vert_kernel),
        cv2.morphologyEx(bw, cv2.MORPH_OPEN, horiz_kernel),
    )
```
- **Morphological opening** with long thin kernels to isolate grid lines:
  - `(1, 150)` — a 1-pixel-wide, 150-pixel-tall kernel. Only vertical structures at least 150px long survive → catches vertical grid lines.
  - `(150, 1)` — same idea horizontally.
- `cv2.add` combines both → full grid mask.
- **Why "opening"?** Opening = erosion followed by dilation. It removes anything smaller than the kernel shape, then restores the surviving structures to original size. Short marks (text, dots, curves) are erased; long straight grid lines survive.

```python
    drawing = cv2.subtract(bw, grid)
```
- Subtracts the grid mask from the thresholded image → leaves only the actual drawing content (lines, text, dots).

```python
    shield = cv2.dilate(drawing, np.ones((3, 3), np.uint8), iterations=1)
```
- **Bridge protection:** Dilates (expands) the drawing content by 1 pixel in all directions. This creates a "shield" around real drawing elements.  
- **Why?** Grid lines sometimes overlap with real drawing lines. Without the shield, erasing the grid would also erase parts of the actual drawing where they intersect.

```python
    eraser = cv2.subtract(grid, shield)
```
- The final eraser mask = grid pixels that DON'T overlap with the shield. Only "pure" grid pixels far from real drawing content get erased.

```python
    clean = cv2.subtract(bw, eraser)
```
- The cleaned binary image: original threshold minus the grid.

```python
    output = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    output[eraser > 0] = [255, 255, 255]
```
- Creates the color output image (starts as a 3-channel copy of the original grayscale).  
- Whites out the grid pixels in the output.

---

##### Line Classification (Lines 168–199)

Now we need to tell apart two types of lines in the drawing:
- **Solid lines** = the proposed road design profile  
- **Dotted lines** = the existing ground/terrain profile

```python
    nlabels, labels, stats, _ = cv2.connectedComponentsWithStats(clean, 8, cv2.CV_32S)
```
- **Connected component analysis:** Groups adjacent white pixels into "blobs" (components).  
- `nlabels` = total number of components found.  
- `labels` = a 2D array where each pixel's value is its component ID.  
- `stats` = for each component: `[x, y, width, height, area]`.  
- `8` = 8-way connectivity (diagonal pixels count as connected).

```python
    border = 3
    cutoff_y = int(h_img * 0.88)
```
- `border = 3` — ignore components touching the image edge (likely page borders).  
- `cutoff_y` — ignore components below 88% of the image height (that's the scale bar/legend area, not actual drawing content).

```python
    solid_mask = np.zeros_like(clean)
    dotted_mask = np.zeros_like(clean)
```
- Two blank masks to accumulate classified pixels:
  - `solid_mask` will hold design line pixels.
  - `dotted_mask` will hold ground line pixels.

```python
    for i in range(1, nlabels):
        x, y, w, h, area = stats[i]
```
- Loop through each component (skip `0` which is the background).  
- Unpack the bounding box and area.

```python
        if x < border or (x + w) > (w_img - border):
            continue
        if y < border or (y + h) > (h_img - border):
            continue
        if y > cutoff_y:
            continue
```
- **Filter out:** Components touching the image border (page frame lines) and components below the cutoff (scale bar text).

```python
        if h >= 16 and w <= 45:
            continue
```
- **Filter out text:** Text characters are typically tall-ish (≥16px) and narrow (≤45px). We don't want text classified as either line type.

```python
        diag = np.sqrt(w**2 + h**2)
        aspect = w / h if h > 0 else 0
```
- `diag` = diagonal length of the bounding box. Larger values indicate bigger components.  
- `aspect` = width-to-height ratio.

```python
        if 3 <= w <= 45 and 2 <= h <= 12 and aspect > 1.0:
            dotted_mask[labels == i] = 255
```
- **Dotted line segments:** Small, wider-than-tall blobs. Dot/dash marks in the ground line are typically small rectangular marks. Width 3–45px, height 2–12px, and wider than tall (aspect > 1.0).

```python
        elif diag > 55 or w > 50 or h > 50:
            solid_mask[labels == i] = 255
```
- **Solid line segments:** Large components — either long diagonal reach (>55px) or wide/tall (>50px in either dimension). These are the continuous lines of the design profile.

---

##### Boundary Healing (Lines 201–204)

```python
    heal_k = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    solid_boundary = cv2.morphologyEx(solid_mask, cv2.MORPH_CLOSE, heal_k)
    dotted_boundary = cv2.morphologyEx(dotted_mask, cv2.MORPH_CLOSE, heal_k)
```
- **Morphological close** with a 25px-wide horizontal kernel.  
- Close = dilation → erosion. It fills small gaps (up to 25px) between nearby segments without changing the overall shape.  
- **Why?** The classification may produce fragmented lines with small horizontal gaps. Closing connects these fragments into continuous boundary lines.

---

##### Column-by-Column Fill Detection (Lines 206–260)

This is the core algorithm: for each vertical column of pixels, find where the solid line and dotted line are, then fill the space between them with the appropriate color.

```python
    red_fill = np.zeros_like(clean)
    green_fill = np.zeros_like(clean)
```
- Two blank masks for accumulating cut (red) and fill (green) zones.

```python
    has_fill = np.zeros(w_img, dtype=bool)
    top_y = np.zeros(w_img, dtype=int)
    bot_y = np.zeros(w_img, dtype=int)
    fill_type = np.zeros(w_img, dtype=int)
```
- Per-column tracking arrays:
  - `has_fill[c]` — whether column `c` has a fill zone.
  - `top_y[c]` / `bot_y[c]` — the top and bottom Y coordinates of the zone in column `c`.
  - `fill_type[c]` — 1 for cut (red), 2 for fill (green).

```python
    window = 25
```
- **Neighbor search window:** If a column has no line pixels, look up to 25 columns left/right for the nearest line position. This handles small gaps in the lines.

```python
    for c in range(w_img):
        solid_rows = np.where(solid_boundary[:, c] == 255)[0]
```
- For each column `c`, find all row positions where the solid (design) line exists.

```python
        if len(solid_rows) == 0:
            left = max(0, c - window)
            right = min(w_img, c + window + 1)
            nearby = np.where(solid_boundary[:, left:right] == 255)
            if len(nearby[0]) > 0:
                st, sb = np.min(nearby[0]), np.max(nearby[0])
            else:
                continue
        else:
            st, sb = np.min(solid_rows), np.max(solid_rows)
```
- If no solid line at this column, search within ±25 columns.  
- `st` = top of solid line, `sb` = bottom of solid line.

```python
        dots_above = np.where(dotted_boundary[:st, c] == 255)[0]
        dots_below = np.where(dotted_boundary[sb:, c] == 255)[0]
```
- Look for dotted (ground) line **above** and **below** the solid line in this column.

```python
        if len(dots_above) == 0 and len(dots_below) == 0:
            left = max(0, c - window)
            right = min(w_img, c + window + 1)
            nearby_above = np.where(dotted_boundary[:st, left:right] == 255)
            nearby_below = np.where(dotted_boundary[sb:, left:right] == 255)
            if len(nearby_above[0]) > 0:
                dots_above = nearby_above[0]
            if len(nearby_below[0]) > 0:
                dots_below = nearby_below[0]
```
- Same neighbor-search fallback for the dotted line.

```python
        if len(dots_above) > 0:
            dot_y = np.max(dots_above)
            if dot_y < st:
                has_fill[c] = True
                top_y[c] = dot_y
                bot_y[c] = st
                fill_type[c] = 1  # cut zone
```
- **Dotted line is ABOVE solid line → CUT zone.**  
  - In cross-section drawings, if the existing ground (dotted) is above the design (solid), you need to **cut** (excavate) material away.  
  - The zone to fill spans from the dotted line down to the solid line.

```python
        elif len(dots_below) > 0:
            dot_y = np.min(dots_below) + sb
            if sb < dot_y:
                has_fill[c] = True
                top_y[c] = sb
                bot_y[c] = dot_y
                fill_type[c] = 2  # fill zone
```
- **Dotted line is BELOW solid line → FILL zone.**  
  - If the design (solid) is above the existing ground (dotted), you need to **fill** (add embankment material).  
  - The zone spans from the solid line down to the dotted line.  
  - Note: `+ sb` adjusts the index since `dots_below` was computed relative to row `sb`.

---

##### Gap Interpolation (Lines 262–290)

```python
    valid = np.where(has_fill)[0]
```
- Find all columns that have a detected fill zone.

```python
    if len(valid) > 1:
        max_gap = 40
```
- Only proceed if at least 2 columns have data. `max_gap = 40` means we'll interpolate across gaps up to 40 columns wide.

```python
        for c in range(np.min(valid), np.max(valid) + 1):
            if has_fill[c]:
                ty, by, ct = top_y[c], bot_y[c], fill_type[c]
```
- For columns that already have data, just use it.

```python
            else:
                lefts = valid[valid < c]
                rights = valid[valid > c]
                if len(lefts) == 0 or len(rights) == 0:
                    continue

                li, ri = lefts[-1], rights[0]
                if (ri - li) > max_gap:
                    continue

                w_frac = (c - li) / (ri - li)
                ty = int(top_y[li] + w_frac * (top_y[ri] - top_y[li]))
                by = int(bot_y[li] + w_frac * (bot_y[ri] - bot_y[li]))
                ct = fill_type[li] if w_frac <= 0.5 else fill_type[ri]
```
- **For gap columns:** Find the nearest valid column to the left (`li`) and right (`ri`).  
- If the gap is too wide (>40px), skip — it's likely a real break between zones.  
- Otherwise, **linearly interpolate** the top and bottom Y positions.  
  - `w_frac` = how far we are from the left neighbor (0.0 = at left, 1.0 = at right).  
  - `ty` and `by` are blended between the left and right values.  
  - `ct` uses the fill type of whichever neighbor is closer.

```python
            if ty < by:
                if ct == 1:
                    red_fill[ty:by, c] = 255
                elif ct == 2:
                    green_fill[ty:by, c] = 255
```
- If the zone is valid (top above bottom), fill the appropriate mask.

---

##### Horizontal Gap Filling (Lines 292–304)

```python
    if len(valid) > 1:
        x_min, x_max = np.min(valid), np.max(valid)
        max_h_gap = 60
```
- **Pass 3:** After vertical filling and interpolation, there may still be small horizontal gaps within a row. This fills them.

```python
        for r in range(cutoff_y):
            for overlay in [green_fill, red_fill]:
                cols = np.where(overlay[r, x_min:x_max + 1] == 255)[0] + x_min
                if len(cols) > 1:
                    for idx in range(len(cols) - 1):
                        gap = cols[idx + 1] - cols[idx]
                        if 1 < gap < max_h_gap:
                            overlay[r, cols[idx]:cols[idx + 1]] = 255
```
- For each row, find all filled pixels.  
- If two filled pixels are close (gap < 60px), fill the space between them.  
- This creates smoother, more continuous fill zones.

---

##### Smoothing (Lines 306–314)

```python
    smooth_h = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
    smooth_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))

    red_clean = cv2.morphologyEx(red_fill, cv2.MORPH_CLOSE, smooth_h)
    red_clean = cv2.morphologyEx(red_clean, cv2.MORPH_CLOSE, smooth_v)

    green_clean = cv2.morphologyEx(green_fill, cv2.MORPH_CLOSE, smooth_h)
    green_clean = cv2.morphologyEx(green_clean, cv2.MORPH_CLOSE, smooth_v)
```
- **Morphological close** to smooth ragged edges.  
- Horizontal close (20px) smooths horizontal jaggedness.  
- Vertical close (3px) smooths tiny vertical gaps.  
- This makes the final output look cleaner.

---

##### Final Overlay (Lines 316–325)

```python
    output[red_clean == 255] = [0, 0, 255]     # red = cut
    output[green_clean == 255] = [0, 210, 0]    # green = fill
```
- Paints the cut zones red and fill zones green on the output image.  
- OpenCV uses BGR order, so `[0, 0, 255]` = red, `[0, 210, 0]` = green.

```python
    output[solid_mask == 255] = [255, 0, 0]
```
- Draws the design line in blue on top of everything (so it's always visible).

```python
    text = (img_gray < 80) & (eraser == 0)
    output[text] = [0, 0, 0]
```
- Restores original dark text/labels in black. Pixels that were very dark (<80) in the original image AND were not grid pixels are considered text and drawn in black on top.

```python
    return output
```
- Returns the final color image with all overlays.

---

## File 3: `app.py`

### Why It Exists

This is the **user interface**. It wraps all the `pipeline.py` logic into an interactive web application using Streamlit. Users can upload PDFs or images, see results visually, and download processed output — without touching any code.

### Line-by-Line

---

#### Lines 1–13: Module Docstring & Imports

```python
"""
Streamlit app for highway cross-section earthwork analysis.
Upload a PDF or images, select a cross-section, and detect Cut/Fill zones.
"""
```
- Docstring explaining what this file does.

```python
import streamlit as st
```
- Streamlit's main library. `st` provides all UI widgets: `st.button()`, `st.file_uploader()`, `st.image()`, `st.progress()`, etc. Every `st.xxx()` call adds a UI element to the page.

```python
import os
```
- File path operations.

```python
import tempfile
```
- Creates temporary directories for storing uploaded files and processed outputs. These are cleaned up when the pipeline is reset.

```python
import shutil
```
- Used to recursively delete temporary directories on pipeline reset (`shutil.rmtree`).

```python
import cv2
```
- Reads images from disk as grayscale arrays (input to `fill_earthwork_zones`).

```python
import pandas as pd
```
- Converts the scale report (list of dicts) into a DataFrame for display as a table.

```python
from io import BytesIO
```
- Imported for potential in-memory byte stream operations (e.g., creating ZIP downloads). Not actively used in the current flow but available for future features.

```python
from PIL import Image
```
- Opens image files for display in Streamlit. Streamlit's `st.image()` works well with Pillow Image objects.

```python
from pipeline import (
    extract_scale_report,
    extract_cross_section_images,
    fill_earthwork_zones,
)
```
- Imports the three core functions from `pipeline.py`. The UI calls these functions when the user clicks buttons.

---

#### Lines 22–29: Page Configuration

```python
st.set_page_config(
    page_title="XDOT Contractor — Road Analyzer",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded",
)
```
- **Must be the first Streamlit command** (Streamlit rule).  
- Sets the browser tab title, favicon emoji, wide layout (uses full browser width), and starts with the sidebar open.

---

#### Lines 32–108: Custom CSS Styling

```python
st.markdown("""<style>...</style>""", unsafe_allow_html=True)
```
- Injects custom CSS to override Streamlit's default look.  
- `unsafe_allow_html=True` is required for raw HTML/CSS injection in Streamlit.

**Key style classes:**

| Class | Purpose |
|-------|---------|
| `.main-header` | The dark gradient header banner at the top of the page |
| `.badge` | The "PHASE 1" purple pill label inside the header |
| `.step-card` | Sidebar pipeline step indicators (numbered cards) |
| `.step-done` | Green border/icon for completed steps |
| `.stat-box` | The statistics boxes showing image count and upload mode |
| `.legend-container` / `.legend-item` | The Cut/Fill/Design color legend |
| `.stButton > button` | Styles all Streamlit buttons with purple gradient |
| `.stDownloadButton > button` | Styles download buttons with green gradient |
| `[data-testid="stFileUploader"]` | Styles the file upload drop zone with a dashed border |
| `.stProgress > div > div` | Styles the progress bar with a purple gradient |

- **Color palette:** Uses a dark slate theme (`#0f172a`, `#1e293b`, `#334155`) with indigo/violet accents (`#6366f1`, `#8b5cf6`). Green (`#22c55e`) for success states.
- **Font:** Google Fonts "Inter" for a clean, modern look.

---

#### Lines 111–124: Session State Initialization

```python
for key, default in {
    "upload_mode": None,
    "pdf_path": None,
    "pdf_name": None,
    "scale_report": None,
    "images": None,
    "selected_idx": None,
    "processed_result": None,
    "work_dir": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default
```
- **Why session state?** Streamlit reruns the entire script on every interaction (button click, dropdown change, etc.). `st.session_state` persists data between reruns.  
- Initializes all state variables only if they don't already exist (so we don't reset them on every rerun).

| Key | What It Stores |
|-----|---------------|
| `upload_mode` | `"pdf"` or `"images"` — how files were uploaded |
| `pdf_path` | Full path to the saved PDF temp file |
| `pdf_name` | Original filename of the uploaded PDF |
| `scale_report` | List of dicts from `extract_scale_report()` |
| `images` | List of dicts, each with `path`, `name`, `page`, `station` |
| `selected_idx` | Index of the currently selected image |
| `processed_result` | Dict with `original`, `processed`, `name`, `label` for the processed image |
| `work_dir` | Path to the temporary working directory |

---

#### Lines 127–130: Temp Directory Helper

```python
def get_work_dir():
    if st.session_state.work_dir is None:
        st.session_state.work_dir = tempfile.mkdtemp(prefix="xdot_")
    return st.session_state.work_dir
```
- Creates a temp directory named like `xdot_abc123` the first time it's needed, then reuses it.

---

#### Lines 133–171: Sidebar

```python
with st.sidebar:
```
- Everything inside this block renders in the left sidebar.

```python
    st.markdown("## 🛣️ Pipeline Steps")
```
- Sidebar heading.

```python
    steps = [
        ("1", "Upload", "Upload a PDF or images",
         st.session_state.images is not None),
        ("2", "Select Image", "Choose a cross-section to process",
         st.session_state.selected_idx is not None),
        ("3", "Process Zones", "Detect Cut/Fill for selected image",
         st.session_state.processed_result is not None),
    ]
```
- Defines the 3 pipeline steps. The last element (`done`) is `True` if that step has been completed.

```python
    for num, title, desc, done in steps:
        cls = "step-card step-done" if done else "step-card"
        mark = " ✓" if done else ""
        st.markdown(f"""
        <div class="{cls}">
            <span class="step-number">{num}</span>
            <span class="step-title">{title}{mark}</span>
            <div class="step-desc">{desc}</div>
        </div>""", unsafe_allow_html=True)
```
- Renders each step as a styled card. Completed steps get a green border and checkmark.

```python
    if st.session_state.scale_report:
        with st.expander("📐 Scale Report", expanded=False):
            df = pd.DataFrame(st.session_state.scale_report)
            df.columns = ["Page #", "H-Scale", "V-Scale"]
            st.dataframe(df, use_container_width=True, hide_index=True)
```
- If a scale report exists (from PDF extraction), shows it in a collapsible expander as a table.

```python
    if st.button("🔄 Reset Pipeline", use_container_width=True):
        if st.session_state.work_dir and os.path.exists(st.session_state.work_dir):
            shutil.rmtree(st.session_state.work_dir, ignore_errors=True)
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()
```
- **Reset button:** Deletes the temp directory, clears all session state, and forces a full page rerun — starting fresh.

---

#### Lines 174–181: Page Header

```python
st.markdown("""
<div class="main-header">
    <div class="badge">PHASE 1 — RPA-DRIVEN AUTOMATED ESTIMATING</div>
    <h1>🛣️ Road Quantity Analyzer</h1>
    <p>Upload highway planning PDFs or cross-section images to detect and visualize Cut & Fill zones.</p>
</div>""", unsafe_allow_html=True)
```
- The dark gradient banner at the top. Uses the CSS classes defined earlier.

---

#### Lines 184–275: Step 1 — Upload

```python
st.markdown("### 📄 Step 1 — Upload")
tab_pdf, tab_images = st.tabs(["📑 Upload PDF", "🖼️ Upload Images"])
```
- Creates two tabs: one for PDF upload, one for direct image upload.

**PDF Tab (lines 192–234):**

```python
    pdf_file = st.file_uploader("Select a PDF file", type=["pdf"], key="pdf_uploader")
```
- A file upload widget that accepts only `.pdf` files.

```python
    if pdf_file is not None:
        is_new = st.session_state.pdf_name != pdf_file.name or st.session_state.upload_mode != "pdf"
```
- Checks if the user uploaded a new/different file.

```python
        if is_new:
            path = os.path.join(get_work_dir(), pdf_file.name)
            with open(path, "wb") as f:
                f.write(pdf_file.getbuffer())
            st.session_state.update(
                pdf_path=path, pdf_name=pdf_file.name, upload_mode="pdf",
                images=None, selected_idx=None, processed_result=None, scale_report=None,
            )
```
- Saves the uploaded PDF to the temp directory and resets downstream state.

```python
        if st.session_state.upload_mode == "pdf" and st.session_state.images is None:
            st.success(f"✅ **{pdf_file.name}** ready.")
            if st.button("🚀 Extract Cross-Sections", key="btn_extract", use_container_width=True):
```
- Shows a success message and an "Extract" button. The button only appears if images haven't been extracted yet.

```python
                with st.spinner("Analyzing PDF..."):
                    st.session_state.scale_report = extract_scale_report(st.session_state.pdf_path)
```
- Runs the scale report extraction with a spinner animation.

```python
                bar = st.progress(0, text="Extracting images...")
                raw = extract_cross_section_images(
                    st.session_state.pdf_path, out_dir, zoom=4.0,
                    progress_callback=lambda cur, tot: bar.progress(cur / tot, f"Page {cur}/{tot}..."),
                )
                bar.progress(1.0, "Done!")
```
- Extracts images with a progress bar. The `progress_callback` lambda updates the bar after each page.

```python
                st.session_state.images = [
                    {"path": r["path"], "name": os.path.basename(r["path"]),
                     "page": r["page"], "station": r["station"]}
                    for r in raw
                ]
                st.session_state.update(selected_idx=None, processed_result=None)
                st.rerun()
```
- Stores the extracted images in session state and forces a rerun to show the next step.

**Images Tab (lines 237–273):**

```python
    img_files = st.file_uploader(
        "Select images", type=["png", "jpg", "jpeg"],
        accept_multiple_files=True, key="img_uploader",
    )
```
- Multi-file uploader for image files.

```python
    if img_files:
        new_names = sorted(f.name for f in img_files)
        old_names = sorted(img["name"] for img in st.session_state.images) if (
            st.session_state.images and st.session_state.upload_mode == "images"
        ) else []

        if new_names != old_names or st.session_state.upload_mode != "images":
```
- Detects if the user uploaded new/different images by comparing sorted filenames.

```python
            items = []
            for uf in img_files:
                p = os.path.join(img_dir, uf.name)
                with open(p, "wb") as f:
                    f.write(uf.getbuffer())
                items.append({
                    "path": p, "name": uf.name,
                    "page": None, "station": os.path.splitext(uf.name)[0],
                })
```
- Saves each uploaded image to disk and records it. Since there's no PDF page info, `page` is `None` and `station` is the filename without extension.

---

#### Lines 278–342: Step 2 — Select Image

```python
st.markdown("### 🎯 Step 2 — Select an Image to Process")
```

```python
if not st.session_state.images:
    st.info("⬆️ Upload a PDF or images in Step 1.")
```
- If no images exist yet, show a hint to go back to Step 1.

```python
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""<div class="stat-box"><div class="stat-value">{len(images)}</div>
            <div class="stat-label">Total Images</div></div>""", unsafe_allow_html=True)
    with c2:
        mode = "PDF Extraction" if st.session_state.upload_mode == "pdf" else "Direct Upload"
        ...
```
- Two stat boxes showing image count and upload mode.

```python
    choice = st.selectbox(
        "Choose a cross-section image:",
        options=labels,
        index=st.session_state.selected_idx or 0,
        key="image_selector",
    )
    idx = labels.index(choice)
```
- A dropdown to select which image to process. Labels show "Page X — Station Y" for PDF extractions, or the filename for direct uploads.

```python
    col_l, col_m, col_r = st.columns([1, 2, 1])
    with col_m:
        try:
            st.image(Image.open(images[idx]["path"]), caption=choice, use_container_width=True)
        except Exception as e:
            st.error(f"Could not load: {e}")
```
- Shows a centered preview of the selected image (1:2:1 column ratio gives centering).

```python
    if st.session_state.selected_idx != idx:
        st.session_state.selected_idx = idx
        st.session_state.processed_result = None
```
- When selection changes, update state and clear any previous processing result.

```python
    with st.expander("📋 View All Images", expanded=False):
        cols = st.columns(min(len(images), 4))
        for i, img in enumerate(images):
            with cols[i % 4]:
                ...
                st.image(Image.open(img["path"]), caption=cap, use_container_width=True)
```
- A collapsible gallery showing all extracted images in a 4-column grid.

---

#### Lines 346–410: Step 3 — Process

```python
st.markdown("### 🎨 Step 3 — Road Zone Detection")
```

```python
st.markdown("""
<div class="legend-container">
    <div class="legend-item"><span class="legend-dot red"></span>Cut Zone (Excavation)</div>
    <div class="legend-item"><span class="legend-dot green"></span>Fill Zone (Embankment)</div>
    <div class="legend-item"><span class="legend-dot blue"></span>Design Profile Line</div>
</div>""", unsafe_allow_html=True)
```
- Color legend explaining what each color means.

```python
    if st.button(f"⚡ Process — {label}", key="btn_process", use_container_width=True):
        ...
        with st.spinner(f"Processing {label}..."):
            gray = cv2.imread(sel["path"], cv2.IMREAD_GRAYSCALE)
            if gray is None:
                st.error("❌ Could not read the image.")
            else:
                result = fill_earthwork_zones(gray)
                out_path = os.path.join(out_dir, f"processed_{sel['name']}")
                cv2.imwrite(out_path, result)
```
- **The processing button.** Reads the selected image as grayscale, runs `fill_earthwork_zones()`, and saves the result.

```python
                st.session_state.processed_result = {
                    "original": sel["path"],
                    "processed": out_path,
                    "name": sel["name"],
                    "label": label,
                }
                st.rerun()
```
- Stores the result paths in session state and reruns to show the comparison.

```python
    else:
        res = st.session_state.processed_result
        st.success(f"✅ Processed **{res['label']}**")

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("##### 📷 Original")
            st.image(Image.open(res["original"]), use_container_width=True)
        with col_b:
            st.markdown("##### 🎨 Processed (Cut / Fill)")
            st.image(Image.open(res["processed"]), use_container_width=True)
```
- **Side-by-side comparison:** Shows the original and processed images in two columns.

```python
        if os.path.exists(res["processed"]):
            with open(res["processed"], "rb") as f:
                st.download_button(
                    "📥 Download Processed Image", f.read(),
                    file_name=f"processed_{res['name']}", mime="image/png",
                    use_container_width=True,
                )
```
- A green download button to save the processed image locally.

---

#### Lines 413–420: Footer

```python
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#64748b; font-size:.8rem;'>"
    "XDOT Contractor — Phase 1: RPA-Driven Automated Estimating</p>",
    unsafe_allow_html=True,
)
```
- A subtle centered footer with the project name and phase.

---

## Summary: How Everything Connects

```
User opens browser → app.py (Streamlit UI)
                        │
                        ├── Upload PDF ─────────► pipeline.extract_scale_report()
                        │                              → returns scale table
                        │
                        ├── Click "Extract" ────► pipeline.extract_cross_section_images()
                        │                              → saves PNGs, returns file list
                        │
                        ├── Select an image ────► preview shown in UI
                        │
                        └── Click "Process" ────► pipeline.fill_earthwork_zones()
                                                       → returns color-coded image
                                                       → displayed as before/after
                                                       → downloadable
```

| Color in Output | Meaning | Engineering Term |
|-----------------|---------|-----------------|
| 🔴 Red | Ground is above design | **Cut** — excavate material |
| 🟢 Green | Design is above ground | **Fill** — add embankment |
| 🔵 Blue | Design profile line | **Design grade** |
| ⬛ Black | Original text and labels | Station names, dimensions |
