"""
Cross-Section Graph Splitter
─────────────────────────────
Splits a PDF where each page contains multiple graphs stacked vertically
into a new PDF where each graph gets its own page.

Detection strategy:
  Render each page at medium resolution, convert to grayscale, then find
  wide horizontal bands of near-white pixels (the gaps between graphs).
  Those gaps define crop boundaries. Each crop is rendered at high res
  and placed on its own landscape page in the output PDF.

UI: Upload → auto-process → download.  That's it.
"""

import io
import os
import re
import tempfile
from scipy.signal import find_peaks

import fitz                         # PyMuPDF
import numpy as np
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Graph Splitter — 1 graph per page", layout="wide")

# ─────────────────────────────────────────────────────────────────────────────
# Tuning constants (no sidebar — sensible defaults)
# ─────────────────────────────────────────────────────────────────────────────

SCAN_ZOOM          = 2.0     # zoom for the analysis render (speed vs accuracy)
OUTPUT_ZOOM        = 3.0     # zoom for the final high-res crop render

# A row is "blank" if at least this fraction of its pixels are near-white


# Minimum height of a blank band to count as a separator (in scan-zoom pixels)


# Minimum height of a graph region (in scan-zoom pixels) — skip tiny slivers





#

# ─────────────────────────────────────────────────────────────────────────────
# Core: find graph boundaries via whitespace-gap analysis
# ─────────────────────────────────────────────────────────────────────────────

def _find_graph_regions(page: fitz.Page) -> list[fitz.Rect]:

    pw = page.rect.width
    ph = page.rect.height

    scan_zoom = 2.0

    pix = page.get_pixmap(
        matrix=fitz.Matrix(scan_zoom, scan_zoom),
        alpha=False
    )

    arr = np.frombuffer(
        pix.samples,
        dtype=np.uint8
    ).reshape(
        pix.height,
        pix.width,
        3
    )

    gray = arr.mean(axis=2)

    # ------------------------------------------------------------------
    # Ignore header/footer
    # ------------------------------------------------------------------
    top_ignore = int(pix.height * 0.06)
    bottom_ignore = int(pix.height * 0.14)

    usable_top = top_ignore
    usable_bottom = pix.height - bottom_ignore

    # ------------------------------------------------------------------
    # Density profile
    # ------------------------------------------------------------------
    density = np.mean(gray < 220, axis=1)

    kernel = np.ones(75) / 75

    density = np.convolve(
        density,
        kernel,
        mode="same"
    )

    # ------------------------------------------------------------------
    # Assume 4 graphs per page
    # ------------------------------------------------------------------
    expected_graphs = 4

    usable_height = usable_bottom - usable_top

    approx_height = usable_height / expected_graphs

    boundaries = [usable_top]

    for i in range(1, expected_graphs):

        expected_split = int(
            usable_top +
            i * approx_height
        )

        search_start = max(
            usable_top,
            expected_split - 120
        )

        search_end = min(
            usable_bottom,
            expected_split + 120
        )

        local_density = density[
            search_start:search_end
        ]

        valley = (
            np.argmin(local_density)
            + search_start
        )

        boundaries.append(valley)

    boundaries.append(usable_bottom)

    # ------------------------------------------------------------------
    # Convert to PDF coords
    # ------------------------------------------------------------------
    scale = 1.0 / scan_zoom

    rects = []

    padding = 8

    for i in range(len(boundaries) - 1):

        top_px = max(
            0,
            boundaries[i] - padding
        )

        bottom_px = min(
            pix.height,
            boundaries[i + 1] + padding
        )

        rects.append(
            fitz.Rect(
                0,
                top_px * scale,
                pw,
                bottom_px * scale
            )
        )

    return rects


def render_crop(page: fitz.Page, crop_rect: fitz.Rect, zoom: float) -> Image.Image:
    """Render a rectangular crop of the page at the given zoom → PIL Image."""
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, clip=crop_rect, alpha=False)
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


def extract_graphs(pdf_bytes: bytes, progress_cb=None) -> list[dict]:
    """
    Main entry point.

    Returns list of dicts:
        { "page": int, "index": int, "image": PIL.Image }
    """
    doc    = fitz.open(stream=pdf_bytes, filetype="pdf")
    total  = len(doc)
    result = []

    for i in range(total):
        page   = doc[i]
        regions = _find_graph_regions(page)

        for j, rect in enumerate(regions):
            img = render_crop(page, rect, OUTPUT_ZOOM)
            result.append({
                "page":  i + 1,
                "index": j + 1,
                "image": img,
            })

        if progress_cb:
            progress_cb(i + 1, total)

    doc.close()
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Output: build a PDF with one graph per page
# ─────────────────────────────────────────────────────────────────────────────

def build_pdf(records: list[dict]) -> io.BytesIO:
    """Create a new PDF — one landscape page per graph."""
    from reportlab.lib.pagesizes import landscape, letter
    from reportlab.pdfgen import canvas as rl_canvas

    buf = io.BytesIO()
    pw, ph = landscape(letter)
    c = rl_canvas.Canvas(buf, pagesize=(pw, ph))

    with tempfile.TemporaryDirectory() as tmpdir:
        for rec in records:
            img       = rec["image"]
            iw, ih    = img.size
            scale     = min(pw / iw, ph / ih) * 0.95
            dw, dh    = iw * scale, ih * scale
            x         = (pw - dw) / 2
            y         = (ph - dh) / 2
            tmp_path  = os.path.join(tmpdir, f"p{rec['page']:04d}_g{rec['index']:03d}.png")
            img.save(tmp_path, "PNG")
            c.drawImage(tmp_path, x, y, width=dw, height=dh)
            c.setFont("Helvetica", 7)
            caption = f"Page {rec['page']}  ·  Graph {rec['index']}"
            c.drawCentredString(pw / 2, 12, caption)
            c.showPage()

    c.save()
    buf.seek(0)
    return buf


# ─────────────────────────────────────────────────────────────────────────────
# Streamlit UI — dead simple: upload → process → download
# ─────────────────────────────────────────────────────────────────────────────

st.title("📊 Graph Splitter")
st.caption("Upload a PDF with multiple graphs per page → get a new PDF with one graph per page.")

uploaded = st.file_uploader("Upload your PDF", type=["pdf"], label_visibility="collapsed")

if not uploaded:
    st.info("👆 Upload a PDF to get started.")
    st.stop()

pdf_bytes = uploaded.read()

# ── Extract ──────────────────────────────────────────────────────────────────
progress_bar = st.progress(0, text="Opening PDF…")

def _prog(done, total):
    progress_bar.progress(done / total, text=f"Scanning page {done} / {total}…")

with st.spinner("Detecting and splitting graphs…"):
    records = extract_graphs(pdf_bytes, progress_cb=_prog)

progress_bar.empty()

# ── Results summary ──────────────────────────────────────────────────────────
st.success(f"Found **{len(records)}** graph(s) across the PDF.")

# ── Build output PDF immediately ─────────────────────────────────────────────
with st.spinner("Building output PDF…"):
    pdf_out = build_pdf(records)

st.download_button(
    "⬇️  Download split PDF  (1 graph per page)",
    data=pdf_out,
    file_name="graphs_split.pdf",
    mime="application/pdf",
    use_container_width=True,
    type="primary",
)

# ── Preview ──────────────────────────────────────────────────────────────────
with st.expander("🔍 Preview extracted graphs", expanded=True):
    for rec in records:
        st.image(
            rec["image"],
            caption=f"Page {rec['page']} · Graph {rec['index']}",
            use_container_width=True,
        )
        st.markdown("---")