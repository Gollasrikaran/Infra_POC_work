import os
import io
import json
import re
import fitz
import pandas as pd
import streamlit as st
from agent import extract_image_data_from_bytes
from dotenv import load_dotenv
from PIL import Image
st.cache_data.clear()

load_dotenv()

st.set_page_config(page_title="Cross-Section Analyzer", page_icon="🛣️", layout="wide")


st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
    html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
    .stApp { background: #0f1117; color: #e8eaf0; }
    .hero { background: linear-gradient(135deg,#0d1b2a,#1b2d42); border:1px solid #2a3f5a;
            border-radius:12px; padding:2rem; margin-bottom:1.5rem; text-align:center; }
    .hero h1 { font-size:1.8rem; color:#7eb8f7; font-weight:600; margin:0 0 .3rem; }
    .hero p  { color:#8899aa; font-size:.9rem; margin:0; }
    .metric-row { display:flex; gap:1rem; margin-bottom:1.2rem; }
    .metric-box { flex:1; background:#1a2332; border:1px solid #2a3f5a;
                  border-radius:8px; padding:.8rem 1rem; }
    .metric-box .val { font-size:1.6rem; font-weight:600; color:#7eb8f7; font-family:'IBM Plex Mono'; }
    .metric-box .lbl { font-size:.75rem; color:#667788; margin-top:2px; }
    .result-cut  { background:#2a1218; border:1px solid #e11d48; border-radius:10px; padding:1.2rem; margin-top:1rem; }
    .result-fill { background:#0d2218; border:1px solid #16a34a; border-radius:10px; padding:1.2rem; margin-top:1rem; }
    .result-unk  { background:#1a1a2e; border:1px solid #4a5568; border-radius:10px; padding:1.2rem; margin-top:1rem; }
    .result-type { font-size:2rem; font-weight:700; font-family:'IBM Plex Mono'; margin-bottom:.5rem; }
    .cut-color  { color:#f87171; }
    .fill-color { color:#4ade80; }
    .info-grid { display:grid; grid-template-columns:1fr 1fr; gap:.8rem; margin-top:1rem; }
    .info-item { background:#0f1117; border-radius:6px; padding:.6rem .8rem; }
    .info-item .k { font-size:.7rem; color:#667788; text-transform:uppercase; letter-spacing:.05em; }
    .info-item .v { font-size:.9rem; color:#c8d8e8; margin-top:2px; }
    .section-hdr { font-size:.7rem; text-transform:uppercase; letter-spacing:.12em;
                   color:#4a6080; font-weight:600; margin:1.2rem 0 .5rem;
                   border-bottom:1px solid #1e2d3d; padding-bottom:4px; }
    div[data-testid="stFileUploader"] { background:#1a2332; border:2px dashed #2a4a6a; border-radius:10px; padding:.5rem; }
    div[data-testid="stFileUploader"]:hover { border-color:#7eb8f7; }
    .stButton>button { background:linear-gradient(135deg,#1d4ed8,#2563eb); color:white; border:none;
                       border-radius:8px; padding:.6rem 1.5rem; font-weight:600; width:100%; transition:all .2s; }
    .stButton>button:hover { background:linear-gradient(135deg,#2563eb,#3b82f6); transform:translateY(-1px); }
    .stSelectbox>div>div { background:#1a2332; border-color:#2a3f5a; color:#e8eaf0; }
    .stSpinner>div { border-top-color:#7eb8f7 !important; }
    div[data-testid="stImage"] img { border-radius:8px; border:1px solid #2a3f5a; }
    .raw-json { background:#0d1117; border:1px solid #2a3f5a; border-radius:8px;
                padding:1rem; font-family:'IBM Plex Mono'; font-size:.8rem;
                color:#8ab4d8; overflow-x:auto; max-height:300px; }
    .opencv-box { background:#0d1f2d; border:1px solid #1e4d6b; border-radius:10px;
                  padding:1.2rem; margin-top:1rem; }
    .opencv-box .title { font-size:.75rem; text-transform:uppercase; letter-spacing:.1em;
                         color:#4a90c4; font-weight:600; margin-bottom:.8rem; }
</style>
""", unsafe_allow_html=True)


# =========================
# HELPERS
# =========================

@st.cache_data(show_spinner=False)
def get_page_count(pdf_bytes: bytes) -> int:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    n = len(doc)
    doc.close()
    return n


@st.cache_data(show_spinner=False)
def render_page(pdf_bytes: bytes, page_index: int, dpi: int = 150) -> bytes:
    doc  = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc.load_page(page_index)
    rect = page.rect
    scale = dpi / 72
    if (rect.width * scale) * (rect.height * scale) > 3_000_000:
        scale = (3_000_000 / (rect.width * rect.height)) ** 0.5
    pix       = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
    img_bytes = pix.tobytes("png")
    doc.close()
    return img_bytes


def parse_result(raw: str) -> dict | None:
    try:
        clean = re.sub(r"```(?:json)?|```", "", raw).strip()
        return json.loads(clean)
    except Exception:
        return None


def _safe_width(lx, rx) -> str:
    try:
        return f"{abs(float(rx) - float(lx)):.1f} ft"
    except (TypeError, ValueError):
        return "—"


# # =========================
# # OPENCV RESULT RENDERER
# # =========================

# def render_opencv_result(cv_data: dict):
#     """Render the OpenCV pixel-based area calculation results."""

#     if cv_data.get("error"):
#         st.error(f"OpenCV Error: {cv_data['error']}")
#         return

#     grid_px    = cv_data.get("grid_px", "—")
#     px_sqft    = cv_data.get("px_per_sqft", "—")
#     regions    = cv_data.get("regions", [])
#     cut_total  = cv_data.get("total_cut_sqft",  0)
#     fill_total = cv_data.get("total_fill_sqft", 0)
#     net        = cv_data.get("net_sqft", 0)
#     net_color  = "#f87171" if net > 0 else "#4ade80" if net < 0 else "#7eb8f7"

#     st.markdown(f"""
#     <div class="opencv-box">
#         <div class="title">📐 OpenCV Pixel-Based Area Measurement</div>
#         <div style="display:flex;gap:2rem;flex-wrap:wrap;margin-bottom:1rem">
#             <div>
#                 <div class="k" style="font-size:.7rem;color:#667788;text-transform:uppercase">Grid Square</div>
#                 <div style="color:#7eb8f7;font-family:'IBM Plex Mono';font-size:1rem">
#                     {grid_px}px × {grid_px}px = 100 sq ft
#                 </div>
#             </div>
#             <div>
#                 <div class="k" style="font-size:.7rem;color:#667788;text-transform:uppercase">Scale</div>
#                 <div style="color:#c8d8e8;font-family:'IBM Plex Mono';font-size:1rem">
#                     {px_sqft} px² = 1 sq ft
#                 </div>
#             </div>
#             <div>
#                 <div class="k" style="font-size:.7rem;color:#667788;text-transform:uppercase">Regions Detected</div>
#                 <div style="color:#7eb8f7;font-family:'IBM Plex Mono';font-size:1rem">{len(regions)}</div>
#             </div>
#         </div>
#     </div>
#     """, unsafe_allow_html=True)

#     # Per-region cards
#     for i, r in enumerate(regions, 1):
#         rtype     = r.get("type", "UNKNOWN")
#         area      = r.get("area_sqft", 0)
#         width_px  = r.get("width_px", 0)

#         if rtype == "CUT":
#             card_cls, color_cls, icon = "result-cut",  "cut-color",  "🔴"
#             desc = "Proposed grade is BELOW existing ground — excavation needed"
#         elif rtype == "FILL":
#             card_cls, color_cls, icon = "result-fill", "fill-color", "🟢"
#             desc = "Proposed grade is ABOVE existing ground — fill material needed"
#         else:
#             card_cls, color_cls, icon = "result-unk",  "",           "⚪"
#             desc = "Undetermined"

#         st.markdown(f"""
#         <div class="{card_cls}">
#             <div style="font-size:.68rem;color:#667788;text-transform:uppercase;
#                         letter-spacing:.1em;margin-bottom:.3rem">Region {i}</div>
#             <div class="result-type {color_cls}">{icon} {rtype}</div>
#             <div style="color:#8899aa;font-size:.85rem;margin-bottom:.8rem">{desc}</div>
#             <div class="info-grid">
#                 <div class="info-item">
#                     <div class="k">Width</div>
#                     <div class="v">{width_px} px</div>
#                 </div>
#                 <div class="info-item">
#                     <div class="k">Area (OpenCV)</div>
#                     <div class="v" style="font-size:1.1rem;color:#7eb8f7;font-family:'IBM Plex Mono'">
#                         {area:,.2f} sq ft
#                     </div>
#                 </div>
#             </div>
#         </div>
#         """, unsafe_allow_html=True)

#     # Summary
#     st.markdown(f"""
#     <div style="background:#1a2332;border:1px solid #2a3f5a;border-radius:10px;
#                 padding:1rem 1.2rem;margin-top:1.5rem;">
#         <div class="section-hdr" style="margin-top:0">OpenCV Summary</div>
#         <div style="display:flex;gap:2.5rem;flex-wrap:wrap;margin-top:.6rem">
#             <div>
#                 <div style="font-size:.7rem;color:#667788;text-transform:uppercase">Total CUT</div>
#                 <div style="font-size:1.4rem;font-weight:600;color:#f87171;font-family:'IBM Plex Mono'">
#                     {cut_total:,.2f} sq ft
#                 </div>
#             </div>
#             <div>
#                 <div style="font-size:.7rem;color:#667788;text-transform:uppercase">Total FILL</div>
#                 <div style="font-size:1.4rem;font-weight:600;color:#4ade80;font-family:'IBM Plex Mono'">
#                     {fill_total:,.2f} sq ft
#                 </div>
#             </div>
#             <div>
#                 <div style="font-size:.7rem;color:#667788;text-transform:uppercase">Net (CUT − FILL)</div>
#                 <div style="font-size:1.4rem;font-weight:600;color:{net_color};font-family:'IBM Plex Mono'">
#                     {net:+,.2f} sq ft
#                 </div>
#             </div>
#         </div>
#     </div>
#     """, unsafe_allow_html=True)


# =========================
# GEMINI RESULT RENDERER
# =========================

def render_result(data: dict):
    """Render multi-intersection Gemini result — one styled card per region."""

    station  = data.get("station")               or "—"
    slopes   = data.get("slopes")                or "—"
    extent   = data.get("proposed_grade_extent") or "—"
    eg_line  = data.get("existing_ground_line")  or "—"
    n_regions = data.get("total_intersections", 0)

    st.markdown(f"""
    <div style="background:#1a2332;border:1px solid #2a3f5a;border-radius:10px;
                padding:1rem 1.2rem;margin-bottom:1rem;">
        <div style="display:flex;gap:2rem;flex-wrap:wrap;align-items:flex-start;">
            <div>
                <div class="k" style="font-size:.7rem;color:#667788;text-transform:uppercase">Station</div>
                <div style="font-size:1.4rem;font-weight:600;color:#7eb8f7;font-family:'IBM Plex Mono'">{station}</div>
            </div>
            <div>
                <div class="k" style="font-size:.7rem;color:#667788;text-transform:uppercase">Slopes</div>
                <div style="color:#c8d8e8;margin-top:4px">{slopes}</div>
            </div>
            <div>
                <div class="k" style="font-size:.7rem;color:#667788;text-transform:uppercase">Proposed Grade Extent</div>
                <div style="color:#c8d8e8;margin-top:4px">{extent}</div>
            </div>
            <div>
                <div class="k" style="font-size:.7rem;color:#667788;text-transform:uppercase">Regions Found</div>
                <div style="font-size:1.4rem;font-weight:600;color:#7eb8f7;font-family:'IBM Plex Mono'">{n_regions}</div>
            </div>
        </div>
        <div style="margin-top:.8rem;padding-top:.8rem;border-top:1px solid #1e2d3d">
            <span style="font-size:.7rem;color:#667788;text-transform:uppercase">Existing Ground — </span>
            <span style="font-size:.85rem;color:#c8d8e8">{eg_line}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    intersections = data.get("intersections", [])
    for region in intersections:
        rid        = region.get("id", "?")
        rtype      = region.get("type", "UNKNOWN").upper()
        lcp        = region.get("left_catch_point",  {})
        rcp        = region.get("right_catch_point", {})
        rnotes     = region.get("notes", "")
        area_calc  = region.get("area_calculation", {})
        total_area = area_calc.get("total_area_sqft")
        samples    = area_calc.get("sample_points", [])
        side       = region.get("side", "")
        lx, lelv   = lcp.get("x", "—"), lcp.get("elevation", "—")
        rx, relv   = rcp.get("x", "—"), rcp.get("elevation", "—")

        formatted_side = "—"
        if side:
            side_upper = side.upper()
            if side_upper == "LEFT":
                formatted_side = "Left"
            elif side_upper == "RIGHT":
                formatted_side = "Right"
            elif side_upper == "CROSSES_CENTERLINE":
                formatted_side = "Crosses Centerline"
            else:
                formatted_side = side

        if rtype == "CUT":
            card_cls, color_cls, icon = "result-cut",  "cut-color",  "🔴"
            desc = "Proposed grade is BELOW existing ground — excavation needed"
        elif rtype == "FILL":
            card_cls, color_cls, icon = "result-fill", "fill-color", "🟢"
            desc = "Proposed grade is ABOVE existing ground — fill material needed"
        else:
            card_cls, color_cls, icon = "result-unk",  "",           "⚪"
            desc = "Undetermined"

        side_badge_html = f'<span style="background:#2a3f5a;color:#7eb8f7;padding:1px 5px;border-radius:4px;margin-left:8px;font-size:0.6rem">{formatted_side}</span>' if side else ''
        side_info_html = f"""
                <div class="info-item">
                    <div class="k">Side</div>
                    <div class="v">{formatted_side}</div>
                </div>
        """ if side else ""

        st.markdown(f"""
        <div class="{card_cls}">
            <div style="font-size:.68rem;color:#667788;text-transform:uppercase;
                        letter-spacing:.1em;margin-bottom:.3rem">Intersection {rid}{side_badge_html}</div>
            <div class="result-type {color_cls}">{icon} {rtype}</div>
            <div style="color:#8899aa;font-size:.85rem;margin-bottom:.8rem">{desc}</div>
            <div class="info-grid">
                <div class="info-item">
                    <div class="k">Left Catch Point</div>
                    <div class="v">x = {lx} ft &nbsp;|&nbsp; elev = {lelv} ft</div>
                </div>
                <div class="info-item">
                    <div class="k">Right Catch Point</div>
                    <div class="v">x = {rx} ft &nbsp;|&nbsp; elev = {relv} ft</div>
                </div>
                <div class="info-item">
                    <div class="k">Width</div>
                    <div class="v">{_safe_width(lx, rx)}</div>
                </div>
                <div class="info-item">
                    <div class="k">Area (Gemini estimate)</div>
                    <div class="v" style="font-size:1.1rem;color:#7eb8f7;font-family:'IBM Plex Mono'">
                        {f"{total_area:,.1f} sq ft" if total_area is not None else "—"}
                    </div>
                </div>
                {side_info_html}
            </div>
        </div>
        """, unsafe_allow_html=True)

        if samples:
            df = pd.DataFrame(samples).rename(columns={
                "x": "Offset (ft)", "existing_elev": "Existing Elev (ft)",
                "proposed_elev": "Proposed Elev (ft)", "gap": "Gap (ft)"
            })
            with st.expander(f"📊 Intersection {rid} — Trapezoidal Sample Points"):
                st.dataframe(df, use_container_width=True, hide_index=True)

        if rnotes:
            st.markdown(
                f'<div class="info-item" style="margin-top:.5rem">'
                f'<div class="k">Notes</div><div class="v">{rnotes}</div></div>',
                unsafe_allow_html=True
            )

    if intersections:
        cut_total  = sum(r.get("area_calculation",{}).get("total_area_sqft",0) or 0
                         for r in intersections if r.get("type","").upper()=="CUT")
        fill_total = sum(r.get("area_calculation",{}).get("total_area_sqft",0) or 0
                         for r in intersections if r.get("type","").upper()=="FILL")
        net = cut_total - fill_total
        net_color = "#f87171" if net > 0 else "#4ade80" if net < 0 else "#7eb8f7"
        st.markdown(f"""
        <div style="background:#1a2332;border:1px solid #2a3f5a;border-radius:10px;
                    padding:1rem 1.2rem;margin-top:1.5rem;">
            <div class="section-hdr" style="margin-top:0">Gemini Summary</div>
            <div style="display:flex;gap:2.5rem;flex-wrap:wrap;margin-top:.6rem">
                <div><div style="font-size:.7rem;color:#667788;text-transform:uppercase">Total CUT</div>
                     <div style="font-size:1.4rem;font-weight:600;color:#f87171;font-family:'IBM Plex Mono'">{cut_total:,.1f} sq ft</div></div>
                <div><div style="font-size:.7rem;color:#667788;text-transform:uppercase">Total FILL</div>
                     <div style="font-size:1.4rem;font-weight:600;color:#4ade80;font-family:'IBM Plex Mono'">{fill_total:,.1f} sq ft</div></div>
                <div><div style="font-size:.7rem;color:#667788;text-transform:uppercase">Net (CUT − FILL)</div>
                     <div style="font-size:1.4rem;font-weight:600;color:{net_color};font-family:'IBM Plex Mono'">{net:+,.1f} sq ft</div></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    if data.get("reasoning"):
        st.markdown('<div class="section-hdr">Reasoning</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="info-item"><div class="v">{data["reasoning"]}</div></div>',
                    unsafe_allow_html=True)
    if data.get("notes"):
        st.markdown('<div class="section-hdr">Notes</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="info-item"><div class="v">{data["notes"]}</div></div>',
                    unsafe_allow_html=True)
    with st.expander("Raw JSON"):
        st.markdown(f'<div class="raw-json">{json.dumps(data, indent=2)}</div>',
                    unsafe_allow_html=True)


# =========================
# UI
# =========================

st.markdown("""
<div class="hero">
    <h1>🛣️ Highway Cross-Section Analyzer</h1>
    <p>Upload a PDF → Select a page → Analyze with Gemini Vision + OpenCV</p>
</div>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["📄 PDF Analysis", "🖼️ Direct Image → Gemini"])

with tab1:

    uploaded_pdf = st.file_uploader("Upload PDF", type=["pdf"], label_visibility="collapsed")

    if not uploaded_pdf:
        st.markdown("""
        <div style="text-align:center;padding:3rem;color:#4a6080;">
            <div style="font-size:3rem">📄</div>
            <p style="margin-top:.8rem">Drop your highway engineering PDF above to get started</p>
        </div>
        """, unsafe_allow_html=True)
    else:

        pdf_bytes = uploaded_pdf.read()

        with st.spinner("Reading PDF..."):
            total_pages = get_page_count(pdf_bytes)

        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-box"><div class="val">{total_pages}</div><div class="lbl">Total Pages</div></div>
            <div class="metric-box"><div class="val">{total_pages * 5}</div><div class="lbl">Est. Cross-Sections</div></div>
            <div class="metric-box"><div class="val">Gemini + OpenCV</div><div class="lbl">Analysis Engine</div></div>
        </div>
        """, unsafe_allow_html=True)

        col_left, col_right = st.columns([1, 2])

        with col_left:
            st.markdown('<div class="section-hdr">Page Selection</div>', unsafe_allow_html=True)
            selected_page = st.selectbox("Page", options=list(range(1, total_pages + 1)),
                                        format_func=lambda x: f"Page {x}",
                                        label_visibility="collapsed")
            page_index = selected_page - 1

            with st.spinner(f"Rendering page {selected_page}..."):
                img_bytes = render_page(pdf_bytes, page_index, dpi=150)

            image = Image.open(io.BytesIO(img_bytes))
            st.image(image, caption=f"Page {selected_page}", use_container_width=True)
            analyze_clicked = st.button("🔍 Analyze This Page", use_container_width=True)

        with col_right:
            st.markdown('<div class="section-hdr">Analysis Result</div>', unsafe_allow_html=True)

            if analyze_clicked:

                # # ── OpenCV area (runs immediately, no API call) ──────────────────────
                # with st.spinner("📐 Measuring pixel area with OpenCV..."):
                #     import importlib, area_calculator
                #     importlib.reload(area_calculator)
                #     from area_calculator import generate_debug_image, crop_to_drawing, calculate_area
                #     img_bytes_cropped = crop_to_drawing(img_bytes)   # ← removes empty whitespace
                #     cv_result = calculate_area(img_bytes_cropped)
                #     debug_png = generate_debug_image(img_bytes_cropped)
                # render_opencv_result(cv_result)

                # # ── Debug visualization ───────────────────────────────────────────────
                # st.markdown('<div class="section-hdr">🔍 OpenCV Detection Visualization</div>',
                #             unsafe_allow_html=True)
                # st.image(debug_png,
                #          caption="Green=grid lines | Red=line1 | Orange=line2 | Cyan=enclosed area | Yellow=1 grid square(100 sqft)",
                #          use_container_width=True)

                st.markdown('<div class="section-hdr" style="margin-top:2rem">Gemini Vision Analysis</div>',
                            unsafe_allow_html=True)

                # ── Gemini labels, station, CUT/FILL, slopes ────────────────────────
                with st.spinner("🤖 Sending to Gemini Vision..."):
                    raw_result = extract_image_data_from_bytes(img_bytes)

                parsed = parse_result(raw_result)
                if parsed:
                    render_result(parsed)
                else:
                    st.warning("Could not parse JSON from Gemini response.")
                    st.code(raw_result, language="json")

            else:
                st.markdown("""
                <div style="text-align:center;padding:4rem 2rem;color:#4a6080;
                            border:1px dashed #2a3f5a;border-radius:10px;">
                    <div style="font-size:2.5rem">📊</div>
                    <p style="margin-top:.8rem;font-size:.9rem">
                        Select a page and click<br>
                        <strong style="color:#7eb8f7">Analyze This Page</strong>
                    </p>
                </div>
                """, unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="section-hdr">Upload a PNG/JPG cross-section image</div>',
                unsafe_allow_html=True)

    uploaded_img = st.file_uploader(
        "Upload Image", type=["png", "jpg", "jpeg"],
        label_visibility="collapsed", key="img_uploader"
    )

    if not uploaded_img:
        st.markdown("""
        <div style="text-align:center;padding:3rem;color:#4a6080;">
            <div style="font-size:3rem">🖼️</div>
            <p style="margin-top:.8rem">Drop a PNG or JPG cross-section image above</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        img_bytes_direct = uploaded_img.read()
        col_l, col_r = st.columns([1, 2])

        with col_l:
            st.markdown('<div class="section-hdr">Uploaded Image</div>', unsafe_allow_html=True)
            image = Image.open(io.BytesIO(img_bytes_direct))
            st.image(image, caption=uploaded_img.name, use_container_width=True)
            analyze_img_clicked = st.button(
                "🔍 Analyze with Gemini", use_container_width=True, key="analyze_img_btn"
            )

        with col_r:
            st.markdown('<div class="section-hdr">Gemini Vision Analysis</div>',
                        unsafe_allow_html=True)
            if analyze_img_clicked:
                with st.spinner("🤖 Sending to Gemini Vision..."):
                    raw_result = extract_image_data_from_bytes(img_bytes_direct)
                parsed = parse_result(raw_result)
                if parsed:
                    render_result(parsed)
                else:
                    st.warning("Could not parse JSON from Gemini response.")
                    st.code(raw_result, language="json")
            else:
                st.markdown("""
                <div style="text-align:center;padding:4rem 2rem;color:#4a6080;
                            border:1px dashed #2a3f5a;border-radius:10px;">
                    <div style="font-size:2.5rem">📊</div>
                    <p style="margin-top:.8rem;font-size:.9rem">
                        Upload an image and click<br>
                        <strong style="color:#7eb8f7">Analyze with Gemini</strong>
                    </p>
                </div>
                """, unsafe_allow_html=True)
