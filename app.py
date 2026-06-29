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


def generate_overlay_from_llm_profiles(img_bytes: bytes, existing_pts: list, proposed_pts: list) -> bytes:
    """
    Interpolates existing ground and proposed grade profiles detected by the LLM
    and draws the CUT (red) / FILL (green) color overlays.
    """
    from PIL import Image, ImageDraw
    from scipy.interpolate import interp1d
    import numpy as np
    
    if not existing_pts or not proposed_pts:
        return img_bytes
        
    try:
        # Load image
        image = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
        width, height = image.size
        
        # Extract and scale points
        ex_x = [float(pt["x"]) * width / 1000.0 for pt in existing_pts]
        ex_y = [float(pt["y"]) * height / 1000.0 for pt in existing_pts]
        
        pr_x = [float(pt["x"]) * width / 1000.0 for pt in proposed_pts]
        pr_y = [float(pt["y"]) * height / 1000.0 for pt in proposed_pts]
        
        # Sort by x coordinate and remove duplicates
        ex_unique = {}
        for x, y in zip(ex_x, ex_y):
            if x not in ex_unique:
                ex_unique[x] = y
        ex_sorted = sorted(ex_unique.items())
        
        pr_unique = {}
        for x, y in zip(pr_x, pr_y):
            if x not in pr_unique:
                pr_unique[x] = y
        pr_sorted = sorted(pr_unique.items())
        
        if len(ex_sorted) < 2 or len(pr_sorted) < 2:
            return img_bytes
            
        ex_x_s, ex_y_s = zip(*ex_sorted)
        pr_x_s, pr_y_s = zip(*pr_sorted)
        
        # Interpolate
        f_existing = interp1d(ex_x_s, ex_y_s, kind='linear', bounds_error=False, fill_value="extrapolate")
        f_proposed = interp1d(pr_x_s, pr_y_s, kind='linear', bounds_error=False, fill_value="extrapolate")
        
        # Find overlap horizontal range
        min_x = max(min(ex_x_s), min(pr_x_s))
        max_x = min(max(ex_x_s), max(pr_x_s))
        
        if min_x >= max_x:
            return img_bytes
            
        # Draw transparent overlay
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        for x in range(int(min_x), int(max_x) + 1):
            y_exist = float(f_existing(x))
            y_prop = float(f_proposed(x))
            
            y_top = min(y_exist, y_prop)
            y_bot = max(y_exist, y_prop)
            
            # Clamp values to image dimensions
            y_top = max(0, min(y_top, height - 1))
            y_bot = max(0, min(y_bot, height - 1))
            
            # y_prop < y_exist means proposed is higher elevation (FILL -> Green)
            # y_exist < y_prop means existing is higher elevation (CUT -> Red)
            if y_prop < y_exist:
                # Green with alpha (semi-transparent green)
                fill_color = (0, 255, 0, 140)
            else:
                # Red with alpha (semi-transparent red)
                fill_color = (255, 0, 0, 140)
                
            draw.line([(x, y_top), (x, y_bot)], fill=fill_color, width=1)
            
        # Combine overlay and original
        combined = Image.alpha_composite(image, overlay).convert("RGB")
        
        out_buf = io.BytesIO()
        combined.save(out_buf, format="PNG")
        return out_buf.getvalue()
        
    except Exception as e:
        st.error(f"Error generating visual overlay: {e}")
        return img_bytes


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
    """Render the visual-only region classification result from Gemini."""

    station = data.get("station") or "—"
    ex_profile = data.get("existing_ground_profile")
    if isinstance(ex_profile, list):
        ex_profile_desc = f"Detected {len(ex_profile)} coordinates tracing the dashed/dotted line."
    else:
        ex_profile_desc = str(ex_profile or "—")

    pr_profile = data.get("proposed_grade_profile")
    if isinstance(pr_profile, list):
        pr_profile_desc = f"Detected {len(pr_profile)} coordinates tracing the solid proposed grade line."
    else:
        pr_profile_desc = str(pr_profile or "—")

    st.markdown(f"""
    <div style="background:#1a2332;border:1px solid #2a3f5a;border-radius:10px;
                padding:1rem 1.2rem;margin-bottom:1rem;">
        <div style="display:flex;gap:2rem;flex-wrap:wrap;align-items:flex-start;">
            <div>
                <div class="k" style="font-size:.7rem;color:#667788;text-transform:uppercase">Station</div>
                <div style="font-size:1.4rem;font-weight:600;color:#7eb8f7;font-family:'IBM Plex Mono'">{station}</div>
            </div>
            <div>
                <div class="k" style="font-size:.7rem;color:#667788;text-transform:uppercase">Existing Ground Profile</div>
                <div style="color:#c8d8e8;margin-top:4px;font-size:.85rem;">{ex_profile_desc}</div>
            </div>
            <div>
                <div class="k" style="font-size:.7rem;color:#667788;text-transform:uppercase">Proposed Grade Profile</div>
                <div style="color:#c8d8e8;margin-top:4px;font-size:.85rem;">{pr_profile_desc}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    regions = data.get("regions", [])
    st.markdown('<div class="section-hdr">Classified Regions</div>', unsafe_allow_html=True)
    if not regions:
        st.info("No regions detected between profiles.")
    
    for region in regions:
        rid = region.get("region_id", "?")
        classification = region.get("classification", "UNKNOWN").upper()
        confidence = region.get("confidence", "—")
        reasoning = region.get("reasoning", "")

        if classification == "CUT":
            card_cls, color_cls, icon = "result-cut", "cut-color", "🔴"
            desc = "Existing Ground is ABOVE Proposed Grade — excavation area"
        elif classification == "FILL":
            card_cls, color_cls, icon = "result-fill", "fill-color", "🟢"
            desc = "Proposed Grade is ABOVE Existing Ground — fill area"
        else:
            card_cls, color_cls, icon = "result-unk", "", "⚪"
            desc = "Undetermined region"

        st.markdown(f"""
        <div class="{card_cls}">
            <div style="font-size:.68rem;color:#667788;text-transform:uppercase;
                        letter-spacing:.1em;margin-bottom:.3rem">Region {rid}</div>
            <div class="result-type {color_cls}">{icon} {classification}</div>
            <div style="color:#8899aa;font-size:.85rem;margin-bottom:.8rem">{desc}</div>
            <div class="info-grid">
                <div class="info-item">
                    <div class="k">Confidence Score</div>
                    <div class="v" style="font-size:1.1rem;color:#7eb8f7;font-family:'IBM Plex Mono'">{confidence}%</div>
                </div>
                <div class="info-item" style="grid-column: span 2;">
                    <div class="k">Visual Reasoning</div>
                    <div class="v" style="margin-top:4px;line-height:1.4;">{reasoning}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Visual Coloring Recommendations
    color_recs = data.get("visual_coloring_recommendations", {})
    if color_recs:
        st.markdown('<div class="section-hdr">Visual Coloring Plan</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="info-item" style="border-left: 4px solid #4ade80; margin-bottom: 0.8rem;">
                <div class="k" style="color: #4ade80;">Green Fill Areas (FILL)</div>
                <div class="v" style="margin-top: 4px;">{color_recs.get("green_regions", "None specified")}</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="info-item" style="border-left: 4px solid #f87171; margin-bottom: 0.8rem;">
                <div class="k" style="color: #f87171;">Red Fill Areas (CUT)</div>
                <div class="v" style="margin-top: 4px;">{color_recs.get("red_regions", "None specified")}</div>
            </div>
            """, unsafe_allow_html=True)
        
        why_selected = color_recs.get("why_selected")
        if why_selected:
            st.markdown(f"""
            <div class="info-item" style="margin-top: 0.5rem;">
                <div class="k">Rationale for Color Placement</div>
                <div class="v" style="margin-top: 4px;">{why_selected}</div>
            </div>
            """, unsafe_allow_html=True)

    uncertainties = data.get("uncertainties")
    if uncertainties and str(uncertainties).lower() != "none" and str(uncertainties).lower() != "null":
        st.markdown('<div class="section-hdr">Uncertainties / Limitations</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="info-item" style="border-left: 4px solid #fbbf24;">
            <div class="k" style="color: #fbbf24;">Visual Challenges</div>
            <div class="v" style="margin-top:4px;">{uncertainties}</div>
        </div>
        """, unsafe_allow_html=True)

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
            <div class="metric-box"><div class="val">Gemini Vision</div><div class="lbl">Analysis Engine</div></div>
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
                    # Color overlay generation based on LLM profiles
                    existing_pts = parsed.get("existing_ground_profile", [])
                    proposed_pts = parsed.get("proposed_grade_profile", [])
                    if existing_pts and proposed_pts:
                        with st.spinner("🎨 Generating color overlay..."):
                            overlay_img_bytes = generate_overlay_from_llm_profiles(img_bytes, existing_pts, proposed_pts)
                        st.markdown('<div class="section-hdr">Color-Filled Cross-Section Overlay</div>', unsafe_allow_html=True)
                        st.image(overlay_img_bytes, caption="Visual Color Overlay (Green = FILL | Red = CUT)", use_container_width=True)
                    
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
                    # Color overlay generation based on LLM profiles
                    existing_pts = parsed.get("existing_ground_profile", [])
                    proposed_pts = parsed.get("proposed_grade_profile", [])
                    if existing_pts and proposed_pts:
                        with st.spinner("🎨 Generating color overlay..."):
                            overlay_img_bytes = generate_overlay_from_llm_profiles(img_bytes_direct, existing_pts, proposed_pts)
                        st.markdown('<div class="section-hdr">Color-Filled Cross-Section Overlay</div>', unsafe_allow_html=True)
                        st.image(overlay_img_bytes, caption="Visual Color Overlay (Green = FILL | Red = CUT)", use_container_width=True)
                    
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
