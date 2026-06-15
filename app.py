"""
Streamlit app for highway cross-section earthwork analysis.
Upload a PDF or images, select a cross-section, and detect Cut/Fill zones.
"""

import streamlit as st
import os
import tempfile
import shutil
import cv2
import pandas as pd
from io import BytesIO
from PIL import Image

from pipeline import (
    extract_scale_report,
    extract_cross_section_images,
    fill_earthwork_zones,
)


# -- Page setup --

st.set_page_config(
    page_title="XDOT Contractor — Road Analyzer",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -- Styling --

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

.stApp { font-family: 'Inter', sans-serif; }

.main-header {
    background: linear-gradient(135deg, #0f172a, #1e293b, #334155);
    padding: 2rem 2.5rem; border-radius: 16px; margin-bottom: 2rem;
    border: 1px solid rgba(99,102,241,0.2);
    box-shadow: 0 4px 24px rgba(0,0,0,0.3);
}
.main-header h1 { color: #f8fafc; font-size: 2rem; font-weight: 800; margin: 0 0 .4rem; }
.main-header p  { color: #94a3b8; margin: 0; }
.main-header .badge {
    display: inline-block; background: linear-gradient(135deg,#6366f1,#8b5cf6);
    color: #fff; padding: .2rem .75rem; border-radius: 20px;
    font-size: .75rem; font-weight: 600; margin-bottom: .75rem; letter-spacing: .5px;
}

.step-card {
    background: linear-gradient(145deg,#1e293b,#0f172a);
    border: 1px solid rgba(99,102,241,0.15); border-radius: 12px;
    padding: 1.5rem; margin-bottom: 1.25rem;
}
.step-card .step-number {
    display: inline-block; background: linear-gradient(135deg,#6366f1,#8b5cf6);
    color: #fff; width: 30px; height: 30px; border-radius: 8px;
    text-align: center; line-height: 30px; font-weight: 700; margin-right: .75rem;
}
.step-card .step-title { color: #e2e8f0; font-size: 1.1rem; font-weight: 600; vertical-align: middle; }
.step-card .step-desc  { color: #94a3b8; font-size: .88rem; margin-top: .75rem; }
.step-done { border-color: rgba(34,197,94,0.3) !important; }
.step-done .step-number { background: linear-gradient(135deg,#22c55e,#16a34a) !important; }

.stat-box {
    background: linear-gradient(145deg,#1e293b,#0f172a);
    border: 1px solid rgba(99,102,241,0.15); border-radius: 12px;
    padding: 1.25rem; text-align: center;
}
.stat-box .stat-value { color: #f8fafc; font-size: 2rem; font-weight: 800; }
.stat-box .stat-label { color: #94a3b8; font-size: .8rem; text-transform: uppercase; letter-spacing: .5px; margin-top: .25rem; }

.legend-container { display: flex; gap: 1.5rem; margin: 1rem 0; }
.legend-item { display: flex; align-items: center; gap: .5rem; color: #cbd5e1; font-size: .88rem; font-weight: 500; }
.legend-dot { width: 14px; height: 14px; border-radius: 4px; }
.legend-dot.red   { background: #ef4444; }
.legend-dot.green { background: #22c55e; }
.legend-dot.blue  { background: #3b82f6; }

section[data-testid="stSidebar"] { background: linear-gradient(180deg,#0f172a,#1e293b) !important; }

.stButton > button {
    background: linear-gradient(135deg,#6366f1,#8b5cf6) !important;
    color: #fff !important; border: none !important; border-radius: 10px !important;
    padding: .6rem 1.5rem !important; font-weight: 600 !important;
    box-shadow: 0 2px 12px rgba(99,102,241,0.3) !important;
}
.stButton > button:hover { transform: translateY(-1px) !important; box-shadow: 0 4px 20px rgba(99,102,241,0.5) !important; }

.stDownloadButton > button {
    background: linear-gradient(135deg,#22c55e,#16a34a) !important;
    color: #fff !important; border: none !important; border-radius: 10px !important;
    font-weight: 600 !important; box-shadow: 0 2px 12px rgba(34,197,94,0.3) !important;
}

[data-testid="stFileUploader"] section {
    border: 2px dashed rgba(99,102,241,0.3) !important; border-radius: 12px !important;
    padding: 2rem !important; background: rgba(99,102,241,0.03) !important;
}

.stProgress > div > div { background: linear-gradient(90deg,#6366f1,#8b5cf6) !important; border-radius: 8px !important; }
hr { border-color: rgba(99,102,241,0.1) !important; margin: 1.5rem 0 !important; }
</style>
""", unsafe_allow_html=True)


# -- Session state --

for key, default in {
    "upload_mode": None,
    "pdf_path": None,
    "pdf_name": None,
    "scale_report": None,
    "images": None,
    "processed_results": None,
    "work_dir": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


def get_work_dir():
    if st.session_state.work_dir is None:
        st.session_state.work_dir = tempfile.mkdtemp(prefix="xdot_")
    return st.session_state.work_dir


# -- Sidebar --

with st.sidebar:
    st.markdown("## 🛣️ Pipeline Steps")

    steps = [
        ("1", "Upload & Extract", "Upload a PDF or images",
         st.session_state.images is not None),
        ("2", "Process All", "Detect Cut/Fill for all images",
         st.session_state.processed_results is not None),
    ]
    for num, title, desc, done in steps:
        cls = "step-card step-done" if done else "step-card"
        mark = " ✓" if done else ""
        st.markdown(f"""
        <div class="{cls}">
            <span class="step-number">{num}</span>
            <span class="step-title">{title}{mark}</span>
            <div class="step-desc">{desc}</div>
        </div>""", unsafe_allow_html=True)

    st.divider()

    # Show scale report if we extracted from a PDF
    if st.session_state.scale_report:
        with st.expander("📐 Scale Report", expanded=False):
            df = pd.DataFrame(st.session_state.scale_report)
            df.columns = ["Page #", "H-Scale", "V-Scale"]
            st.dataframe(df, width='stretch', hide_index=True)
        st.divider()

    if st.button("🔄 Reset Pipeline", width='stretch'):
        if st.session_state.work_dir and os.path.exists(st.session_state.work_dir):
            shutil.rmtree(st.session_state.work_dir, ignore_errors=True)
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()


# -- Header --

st.markdown("""
<div class="main-header">
    <div class="badge">PHASE 1 — RPA-DRIVEN AUTOMATED ESTIMATING</div>
    <h1>🛣️ Road Quantity Analyzer</h1>
    <p>Upload highway planning PDFs or cross-section images to detect and visualize Cut & Fill zones.</p>
</div>""", unsafe_allow_html=True)


# ===========================
# STEP 1 — Upload
# ===========================

st.markdown("### Step 1 — Upload")

tab_pdf, tab_images = st.tabs(["📑 Upload PDF", "🖼️ Upload Images"])

with tab_pdf:
    st.markdown("Upload a highway planning PDF — the app will extract cross-section images from it.")

    pdf_file = st.file_uploader("Select a PDF file", type=["pdf"], key="pdf_uploader")

    if pdf_file is not None:
        is_new = st.session_state.pdf_name != pdf_file.name or st.session_state.upload_mode != "pdf"

        if is_new:
            path = os.path.join(get_work_dir(), pdf_file.name)
            with open(path, "wb") as f:
                f.write(pdf_file.getbuffer())
            st.session_state.update(
                pdf_path=path, pdf_name=pdf_file.name, upload_mode="pdf",
                images=None, processed_results=None, scale_report=None,
            )

        if st.session_state.upload_mode == "pdf" and st.session_state.images is None:
            st.success(f"✅ **{pdf_file.name}** ready.")
            if st.button("🚀 Extract Cross-Sections", key="btn_extract", width='stretch'):
                out_dir = os.path.join(get_work_dir(), "extracted")
                os.makedirs(out_dir, exist_ok=True)

                with st.spinner("Analyzing PDF..."):
                    st.session_state.scale_report = extract_scale_report(st.session_state.pdf_path)

                bar = st.progress(0, text="Extracting images...")
                raw = extract_cross_section_images(
                    st.session_state.pdf_path, out_dir, zoom=4.0,
                    progress_callback=lambda cur, tot: bar.progress(cur / tot, f"Page {cur}/{tot}..."),
                )
                bar.progress(1.0, "Done!")

                st.session_state.images = [
                    {"path": r["path"], "name": os.path.basename(r["path"]),
                     "page": r["page"], "station": r["station"]}
                    for r in raw
                ]
                st.session_state.processed_results = None
                st.rerun()

        elif st.session_state.upload_mode == "pdf" and st.session_state.images is not None:
            st.success(f"✅ Extracted **{len(st.session_state.images)}** images from **{st.session_state.pdf_name}**")


with tab_images:
    st.markdown("Upload cross-section images directly (PNG, JPG).")

    img_files = st.file_uploader(
        "Select images", type=["png", "jpg", "jpeg"],
        accept_multiple_files=True, key="img_uploader",
    )

    if img_files:
        new_names = sorted(f.name for f in img_files)
        old_names = sorted(img["name"] for img in st.session_state.images) if (
            st.session_state.images and st.session_state.upload_mode == "images"
        ) else []

        if new_names != old_names or st.session_state.upload_mode != "images":
            img_dir = os.path.join(get_work_dir(), "uploaded_images")
            os.makedirs(img_dir, exist_ok=True)

            items = []
            for uf in img_files:
                p = os.path.join(img_dir, uf.name)
                with open(p, "wb") as f:
                    f.write(uf.getbuffer())
                items.append({
                    "path": p, "name": uf.name,
                    "page": None, "station": os.path.splitext(uf.name)[0],
                })

            st.session_state.update(
                upload_mode="images", images=items,
                pdf_path=None, pdf_name=None, scale_report=None,
                processed_results=None,
            )
            st.rerun()

        if st.session_state.upload_mode == "images" and st.session_state.images:
            st.success(f"✅ **{len(st.session_state.images)}** image(s) uploaded.")

st.divider()


# ===========================
# STEP 2 — Process All Images
# ===========================

st.markdown("### Step 2 — Process All Images")

st.markdown("""
<div class="legend-container">
    <div class="legend-item"><span class="legend-dot red"></span>Cut Zone (Excavation)</div>
    <div class="legend-item"><span class="legend-dot green"></span>Fill Zone (Embankment)</div>
    <div class="legend-item"><span class="legend-dot blue"></span>Design Profile Line</div>
</div>""", unsafe_allow_html=True)

if not st.session_state.images:
    st.info("Upload a PDF or images in Step 1.")
else:
    images = st.session_state.images

    # Stats
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""<div class="stat-box"><div class="stat-value">{len(images)}</div>
            <div class="stat-label">Total Images</div></div>""", unsafe_allow_html=True)
    with c2:
        mode = "PDF Extraction" if st.session_state.upload_mode == "pdf" else "Direct Upload"
        icon = "📑" if st.session_state.upload_mode == "pdf" else "🖼️"
        st.markdown(f"""<div class="stat-box"><div class="stat-value">{icon}</div>
            <div class="stat-label">{mode}</div></div>""", unsafe_allow_html=True)

    st.markdown("")

    # Preview gallery of extracted images
    with st.expander("📋 Preview Extracted Images", expanded=False):
        cols = st.columns(min(len(images), 4))
        for i, img in enumerate(images):
            with cols[i % 4]:
                try:
                    cap = f"Sta {img['station']}" if img["page"] else img["name"]
                    st.image(Image.open(img["path"]), caption=cap, use_container_width=True)
                except Exception:
                    st.error("Load error")

    debug_on = st.checkbox("🔍 Save debug images (intermediate pipeline stages)", value=False, key="debug_toggle")

    if st.session_state.processed_results is None:
        if st.button(f"🚀 Process All {len(images)} Images", key="btn_process_all", use_container_width=True):
            out_dir = os.path.join(get_work_dir(), "processed")
            os.makedirs(out_dir, exist_ok=True)

            results = []
            bar = st.progress(0, text="Processing images...")

            for idx, img_info in enumerate(images):
                img_label = f"Station {img_info['station']}" if img_info["page"] else img_info["name"]
                bar.progress((idx) / len(images), text=f"Processing {img_label} ({idx + 1}/{len(images)})...")

                gray = cv2.imread(img_info["path"], cv2.IMREAD_GRAYSCALE)
                if gray is None:
                    results.append({
                        "original": img_info["path"],
                        "processed": None,
                        "name": img_info["name"],
                        "label": img_label,
                        "debug_dir": None,
                        "error": True,
                    })
                    continue

                # Debug dir per image (only if enabled)
                dbg_dir = None
                if debug_on:
                    safe_name = os.path.splitext(img_info["name"])[0]
                    dbg_dir = os.path.join(out_dir, "debug", safe_name)

                result = fill_earthwork_zones(gray, debug_dir=dbg_dir)
                out_path = os.path.join(out_dir, f"processed_{img_info['name']}")
                cv2.imwrite(out_path, result)

                results.append({
                    "original": img_info["path"],
                    "processed": out_path,
                    "name": img_info["name"],
                    "label": img_label,
                    "debug_dir": dbg_dir,
                    "error": False,
                })

            bar.progress(1.0, text="Done!")
            st.session_state.processed_results = results
            st.rerun()
    else:
        results = st.session_state.processed_results
        success_count = sum(1 for r in results if not r.get("error"))
        st.success(f"✅ Processed **{success_count}/{len(results)}** images successfully.")

        # Show each result as an expandable section
        for i, res in enumerate(results):
            if res.get("error"):
                st.error(f"❌ **{res['label']}** — Could not read image.")
                continue

            with st.expander(f"📊 {res['label']}", expanded=(i == 0)):
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown("##### Original")
                    try:
                        st.image(Image.open(res["original"]), use_container_width=True)
                    except Exception:
                        st.error("Could not load original.")
                with col_b:
                    st.markdown("##### Processed (Cut / Fill)")
                    try:
                        st.image(Image.open(res["processed"]), use_container_width=True)
                    except Exception:
                        st.error("Could not load result.")

                st.markdown("")
                if res["processed"] and os.path.exists(res["processed"]):
                    with open(res["processed"], "rb") as f:
                        st.download_button(
                            f"⬇️ Download — {res['label']}", f.read(),
                            file_name=f"processed_{res['name']}", mime="image/png",
                            use_container_width=True,
                            key=f"dl_{i}",
                        )

                # Show debug stage images if they were generated
                dbg = res.get("debug_dir")
                if dbg and os.path.isdir(dbg):
                    st.markdown("---")
                    st.markdown("**🔬 Debug: Intermediate Pipeline Stages**")
                    debug_files = [
                        ("1_binary.png", "Stage 1 — Binary threshold"),
                        ("2_grid_detected.png", "Stage 2 — Grid lines detected"),
                        ("3_after_grid_removal.png", "Stage 3 — After grid removal"),
                        ("4_design_mask.png", "Stage 4 — Design (solid) mask"),
                        ("5_dotted_mask.png", "Stage 5 — Ground (dotted) mask"),
                        ("6_final.png", "Stage 6 — Final output"),
                    ]
                    for fname, caption in debug_files:
                        fpath = os.path.join(dbg, fname)
                        if os.path.exists(fpath):
                            st.markdown(f"**{caption}**")
                            st.image(Image.open(fpath), use_container_width=True)
                            st.markdown("")


# -- Footer --

st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#64748b; font-size:.8rem;'>"
    "XDOT Contractor — Phase 1: RPA-Driven Automated Estimating</p>",
    unsafe_allow_html=True,
)
