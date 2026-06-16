"""
Streamlit app — Direct PDF Vector Extraction pipeline.
Upload a highway PDF → classify pages → extract profiles → compute cut/fill volumes.
"""

import streamlit as st
import os
import tempfile
import shutil
import pandas as pd
from PIL import Image

from orchestrator import VectorPipeline
from pdf_classifier import PDFClassifier


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="XDOT Contractor — Road Quantity Analyzer",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Theme ─────────────────────────────────────────────────────────────────────

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

.vol-card {
    background: linear-gradient(145deg,#1e293b,#0f172a);
    border: 1px solid rgba(99,102,241,0.15); border-radius: 12px;
    padding: 1.5rem; text-align: center;
}
.vol-card.cut  { border-color: rgba(239,68,68,0.4); }
.vol-card.fill { border-color: rgba(34,197,94,0.4); }
.vol-card.net  { border-color: rgba(99,102,241,0.4); }
.vol-card .vol-value { font-size: 1.8rem; font-weight: 800; }
.vol-card .vol-unit  { font-size: .85rem; color: #94a3b8; }
.vol-card .vol-label { font-size: .75rem; color: #64748b; text-transform: uppercase; letter-spacing: .5px; margin-top: .25rem; }
.vol-card.cut  .vol-value { color: #ef4444; }
.vol-card.fill .vol-value { color: #22c55e; }
.vol-card.net  .vol-value { color: #818cf8; }

.legend-container { display: flex; gap: 1.5rem; margin: 1rem 0; flex-wrap: wrap; }
.legend-item { display: flex; align-items: center; gap: .5rem; color: #cbd5e1; font-size: .88rem; font-weight: 500; }
.legend-dot { width: 14px; height: 14px; border-radius: 4px; }
.legend-dot.red   { background: #ef4444; }
.legend-dot.green { background: #22c55e; }
.legend-dot.blue  { background: #3b82f6; }
.legend-dot.gray  { background: #64748b; }

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


# ── State ─────────────────────────────────────────────────────────────────────

for key, default in {
    "pdf_path": None, "pdf_name": None, "doc_analysis": None,
    "selected_pages": [], "pipeline_result": None, "work_dir": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


def get_work_dir():
    if st.session_state.work_dir is None:
        st.session_state.work_dir = tempfile.mkdtemp(prefix="xdot_vec_")
    return st.session_state.work_dir


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🛣️ Vector Pipeline")

    steps = [
        ("1", "Upload & Classify", "Upload PDF, analyze structure",
         st.session_state.doc_analysis is not None),
        ("2", "Select Pages", "Choose pages to process",
         len(st.session_state.selected_pages) > 0),
        ("3", "Run Pipeline", "Extract profiles, compute volumes",
         st.session_state.pipeline_result is not None),
        ("4", "Results", "View volumes, plots, reports",
         st.session_state.pipeline_result is not None
         and len(getattr(st.session_state.pipeline_result, "successful_pages", [])) > 0),
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

    if st.session_state.doc_analysis:
        da = st.session_state.doc_analysis
        with st.expander("📊 Page Classification", expanded=False):
            st.markdown(f"""
            | Type | Count |
            |------|-------|
            | 🟢 Vector | **{len(da.vector_pages)}** |
            | 🟡 Hybrid | **{len(da.hybrid_pages)}** |
            | 🔴 Raster | **{len(da.raster_pages)}** |
            | 📐 Cross-Section | **{len(da.cross_section_pages)}** |
            """)

    st.divider()

    if st.button("🔄 Reset Pipeline", use_container_width=True):
        if st.session_state.work_dir and os.path.exists(st.session_state.work_dir):
            shutil.rmtree(st.session_state.work_dir, ignore_errors=True)
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()


# ── Header ────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="main-header">
    <div class="badge">APPROACH 2 — DIRECT PDF VECTOR EXTRACTION</div>
    <h1>🛣️ Road Quantity Analyzer</h1>
    <p>Extract profile geometry from CAD-generated PDFs and compute earthwork cut/fill volumes.</p>
</div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Upload & Classify
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("### Step 1 — Upload & Classify PDF")

pdf_file = st.file_uploader("Select a highway engineering PDF", type=["pdf"], key="pdf_uploader")

if pdf_file is not None:
    if st.session_state.pdf_name != pdf_file.name:
        path = os.path.join(get_work_dir(), pdf_file.name)
        with open(path, "wb") as f:
            f.write(pdf_file.getbuffer())
        st.session_state.update(
            pdf_path=path, pdf_name=pdf_file.name,
            doc_analysis=None, selected_pages=[], pipeline_result=None,
        )

    if st.session_state.doc_analysis is None:
        st.success(f"✅ **{pdf_file.name}** uploaded.")
        if st.button("🔍 Analyze PDF Structure", key="btn_classify", use_container_width=True):
            with st.spinner("Analyzing PDF pages..."):
                classifier = PDFClassifier(st.session_state.pdf_path)
                st.session_state.doc_analysis = classifier.analyze()
            st.rerun()
    else:
        da = st.session_state.doc_analysis
        st.success(f"✅ **{da.total_pages}** pages analyzed in **{st.session_state.pdf_name}**")

        with st.expander("📋 Full Page Classification Table", expanded=False):
            classifier = PDFClassifier(st.session_state.pdf_path)
            st.dataframe(pd.DataFrame(classifier.to_dict_list(da)),
                         use_container_width=True, hide_index=True)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f'<div class="stat-box"><div class="stat-value">{da.total_pages}</div>'
                        '<div class="stat-label">Total Pages</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="stat-box"><div class="stat-value">{len(da.vector_pages)}</div>'
                        '<div class="stat-label">Vector Pages</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="stat-box"><div class="stat-value">{len(da.cross_section_pages)}</div>'
                        '<div class="stat-label">Cross-Sections</div></div>', unsafe_allow_html=True)
        with c4:
            st.markdown(f'<div class="stat-box"><div class="stat-value">{len(da.processable_pages)}</div>'
                        '<div class="stat-label">Processable</div></div>', unsafe_allow_html=True)

st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Select Pages
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("### Step 2 — Select Pages to Process")

if not st.session_state.doc_analysis:
    st.info("Upload and analyze a PDF in Step 1.")
else:
    da = st.session_state.doc_analysis
    processable = da.processable_pages

    if not processable:
        st.warning("No processable cross-section pages found.")
        all_vec = [p for p in da.pages if p.classification in ("VECTOR", "HYBRID")]
        if all_vec:
            st.info(f"Found {len(all_vec)} vector/hybrid pages — select manually:")
            selected = st.multiselect("Select pages", [p.page_number for p in all_vec],
                                      default=[], key="manual_page_select")
            if selected:
                st.session_state.selected_pages = selected
    else:
        st.markdown(f"**{len(processable)}** pages ready for vector extraction.")
        page_nums = [p.page_number for p in processable]

        col_a, col_b = st.columns([3, 1])
        with col_a:
            selected = st.multiselect("Select pages (all by default)", page_nums,
                                      default=page_nums, key="page_select")
            st.session_state.selected_pages = selected
        with col_b:
            st.markdown("")
            st.markdown("")
            if st.button("Select All", key="btn_all"):
                st.session_state.selected_pages = page_nums
                st.rerun()

        with st.expander("⚙️ Scale Overrides (optional)", expanded=False):
            st.markdown("Leave at **0** to auto-detect from PDF text.")
            sc1, sc2 = st.columns(2)
            with sc1:
                h_over = st.number_input("Horizontal Scale (ft/inch)", min_value=0.0,
                                         value=0.0, step=1.0, key="h_scale_override")
            with sc2:
                v_over = st.number_input("Vertical Scale (ft/inch)", min_value=0.0,
                                         value=0.0, step=1.0, key="v_scale_override")

st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Run Pipeline
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("### Step 3 — Run Vector Pipeline")

st.markdown("""
<div class="legend-container">
    <div class="legend-item"><span class="legend-dot red"></span>Cut (Excavation)</div>
    <div class="legend-item"><span class="legend-dot green"></span>Fill (Embankment)</div>
    <div class="legend-item"><span class="legend-dot blue"></span>Proposed Grade</div>
    <div class="legend-item"><span class="legend-dot gray"></span>Existing Ground</div>
</div>""", unsafe_allow_html=True)

if not st.session_state.selected_pages:
    st.info("Select pages in Step 2.")
elif st.session_state.pipeline_result is None:
    pages = st.session_state.selected_pages
    st.markdown(f"Ready to process **{len(pages)}** page(s).")

    with st.expander("⚙️ Pipeline Settings", expanded=False):
        s1, s2 = st.columns(2)
        with s1:
            interval = st.number_input("Station Interval (ft)", min_value=0.1,
                                       value=1.0, step=0.5, key="station_interval")
        with s2:
            method = st.selectbox("Interpolation", ["linear", "cubic"], index=0, key="interp_method")

    if st.button(f"🚀 Run Pipeline on {len(pages)} Page(s)", key="btn_run", use_container_width=True):
        report_dir = os.path.join(get_work_dir(), "reports")
        os.makedirs(report_dir, exist_ok=True)

        h_s = st.session_state.get("h_scale_override", 0.0)
        v_s = st.session_state.get("v_scale_override", 0.0)

        pipeline = VectorPipeline(
            station_interval=st.session_state.get("station_interval", 1.0),
            interpolation_method=st.session_state.get("interp_method", "linear"),
            output_dir=report_dir,
        )

        bar = st.progress(0, text="Starting...")
        result = pipeline.run(
            pdf_path=st.session_state.pdf_path,
            page_numbers=pages,
            h_scale_override=h_s if h_s > 0 else None,
            v_scale_override=v_s if v_s > 0 else None,
            progress_callback=lambda c, t, m: bar.progress(c / max(t, 1), text=m),
        )
        bar.progress(1.0, text="Done!")

        st.session_state.pipeline_result = result
        st.rerun()
else:
    result = st.session_state.pipeline_result
    ok = len(result.successful_pages)
    fail = len(result.failed_pages)

    if ok > 0:
        st.success(f"✅ **{ok}/{len(result.page_results)}** pages processed.")
    if fail > 0:
        st.warning(f"⚠️ **{fail}** page(s) had errors.")

st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Results
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("### Step 4 — Results")

if st.session_state.pipeline_result is None:
    st.info("Run the pipeline in Step 3.")
else:
    result = st.session_state.pipeline_result

    # aggregate volume cards
    if result.successful_pages:
        st.markdown("#### 📊 Aggregate Volumes")
        v1, v2, v3 = st.columns(3)
        with v1:
            st.markdown(f'<div class="vol-card cut"><div class="vol-value">'
                        f'{result.total_cut_volume_cy:,.1f}</div>'
                        '<div class="vol-unit">cu yd</div>'
                        '<div class="vol-label">Total Cut</div></div>', unsafe_allow_html=True)
        with v2:
            st.markdown(f'<div class="vol-card fill"><div class="vol-value">'
                        f'{result.total_fill_volume_cy:,.1f}</div>'
                        '<div class="vol-unit">cu yd</div>'
                        '<div class="vol-label">Total Fill</div></div>', unsafe_allow_html=True)
        with v3:
            net = result.total_cut_volume_cy - result.total_fill_volume_cy
            st.markdown(f'<div class="vol-card net"><div class="vol-value">'
                        f'{net:,.1f}</div>'
                        '<div class="vol-unit">cu yd</div>'
                        '<div class="vol-label">Net (Cut − Fill)</div></div>', unsafe_allow_html=True)
        st.markdown("")

    # per-page results
    for pr in result.page_results:
        if pr.success:
            with st.expander(f"📊 {pr.page_label} — ✅", expanded=(len(result.page_results) == 1)):
                if pr.earthwork:
                    pc1, pc2, pc3 = st.columns(3)
                    with pc1:
                        st.metric("Cut", f"{pr.earthwork.total_cut_volume_cy:,.2f} cu yd")
                    with pc2:
                        st.metric("Fill", f"{pr.earthwork.total_fill_volume_cy:,.2f} cu yd")
                    with pc3:
                        st.metric("Net", f"{pr.earthwork.net_volume_cy:,.2f} cu yd")

                if pr.plot_path and os.path.exists(pr.plot_path):
                    st.image(Image.open(pr.plot_path), use_container_width=True)

                if pr.validation:
                    for w in pr.validation.warnings:
                        st.warning(f"⚠️ {w}")
                    for e in pr.validation.errors:
                        st.error(f"❌ {e}")
                    if pr.validation.is_valid and not pr.validation.warnings:
                        st.success("✅ All checks passed.")

                if pr.earthwork:
                    with st.expander("📋 Station Data"):
                        st.dataframe(pr.earthwork.to_dataframe(), use_container_width=True, hide_index=True)
                    with st.expander("📋 Segment Volumes"):
                        st.dataframe(pr.earthwork.to_volume_dataframe(), use_container_width=True, hide_index=True)

                # downloads
                dl1, dl2, dl3 = st.columns(3)
                for col, path, label, mime, key_suffix in [
                    (dl1, pr.csv_report_path, "⬇️ CSV", "text/csv", "csv"),
                    (dl2, pr.json_report_path, "⬇️ JSON", "application/json", "json"),
                    (dl3, pr.plot_path, "⬇️ Plot", "image/png", "plot"),
                ]:
                    if path and os.path.exists(path):
                        with col:
                            with open(path, "rb") as f:
                                st.download_button(label, f.read(), os.path.basename(path),
                                                   mime, use_container_width=True,
                                                   key=f"dl_{key_suffix}_{pr.page_number}")

                with st.expander("🔧 Diagnostics"):
                    diag = {"classification": pr.classification, "workflow": pr.workflow,
                            "stage": pr.stage_reached}
                    if pr.scale:
                        diag["h_scale"] = f"{pr.scale.h_scale} ft/in"
                        diag["v_scale"] = f"{pr.scale.v_scale} ft/in"
                        diag["origin"] = (pr.scale.origin_x, pr.scale.origin_y)
                        diag["start_station"] = pr.scale.start_station
                        diag["base_elevation"] = pr.scale.base_elevation
                    if pr.profiles and pr.profiles.diagnostics:
                        diag.update(pr.profiles.diagnostics)
                    if pr.earthwork and pr.earthwork.diagnostics:
                        diag["earthwork"] = pr.earthwork.diagnostics
                    st.json(diag)

        else:
            with st.expander(f"❌ {pr.page_label} — Failed [{pr.stage_reached}]"):
                st.error(pr.error)


# ── Footer ────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("<p style='text-align:center; color:#64748b; font-size:.8rem;'>"
            "XDOT Contractor — Approach 2: Direct PDF Vector Extraction</p>",
            unsafe_allow_html=True)
