"""
Streamlit app — Cross-Section Vector Extraction Pipeline.
Upload a highway PDF → classify pages → split into stations →
extract profiles → compute cut/fill areas → Average End Area volumes.
"""

import streamlit as st
import os
import tempfile
import shutil
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from orchestrator import VectorPipeline
from pdf_classifier import PDFClassifier
import fitz as pymupdf  # PyMuPDF for rendering PDF pages as images


def render_pdf_region(pdf_path, page_number, y_top=None, y_bottom=None, dpi=150):
    """Render a region of a PDF page as a PIL Image.
    If y_top/y_bottom are given, crop to that vertical range.
    page_number is 1-indexed.
    """
    try:
        doc = pymupdf.open(pdf_path)
        page = doc[page_number - 1]
        if y_top is not None and y_bottom is not None:
            margin = 10
            clip = pymupdf.Rect(
                page.rect.x0, max(y_top - margin, page.rect.y0),
                page.rect.x1, min(y_bottom + margin, page.rect.y1),
            )
        else:
            clip = page.rect
        mat = pymupdf.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat, clip=clip)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        doc.close()
        return img
    except Exception:
        return None


st.set_page_config(
    page_title="XDOT Contractor — Road Quantity Analyzer",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded",
)

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

section[data-testid="stSidebar"] { background: linear-gradient(180deg,#0f172a,#1e293b) !important; }

.stButton > button {
    background: linear-gradient(135deg,#6366f1,#8b5cf6) !important;
    color: #fff !important; border: none !important; border-radius: 10px !important;
    padding: .6rem 1.5rem !important; font-weight: 600 !important;
}
.stDownloadButton > button {
    background: linear-gradient(135deg,#22c55e,#16a34a) !important;
    color: #fff !important; border: none !important; border-radius: 10px !important;
    font-weight: 600 !important;
}
hr { border-color: rgba(99,102,241,0.1) !important; margin: 1.5rem 0 !important; }
</style>
""", unsafe_allow_html=True)

# state
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


# sidebar
with st.sidebar:
    st.markdown("## 🛣️ Cross-Section Pipeline")

    if st.session_state.doc_analysis:
        da = st.session_state.doc_analysis
        st.markdown(f"**{da.total_pages}** pages | **{len(da.cross_section_pages)}** cross-section pages")

    if st.session_state.pipeline_result:
        pr = st.session_state.pipeline_result
        ok = len(pr.successful_stations)
        fail = len(pr.failed_stations)
        st.markdown(f"**{ok}** stations OK | **{fail}** failed")

    st.divider()
    if st.button("🔄 Reset", use_container_width=True):
        if st.session_state.work_dir and os.path.exists(st.session_state.work_dir):
            shutil.rmtree(st.session_state.work_dir, ignore_errors=True)
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()


# header
st.markdown("""
<div class="main-header">
    <div class="badge">CROSS-SECTION VECTOR EXTRACTION</div>
    <h1>🛣️ Road Quantity Analyzer</h1>
    <p>Extract cross-section profiles from CAD PDFs. Compute cut/fill areas per station and volumes via Average End Area.</p>
</div>""", unsafe_allow_html=True)


# ── Step 1: Upload ──────────────────────────────────────────────────────────

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
        if st.button("🔍 Analyze PDF", key="btn_classify", use_container_width=True):
            with st.spinner("Analyzing..."):
                classifier = PDFClassifier(st.session_state.pdf_path)
                st.session_state.doc_analysis = classifier.analyze()
            st.rerun()
    else:
        da = st.session_state.doc_analysis
        st.success(f"✅ **{da.total_pages}** pages analyzed")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f'<div class="stat-box"><div class="stat-value">{da.total_pages}</div>'
                        '<div class="stat-label">Total Pages</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="stat-box"><div class="stat-value">{len(da.vector_pages)}</div>'
                        '<div class="stat-label">Vector</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="stat-box"><div class="stat-value">{len(da.cross_section_pages)}</div>'
                        '<div class="stat-label">Cross-Section</div></div>', unsafe_allow_html=True)
        with c4:
            st.markdown(f'<div class="stat-box"><div class="stat-value">{len(da.processable_pages)}</div>'
                        '<div class="stat-label">Processable</div></div>', unsafe_allow_html=True)

        with st.expander("📋 Page Classification", expanded=False):
            classifier = PDFClassifier(st.session_state.pdf_path)
            st.dataframe(pd.DataFrame(classifier.to_dict_list(da)),
                         use_container_width=True, hide_index=True)

st.divider()


# ── Step 2: Select Pages ─────────────────────────────────────────────────────

st.markdown("### Step 2 — Select Pages")

if not st.session_state.doc_analysis:
    st.info("Upload and analyze a PDF first.")
else:
    da = st.session_state.doc_analysis
    processable = da.processable_pages

    if not processable:
        st.warning("No processable cross-section pages found.")
    else:
        page_nums = [p.page_number for p in processable]
        st.markdown(f"**{len(processable)}** cross-section pages ready.")

        selected = st.multiselect("Select pages", page_nums,
                                  default=page_nums, key="page_select")
        st.session_state.selected_pages = selected

st.divider()


# ── Step 3: Run ──────────────────────────────────────────────────────────────

st.markdown("### Step 3 — Run Pipeline")

if not st.session_state.selected_pages:
    st.info("Select pages in Step 2.")
elif st.session_state.pipeline_result is None:
    pages = st.session_state.selected_pages
    st.markdown(f"Ready to process **{len(pages)}** page(s).")

    with st.expander("⚙️ Settings", expanded=False):
        s1, s2 = st.columns(2)
        with s1:
            interval = st.number_input("Offset Interval (ft)", min_value=0.1,
                                       value=1.0, step=0.5, key="offset_interval")
        with s2:
            method = st.selectbox("Interpolation", ["linear", "cubic"], index=0, key="interp_method")

    if st.button(f"🚀 Run on {len(pages)} Pages", key="btn_run", use_container_width=True):
        report_dir = os.path.join(get_work_dir(), "reports")
        os.makedirs(report_dir, exist_ok=True)

        pipeline = VectorPipeline(
            offset_interval=st.session_state.get("offset_interval", 1.0),
            interpolation_method=st.session_state.get("interp_method", "linear"),
            output_dir=report_dir,
        )

        bar = st.progress(0, text="Starting...")
        result = pipeline.run(
            pdf_path=st.session_state.pdf_path,
            page_numbers=pages,
            progress_callback=lambda c, t, m: bar.progress(min(c / max(t, 1), 1.0), text=m),
        )
        bar.progress(1.0, text="Done!")

        st.session_state.pipeline_result = result
        st.rerun()
else:
    result = st.session_state.pipeline_result
    ok = len(result.successful_stations)
    fail = len(result.failed_stations)
    if ok > 0:
        st.success(f"✅ **{ok}** stations processed successfully.")
    if fail > 0:
        st.warning(f"⚠️ **{fail}** station(s) failed.")

st.divider()


# ── Step 4: Results ──────────────────────────────────────────────────────────

st.markdown("### Step 4 — Results")

if st.session_state.pipeline_result is None:
    st.info("Run the pipeline in Step 3.")
else:
    result = st.session_state.pipeline_result

    # volume cards
    if result.earthwork and len(result.earthwork.station_areas) >= 2:
        ew = result.earthwork
        st.markdown("#### 📊 Aggregate Volumes (Average End Area)")
        v1, v2, v3 = st.columns(3)
        with v1:
            st.markdown(f'<div class="vol-card cut"><div class="vol-value">'
                        f'{ew.total_cut_volume_cy:,.1f}</div>'
                        '<div class="vol-unit">cu yd</div>'
                        '<div class="vol-label">Total Cut</div></div>', unsafe_allow_html=True)
        with v2:
            st.markdown(f'<div class="vol-card fill"><div class="vol-value">'
                        f'{ew.total_fill_volume_cy:,.1f}</div>'
                        '<div class="vol-unit">cu yd</div>'
                        '<div class="vol-label">Total Fill</div></div>', unsafe_allow_html=True)
        with v3:
            st.markdown(f'<div class="vol-card net"><div class="vol-value">'
                        f'{ew.net_volume_cy:,.1f}</div>'
                        '<div class="vol-unit">cu yd</div>'
                        '<div class="vol-label">Net (Cut − Fill)</div></div>', unsafe_allow_html=True)

        st.markdown("")

        # aggregate plot
        if result.plot_path and os.path.exists(result.plot_path):
            st.image(Image.open(result.plot_path), use_container_width=True)

        # area table
        with st.expander("📋 Cut/Fill Areas per Station", expanded=True):
            st.dataframe(ew.to_area_dataframe(), use_container_width=True, hide_index=True)

        # volume segments table
        with st.expander("📋 Segment Volumes (Average End Area)", expanded=False):
            st.dataframe(ew.to_volume_dataframe(), use_container_width=True, hide_index=True)

        # downloads
        dl1, dl2 = st.columns(2)
        if result.csv_report_path and os.path.exists(result.csv_report_path):
            with dl1:
                with open(result.csv_report_path, "rb") as f:
                    st.download_button("⬇️ Download CSV", f.read(),
                                       os.path.basename(result.csv_report_path),
                                       "text/csv", use_container_width=True, key="dl_csv")
        if result.json_report_path and os.path.exists(result.json_report_path):
            with dl2:
                with open(result.json_report_path, "rb") as f:
                    st.download_button("⬇️ Download JSON", f.read(),
                                       os.path.basename(result.json_report_path),
                                       "application/json", use_container_width=True, key="dl_json")

    elif result.earthwork and len(result.earthwork.station_areas) == 1:
        st.warning("Only 1 station found — need at least 2 for volume calculation.")

    st.divider()

    # per-station details
    st.markdown("#### 📐 Per-Station Details")

    for sr in result.station_results:
        if sr.success:
            icon = "✅"
            title = f"{icon} STA {sr.station_label} (Page {sr.page_number})"
            with st.expander(title, expanded=False):
                if sr.station_area:
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.metric("Cut Area", f"{sr.station_area.cut_area:,.1f} sq ft")
                    with c2:
                        st.metric("Fill Area", f"{sr.station_area.fill_area:,.1f} sq ft")
                    with c3:
                        st.metric("Net Area", f"{sr.station_area.net_area:,.1f} sq ft")

                # validation plot
                if sr.normalized:
                    fig, ax = plt.subplots(figsize=(12, 5))

                    offsets = sr.normalized.stations
                    ex = sr.normalized.existing_elevations
                    pr = sr.normalized.proposed_elevations
                    diff = pr - ex

                    ax.plot(offsets, ex, color="#6b7280", lw=1.8, ls="--",
                            label="Existing Ground", zorder=3)
                    ax.plot(offsets, pr, color="#3b82f6", lw=2.2, ls="-",
                            label="Proposed Grade", zorder=3)

                    ax.fill_between(offsets, ex, pr, where=(diff < 0), interpolate=True,
                                    color="#ef4444", alpha=0.25, label="Cut", zorder=2)
                    ax.fill_between(offsets, ex, pr, where=(diff > 0), interpolate=True,
                                    color="#22c55e", alpha=0.25, label="Fill", zorder=2)

                    # centerline marker
                    ax.axvline(x=0, color="#94a3b8", lw=0.8, ls=":", alpha=0.6)
                    ax.text(0, ax.get_ylim()[1], " CL", fontsize=8, color="#94a3b8",
                            va="top", ha="left")

                    ax.set_xlabel("Offset from Centerline (ft)", fontsize=11, fontweight="600")
                    ax.set_ylabel("Elevation (ft)", fontsize=11, fontweight="600")
                    ax.set_title(f"Validation Plot — STA {sr.station_label}",
                                 fontsize=13, fontweight="700", pad=10)
                    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
                    ax.grid(True, alpha=0.2, lw=0.5)
                    ax.set_facecolor("#f8fafc")

                    # area annotation
                    if sr.station_area:
                        note = (
                            f"Cut: {sr.station_area.cut_area:,.1f} sq ft  |  "
                            f"Fill: {sr.station_area.fill_area:,.1f} sq ft  |  "
                            f"Net: {sr.station_area.net_area:,.1f} sq ft"
                        )
                        ax.text(0.5, -0.13, note, transform=ax.transAxes,
                                ha="center", fontsize=10, color="#475569", fontweight="500")

                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

                if sr.station_area and sr.station_area.diagnostics:
                    with st.expander("🔍 Step-by-Step Pipeline Diagnostics", expanded=False):
                        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                            "1. Region Splitting",
                            "2. Vector Extraction",
                            "3. Profile ID",
                            "4. Coordinate Xform",
                            "5. Grid Alignment",
                            "6. Earthwork Calc"
                        ])

                        # --- Tab 1: Region Splitting ---
                        with tab1:
                            st.markdown("#### Step 1: Station Region & Text Label Detection")
                            c1, c2 = st.columns(2)
                            with c1:
                                st.markdown(f"""
                                **Region Boundaries:**
                                * **Station Label:** `{sr.station_label}`
                                * **Station (ft):** `{sr.station_ft:,.1f}`
                                * **PDF Page:** `{sr.page_number}`
                                * **Drawing Sheet:** `{sr.drawing_number or 'N/A'}`
                                * **Vertical Bounds:** `{sr.region.y_top:.1f}` to `{sr.region.y_bottom:.1f}` pt
                                """)
                            with c2:
                                st.markdown("**Elevation Labels Found on Left Margin (Y):**")
                                if sr.region.elevations:
                                    st.write(", ".join(map(str, sorted(sr.region.elevations))))
                                else:
                                    st.write("*None*")

                                st.markdown("**Offset Tick Labels Found on X-Axis Row:**")
                                if sr.region.offsets:
                                    st.write(", ".join(map(str, sorted(sr.region.offsets))))
                                else:
                                    st.write("*None*")

                        # --- Tab 2: Vector Extraction ---
                        with tab2:
                            st.markdown("#### Step 2: Vector Geometry Extraction")
                            if sr.profiles and sr.profiles.diagnostics:
                                diag_data = sr.profiles.diagnostics
                                st.markdown(f"""
                                **Extraction Summary:**
                                * **Total Drawing Paths in Region Bounds:** `{diag_data.get('total_paths', 0)}`
                                * **Grid Line Segments Removed:** `{diag_data.get('grid_lines_removed', 0)}`
                                * **Paths Before Merge:** `{diag_data.get('pre_merge_paths', '—')}`
                                * **After Strict Merge (same color):** `{diag_data.get('post_merge_strict', '—')}`
                                * **After Tolerant Merge (any color):** `{diag_data.get('post_merge_tolerant', '—')}`
                                * **Total Segments Merged:** `{diag_data.get('paths_merged', 0)}`
                                * **Noise Paths Filtered (too short/small):** `{diag_data.get('noise_removed', 0)}`
                                * **Profile Candidates Remaining:** `{diag_data.get('candidates_remaining', 0)}`
                                * **Scored Candidates (passed coverage):** `{diag_data.get('scored_candidates', 0)}`
                                """)
                            else:
                                st.info("No extraction diagnostics available.")

                        # --- Tab 3: Profile ID ---
                        with tab3:
                            st.markdown("#### Step 3: Profile Identification & Scoring")

                            # ── Classification confidence indicator ──
                            _diag = sr.profiles.diagnostics if sr.profiles else {}
                            _conf = _diag.get("classification_confidence", 0)
                            _rule = _diag.get("classification_rule", "—")
                            _signals = _diag.get("classification_signals", {})

                            if _conf >= 0.8:
                                _conf_color = "#22c55e"
                                _conf_label = "HIGH"
                            elif _conf >= 0.5:
                                _conf_color = "#f59e0b"
                                _conf_label = "MEDIUM"
                            else:
                                _conf_color = "#ef4444"
                                _conf_label = "LOW"

                            st.markdown(f"""
                            <div style="background: linear-gradient(135deg, #1e293b, #0f172a);
                                        border: 2px solid {_conf_color}; border-radius: 12px;
                                        padding: 1rem 1.5rem; margin-bottom: 1rem;">
                                <div style="display:flex; align-items:center; gap: 1rem; flex-wrap: wrap;">
                                    <span style="background:{_conf_color}; color:#fff; padding:3px 12px;
                                                 border-radius:20px; font-size:.75rem; font-weight:700;">
                                        {_conf_label} CONFIDENCE ({_conf:.0%})
                                    </span>
                                    <span style="color:#94a3b8; font-size:.85rem;">
                                        Method: <b style="color:#f8fafc;">{_rule}</b>
                                    </span>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                            # ── Classification signal votes ──
                            _votes = _diag.get("classification_votes", {})
                            if _votes:
                                st.markdown("**Signal Votes:**")
                                vote_rows = []
                                for sig_name, sig_val in _votes.items():
                                    direction = "→ Candidate 1 is EG" if sig_val > 0 else "→ Candidate 2 is EG"
                                    vote_rows.append({
                                        "Signal": sig_name,
                                        "Vote": f"{sig_val:+.1f}",
                                        "Interpretation": direction,
                                    })
                                st.dataframe(pd.DataFrame(vote_rows), use_container_width=True, hide_index=True)

                            # ── Profile summary table with enhanced metadata ──
                            prof_data = []
                            if sr.profiles and sr.profiles.existing_ground:
                                eg = sr.profiles.existing_ground
                                eg_dr = eg.dash_segments / max(eg.dash_segments + eg.solid_segments, 1)
                                prof_data.append({
                                    "Profile": "Existing Ground",
                                    "Points": eg.point_count,
                                    "Length (pt)": round(eg.length, 1),
                                    "Width (pt)": round(eg.width, 1),
                                    "Score": round(sr.profiles.existing_ground_score, 2),
                                    "Dash Ratio": f"{eg_dr:.0%}",
                                    "Segments": eg.segment_count,
                                    "Stroke W": round(eg.stroke_width, 2),
                                    "Color": str(eg.color),
                                })
                            if sr.profiles and sr.profiles.proposed_grade:
                                pg = sr.profiles.proposed_grade
                                pg_dr = pg.dash_segments / max(pg.dash_segments + pg.solid_segments, 1)
                                prof_data.append({
                                    "Profile": "Proposed Grade",
                                    "Points": pg.point_count,
                                    "Length (pt)": round(pg.length, 1),
                                    "Width (pt)": round(pg.width, 1),
                                    "Score": round(sr.profiles.proposed_grade_score, 2),
                                    "Dash Ratio": f"{pg_dr:.0%}",
                                    "Segments": pg.segment_count,
                                    "Stroke W": round(pg.stroke_width, 2),
                                    "Color": str(pg.color),
                                })
                            if prof_data:
                                st.dataframe(pd.DataFrame(prof_data), use_container_width=True, hide_index=True)

                            # ── Overlay comparison: polylines ON TOP of PDF raster ──
                            st.markdown("---")
                            st.markdown("**🔍 Cross-Verification Overlay (Extracted Lines on PDF)**")
                            if sr.region and st.session_state.get("pdf_path"):
                                _overlay_img = render_pdf_region(
                                    st.session_state.pdf_path,
                                    sr.page_number,
                                    sr.region.y_top,
                                    sr.region.y_bottom,
                                    dpi=150,
                                )
                                if _overlay_img:
                                    _dpi = 150
                                    _scale = _dpi / 72.0
                                    _margin = 10
                                    _y_off = max(sr.region.y_top - _margin, 0)
                                    _page_x0 = 0  # x0 for the clip is always 0

                                    fig_ov, ax_ov = plt.subplots(figsize=(14, 6))
                                    ax_ov.imshow(_overlay_img, aspect="auto",
                                                 extent=[0, _overlay_img.width, _overlay_img.height, 0])

                                    # plot EG polyline on overlay (use original path order, not X-sorted)
                                    if sr.profiles and sr.profiles.existing_ground:
                                        eg_pts = sr.profiles.existing_ground.points
                                        if eg_pts:
                                            ox = [(p[0] - _page_x0) * _scale for p in eg_pts]
                                            oy = [(p[1] - _y_off) * _scale for p in eg_pts]
                                            ax_ov.plot(ox, oy, color="#ff4444", lw=2.5, ls="--",
                                                       label="Existing Ground (extracted)", zorder=5, alpha=0.85)

                                    # plot PG polyline on overlay (use original path order, not X-sorted)
                                    if sr.profiles and sr.profiles.proposed_grade:
                                        pg_pts = sr.profiles.proposed_grade.points
                                        if pg_pts:
                                            ox = [(p[0] - _page_x0) * _scale for p in pg_pts]
                                            oy = [(p[1] - _y_off) * _scale for p in pg_pts]
                                            ax_ov.plot(ox, oy, color="#00bbff", lw=2.5, ls="-",
                                                       label="Proposed Grade (extracted)", zorder=5, alpha=0.85)

                                    ax_ov.set_title(f"Overlay Verification — STA {sr.station_label}",
                                                    fontsize=13, fontweight="700")
                                    ax_ov.legend(fontsize=9, loc="upper right",
                                                 facecolor="white", edgecolor="#ccc", framealpha=0.9)
                                    ax_ov.axis("off")
                                    plt.tight_layout()
                                    st.pyplot(fig_ov)
                                    plt.close(fig_ov)
                                else:
                                    st.warning("Could not render PDF region for overlay.")
                            else:
                                st.info("No region data or PDF path available for overlay.")

                            # ── All candidates overlay ──
                            st.markdown("---")
                            st.markdown("**📊 All Scored Candidates (PDF Coordinate Space)**")
                            if sr.profiles and sr.profiles.scored_list:
                                fig_ac, ax_ac = plt.subplots(figsize=(14, 6))
                                _has_bg = False

                                # render background PDF if available
                                if sr.region and st.session_state.get("pdf_path"):
                                    _bg_img = render_pdf_region(
                                        st.session_state.pdf_path,
                                        sr.page_number,
                                        sr.region.y_top,
                                        sr.region.y_bottom,
                                        dpi=100,
                                    )
                                    if _bg_img:
                                        _s100 = 100 / 72.0
                                        _m = 10
                                        _yo = max(sr.region.y_top - _m, 0)
                                        ax_ac.imshow(_bg_img, aspect="auto", alpha=0.3,
                                                     extent=[0, _bg_img.width, _bg_img.height, 0])
                                        _has_bg = True

                                cand_colors = plt.cm.Set1(np.linspace(0, 1, max(len(sr.profiles.scored_list), 1)))
                                for k, (cpath, cscore) in enumerate(sr.profiles.scored_list[:8]):
                                    # Use original path order (not X-sorted) to avoid zigzag
                                    cx = [p[0] for p in cpath.points]
                                    cy = [p[1] for p in cpath.points]
                                    _ls = "--" if cpath.dashes and str(cpath.dashes).strip() not in ("[] 0", "[]") else "-"
                                    _lbl = f"#{k+1} score={cscore:.1f}"
                                    _dr = cpath.dash_segments / max(cpath.dash_segments + cpath.solid_segments, 1)
                                    _lbl += f" dash={_dr:.0%}"
                                    if cpath is (sr.profiles.existing_ground if sr.profiles else None):
                                        _lbl += " ★EG"
                                    elif cpath is (sr.profiles.proposed_grade if sr.profiles else None):
                                        _lbl += " ★PG"
                                    ax_ac.plot(cx, cy, color=cand_colors[k], lw=1.8, ls=_ls,
                                               alpha=0.85, label=_lbl, zorder=3 + k)

                                ax_ac.set_xlabel("PDF X (pt)", fontsize=10)
                                ax_ac.set_ylabel("PDF Y (pt)", fontsize=10)
                                ax_ac.set_title(f"All Scored Candidates — STA {sr.station_label}",
                                                fontsize=12, fontweight="600")
                                ax_ac.legend(fontsize=7, loc="best", framealpha=0.9)
                                ax_ac.grid(True, alpha=0.15)
                                # Only invert Y if no background image
                                # (imshow with extent=[0,w,h,0] already inverts)
                                if not _has_bg:
                                    ax_ac.invert_yaxis()
                                ax_ac.set_facecolor("#fefce8")
                                plt.tight_layout()
                                st.pyplot(fig_ac)
                                plt.close(fig_ac)

                            # ── Raw polyline plot (legacy) ──
                            st.markdown("---")
                            st.markdown("**Raw Candidate Polylines (PDF Coordinate Space)**")
                            fig2, ax2 = plt.subplots(figsize=(12, 5))

                            if sr.profiles and sr.profiles.existing_ground:
                                eg_pts = sr.profiles.existing_ground_points
                                if eg_pts:
                                    xs = [p[0] for p in eg_pts]
                                    ys = [p[1] for p in eg_pts]
                                    ax2.plot(xs, ys, color="#6b7280", lw=2.0, ls="--",
                                             label=f"Existing Ground ({len(eg_pts)} pts, score={sr.profiles.existing_ground_score:.1f})",
                                             zorder=3)

                            if sr.profiles and sr.profiles.proposed_grade:
                                pg_pts = sr.profiles.proposed_grade_points
                                if pg_pts:
                                    xs = [p[0] for p in pg_pts]
                                    ys = [p[1] for p in pg_pts]
                                    ax2.plot(xs, ys, color="#3b82f6", lw=2.0, ls="-",
                                             label=f"Proposed Grade ({len(pg_pts)} pts, score={sr.profiles.proposed_grade_score:.1f})",
                                             zorder=3)

                            ax2.set_xlabel("PDF X (pt)", fontsize=10)
                            ax2.set_ylabel("PDF Y (pt)", fontsize=10)
                            ax2.set_title(f"Raw PDF Polylines — STA {sr.station_label}", fontsize=12, fontweight="600")
                            ax2.legend(fontsize=8, loc="best")
                            ax2.grid(True, alpha=0.2)
                            ax2.invert_yaxis()  # PDF Y is top-down
                            ax2.set_facecolor("#fefce8")
                            plt.tight_layout()
                            st.pyplot(fig2)
                            plt.close(fig2)

                            # top candidates table
                            if sr.profiles and sr.profiles.diagnostics.get("top_candidates"):
                                st.markdown("**Top 10 Scored Candidates**")
                                st.dataframe(pd.DataFrame(sr.profiles.diagnostics["top_candidates"]),
                                             use_container_width=True, hide_index=True)

                            # ── Original PDF Region Image for comparison ──
                            st.markdown("---")
                            st.markdown("**📄 Original PDF Region (for visual comparison)**")
                            if sr.region and st.session_state.get("pdf_path"):
                                orig_img = render_pdf_region(
                                    st.session_state.pdf_path,
                                    sr.page_number,
                                    sr.region.y_top,
                                    sr.region.y_bottom,
                                )
                                if orig_img:
                                    st.image(orig_img, caption=f"Original PDF — STA {sr.station_label} (Page {sr.page_number})",
                                             use_container_width=True)
                                else:
                                    st.warning("Could not render PDF region image.")
                            else:
                                st.info("No region data or PDF path available.")

                        # --- Tab 4: Coordinate Xform ---
                        with tab4:
                            st.markdown("#### Step 4: Scale Calibration & Coordinate Transformation")
                            if sr.scale:
                                c1, c2 = st.columns(2)
                                with c1:
                                    st.markdown(f"**Horizontal X-Mapping (PDF x ➔ Offset ft)**")
                                    st.markdown(f"**Equation:** `Offset = x * {sr.scale.x_slope:.5f} + {sr.scale.x_intercept:.2f}`")
                                    if sr.scale.x_ticks:
                                        x_ticks_df = pd.DataFrame(sr.scale.x_ticks, columns=["PDF X (pt)", "Offset (ft)"])
                                        st.dataframe(x_ticks_df, use_container_width=True, hide_index=True)
                                    else:
                                        st.write("*No calibration X-ticks found.*")
                                with c2:
                                    st.markdown(f"**Vertical Y-Mapping (PDF y ➔ Elevation ft)**")
                                    st.markdown(f"**Equation:** `Elevation = y * {sr.scale.y_slope:.5f} + {sr.scale.y_intercept:.2f}`")
                                    if sr.scale.y_ticks:
                                        y_ticks_df = pd.DataFrame(sr.scale.y_ticks, columns=["PDF Y (pt)", "Elevation (ft)"])
                                        st.dataframe(y_ticks_df, use_container_width=True, hide_index=True)
                                    else:
                                        st.write("*No calibration Y-ticks found.*")

                                st.markdown("---")
                                st.markdown("**Transformed Coordinates Bounding Boxes:**")
                                tx_data = []
                                if sr.existing_transformed:
                                    ex = sr.existing_transformed
                                    tx_data.append({
                                        "Profile": "Existing Ground",
                                        "Min Offset (ft)": round(ex.station_range[0], 2),
                                        "Max Offset (ft)": round(ex.station_range[1], 2),
                                        "Min Elev (ft)": round(ex.elevation_range[0], 2),
                                        "Max Elev (ft)": round(ex.elevation_range[1], 2),
                                        "Points": len(ex.offsets),
                                    })
                                if sr.proposed_transformed:
                                    pr = sr.proposed_transformed
                                    tx_data.append({
                                        "Profile": "Proposed Grade",
                                        "Min Offset (ft)": round(pr.station_range[0], 2),
                                        "Max Offset (ft)": round(pr.station_range[1], 2),
                                        "Min Elev (ft)": round(pr.elevation_range[0], 2),
                                        "Max Elev (ft)": round(pr.elevation_range[1], 2),
                                        "Points": len(pr.offsets),
                                    })
                                if tx_data:
                                    st.dataframe(pd.DataFrame(tx_data), use_container_width=True, hide_index=True)
                            else:
                                st.info("No scale calibration data available.")

                        # --- Tab 5: Grid Alignment ---
                        with tab5:
                            st.markdown("#### Step 5: Horizontal Grid Normalization & Alignment")
                            if sr.normalized:
                                nd = sr.normalized.diagnostics
                                eg_r = nd.get("existing_range", sr.normalized.station_range)
                                pg_r = nd.get("proposed_range", sr.normalized.station_range)
                                ov_r = nd.get("overlap_range", sr.normalized.station_range)
                                st.markdown(f"""
                                **Profile Ranges (engineering ft):**
                                * **Existing Ground:** `{eg_r[0]:,.1f}` to `{eg_r[1]:,.1f}` ft (span: `{eg_r[1]-eg_r[0]:,.1f}`)
                                * **Proposed Grade:** `{pg_r[0]:,.1f}` to `{pg_r[1]:,.1f}` ft (span: `{pg_r[1]-pg_r[0]:,.1f}`)
                                * **Overlap (earthwork zone):** `{ov_r[0]:,.1f}` to `{ov_r[1]:,.1f}` ft (width: `{nd.get('overlap_width', ov_r[1]-ov_r[0]):,.1f}`)

                                **Grid Details:**
                                * **Grid Step Interval:** `{sr.normalized.station_interval:.2f}` ft
                                * **Total Grid Points:** `{len(sr.normalized.stations)}`
                                * **Interpolation Method:** `{nd.get('method', 'linear')}`
                                """)
                            else:
                                st.info("No normalization data available.")

                        # --- Tab 6: Earthwork Calc ---
                        with tab6:
                            st.markdown("#### Step 6: Earthwork Calculations (Trapezoidal Rule)")
                            if sr.station_area:
                                c1, c2, c3 = st.columns(3)
                                with c1:
                                    st.metric("Cut Area (trapz)", f"{sr.station_area.cut_area:,.2f} sq ft")
                                with c2:
                                    st.metric("Fill Area (trapz)", f"{sr.station_area.fill_area:,.2f} sq ft")
                                with c3:
                                    st.metric("Net Area", f"{sr.station_area.net_area:,.2f} sq ft")

                                # max bounds
                                max_cut = sr.station_area.diagnostics.get("max_cut_depth", 0.0)
                                max_fill = sr.station_area.diagnostics.get("max_fill_height", 0.0)
                                st.write(f"**Max Cut Depth:** `{max_cut:.2f} ft`  |  **Max Fill Height:** `{max_fill:.2f} ft`")

                                if sr.normalized:
                                    st.markdown("**Aligned Point-by-Point Data Table:**")
                                    offsets = sr.normalized.stations
                                    ex_elevs = sr.normalized.existing_elevations
                                    pr_elevs = sr.normalized.proposed_elevations
                                    diff = pr_elevs - ex_elevs

                                    align_rows = []
                                    for o, ex, pr, d in zip(offsets, ex_elevs, pr_elevs, diff):
                                        if d < 0:
                                            ew_type = "Cut"
                                            depth_val = abs(d)
                                        elif d > 0:
                                            ew_type = "Fill"
                                            depth_val = d
                                        else:
                                            ew_type = "Match"
                                            depth_val = 0.0
                                        
                                        align_rows.append({
                                            "Offset (ft)": round(o, 1),
                                            "Existing Ground (ft)": round(ex, 2),
                                            "Proposed Grade (ft)": round(pr, 2),
                                            "Difference (ft)": round(d, 2),
                                            "Type": ew_type,
                                            "Depth/Height (ft)": round(depth_val, 2),
                                        })
                                    
                                    st.dataframe(pd.DataFrame(align_rows), use_container_width=True, hide_index=True)
                            else:
                                st.info("No earthwork area calculation data available.")
        else:
            title = f"❌ STA {sr.station_label} (Page {sr.page_number}) — {sr.stage_reached}"
            with st.expander(title, expanded=True):
                st.error(sr.error)

                # Show all available debugging information for the failed station
                st.markdown("---")
                st.markdown("##### 🔍 Failure Diagnostics")

                # ── Region info ──
                if sr.region:
                    st.markdown(f"""
                    **Region Info:**
                    * **Station:** `{sr.station_label}` ({sr.station_ft:,.0f} ft)
                    * **Page:** {sr.page_number} | Drawing: `{sr.drawing_number or 'N/A'}`
                    * **Y bounds:** `{sr.region.y_top:.1f}` — `{sr.region.y_bottom:.1f}` pt
                    * **Raw paths extracted:** `{sr.raw_paths_count}`
                    """)

                # ── Profile identification diagnostics ──
                if sr.profiles and sr.profiles.diagnostics:
                    diag = sr.profiles.diagnostics

                    st.markdown("##### Pipeline Filtering Breakdown")
                    # Funnel visualization
                    funnel_data = {
                        "Stage": [
                            "1. Raw Paths in Region",
                            "2. After Grid Removal",
                            "3. After Path Merging",
                            "4. After Noise Removal",
                            "5. After Filled Removal",
                            "6. Passed Coverage Scoring",
                        ],
                        "Count": [
                            diag.get("total_paths", 0),
                            diag.get("after_grid_filter", diag.get("total_paths", 0) - diag.get("grid_lines_removed", 0)),
                            diag.get("post_merge_paths", "—"),
                            diag.get("candidates_remaining", 0) + diag.get("filled_removed", 0),
                            diag.get("candidates_remaining", 0),
                            diag.get("scored_candidates", 0),
                        ],
                        "Removed": [
                            "—",
                            f"-{diag.get('grid_lines_removed', 0)} grid lines",
                            f"-{diag.get('paths_merged', 0)} (merged into fewer)",
                            f"-{diag.get('noise_removed', 0)} noise",
                            f"-{diag.get('filled_removed', 0)} filled",
                            f"-{diag.get('rejected_by_coverage', 0)} below threshold",
                        ],
                    }
                    st.dataframe(pd.DataFrame(funnel_data), use_container_width=True, hide_index=True)

                    # Failure reason
                    if diag.get("failure_reason"):
                        st.warning(f"**Root Cause:** {diag['failure_reason']}")

                    if diag.get("classification_rule"):
                        st.info(f"**Classification Rule Applied:** {diag['classification_rule']}")

                    # Rejected paths details
                    if diag.get("rejected_details"):
                        st.markdown("##### Paths Rejected by Coverage Threshold")
                        st.markdown(f"*Minimum required: `{0.08:.0%}` of page width*")
                        st.dataframe(pd.DataFrame(diag["rejected_details"]),
                                     use_container_width=True, hide_index=True)

                    # Top candidates (if any made it through)
                    if diag.get("top_candidates"):
                        st.markdown("##### Top Scored Candidates")
                        st.dataframe(pd.DataFrame(diag["top_candidates"]),
                                     use_container_width=True, hide_index=True)

                    # Plot ALL remaining paths so user can visually inspect
                    if sr.profiles.all_after_filter:
                        st.markdown("##### All Surviving Paths After Filtering (PDF coordinates)")
                        fig_dbg, ax_dbg = plt.subplots(figsize=(12, 5))

                        colors = plt.cm.tab20(np.linspace(0, 1, max(len(sr.profiles.all_after_filter), 1)))
                        for k, path in enumerate(sr.profiles.all_after_filter[:30]):
                            xs = [p[0] for p in path.points]
                            ys = [p[1] for p in path.points]
                            ls = "--" if path.dashes else "-"
                            ax_dbg.plot(xs, ys, color=colors[k % len(colors)], lw=1.2, ls=ls,
                                        alpha=0.8, label=f"#{path.path_id} ({path.point_count}pts)" if k < 8 else None)

                        ax_dbg.set_xlabel("PDF X (pt)")
                        ax_dbg.set_ylabel("PDF Y (pt)")
                        ax_dbg.set_title(f"All Filtered Paths — STA {sr.station_label} (FAILED)", fontweight="600")
                        if len(sr.profiles.all_after_filter) <= 8:
                            ax_dbg.legend(fontsize=7, loc="best")
                        ax_dbg.grid(True, alpha=0.2)
                        ax_dbg.invert_yaxis()
                        ax_dbg.set_facecolor("#fff1f2")
                        plt.tight_layout()
                        st.pyplot(fig_dbg)
                        plt.close(fig_dbg)

                    # ── Original PDF Region Image for comparison ──
                    if sr.region and st.session_state.get("pdf_path"):
                        st.markdown("---")
                        st.markdown("##### 📄 Original PDF Region (for visual comparison)")
                        orig_img = render_pdf_region(
                            st.session_state.pdf_path,
                            sr.page_number,
                            sr.region.y_top,
                            sr.region.y_bottom,
                        )
                        if orig_img:
                            st.image(orig_img, caption=f"Original PDF — STA {sr.station_label} (Page {sr.page_number})",
                                     use_container_width=True)
                        else:
                            st.warning("Could not render PDF region image.")
                elif sr.stage_reached == "extraction":
                    st.info(f"Failure occurred at extraction — {sr.raw_paths_count} paths were found in the region bounds.")

    # failed stations summary
    if result.failed_stations:
        st.divider()
        st.markdown("#### ⚠️ Failed Stations")
        for sr in result.failed_stations:
            st.error(f"STA {sr.station_label} (Page {sr.page_number}): {sr.error}")


# footer
st.markdown("---")
st.markdown("<p style='text-align:center; color:#64748b; font-size:.8rem;'>"
            "XDOT Contractor — Cross-Section Vector Extraction Pipeline</p>",
            unsafe_allow_html=True)