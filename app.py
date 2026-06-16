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
                    with st.expander("🔧 Diagnostics & Profile Verification"):
                        # profile summary metrics
                        st.markdown("**Profile Selection Summary**")
                        prof_data = []
                        if sr.profiles and sr.profiles.existing_ground:
                            eg = sr.profiles.existing_ground
                            prof_data.append({
                                "Profile": "Existing Ground",
                                "Points": eg.point_count,
                                "Length (pt)": round(eg.length, 1),
                                "Width (pt)": round(eg.width, 1),
                                "Height (pt)": round(eg.height, 1),
                                "Score": round(sr.profiles.existing_ground_score, 2),
                                "Dashes": str(eg.dashes) if eg.dashes else "None (solid)",
                                "Stroke Width": round(eg.stroke_width, 2),
                                "Color": str(eg.color),
                            })
                        if sr.profiles and sr.profiles.proposed_grade:
                            pg = sr.profiles.proposed_grade
                            prof_data.append({
                                "Profile": "Proposed Grade",
                                "Points": pg.point_count,
                                "Length (pt)": round(pg.length, 1),
                                "Width (pt)": round(pg.width, 1),
                                "Height (pt)": round(pg.height, 1),
                                "Score": round(sr.profiles.proposed_grade_score, 2),
                                "Dashes": str(pg.dashes) if pg.dashes else "None (solid)",
                                "Stroke Width": round(pg.stroke_width, 2),
                                "Color": str(pg.color),
                            })
                        if prof_data:
                            st.dataframe(pd.DataFrame(prof_data), use_container_width=True, hide_index=True)

                        # raw polyline plot (PDF coordinates)
                        st.markdown("**Raw Candidate Polylines (PDF Coordinates)**")
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

                        # region/scale info
                        st.markdown("**Scale & Region**")
                        info = {
                            "station": sr.station_label,
                            "station_ft": sr.station_ft,
                            "page": sr.page_number,
                            "drawing": sr.drawing_number,
                        }
                        if sr.region:
                            info["region_y"] = f"{sr.region.y_top:.0f} — {sr.region.y_bottom:.0f}"
                        if sr.scale:
                            info["x_ticks_count"] = len(sr.scale.x_ticks)
                            info["y_ticks_count"] = len(sr.scale.y_ticks)
                            info["x_slope"] = f"{sr.scale.x_slope:.6f}"
                            info["x_intercept"] = f"{sr.scale.x_intercept:.2f}"
                            info["y_slope"] = f"{sr.scale.y_slope:.6f}"
                            info["y_intercept"] = f"{sr.scale.y_intercept:.2f}"
                        info.update(sr.station_area.diagnostics)
                        st.json(info)
        else:
            title = f"❌ STA {sr.station_label} (Page {sr.page_number}) — {sr.stage_reached}"
            with st.expander(title, expanded=False):
                st.error(sr.error)

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