"""
Batch cross-verification script for Approach 2 (Ground vs. Proposed Line ID).

Processes all cross-section pages in a PDF, extracts profiles, generates
overlay images and a summary CSV for systematic manual review.

Usage:
    python cross_verify.py <pdf_path> [--output-dir output_merged4] [--pages 45,46,47]
"""

import argparse
import csv
import os
import sys

import fitz
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from pdf_classifier import PDFClassifier
from cross_section_splitter import CrossSectionSplitter
from geometry_extractor import GeometryExtractor
from profile_identifier import ProfileIdentifier


def render_region(pdf_path, page_number, y_top, y_bottom, dpi=150):
    """Render a region of a PDF page as a PIL Image."""
    doc = fitz.open(pdf_path)
    page = doc[page_number - 1]
    margin = 10
    clip = fitz.Rect(
        page.rect.x0, max(y_top - margin, page.rect.y0),
        page.rect.x1, min(y_bottom + margin, page.rect.y1),
    )
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, clip=clip)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    doc.close()
    return img


def process_station(pdf_path, page, region, extractor, identifier):
    """Process one cross-section region and return diagnostics dict."""
    pw = page.rect.width
    region_h = region.y_bottom - region.y_top

    paths = extractor.extract_region(page, region.y_top, region.y_bottom)
    if not paths:
        return {"status": "no_paths", "raw_paths": 0}

    profiles = identifier.identify(paths, pw, region_h)
    diag = profiles.diagnostics.copy()

    # Build summary row
    row = {
        "station": region.station_label,
        "station_ft": region.station_ft,
        "page": region.page_number,
        "raw_paths": len(paths),
        "after_grid_filter": diag.get("after_grid_filter", ""),
        "post_merge_strict": diag.get("post_merge_strict", ""),
        "post_merge_tolerant": diag.get("post_merge_tolerant", ""),
        "noise_removed": diag.get("noise_removed", ""),
        "filled_removed": diag.get("filled_removed", ""),
        "candidates_remaining": diag.get("candidates_remaining", ""),
        "scored_candidates": diag.get("scored_candidates", 0),
        "classification_rule": diag.get("classification_rule", "--"),
        "classification_confidence": diag.get("classification_confidence", 0),
        "eg_found": profiles.existing_ground is not None,
        "pg_found": profiles.proposed_grade is not None,
        "status": "ok" if (profiles.existing_ground and profiles.proposed_grade) else "incomplete",
    }

    if profiles.existing_ground:
        eg = profiles.existing_ground
        eg_total = eg.dash_segments + eg.solid_segments
        row["eg_points"] = eg.point_count
        row["eg_length"] = round(eg.length, 1)
        row["eg_dash_ratio"] = round(eg.dash_segments / max(eg_total, 1), 2)
        row["eg_segments_merged"] = eg.segment_count
        row["eg_color"] = str(eg.color)
        row["eg_stroke_w"] = round(eg.stroke_width, 2)
    else:
        row.update({"eg_points": 0, "eg_length": 0, "eg_dash_ratio": 0,
                     "eg_segments_merged": 0, "eg_color": "--", "eg_stroke_w": 0})

    if profiles.proposed_grade:
        pg = profiles.proposed_grade
        pg_total = pg.dash_segments + pg.solid_segments
        row["pg_points"] = pg.point_count
        row["pg_length"] = round(pg.length, 1)
        row["pg_dash_ratio"] = round(pg.dash_segments / max(pg_total, 1), 2)
        row["pg_segments_merged"] = pg.segment_count
        row["pg_color"] = str(pg.color)
        row["pg_stroke_w"] = round(pg.stroke_width, 2)
    else:
        row.update({"pg_points": 0, "pg_length": 0, "pg_dash_ratio": 0,
                     "pg_segments_merged": 0, "pg_color": "--", "pg_stroke_w": 0})

    # classification signal details
    signals = diag.get("classification_signals", {})
    row["signals_used"] = str(signals.get("signals_used", []))
    votes = diag.get("classification_votes", {})
    row["signal_votes"] = str(votes) if votes else "--"

    # failure reason for INCOMPLETE
    row["failure_reason"] = diag.get("failure_reason", "")

    return {"row": row, "profiles": profiles, "diag": diag}


def generate_overlay(pdf_path, region, profiles, output_path, dpi=150):
    """Generate overlay PNG: extracted polylines on top of PDF raster."""
    img = render_region(pdf_path, region.page_number,
                        region.y_top, region.y_bottom, dpi=dpi)

    scale = dpi / 72.0
    margin = 10
    y_off = max(region.y_top - margin, 0)

    fig, ax = plt.subplots(figsize=(16, 7))
    ax.imshow(img, aspect="auto", extent=[0, img.width, img.height, 0])

    # overlay EG (use original path order, not X-sorted, to avoid zigzag)
    if profiles.existing_ground:
        eg_pts = profiles.existing_ground.points
        ox = [(p[0]) * scale for p in eg_pts]
        oy = [(p[1] - y_off) * scale for p in eg_pts]
        eg = profiles.existing_ground
        eg_dr = eg.dash_segments / max(eg.dash_segments + eg.solid_segments, 1)
        ax.plot(ox, oy, color="#ff4444", lw=2.5, ls="--", alpha=0.85, zorder=5,
                label=f"EG (dash={eg_dr:.0%}, pts={eg.point_count})")

    # overlay PG (use original path order, not X-sorted, to avoid zigzag)
    if profiles.proposed_grade:
        pg_pts = profiles.proposed_grade.points
        ox = [(p[0]) * scale for p in pg_pts]
        oy = [(p[1] - y_off) * scale for p in pg_pts]
        pg = profiles.proposed_grade
        pg_dr = pg.dash_segments / max(pg.dash_segments + pg.solid_segments, 1)
        ax.plot(ox, oy, color="#00bbff", lw=2.5, ls="-", alpha=0.85, zorder=5,
                label=f"PG (dash={pg_dr:.0%}, pts={pg.point_count})")

    # confidence badge
    conf = profiles.classification_confidence
    conf_str = f"Confidence: {conf:.0%}"
    rule = profiles.diagnostics.get("classification_rule", "--")

    ax.set_title(
        f"STA {region.station_label} (Page {region.page_number}) -- {conf_str}\n"
        f"Method: {rule}",
        fontsize=12, fontweight="700", pad=10,
    )
    ax.legend(fontsize=9, loc="upper right", facecolor="white",
              edgecolor="#ccc", framealpha=0.92)
    ax.axis("off")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Batch cross-verification for Approach 2")
    parser.add_argument("pdf_path", help="Path to the highway PDF")
    parser.add_argument("--output-dir", default="output_merged4", help="Directory for overlay PNGs and CSV")
    parser.add_argument("--pages", default=None, help="Comma-separated page numbers to process (default: all cross-section pages)")
    args = parser.parse_args()

    pdf_path = args.pdf_path
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # classify pages
    print(f"Analyzing {pdf_path}...")
    classifier = PDFClassifier(pdf_path)
    doc_analysis = classifier.analyze()

    if args.pages:
        page_nums = [int(p.strip()) for p in args.pages.split(",")]
        targets = [p for p in doc_analysis.pages if p.page_number in page_nums]
    else:
        targets = doc_analysis.processable_pages

    print(f"Found {len(targets)} cross-section page(s) to process.")

    # setup pipeline components
    splitter = CrossSectionSplitter()
    extractor = GeometryExtractor()
    identifier = ProfileIdentifier()

    doc = fitz.open(pdf_path)
    all_rows = []
    station_count = 0
    ok_count = 0
    fail_count = 0

    for page_info in targets:
        page = doc[page_info.page_number - 1]
        dwg = page_info.drawing_number or ""
        regions = splitter.split(page, page_info.page_number, dwg)

        if not regions:
            print(f"  Page {page_info.page_number}: no cross-section regions found")
            continue

        for region in regions:
            station_count += 1
            print(f"  STA {region.station_label} (Page {region.page_number})...", end=" ")

            try:
                result = process_station(pdf_path, page, region, extractor, identifier)
            except Exception as e:
                print(f"ERROR: {e}")
                fail_count += 1
                all_rows.append({
                    "station": region.station_label, "station_ft": region.station_ft,
                    "page": region.page_number, "status": f"error: {e}",
                })
                continue

            if result.get("status") == "no_paths":
                print("no paths")
                fail_count += 1
                all_rows.append({
                    "station": region.station_label, "station_ft": region.station_ft,
                    "page": region.page_number, "status": "no_paths",
                })
                continue

            row = result["row"]
            profiles = result["profiles"]
            all_rows.append(row)

            if row["status"] == "ok":
                ok_count += 1
                # generate overlay
                safe_name = region.station_label.replace("+", "_")
                overlay_path = os.path.join(output_dir, f"overlay_sta_{safe_name}_p{region.page_number}.png")
                try:
                    generate_overlay(pdf_path, region, profiles, overlay_path)
                    print(f"OK -> {overlay_path}")
                except Exception as e:
                    print(f"OK (overlay failed: {e})")
            else:
                fail_count += 1
                reason = row.get("failure_reason", row.get("classification_rule", "--"))
                scored = row.get("scored_candidates", 0)
                cands = row.get("candidates_remaining", 0)
                print(f"INCOMPLETE: scored={scored}/{cands} | {reason}")

    doc.close()

    # write summary CSV (use utf-8 encoding to avoid cp1252 issues)
    csv_path = os.path.join(output_dir, "cross_verify_summary.csv")
    if all_rows:
        # collect all possible keys across all rows
        all_keys = []
        seen = set()
        for r in all_rows:
            for k in r.keys():
                if k not in seen:
                    all_keys.append(k)
                    seen.add(k)

        # ensure all rows have all keys
        for r in all_rows:
            for k in all_keys:
                r.setdefault(k, "")

        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys)
            writer.writeheader()
            writer.writerows(all_rows)

    print(f"\n{'='*60}")
    print(f"Total stations: {station_count}")
    print(f"  OK:         {ok_count}")
    print(f"  Failed:     {fail_count}")
    if station_count > 0:
        print(f"  Success %:  {ok_count/station_count:.0%}")
    print(f"Summary CSV:  {csv_path}")
    print(f"Overlays in:  {output_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
