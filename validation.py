"""
Validates cross-section results and generates reports (plots, JSON, CSV).
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


class ValidationResult:
    def __init__(self):
        self.is_valid = True
        self.warnings = []
        self.errors = []
        self.metrics = {}

    def warn(self, msg):
        self.warnings.append(msg)

    def error(self, msg):
        self.errors.append(msg)
        self.is_valid = False


class Validator:

    def validate_cross_sections(self, earthwork, station_results):
        result = ValidationResult()

        n = len(earthwork.station_areas)
        if n < 2:
            result.warn(f"Only {n} station(s) -- need at least 2 for volume calc.")

        if n >= 2:
            areas = earthwork.station_areas
            for sa in areas:
                if sa.cut_area < 0:
                    result.error(f"Negative cut area at {sa.station_label}")
                if sa.fill_area < 0:
                    result.error(f"Negative fill area at {sa.station_label}")

            for i in range(len(areas) - 1):
                dist = areas[i + 1].station_ft - areas[i].station_ft
                if dist <= 0:
                    result.error(f"Non-increasing stations: {areas[i].station_label} -> {areas[i+1].station_label}")
                if dist > 500:
                    result.warn(f"Large gap ({dist:.0f} ft) between {areas[i].station_label} and {areas[i+1].station_label}")

        if earthwork.total_cut_volume == 0 and earthwork.total_fill_volume == 0:
            result.warn("Both cut and fill volumes are zero.")

        failed = [s for s in station_results if not s.success]
        if failed:
            result.warn(f"{len(failed)} station(s) failed processing.")

        result.metrics = {
            "total_stations": n,
            "successful_stations": n - len(failed) if station_results else n,
            "cut_volume_cy": earthwork.total_cut_volume_cy,
            "fill_volume_cy": earthwork.total_fill_volume_cy,
            "net_volume_cy": earthwork.net_volume_cy,
        }

        return result


class ReportGenerator:

    def __init__(self, output_dir="reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_cross_sections(self, station_results, earthwork, label=""):
        """Bar chart of cut/fill areas + volume summary table."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 1]})

        labels = [s.station_area.station_label for s in station_results if s.station_area]
        cuts = [s.station_area.cut_area for s in station_results if s.station_area]
        fills = [s.station_area.fill_area for s in station_results if s.station_area]

        x = np.arange(len(labels))
        width = 0.35

        ax1.bar(x - width / 2, cuts, width, label="Cut Area", color="#ef4444", alpha=0.8)
        ax1.bar(x + width / 2, fills, width, label="Fill Area", color="#22c55e", alpha=0.8)
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax1.set_ylabel("Area (sq ft)", fontsize=11, fontweight="600")
        ax1.set_title("Cut / Fill Areas per Station", fontsize=13, fontweight="700", pad=10)
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.2, axis="y")
        ax1.set_facecolor("#f8fafc")

        # summary table below the chart
        ax2.axis("off")
        table_data = [
            ["Total Cut Volume", f"{earthwork.total_cut_volume_cy:,.1f} cu yd"],
            ["Total Fill Volume", f"{earthwork.total_fill_volume_cy:,.1f} cu yd"],
            ["Net Volume", f"{earthwork.net_volume_cy:,.1f} cu yd"],
            ["Stations Processed", str(len(labels))],
        ]
        table = ax2.table(cellText=table_data, colLabels=["Metric", "Value"],
                          loc="center", cellLoc="left")
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(0.8, 1.5)

        for j in range(2):
            table[0, j].set_facecolor("#1e293b")
            table[0, j].set_text_props(color="white", fontweight="bold")

        plt.tight_layout()

        safe = label.replace(" ", "_").replace("/", "_") or "cross_sections"
        save_path = str(self.output_dir / f"{safe}_plot.png")
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return save_path

    def plot_station_profile(self, normalized, station_area, label="", save_path=None):
        """Single cross-section with cut/fill shading."""
        fig, ax = plt.subplots(figsize=(12, 5))

        offsets = normalized.stations
        ex = normalized.existing_elevations
        pr = normalized.proposed_elevations
        diff = pr - ex

        ax.plot(offsets, ex, color="#ef4444", lw=1.5, ls="--", label="Existing Ground", zorder=3)
        ax.plot(offsets, pr, color="#3b82f6", lw=2.0, ls="-", label="Proposed Grade", zorder=3)

        ax.fill_between(offsets, ex, pr, where=(diff < 0), interpolate=True,
                        color="#ef4444", alpha=0.3, label="Cut", zorder=2)
        ax.fill_between(offsets, ex, pr, where=(diff > 0), interpolate=True,
                        color="#22c55e", alpha=0.3, label="Fill", zorder=2)

        ax.set_xlabel("Offset from CL (ft)", fontsize=11, fontweight="600")
        ax.set_ylabel("Elevation (ft)", fontsize=11, fontweight="600")
        title = f"Cross-Section at STA {station_area.station_label}"
        if label:
            title += f"  --  {label}"
        ax.set_title(title, fontsize=13, fontweight="700", pad=10)
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3, lw=0.5)
        ax.set_facecolor("#f8fafc")

        summary = (
            f"Cut: {station_area.cut_area:,.1f} sq ft  |  "
            f"Fill: {station_area.fill_area:,.1f} sq ft"
        )
        ax.text(0.5, -0.12, summary, transform=ax.transAxes,
                ha="center", fontsize=10, color="#475569", fontweight="500")

        plt.tight_layout()

        if save_path is None:
            safe = station_area.station_label.replace("+", "_")
            save_path = str(self.output_dir / f"sta_{safe}_plot.png")

        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return save_path

    def save_json(self, earthwork, validation, label="", save_path=None):
        report = {
            "label": label,
            "summary": earthwork.summary_dict(),
            "validation": {
                "is_valid": validation.is_valid,
                "warnings": validation.warnings,
                "errors": validation.errors,
                "metrics": {
                    k: list(v) if isinstance(v, tuple) else v
                    for k, v in validation.metrics.items()
                },
            },
            "diagnostics": earthwork.diagnostics,
        }

        if save_path is None:
            safe = label.replace(" ", "_").replace("/", "_") or "earthwork"
            save_path = str(self.output_dir / f"{safe}_report.json")

        with open(save_path, "w") as f:
            json.dump(report, f, indent=4, default=str)
        return save_path

    def save_csv(self, earthwork, label="", save_path=None):
        df = earthwork.to_area_dataframe()
        if save_path is None:
            safe = label.replace(" ", "_").replace("/", "_") or "earthwork"
            save_path = str(self.output_dir / f"{safe}_stations.csv")
        df.to_csv(save_path, index=False)
        return save_path