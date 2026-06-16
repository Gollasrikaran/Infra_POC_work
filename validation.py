"""
Validates extracted profiles and generates reports (plots + CSV/JSON).
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

    def validate(self, profiles, earthwork):
        result = ValidationResult()

        stations = profiles.stations
        existing = profiles.existing_elevations
        proposed = profiles.proposed_elevations

        if len(stations) < 10:
            result.warn(f"Very few station points: {len(stations)} — profile may be incomplete.")

        for label, elev in [("existing", existing), ("proposed", proposed)]:
            nans = int(np.sum(np.isnan(elev)))
            if nans > 0:
                result.error(f"{label.title()} has {nans} NaN elevations.")
                continue

            span = float(np.nanmax(elev) - np.nanmin(elev))
            if span > 500:
                result.warn(f"{label.title()} elevation range is very large ({span:.1f} ft) — check scale.")
            if span < 0.1:
                result.warn(f"{label.title()} elevation range is tiny ({span:.4f} ft) — profile may be flat.")

        if len(stations) > 1 and np.any(np.diff(stations) <= 0):
            result.error("Stations are not monotonically increasing.")

        if earthwork.total_cut_volume == 0 and earthwork.total_fill_volume == 0:
            result.warn("Both cut and fill volumes are zero — profiles may be identical.")

        if earthwork.total_cut_volume < 0 or earthwork.total_fill_volume < 0:
            result.error("Negative volume — calculation bug.")

        result.metrics = {
            "station_count": len(stations),
            "station_range": profiles.station_range,
            "cut_volume_cy": earthwork.total_cut_volume_cy,
            "fill_volume_cy": earthwork.total_fill_volume_cy,
            "net_volume_cy": earthwork.net_volume_cy,
        }

        return result


class ReportGenerator:

    def __init__(self, output_dir="reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_profiles(self, profiles, earthwork, label="", save_path=None):
        """Profile overlay plot with cut/fill shading. Returns save path."""
        fig, ax = plt.subplots(figsize=(14, 6))

        sta = profiles.stations
        ex = profiles.existing_elevations
        pr = profiles.proposed_elevations
        diff = pr - ex

        ax.plot(sta, ex, color="#ef4444", lw=1.5, ls="--", label="Existing Ground", zorder=3)
        ax.plot(sta, pr, color="#3b82f6", lw=2.0, ls="-", label="Proposed Grade", zorder=3)

        ax.fill_between(sta, ex, pr, where=(diff < 0), interpolate=True,
                        color="#ef4444", alpha=0.3, label="Cut", zorder=2)
        ax.fill_between(sta, ex, pr, where=(diff > 0), interpolate=True,
                        color="#22c55e", alpha=0.3, label="Fill", zorder=2)

        ax.set_xlabel("Station (ft)", fontsize=11, fontweight="600")
        ax.set_ylabel("Elevation (ft)", fontsize=11, fontweight="600")
        title = "Existing Ground vs Proposed Grade"
        if label:
            title += f"  —  {label}"
        ax.set_title(title, fontsize=13, fontweight="700", pad=12)
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3, lw=0.5)
        ax.set_facecolor("#f8fafc")

        summary = (
            f"Cut: {earthwork.total_cut_volume_cy:,.1f} cy  |  "
            f"Fill: {earthwork.total_fill_volume_cy:,.1f} cy  |  "
            f"Net: {earthwork.net_volume_cy:,.1f} cy"
        )
        ax.text(0.5, -0.12, summary, transform=ax.transAxes,
                ha="center", fontsize=10, color="#475569", fontweight="500")

        plt.tight_layout()

        if save_path is None:
            safe = label.replace(" ", "_").replace("/", "_") or "profile"
            save_path = str(self.output_dir / f"{safe}_plot.png")

        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return save_path

    def save_json(self, earthwork, validation, label="", save_path=None):
        report = {
            "page": label,
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
        df = earthwork.to_dataframe()
        if save_path is None:
            safe = label.replace(" ", "_").replace("/", "_") or "earthwork"
            save_path = str(self.output_dir / f"{safe}_stations.csv")
        df.to_csv(save_path, index=False)
        return save_path
