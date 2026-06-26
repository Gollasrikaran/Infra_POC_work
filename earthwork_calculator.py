"""
Earthwork calculations -- cut/fill area per station + volumes between stations.
Uses trapezoidal integration for areas, average end area for volumes.
"""

import numpy as np
import pandas as pd

# numpy 2.0 renamed trapz -> trapezoid
_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")


class StationArea:
    """Stores cut/fill areas for one cross-section."""

    def __init__(self, station_label, station_ft, page_number):
        self.station_label = station_label
        self.station_ft = station_ft
        self.page_number = page_number
        self.cut_area = 0.0         # sq ft
        self.fill_area = 0.0
        self.net_area = 0.0         # cut minus fill
        self.offset_range = (0.0, 0.0)
        self.existing_points = 0
        self.proposed_points = 0
        self.diagnostics = {}


class EarthworkResult:

    def __init__(self):
        self.station_areas = []
        self.segment_volumes = []       # dicts for each pair of consecutive stations
        self.total_cut_area = 0.0
        self.total_fill_area = 0.0
        self.total_cut_volume = 0.0     # cu ft
        self.total_fill_volume = 0.0
        self.total_cut_volume_cy = 0.0  # cu yd
        self.total_fill_volume_cy = 0.0
        self.net_volume = 0.0
        self.net_volume_cy = 0.0
        self.diagnostics = {}

    def to_area_dataframe(self):
        rows = []
        for sa in self.station_areas:
            rows.append({
                "Station": sa.station_label,
                "Station (ft)": sa.station_ft,
                "Page": sa.page_number,
                "Cut Area (sq ft)": round(sa.cut_area, 2),
                "Fill Area (sq ft)": round(sa.fill_area, 2),
                "Net Area (sq ft)": round(sa.net_area, 2),
            })
        return pd.DataFrame(rows)

    def to_volume_dataframe(self):
        return pd.DataFrame(self.segment_volumes)

    # backwards compat
    def to_dataframe(self):
        return self.to_area_dataframe()

    def to_volume_dataframe(self):
        return pd.DataFrame(self.segment_volumes)

    def summary_dict(self):
        return {
            "Total Stations": len(self.station_areas),
            "Total Cut Area (sq ft)": round(self.total_cut_area, 2),
            "Total Fill Area (sq ft)": round(self.total_fill_area, 2),
            "Total Cut Volume (cu ft)": round(self.total_cut_volume, 2),
            "Total Fill Volume (cu ft)": round(self.total_fill_volume, 2),
            "Total Cut Volume (cu yd)": round(self.total_cut_volume_cy, 2),
            "Total Fill Volume (cu yd)": round(self.total_fill_volume_cy, 2),
            "Net Volume (cu yd)": round(self.net_volume_cy, 2),
        }


class EarthworkCalculator:

    def __init__(self, offset_interval=1.0):
        self.offset_interval = offset_interval

    def compute_area(self, normalized, station_label, station_ft, page_number):
        """Cut/fill area at one station using trapezoidal integration."""
        sa = StationArea(station_label, station_ft, page_number)

        offsets = normalized.stations
        existing = normalized.existing_elevations
        proposed = normalized.proposed_elevations

        if len(offsets) < 2:
            return sa

        sa.offset_range = (float(offsets.min()), float(offsets.max()))
        sa.existing_points = len(existing)
        sa.proposed_points = len(proposed)

        diff = proposed - existing

        # cut where existing is above proposed, fill where it's below
        cut_depths = np.where(diff < 0, np.abs(diff), 0.0)
        fill_heights = np.where(diff > 0, diff, 0.0)

        sa.cut_area = float(_trapz(cut_depths, offsets))
        sa.fill_area = float(_trapz(fill_heights, offsets))
        sa.net_area = sa.cut_area - sa.fill_area

        sa.diagnostics = {
            "offset_range": sa.offset_range,
            "existing_elev_range": (float(np.nanmin(existing)), float(np.nanmax(existing))),
            "proposed_elev_range": (float(np.nanmin(proposed)), float(np.nanmax(proposed))),
            "max_cut_depth": float(np.max(cut_depths)),
            "max_fill_height": float(np.max(fill_heights)),
        }

        return sa

    def compute_volumes(self, station_areas):
        """Average end area method between consecutive stations."""
        result = EarthworkResult()

        sorted_areas = sorted(station_areas, key=lambda sa: sa.station_ft)
        result.station_areas = sorted_areas

        result.total_cut_area = sum(sa.cut_area for sa in sorted_areas)
        result.total_fill_area = sum(sa.fill_area for sa in sorted_areas)

        if len(sorted_areas) < 2:
            result.diagnostics = {"stations": len(sorted_areas), "note": "need >= 2 for volume"}
            return result

        total_cut_vol = 0.0
        total_fill_vol = 0.0

        for i in range(len(sorted_areas) - 1):
            s1 = sorted_areas[i]
            s2 = sorted_areas[i + 1]
            dist = s2.station_ft - s1.station_ft

            if dist <= 0:
                continue

            # V = (A1 + A2) / 2 * L
            cut_vol = ((s1.cut_area + s2.cut_area) / 2.0) * dist
            fill_vol = ((s1.fill_area + s2.fill_area) / 2.0) * dist

            total_cut_vol += cut_vol
            total_fill_vol += fill_vol

            result.segment_volumes.append({
                "Sta Start": s1.station_label,
                "Sta End": s2.station_label,
                "Distance (ft)": round(dist, 1),
                "Cut Area 1 (sq ft)": round(s1.cut_area, 2),
                "Cut Area 2 (sq ft)": round(s2.cut_area, 2),
                "Fill Area 1 (sq ft)": round(s1.fill_area, 2),
                "Fill Area 2 (sq ft)": round(s2.fill_area, 2),
                "Cut Volume (cu ft)": round(cut_vol, 2),
                "Fill Volume (cu ft)": round(fill_vol, 2),
                "Cut Volume (cu yd)": round(cut_vol / 27, 2),
                "Fill Volume (cu yd)": round(fill_vol / 27, 2),
            })

        result.total_cut_volume = total_cut_vol
        result.total_fill_volume = total_fill_vol
        result.total_cut_volume_cy = total_cut_vol / 27.0
        result.total_fill_volume_cy = total_fill_vol / 27.0
        result.net_volume = total_cut_vol - total_fill_vol
        result.net_volume_cy = result.net_volume / 27.0

        result.diagnostics = {
            "total_stations": len(sorted_areas),
            "station_range": f"{sorted_areas[0].station_label} — {sorted_areas[-1].station_label}",
            "total_distance": sorted_areas[-1].station_ft - sorted_areas[0].station_ft,
            "segments": len(result.segment_volumes),
        }

        return result