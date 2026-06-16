"""
Earthwork cut/fill calculation.

Cut  = existing is above proposed (needs excavation)
Fill = proposed is above existing (needs embankment)

Volume uses the Average End Area Method:
    V = ((A1 + A2) / 2) x distance
"""

import numpy as np
import pandas as pd

# numpy 2.0 renamed trapz to trapezoid
_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")


class EarthworkResult:

    def __init__(self):
        self.station_data = []
        self.segment_volumes = []
        self.total_cut_area = 0.0
        self.total_fill_area = 0.0
        self.total_cut_volume = 0.0
        self.total_fill_volume = 0.0
        self.total_cut_volume_cy = 0.0
        self.total_fill_volume_cy = 0.0
        self.net_volume = 0.0
        self.net_volume_cy = 0.0
        self.station_range = (0.0, 0.0)
        self.diagnostics = {}

    def to_dataframe(self):
        return pd.DataFrame(self.station_data)

    def to_volume_dataframe(self):
        return pd.DataFrame(self.segment_volumes)

    def summary_dict(self):
        return {
            "Station Range": f"{self.station_range[0]:.0f} — {self.station_range[1]:.0f}",
            "Total Cut Area (sq ft)": round(self.total_cut_area, 2),
            "Total Fill Area (sq ft)": round(self.total_fill_area, 2),
            "Total Cut Volume (cu ft)": round(self.total_cut_volume, 2),
            "Total Fill Volume (cu ft)": round(self.total_fill_volume, 2),
            "Total Cut Volume (cu yd)": round(self.total_cut_volume_cy, 2),
            "Total Fill Volume (cu yd)": round(self.total_fill_volume_cy, 2),
            "Net Volume (cu yd)": round(self.net_volume_cy, 2),
        }


class EarthworkCalculator:

    def __init__(self, unit_width=1.0):
        self.unit_width = unit_width

    def calculate(self, profiles):
        result = EarthworkResult()
        result.station_range = profiles.station_range

        stations = profiles.stations
        existing = profiles.existing_elevations
        proposed = profiles.proposed_elevations

        if len(stations) < 2:
            return result

        diffs = proposed - existing

        for i in range(len(stations)):
            d = float(diffs[i])
            if d < -0.01:
                zone, cut, fill = "CUT", abs(d), 0.0
            elif d > 0.01:
                zone, cut, fill = "FILL", 0.0, d
            else:
                zone, cut, fill = "ZERO", 0.0, 0.0

            result.station_data.append({
                "Station": float(stations[i]),
                "Existing Elev (ft)": round(float(existing[i]), 2),
                "Proposed Elev (ft)": round(float(proposed[i]), 2),
                "Difference (ft)": round(d, 2),
                "Zone": zone,
                "Cut Depth (ft)": round(cut, 2),
                "Fill Height (ft)": round(fill, 2),
            })

        cut_arr = np.where(diffs < 0, np.abs(diffs), 0.0)
        fill_arr = np.where(diffs > 0, diffs, 0.0)
        result.total_cut_area = float(_trapz(cut_arr, stations))
        result.total_fill_area = float(_trapz(fill_arr, stations))

        total_cut, total_fill = 0.0, 0.0

        for i in range(len(stations) - 1):
            ds = float(stations[i + 1] - stations[i])
            d0 = float(diffs[i])
            d1 = float(diffs[i + 1])

            a0_cut = abs(d0) * self.unit_width if d0 < 0 else 0.0
            a0_fill = d0 * self.unit_width if d0 > 0 else 0.0
            a1_cut = abs(d1) * self.unit_width if d1 < 0 else 0.0
            a1_fill = d1 * self.unit_width if d1 > 0 else 0.0

            if d0 * d1 < 0:
                t = abs(d0) / (abs(d0) + abs(d1))
                dz = ds * t

                if d0 < 0:
                    vc = (a0_cut / 2) * dz
                    vf = (a1_fill / 2) * (ds - dz)
                else:
                    vf = (a0_fill / 2) * dz
                    vc = (a1_cut / 2) * (ds - dz)

                total_cut += vc
                total_fill += vf
                zone = "MIXED"
            else:
                vc = ((a0_cut + a1_cut) / 2) * ds
                vf = ((a0_fill + a1_fill) / 2) * ds
                total_cut += vc
                total_fill += vf
                zone = "CUT" if (a0_cut + a1_cut) > 0 else ("FILL" if (a0_fill + a1_fill) > 0 else "ZERO")

            result.segment_volumes.append({
                "Sta Start": float(stations[i]),
                "Sta End": float(stations[i + 1]),
                "Distance (ft)": round(ds, 2),
                "Area Start (sq ft)": round(max(a0_cut, a0_fill), 2),
                "Area End (sq ft)": round(max(a1_cut, a1_fill), 2),
                "Volume (cu ft)": round(vc + vf, 2),
                "Volume (cu yd)": round((vc + vf) / 27, 2),
                "Zone": zone,
            })

        result.total_cut_volume = total_cut
        result.total_fill_volume = total_fill
        result.total_cut_volume_cy = total_cut / 27.0
        result.total_fill_volume_cy = total_fill / 27.0
        result.net_volume = total_cut - total_fill
        result.net_volume_cy = result.net_volume / 27.0

        result.diagnostics = {
            "total_stations": len(stations),
            "cut_stations": sum(1 for s in result.station_data if s["Zone"] == "CUT"),
            "fill_stations": sum(1 for s in result.station_data if s["Zone"] == "FILL"),
            "zero_stations": sum(1 for s in result.station_data if s["Zone"] == "ZERO"),
            "transitions": sum(1 for s in result.segment_volumes if s["Zone"] == "MIXED"),
        }

        return result
