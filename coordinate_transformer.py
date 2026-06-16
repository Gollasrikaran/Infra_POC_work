"""
Converts PDF page coordinates (top-left origin, points) into
engineering coordinates (station in feet, elevation in feet).
"""

import re
import fitz
import numpy as np


class ScaleInfo:
    """Everything needed to map PDF coords to engineering coords."""

    def __init__(self):
        self.h_scale = 10.0
        self.v_scale = 10.0
        self.origin_x = 0.0
        self.origin_y = 0.0
        self.start_station = 0.0
        self.base_elevation = 0.0
        self.ppi = 72.0


class TransformedProfile:
    """Profile with real station/elevation arrays."""

    def __init__(self, stations, elevations, label=""):
        self.stations = stations
        self.elevations = elevations
        self.label = label

    @property
    def station_range(self):
        if len(self.stations) == 0:
            return (0.0, 0.0)
        return (float(self.stations.min()), float(self.stations.max()))

    @property
    def elevation_range(self):
        if len(self.elevations) == 0:
            return (0.0, 0.0)
        return (float(self.elevations.min()), float(self.elevations.max()))


class CoordinateTransformer:

    def detect_scale(self, page, h_override=None, v_override=None):
        """Read scale from page text. Overrides win if provided."""
        rect = page.rect
        info = ScaleInfo()

        if h_override is not None:
            info.h_scale = h_override
        if v_override is not None:
            info.v_scale = v_override

        if h_override is None or v_override is None:
            scale_area = fitz.Rect(0, rect.height * 0.70, rect.width, rect.height)
            text = page.get_text("text", clip=scale_area)

            if h_override is None:
                m = re.search(r"(?i)HORIZONTAL[:\s]*1\"\s*=\s*(\d+)'?", text)
                if m:
                    info.h_scale = float(m.group(1))

            if v_override is None:
                m = re.search(r"(?i)VERTICAL[:\s]*1\"\s*=\s*(\d+)'?", text)
                if m:
                    info.v_scale = float(m.group(1))

        self._read_axis_labels(page, rect, info)
        return info

    def transform(self, pdf_points, scale, label=""):
        """Convert list of (x, y) PDF coords into station/elevation arrays."""
        if not pdf_points:
            return TransformedProfile(np.array([]), np.array([]), label)

        pts = np.array(pdf_points)

        x_in = (pts[:, 0] - scale.origin_x) / scale.ppi
        y_in = (pts[:, 1] - scale.origin_y) / scale.ppi

        stations = x_in * scale.h_scale + scale.start_station
        elevations = -y_in * scale.v_scale + scale.base_elevation

        order = np.argsort(stations)
        stations = stations[order]
        elevations = elevations[order]

        mask = np.diff(stations, prepend=-np.inf) > 0.001
        return TransformedProfile(stations[mask], elevations[mask], label)

    def _read_axis_labels(self, page, rect, info):
        """Find station labels (bottom) and elevation labels (left) to set origin."""

        bottom = fitz.Rect(rect.width * 0.05, rect.height * 0.80, rect.width * 0.95, rect.height)
        bottom_dict = page.get_text("dict", clip=bottom)

        sta_vals, sta_pos = [], []
        if "blocks" in bottom_dict:
            for block in bottom_dict["blocks"]:
                for line in block.get("lines", []):
                    for span in line["spans"]:
                        m = re.match(r"(\d+)\+(\d{2})", span["text"].strip())
                        if m:
                            sta_ft = int(m.group(1)) * 100 + int(m.group(2))
                            sta_vals.append(sta_ft)
                            sta_pos.append(span["origin"][0])

        left = fitz.Rect(0, rect.height * 0.05, rect.width * 0.15, rect.height * 0.85)
        left_dict = page.get_text("dict", clip=left)

        elev_vals, elev_pos = [], []
        if "blocks" in left_dict:
            for block in left_dict["blocks"]:
                for line in block.get("lines", []):
                    for span in line["spans"]:
                        m = re.match(r"^(\d{3,5})$", span["text"].strip())
                        if m:
                            elev_vals.append(float(m.group(1)))
                            elev_pos.append(span["origin"][1])

        if len(sta_vals) >= 2:
            idx = int(np.argmin(sta_pos))
            info.origin_x = sta_pos[idx]
            info.start_station = sta_vals[idx]

            pairs = sorted(zip(sta_pos, sta_vals), key=lambda x: x[0])
            dx = pairs[-1][0] - pairs[0][0]
            ds = pairs[-1][1] - pairs[0][1]
            if dx > 0 and ds > 0:
                info.h_scale = ds / (dx / info.ppi)

        if len(elev_vals) >= 2:
            idx = int(np.argmax(elev_pos))
            info.origin_y = elev_pos[idx]
            info.base_elevation = elev_vals[idx]

            pairs = sorted(zip(elev_pos, elev_vals), key=lambda x: x[0])
            dy = abs(pairs[0][0] - pairs[-1][0])
            de = abs(pairs[0][1] - pairs[-1][1])
            if dy > 0 and de > 0:
                info.v_scale = de / (dy / info.ppi)
