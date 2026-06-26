"""
Converts PDF page coordinates into real-world engineering coords.
X axis = offset from centerline (ft), Y axis = elevation (ft).
Reads tick labels from the PDF text to build a linear mapping.
"""

import re
import fitz
import numpy as np


class RegionScale:
    """Linear mapping params for one cross-section graph region."""

    def __init__(self):
        self.x_ticks = []       # [(pdf_x, offset_ft), ...]
        self.x_slope = 1.0
        self.x_intercept = 0.0

        self.y_ticks = []       # [(pdf_y, elevation_ft), ...]
        self.y_slope = -1.0     # negative because PDF Y is flipped
        self.y_intercept = 0.0

        self.h_scale_ft = 10.0  # nominal ft/inch
        self.v_scale_ft = 10.0


class TransformedProfile:
    """Profile with real offset/elevation arrays."""

    def __init__(self, offsets, elevations, label=""):
        self.offsets = offsets
        self.elevations = elevations
        self.stations = offsets       # alias
        self.label = label

    @property
    def station_range(self):
        if len(self.offsets) == 0:
            return (0.0, 0.0)
        return (float(self.offsets.min()), float(self.offsets.max()))

    @property
    def elevation_range(self):
        if len(self.elevations) == 0:
            return (0.0, 0.0)
        return (float(self.elevations.min()), float(self.elevations.max()))


class CoordinateTransformer:

    def build_scale(self, page, region):
        """Build scale mapping from text labels in the cross-section region."""
        scale = RegionScale()
        spans = self._get_spans(page)

        # offset tick labels along the X axis (at y_bottom)
        x_ticks = []
        for y, x, txt, sz in spans:
            if abs(y - region.y_bottom) < 3.0:
                m = re.match(r"^(-?\d{1,3})$", txt.strip())
                if m:
                    offset_ft = float(m.group(1))
                    if abs(offset_ft) > 200:  # skip elevation values that leak in
                        continue
                    x_ticks.append((x, offset_ft))

        # elevation labels on left edge
        y_ticks = []
        for y, x, txt, sz in spans:
            if region.y_top - 5 <= y <= region.y_bottom + 5 and x < page.rect.width * 0.12:
                m = re.match(r"^(\d{3,4})$", txt.strip())
                if m:
                    elev_ft = float(m.group(1))
                    y_ticks.append((y, elev_ft))

        # least-squares fit for X mapping
        if len(x_ticks) >= 2:
            scale.x_ticks = x_ticks
            px = np.array([t[0] for t in x_ticks])
            ft = np.array([t[1] for t in x_ticks])
            A = np.vstack([px, np.ones(len(px))]).T
            slope, intercept = np.linalg.lstsq(A, ft, rcond=None)[0]
            scale.x_slope = slope
            scale.x_intercept = intercept

        # same for Y mapping
        if len(y_ticks) >= 2:
            scale.y_ticks = y_ticks
            py = np.array([t[0] for t in y_ticks])
            ft = np.array([t[1] for t in y_ticks])
            A = np.vstack([py, np.ones(len(py))]).T
            slope, intercept = np.linalg.lstsq(A, ft, rcond=None)[0]
            scale.y_slope = slope
            scale.y_intercept = intercept

        self._read_nominal_scale(page, scale)
        return scale

    def transform(self, pdf_points, scale, label=""):
        """Convert (x,y) PDF coords into offset/elevation arrays."""
        if not pdf_points:
            return TransformedProfile(np.array([]), np.array([]), label)

        pts = np.array(pdf_points)

        offsets = pts[:, 0] * scale.x_slope + scale.x_intercept
        elevations = pts[:, 1] * scale.y_slope + scale.y_intercept

        # sort left-to-right
        order = np.argsort(offsets)
        offsets = offsets[order]
        elevations = elevations[order]

        # drop duplicate offsets
        mask = np.diff(offsets, prepend=-np.inf) > 0.01
        return TransformedProfile(offsets[mask], elevations[mask], label)

    def _read_nominal_scale(self, page, scale):
        """Try to grab HORIZONTAL/VERTICAL scale text from page bottom."""
        rect = page.rect
        bottom = fitz.Rect(0, rect.height * 0.85, rect.width, rect.height)
        text = page.get_text("text", clip=bottom)

        m = re.search(r'(?i)HORIZONTAL[:\s]*1"\s*=\s*(\d+)', text)
        if m:
            scale.h_scale_ft = float(m.group(1))

        m = re.search(r'(?i)VERTICAL[:\s]*1"\s*=\s*(\d+)', text)
        if m:
            scale.v_scale_ft = float(m.group(1))

    def _get_spans(self, page):
        text_dict = page.get_text("dict")
        spans = []
        for block in text_dict.get("blocks", []):
            for line in block.get("lines", []):
                for span in line["spans"]:
                    txt = span["text"].strip()
                    if txt:
                        ox, oy = span["origin"]
                        spans.append((round(oy, 1), round(ox, 1), txt, round(span["size"], 1)))
        return spans