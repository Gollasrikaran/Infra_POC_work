"""
Converts PDF page coordinates into engineering coordinates for cross-sections.

Cross-section axes:
  X-axis = offset from centerline (ft), using tick labels -140..0..140
  Y-axis = elevation (ft), using labels like 650, 660, 670...

Uses known text label positions within each graph region to build
a precise linear mapping.
"""

import re
import fitz
import numpy as np


class RegionScale:
    """Mapping parameters for one cross-section graph region."""

    def __init__(self):
        # offset (X) mapping: PDF x -> offset ft
        self.x_ticks = []       # [(pdf_x, offset_ft), ...]
        self.x_slope = 1.0      # offset_ft per PDF pt
        self.x_intercept = 0.0

        # elevation (Y) mapping: PDF y -> elevation ft
        self.y_ticks = []       # [(pdf_y, elevation_ft), ...]
        self.y_slope = -1.0     # elevation per PDF pt (negative: PDF Y is flipped)
        self.y_intercept = 0.0

        self.h_scale_ft = 10.0  # nominal ft per inch
        self.v_scale_ft = 10.0


class TransformedProfile:
    """Profile with real offset/elevation arrays."""

    def __init__(self, offsets, elevations, label=""):
        self.offsets = offsets
        self.elevations = elevations
        self.stations = offsets       # alias for compatibility
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
        """Build scale mapping from text labels within a cross-section region."""
        scale = RegionScale()

        spans = self._get_spans(page)

        # collect offset tick labels on the X-axis row (at y_bottom)
        x_ticks = []
        for y, x, txt, sz in spans:
            if abs(y - region.y_bottom) < 3.0:
                m = re.match(r"^(-?\d{1,3})$", txt.strip())
                if m:
                    offset_ft = float(m.group(1))
                    x_ticks.append((x, offset_ft))

        # collect elevation labels on the left edge within region
        y_ticks = []
        for y, x, txt, sz in spans:
            if region.y_top - 5 <= y <= region.y_bottom + 5 and x < page.rect.width * 0.12:
                m = re.match(r"^(\d{3,4})$", txt.strip())
                if m:
                    elev_ft = float(m.group(1))
                    y_ticks.append((y, elev_ft))

        # fit linear mapping for X: pdf_x -> offset_ft
        if len(x_ticks) >= 2:
            scale.x_ticks = x_ticks
            px = np.array([t[0] for t in x_ticks])
            ft = np.array([t[1] for t in x_ticks])
            # least-squares fit
            A = np.vstack([px, np.ones(len(px))]).T
            slope, intercept = np.linalg.lstsq(A, ft, rcond=None)[0]
            scale.x_slope = slope
            scale.x_intercept = intercept

        # fit linear mapping for Y: pdf_y -> elevation_ft
        if len(y_ticks) >= 2:
            scale.y_ticks = y_ticks
            py = np.array([t[0] for t in y_ticks])
            ft = np.array([t[1] for t in y_ticks])
            A = np.vstack([py, np.ones(len(py))]).T
            slope, intercept = np.linalg.lstsq(A, ft, rcond=None)[0]
            scale.y_slope = slope
            scale.y_intercept = intercept

        # read nominal scale from bottom of page
        self._read_nominal_scale(page, scale)

        return scale

    def transform(self, pdf_points, scale, label=""):
        """Convert list of (x, y) PDF coords into offset/elevation arrays."""
        if not pdf_points:
            return TransformedProfile(np.array([]), np.array([]), label)

        pts = np.array(pdf_points)

        offsets = pts[:, 0] * scale.x_slope + scale.x_intercept
        elevations = pts[:, 1] * scale.y_slope + scale.y_intercept

        # sort by offset (left to right)
        order = np.argsort(offsets)
        offsets = offsets[order]
        elevations = elevations[order]

        # remove duplicate offsets
        mask = np.diff(offsets, prepend=-np.inf) > 0.01
        return TransformedProfile(offsets[mask], elevations[mask], label)

    def _read_nominal_scale(self, page, scale):
        """Read HORIZONTAL/VERTICAL scale text from page bottom."""
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