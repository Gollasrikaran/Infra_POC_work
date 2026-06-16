"""
Splits a cross-section PDF page into individual station graphs.

Each page has N cross-section graphs stacked vertically. This module
detects graph boundaries by finding X-axis label rows and station labels
in the right margin.
"""

import re
import fitz


class CrossSectionRegion:
    """One cross-section graph within a page."""

    def __init__(self, station_label, station_ft, y_top, y_bottom, page_number,
                 drawing_number="", elevations=None, offsets=None):
        self.station_label = station_label
        self.station_ft = station_ft
        self.y_top = y_top
        self.y_bottom = y_bottom
        self.page_number = page_number
        self.drawing_number = drawing_number
        self.elevations = elevations or []
        self.offsets = offsets or []

    @property
    def elevation_range(self):
        if not self.elevations:
            return (0.0, 0.0)
        return (min(self.elevations), max(self.elevations))

    @property
    def offset_range(self):
        if not self.offsets:
            return (-140.0, 140.0)
        return (min(self.offsets), max(self.offsets))


class CrossSectionSplitter:

    def __init__(self, right_margin_fraction=0.85, axis_label_tolerance=2.0):
        self.right_margin_x = right_margin_fraction
        self.axis_tol = axis_label_tolerance

    def split(self, page, page_number, drawing_number=""):
        """Split a page into cross-section regions. Returns list of CrossSectionRegion."""
        rect = page.rect
        pw, ph = rect.width, rect.height

        # collect all text spans with positions
        spans = self._get_spans(page)

        # find station labels in right margin (format: XX+XX)
        right_x = pw * self.right_margin_x
        station_labels = []
        for y, x, txt, sz in spans:
            if x > right_x and sz > 8.0:
                m = re.match(r"^(\d+)\+(\d{2})$", txt.strip())
                if m:
                    sta_ft = int(m.group(1)) * 100 + int(m.group(2))
                    station_labels.append((y, txt.strip(), sta_ft))

        if not station_labels:
            return []

        station_labels.sort(key=lambda s: s[0])

        # find X-axis label rows — clusters of offset labels at the same Y
        axis_rows = self._find_axis_rows(spans, pw)

        if not axis_rows:
            return []

        # find elevation labels on left edge (x < 15% of page width)
        left_x = pw * 0.12
        elev_spans = []
        for y, x, txt, sz in spans:
            if x < left_x:
                m = re.match(r"^(\d{3,4})$", txt.strip())
                if m:
                    val = float(m.group(1))
                    if 100 < val < 9999:
                        elev_spans.append((y, val))

        # build regions by pairing station labels with axis rows
        regions = []
        for i, axis_y in enumerate(axis_rows):
            # top boundary: previous axis row (or page header area)
            if i == 0:
                y_top = max(axis_y - (axis_rows[1] - axis_rows[0]) if len(axis_rows) > 1 else axis_y - 180, 30.0)
            else:
                y_top = axis_rows[i - 1]

            y_bottom = axis_y

            # find station label closest to this region (Y between top and bottom)
            best_sta = None
            best_dist = float("inf")
            for sy, label, ft in station_labels:
                if y_top <= sy <= y_bottom:
                    dist = abs(sy - (y_top + y_bottom) / 2)
                    if dist < best_dist:
                        best_dist = dist
                        best_sta = (label, ft)

            if not best_sta:
                continue

            # collect elevation values within this region
            region_elevs = sorted(set(
                val for ey, val in elev_spans
                if y_top - 5 <= ey <= y_bottom + 5
            ))

            # collect offset values from this axis row
            region_offsets = self._get_offsets_at_row(spans, axis_y)

            regions.append(CrossSectionRegion(
                station_label=best_sta[0],
                station_ft=best_sta[1],
                y_top=y_top,
                y_bottom=y_bottom,
                page_number=page_number,
                drawing_number=drawing_number,
                elevations=region_elevs,
                offsets=region_offsets,
            ))

        regions.sort(key=lambda r: r.station_ft)
        return regions

    def _get_spans(self, page):
        """Extract all text spans as (y, x, text, font_size) tuples."""
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

    def _find_axis_rows(self, spans, page_width):
        """Find Y values where X-axis offset labels cluster (the bottom of each graph)."""
        # offset labels are numbers like -140, -130, ..., 0, ..., 130, 140
        # they appear in the middle of the page width
        offset_spans = []
        for y, x, txt, sz in spans:
            m = re.match(r"^-?\d{1,3}$", txt)
            if m:
                val = int(txt)
                if -150 <= val <= 150 and 0.1 * page_width < x < 0.95 * page_width:
                    offset_spans.append((y, val))

        if not offset_spans:
            return []

        # cluster by Y value (offset labels on the same axis row share a Y within tolerance)
        offset_spans.sort()
        rows = {}
        for y, val in offset_spans:
            placed = False
            for row_y in rows:
                if abs(y - row_y) < self.axis_tol:
                    rows[row_y].append(val)
                    placed = True
                    break
            if not placed:
                rows[y] = [val]

        # an axis row should have at least 10 offset labels
        axis_ys = sorted(y for y, vals in rows.items() if len(vals) >= 10)
        return axis_ys

    def _get_offsets_at_row(self, spans, axis_y):
        """Get the offset values from a specific axis label row."""
        offsets = []
        for y, x, txt, sz in spans:
            if abs(y - axis_y) < self.axis_tol:
                m = re.match(r"^-?\d{1,3}$", txt)
                if m:
                    offsets.append(int(txt))
        return sorted(set(offsets))
