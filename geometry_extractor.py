"""
Pulls vector geometry (lines, curves) from a PDF page using PyMuPDF's
drawing API. Returns structured polyline data for profile identification.
"""

import fitz
import numpy as np


class ExtractedPath:
    """One vector path from the PDF."""

    def __init__(self, path_id, points, color=None, fill_color=None,
                 stroke_width=0.0, dashes=None, is_filled=False,
                 bbox=None, item_types=None, raw_dashes=None,
                 segment_count=1, dash_segments=0, solid_segments=0):
        self.path_id = path_id
        self.points = points
        self.color = color
        self.fill_color = fill_color
        self.stroke_width = stroke_width
        self.dashes = dashes
        self.raw_dashes = raw_dashes
        self.is_filled = is_filled
        self.bbox = bbox
        self.item_types = item_types or []
        # these get accumulated during merging
        self.segment_count = segment_count
        self.dash_segments = dash_segments
        self.solid_segments = solid_segments

    @property
    def point_count(self):
        return len(self.points)

    @property
    def width(self):
        if not self.bbox:
            return 0.0
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self):
        if not self.bbox:
            return 0.0
        return self.bbox[3] - self.bbox[1]

    @property
    def length(self):
        if len(self.points) < 2:
            return 0.0
        pts = np.array(self.points)
        diffs = np.diff(pts, axis=0)
        return float(np.sum(np.sqrt(np.sum(diffs ** 2, axis=1))))


class GeometryExtractor:

    def __init__(self, bezier_steps=8):
        self.bezier_steps = bezier_steps

    def extract(self, page):
        """Pull all vector paths from a page."""
        return self._build_paths(page.get_drawings())

    def extract_region(self, page, y_top, y_bottom, margin=5.0):
        """Pull vector paths only from drawings that overlap a Y range."""
        drawings = page.get_drawings()
        filtered = []
        for d in drawings:
            r = d.get("rect")
            if not r:
                continue
            if r.y1 >= (y_top - margin) and r.y0 <= (y_bottom + margin):
                filtered.append(d)
        return self._build_paths(filtered)

    def _build_paths(self, drawings):
        paths = []

        for idx, drawing in enumerate(drawings):
            points = self._parse_items(drawing.get("items", []))
            if len(points) < 2:
                continue

            rect = drawing.get("rect")
            bbox = None
            if rect:
                bbox = (round(rect.x0, 4), round(rect.y0, 4),
                        round(rect.x1, 4), round(rect.y1, 4))

            color = drawing.get("color")
            if color and isinstance(color, (list, tuple)):
                color = tuple(round(c, 4) for c in color)

            fill = drawing.get("fill")
            if fill and isinstance(fill, (list, tuple)):
                fill = tuple(round(c, 4) for c in fill)

            raw_dashes = drawing.get("dashes")
            dash_str = str(raw_dashes) if raw_dashes else None

            is_seg_dashed = self._has_real_dash(raw_dashes)

            item_types = [item[0] for item in drawing.get("items", [])]

            paths.append(ExtractedPath(
                path_id=idx,
                points=points,
                color=color,
                fill_color=fill,
                stroke_width=drawing.get("width", 0.0) or 0.0,
                dashes=dash_str,
                raw_dashes=raw_dashes,
                is_filled=(fill is not None),
                bbox=bbox,
                item_types=item_types,
                segment_count=1,
                dash_segments=1 if is_seg_dashed else 0,
                solid_segments=0 if is_seg_dashed else 1,
            ))

        return paths

    def _parse_items(self, items):
        """Turn PyMuPDF drawing items into a flat list of (x, y) tuples."""
        points = []

        for item in items:
            kind = item[0]

            if kind == "l":
                self._add(points, item[1])
                self._add(points, item[2])

            elif kind == "c":
                for bp in self._bezier(item[1], item[2], item[3], item[4]):
                    self._add(points, bp)

            elif kind == "re":
                rect = item[1]
                if isinstance(rect, fitz.Rect):
                    for corner in [
                        fitz.Point(rect.x0, rect.y0), fitz.Point(rect.x1, rect.y0),
                        fitz.Point(rect.x1, rect.y1), fitz.Point(rect.x0, rect.y1),
                        fitz.Point(rect.x0, rect.y0),
                    ]:
                        self._add(points, corner)

            elif kind == "qu":
                quad = item[1]
                if isinstance(quad, fitz.Quad):
                    for corner in [quad.ul, quad.ur, quad.lr, quad.ll, quad.ul]:
                        self._add(points, corner)

        return points

    def _add(self, points, pt):
        """Append point, skip consecutive duplicates."""
        if isinstance(pt, fitz.Point):
            x, y = round(pt.x, 4), round(pt.y, 4)
        elif isinstance(pt, (tuple, list)) and len(pt) >= 2:
            x, y = round(pt[0], 4), round(pt[1], 4)
        else:
            return

        if points and points[-1] == (x, y):
            return
        points.append((x, y))

    def _bezier(self, p0, p1, p2, p3):
        """Cubic bezier -> list of interpolated points."""
        def to_arr(p):
            if isinstance(p, fitz.Point):
                return np.array([p.x, p.y])
            return np.array(p[:2])

        P0, P1, P2, P3 = to_arr(p0), to_arr(p1), to_arr(p2), to_arr(p3)
        result = []

        for i in range(self.bezier_steps + 1):
            t = i / self.bezier_steps
            mt = 1 - t
            point = mt**3 * P0 + 3 * mt**2 * t * P1 + 3 * mt * t**2 * P2 + t**3 * P3
            result.append((round(float(point[0]), 4), round(float(point[1]), 4)))

        return result

    @staticmethod
    def _has_real_dash(raw_dashes):
        """Check if dashes value is an actual dash pattern vs just solid."""
        if raw_dashes is None:
            return False
        if isinstance(raw_dashes, str):
            s = raw_dashes.strip()
            return s not in ("", "[] 0", "[]")
        # PyMuPDF gives dashes as tuple/list e.g. ([6], 0) or ([6, 2], 0)
        if isinstance(raw_dashes, (tuple, list)):
            if len(raw_dashes) == 0:
                return False
            dash_array = raw_dashes[0] if isinstance(raw_dashes[0], (list, tuple)) else raw_dashes
            nums = [v for v in dash_array if isinstance(v, (int, float)) and v > 0]
            return len(nums) > 0
        return False