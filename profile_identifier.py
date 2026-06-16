"""
Figures out which extracted paths are the Existing Ground profile
and which is the Proposed Grade. Filters out grid lines and noise first.
"""

import numpy as np
from geometry_extractor import ExtractedPath


class IdentifiedProfiles:
    def __init__(self):
        self.existing_ground = None
        self.proposed_grade = None
        self.existing_ground_points = []
        self.proposed_grade_points = []
        self.grid_lines = []
        self.noise = []
        self.candidates = []
        self.diagnostics = {}


class ProfileIdentifier:

    def __init__(self, grid_coverage=0.75, min_profile_coverage=0.25,
                 min_length=10.0, noise_max_pts=3, noise_max_len=15.0):
        self.grid_coverage = grid_coverage
        self.min_profile_coverage = min_profile_coverage
        self.min_length = min_length
        self.noise_max_pts = noise_max_pts
        self.noise_max_len = noise_max_len

    def identify(self, paths, page_width, page_height):
        result = IdentifiedProfiles()
        diag = {"total_paths": len(paths), "page_width": page_width, "page_height": page_height}

        remaining, grids = self._filter_grids(paths, page_width, page_height)
        result.grid_lines = grids
        diag["grid_lines_removed"] = len(grids)

        remaining, noise = self._filter_noise(remaining)
        result.noise = noise
        diag["noise_removed"] = len(noise)
        diag["candidates_remaining"] = len(remaining)

        remaining = [p for p in remaining if not p.is_filled]

        scored = self._score(remaining, page_width, page_height)
        diag["scored_candidates"] = len(scored)

        if not scored:
            result.diagnostics = diag
            return result

        result.candidates = [s[0] for s in scored[:20]]

        self._classify(scored, result)

        if result.existing_ground:
            result.existing_ground_points = sorted(result.existing_ground.points, key=lambda p: p[0])
        if result.proposed_grade:
            result.proposed_grade_points = sorted(result.proposed_grade.points, key=lambda p: p[0])

        diag["existing_ground_points"] = len(result.existing_ground_points)
        diag["proposed_grade_points"] = len(result.proposed_grade_points)
        result.diagnostics = diag
        return result

    def _filter_grids(self, paths, page_w, page_h):
        grids, keep = [], []
        h_thresh = page_w * self.grid_coverage
        v_thresh = page_h * self.grid_coverage

        for path in paths:
            if path.point_count < 2:
                keep.append(path)
                continue

            pts = np.array(path.points)
            x_span = pts[:, 0].max() - pts[:, 0].min()
            y_span = pts[:, 1].max() - pts[:, 1].min()

            if (x_span > h_thresh and y_span < 2.0) or (y_span > v_thresh and x_span < 2.0):
                grids.append(path)
            else:
                keep.append(path)

        return keep, grids

    def _filter_noise(self, paths):
        noise, keep = [], []
        for path in paths:
            too_short = path.point_count <= self.noise_max_pts and path.length < self.noise_max_len
            if too_short or path.length < self.min_length:
                noise.append(path)
            else:
                keep.append(path)
        return keep, noise

    def _score(self, paths, page_w, page_h):
        scored = []

        for path in paths:
            pts = np.array(path.points)
            x_range = pts[:, 0].max() - pts[:, 0].min()
            y_range = pts[:, 1].max() - pts[:, 1].min()
            h_cov = x_range / page_w

            if h_cov < self.min_profile_coverage:
                continue

            score = 0.0
            score += h_cov * 40
            score += min(path.point_count / 50, 1.0) * 20
            score += min(path.length / (page_w * 0.8), 1.0) * 20
            if y_range > 0:
                score += min((x_range / y_range) / 10, 1.0) * 10
            y_center = pts[:, 1].mean()
            score += (1 - abs(y_center - page_h / 2) / (page_h / 2)) * 10

            scored.append((path, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def _classify(self, scored, result):
        """Dashed = existing ground, Solid = proposed grade."""
        top = scored[:min(10, len(scored))]
        dashed = [(p, s) for p, s in top if p.dashes is not None]
        solid = [(p, s) for p, s in top if p.dashes is None]

        if dashed and solid:
            result.existing_ground = dashed[0][0]
            result.proposed_grade = solid[0][0]
        elif len(solid) >= 2:
            solid.sort(key=lambda x: x[0].stroke_width, reverse=True)
            result.proposed_grade = solid[0][0]
            result.existing_ground = solid[1][0]
        elif len(dashed) >= 2:
            dashed.sort(key=lambda x: x[0].stroke_width, reverse=True)
            result.proposed_grade = dashed[0][0]
            result.existing_ground = dashed[1][0]
        elif len(scored) >= 2:
            result.proposed_grade = scored[0][0]
            result.existing_ground = scored[1][0]
        elif len(scored) == 1:
            result.proposed_grade = scored[0][0]
