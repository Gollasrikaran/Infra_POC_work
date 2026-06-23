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
        self.existing_ground_score = 0.0
        self.proposed_grade_score = 0.0
        self.grid_lines = []
        self.noise = []
        self.candidates = []
        self.scored_list = []       # [(path, score), ...] top candidates
        self.diagnostics = {}


class ProfileIdentifier:

    def __init__(self, grid_coverage=0.75, min_profile_coverage=0.25,
                 min_length=10.0, noise_max_pts=3, noise_max_len=15.0,
                 min_grid_lines=5, grid_straightness=3.0):
        self.grid_coverage = grid_coverage
        self.min_profile_coverage = min_profile_coverage
        self.min_length = min_length
        self.noise_max_pts = noise_max_pts
        self.noise_max_len = noise_max_len
        self.min_grid_lines = min_grid_lines
        self.grid_straightness = grid_straightness

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
        result.scored_list = scored[:10]

        # store top-10 candidate details for debugging
        diag["top_candidates"] = []
        for i, (p, s) in enumerate(scored[:10]):
            diag["top_candidates"].append({
                "rank": i + 1,
                "score": round(s, 2),
                "points": p.point_count,
                "length": round(p.length, 1),
                "width": round(p.width, 1),
                "height": round(p.height, 1),
                "dashes": p.dashes,
                "is_dashed": self._is_dashed(p),
                "stroke_w": round(p.stroke_width, 2),
                "color": str(p.color),
            })

        self._classify(scored, result)

        if result.existing_ground:
            result.existing_ground_points = sorted(result.existing_ground.points, key=lambda p: p[0])
            result.existing_ground_score = next(
                (s for p, s in scored if p is result.existing_ground), 0.0
            )
        if result.proposed_grade:
            result.proposed_grade_points = sorted(result.proposed_grade.points, key=lambda p: p[0])
            result.proposed_grade_score = next(
                (s for p, s in scored if p is result.proposed_grade), 0.0
            )

        diag["existing_ground_points"] = len(result.existing_ground_points)
        diag["existing_ground_length"] = round(result.existing_ground.length, 1) if result.existing_ground else 0
        diag["existing_ground_score"] = round(result.existing_ground_score, 2)
        diag["existing_ground_dashes"] = result.existing_ground.dashes if result.existing_ground else None
        diag["proposed_grade_points"] = len(result.proposed_grade_points)
        diag["proposed_grade_length"] = round(result.proposed_grade.length, 1) if result.proposed_grade else 0
        diag["proposed_grade_score"] = round(result.proposed_grade_score, 2)
        diag["proposed_grade_dashes"] = result.proposed_grade.dashes if result.proposed_grade else None
        result.diagnostics = diag
        return result

    def _filter_grids(self, paths, page_w, page_h):
        """Filter grid lines using clustering of nearly-straight segments.

        Detects grid patterns by:
        1. Identifying all nearly-horizontal segments (small y_span)
           and nearly-vertical segments (small x_span)
        2. Clustering horizontal segments by Y position and vertical by X position
        3. If enough clusters exist in both directions → grid detected
        4. Removing all segments belonging to grid clusters

        This handles PDFs where grid lines are drawn as many short segments
        rather than single page-spanning lines.
        """
        CLUSTER_TOL = 2.5     # merge segments within this distance as same grid line
        MIN_SEG_LEN = 5.0     # ignore very tiny segments for grid detection

        horizontal = []   # (path, y_center)
        vertical = []     # (path, x_center)

        for path in paths:
            if path.point_count < 2:
                continue

            pts = np.array(path.points)
            x_span = pts[:, 0].max() - pts[:, 0].min()
            y_span = pts[:, 1].max() - pts[:, 1].min()

            # Nearly-horizontal: small vertical variation, some horizontal extent
            if y_span <= self.grid_straightness and x_span >= MIN_SEG_LEN:
                horizontal.append((path, float(pts[:, 1].mean())))
            # Nearly-vertical: small horizontal variation, some vertical extent
            elif x_span <= self.grid_straightness and y_span >= MIN_SEG_LEN:
                vertical.append((path, float(pts[:, 0].mean())))

        # Cluster and count distinct grid lines in each direction
        h_grid_ids, h_cluster_count = self._find_grid_clusters(horizontal, CLUSTER_TOL)
        v_grid_ids, v_cluster_count = self._find_grid_clusters(vertical, CLUSTER_TOL)

        # A real grid has multiple lines in BOTH directions
        has_grid = (h_cluster_count >= self.min_grid_lines
                    and v_cluster_count >= self.min_grid_lines)

        if not has_grid:
            # No grid detected — return all paths unchanged
            return paths, []

        # Removal pass: separate grid segments from real geometry
        grid_ids = h_grid_ids | v_grid_ids
        grids = [p for p in paths if id(p) in grid_ids]
        keep = [p for p in paths if id(p) not in grid_ids]

        return keep, grids

    def _find_grid_clusters(self, items, tolerance):
        """Cluster (path, position) items by position to find grid lines.

        Groups nearly-colinear segments together (e.g., multiple short
        horizontal segments at the same Y coordinate = one grid line).

        Returns:
            (grid_path_ids, num_clusters) — set of path ids and count of
            distinct grid lines detected.
        """
        if not items:
            return set(), 0

        # Sort by position (Y for horizontal, X for vertical)
        sorted_items = sorted(items, key=lambda x: x[1])

        # Build clusters of segments at similar positions
        clusters = []
        current_cluster = [sorted_items[0]]

        for i in range(1, len(sorted_items)):
            if sorted_items[i][1] - current_cluster[-1][1] <= tolerance:
                current_cluster.append(sorted_items[i])
            else:
                clusters.append(current_cluster)
                current_cluster = [sorted_items[i]]
        clusters.append(current_cluster)

        # Collect all path ids from all clusters
        grid_ids = set()
        for cluster in clusters:
            for path, _ in cluster:
                grid_ids.add(id(path))

        return grid_ids, len(clusters)

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
            if self._is_frame_or_border(path, page_w, page_h):
                continue

            pts = np.array(path.points)
            x_range = pts[:, 0].max() - pts[:, 0].min()
            y_range = pts[:, 1].max() - pts[:, 1].min()
            h_cov = x_range / page_w

            is_dashed = self._is_dashed(path)
            min_cov = self.min_profile_coverage if is_dashed else 0.005
            if h_cov < min_cov:
                continue

            score = 0.0
            score += h_cov * 40
            score += min(path.point_count / 50, 1.0) * 20
            score += min(path.length / (page_w * 0.8), 1.0) * 20
            if y_range > 0:
                score += min((x_range / y_range) / 10, 1.0) * 10
            y_center = pts[:, 1].mean()
            center_bonus = 1 - abs(y_center - page_h / 2) / (page_h / 2)
            score += max(0.0, center_bonus) * 10

            scored.append((path, float(score)))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def _classify(self, scored, result):
        """Dashed = existing ground, Solid = proposed grade."""
        top = scored[:min(20, len(scored))]
        dashed = [(p, s) for p, s in top if self._is_dashed(p)]
        solid = [(p, s) for p, s in top if not self._is_dashed(p)]

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

    def _is_dashed(self, path):
        """Return True only for actual dash patterns, not PyMuPDF's solid [] marker."""
        if not path.dashes:
            return False
        dash = str(path.dashes).strip()
        return dash not in ("[] 0", "[]")

    def _is_frame_or_border(self, path, page_w, region_h):
        """Reject sheet/viewport borders that overlap a region but are not profiles."""
        if not path.bbox:
            return False

        very_wide = path.width >= page_w * 0.85
        much_taller_than_region = path.height >= max(region_h * 1.5, 100.0)
        is_closed_quad = "qu" in path.item_types and path.point_count <= 5

        return very_wide and much_taller_than_region and is_closed_quad
