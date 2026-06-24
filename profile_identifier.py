"""
Figures out which extracted paths are the Existing Ground profile
and which is the Proposed Grade. Filters out grid lines and noise first.

Includes path-merging to reassemble profiles that CAD software exported
as many short connected segments.
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
        self.all_after_filter = []  # all paths that survived filtering (for debug plots)
        self.diagnostics = {}


class ProfileIdentifier:

    def __init__(self, grid_coverage=0.75, min_profile_coverage=0.08,
                 min_length=10.0, noise_max_pts=3, noise_max_len=15.0,
                 merge_tolerance=5.0):
        self.grid_coverage = grid_coverage
        self.min_profile_coverage = min_profile_coverage
        self.min_length = min_length
        self.noise_max_pts = noise_max_pts
        self.noise_max_len = noise_max_len
        self.merge_tolerance = merge_tolerance

    def identify(self, paths, page_width, page_height):
        result = IdentifiedProfiles()
        diag = {"total_paths": len(paths), "page_width": page_width, "page_height": page_height}

        # Step 1: filter grid lines
        remaining, grids = self._filter_grids(paths, page_width, page_height)
        result.grid_lines = grids
        diag["grid_lines_removed"] = len(grids)
        diag["after_grid_filter"] = len(remaining)

        # Step 2a: merge nearby connected segments (same color/dash)
        pre_merge_count = len(remaining)
        remaining = self._merge_nearby_paths(remaining)
        diag["pre_merge_paths"] = pre_merge_count
        diag["post_merge_strict"] = len(remaining)

        # Step 2b: second-pass color-tolerant merge (endpoint proximity only)
        pre_pass2 = len(remaining)
        remaining = self._merge_color_tolerant(remaining)
        diag["post_merge_tolerant"] = len(remaining)
        diag["paths_merged"] = pre_merge_count - len(remaining)

        # Step 3: filter noise (short/small paths)
        remaining, noise = self._filter_noise(remaining)
        result.noise = noise
        diag["noise_removed"] = len(noise)

        # Step 4: remove filled paths
        filled_count = sum(1 for p in remaining if p.is_filled)
        remaining = [p for p in remaining if not p.is_filled]
        diag["filled_removed"] = filled_count
        diag["candidates_remaining"] = len(remaining)

        # Save all filtered paths for debug plots
        result.all_after_filter = list(remaining)

        # Step 5: score candidates
        scored, rejected = self._score(remaining, page_width, page_height)
        diag["scored_candidates"] = len(scored)
        diag["rejected_by_coverage"] = len(rejected)

        # Store rejected path details for debugging
        diag["rejected_details"] = rejected[:20]  # top 20 for UI

        if not scored:
            diag["failure_reason"] = (
                f"No paths passed the minimum horizontal coverage threshold "
                f"({self.min_profile_coverage:.0%} of page width = "
                f"{page_width * self.min_profile_coverage:.0f} pt). "
                f"{len(remaining)} candidates were checked."
            )
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

        # Step 6: classify (dashed = existing, solid = proposed)
        self._classify(scored, result, diag)

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

    # ── Grid Filter ──────────────────────────────────────────────────────────

    def _filter_grids(self, paths, page_w, page_h):
        """Remove nearly-straight horizontal/vertical lines spanning most of the region."""
        CLUSTER_TOL = 2.5
        MIN_SEG_LEN = 5.0

        horizontal = []   # (path, y_center)
        vertical = []     # (path, x_center)

        for path in paths:
            if path.point_count < 2:
                continue
            pts = np.array(path.points)
            x_span = pts[:, 0].max() - pts[:, 0].min()
            y_span = pts[:, 1].max() - pts[:, 1].min()

            if y_span <= 3.0 and x_span >= MIN_SEG_LEN:
                horizontal.append((path, float(pts[:, 1].mean())))
            elif x_span <= 3.0 and y_span >= MIN_SEG_LEN:
                vertical.append((path, float(pts[:, 0].mean())))

        h_grid_ids, h_count = self._cluster_grid_lines(horizontal, CLUSTER_TOL)
        v_grid_ids, v_count = self._cluster_grid_lines(vertical, CLUSTER_TOL)

        has_grid = h_count >= 5 and v_count >= 5

        if not has_grid:
            return paths, []

        grid_ids = h_grid_ids | v_grid_ids
        grids = [p for p in paths if id(p) in grid_ids]
        keep = [p for p in paths if id(p) not in grid_ids]
        return keep, grids

    def _cluster_grid_lines(self, items, tolerance):
        """Cluster (path, position) items to find parallel grid lines."""
        if not items:
            return set(), 0
        sorted_items = sorted(items, key=lambda x: x[1])
        clusters = []
        current = [sorted_items[0]]
        for i in range(1, len(sorted_items)):
            if sorted_items[i][1] - current[-1][1] <= tolerance:
                current.append(sorted_items[i])
            else:
                clusters.append(current)
                current = [sorted_items[i]]
        clusters.append(current)

        grid_ids = set()
        for cluster in clusters:
            for path, _ in cluster:
                grid_ids.add(id(path))
        return grid_ids, len(clusters)

    # ── Path Merging ─────────────────────────────────────────────────────────

    def _merge_nearby_paths(self, paths, tolerance=None):
        """Merge unfilled paths with close endpoints and matching visual style.

        CAD software often exports a single continuous profile as many
        short line segments.  This reassembles them by chaining segments
        whose endpoints are within *tolerance* PDF points of each other
        and share the same color + dash pattern.
        """
        if tolerance is None:
            tolerance = self.merge_tolerance
        if len(paths) < 2:
            return paths

        mergeable = [p for p in paths if not p.is_filled and p.point_count >= 2]
        non_mergeable = [p for p in paths if p.is_filled or p.point_count < 2]

        if len(mergeable) < 2:
            return paths

        used = [False] * len(mergeable)
        merged = []

        for i in range(len(mergeable)):
            if used[i]:
                continue

            chain_pts = list(mergeable[i].points)
            chain_color = mergeable[i].color
            chain_dashes = mergeable[i].dashes
            chain_stroke = mergeable[i].stroke_width
            chain_items = list(mergeable[i].item_types)
            used[i] = True

            changed = True
            while changed:
                changed = False
                for j in range(len(mergeable)):
                    if used[j]:
                        continue
                    # only merge matching visual style
                    if mergeable[j].color != chain_color:
                        continue
                    if mergeable[j].dashes != chain_dashes:
                        continue

                    q_start = mergeable[j].points[0]
                    q_end = mergeable[j].points[-1]

                    # try 4 connection orientations
                    if self._pt_dist(chain_pts[-1], q_start) < tolerance:
                        chain_pts.extend(mergeable[j].points[1:])
                        used[j] = True
                        changed = True
                    elif self._pt_dist(chain_pts[-1], q_end) < tolerance:
                        chain_pts.extend(reversed(mergeable[j].points[:-1]))
                        used[j] = True
                        changed = True
                    elif self._pt_dist(chain_pts[0], q_end) < tolerance:
                        chain_pts = list(mergeable[j].points[:-1]) + chain_pts
                        used[j] = True
                        changed = True
                    elif self._pt_dist(chain_pts[0], q_start) < tolerance:
                        chain_pts = list(reversed(mergeable[j].points[1:])) + chain_pts
                        used[j] = True
                        changed = True

            # build merged ExtractedPath
            pts_arr = np.array(chain_pts)
            bbox = (
                round(float(pts_arr[:, 0].min()), 4),
                round(float(pts_arr[:, 1].min()), 4),
                round(float(pts_arr[:, 0].max()), 4),
                round(float(pts_arr[:, 1].max()), 4),
            )
            merged.append(ExtractedPath(
                path_id=mergeable[i].path_id,
                points=chain_pts,
                color=chain_color,
                fill_color=None,
                stroke_width=chain_stroke,
                dashes=chain_dashes,
                is_filled=False,
                bbox=bbox,
                item_types=chain_items,
            ))

        return merged + non_mergeable

    def _merge_color_tolerant(self, paths, tolerance=None):
        """Second-pass merge: join segments by endpoint proximity only.

        Ignores color/dash differences.  Requires similar stroke width
        (within 50%) to avoid merging unrelated drawing elements.
        """
        if tolerance is None:
            tolerance = self.merge_tolerance * 1.5  # slightly larger window
        if len(paths) < 2:
            return paths

        mergeable = [p for p in paths if not p.is_filled and p.point_count >= 2]
        non_mergeable = [p for p in paths if p.is_filled or p.point_count < 2]

        if len(mergeable) < 2:
            return paths

        used = [False] * len(mergeable)
        merged = []

        for i in range(len(mergeable)):
            if used[i]:
                continue

            chain_pts = list(mergeable[i].points)
            chain_stroke = mergeable[i].stroke_width
            chain_color = mergeable[i].color
            chain_dashes = mergeable[i].dashes
            chain_items = list(mergeable[i].item_types)
            used[i] = True

            changed = True
            while changed:
                changed = False
                for j in range(len(mergeable)):
                    if used[j]:
                        continue
                    # require similar stroke width (within 50%)
                    sw_ratio = (mergeable[j].stroke_width / chain_stroke
                                if chain_stroke > 0 else 1.0)
                    if sw_ratio < 0.5 or sw_ratio > 2.0:
                        continue

                    q_start = mergeable[j].points[0]
                    q_end = mergeable[j].points[-1]

                    if self._pt_dist(chain_pts[-1], q_start) < tolerance:
                        chain_pts.extend(mergeable[j].points[1:])
                        used[j] = True
                        changed = True
                    elif self._pt_dist(chain_pts[-1], q_end) < tolerance:
                        chain_pts.extend(reversed(mergeable[j].points[:-1]))
                        used[j] = True
                        changed = True
                    elif self._pt_dist(chain_pts[0], q_end) < tolerance:
                        chain_pts = list(mergeable[j].points[:-1]) + chain_pts
                        used[j] = True
                        changed = True
                    elif self._pt_dist(chain_pts[0], q_start) < tolerance:
                        chain_pts = list(reversed(mergeable[j].points[1:])) + chain_pts
                        used[j] = True
                        changed = True

            pts_arr = np.array(chain_pts)
            bbox = (
                round(float(pts_arr[:, 0].min()), 4),
                round(float(pts_arr[:, 1].min()), 4),
                round(float(pts_arr[:, 0].max()), 4),
                round(float(pts_arr[:, 1].max()), 4),
            )
            merged.append(ExtractedPath(
                path_id=mergeable[i].path_id,
                points=chain_pts,
                color=chain_color,
                fill_color=None,
                stroke_width=chain_stroke,
                dashes=chain_dashes,
                is_filled=False,
                bbox=bbox,
                item_types=chain_items,
            ))

        return merged + non_mergeable

    @staticmethod
    def _pt_dist(p1, p2):
        return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

    # ── Noise Filter ─────────────────────────────────────────────────────────

    def _filter_noise(self, paths):
        noise, keep = [], []
        for path in paths:
            too_short = path.point_count <= self.noise_max_pts and path.length < self.noise_max_len
            if too_short or path.length < self.min_length:
                noise.append(path)
            else:
                keep.append(path)
        return keep, noise

    # ── Scoring ──────────────────────────────────────────────────────────────

    def _score(self, paths, page_w, page_h):
        """Score candidates. Returns (scored, rejected_details)."""
        scored = []
        rejected = []

        min_height = max(8.0, page_h * 0.03)  # reject nearly-flat paths

        for path in paths:
            pts = np.array(path.points)
            x_range = pts[:, 0].max() - pts[:, 0].min()
            y_range = pts[:, 1].max() - pts[:, 1].min()
            h_cov = x_range / page_w

            rej_info = {
                "path_id": path.path_id,
                "h_cov": round(h_cov, 4),
                "h_cov_pct": f"{h_cov:.1%}",
                "points": path.point_count,
                "length": round(path.length, 1),
                "width": round(x_range, 1),
                "height": round(y_range, 1),
                "dashes": path.dashes,
                "color": str(path.color),
            }

            if h_cov < self.min_profile_coverage:
                rej_info["reason"] = f"coverage {h_cov:.1%} < {self.min_profile_coverage:.0%}"
                rejected.append(rej_info)
                continue

            # reject paths much taller than wide — likely borders
            if y_range > page_h * 0.6:
                rej_info["reason"] = f"too tall: height {y_range:.0f} > {page_h * 0.6:.0f}"
                rejected.append(rej_info)
                continue

            # reject nearly-flat paths — zigzag hatch marks, decorative elements
            if y_range < min_height:
                rej_info["reason"] = f"too flat: height {y_range:.1f} < {min_height:.1f} (zigzag/decoration)"
                rejected.append(rej_info)
                continue

            # detect zigzag patterns: many rapid Y-reversals relative to height
            if path.point_count >= 6:
                y_vals = pts[:, 1]
                dy = np.diff(y_vals)
                sign_changes = np.sum(np.abs(np.diff(np.sign(dy))) > 0)
                reversal_rate = sign_changes / max(path.point_count, 1)
                # zigzag: high reversal rate AND low height → decorative hatching
                if reversal_rate > 0.5 and y_range < page_h * 0.08:
                    rej_info["reason"] = (
                        f"zigzag pattern: {sign_changes} reversals in "
                        f"{path.point_count} pts (rate={reversal_rate:.2f}), "
                        f"height={y_range:.1f}"
                    )
                    rejected.append(rej_info)
                    continue

            score = 0.0
            score += h_cov * 40
            score += min(path.point_count / 50, 1.0) * 20
            score += min(path.length / (page_w * 0.8), 1.0) * 20

            # moderate aspect ratio bonus — profiles are wide-ish but NOT perfectly flat
            if y_range > 0:
                aspect = x_range / y_range
                # sweet spot: aspect ratio 3-15 (typical profile), penalize extremes
                if aspect > 30:
                    score += 2  # very flat → low bonus (likely decoration)
                elif aspect > 15:
                    score += 6
                else:
                    score += 10  # good profile-like proportions

            y_center = pts[:, 1].mean()
            center_bonus = 1 - abs(y_center - page_h / 2) / (page_h / 2)
            score += max(0.0, center_bonus) * 10

            scored.append((path, float(score)))

        scored.sort(key=lambda x: x[1], reverse=True)
        rejected.sort(key=lambda x: x["h_cov"], reverse=True)
        return scored, rejected

    # ── Classification ───────────────────────────────────────────────────────

    def _classify(self, scored, result, diag=None):
        """Dashed = existing ground, Solid = proposed grade."""
        if diag is None:
            diag = {}

        top = scored[:min(20, len(scored))]
        dashed = [(p, s) for p, s in top if self._is_dashed(p)]
        solid = [(p, s) for p, s in top if not self._is_dashed(p)]

        diag["classify_dashed_count"] = len(dashed)
        diag["classify_solid_count"] = len(solid)

        if dashed and solid:
            result.existing_ground = dashed[0][0]
            result.proposed_grade = solid[0][0]
            diag["classification_rule"] = "dashed→EG, solid→PG"
        elif len(solid) >= 2:
            solid.sort(key=lambda x: x[0].stroke_width, reverse=True)
            result.proposed_grade = solid[0][0]
            result.existing_ground = solid[1][0]
            diag["classification_rule"] = "all solid → stroke width fallback"
        elif len(dashed) >= 2:
            dashed.sort(key=lambda x: x[0].stroke_width, reverse=True)
            result.proposed_grade = dashed[0][0]
            result.existing_ground = dashed[1][0]
            diag["classification_rule"] = "all dashed → stroke width fallback"
        elif len(scored) >= 2:
            result.proposed_grade = scored[0][0]
            result.existing_ground = scored[1][0]
            diag["classification_rule"] = "score-rank fallback (top 2)"
        elif len(scored) == 1:
            result.proposed_grade = scored[0][0]
            diag["classification_rule"] = "only 1 candidate → PG only, no EG"
            diag["failure_reason"] = "Only 1 scored candidate — cannot assign Existing Ground."
        else:
            diag["classification_rule"] = "no candidates"
            diag["failure_reason"] = "No scored candidates at all."

    def _is_dashed(self, path):
        """Return True only for actual dash patterns, not PyMuPDF's solid [] marker."""
        if not path.dashes:
            return False
        dash = str(path.dashes).strip()
        return dash not in ("[] 0", "[]")