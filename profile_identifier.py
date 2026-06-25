"""
Figures out which extracted paths are the Existing Ground profile
and which is the Proposed Grade. Filters out grid lines and noise first.

Includes path-merging to reassemble profiles that CAD software exported
as many short connected segments.

Classification signals (in priority order):
  1. Dash-ratio (statistical, surviving merge)
  2. Color grouping (EG often brown/red/gray; PG often black/blue)
  3. Roughness / curvature (EG = natural terrain = more jagged)
  4. Vertical position (EG often lower on the graph)
  5. Stroke width (thicker = proposed, fallback)
"""

import numpy as np
from geometry_extractor import ExtractedPath, GeometryExtractor


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
        self.classification_confidence = 0.0   # 0..1 confidence
        self.classification_signals = {}       # detailed signal breakdown
        self.diagnostics = {}


class ProfileIdentifier:

    def __init__(self, grid_coverage=0.75, min_profile_coverage=0.05,
                 min_length=10.0, noise_max_pts=3, noise_max_len=15.0,
                 merge_tolerance=6.0):
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
        #          but WITH dash-pattern protection to prevent cross-profile merging
        pre_pass2 = len(remaining)
        remaining = self._merge_color_tolerant(remaining, page_width)
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

        # ── Rescue pass: if only 0-1 candidates, retry with lower threshold ──
        # The PG line often spans only the road prism (±40-60 ft), giving
        # only 5-7% coverage of the full page width.  A rescue pass with 3%
        # threshold catches these narrow PG fragments.
        RESCUE_COVERAGE = 0.03
        if len(scored) <= 1 and RESCUE_COVERAGE < self.min_profile_coverage:
            rescued, rescued_rej = self._score(
                remaining, page_width, page_height,
                override_min_coverage=RESCUE_COVERAGE
            )
            # deduplicate: only add paths not already in scored
            existing_path_ids = {id(p) for p, s in scored}
            new_rescued = [(p, s) for p, s in rescued if id(p) not in existing_path_ids]
            if new_rescued:
                scored = scored + new_rescued
                scored.sort(key=lambda x: x[1], reverse=True)
                diag["rescue_pass"] = True
                diag["rescue_candidates_added"] = len(new_rescued)
                diag["scored_candidates"] = len(scored)
            else:
                diag["rescue_pass"] = True
                diag["rescue_candidates_added"] = 0

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
                "dash_ratio": round(self._dash_ratio(p), 2),
                "roughness": round(self._roughness(p), 4),
                "stroke_w": round(p.stroke_width, 2),
                "color": str(p.color),
                "segment_count": p.segment_count,
                "dash_segments": p.dash_segments,
                "solid_segments": p.solid_segments,
            })

        # Step 6: classify (multi-signal cascade)
        self._classify(scored, result, diag, page_height)

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
        diag["existing_ground_dash_ratio"] = round(self._dash_ratio(result.existing_ground), 2) if result.existing_ground else None
        diag["proposed_grade_points"] = len(result.proposed_grade_points)
        diag["proposed_grade_length"] = round(result.proposed_grade.length, 1) if result.proposed_grade else 0
        diag["proposed_grade_score"] = round(result.proposed_grade_score, 2)
        diag["proposed_grade_dashes"] = result.proposed_grade.dashes if result.proposed_grade else None
        diag["proposed_grade_dash_ratio"] = round(self._dash_ratio(result.proposed_grade), 2) if result.proposed_grade else None
        diag["classification_confidence"] = round(result.classification_confidence, 2)
        diag["classification_signals"] = result.classification_signals
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

        Now also tracks dash/solid segment counts through the merge.
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
            # track merge statistics
            chain_seg_count = mergeable[i].segment_count
            chain_dash_segs = mergeable[i].dash_segments
            chain_solid_segs = mergeable[i].solid_segments
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
                    connected, new_pts = self._try_connect(
                        chain_pts, mergeable[j].points, tolerance
                    )
                    if connected:
                        chain_pts = new_pts
                        chain_seg_count += mergeable[j].segment_count
                        chain_dash_segs += mergeable[j].dash_segments
                        chain_solid_segs += mergeable[j].solid_segments
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
                segment_count=chain_seg_count,
                dash_segments=chain_dash_segs,
                solid_segments=chain_solid_segs,
            ))

        return merged + non_mergeable

    def _merge_color_tolerant(self, paths, page_width=None, tolerance=None):
        """Second-pass merge: join segments by endpoint proximity only.

        Ignores color differences but RESPECTS dash pattern differences
        to prevent merging EG (dashed) with PG (solid) at their crossing
        points. Requires similar stroke width (within 50%).

        PROTECTION:
         - Paths whose horizontal span > 25% of page width won't merge
           with each other (they're likely two separate profiles).
         - Dashed chains won't absorb solid segments, and vice versa.
         - Angle consistency check prevents sharp-turn joins.
        """
        if tolerance is None:
            tolerance = self.merge_tolerance * 1.2
        if len(paths) < 2:
            return paths

        mergeable = [p for p in paths if not p.is_filled and p.point_count >= 2]
        non_mergeable = [p for p in paths if p.is_filled or p.point_count < 2]

        if len(mergeable) < 2:
            return paths

        # Pre-compute widths for cross-merge protection
        WIDE_THRESHOLD = (page_width or 1000) * 0.25

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
            chain_seg_count = mergeable[i].segment_count
            chain_dash_segs = mergeable[i].dash_segments
            chain_solid_segs = mergeable[i].solid_segments
            # track whether the chain is predominantly dashed or solid
            chain_is_dashed = self._is_dashed(mergeable[i])
            used[i] = True

            changed = True
            while changed:
                changed = False
                # compute current chain width for protection check
                chain_arr = np.array(chain_pts)
                chain_width = chain_arr[:, 0].max() - chain_arr[:, 0].min()

                for j in range(len(mergeable)):
                    if used[j]:
                        continue
                    # require similar stroke width (within 50%)
                    sw_ratio = (mergeable[j].stroke_width / chain_stroke
                                if chain_stroke > 0 else 1.0)
                    if sw_ratio < 0.5 or sw_ratio > 2.0:
                        continue

                    # PROTECTION 1: don't merge two wide paths together
                    j_width = mergeable[j].width
                    if chain_width > WIDE_THRESHOLD and j_width > WIDE_THRESHOLD:
                        continue

                    # PROTECTION 2: don't merge dashed with solid
                    # This prevents EG (dashed) from absorbing PG (solid) segments
                    # at intersection points
                    j_is_dashed = self._is_dashed(mergeable[j])
                    if chain_is_dashed != j_is_dashed:
                        continue

                    # Try connection with angle consistency check
                    connected, new_pts = self._try_connect_with_angle(
                        chain_pts, mergeable[j].points, tolerance,
                        max_angle_deg=120  # reject sharp turns > 120°
                    )
                    if connected:
                        chain_pts = new_pts
                        chain_seg_count += mergeable[j].segment_count
                        chain_dash_segs += mergeable[j].dash_segments
                        chain_solid_segs += mergeable[j].solid_segments
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
                segment_count=chain_seg_count,
                dash_segments=chain_dash_segs,
                solid_segments=chain_solid_segs,
            ))

        return merged + non_mergeable

    # ── Connection Helpers ────────────────────────────────────────────────────

    def _try_connect(self, chain_pts, other_pts, tolerance):
        """Try 4 orientations to connect other_pts to chain_pts.
        Returns (connected, new_chain_pts) or (False, chain_pts)."""
        q_start = other_pts[0]
        q_end = other_pts[-1]

        if self._pt_dist(chain_pts[-1], q_start) < tolerance:
            return True, chain_pts + list(other_pts[1:])
        elif self._pt_dist(chain_pts[-1], q_end) < tolerance:
            return True, chain_pts + list(reversed(other_pts[:-1]))
        elif self._pt_dist(chain_pts[0], q_end) < tolerance:
            return True, list(other_pts[:-1]) + chain_pts
        elif self._pt_dist(chain_pts[0], q_start) < tolerance:
            return True, list(reversed(other_pts[1:])) + chain_pts
        return False, chain_pts

    def _try_connect_with_angle(self, chain_pts, other_pts, tolerance,
                                max_angle_deg=120):
        """Like _try_connect but also checks the angle at the junction.
        Prevents merging segments that would create a sharp turn."""
        q_start = other_pts[0]
        q_end = other_pts[-1]

        candidates = []

        # orientation 1: chain_end → other_start
        d = self._pt_dist(chain_pts[-1], q_start)
        if d < tolerance:
            new_pts = chain_pts + list(other_pts[1:])
            join_idx = len(chain_pts) - 1
            candidates.append((d, new_pts, join_idx))

        # orientation 2: chain_end → other_end (reversed)
        d = self._pt_dist(chain_pts[-1], q_end)
        if d < tolerance:
            new_pts = chain_pts + list(reversed(other_pts[:-1]))
            join_idx = len(chain_pts) - 1
            candidates.append((d, new_pts, join_idx))

        # orientation 3: other_end → chain_start
        d = self._pt_dist(chain_pts[0], q_end)
        if d < tolerance:
            new_pts = list(other_pts[:-1]) + chain_pts
            join_idx = len(other_pts) - 2
            candidates.append((d, new_pts, join_idx))

        # orientation 4: other_start (reversed) → chain_start
        d = self._pt_dist(chain_pts[0], q_start)
        if d < tolerance:
            new_pts = list(reversed(other_pts[1:])) + chain_pts
            join_idx = len(other_pts) - 2
            candidates.append((d, new_pts, join_idx))

        if not candidates:
            return False, chain_pts

        # pick the closest connection that passes the angle check
        candidates.sort(key=lambda x: x[0])  # sort by distance

        for dist, new_pts, join_idx in candidates:
            if self._angle_ok(new_pts, join_idx, max_angle_deg):
                return True, new_pts

        # none passed angle check — fall back to closest if angle is moderate
        # (be more permissive: allow up to 150° for the closest match)
        for dist, new_pts, join_idx in candidates:
            if self._angle_ok(new_pts, join_idx, 150):
                return True, new_pts

        return False, chain_pts

    def _angle_ok(self, pts, join_idx, max_angle_deg):
        """Check that the angle at join_idx is not too sharp."""
        if join_idx <= 0 or join_idx >= len(pts) - 1:
            return True  # can't check angle at endpoints

        p_before = np.array(pts[join_idx - 1])
        p_join = np.array(pts[join_idx])
        p_after = np.array(pts[join_idx + 1])

        v1 = p_join - p_before
        v2 = p_after - p_join

        len1 = np.linalg.norm(v1)
        len2 = np.linalg.norm(v2)

        if len1 < 0.01 or len2 < 0.01:
            return True  # degenerate segment, allow

        cos_angle = np.dot(v1, v2) / (len1 * len2)
        cos_angle = np.clip(cos_angle, -1, 1)
        angle_deg = np.degrees(np.arccos(cos_angle))

        # angle_deg = 0 means straight continuation, 180 means reversal
        # We want to REJECT if the turn is too sharp
        # A "sharp turn" is when the angle deviates significantly from straight (0°)
        return angle_deg < max_angle_deg

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

    def _score(self, paths, page_w, page_h, override_min_coverage=None):
        """Score candidates. Returns (scored, rejected_details).

        If *override_min_coverage* is given it replaces the instance-level
        ``min_profile_coverage`` for this call (used by the rescue pass).
        """
        min_cov = override_min_coverage if override_min_coverage is not None else self.min_profile_coverage
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

            if h_cov < min_cov:
                rej_info["reason"] = f"coverage {h_cov:.1%} < {min_cov:.0%}"
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

    # ── Dash Analysis Helpers ────────────────────────────────────────────────

    def _is_dashed(self, path):
        """Return True only for actual dash patterns, not PyMuPDF's solid [] marker."""
        if not path.dashes:
            return False
        dash = str(path.dashes).strip()
        return dash not in ("[] 0", "[]")

    def _dash_ratio(self, path):
        """Fraction of constituent segments that were dashed (0.0 = fully solid,
        1.0 = fully dashed). Falls back to binary _is_dashed if no merge stats."""
        if path is None:
            return 0.0
        total = path.dash_segments + path.solid_segments
        if total == 0:
            return 1.0 if self._is_dashed(path) else 0.0
        return path.dash_segments / total

    # ── Roughness / Curvature ────────────────────────────────────────────────

    @staticmethod
    def _roughness(path):
        """Measure profile roughness as mean absolute second-derivative of Y
        with respect to X. Natural terrain (EG) is rougher; designed grades (PG)
        are smoother with longer linear runs.

        Returns 0.0 for paths with < 4 points.
        """
        if path.point_count < 4:
            return 0.0
        pts = np.array(path.points)
        # sort by x
        order = np.argsort(pts[:, 0])
        x = pts[order, 0]
        y = pts[order, 1]

        # remove duplicate x values
        dx = np.diff(x)
        mask = dx > 0.01
        x = np.concatenate([[x[0]], x[1:][mask]])
        y = np.concatenate([[y[0]], y[1:][mask]])

        if len(x) < 4:
            return 0.0

        # first derivative (slope)
        dy_dx = np.diff(y) / np.diff(x)
        # second derivative (curvature proxy)
        d2y = np.diff(dy_dx)
        dx2 = (x[2:] - x[:-2]) / 2.0  # average spacing for second derivative
        d2y_dx2 = d2y / np.where(dx2 > 0.01, dx2, 0.01)

        return float(np.mean(np.abs(d2y_dx2)))

    # ── Color Analysis ───────────────────────────────────────────────────────

    @staticmethod
    def _color_category(color):
        """Classify a PyMuPDF RGB color tuple into broad categories useful for
        distinguishing EG from PG lines.

        Returns one of: 'black', 'red', 'blue', 'green', 'brown', 'gray', 'other'
        """
        if color is None:
            return "black"  # default stroke is black
        if not isinstance(color, (tuple, list)) or len(color) < 3:
            return "other"

        r, g, b = color[0], color[1], color[2]

        # black / near-black
        if r < 0.15 and g < 0.15 and b < 0.15:
            return "black"
        # gray
        if abs(r - g) < 0.1 and abs(g - b) < 0.1 and r > 0.15:
            return "gray"
        # red / brown
        if r > 0.4 and g < 0.35 and b < 0.35:
            return "brown" if r < 0.7 else "red"
        # blue
        if b > 0.4 and r < 0.35 and g < 0.35:
            return "blue"
        # green
        if g > 0.4 and r < 0.35 and b < 0.35:
            return "green"
        return "other"

    # ── Vertical Position ────────────────────────────────────────────────────

    @staticmethod
    def _mean_y(path):
        """Mean Y coordinate in PDF space (higher Y = lower on page = typically lower elevation)."""
        if path.point_count == 0:
            return 0.0
        pts = np.array(path.points)
        return float(pts[:, 1].mean())

    # ── Classification ───────────────────────────────────────────────────────

    def _classify(self, scored, result, diag=None, page_height=None):
        """Multi-signal classification cascade for EG vs PG.

        Priority:
          1. Dash ratio (dashed -> EG, solid -> PG)
          2. Color grouping (brown/red/gray -> EG, black/blue -> PG)
          3. Roughness (rougher -> EG, smoother -> PG)
          4. Vertical position (higher mean Y in PDF = lower elevation -> EG)
          5. Stroke width (fallback: thicker -> PG)
        """
        if diag is None:
            diag = {}

        top = scored[:min(20, len(scored))]
        if len(top) == 0:
            diag["classification_rule"] = "no candidates"
            diag["failure_reason"] = "No scored candidates at all."
            return

        if len(top) == 1:
            result.proposed_grade = top[0][0]
            result.classification_confidence = 0.3
            result.classification_signals = {"method": "single_candidate"}
            diag["classification_rule"] = "only 1 candidate -> PG only, no EG"
            diag["failure_reason"] = "Only 1 scored candidate -- cannot assign Existing Ground."
            return

        # -- Signal 1: Dash ratio --
        # Compute dash ratios for all top candidates
        dash_ratios = [(p, s, self._dash_ratio(p)) for p, s in top]

        # Split by dash confidence
        dashed_cands = [(p, s, dr) for p, s, dr in dash_ratios if dr > 0.6]
        solid_cands = [(p, s, dr) for p, s, dr in dash_ratios if dr < 0.4]
        ambiguous_cands = [(p, s, dr) for p, s, dr in dash_ratios if 0.4 <= dr <= 0.6]

        diag["classify_dashed_count"] = len(dashed_cands)
        diag["classify_solid_count"] = len(solid_cands)
        diag["classify_ambiguous_count"] = len(ambiguous_cands)

        signals_used = []
        confidence = 0.0

        if dashed_cands and solid_cands:
            # Clear dash/solid separation -- highest confidence
            # Pick the best-scored from each group
            eg_cand = max(dashed_cands, key=lambda x: x[1])[0]
            pg_cand = max(solid_cands, key=lambda x: x[1])[0]
            result.existing_ground = eg_cand
            result.proposed_grade = pg_cand
            signals_used.append("dash_ratio")
            confidence = 0.9
            diag["classification_rule"] = "dash_ratio: dashed=EG, solid=PG"

        else:
            # Dash detection inconclusive -- use secondary signals
            # Take top-2 candidates by score
            c1, s1 = top[0]
            c2, s2 = top[1]

            # Gather signal votes: each signal votes for which candidate is EG
            # vote = +1 means c1 is EG, vote = -1 means c2 is EG
            votes = {}

            # Signal 2: Color grouping
            cat1 = self._color_category(c1.color)
            cat2 = self._color_category(c2.color)
            EG_COLORS = {"brown", "red", "gray"}
            PG_COLORS = {"black", "blue"}
            if cat1 in EG_COLORS and cat2 in PG_COLORS:
                votes["color"] = +1  # c1 is EG
            elif cat2 in EG_COLORS and cat1 in PG_COLORS:
                votes["color"] = -1  # c2 is EG
            elif cat1 != cat2:
                # different colors, slight signal
                if cat1 in EG_COLORS:
                    votes["color"] = +0.5
                elif cat2 in EG_COLORS:
                    votes["color"] = -0.5

            # Signal 3: Roughness
            rough1 = self._roughness(c1)
            rough2 = self._roughness(c2)
            if rough1 > 0 or rough2 > 0:
                max_rough = max(rough1, rough2, 0.001)
                rough_diff = (rough1 - rough2) / max_rough
                if abs(rough_diff) > 0.2:  # meaningful difference
                    votes["roughness"] = +1 if rough_diff > 0 else -1
                    signals_used.append("roughness")

            # Signal 4: Vertical position (higher mean Y in PDF -> lower on page -> typically EG)
            my1 = self._mean_y(c1)
            my2 = self._mean_y(c2)
            if page_height and page_height > 0:
                y_diff = (my1 - my2) / page_height
                if abs(y_diff) > 0.02:  # meaningful vertical separation
                    votes["vertical_pos"] = +1 if y_diff > 0 else -1
                    signals_used.append("vertical_pos")

            # Signal 5: Stroke width (thicker -> PG)
            sw1 = c1.stroke_width
            sw2 = c2.stroke_width
            if sw1 > 0 and sw2 > 0:
                sw_ratio = sw1 / sw2
                if sw_ratio > 1.3:
                    votes["stroke_width"] = -1  # c1 thicker -> c1 is PG -> c2 is EG
                elif sw_ratio < 0.7:
                    votes["stroke_width"] = +1  # c2 thicker -> c2 is PG -> c1 is EG

            # Also consider partial dash info even when not clear-cut
            dr1 = self._dash_ratio(c1)
            dr2 = self._dash_ratio(c2)
            if dr1 > dr2 + 0.2:
                votes["dash_partial"] = +1  # c1 more dashed -> c1 is EG
                signals_used.append("dash_partial")
            elif dr2 > dr1 + 0.2:
                votes["dash_partial"] = -1  # c2 more dashed -> c2 is EG
                signals_used.append("dash_partial")

            # Tally weighted votes
            weights = {
                "dash_partial": 3.0,
                "color": 2.0,
                "roughness": 1.5,
                "vertical_pos": 1.0,
                "stroke_width": 1.0,
            }

            total_vote = sum(votes.get(k, 0) * weights.get(k, 1.0) for k in weights)

            if total_vote > 0:
                result.existing_ground = c1
                result.proposed_grade = c2
            elif total_vote < 0:
                result.existing_ground = c2
                result.proposed_grade = c1
            else:
                # Complete tie -- fall back to score rank
                # Higher-scored path as PG (proposed grade is often more prominent)
                result.proposed_grade = c1
                result.existing_ground = c2

            # Compute confidence from vote strength
            max_possible = sum(abs(weights[k]) for k in weights)
            confidence = min(0.8, abs(total_vote) / max_possible + 0.2)

            method_parts = [f"{k}={v:+.1f}" for k, v in votes.items()]
            diag["classification_rule"] = f"multi-signal cascade: {', '.join(method_parts)} -> total={total_vote:+.1f}"
            diag["classification_votes"] = votes

        result.classification_confidence = confidence
        result.classification_signals = {
            "signals_used": signals_used,
            "confidence": round(confidence, 2),
            "method": diag.get("classification_rule", "unknown"),
        }