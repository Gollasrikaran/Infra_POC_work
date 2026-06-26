"""
Figures out which PDF vector paths are Existing Ground vs Proposed Grade.
Strips grids, merges fragmented CAD segments, then classifies using
dash patterns, color, roughness, position and stroke width.
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
        self.scored_list = []       # (path, score) tuples
        self.all_after_filter = []  # everything that made it past filtering
        self.classification_confidence = 0.0
        self.classification_signals = {}
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

        # get rid of grid lines
        remaining, grids = self._filter_grids(paths, page_width, page_height)
        result.grid_lines = grids
        diag["grid_lines_removed"] = len(grids)
        diag["after_grid_filter"] = len(remaining)

        # yank out bridge outlines early so merge doesn't glue them into PG
        remaining, bridge_pre = self._filter_bridge_structures(remaining)
        diag["bridge_structures_removed_pre_merge"] = len(bridge_pre)

        # first merge pass -- strict color+dash matching
        pre_merge = len(remaining)
        remaining = self._merge_nearby_paths(remaining)
        diag["pre_merge_paths"] = pre_merge
        diag["post_merge_strict"] = len(remaining)

        # second merge -- looser on color, still won't mix dashed with solid
        pre_pass2 = len(remaining)
        remaining = self._merge_color_tolerant(remaining, page_width)
        diag["post_merge_tolerant"] = len(remaining)
        diag["paths_merged"] = pre_merge - len(remaining)

        # throw away small stuff
        remaining, noise = self._filter_noise(remaining)
        result.noise = noise
        diag["noise_removed"] = len(noise)

        # second bridge check -- some may have formed during merge
        remaining, bridge_post = self._filter_bridge_structures(remaining)
        diag["bridge_structures_removed"] = len(bridge_pre) + len(bridge_post)
        diag["bridge_structures_removed_post_merge"] = len(bridge_post)

        # kill slope marker lines (short steep solid segments near "2:1" labels)
        remaining, anno_lines = self._filter_annotation_lines(
            remaining, page_width, page_height
        )
        diag["annotation_lines_removed"] = len(anno_lines)

        # toss filled paths (hatching, regions)
        n_filled = sum(1 for p in remaining if p.is_filled)
        remaining = [p for p in remaining if not p.is_filled]
        diag["filled_removed"] = n_filled
        diag["candidates_remaining"] = len(remaining)

        result.all_after_filter = list(remaining)

        # score what's left
        scored, rejected = self._score(remaining, page_width, page_height)
        diag["scored_candidates"] = len(scored)
        diag["rejected_by_coverage"] = len(rejected)
        diag["rejected_details"] = rejected[:20]

        # rescue pass -- PG sometimes only covers 5-7% of page width,
        # which is below the default 5% threshold after rounding.  retry at 3%.
        RESCUE_COV = 0.03
        if len(scored) <= 1 and RESCUE_COV < self.min_profile_coverage:
            rescued, rescued_rej = self._score(
                remaining, page_width, page_height,
                override_min_coverage=RESCUE_COV
            )
            already = {id(p) for p, s in scored}
            new_ones = [(p, s) for p, s in rescued if id(p) not in already]
            if new_ones:
                scored = scored + new_ones
                scored.sort(key=lambda x: x[1], reverse=True)
                diag["rescue_pass"] = True
                diag["rescue_candidates_added"] = len(new_ones)
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

        # dump top 10 for debug UI
        diag["top_candidates"] = []
        for rank, (p, sc) in enumerate(scored[:10]):
            diag["top_candidates"].append({
                "rank": rank + 1,
                "score": round(sc, 2),
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

        # figure out which is EG and which is PG
        self._classify(scored, result, diag, page_height)

        # if PG barely has any points, try stitching nearby solid fragments
        # into a composite road template
        if result.proposed_grade and result.proposed_grade.point_count < 10:
            comp = self._reassemble_pg(
                remaining, result.proposed_grade, result.existing_ground,
                page_width, page_height
            )
            if comp and comp.point_count > result.proposed_grade.point_count:
                diag["pg_reassembly"] = True
                diag["pg_reassembly_old_pts"] = result.proposed_grade.point_count
                diag["pg_reassembly_new_pts"] = comp.point_count
                result.proposed_grade = comp
                comp_scored, _ = self._score(
                    [comp], page_width, page_height,
                    override_min_coverage=0.01
                )
                if comp_scored:
                    scored.append(comp_scored[0])
                    scored.sort(key=lambda x: x[1], reverse=True)

        # populate final output
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

    # -----------------------------------------------------------------
    #  grid line removal
    # -----------------------------------------------------------------

    def _filter_grids(self, paths, page_w, page_h):
        """Zap horizontal/vertical lines that form the background grid."""
        CLUSTER_TOL = 2.5
        MIN_SEG_LEN = 5.0

        horiz = []  # (path, y_center)
        vert = []   # (path, x_center)

        for path in paths:
            if path.point_count < 2:
                continue
            pts = np.array(path.points)
            xs = pts[:, 0].max() - pts[:, 0].min()
            ys = pts[:, 1].max() - pts[:, 1].min()

            if ys <= 3.0 and xs >= MIN_SEG_LEN:
                horiz.append((path, float(pts[:, 1].mean())))
            elif xs <= 3.0 and ys >= MIN_SEG_LEN:
                vert.append((path, float(pts[:, 0].mean())))

        h_ids, h_cnt = self._cluster_grid_lines(horiz, CLUSTER_TOL)
        v_ids, v_cnt = self._cluster_grid_lines(vert, CLUSTER_TOL)

        if h_cnt < 5 or v_cnt < 5:
            return paths, []

        all_grid_ids = h_ids | v_ids
        grids = [p for p in paths if id(p) in all_grid_ids]
        keep = [p for p in paths if id(p) not in all_grid_ids]
        return keep, grids

    def _cluster_grid_lines(self, items, tol):
        """Group lines by position to find parallel grid lines."""
        if not items:
            return set(), 0
        items_sorted = sorted(items, key=lambda x: x[1])
        clusters = []
        cur = [items_sorted[0]]
        for i in range(1, len(items_sorted)):
            if items_sorted[i][1] - cur[-1][1] <= tol:
                cur.append(items_sorted[i])
            else:
                clusters.append(cur)
                cur = [items_sorted[i]]
        clusters.append(cur)

        ids = set()
        for cl in clusters:
            for path, _ in cl:
                ids.add(id(path))
        return ids, len(clusters)

    # -----------------------------------------------------------------
    #  merging fragmented paths back together
    # -----------------------------------------------------------------

    def _merge_nearby_paths(self, paths, tolerance=None):
        """Reconnect paths that CAD exported as separate segments.
        Only merges when color and dash pattern match exactly."""
        if tolerance is None:
            tolerance = self.merge_tolerance
        if len(paths) < 2:
            return paths

        can_merge = [p for p in paths if not p.is_filled and p.point_count >= 2]
        cant_merge = [p for p in paths if p.is_filled or p.point_count < 2]

        if len(can_merge) < 2:
            return paths

        used = [False] * len(can_merge)
        merged = []

        for i in range(len(can_merge)):
            if used[i]:
                continue

            chain = list(can_merge[i].points)
            col = can_merge[i].color
            dsh = can_merge[i].dashes
            sw = can_merge[i].stroke_width
            itypes = list(can_merge[i].item_types)
            seg_cnt = can_merge[i].segment_count
            d_segs = can_merge[i].dash_segments
            s_segs = can_merge[i].solid_segments
            used[i] = True

            keep_going = True
            while keep_going:
                keep_going = False
                for j in range(len(can_merge)):
                    if used[j]:
                        continue
                    if can_merge[j].color != col:
                        continue
                    if can_merge[j].dashes != dsh:
                        continue

                    # try connecting with angle check -- 140 degrees lets
                    # road template slope transitions through (2:1 to 4:1 etc)
                    ok, new_chain = self._try_connect_with_angle(
                        chain, can_merge[j].points, tolerance,
                        max_angle_deg=140
                    )
                    if ok:
                        chain = new_chain
                        seg_cnt += can_merge[j].segment_count
                        d_segs += can_merge[j].dash_segments
                        s_segs += can_merge[j].solid_segments
                        used[j] = True
                        keep_going = True

            arr = np.array(chain)
            bb = (
                round(float(arr[:, 0].min()), 4),
                round(float(arr[:, 1].min()), 4),
                round(float(arr[:, 0].max()), 4),
                round(float(arr[:, 1].max()), 4),
            )
            merged.append(ExtractedPath(
                path_id=can_merge[i].path_id,
                points=chain,
                color=col,
                fill_color=None,
                stroke_width=sw,
                dashes=dsh,
                is_filled=False,
                bbox=bb,
                item_types=itypes,
                segment_count=seg_cnt,
                dash_segments=d_segs,
                solid_segments=s_segs,
            ))

        return merged + cant_merge

    def _merge_color_tolerant(self, paths, page_width=None, tolerance=None):
        """Second merge pass -- ignores color but won't cross dashed/solid
        boundary. Also blocks merging two already-wide paths together."""
        if tolerance is None:
            tolerance = self.merge_tolerance * 1.2
        if len(paths) < 2:
            return paths

        can_merge = [p for p in paths if not p.is_filled and p.point_count >= 2]
        cant_merge = [p for p in paths if p.is_filled or p.point_count < 2]

        if len(can_merge) < 2:
            return paths

        WIDE = (page_width or 1000) * 0.25

        used = [False] * len(can_merge)
        merged = []

        for i in range(len(can_merge)):
            if used[i]:
                continue

            chain = list(can_merge[i].points)
            sw = can_merge[i].stroke_width
            col = can_merge[i].color
            dsh = can_merge[i].dashes
            itypes = list(can_merge[i].item_types)
            seg_cnt = can_merge[i].segment_count
            d_segs = can_merge[i].dash_segments
            s_segs = can_merge[i].solid_segments
            is_dashed = self._is_dashed(can_merge[i])
            used[i] = True

            keep_going = True
            while keep_going:
                keep_going = False
                c_arr = np.array(chain)
                c_width = c_arr[:, 0].max() - c_arr[:, 0].min()

                for j in range(len(can_merge)):
                    if used[j]:
                        continue
                    # stroke width sanity check
                    sw_r = (can_merge[j].stroke_width / sw if sw > 0 else 1.0)
                    if sw_r < 0.5 or sw_r > 2.0:
                        continue

                    # don't glue two big paths together
                    jw = can_merge[j].width
                    if c_width > WIDE and jw > WIDE:
                        continue

                    # keep dashed and solid apart
                    if self._is_dashed(can_merge[j]) != is_dashed:
                        continue

                    ok, new_chain = self._try_connect_with_angle(
                        chain, can_merge[j].points, tolerance,
                        max_angle_deg=120
                    )
                    if ok:
                        chain = new_chain
                        seg_cnt += can_merge[j].segment_count
                        d_segs += can_merge[j].dash_segments
                        s_segs += can_merge[j].solid_segments
                        used[j] = True
                        keep_going = True

            arr = np.array(chain)
            bb = (
                round(float(arr[:, 0].min()), 4),
                round(float(arr[:, 1].min()), 4),
                round(float(arr[:, 0].max()), 4),
                round(float(arr[:, 1].max()), 4),
            )
            merged.append(ExtractedPath(
                path_id=can_merge[i].path_id,
                points=chain,
                color=col,
                fill_color=None,
                stroke_width=sw,
                dashes=dsh,
                is_filled=False,
                bbox=bb,
                item_types=itypes,
                segment_count=seg_cnt,
                dash_segments=d_segs,
                solid_segments=s_segs,
            ))

        return merged + cant_merge

    # -----------------------------------------------------------------
    #  endpoint connection helpers
    # -----------------------------------------------------------------

    def _try_connect(self, chain, other, tol):
        """Try all 4 ways to attach other to chain. Return (ok, new_pts)."""
        s, e = other[0], other[-1]

        if self._pt_dist(chain[-1], s) < tol:
            return True, chain + list(other[1:])
        if self._pt_dist(chain[-1], e) < tol:
            return True, chain + list(reversed(other[:-1]))
        if self._pt_dist(chain[0], e) < tol:
            return True, list(other[:-1]) + chain
        if self._pt_dist(chain[0], s) < tol:
            return True, list(reversed(other[1:])) + chain
        return False, chain

    def _try_connect_with_angle(self, chain, other, tol, max_angle_deg=120):
        """Same as _try_connect but rejects connections that make a sharp bend."""
        s, e = other[0], other[-1]

        options = []

        # chain_end -> other_start
        d = self._pt_dist(chain[-1], s)
        if d < tol:
            pts = chain + list(other[1:])
            options.append((d, pts, len(chain) - 1))

        # chain_end -> other_end (flip other)
        d = self._pt_dist(chain[-1], e)
        if d < tol:
            pts = chain + list(reversed(other[:-1]))
            options.append((d, pts, len(chain) - 1))

        # other_end -> chain_start
        d = self._pt_dist(chain[0], e)
        if d < tol:
            pts = list(other[:-1]) + chain
            options.append((d, pts, len(other) - 2))

        # other_start (flip) -> chain_start
        d = self._pt_dist(chain[0], s)
        if d < tol:
            pts = list(reversed(other[1:])) + chain
            options.append((d, pts, len(other) - 2))

        if not options:
            return False, chain

        # prefer closest that passes the angle test
        options.sort(key=lambda x: x[0])

        for dist, pts, jidx in options:
            if self._check_angle(pts, jidx, max_angle_deg):
                return True, pts

        # nothing passed -- try again more leniently (150 deg)
        for dist, pts, jidx in options:
            if self._check_angle(pts, jidx, 150):
                return True, pts

        return False, chain

    def _check_angle(self, pts, idx, max_deg):
        """Make sure the bend at pts[idx] isn't too sharp."""
        if idx <= 0 or idx >= len(pts) - 1:
            return True

        a = np.array(pts[idx - 1])
        b = np.array(pts[idx])
        c = np.array(pts[idx + 1])

        v1 = b - a
        v2 = c - b
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)

        if n1 < 0.01 or n2 < 0.01:
            return True  # degenerate, just allow it

        cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)
        deg = np.degrees(np.arccos(cos_a))
        return deg < max_deg

    @staticmethod
    def _pt_dist(p1, p2):
        return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

    # -----------------------------------------------------------------
    #  noise removal
    # -----------------------------------------------------------------

    def _filter_noise(self, paths):
        noise, keep = [], []
        for p in paths:
            tiny = p.point_count <= self.noise_max_pts and p.length < self.noise_max_len
            if tiny or p.length < self.min_length:
                noise.append(p)
            else:
                keep.append(p)
        return keep, noise

    # -----------------------------------------------------------------
    #  bridge / structure outlines
    # -----------------------------------------------------------------

    def _filter_bridge_structures(self, paths):
        """Catch rectangular bridge deck outlines before they get absorbed
        into profiles. They're flat, boxy, and have sharp corners."""
        keep, bridges = [], []

        for p in paths:
            if p.point_count < 4:
                keep.append(p)
                continue

            pts = np.array(p.points)
            xspan = pts[:, 0].max() - pts[:, 0].min()
            yspan = pts[:, 1].max() - pts[:, 1].min()

            if xspan < 1.0:
                keep.append(p)
                continue

            aspect = yspan / xspan
            corners = self._count_sharp_corners(pts, min_angle_deg=70)
            start_end = self._pt_dist(p.points[0], p.points[-1])
            nearly_closed = start_end < max(p.length * 0.10, 5.0)

            is_bridge = False

            # closed rectangular outline
            if nearly_closed and aspect < 0.20 and corners >= 3:
                is_bridge = True

            # open but very flat with lots of right-angle turns
            if not is_bridge and aspect < 0.12 and corners >= 4:
                is_bridge = True

            # Y values cluster into 2-3 levels (deck top + bottom)
            if not is_bridge and aspect < 0.18 and corners >= 2:
                ylev = self._count_y_levels(pts[:, 1], tolerance=2.0)
                if ylev <= 3 and p.point_count >= 6:
                    is_bridge = True

            if is_bridge:
                bridges.append(p)
            else:
                keep.append(p)

        return keep, bridges

    @staticmethod
    def _count_sharp_corners(pts, min_angle_deg=70):
        """How many corners deviate more than min_angle_deg from straight."""
        if len(pts) < 3:
            return 0
        n = 0
        for i in range(1, len(pts) - 1):
            v1 = pts[i] - pts[i - 1]
            v2 = pts[i + 1] - pts[i]
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 < 0.01 or n2 < 0.01:
                continue
            cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)
            if np.degrees(np.arccos(cos_a)) > min_angle_deg:
                n += 1
        return n

    @staticmethod
    def _count_y_levels(y_vals, tolerance=2.0):
        """Count distinct Y clusters."""
        if len(y_vals) == 0:
            return 0
        sy = np.sort(y_vals)
        lvl = 1
        prev = sy[0]
        for y in sy[1:]:
            if y - prev > tolerance:
                lvl += 1
            prev = y
        return lvl

    # -----------------------------------------------------------------
    #  annotation / slope marker line filter
    # -----------------------------------------------------------------

    def _filter_annotation_lines(self, paths, page_w, page_h):
        """Removes the short steep lines near slope labels (2:1, 4:1 etc).
        These are solid, have few points, and angle steeply from horizontal."""
        MAX_LEN_FRAC = 0.12
        MIN_ANGLE = 50  # degrees from horizontal
        MAX_PTS = 4

        keep, removed = [], []
        max_len = page_w * MAX_LEN_FRAC

        for p in paths:
            if p.point_count > MAX_PTS or self._is_dashed(p):
                keep.append(p)
                continue

            pts = np.array(p.points)
            xspan = pts[:, 0].max() - pts[:, 0].min()
            yspan = pts[:, 1].max() - pts[:, 1].min()

            if p.length > max_len:
                keep.append(p)
                continue

            if xspan < 0.1:
                ang = 90.0
            else:
                ang = np.degrees(np.arctan2(yspan, xspan))

            if ang > MIN_ANGLE:
                removed.append(p)
            else:
                keep.append(p)

        return keep, removed

    # -----------------------------------------------------------------
    #  PG reassembly from fragments
    # -----------------------------------------------------------------

    def _reassemble_pg(self, all_remaining, pg_cand, eg_cand, page_w, page_h):
        """When PG only has a handful of points, gather nearby solid fragments
        in the same Y band and chain them left to right."""
        if pg_cand is None:
            return None

        pg_pts = np.array(pg_cand.points)
        pg_y = float(pg_pts[:, 1].mean())

        # look within +/- 20% of region height
        band = page_h * 0.20
        y_lo, y_hi = pg_y - band, pg_y + band

        eg_id = id(eg_cand) if eg_cand else None
        pg_id = id(pg_cand)

        frags = []
        for p in all_remaining:
            if id(p) == eg_id or id(p) == pg_id:
                continue
            if self._is_dashed(p) or p.is_filled or p.point_count < 2:
                continue
            mean_y = float(np.array(p.points)[:, 1].mean())
            if y_lo <= mean_y <= y_hi:
                frags.append(p)

        frags.append(pg_cand)  # include original PG too

        if len(frags) < 2:
            return None

        # sort by leftmost x
        frags.sort(key=lambda f: min(pt[0] for pt in f.points))

        # greedy left-to-right chaining
        comp = list(frags[0].points)
        taken = {0}
        tot_d = frags[0].dash_segments
        tot_s = frags[0].solid_segments
        tot_n = frags[0].segment_count

        gap_limit = self.merge_tolerance * 3

        changed = True
        while changed:
            changed = False
            cend = comp[-1]
            cstart = comp[0]

            best_j, best_d, best_how, best_rev = None, float("inf"), None, False

            for j in range(len(frags)):
                if j in taken:
                    continue
                fp = frags[j].points
                fs, fe = fp[0], fp[-1]

                d = self._pt_dist(cend, fs)
                if d < best_d:
                    best_d, best_j, best_how, best_rev = d, j, "append", False

                d = self._pt_dist(cend, fe)
                if d < best_d:
                    best_d, best_j, best_how, best_rev = d, j, "append", True

                d = self._pt_dist(fe, cstart)
                if d < best_d:
                    best_d, best_j, best_how, best_rev = d, j, "prepend", False

                d = self._pt_dist(fs, cstart)
                if d < best_d:
                    best_d, best_j, best_how, best_rev = d, j, "prepend", True

            if best_j is not None and best_d < gap_limit:
                frag = frags[best_j]
                fp = list(frag.points)
                if best_rev:
                    fp = list(reversed(fp))

                if best_how == "append":
                    test = comp + fp[1:] if best_d < 2.0 else comp + fp
                    jidx = len(comp) - 1
                    if self._check_angle(test, jidx, 150):
                        comp = test
                        taken.add(best_j)
                        tot_d += frag.dash_segments
                        tot_s += frag.solid_segments
                        tot_n += frag.segment_count
                        changed = True
                else:
                    test = fp[:-1] + comp if best_d < 2.0 else fp + comp
                    jidx = len(fp) - 2 if best_d < 2.0 else len(fp) - 1
                    if self._check_angle(test, jidx, 150):
                        comp = test
                        taken.add(best_j)
                        tot_d += frag.dash_segments
                        tot_s += frag.solid_segments
                        tot_n += frag.segment_count
                        changed = True

        if len(comp) <= pg_cand.point_count:
            return None

        arr = np.array(comp)
        bb = (
            round(float(arr[:, 0].min()), 4),
            round(float(arr[:, 1].min()), 4),
            round(float(arr[:, 0].max()), 4),
            round(float(arr[:, 1].max()), 4),
        )
        return ExtractedPath(
            path_id=pg_cand.path_id,
            points=comp,
            color=pg_cand.color,
            fill_color=None,
            stroke_width=pg_cand.stroke_width,
            dashes=pg_cand.dashes,
            is_filled=False,
            bbox=bb,
            item_types=pg_cand.item_types,
            segment_count=tot_n,
            dash_segments=tot_d,
            solid_segments=tot_s,
        )

    # -----------------------------------------------------------------
    #  scoring
    # -----------------------------------------------------------------

    def _score(self, paths, page_w, page_h, override_min_coverage=None):
        """Rate each path on how profile-like it is. Returns (scored, rejected)."""
        min_cov = override_min_coverage if override_min_coverage is not None else self.min_profile_coverage
        scored = []
        rejected = []

        min_h = max(8.0, page_h * 0.03)

        for path in paths:
            pts = np.array(path.points)
            xr = pts[:, 0].max() - pts[:, 0].min()
            yr = pts[:, 1].max() - pts[:, 1].min()
            hcov = xr / page_w

            info = {
                "path_id": path.path_id,
                "h_cov": round(hcov, 4),
                "h_cov_pct": f"{hcov:.1%}",
                "points": path.point_count,
                "length": round(path.length, 1),
                "width": round(xr, 1),
                "height": round(yr, 1),
                "dashes": path.dashes,
                "color": str(path.color),
            }

            if hcov < min_cov:
                info["reason"] = f"coverage {hcov:.1%} < {min_cov:.0%}"
                rejected.append(info)
                continue

            if yr > page_h * 0.6:
                info["reason"] = f"too tall: {yr:.0f} > {page_h * 0.6:.0f}"
                rejected.append(info)
                continue

            if yr < min_h:
                info["reason"] = f"too flat: {yr:.1f} < {min_h:.1f}"
                rejected.append(info)
                continue

            # catch zigzag hatching: tons of Y reversals but very short height
            if path.point_count >= 6:
                yv = pts[:, 1]
                dy = np.diff(yv)
                flips = np.sum(np.abs(np.diff(np.sign(dy))) > 0)
                rate = flips / max(path.point_count, 1)
                if rate > 0.5 and yr < page_h * 0.08:
                    info["reason"] = (
                        f"zigzag: {flips} reversals in {path.point_count} pts "
                        f"(rate={rate:.2f}), height={yr:.1f}"
                    )
                    rejected.append(info)
                    continue

            sc = 0.0
            sc += hcov * 40
            sc += min(path.point_count / 50, 1.0) * 20
            sc += min(path.length / (page_w * 0.8), 1.0) * 20

            # aspect ratio bonus -- profiles are wide but not perfectly flat
            if yr > 0:
                ar = xr / yr
                if ar > 30:
                    sc += 2   # way too flat, probably not a profile
                elif ar > 15:
                    sc += 6
                else:
                    sc += 10  # good proportions

            # slight bonus for being near vertical center of region
            yc = pts[:, 1].mean()
            cb = 1 - abs(yc - page_h / 2) / (page_h / 2)
            sc += max(0.0, cb) * 10

            scored.append((path, float(sc)))

        scored.sort(key=lambda x: x[1], reverse=True)
        rejected.sort(key=lambda x: x["h_cov"], reverse=True)
        return scored, rejected

    # -----------------------------------------------------------------
    #  dash helpers
    # -----------------------------------------------------------------

    def _is_dashed(self, path):
        """True if path has a real dash pattern (not just the default [] solid)."""
        if not path.dashes:
            return False
        d = str(path.dashes).strip()
        return d not in ("[] 0", "[]")

    def _dash_ratio(self, path):
        """What fraction of the merged segments were dashed (0 = solid, 1 = dashed)."""
        if path is None:
            return 0.0
        total = path.dash_segments + path.solid_segments
        if total == 0:
            return 1.0 if self._is_dashed(path) else 0.0
        return path.dash_segments / total

    # -----------------------------------------------------------------
    #  roughness / curvature
    # -----------------------------------------------------------------

    @staticmethod
    def _roughness(path):
        """Mean absolute 2nd derivative of Y w.r.t. X.
        Natural terrain is jagged, designed grades are smooth."""
        if path.point_count < 4:
            return 0.0
        pts = np.array(path.points)
        order = np.argsort(pts[:, 0])
        x = pts[order, 0]
        y = pts[order, 1]

        # toss duplicate x
        dx = np.diff(x)
        mask = dx > 0.01
        x = np.concatenate([[x[0]], x[1:][mask]])
        y = np.concatenate([[y[0]], y[1:][mask]])

        if len(x) < 4:
            return 0.0

        dy = np.diff(y) / np.diff(x)
        d2y = np.diff(dy)
        dx2 = (x[2:] - x[:-2]) / 2.0
        d2y_dx2 = d2y / np.where(dx2 > 0.01, dx2, 0.01)

        return float(np.mean(np.abs(d2y_dx2)))

    # -----------------------------------------------------------------
    #  color bucketing
    # -----------------------------------------------------------------

    @staticmethod
    def _color_category(color):
        """Bucket an RGB tuple into broad categories for EG/PG heuristics."""
        if color is None:
            return "black"
        if not isinstance(color, (tuple, list)) or len(color) < 3:
            return "other"

        r, g, b = color[0], color[1], color[2]

        if r < 0.15 and g < 0.15 and b < 0.15:
            return "black"
        if abs(r - g) < 0.1 and abs(g - b) < 0.1 and r > 0.15:
            return "gray"
        if r > 0.4 and g < 0.35 and b < 0.35:
            return "brown" if r < 0.7 else "red"
        if b > 0.4 and r < 0.35 and g < 0.35:
            return "blue"
        if g > 0.4 and r < 0.35 and b < 0.35:
            return "green"
        return "other"

    @staticmethod
    def _mean_y(path):
        """Average Y in PDF coords (bigger = lower on page)."""
        if path.point_count == 0:
            return 0.0
        return float(np.array(path.points)[:, 1].mean())

    # -----------------------------------------------------------------
    #  EG vs PG classification
    # -----------------------------------------------------------------

    def _classify(self, scored, result, diag=None, page_height=None):
        """Decide which candidate is EG and which is PG.

        Tries dash ratio first (best signal), then falls back to
        color, roughness, position and stroke width."""
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

        # primary signal: dash ratio
        dr_info = [(p, s, self._dash_ratio(p)) for p, s in top]

        dashed = [(p, s, dr) for p, s, dr in dr_info if dr > 0.6]
        solid = [(p, s, dr) for p, s, dr in dr_info if dr < 0.4]
        ambig = [(p, s, dr) for p, s, dr in dr_info if 0.4 <= dr <= 0.6]

        diag["classify_dashed_count"] = len(dashed)
        diag["classify_solid_count"] = len(solid)
        diag["classify_ambiguous_count"] = len(ambig)

        signals = []
        conf = 0.0

        if dashed and solid:
            # clear split -- best case
            eg = max(dashed, key=lambda x: x[1])[0]
            pg = max(solid, key=lambda x: x[1])[0]
            result.existing_ground = eg
            result.proposed_grade = pg
            signals.append("dash_ratio")
            conf = 0.9
            diag["classification_rule"] = "dash_ratio: dashed=EG, solid=PG"

        else:
            # ambiguous -- fall back to secondary signals on top 2
            c1, s1 = top[0]
            c2, s2 = top[1]

            # each signal votes: +1 means c1 is EG, -1 means c2 is EG
            votes = {}

            # color
            cat1 = self._color_category(c1.color)
            cat2 = self._color_category(c2.color)
            EG_COLS = {"brown", "red", "gray"}
            PG_COLS = {"black", "blue"}
            if cat1 in EG_COLS and cat2 in PG_COLS:
                votes["color"] = +1
            elif cat2 in EG_COLS and cat1 in PG_COLS:
                votes["color"] = -1
            elif cat1 != cat2:
                if cat1 in EG_COLS:
                    votes["color"] = +0.5
                elif cat2 in EG_COLS:
                    votes["color"] = -0.5

            # roughness
            r1 = self._roughness(c1)
            r2 = self._roughness(c2)
            if r1 > 0 or r2 > 0:
                mx = max(r1, r2, 0.001)
                rdiff = (r1 - r2) / mx
                if abs(rdiff) > 0.2:
                    votes["roughness"] = +1 if rdiff > 0 else -1
                    signals.append("roughness")

            # vertical position (higher Y in PDF = lower on page = usually EG)
            my1 = self._mean_y(c1)
            my2 = self._mean_y(c2)
            if page_height and page_height > 0:
                ydiff = (my1 - my2) / page_height
                if abs(ydiff) > 0.02:
                    votes["vertical_pos"] = +1 if ydiff > 0 else -1
                    signals.append("vertical_pos")

            # stroke width (thicker usually = PG)
            w1, w2 = c1.stroke_width, c2.stroke_width
            if w1 > 0 and w2 > 0:
                wr = w1 / w2
                if wr > 1.3:
                    votes["stroke_width"] = -1  # c1 thicker -> c1 is PG -> c2 is EG
                elif wr < 0.7:
                    votes["stroke_width"] = +1

            # partial dash info
            dr1 = self._dash_ratio(c1)
            dr2 = self._dash_ratio(c2)
            if dr1 > dr2 + 0.2:
                votes["dash_partial"] = +1
                signals.append("dash_partial")
            elif dr2 > dr1 + 0.2:
                votes["dash_partial"] = -1
                signals.append("dash_partial")

            # tally
            wt = {
                "dash_partial": 3.0,
                "color": 2.0,
                "roughness": 1.5,
                "vertical_pos": 1.0,
                "stroke_width": 1.0,
            }

            total = sum(votes.get(k, 0) * wt.get(k, 1.0) for k in wt)

            if total > 0:
                result.existing_ground = c1
                result.proposed_grade = c2
            elif total < 0:
                result.existing_ground = c2
                result.proposed_grade = c1
            else:
                # dead tie -- higher score gets PG (it's usually more prominent)
                result.proposed_grade = c1
                result.existing_ground = c2

            max_w = sum(abs(wt[k]) for k in wt)
            conf = min(0.8, abs(total) / max_w + 0.2)

            parts = [f"{k}={v:+.1f}" for k, v in votes.items()]
            diag["classification_rule"] = f"multi-signal: {', '.join(parts)} -> total={total:+.1f}"
            diag["classification_votes"] = votes

        result.classification_confidence = conf
        result.classification_signals = {
            "signals_used": signals,
            "confidence": round(conf, 2),
            "method": diag.get("classification_rule", "unknown"),
        }