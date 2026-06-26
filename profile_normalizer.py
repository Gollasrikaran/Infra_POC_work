"""
Aligns EG and PG profiles to a common set of offsets so we can do
point-by-point cut/fill comparison. Only the overlapping range is used.
"""

import numpy as np
from scipy import interpolate


class NormalizedProfiles:
    def __init__(self, stations, existing, proposed, interval, station_range):
        self.stations = stations
        self.existing_elevations = existing
        self.proposed_elevations = proposed
        self.station_interval = interval
        self.station_range = station_range
        self.diagnostics = {}


class ProfileNormalizer:

    def __init__(self, interval=1.0, method="linear"):
        self.interval = interval
        self.method = method

    def normalize(self, existing, proposed, sta_start=None, sta_end=None):
        if len(existing.stations) < 2:
            raise ValueError(f"EG has too few points: {len(existing.stations)}")
        if len(proposed.stations) < 2:
            raise ValueError(f"PG has too few points: {len(proposed.stations)}")

        # overlap range -- only compute earthwork where both profiles exist
        eg_min, eg_max = float(existing.stations.min()), float(existing.stations.max())
        pg_min, pg_max = float(proposed.stations.min()), float(proposed.stations.max())

        if sta_start is None:
            sta_start = max(eg_min, pg_min)
        if sta_end is None:
            sta_end = min(eg_max, pg_max)

        if sta_end <= sta_start:
            raise ValueError(
                f"Profiles don't overlap -- EG [{existing.station_range}], "
                f"PG [{proposed.station_range}]"
            )

        stations = np.arange(sta_start, sta_end + self.interval / 2, self.interval)

        ex_interp = self._interp(existing.stations, existing.elevations, stations)
        pr_interp = self._interp(proposed.stations, proposed.elevations, stations)

        result = NormalizedProfiles(stations, ex_interp, pr_interp, self.interval,
                                   (float(sta_start), float(sta_end)))
        result.diagnostics = {
            "existing_points": len(existing.stations),
            "proposed_points": len(proposed.stations),
            "existing_range": (eg_min, eg_max),
            "proposed_range": (pg_min, pg_max),
            "overlap_range": (sta_start, sta_end),
            "overlap_width": sta_end - sta_start,
            "common_stations": len(stations),
            "method": self.method,
            "interval": self.interval,
        }
        return result

    def _interp(self, x, y, x_new):
        """Interpolate within known range, clamp to edge values outside."""
        valid = ~(np.isnan(x) | np.isnan(y))
        xc, yc = x[valid], y[valid]

        if len(xc) < 2:
            raise ValueError(f"Not enough valid points for interpolation: {len(xc)}")

        if self.method == "cubic" and len(xc) >= 4:
            spline = interpolate.CubicSpline(xc, yc, bc_type="natural", extrapolate=False)
            y_new = spline(x_new)
            y_new[x_new < xc.min()] = yc[0]
            y_new[x_new > xc.max()] = yc[-1]
        else:
            y_new = np.interp(x_new, xc, yc)

        return y_new