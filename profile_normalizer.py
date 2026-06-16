"""
Aligns both profiles onto common station intervals so they can be
compared point-by-point for cut/fill calculation.
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
            raise ValueError(f"Existing ground has too few points: {len(existing.stations)}")
        if len(proposed.stations) < 2:
            raise ValueError(f"Proposed grade has too few points: {len(proposed.stations)}")

        if sta_start is None:
            sta_start = max(existing.stations.min(), proposed.stations.min())
        if sta_end is None:
            sta_end = min(existing.stations.max(), proposed.stations.max())

        if sta_end <= sta_start:
            raise ValueError(
                f"No overlap — existing [{existing.station_range}], "
                f"proposed [{proposed.station_range}]"
            )

        stations = np.arange(sta_start, sta_end + self.interval / 2, self.interval)

        ex_interp = self._interp(existing.stations, existing.elevations, stations)
        pr_interp = self._interp(proposed.stations, proposed.elevations, stations)

        result = NormalizedProfiles(stations, ex_interp, pr_interp, self.interval,
                                   (float(sta_start), float(sta_end)))
        result.diagnostics = {
            "existing_points": len(existing.stations),
            "proposed_points": len(proposed.stations),
            "common_stations": len(stations),
            "method": self.method,
            "interval": self.interval,
        }
        return result

    def _interp(self, x, y, x_new):
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
