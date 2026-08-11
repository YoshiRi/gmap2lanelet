"""Road-aligned resampling: the bridge between the prior and the imagery.

Every fusion decision is made in a *road-aligned* frame: arc length ``s`` along
a prior centreline, lateral offset ``u`` across it.  Rectifying the evidence
into that frame turns "find the lane boundaries" from a 2-D curve-tracing
problem into a 1-D peak-finding problem, and it is the reason the topology
prior is worth having even when its geometry is wrong: it only has to be right
enough to define the direction of the cross-section.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..geo import normals, resample_polyline, tangents
from ..observation.base import Evidence


@dataclass
class Profile:
    """Evidence rectified onto (station, lateral offset)."""

    center: np.ndarray          # (S, 2) the polyline the profile is taken about
    tangent: np.ndarray         # (S, 2)
    normal: np.ndarray          # (S, 2) left-hand
    stations: np.ndarray        # (S,) arc length, metres
    curvature: np.ndarray       # (S,) signed curvature 1/m, + = turning left
    offsets: np.ndarray         # (U,) lateral offsets, metres, +left
    road: np.ndarray            # (S, U)
    mark: np.ndarray            # (S, U)
    veg: np.ndarray             # (S, U)
    shadow: np.ndarray          # (S, U)
    inside: np.ndarray          # (S, U) bool: sample fell inside the imagery

    @property
    def n_stations(self) -> int:
        return len(self.stations)

    def u_index(self, u: float) -> int:
        return int(np.clip(np.searchsorted(self.offsets, u), 0, len(self.offsets) - 1))

    def to_world(self, s_idx, u) -> np.ndarray:
        """Road-aligned (station index, offset) -> world xy."""
        s_idx = np.asarray(s_idx)
        u = np.asarray(u, dtype=float)
        return self.center[s_idx] + self.normal[s_idx] * u[..., None]


def build_profile(center: np.ndarray, evidence: Evidence, *, half_width: float,
                  ds: float = 1.0, du: float = 0.25) -> Profile:
    """Sample the evidence rasters along cross-sections of ``center``."""
    c = resample_polyline(np.asarray(center, dtype=float), ds)
    if len(c) < 2:
        c = np.asarray(center, dtype=float)
    t = tangents(c)
    n = normals(c)
    s = np.arange(len(c)) * ds
    u = np.arange(-half_width, half_width + 1e-9, du)

    # (S, U, 2) sample positions
    pts = c[:, None, :] + n[:, None, :] * u[None, :, None]
    x, y = pts[:, :, 0], pts[:, :, 1]

    return Profile(
        center=c, tangent=t, normal=n, stations=s, curvature=_curvature(t, ds), offsets=u,
        road=evidence.road_prob.sample(x, y, fill=0.0),
        mark=evidence.marking.sample(x, y, fill=0.0),
        veg=evidence.vegetation.sample(x, y, fill=0.0),
        shadow=evidence.shadow.sample(x, y, fill=0.0),
        inside=evidence.road_prob.contains(x, y),
    )


def _curvature(t: np.ndarray, ds: float) -> np.ndarray:
    """Signed curvature from the turn rate of the unit tangent."""
    th = np.unwrap(np.arctan2(t[:, 1], t[:, 0]))
    k = np.gradient(th) / max(ds, 1e-6)
    # a couple of metres of smoothing: raw per-station turn rate is very noisy
    return smooth_along(k, max(3, int(round(3.0 / max(ds, 1e-6))) | 1))


def smooth_along(a: np.ndarray, window: int, axis: int = 0) -> np.ndarray:
    """Moving average with edge replication."""
    if window < 2:
        return a
    k = np.ones(window) / window
    pad = window // 2
    a_sw = np.moveaxis(a, axis, -1)
    padded = np.pad(a_sw, [(0, 0)] * (a_sw.ndim - 1) + [(pad, pad)], mode="edge")
    out = np.apply_along_axis(lambda v: np.convolve(v, k, mode="valid"), -1, padded)
    out = out[..., :a_sw.shape[-1]]
    return np.moveaxis(out, -1, axis)


def median_along(a: np.ndarray, window: int) -> np.ndarray:
    """1-D rolling median (odd window), edge-replicated."""
    from scipy.ndimage import median_filter

    if window < 3:
        return a
    if window % 2 == 0:
        window += 1
    return median_filter(a, size=window, mode="nearest")
