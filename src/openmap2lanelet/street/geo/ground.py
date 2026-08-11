"""Ground-height surface: the "public 3-D geometry" input.

Having a real ground surface rather than a flat-plane assumption matters twice
over: the inverse perspective mapping stops smearing on graded roads, and a
traffic light's *height above ground* becomes a usable plausibility test rather
than a guess about the local datum.

The interface is deliberately tiny so any raster DEM (AV2's per-log surface, a
national lidar DTM, a photogrammetric DSM) can be dropped in.
"""

from __future__ import annotations

import numpy as np


class GroundSurface:
    """Bilinearly-interpolated ground elevation over a regular grid."""

    def __init__(self, heights: np.ndarray, x0: float, y0: float, res: float,
                 name: str = "ground"):
        """``heights[r, c]`` is the elevation at ``(x0 + c*res, y0 + r*res)``."""
        self.h = np.asarray(heights, dtype=np.float32)
        self.x0 = float(x0)
        self.y0 = float(y0)
        self.res = float(res)
        self.name = name

    @property
    def coverage(self) -> float:
        return float(np.isfinite(self.h).mean())

    def elevation(self, x, y) -> np.ndarray:
        """Elevation at metric (x, y); NaN outside the mapped area."""
        c = (np.asarray(x, dtype=float) - self.x0) / self.res
        r = (np.asarray(y, dtype=float) - self.y0) / self.res
        H, W = self.h.shape
        ok = (c >= 0) & (c <= W - 1) & (r >= 0) & (r <= H - 1)
        c0 = np.clip(np.floor(c), 0, W - 1).astype(int)
        r0 = np.clip(np.floor(r), 0, H - 1).astype(int)
        c1 = np.minimum(c0 + 1, W - 1)
        r1 = np.minimum(r0 + 1, H - 1)
        fc = np.clip(c - c0, 0, 1)
        fr = np.clip(r - r0, 0, 1)
        v = (self.h[r0, c0] * (1 - fc) * (1 - fr) + self.h[r0, c1] * fc * (1 - fr)
             + self.h[r1, c0] * (1 - fc) * fr + self.h[r1, c1] * fc * fr)
        return np.where(ok, v, np.nan)

    def height_above(self, xyz: np.ndarray) -> np.ndarray:
        """Height of 3-D points above the ground beneath them."""
        p = np.atleast_2d(np.asarray(xyz, dtype=float))
        return p[:, 2] - self.elevation(p[:, 0], p[:, 1])

    # -- ray casting --------------------------------------------------------

    def intersect_ray(self, origin: np.ndarray, direction: np.ndarray,
                      t_max: float = 80.0, step: float = 0.5) -> np.ndarray | None:
        """First intersection of a ray with the surface, or ``None``.

        Marching then bisecting is more robust here than a closed-form plane
        solve, because the surface is a raster with holes and the ray may graze
        it at a very shallow angle.
        """
        o = np.asarray(origin, dtype=float)
        d = np.asarray(direction, dtype=float)
        d = d / max(np.linalg.norm(d), 1e-12)
        ts = np.arange(step, t_max, step)
        pts = o[None, :] + ts[:, None] * d[None, :]
        gap = pts[:, 2] - self.elevation(pts[:, 0], pts[:, 1])
        valid = np.isfinite(gap)
        if not valid.any():
            return None
        idx = np.flatnonzero(valid & (gap <= 0))
        if len(idx) == 0:
            return None
        i = int(idx[0])
        if i == 0:
            return pts[0]
        lo, hi = ts[i - 1], ts[i]
        for _ in range(24):
            mid = 0.5 * (lo + hi)
            p = o + mid * d
            g = float(self.elevation(p[0], p[1]))
            if not np.isfinite(g):
                break
            if p[2] - g > 0:
                lo = mid
            else:
                hi = mid
        return o + 0.5 * (lo + hi) * d
