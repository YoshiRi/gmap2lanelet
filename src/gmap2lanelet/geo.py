"""Coordinate frames.

The whole pipeline works in a **local metric frame**: a tangent plane centred on
the AOI, x pointing east, y pointing north, metres.  For AOIs of a few hundred
metres an equirectangular projection about the AOI centre is accurate to well
below a centimetre, and -- more usefully -- it keeps the mapping from a
north-up lat/lon raster to the metric frame *exactly affine*, so no resampling
is ever required.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# WGS84 mean radius; the equirectangular approximation does not warrant the
# full ellipsoid here (see module docstring).
_EARTH_R = 6_378_137.0


@dataclass(frozen=True)
class AOI:
    """Area of interest, WGS84 degrees."""

    name: str
    west: float
    south: float
    east: float
    north: float

    @property
    def center(self) -> tuple[float, float]:
        return (0.5 * (self.west + self.east), 0.5 * (self.south + self.north))

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return (self.west, self.south, self.east, self.north)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "west": self.west,
            "south": self.south,
            "east": self.east,
            "north": self.north,
        }

    @staticmethod
    def from_dict(d: dict) -> "AOI":
        return AOI(d["name"], d["west"], d["south"], d["east"], d["north"])


class LocalFrame:
    """Equirectangular tangent plane about ``(lat0, lon0)``."""

    def __init__(self, lat0: float, lon0: float):
        self.lat0 = float(lat0)
        self.lon0 = float(lon0)
        self._mx = _EARTH_R * math.cos(math.radians(self.lat0)) * math.pi / 180.0
        self._my = _EARTH_R * math.pi / 180.0

    @staticmethod
    def for_aoi(aoi: AOI) -> "LocalFrame":
        lon0, lat0 = aoi.center
        return LocalFrame(lat0, lon0)

    def to_local(self, lon, lat):
        """lon/lat (deg, array-like) -> x/y (m) in the local frame."""
        lon = np.asarray(lon, dtype=float)
        lat = np.asarray(lat, dtype=float)
        return (lon - self.lon0) * self._mx, (lat - self.lat0) * self._my

    def to_wgs84(self, x, y):
        """x/y (m) -> lon/lat (deg)."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        return x / self._mx + self.lon0, y / self._my + self.lat0

    def to_dict(self) -> dict:
        return {"lat0": self.lat0, "lon0": self.lon0}


# --------------------------------------------------------------------------- #
# polyline helpers (local metric frame, arrays of shape (N, 2))
# --------------------------------------------------------------------------- #


def polyline_length(pts: np.ndarray) -> float:
    if len(pts) < 2:
        return 0.0
    return float(np.hypot(*np.diff(pts, axis=0).T).sum())


def cumulative_length(pts: np.ndarray) -> np.ndarray:
    if len(pts) < 2:
        return np.zeros(len(pts))
    seg = np.hypot(*np.diff(pts, axis=0).T)
    return np.concatenate([[0.0], np.cumsum(seg)])


def resample_polyline(pts: np.ndarray, step: float) -> np.ndarray:
    """Resample to (approximately) constant arc-length spacing."""
    pts = np.asarray(pts, dtype=float)
    if len(pts) < 2:
        return pts.copy()
    s = cumulative_length(pts)
    total = s[-1]
    if total <= 1e-9:
        return pts[:1].copy()
    n = max(2, int(round(total / step)) + 1)
    st = np.linspace(0.0, total, n)
    return np.column_stack([np.interp(st, s, pts[:, 0]), np.interp(st, s, pts[:, 1])])


def tangents(pts: np.ndarray) -> np.ndarray:
    """Unit tangents, central differences, shape (N, 2)."""
    pts = np.asarray(pts, dtype=float)
    if len(pts) < 2:
        return np.tile(np.array([1.0, 0.0]), (len(pts), 1))
    d = np.gradient(pts, axis=0)
    n = np.hypot(d[:, 0], d[:, 1])
    n[n < 1e-12] = 1.0
    return d / n[:, None]


def normals(pts: np.ndarray) -> np.ndarray:
    """Left-hand normals (rotate tangent +90 deg)."""
    t = tangents(pts)
    return np.column_stack([-t[:, 1], t[:, 0]])


def offset_polyline(pts: np.ndarray, offsets) -> np.ndarray:
    """Offset ``pts`` laterally by ``offsets`` (scalar or per-vertex, +left)."""
    n = normals(pts)
    off = np.asarray(offsets, dtype=float)
    if off.ndim == 0:
        off = np.full(len(pts), float(off))
    return pts + n * off[:, None]


def smooth_polyline(pts: np.ndarray, window: int = 9) -> np.ndarray:
    """Moving-average smoothing with endpoints held fixed."""
    pts = np.asarray(pts, dtype=float)
    if len(pts) < 3 or window < 3:
        return pts.copy()
    window = min(window, len(pts) if len(pts) % 2 else len(pts) - 1)
    if window < 3:
        return pts.copy()
    k = np.ones(window) / window
    pad = window // 2
    out = np.empty_like(pts)
    for i in range(2):
        padded = np.concatenate([np.full(pad, pts[0, i]), pts[:, i], np.full(pad, pts[-1, i])])
        out[:, i] = np.convolve(padded, k, mode="valid")
    out[0] = pts[0]
    out[-1] = pts[-1]
    return out


def heading(pts: np.ndarray, at_start: bool) -> float:
    """Heading (rad) at the first or last vertex."""
    pts = np.asarray(pts, dtype=float)
    if len(pts) < 2:
        return 0.0
    d = pts[1] - pts[0] if at_start else pts[-1] - pts[-2]
    return math.atan2(d[1], d[0])


def angle_diff(a: float, b: float) -> float:
    """Signed smallest difference a - b, wrapped to (-pi, pi]."""
    return (a - b + math.pi) % (2 * math.pi) - math.pi


def hermite(p0: np.ndarray, h0: float, p1: np.ndarray, h1: float, n: int = 24,
            tension: float = 0.55) -> np.ndarray:
    """Cubic Hermite between poses (position + heading). G1-continuous."""
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    d = float(np.hypot(*(p1 - p0)))
    m = max(d * tension, 1e-3)
    t0 = np.array([math.cos(h0), math.sin(h0)]) * m
    t1 = np.array([math.cos(h1), math.sin(h1)]) * m
    t = np.linspace(0.0, 1.0, n)[:, None]
    h00 = 2 * t**3 - 3 * t**2 + 1
    h10 = t**3 - 2 * t**2 + t
    h01 = -2 * t**3 + 3 * t**2
    h11 = t**3 - t**2
    return h00 * p0 + h10 * t0 + h01 * p1 + h11 * t1


def simplify_polyline(pts: np.ndarray, tolerance: float) -> np.ndarray:
    """Douglas-Peucker, keeping the first and last vertex."""
    pts = np.asarray(pts, dtype=float)
    if tolerance <= 0 or len(pts) < 3:
        return pts
    from shapely.geometry import LineString

    out = np.asarray(LineString(pts).simplify(tolerance, preserve_topology=False).coords)
    return out if len(out) >= 2 else pts
