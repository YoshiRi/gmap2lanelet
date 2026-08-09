"""Georeferenced rasters in the local metric frame."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geo import AOI, LocalFrame


@dataclass
class GeoRaster:
    """A north-up raster with an affine pixel -> local-metric mapping.

    Source imagery is a north-up lat/lon (or Web-Mercator) grid, and the local
    frame is linear in lat/lon, so ``pixel -> metres`` is affine and no
    resampling is needed anywhere in the pipeline.
    """

    data: np.ndarray            # (H, W) or (H, W, C)
    x0: float                   # metric x of pixel-column 0 (left edge of pixel 0)
    y0: float                   # metric y of pixel-row 0 (top edge of pixel 0)
    dx: float                   # metres per column (> 0)
    dy: float                   # metres per row (> 0, rows run *southward*)
    name: str = "raster"

    # -- construction -------------------------------------------------------

    @staticmethod
    def from_bounds(data: np.ndarray, frame: LocalFrame, west: float, south: float,
                    east: float, north: float, name: str = "raster") -> "GeoRaster":
        h, w = data.shape[:2]
        (x_w, x_e), (y_s, y_n) = frame.to_local([west, east], [south, north])
        return GeoRaster(data, float(x_w), float(y_n), (float(x_e) - float(x_w)) / w,
                         (float(y_n) - float(y_s)) / h, name)

    def like(self, data: np.ndarray, name: str) -> "GeoRaster":
        """A raster with the same georeferencing but different pixel content."""
        return GeoRaster(data, self.x0, self.y0, self.dx, self.dy, name)

    # -- geometry -----------------------------------------------------------

    @property
    def shape(self) -> tuple[int, int]:
        return self.data.shape[0], self.data.shape[1]

    @property
    def gsd(self) -> float:
        """Nominal ground sample distance (m/px)."""
        return float(np.sqrt(self.dx * self.dy))

    @property
    def extent(self) -> tuple[float, float, float, float]:
        """(xmin, xmax, ymin, ymax) for matplotlib ``imshow``."""
        h, w = self.shape
        return (self.x0, self.x0 + w * self.dx, self.y0 - h * self.dy, self.y0)

    def world_to_pixel(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        """Metric -> fractional (col, row)."""
        return (np.asarray(x, float) - self.x0) / self.dx, (self.y0 - np.asarray(y, float)) / self.dy

    def pixel_to_world(self, col, row) -> tuple[np.ndarray, np.ndarray]:
        return self.x0 + np.asarray(col, float) * self.dx, self.y0 - np.asarray(row, float) * self.dy

    def contains(self, x, y) -> np.ndarray:
        c, r = self.world_to_pixel(x, y)
        h, w = self.shape
        return (c >= 0) & (c <= w - 1) & (r >= 0) & (r <= h - 1)

    # -- sampling -----------------------------------------------------------

    def sample(self, x, y, fill: float = 0.0) -> np.ndarray:
        """Bilinear sample at metric coordinates.  Out-of-bounds -> ``fill``."""
        c, r = self.world_to_pixel(x, y)
        return _bilinear(self.data, c, r, fill)

    def aoi_mask(self, aoi: AOI, frame: LocalFrame) -> np.ndarray:
        h, w = self.shape
        cols = np.arange(w)
        rows = np.arange(h)
        xs, _ = self.pixel_to_world(cols, np.zeros_like(cols))
        _, ys = self.pixel_to_world(np.zeros_like(rows), rows)
        (xw, xe), (ys_, yn) = frame.to_local([aoi.west, aoi.east], [aoi.south, aoi.north])
        mx = (xs >= xw) & (xs <= xe)
        my = (ys >= ys_) & (ys <= yn)
        return my[:, None] & mx[None, :]


def _bilinear(img: np.ndarray, c: np.ndarray, r: np.ndarray, fill: float) -> np.ndarray:
    c = np.asarray(c, float)
    r = np.asarray(r, float)
    h, w = img.shape[:2]
    chan = img.shape[2] if img.ndim == 3 else 0

    ok = (c >= 0) & (c <= w - 1) & (r >= 0) & (r <= h - 1)
    c_ = np.clip(c, 0, w - 1)
    r_ = np.clip(r, 0, h - 1)
    c0 = np.floor(c_).astype(int)
    r0 = np.floor(r_).astype(int)
    c1 = np.minimum(c0 + 1, w - 1)
    r1 = np.minimum(r0 + 1, h - 1)
    fc = (c_ - c0)[..., None] if chan else (c_ - c0)
    fr = (r_ - r0)[..., None] if chan else (r_ - r0)

    a = img[r0, c0].astype(np.float32)
    b = img[r0, c1].astype(np.float32)
    d = img[r1, c0].astype(np.float32)
    e = img[r1, c1].astype(np.float32)
    out = (a * (1 - fc) * (1 - fr) + b * fc * (1 - fr) + d * (1 - fc) * fr + e * fc * fr)

    m = ok[..., None] if chan else ok
    return np.where(m, out, fill)
