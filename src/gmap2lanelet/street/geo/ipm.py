"""Inverse perspective mapping: street imagery -> a bird's-eye road raster.

This is what lets the street layer feed the *existing* geometry stage instead of
running beside it.  The output is a plain :class:`GeoRaster` in the local metric
frame, so ``ClassicalBackend`` and the whole corridor/lane-structure pipeline
consume it unchanged -- only now at 5-8 cm per pixel instead of the 27 cm that
public satellite imagery affords.

Two details matter more than the projection itself:

* **The ground is not a plane.**  Rays are intersected with the ground-height
  surface, so a crest or a crowned carriageway does not smear the markings.
* **The scene moves.**  A weighted mean over a hundred frames would paint every
  passing vehicle into the road, so the accumulation is two-pass: the first pass
  estimates a mean and variance per cell, the second re-accumulates only the
  samples that agree with it.  Transient objects fall out; the road stays.
"""

from __future__ import annotations

import logging

import numpy as np

from ...geo import LocalFrame
from ...raster import GeoRaster
from .ground import GroundSurface

log = logging.getLogger(__name__)


class BevMosaic:
    def __init__(self, *, resolution: float = 0.06, near_m: float = 4.0,
                 far_m: float = 26.0, half_width_m: float = 14.0,
                 margin_m: float = 8.0, robust_sigma: float = 1.6,
                 max_gsd_m: float = 0.22):
        self.res = resolution
        self.near = near_m
        self.far = far_m
        self.half_width = half_width_m
        self.margin = margin_m
        self.robust_sigma = robust_sigma
        # A ground cell is only worth sampling while the camera still resolves
        # it: the along-range footprint of one pixel grows as range^2, so beyond
        # a few tens of metres a single pixel smears over many BEV cells and
        # paints radial streaks instead of road.
        self.max_gsd = max_gsd_m

    # -- canvas -------------------------------------------------------------

    def _canvas(self, frames):
        c = np.array([f.camera.center for f in frames])
        pad = self.far + self.margin
        x0, x1 = c[:, 0].min() - pad, c[:, 0].max() + pad
        y0, y1 = c[:, 1].min() - pad, c[:, 1].max() + pad
        w = int(np.ceil((x1 - x0) / self.res))
        h = int(np.ceil((y1 - y0) / self.res))
        return x0, y1, w, h

    def build(self, frames, ground: GroundSurface | None, local_frame: LocalFrame,
              roi_polylines: list | None = None, roi_halfwidth_m: float = 16.0,
              progress_every: int = 40) -> tuple[GeoRaster, GeoRaster]:
        """Return ``(rgb, coverage)`` rasters over the trajectory.

        ``roi_polylines`` restricts accumulation to a buffer around the prior
        road centrelines.  Inverse perspective mapping is only valid *on the
        ground*: anything with height -- buildings, kerbs, snow banks, parked
        vehicles -- is smeared radially away from the camera.  Confining the
        mosaic to the road corridor removes almost all of it, and the prior is
        exactly the right thing to define that corridor, since it is already
        trusted for topology and not for geometry.
        """
        x0, ytop, w, h = self._canvas(frames)
        log.info("bev canvas %sx%s px @ %.2f m (%.0f x %.0f m)",
                 w, h, self.res, w * self.res, h * self.res)

        roi = self._roi_mask(roi_polylines, roi_halfwidth_m, x0, ytop, w, h)

        acc = np.zeros((h, w, 3), dtype=np.float32)
        wsum = np.zeros((h, w), dtype=np.float32)
        sq = np.zeros((h, w, 3), dtype=np.float32)

        # pass 1: weighted mean and variance
        for i, f in enumerate(frames):
            self._accumulate(f, ground, x0, ytop, w, h, acc, wsum, sq, None, None, roi)
            if progress_every and (i + 1) % progress_every == 0:
                log.info("bev pass 1: %s/%s frames", i + 1, len(frames))

        ok = wsum > 1e-6
        mean = np.zeros_like(acc)
        mean[ok] = acc[ok] / wsum[ok][..., None]
        var = np.zeros_like(acc)
        var[ok] = np.maximum(sq[ok] / wsum[ok][..., None] - mean[ok] ** 2, 0.0)
        std = np.sqrt(var).mean(axis=2)

        # pass 2: re-accumulate agreeing samples only
        acc2 = np.zeros_like(acc)
        wsum2 = np.zeros_like(wsum)
        for i, f in enumerate(frames):
            self._accumulate(f, ground, x0, ytop, w, h, acc2, wsum2, None, mean, std, roi)
            if progress_every and (i + 1) % progress_every == 0:
                log.info("bev pass 2: %s/%s frames", i + 1, len(frames))

        ok2 = wsum2 > 1e-6
        out = np.zeros((h, w, 3), dtype=np.uint8)
        out[ok2] = np.clip(acc2[ok2] / wsum2[ok2][..., None], 0, 255).astype(np.uint8)
        # cells that lost every sample to the outlier test keep the pass-1 mean
        fallback = ok & ~ok2
        out[fallback] = np.clip(mean[fallback], 0, 255).astype(np.uint8)

        rgb = GeoRaster(out, x0, ytop, self.res, self.res, "bev")
        cov = GeoRaster(np.minimum(wsum2, 20.0).astype(np.float32) / 20.0,
                        x0, ytop, self.res, self.res, "bev_coverage")
        log.info("bev mosaic: %.1f%% of canvas covered", 100 * float(ok2.mean()))
        return rgb, cov

    # -- one frame ----------------------------------------------------------

    def _roi_mask(self, polylines, halfwidth, x0, ytop, w, h) -> np.ndarray | None:
        if not polylines:
            return None
        import cv2

        mask = np.zeros((h, w), dtype=np.uint8)
        thick = max(1, int(round(2 * halfwidth / self.res)))
        for line in polylines:
            p = np.asarray(line, dtype=float)[:, :2]
            cols = (p[:, 0] - x0) / self.res
            rows = (ytop - p[:, 1]) / self.res
            pts = np.column_stack([cols, rows]).astype(np.int32)
            if len(pts) >= 2:
                cv2.polylines(mask, [pts], False, 1, thickness=thick)
        log.info("bev roi: %.1f%% of canvas within %.0f m of a prior road",
                 100 * float(mask.mean()), halfwidth)
        return mask.astype(bool)

    def _accumulate(self, f, ground, x0, ytop, w, h, acc, wsum, sq, mean, std, roi=None):
        cam = f.camera
        c = cam.center
        fwd = cam.forward
        yaw = np.arctan2(fwd[1], fwd[0])

        # candidate cells: a rectangle in front of the camera, in canvas indices
        ca, sa = np.cos(yaw), np.sin(yaw)
        corners = []
        for dl in (self.near, self.far):
            for dt in (-self.half_width, self.half_width):
                corners.append([c[0] + dl * ca - dt * sa, c[1] + dl * sa + dt * ca])
        corners = np.array(corners)
        cx0 = int(np.floor((corners[:, 0].min() - x0) / self.res))
        cx1 = int(np.ceil((corners[:, 0].max() - x0) / self.res))
        ry0 = int(np.floor((ytop - corners[:, 1].max()) / self.res))
        ry1 = int(np.ceil((ytop - corners[:, 1].min()) / self.res))
        cx0, cx1 = max(cx0, 0), min(cx1, w)
        ry0, ry1 = max(ry0, 0), min(ry1, h)
        if cx1 <= cx0 or ry1 <= ry0:
            return

        gx = x0 + (np.arange(cx0, cx1) + 0.5) * self.res
        gy = ytop - (np.arange(ry0, ry1) + 0.5) * self.res
        X, Y = np.meshgrid(gx, gy)

        # keep only cells actually inside the viewing rectangle
        dx, dy = X - c[0], Y - c[1]
        along = dx * ca + dy * sa
        across = -dx * sa + dy * ca
        sel = (along >= self.near) & (along <= self.far) & (np.abs(across) <= self.half_width)
        if roi is not None:
            sel &= roi[ry0:ry1, cx0:cx1]
        if not sel.any():
            return

        Z = ground.elevation(X, Y) if ground is not None else np.full(X.shape, c[2] - 1.6)
        sel &= np.isfinite(Z)
        if not sel.any():
            return

        pts = np.column_stack([X[sel], Y[sel], Z[sel]])
        uv, ok = cam.project(pts)
        if not ok.any():
            return

        img = f.load()
        ih, iw = img.shape[:2]
        u = np.clip(np.round(uv[ok, 0]).astype(int), 0, iw - 1)
        v = np.clip(np.round(uv[ok, 1]).astype(int), 0, ih - 1)
        colour = img[v, u].astype(np.float32)

        rng = along[sel][ok]
        # along-range footprint of one pixel on the ground
        cam_h = max(float(c[2] - np.nanmean(Z[sel])), 0.8)
        gsd = (rng ** 2) / max(cam.intrinsics.fy * cam_h, 1e-6)
        sharp = gsd <= self.max_gsd
        if not sharp.any():
            return
        weight = (1.0 / np.maximum(gsd, self.res) ** 2).astype(np.float32)

        rows, cols = np.nonzero(sel)
        rows = rows[ok] + ry0
        cols = cols[ok] + cx0
        rows, cols = rows[sharp], cols[sharp]
        colour, weight = colour[sharp], weight[sharp]

        if mean is not None:
            ref = mean[rows, cols]
            tol = self.robust_sigma * np.maximum(std[rows, cols], 6.0)[:, None]
            agree = (np.abs(colour - ref) <= tol).all(axis=1)
            rows, cols, colour, weight = rows[agree], cols[agree], colour[agree], weight[agree]
            if len(rows) == 0:
                return

        np.add.at(acc, (rows, cols), colour * weight[:, None])
        np.add.at(wsum, (rows, cols), weight)
        if sq is not None:
            np.add.at(sq, (rows, cols), (colour ** 2) * weight[:, None])
