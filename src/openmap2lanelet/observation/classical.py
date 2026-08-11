"""Classical (weight-free) aerial observation backend.

Rationale
---------
The brief allows an existing pretrained segmentation model.  In practice a
model that segments *lane markings* from 0.3 m satellite imagery is not
something you can just download and trust on a new city -- the published ones
(SkyScapes, DeepAerialMapper's segmenter) are trained on 0.05-0.13 m aerial
orthophotos.  So the default backend here is classical and self-calibrating:

* **road surface** -- the prior tells us where road *probably* is, so we sample
  colour statistics along the prior centrelines and fit a robust model in
  illumination-normalised chromaticity space.  This is weak supervision from
  the map, not a decision by the map: the model is then evaluated everywhere,
  and it happily finds pavement OSM never mentioned (parking lots), which is
  precisely the conflict we want to detect.
* **markings** -- white top-hat at the painted-line scale, suppressed off the
  road surface and normalised by local pavement brightness, combined with a
  Sato ridge filter so that *linear* bright structures beat compact ones
  (vehicles, roof furniture).

``PretrainedBackend`` in ``pretrained.py`` implements the same interface for
when a suitable segmentation network is available.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy import ndimage as ndi

from ..raster import GeoRaster
from .base import Evidence

log = logging.getLogger(__name__)


class ClassicalBackend:
    name = "classical-v1"

    def __init__(self, *, sample_halfwidth: float = 2.0, marking_scale_m: float = 1.0,
                 min_road_component_m2: float = 200.0, lum_weight: float = 0.35,
                 local_norm_m: float = 30.0, local_norm_gain: float = 4.0,
                 min_marking_contrast: float = 0.008):
        # gain / min_marking_contrast were calibrated jointly against a
        # synthetic scene with known paint (must find every line), the same
        # scene with the paint removed (must find none), and three real 0.27 m
        # tiles (must keep finding lines on major roads).  See
        # docs/CALIBRATION.md.
        self.sample_halfwidth = sample_halfwidth
        self.marking_scale_m = marking_scale_m
        self.min_road_component_m2 = min_road_component_m2
        self.lum_weight = lum_weight
        self.local_norm_m = local_norm_m
        self.local_norm_gain = local_norm_gain
        self.min_marking_contrast = min_marking_contrast

    # -- main ---------------------------------------------------------------

    def run(self, imagery: GeoRaster, prior) -> Evidence:
        rgb = imagery.data.astype(np.float32) / 255.0
        gsd = imagery.gsd
        h, w = rgb.shape[:2]

        lum = rgb.mean(axis=2)
        chroma = _chromaticity(rgb)                       # (H, W, 2), illumination-normalised
        veg = _vegetation(rgb)
        nodata = lum < 1e-3

        # --- weak supervision: sample pavement colour along the prior --------
        sample_mask = self._sample_mask(imagery, prior, lum, veg, nodata)
        samples = _features_at(chroma, lum, sample_mask, limit=20000)
        if len(samples) < 200:
            log.warning("only %s pavement samples from the prior; falling back to "
                        "a global low-saturation model", len(samples))
            fallback = (veg < 0.5) & (~nodata) & (lum > 0.05) & (lum < 0.6)
            samples = _features_at(chroma, lum, fallback, limit=20000)

        road_prob = self._road_probability(chroma, lum, samples, veg, nodata, gsd)
        shadow = _shadow(lum, samples)
        marking = self._marking_response(lum, road_prob, gsd)

        detail = {
            "gsd": round(gsd, 3),
            "pavement_samples": int(len(samples)),
            "road_fraction": round(float((road_prob > 0.5).mean()), 4),
            "marking_fraction": round(float((marking > 0.4).mean()), 5),
            "sampled_pixels": int(sample_mask.sum()),
        }
        log.info("observation(%s): road=%.1f%% of image, markings=%.3f%%",
                 self.name, 100 * detail["road_fraction"], 100 * detail["marking_fraction"])

        return Evidence(
            road_prob=imagery.like(road_prob.astype(np.float32), "road_prob"),
            marking=imagery.like(marking.astype(np.float32), "marking"),
            vegetation=imagery.like(veg.astype(np.float32), "vegetation"),
            shadow=imagery.like(shadow.astype(np.float32), "shadow"),
            backend=self.name,
            detail=detail,
        )

    # -- steps --------------------------------------------------------------

    def _sample_mask(self, imagery, prior, lum, veg, nodata) -> np.ndarray:
        """Pixels in a thin buffer around the prior centrelines, tails trimmed."""
        h, w = lum.shape
        mask = np.zeros((h, w), dtype=bool)
        half_px = max(1, int(round(self.sample_halfwidth / imagery.gsd)))

        for edge in prior.edges.values():
            c, r = imagery.world_to_pixel(edge.points[:, 0], edge.points[:, 1])
            for (c0, r0), (c1, r1) in zip(np.column_stack([c, r]), np.column_stack([c, r])[1:]):
                n = max(2, int(max(abs(c1 - c0), abs(r1 - r0))) + 1)
                xs = np.round(np.linspace(c0, c1, n)).astype(int)
                ys = np.round(np.linspace(r0, r1, n)).astype(int)
                ok = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
                mask[ys[ok], xs[ok]] = True
        mask = ndi.binary_dilation(mask, _disk(half_px))
        mask &= (veg < 0.5) & ~nodata

        vals = lum[mask]
        if vals.size < 50:
            return mask
        # Drop the brightest (markings, vehicles, sun glint) and darkest (deep
        # shadow) tails so the pavement model is not dragged by them.
        lo, hi = np.percentile(vals, (10, 80))
        return mask & (lum >= lo) & (lum <= hi)

    def _road_probability(self, chroma, lum, feats, veg, nodata, gsd):
        """Distance to the sampled pavement model -> probability.

        Pavement is *not* unimodal: fresh asphalt, weathered asphalt and
        concrete differ by a factor of two in brightness while sharing a
        near-neutral chromaticity.  A single Gaussian fitted over all of them
        loses whichever mode is in the minority (typically the dark arterial,
        because most prior edges by count are pale parking aisles).  So we fit
        a small mixture and take the best-matching component, and we weight
        luminance well below chromaticity because chromaticity is the part that
        survives shading.
        """
        components = _fit_pavement_modes(feats, k=3, min_share=0.10)
        f = np.dstack([chroma, lum])

        best = None
        for mu, mad in components:
            d2 = (((f[:, :, :2] - mu[:2]) / mad[:2]) ** 2).sum(axis=2)
            d2 = d2 + self.lum_weight * ((f[:, :, 2] - mu[2]) / mad[2]) ** 2
            best = d2 if best is None else np.minimum(best, d2)

        # d2 ~ 9 (3 sigma) -> p ~ 0.5
        p = 1.0 / (1.0 + (best / 9.0) ** 1.5)

        p[veg > 0.5] *= 0.15
        p[nodata] = 0.0

        # Spatial regularisation + small-blob removal (a road is a big object).
        p = ndi.uniform_filter(p, size=max(3, int(round(1.5 / gsd))))
        binary = p > 0.5
        min_px = int(self.min_road_component_m2 / (gsd * gsd))
        lab, n = ndi.label(binary)
        if n:
            sizes = ndi.sum(binary, lab, index=np.arange(1, n + 1))
            small = np.isin(lab, np.nonzero(sizes < min_px)[0] + 1)
            p[small] *= 0.3
        return np.clip(p, 0, 1)

    def _marking_response(self, lum, road_prob, gsd):
        """Bright, thin, linear structures lying on the road surface."""
        from skimage.filters import sato
        from skimage.morphology import white_tophat

        r = max(2, int(round(self.marking_scale_m / gsd)))
        th = white_tophat(lum, _disk(r))

        # Line-ness: Sato favours tubular bright ridges over compact blobs.
        sig = max(0.7, 0.5 * (0.25 / max(gsd, 0.05)))
        ridge = sato(lum, sigmas=[sig, sig * 1.6, sig * 2.4], black_ridges=False)

        # Normalise *locally*.  A single global scale is set by whatever is
        # brightest in the scene -- typically fresh parking-bay stripes on pale
        # concrete -- which drives the response on a dark, weathered arterial
        # to nearly zero even where its lines are perfectly visible.  Dividing
        # by the local response level on nearby pavement makes the threshold
        # mean the same thing everywhere.
        on_road = road_prob > 0.5
        th_n = _local_normalise(th, on_road, gsd, self.local_norm_m,
                                self.local_norm_gain, self.min_marking_contrast)
        ridge_n = _local_normalise(ridge, on_road, gsd, self.local_norm_m,
                                   self.local_norm_gain, self.min_marking_contrast)

        m = np.sqrt(np.clip(th_n, 0, 1) * np.clip(ridge_n, 0, 1))
        # Markings only exist on pavement; keep a little tolerance so a marking
        # at the very edge of the road model is not erased.
        return np.clip(m * np.clip(road_prob * 1.25, 0, 1), 0, 1)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _local_normalise(resp: np.ndarray, on_road: np.ndarray, gsd: float,
                     window_m: float, gain: float, abs_floor: float = 0.0) -> np.ndarray:
    """Scale a filter response by its own local level over nearby pavement.

    ``resp / (gain * local_mean_on_road(resp))`` is ~1 where the response is
    ``gain`` times the local pavement texture, so a fixed threshold means
    "clearly brighter than this road's own noise" rather than "bright compared
    with the brightest thing in the tile".

    ``abs_floor`` stops the scale collapsing on featureless pavement.  Without
    it, a road with no markings at all normalises its own sensor noise up to
    marking level and the pipeline invents lane lines out of grain -- which is
    much worse than reporting that nothing was visible.
    """
    from scipy.ndimage import uniform_filter

    w = max(9, int(round(window_m / max(gsd, 0.05))) | 1)
    m = on_road.astype(np.float32)
    num = uniform_filter(resp * m, w, mode="nearest")
    den = uniform_filter(m, w, mode="nearest")

    glob = float(np.percentile(resp[on_road], 90.0)) if on_road.sum() > 500 \
        else float(np.percentile(resp, 99.0))
    glob = max(glob, 1e-6)
    local = np.where(den > 0.05, num / np.maximum(den, 1e-6), glob)
    # Never let the local scale collapse on a patch of unusually clean asphalt.
    local = np.maximum(local, max(0.15 * glob, abs_floor))
    return np.clip(resp / (gain * local + 1e-9), 0, 1)


def _disk(r: int) -> np.ndarray:
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y <= r * r)


def _chromaticity(rgb: np.ndarray) -> np.ndarray:
    """(r, g) normalised chromaticity: cancels most multiplicative shading."""
    s = rgb.sum(axis=2, keepdims=True)
    s = np.where(s < 1e-4, 1e-4, s)
    n = rgb / s
    return n[:, :, :2]


def _fit_pavement_modes(feats: np.ndarray, k: int = 3, min_share: float = 0.1,
                        iters: int = 25) -> list[tuple[np.ndarray, np.ndarray]]:
    """1-D k-means on luminance, then robust per-mode statistics in 3-D.

    Splitting on luminance only (rather than full 3-D k-means) keeps the modes
    interpretable -- they are "dark asphalt / mid asphalt / concrete" -- and
    stops the clustering from chasing chromatic noise.
    """
    floor = np.array([0.006, 0.006, 0.045])
    if len(feats) < 50:
        mu = np.median(feats, axis=0) if len(feats) else np.array([0.333, 0.333, 0.3])
        return [(mu, np.maximum(np.full(3, 0.05), floor))]

    lum = feats[:, 2]
    centers = np.percentile(lum, np.linspace(15, 85, k))
    labels = np.zeros(len(lum), dtype=int)
    for _ in range(iters):
        labels = np.argmin(np.abs(lum[:, None] - centers[None, :]), axis=1)
        new = np.array([lum[labels == i].mean() if np.any(labels == i) else centers[i]
                        for i in range(k)])
        if np.allclose(new, centers, atol=1e-4):
            break
        centers = new

    out: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(k):
        sel = feats[labels == i]
        if len(sel) < max(30, min_share * len(feats)):
            continue
        mu = np.median(sel, axis=0)
        mad = np.median(np.abs(sel - mu), axis=0) * 1.4826
        out.append((mu, np.maximum(mad, floor)))
    if not out:
        mu = np.median(feats, axis=0)
        mad = np.median(np.abs(feats - mu), axis=0) * 1.4826
        out = [(mu, np.maximum(mad, floor))]
    return out


def _features_at(chroma: np.ndarray, lum: np.ndarray, mask: np.ndarray,
                 limit: int = 20000) -> np.ndarray:
    """Stack (r, g, luminance) features for the masked pixels, subsampled."""
    idx = np.argwhere(mask)
    if len(idx) == 0:
        return np.empty((0, 3))
    if len(idx) > limit:
        idx = idx[np.random.default_rng(0).choice(len(idx), limit, replace=False)]
    return np.column_stack([chroma[idx[:, 0], idx[:, 1]], lum[idx[:, 0], idx[:, 1]]])


def _vegetation(rgb: np.ndarray) -> np.ndarray:
    """Green-excess index; no NIR band is available in RGB imagery."""
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    exg = 2 * g - r - b
    return np.clip((exg - 0.02) / 0.10, 0, 1)


def _shadow(lum: np.ndarray, feats) -> np.ndarray:
    """Pixels much darker than the sampled pavement."""
    has_chroma = feats.ndim == 2 and feats.shape[1] == 3
    ref = float(np.median(feats[:, 2])) if has_chroma else float(np.median(lum))
    thr = max(0.35 * ref, 0.02)
    return np.clip((thr - lum) / max(thr, 1e-3), 0, 1)
