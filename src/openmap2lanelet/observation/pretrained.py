"""Pretrained-segmentation observation backend.

Implements the same :class:`~openmap2lanelet.observation.base.ObservationBackend`
contract as the classical backend, so swapping it changes nothing downstream.

Read this before using it
-------------------------
The published aerial lane-marking segmenters (SkyScapes, the model behind
DeepAerialMapper) are trained on 5-13 cm orthophotos, where a lane marking is
2-3 px wide.  At the 25-35 cm available from public satellite imagery a marking
is *sub-pixel*.  Running such a model out of domain produces confident output
that nobody can check, which is the opposite of what this PoC is for -- hence
the classical backend is the default.

This backend is the right choice when you actually have high-resolution
orthophotos (a national aerial survey, a drone flight) and a model trained on
them.  Point ``model`` at anything callable that maps an ``(H, W, 3)`` uint8
array to per-class probabilities ``(H, W, C)``, and describe the class layout
with ``class_map``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np

from ..raster import GeoRaster
from .base import Evidence
from .classical import _vegetation

log = logging.getLogger(__name__)

# A DeepAerialMapper-style palette, for reference.
DEEPAERIALMAPPER_CLASSES = {
    "road": 1, "vegetation": 2, "traffic_island": 3, "sidewalk": 4,
    "parking": 5, "symbol": 6, "lane_marking": 7,
}


class PretrainedBackend:
    """Wrap a segmentation model as a geometry-evidence source."""

    def __init__(self, model: Callable[[np.ndarray], np.ndarray],
                 class_map: dict[str, int], *, name: str = "pretrained",
                 tile: int = 1024, overlap: int = 128,
                 drivable_classes: tuple[str, ...] = ("road", "parking", "traffic_island"),
                 marking_classes: tuple[str, ...] = ("lane_marking",)):
        self.model = model
        self.class_map = class_map
        self.name = name
        self.tile = tile
        self.overlap = overlap
        self.drivable_classes = drivable_classes
        self.marking_classes = marking_classes

    def run(self, imagery: GeoRaster, prior) -> Evidence:
        probs = self._infer_tiled(imagery.data)

        def combine(names) -> np.ndarray:
            idx = [self.class_map[n] for n in names if n in self.class_map]
            if not idx:
                return np.zeros(imagery.shape, dtype=np.float32)
            return np.clip(probs[:, :, idx].sum(axis=2), 0, 1).astype(np.float32)

        road = combine(self.drivable_classes)
        marking = combine(self.marking_classes)
        veg = (combine(("vegetation",)) if "vegetation" in self.class_map
               else _vegetation(imagery.data.astype(np.float32) / 255.0).astype(np.float32))

        lum = imagery.data.astype(np.float32).mean(axis=2) / 255.0
        on_road = road > 0.5
        ref = float(np.median(lum[on_road])) if on_road.sum() > 500 else float(np.median(lum))
        shadow = np.clip((0.35 * ref - lum) / max(0.35 * ref, 1e-3), 0, 1).astype(np.float32)

        log.info("observation(%s): road=%.1f%% of image, markings=%.3f%%",
                 self.name, 100 * float((road > 0.5).mean()),
                 100 * float((marking > 0.4).mean()))

        return Evidence(
            road_prob=imagery.like(road, "road_prob"),
            marking=imagery.like(marking, "marking"),
            vegetation=imagery.like(veg, "vegetation"),
            shadow=imagery.like(shadow, "shadow"),
            backend=self.name,
            detail={"gsd": round(imagery.gsd, 3), "classes": sorted(self.class_map),
                    "tile": self.tile,
                    "warning": "check the model's training GSD against the imagery GSD"},
        )

    def _infer_tiled(self, rgb: np.ndarray) -> np.ndarray:
        """Run the model over overlapping tiles and average the seams."""
        h, w = rgb.shape[:2]
        n_classes = max(self.class_map.values()) + 1
        acc = np.zeros((h, w, n_classes), dtype=np.float32)
        cnt = np.zeros((h, w, 1), dtype=np.float32)
        step = max(1, self.tile - self.overlap)

        for y in range(0, max(h - self.overlap, 1), step):
            for x in range(0, max(w - self.overlap, 1), step):
                y1, x1 = min(y + self.tile, h), min(x + self.tile, w)
                out = np.asarray(self.model(rgb[y:y1, x:x1]), dtype=np.float32)
                if out.ndim != 3 or out.shape[:2] != (y1 - y, x1 - x):
                    raise ValueError(
                        f"model must return (H, W, C) matching its input; got {out.shape}")
                acc[y:y1, x:x1, :out.shape[2]] += out
                cnt[y:y1, x:x1] += 1
        return acc / np.maximum(cnt, 1e-6)
