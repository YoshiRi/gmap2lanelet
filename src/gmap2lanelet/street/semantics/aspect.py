"""Signal-head lens layout -> ``SignalAspect``, from a user-supplied classifier.

This module never trains or ships a model. It crops the best available view of
each already-triangulated traffic-light ``Landmark`` and hands it to an
externally trained classifier the caller provides, then writes the normalized
result back onto the landmark. Association, lane membership and export
otherwise proceed exactly as before -- aspect is additive evidence, not a
filter (see ``associate.py`` and ``export/lanelet2_osm.py``).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from ..types import Detection, Landmark, LandmarkKind, SignalAspect

log = logging.getLogger(__name__)

_PAD_FRAC = 0.15


@dataclass
class AspectResult:
    """One classifier output: its own label plus a confidence in [0, 1]."""

    label: str
    confidence: float


class AspectClassifier(Protocol):
    name: str

    def classify(self, crops: list[np.ndarray]) -> list[AspectResult]:
        """Classify a batch of RGB (H, W, 3) uint8 crops, one result each."""
        ...


def best_view(lm: Landmark) -> Detection | None:
    """The most promising detection to crop for classification.

    Ranked by pixel bbox area: a larger box is a closer, more resolved view,
    and it is available today with no change to triangulation. A future
    refinement could weigh in per-detection reprojection error once
    ``LandmarkBuilder._fit()`` retains it (currently only the mean survives
    into ``Landmark.residual_px``).
    """
    if not lm.detections:
        return None
    return max(lm.detections, key=lambda d: d.size[0] * d.size[1])


def _crop(frame, bbox: tuple[float, float, float, float]) -> np.ndarray:
    img = frame.load()
    h, w = img.shape[:2]
    x0, y0, x1, y1 = bbox
    bw, bh = x1 - x0, y1 - y0
    x0 -= bw * _PAD_FRAC
    x1 += bw * _PAD_FRAC
    y0 -= bh * _PAD_FRAC
    y1 += bh * _PAD_FRAC
    c0, r0 = max(0, int(x0)), max(0, int(y0))
    c1, r1 = min(w, int(np.ceil(x1))), min(h, int(np.ceil(y1)))
    return img[r0:r1, c0:c1]


def _normalize(result: AspectResult, label_map: dict[str, str],
               min_confidence: float) -> tuple[SignalAspect, bool]:
    mapped = label_map.get(result.label, result.label)
    try:
        aspect = SignalAspect(mapped)
    except ValueError:
        return SignalAspect.UNKNOWN, True
    if result.confidence < min_confidence:
        return SignalAspect.UNKNOWN, True
    return aspect, False


def classify_aspects(landmarks: list[Landmark], frames, classifier: AspectClassifier, *,
                      label_map: dict[str, str] | None = None,
                      min_confidence: float = 0.35,
                      cache: str | Path | None = None) -> dict:
    """Crop the best view of every traffic-light landmark and tag its aspect.

    Mutates ``TRAFFIC_LIGHT`` landmarks in place (``aspect``,
    ``aspect_confidence``, ``provenance.detail["aspect"]``); other kinds are
    left untouched. Returns a stats dict meant for ``SemanticLayer.detail``,
    the same convention ``apply_arrows()`` uses in ``associate.py``.
    """
    label_map = label_map or {}
    by_frame = {f.id: f for f in frames}
    tls = [lm for lm in landmarks if lm.kind is LandmarkKind.TRAFFIC_LIGHT]

    # The cache holds the classifier's *raw* output (the expensive part),
    # mirroring how experiment.py's detections.json caches raw Detections
    # rather than the derived Landmarks -- thresholding/label_map are cheap
    # and re-applied on every load so tuning them doesn't require a rerun.
    cached: dict[str, dict] = {}
    cache_path = Path(cache) if cache else None
    if cache_path and cache_path.exists():
        cached = {row["landmark"]: row for row in json.loads(cache_path.read_text())}

    views: dict[str, Detection] = {}
    to_classify: list[Landmark] = []
    for lm in tls:
        det = best_view(lm)
        if det is None or det.frame_id not in by_frame:
            continue
        views[lm.id] = det
        if lm.id not in cached:
            to_classify.append(lm)

    if to_classify:
        crops = [_crop(by_frame[views[lm.id].frame_id], views[lm.id].bbox) for lm in to_classify]
        for lm, result in zip(to_classify, classifier.classify(crops), strict=True):
            det = views[lm.id]
            cached[lm.id] = {"landmark": lm.id, "raw_label": result.label,
                             "confidence": round(result.confidence, 3),
                             "frame": det.frame_id, "camera": det.camera}

    low_confidence = 0
    for lm in tls:
        row = cached.get(lm.id)
        if row is None:
            continue
        aspect, low = _normalize(AspectResult(row["raw_label"], row["confidence"]),
                                 label_map, min_confidence)
        lm.aspect = aspect
        lm.aspect_confidence = row["confidence"]
        lm.provenance.detail["aspect"] = {
            "raw_label": row["raw_label"], "frame": row["frame"], "camera": row["camera"],
        }
        if low:
            low_confidence += 1
            lm.flags.append("aspect_low_confidence")

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(list(cached.values())))

    classified = sum(1 for lm in tls if lm.aspect is not None)
    log.info("aspect: %s/%s traffic lights classified (%s low-confidence)",
             classified, len(tls), low_confidence)
    return {"traffic_lights_classified": classified,
            "traffic_lights_low_confidence": low_confidence}
