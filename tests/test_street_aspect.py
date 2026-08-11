"""Signal aspect classification: best-view selection, crop/classify, caching.

Fully offline and synthetic, following the convention set by
test_street_semantics.py: hand-built dataclasses and a tiny on-disk image, no
network and no real classifier model.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from gmap2lanelet.street.geo.camera import Camera, Intrinsics
from gmap2lanelet.street.semantics.aspect import (
    AspectResult,
    best_view,
    classify_aspects,
)
from gmap2lanelet.street.sources.base import StreetFrame
from gmap2lanelet.street.types import Detection, Landmark, LandmarkKind, SignalAspect


def _camera(name: str = "c0") -> Camera:
    k = Intrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0, width=640, height=480)
    return Camera(name=name, intrinsics=k, world_T_cam=np.eye(4))


def _frame(tmp_path, fid: str = "f0") -> StreetFrame:
    path = tmp_path / f"{fid}.png"
    Image.new("RGB", (640, 480), color=(10, 20, 30)).save(path)
    return StreetFrame(id=fid, camera=_camera(), image_path=str(path))


def _det(frame_id: str, bbox: tuple[float, float, float, float]) -> Detection:
    return Detection(frame_id=frame_id, kind=LandmarkKind.TRAFFIC_LIGHT, bbox=bbox,
                     score=0.9, camera="c0")


def _landmark(lid: str, detections: list[Detection], *,
             kind: LandmarkKind = LandmarkKind.TRAFFIC_LIGHT) -> Landmark:
    return Landmark(id=lid, kind=kind, position=np.array([0.0, 0.0, 5.0]),
                    detections=detections, n_views=len(detections))


class _StubClassifier:
    name = "stub"

    def __init__(self, results: list[AspectResult]):
        self._results = results
        self.calls = 0

    def classify(self, crops: list[np.ndarray]) -> list[AspectResult]:
        self.calls += 1
        assert len(crops) == len(self._results)
        return self._results


class _ExplodingClassifier:
    name = "exploding"

    def classify(self, crops: list[np.ndarray]) -> list[AspectResult]:
        raise AssertionError("classifier must not be called on a full cache hit")


def test_best_view_picks_largest_bbox():
    small = _det("f0", (0, 0, 10, 10))
    big = _det("f1", (0, 0, 40, 30))
    lm = _landmark("tl0", [small, big])
    assert best_view(lm) is big


def test_best_view_none_without_detections():
    assert best_view(_landmark("tl0", [])) is None


def test_classify_aspects_end_to_end(tmp_path):
    frame = _frame(tmp_path)
    lm = _landmark("tl0", [_det(frame.id, (100, 100, 160, 220))])
    clf = _StubClassifier([AspectResult("left_arrow", 0.9)])

    stats = classify_aspects([lm], [frame], clf,
                             label_map={"left_arrow": "arrow_left"})

    assert lm.aspect is SignalAspect.ARROW_LEFT
    assert lm.aspect_confidence == pytest.approx(0.9)
    assert lm.provenance.detail["aspect"] == {
        "raw_label": "left_arrow", "frame": frame.id, "camera": "c0"}
    assert "aspect_low_confidence" not in lm.flags
    assert stats == {"traffic_lights_classified": 1, "traffic_lights_low_confidence": 0}
    assert clf.calls == 1


def test_low_confidence_becomes_unknown(tmp_path):
    frame = _frame(tmp_path)
    lm = _landmark("tl0", [_det(frame.id, (100, 100, 160, 220))])
    clf = _StubClassifier([AspectResult("ball", 0.1)])

    classify_aspects([lm], [frame], clf, min_confidence=0.35)

    assert lm.aspect is SignalAspect.UNKNOWN
    assert "aspect_low_confidence" in lm.flags


def test_unmapped_label_becomes_unknown(tmp_path):
    frame = _frame(tmp_path)
    lm = _landmark("tl0", [_det(frame.id, (100, 100, 160, 220))])
    clf = _StubClassifier([AspectResult("some_unrecognized_class", 0.95)])

    classify_aspects([lm], [frame], clf)

    assert lm.aspect is SignalAspect.UNKNOWN
    assert "aspect_low_confidence" in lm.flags


def test_cache_round_trip_skips_the_classifier(tmp_path):
    frame = _frame(tmp_path)
    lm = _landmark("tl0", [_det(frame.id, (100, 100, 160, 220))])
    cache = tmp_path / "cache" / "aspects.json"

    classify_aspects([lm], [frame], _StubClassifier([AspectResult("ball", 0.8)]), cache=cache)
    assert cache.exists()
    assert json.loads(cache.read_text())[0]["landmark"] == "tl0"

    lm2 = _landmark("tl0", [_det(frame.id, (100, 100, 160, 220))])
    classify_aspects([lm2], [frame], _ExplodingClassifier(), cache=cache)
    assert lm2.aspect is SignalAspect.BALL
    assert lm2.aspect_confidence == pytest.approx(0.8)


def test_non_traffic_light_landmarks_are_untouched(tmp_path):
    frame = _frame(tmp_path)
    sign = _landmark("s0", [_det(frame.id, (100, 100, 160, 220))], kind=LandmarkKind.STOP_SIGN)

    stats = classify_aspects([sign], [frame], _ExplodingClassifier())

    assert sign.aspect is None
    assert stats == {"traffic_lights_classified": 0, "traffic_lights_low_confidence": 0}
