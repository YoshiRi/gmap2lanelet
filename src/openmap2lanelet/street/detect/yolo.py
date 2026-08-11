"""COCO detector for the objects that carry road rules.

COCO already contains the two classes that matter most for a first pass --
``traffic light`` and ``stop sign`` -- so an off-the-shelf detector is a genuine
pretrained model for this task rather than a stand-in.  What COCO does *not*
contain is every other regulatory sign, and that gap is reported rather than
papered over (see ``docs/STREET_RESULTS.md``).

Weights come from the Ultralytics release assets and are cached on disk.
"""

from __future__ import annotations

import logging
from pathlib import Path

from ..types import Detection, LandmarkKind

log = logging.getLogger(__name__)

COCO_TRAFFIC_LIGHT = 9
COCO_STOP_SIGN = 11

KIND_OF = {
    COCO_TRAFFIC_LIGHT: LandmarkKind.TRAFFIC_LIGHT,
    COCO_STOP_SIGN: LandmarkKind.STOP_SIGN,
}

WEIGHTS_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/{name}"


class YoloDetector:
    """Ultralytics YOLO, restricted to the regulatory COCO classes."""

    def __init__(self, weights: str = "yolov8m.pt", imgsz: int = 1536,
                 conf: float = 0.25, min_box_px: float = 6.0,
                 cache_dir: str | None = None, device: str = "cpu"):
        self.weights_name = weights
        self.imgsz = imgsz
        self.conf = conf
        self.min_box_px = min_box_px
        self.device = device
        base = Path(cache_dir) if cache_dir else \
            Path(__file__).resolve().parents[4] / "data" / "cache" / "models"
        base.mkdir(parents=True, exist_ok=True)
        self.weights_path = base / weights
        self._model = None

    name = "yolov8-coco"

    def _ensure_weights(self) -> Path:
        if not self.weights_path.exists() or self.weights_path.stat().st_size == 0:
            from ...sources.cache import fetch_bytes

            url = WEIGHTS_URL.format(name=self.weights_name)
            log.info("downloading detector weights %s", url)
            self.weights_path.write_bytes(fetch_bytes(url, use_cache=False))
        return self.weights_path

    @property
    def model(self):
        if self._model is None:
            from ultralytics import YOLO

            self._model = YOLO(str(self._ensure_weights()))
        return self._model

    def detect(self, frames) -> list[Detection]:
        out: list[Detection] = []
        classes = list(KIND_OF)
        for i, f in enumerate(frames):
            res = self.model.predict(f.image_path, imgsz=self.imgsz, conf=self.conf,
                                     classes=classes, verbose=False,
                                     device=self.device)[0]
            for cls, score, xyxy in zip(res.boxes.cls.tolist(), res.boxes.conf.tolist(),
                                        res.boxes.xyxy.tolist()):
                w, h = xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]
                if min(w, h) < self.min_box_px:
                    continue
                out.append(Detection(frame_id=f.id, kind=KIND_OF[int(cls)],
                                     bbox=tuple(float(v) for v in xyxy),
                                     score=float(score), camera=f.camera.name))
            if (i + 1) % 25 == 0:
                log.info("detection %s/%s frames, %s boxes so far", i + 1, len(frames), len(out))
        n_tl = sum(1 for d in out if d.kind is LandmarkKind.TRAFFIC_LIGHT)
        log.info("detector(%s): %s boxes over %s frames (%s traffic lights, %s stop signs)",
                 self.name, len(out), len(frames), n_tl, len(out) - n_tl)
        return out
