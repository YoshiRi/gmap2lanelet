"""Street-level imagery source interface.

A source yields **posed frames**: an image plus a calibrated camera whose pose is
known in the local metric frame.  Pose is the whole point -- an image without one
can be classified but not mapped.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from ..geo.camera import Camera


@dataclass
class StreetFrame:
    id: str
    camera: Camera
    image_path: str
    timestamp_ns: int = 0
    detail: dict = field(default_factory=dict)

    def load(self) -> np.ndarray:
        from PIL import Image

        return np.asarray(Image.open(self.image_path).convert("RGB"))


@dataclass
class StreetSequence:
    """A posed image sequence plus whatever 3-D context the source provides."""

    frames: list[StreetFrame]
    attribution: str
    ground: object | None = None                  # GroundSurface, if available
    detail: dict = field(default_factory=dict)

    def bounds(self) -> tuple[float, float, float, float]:
        c = np.array([f.camera.center for f in self.frames])
        return (float(c[:, 0].min()), float(c[:, 1].min()),
                float(c[:, 0].max()), float(c[:, 1].max()))


class StreetImagerySource(Protocol):
    name: str

    def fetch(self, aoi, frame) -> StreetSequence: ...
