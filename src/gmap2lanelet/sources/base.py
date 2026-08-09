"""Source interfaces.

A *prior source* yields a road-centreline graph with OSM-style tags.  An
*imagery source* yields a georeferenced RGB raster.  Both are keyed by AOI so
that any prior can be combined with any imagery.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from ..geo import AOI, LocalFrame
from ..raster import GeoRaster


@dataclass
class PriorWay:
    """One road centreline with OSM-style tags, in WGS84."""

    id: str
    coords: np.ndarray                 # (N, 2) lon/lat degrees
    tags: dict[str, str] = field(default_factory=dict)


@dataclass
class PriorData:
    ways: list[PriorWay]
    attribution: str
    kind: str                          # "osm" | "osm-like"
    detail: dict = field(default_factory=dict)


@dataclass
class ImageryData:
    raster: GeoRaster                  # RGB uint8, local metric frame
    attribution: str
    gsd: float
    detail: dict = field(default_factory=dict)


class PriorSource(Protocol):
    name: str

    def fetch(self, aoi: AOI) -> PriorData: ...


class ImagerySource(Protocol):
    name: str

    def fetch(self, aoi: AOI, frame: LocalFrame) -> ImageryData: ...
