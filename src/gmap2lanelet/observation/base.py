"""Imagery observation interface.

An observation backend converts an aerial raster into *geometry evidence*:
where the drivable surface is, where painted markings are, and what is in the
way.  It deliberately knows nothing about lanes, connectivity or road classes
-- that is the prior's job.  Swapping a classical backend for a learned one
must not change anything downstream.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from ..raster import GeoRaster


@dataclass
class Evidence:
    """Per-pixel geometry evidence, all rasters share the imagery grid."""

    road_prob: GeoRaster        # float32 [0, 1]: probability of drivable paved surface
    marking: GeoRaster          # float32 [0, 1]: painted-marking response
    vegetation: GeoRaster       # float32 [0, 1]
    shadow: GeoRaster           # float32 [0, 1]
    backend: str = "unknown"
    detail: dict = field(default_factory=dict)

    @property
    def gsd(self) -> float:
        return self.road_prob.gsd


class ObservationBackend(Protocol):
    name: str

    def run(self, imagery: GeoRaster, prior) -> Evidence:
        """``prior`` is a :class:`~gmap2lanelet.prior.road_graph.RoadPrior`.

        Backends may use it for weak supervision (sampling road colour along
        centrelines) but must not use it to *decide* where roads are.
        """
        ...
