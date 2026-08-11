"""Core data model.

Everything the pipeline produces carries *provenance* (which input decided it)
and a *confidence* in [0, 1].  That is a hard requirement of the PoC: the point
is not to be right everywhere, it is to know where we are probably wrong.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from .geo import AOI, LocalFrame


class Source(str, Enum):
    """Which input is responsible for a value."""

    OSM = "osm"                    # straight from map tags / map geometry
    IMAGE = "image"                # measured in the aerial image
    FUSED = "fused"                # image measurement, arbitrated by the prior
    DEFAULT = "default"            # neither: fell back to a modelling assumption
    INFERRED = "inferred"          # derived by a rule (e.g. intersection model)


class MarkingType(str, Enum):
    SOLID = "solid"
    DASHED = "dashed"
    DOUBLE_SOLID = "double_solid"
    ROAD_EDGE = "road_edge"        # corridor boundary, no painted line observed
    VIRTUAL = "virtual"            # invented (intersection interior, fallback split)
    UNKNOWN = "unknown"


@dataclass
class Provenance:
    source: Source
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"source": self.source.value, **{k: _jsonable(v) for k, v in self.detail.items()}}


@dataclass
class Boundary:
    """A lane boundary polyline in the local metric frame."""

    id: str
    points: np.ndarray                    # (N, 2) metres
    marking: MarkingType
    confidence: float
    provenance: Provenance
    # lateral offset from the corrected road centreline, if it came from a
    # corridor cross-section (used by the debug views)
    offset: float | None = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "marking": self.marking.value,
            "confidence": round(self.confidence, 3),
            "provenance": self.provenance.to_dict(),
            "offset": None if self.offset is None else round(self.offset, 2),
            "points": np.round(self.points, 3).tolist(),
        }


@dataclass
class Lane:
    """One drivable lane: a lanelet in the making."""

    id: str
    left_id: str
    right_id: str
    centerline: np.ndarray                # (N, 2) metres
    segment_id: str                       # prior edge (or junction) it belongs to
    kind: str = "road"                    # "road" | "turn"
    index_from_left: int = 0
    width: float = 3.5
    speed_limit_kph: float | None = None
    one_way: bool = True
    confidence: float = 0.5
    provenance: Provenance = field(default_factory=lambda: Provenance(Source.DEFAULT))
    predecessors: list[str] = field(default_factory=list)
    successors: list[str] = field(default_factory=list)
    turn_direction: str | None = None     # "left" | "right" | "straight" | "u_turn"
    attributes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "left": self.left_id,
            "right": self.right_id,
            "segment": self.segment_id,
            "kind": self.kind,
            "index_from_left": self.index_from_left,
            "width": round(self.width, 2),
            "speed_limit_kph": self.speed_limit_kph,
            "one_way": self.one_way,
            "confidence": round(self.confidence, 3),
            "provenance": self.provenance.to_dict(),
            "predecessors": self.predecessors,
            "successors": self.successors,
            "turn_direction": self.turn_direction,
            "attributes": {k: _jsonable(v) for k, v in self.attributes.items()},
            "centerline": np.round(self.centerline, 3).tolist(),
        }


@dataclass
class Intersection:
    """A junction area plus the connectivity that was inferred inside it."""

    id: str
    center: np.ndarray                    # (2,) metres
    radius: float
    prior_node_ids: list[str]
    approach_count: int
    turn_lane_ids: list[str] = field(default_factory=list)
    confidence: float = 0.4
    provenance: Provenance = field(default_factory=lambda: Provenance(Source.INFERRED))
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "center": np.round(self.center, 3).tolist(),
            "radius": round(self.radius, 2),
            "prior_nodes": self.prior_node_ids,
            "approaches": self.approach_count,
            "turn_lanes": self.turn_lane_ids,
            "confidence": round(self.confidence, 3),
            "provenance": self.provenance.to_dict(),
            "notes": self.notes,
        }


@dataclass
class LaneGraph:
    """The intermediate representation: what actually gets exported."""

    aoi: AOI
    frame: LocalFrame
    lanes: dict[str, Lane] = field(default_factory=dict)
    boundaries: dict[str, Boundary] = field(default_factory=dict)
    intersections: dict[str, Intersection] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def add_boundary(self, b: Boundary) -> str:
        self.boundaries[b.id] = b
        return b.id

    def add_lane(self, ln: Lane) -> str:
        self.lanes[ln.id] = ln
        return ln.id

    def connect(self, upstream: str, downstream: str) -> None:
        if downstream not in self.lanes[upstream].successors:
            self.lanes[upstream].successors.append(downstream)
        if upstream not in self.lanes[downstream].predecessors:
            self.lanes[downstream].predecessors.append(upstream)

    def stats(self) -> dict:
        road = [ln for ln in self.lanes.values() if ln.kind == "road"]
        turn = [ln for ln in self.lanes.values() if ln.kind == "turn"]
        import numpy as _np

        from .geo import polyline_length

        return {
            "lanes": len(self.lanes),
            "road_lanes": len(road),
            "turn_lanes": len(turn),
            "boundaries": len(self.boundaries),
            "intersections": len(self.intersections),
            "lane_km": round(
                sum(polyline_length(l.centerline) for l in self.lanes.values()) / 1000, 3),
            "mean_confidence": round(
                float(_np.mean([l.confidence for l in self.lanes.values()])), 3)
            if self.lanes else 0.0,
        }


@dataclass
class ReviewItem:
    """Something a human should look at."""

    id: str
    kind: str                              # failure-mode code, see qa.failures
    severity: str                          # "high" | "medium" | "low"
    message: str
    position: np.ndarray                   # (2,) metres, where to look
    element_ids: list[str] = field(default_factory=list)
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, frame: LocalFrame | None = None) -> dict:
        d = {
            "id": self.id,
            "kind": self.kind,
            "severity": self.severity,
            "message": self.message,
            "position": np.round(self.position, 2).tolist(),
            "elements": self.element_ids,
            "detail": {k: _jsonable(v) for k, v in self.detail.items()},
        }
        if frame is not None:
            lon, lat = frame.to_wgs84(self.position[0], self.position[1])
            d["lon"] = round(float(lon), 7)
            d["lat"] = round(float(lat), 7)
        return d


def _jsonable(v: Any) -> Any:
    if isinstance(v, np.ndarray):
        return np.round(v, 4).tolist()
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, Enum):
        return v.value
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    return v
