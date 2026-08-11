"""Data model for the street-level semantic layer.

The street layer never produces geometry that competes with the lane graph.  It
produces *evidence about meaning*: where a signal head is, which stop line it
governs, which way an arrow points.  Everything it emits therefore carries the
same ``Provenance`` / ``confidence`` machinery as the rest of the pipeline, and
anything it cannot decide is left undecided and flagged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from ..types import Provenance, Source


class LandmarkKind(str, Enum):
    TRAFFIC_LIGHT = "traffic_light"
    STOP_SIGN = "stop_sign"
    TRAFFIC_SIGN = "traffic_sign"


class SignalAspect(str, Enum):
    """The lens layout a traffic-light head displays.

    Describes hardware, not a permitted manoeuvre -- kept distinct from the
    "left"/"right"/"through"/"straight" vocabularies used elsewhere for
    painted-arrow manoeuvres (``arrows.py``, ``Lane.turn_direction``), since
    ``BALL``/``PEDESTRIAN`` have no analogue there.
    """

    BALL = "ball"
    ARROW_LEFT = "arrow_left"
    ARROW_RIGHT = "arrow_right"
    ARROW_STRAIGHT = "arrow_straight"
    PEDESTRIAN = "pedestrian"
    UNKNOWN = "unknown"


@dataclass
class Detection:
    """One 2-D detection in one frame."""

    frame_id: str
    kind: LandmarkKind
    bbox: tuple[float, float, float, float]      # x0, y0, x1, y1 in pixels
    score: float
    camera: str

    @property
    def center(self) -> tuple[float, float]:
        x0, y0, x1, y1 = self.bbox
        return (0.5 * (x0 + x1), 0.5 * (y0 + y1))

    @property
    def size(self) -> tuple[float, float]:
        x0, y0, x1, y1 = self.bbox
        return (x1 - x0, y1 - y0)

    def to_dict(self) -> dict:
        return {"frame": self.frame_id, "kind": self.kind.value, "camera": self.camera,
                "bbox": [round(v, 1) for v in self.bbox], "score": round(self.score, 3)}


@dataclass
class Landmark:
    """A 3-D object triangulated from several views.

    ``position`` is in the pipeline's local metric frame with ``z`` in metres
    above the WGS84-ish local datum; ``height_above_ground`` is the physically
    meaningful quantity and is what the plausibility checks use.
    """

    id: str
    kind: LandmarkKind
    position: np.ndarray                          # (3,) local metric x, y, z
    detections: list[Detection] = field(default_factory=list)
    n_views: int = 0
    baseline_m: float = 0.0                       # max separation of observing cameras
    residual_px: float = 0.0                      # mean reprojection error
    position_sigma_m: float = 0.0                 # 1-sigma from the ray geometry
    height_above_ground: float | None = None
    facing: float | None = None                   # rad, direction the face points
    confidence: float = 0.0
    provenance: Provenance = field(default_factory=lambda: Provenance(Source.IMAGE))
    flags: list[str] = field(default_factory=list)
    aspect: SignalAspect | None = None
    aspect_confidence: float = 0.0

    def to_dict(self) -> dict:
        return {
            "id": self.id, "kind": self.kind.value,
            "position": [round(float(v), 3) for v in self.position],
            "n_views": self.n_views, "baseline_m": round(self.baseline_m, 2),
            "residual_px": round(self.residual_px, 2),
            "position_sigma_m": round(self.position_sigma_m, 2),
            "height_above_ground": None if self.height_above_ground is None
            else round(self.height_above_ground, 2),
            "facing_deg": None if self.facing is None
            else round(float(np.degrees(self.facing)), 1),
            "confidence": round(self.confidence, 3),
            "provenance": self.provenance.to_dict(),
            "flags": self.flags,
            "aspect": None if self.aspect is None else self.aspect.value,
            "aspect_confidence": round(self.aspect_confidence, 3),
            "detections": [d.to_dict() for d in self.detections],
        }


@dataclass
class StopLine:
    """A transverse marking across one approach."""

    id: str
    points: np.ndarray                            # (2, 2) local metric, left -> right
    segment_id: str                               # prior edge it belongs to
    lane_ids: list[str] = field(default_factory=list)
    observed: bool = True                         # False -> placed at the junction edge
    confidence: float = 0.0
    provenance: Provenance = field(default_factory=lambda: Provenance(Source.IMAGE))
    flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"id": self.id, "points": np.round(self.points, 3).tolist(),
                "segment": self.segment_id, "lanes": self.lane_ids,
                "observed": self.observed, "confidence": round(self.confidence, 3),
                "provenance": self.provenance.to_dict(), "flags": self.flags}


@dataclass
class LaneArrow:
    """A painted arrow observed inside one lane."""

    id: str
    lane_id: str
    position: np.ndarray                          # (2,) local metric
    manoeuvres: set[str]                          # subset of left/through/right/u_turn
    score: float = 0.0
    confidence: float = 0.0
    provenance: Provenance = field(default_factory=lambda: Provenance(Source.IMAGE))
    flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"id": self.id, "lane": self.lane_id,
                "position": [round(float(v), 2) for v in self.position],
                "manoeuvres": sorted(self.manoeuvres), "score": round(self.score, 3),
                "confidence": round(self.confidence, 3),
                "provenance": self.provenance.to_dict(), "flags": self.flags}


@dataclass
class TrafficLightAssignment:
    """A signal head, the stop line it governs, and the lanes behind it."""

    id: str
    landmark_id: str
    stop_line_id: str | None
    lane_ids: list[str]
    intersection_id: str | None
    confidence: float
    reason: str
    flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"id": self.id, "landmark": self.landmark_id,
                "stop_line": self.stop_line_id, "lanes": self.lane_ids,
                "intersection": self.intersection_id,
                "confidence": round(self.confidence, 3), "reason": self.reason,
                "flags": self.flags}


@dataclass
class SemanticLayer:
    """Everything the street-level stage produced for one AOI."""

    landmarks: dict[str, Landmark] = field(default_factory=dict)
    stop_lines: dict[str, StopLine] = field(default_factory=dict)
    arrows: dict[str, LaneArrow] = field(default_factory=dict)
    assignments: dict[str, TrafficLightAssignment] = field(default_factory=dict)
    detail: dict = field(default_factory=dict)

    def stats(self) -> dict:
        tls = [l for l in self.landmarks.values() if l.kind is LandmarkKind.TRAFFIC_LIGHT]
        signs = [l for l in self.landmarks.values() if l.kind is not LandmarkKind.TRAFFIC_LIGHT]
        assigned = {a.landmark_id for a in self.assignments.values() if a.lane_ids}
        return {
            "traffic_lights": len(tls),
            "traffic_signs": len(signs),
            "stop_lines": len(self.stop_lines),
            "stop_lines_observed": sum(1 for s in self.stop_lines.values() if s.observed),
            "lane_arrows": len(self.arrows),
            "assignments": len(self.assignments),
            "traffic_lights_assigned": len(assigned & {l.id for l in tls}),
            "mean_tl_views": round(float(np.mean([l.n_views for l in tls])), 2) if tls else 0.0,
            "median_tl_residual_px": round(float(np.median([l.residual_px for l in tls])), 2)
            if tls else 0.0,
            **self.detail,
        }

    def to_dict(self) -> dict:
        return {
            "stats": self.stats(),
            "landmarks": [l.to_dict() for l in self.landmarks.values()],
            "stop_lines": [s.to_dict() for s in self.stop_lines.values()],
            "arrows": [a.to_dict() for a in self.arrows.values()],
            "assignments": [a.to_dict() for a in self.assignments.values()],
        }
