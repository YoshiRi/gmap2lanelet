"""Lane strips: a lane before it becomes a lanelet.

A strip carries its centreline *and* both boundaries sampled at the same
stations, so trimming it at an intersection is a slice, and the boundaries
never drift out of correspondence with the centreline.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..geo import polyline_length
from ..types import MarkingType, Provenance, Source


@dataclass
class StripBoundary:
    points: np.ndarray                  # (S, 2) world metres
    marking: MarkingType
    source: Source
    coverage: float = 0.0
    strength: float = 0.0
    detail: dict = field(default_factory=dict)
    # boundaries are shared between adjacent lanes: this key identifies them
    key: str = ""


@dataclass
class LaneStrip:
    """One lane, oriented in its direction of travel."""

    id: str
    segment_id: str                     # prior edge id
    group: str                          # strips sharing a station grid and orientation
    center: np.ndarray                  # (S, 2) world metres
    left: StripBoundary
    right: StripBoundary
    start_node: str                     # prior node the strip leaves
    end_node: str                       # prior node the strip enters
    index_from_left: int
    width: float
    speed_limit_kph: float | None
    one_way: bool
    confidence: float
    provenance: Provenance
    flags: list[str] = field(default_factory=list)
    detail: dict = field(default_factory=dict)
    trimmed_start: bool = False
    trimmed_end: bool = False

    @property
    def length(self) -> float:
        return polyline_length(self.center)

    def entry_pose(self) -> tuple[np.ndarray, float]:
        return _pose(self.center, at_start=True)

    def exit_pose(self) -> tuple[np.ndarray, float]:
        return _pose(self.center, at_start=False)


def slice_group(strips: list[LaneStrip], i0: int, i1: int) -> None:
    """Keep stations ``[i0, i1)`` for a whole group of strips.

    Lanes of one carriageway *share* their boundary objects (lane k's left edge
    is lane k+1's right edge), so a boundary must be sliced exactly once no
    matter how many lanes reference it -- and every lane of the group must be
    cut at the same station, or the shared boundaries stop corresponding to the
    centrelines.
    """
    if not strips:
        return
    n = len(strips[0].center)
    i0 = max(0, min(i0, n - 2))
    i1 = max(i0 + 2, min(i1, n))

    seen: set[int] = set()
    for s in strips:
        s.center = s.center[i0:i1]
        for b in (s.left, s.right):
            if id(b) in seen:
                continue
            seen.add(id(b))
            b.points = b.points[i0:i1]


def _pose(pts: np.ndarray, at_start: bool) -> tuple[np.ndarray, float]:
    import math

    if at_start:
        p, d = pts[0], pts[1] - pts[0]
    else:
        p, d = pts[-1], pts[-1] - pts[-2]
    return p.copy(), math.atan2(d[1], d[0])
