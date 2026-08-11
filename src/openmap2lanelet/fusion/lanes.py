"""Lane structure inside a corridor: how many lanes, and where their edges are.

This is where the two information sources are actually arbitrated.

* The **imagery** proposes lane boundaries: persistent bright linear structures
  running along the corridor.  It is good at *where* a boundary is and blind to
  what it means.
* The **prior** proposes a lane count and a direction split.  It is often right
  about the count and always right about one-way-ness, but it says nothing
  about position.

The arbitration is deliberately explicit and logged, because "which source won,
and why" is one of the outputs of the PoC.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from scipy.signal import find_peaks

from ..prior import osm_tags
from ..types import MarkingType, Source
from .corridor import Corridor
from .profile import Profile, median_along, smooth_along

log = logging.getLogger(__name__)

MIN_LANE_W = 2.4
MAX_LANE_W = 5.0


@dataclass
class BoundaryTrack:
    """A lane boundary as a per-station lateral offset from the prior line."""

    offsets: np.ndarray                 # (S,) metres, +left
    marking: MarkingType
    source: Source
    coverage: float = 0.0               # fraction of stations with a marking hit
    strength: float = 0.0               # mean marking response along the track
    detail: dict = field(default_factory=dict)

    @property
    def observed(self) -> bool:
        return self.source is Source.IMAGE


@dataclass
class LaneSlot:
    """One lane: the interval between two consecutive boundaries."""

    left_idx: int
    right_idx: int
    forward: bool
    width: float
    confidence: float


@dataclass
class CarriagewaySolution:
    boundaries: list[BoundaryTrack]     # ordered left -> right, len == n_lanes + 1
    lanes: list[LaneSlot]
    n_osm: int
    n_image: int | None
    count_source: Source
    confidence: float
    flags: list[str] = field(default_factory=list)
    detail: dict = field(default_factory=dict)

    @property
    def n_lanes(self) -> int:
        return len(self.lanes)


def solve_lanes(prof: Profile, cor: Corridor, tags: dict[str, str], *,
                drive_on_right: bool = True, marking_thr: float = 0.30,
                min_coverage: float = 0.18) -> CarriagewaySolution:
    """Fuse marking observations with the prior's lane metadata."""
    flags: list[str] = []
    n_osm_v = osm_tags.lane_count(tags)
    w_lane = float(osm_tags.lane_width(tags).value)
    is_dual = bool(cor.detail.get("dual_carriageway"))
    oneway = bool(osm_tags.is_oneway(tags).value)

    n_osm = int(n_osm_v.value)
    if is_dual:
        fwd, bwd = osm_tags.directional_split(tags, n_osm)
        n_osm = max(1, fwd if cor.side == ("right" if drive_on_right else "left") else bwd)

    peaks, score, v_grid = _marking_peaks(prof, cor, marking_thr)
    tracks = [_track_boundary(prof, cor, v, marking_thr) for v in peaks]
    tracks = [t for t in tracks if t.coverage >= min_coverage]
    tracks.sort(key=lambda t: -_median(t.offsets))          # left -> right

    edges = _corridor_edges(cor)
    interior = [t for t in tracks
                if _median(t.offsets) < _median(edges[0].offsets) - 1.0
                and _median(t.offsets) > _median(edges[1].offsets) + 1.0]

    boundaries = [edges[0]] + interior + [edges[1]]
    n_image = len(boundaries) - 1 if interior else None

    boundaries, subdivided, merged = _regularise(boundaries, w_lane)
    n_geom = len(boundaries) - 1

    n_lanes, source, conf = _arbitrate(n_geom, n_osm, n_osm_v.source, bool(interior),
                                       cor.mean_width(), w_lane, flags)

    if n_lanes != n_geom:
        # The arbitration overruled the geometry: fall back to an even split of
        # the corridor, keeping any observed boundary that still fits.
        boundaries = _uniform_split(cor, n_lanes)
        flags.append("lane_geometry_overruled_by_prior")

    if not interior:
        flags.append("no_lane_markings_observed")
    if subdivided:
        flags.append("boundary_gap_filled")
    if merged:
        flags.append("boundary_pair_merged")
    if cor.clamped is not None and cor.valid.any() and cor.clamped[cor.valid].mean() > 0.25:
        flags.append("corridor_width_clamped")
    if cor.coverage < 0.7:
        flags.append("corridor_partially_unobserved")

    lanes = _assign_directions(boundaries, tags, n_lanes, cor, is_dual, oneway,
                               drive_on_right, conf, flags)

    detail = {
        "n_osm": n_osm,
        "n_osm_source": n_osm_v.source.value,
        "n_image": n_image,
        "n_geometry": n_geom,
        "corridor_width": round(cor.mean_width(), 2),
        "corridor_coverage": round(cor.coverage, 3),
        "observed_boundaries": len(interior),
        "dual_carriageway": is_dual,
        "carriageway_side": cor.side,
        "lane_width_prior": w_lane,
    }
    return CarriagewaySolution(boundaries, lanes, n_osm, n_image, source, conf, flags, detail)


# --------------------------------------------------------------------------- #
# marking detection
# --------------------------------------------------------------------------- #


def _marking_peaks(prof: Profile, cor: Corridor, thr: float):
    """Aggregate the marking response in corridor-relative coordinates."""
    du = float(prof.offsets[1] - prof.offsets[0])
    half = max(3.0, 0.5 * cor.mean_width() + 1.0)
    v_grid = np.arange(-half, half + 1e-9, du)

    rows = []
    for i in range(prof.n_stations):
        if not cor.valid[i]:
            continue
        u = v_grid + cor.center_offset[i]
        row = np.interp(u, prof.offsets, prof.mark[i], left=0.0, right=0.0)
        inside = (u <= cor.left[i] - 0.25) & (u >= cor.right[i] + 0.25)
        rows.append(np.where(inside, row, np.nan))
    if not rows:
        return [], np.zeros_like(v_grid), v_grid

    stack = np.array(rows)
    seen = np.isfinite(stack).sum(axis=0)

    # Score a lateral offset by *how often* a marking is present there, not by
    # its mean response.  A dashed line is painted about a quarter of the time,
    # so averaging amplitude buries it under the solid centre line; a hit rate
    # separates "dashed line" (~0.25) from "sensor noise" (~0.02) cleanly and
    # is invariant to how bright the paint happens to be.
    with np.errstate(invalid="ignore"):
        hits = np.nansum(np.where(np.isfinite(stack), stack >= thr, 0.0), axis=0)
        amp = np.nansum(np.nan_to_num(stack), axis=0)
    denom = np.maximum(seen, 1)
    score = hits / denom + 0.25 * (amp / denom)
    # An offset only seen by a handful of stations is not evidence of a line.
    score = np.where(seen >= max(3, 0.15 * len(rows)), score, 0.0)
    score = smooth_along(score, max(3, int(round(0.4 / du))))

    if score.max() <= 1e-6:
        return [], score, v_grid
    idx, props = find_peaks(score, distance=max(2, int(round(2.0 / du))),
                            height=max(0.12, 0.22 * score.max()),
                            prominence=0.12 * score.max())
    order = np.argsort(-props["peak_heights"])
    return [float(v_grid[i]) for i in idx[order]], score, v_grid


def _track_boundary(prof: Profile, cor: Corridor, v0: float, thr: float,
                    search: float = 0.7) -> BoundaryTrack:
    """Follow a marking peak along the segment, station by station."""
    S = prof.n_stations
    offs = np.full(S, np.nan)
    hits = np.zeros(S, dtype=bool)
    strengths = []

    for i in range(S):
        if not cor.valid[i]:
            continue
        u0 = v0 + cor.center_offset[i]
        lo, hi = u0 - search, u0 + search
        sel = (prof.offsets >= lo) & (prof.offsets <= hi)
        if not sel.any():
            continue
        vals = prof.mark[i][sel]
        j = int(np.argmax(vals))
        offs[i] = float(prof.offsets[sel][j])
        if vals[j] >= thr:
            hits[i] = True
            strengths.append(float(vals[j]))

    valid = np.isfinite(offs)
    if not valid.any():
        return BoundaryTrack(np.full(S, v0), MarkingType.UNKNOWN, Source.IMAGE, 0.0, 0.0)

    filled = np.interp(np.arange(S), np.flatnonzero(valid), offs[valid])
    # Only smooth where the track was actually seen; unobserved stretches are
    # already linear interpolations.
    smoothed = smooth_along(median_along(filled, 9), 7)

    coverage = float(hits[cor.valid].mean()) if cor.valid.any() else 0.0
    strength = float(np.mean(strengths)) if strengths else 0.0
    marking = _classify_marking(coverage)
    return BoundaryTrack(smoothed, marking, Source.IMAGE, coverage, strength,
                         {"v0": round(v0, 2)})


def _classify_marking(coverage: float) -> MarkingType:
    """Solid lines are continuous; dashed lines are hit ~1 station in 4."""
    if coverage >= 0.72:
        return MarkingType.SOLID
    if coverage >= 0.18:
        return MarkingType.DASHED
    return MarkingType.UNKNOWN


def _corridor_edges(cor: Corridor) -> tuple[BoundaryTrack, BoundaryTrack]:
    left = BoundaryTrack(cor.left.copy(), MarkingType.ROAD_EDGE, Source.IMAGE,
                         coverage=cor.coverage, detail={"edge": "left"})
    right = BoundaryTrack(cor.right.copy(), MarkingType.ROAD_EDGE, Source.IMAGE,
                          coverage=cor.coverage, detail={"edge": "right"})
    return left, right


# --------------------------------------------------------------------------- #
# arbitration
# --------------------------------------------------------------------------- #


def _median(a: np.ndarray) -> float:
    return float(np.nanmedian(a))


def _regularise(bounds: list[BoundaryTrack], w_lane: float
                ) -> tuple[list[BoundaryTrack], bool, bool]:
    """Make the boundary sequence produce plausible lane widths.

    Two edits, both of which encode real-world knowledge rather than the map:
    a gap wide enough for k lanes but with no marking in it gets k-1 *virtual*
    boundaries (the markings were occluded or worn away); two boundaries closer
    together than a lane are one boundary (a double line, or a line detected
    twice).
    """
    merged = False
    out: list[BoundaryTrack] = [bounds[0]]
    for b in bounds[1:-1]:
        if _median(out[-1].offsets) - _median(b.offsets) < MIN_LANE_W * 0.75:
            # keep the stronger of the pair, and remember it was a double line
            if b.strength > out[-1].strength and out[-1].marking is not MarkingType.ROAD_EDGE:
                out[-1] = b
            if out[-1].marking is not MarkingType.ROAD_EDGE:
                out[-1].marking = MarkingType.DOUBLE_SOLID
                out[-1].detail["merged_pair"] = True
            merged = True
            continue
        out.append(b)
    if _median(out[-1].offsets) - _median(bounds[-1].offsets) < MIN_LANE_W * 0.75 and len(out) > 1:
        out.pop()
        merged = True
    out.append(bounds[-1])

    subdivided = False
    filled: list[BoundaryTrack] = [out[0]]
    for a, b in zip(out, out[1:]):
        w = _median(a.offsets) - _median(b.offsets)
        k = int(round(w / w_lane))
        if k >= 2 and w / k >= MIN_LANE_W:
            for t in range(1, k):
                f = t / k
                filled.append(BoundaryTrack(a.offsets * (1 - f) + b.offsets * f,
                                            MarkingType.VIRTUAL, Source.INFERRED,
                                            detail={"reason": "unmarked lane split"}))
            subdivided = True
        filled.append(b)
    return filled, subdivided, merged


def _arbitrate(n_geom: int, n_osm: int, osm_source: Source, saw_markings: bool,
               corridor_width: float, w_lane: float, flags: list[str]
               ) -> tuple[int, Source, float]:
    """Decide the lane count and how much to believe it."""
    n_width = int(round(corridor_width / w_lane)) if corridor_width > 0 else 0
    n_geom = max(1, n_geom)

    if not saw_markings:
        # Nothing observed: the prior is all we have.
        if n_width and abs(n_width - n_osm) <= 1:
            return n_osm, Source.OSM, 0.45
        flags.append("lane_count_unverified")
        return n_osm, Source.OSM, 0.30

    if n_geom == n_osm:
        return n_geom, Source.FUSED, 0.85

    flags.append("lane_count_conflict")
    if abs(n_geom - n_osm) == 1 and n_geom == n_width:
        # geometry is self-consistent (markings agree with corridor width)
        return n_geom, Source.IMAGE, 0.60
    if abs(n_geom - n_osm) <= 1:
        return n_geom, Source.IMAGE, 0.50
    if osm_source is Source.OSM and n_width and abs(n_width - n_osm) <= 1:
        # the map is explicit and the corridor width backs it: markings were
        # probably mis-detected (parking bays, kerb shadows)
        flags.append("markings_rejected_by_prior")
        return n_osm, Source.OSM, 0.40
    return n_geom, Source.IMAGE, 0.35


def _uniform_split(cor: Corridor, n: int) -> list[BoundaryTrack]:
    out = []
    for k in range(n + 1):
        f = k / n
        offs = cor.left * (1 - f) + cor.right * f
        marking = MarkingType.ROAD_EDGE if k in (0, n) else MarkingType.VIRTUAL
        src = Source.IMAGE if k in (0, n) else Source.INFERRED
        out.append(BoundaryTrack(offs, marking, src, detail={"reason": "uniform split"}))
    return out


def _assign_directions(bounds: list[BoundaryTrack], tags: dict[str, str], n: int,
                       cor: Corridor, is_dual: bool, oneway: bool,
                       drive_on_right: bool, conf: float, flags: list[str]
                       ) -> list[LaneSlot]:
    """Split the lanes of one corridor into forward / backward."""
    widths = [_median(bounds[i].offsets) - _median(bounds[i + 1].offsets) for i in range(n)]

    if is_dual:
        forward_side = "right" if drive_on_right else "left"
        fwd_all = cor.side == forward_side
        return [LaneSlot(i, i + 1, fwd_all, widths[i], conf) for i in range(n)]

    if oneway or n == 1:
        return [LaneSlot(i, i + 1, True, widths[i], conf) for i in range(n)]

    fwd, bwd = osm_tags.directional_split(tags, n)
    if fwd + bwd != n:
        fwd = n // 2
        bwd = n - fwd
    # Boundaries run left -> right. With right-hand traffic the forward lanes
    # (along the prior edge direction) are the right-hand ones.
    slots = []
    for i in range(n):
        from_right = n - 1 - i
        forward = (from_right < fwd) if drive_on_right else (i < fwd)
        slots.append(LaneSlot(i, i + 1, forward, widths[i], conf))

    divider = n - fwd if drive_on_right else fwd
    if 0 < divider < n:
        b = bounds[divider]
        if b.marking is MarkingType.VIRTUAL:
            flags.append("centre_divider_unobserved")
        elif b.marking is MarkingType.DASHED:
            flags.append("centre_divider_dashed")
    return slots
