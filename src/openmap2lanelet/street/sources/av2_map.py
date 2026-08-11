"""The AV2 HD map, used two ways -- and never both at once for the same claim.

1. **As an OSM stand-in prior.**  OSM is not reachable from this environment, so
   the topology prior is synthesised from the HD map by throwing away
   everything OSM would not have: lane-level geometry, the true lane count,
   mark types, turn permissions and intersection interiors.  What survives is a
   simplified road centreline plus a road class -- which is what an OSM way is.
   The degradation is explicit and controlled by ``lanes_tag``.

2. **As evaluation ground truth.**  The full lane segments, with their true
   count, boundaries and intersection flags, are loaded separately and are
   never seen by the pipeline.

Keeping these in one module makes the separation auditable: ``build_prior``
cannot reach the fields ``ground_truth`` returns.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ...geo import AOI, LocalFrame, polyline_length, resample_polyline, simplify_polyline
from ...sources.base import PriorData, PriorWay

log = logging.getLogger(__name__)


@dataclass
class LaneSegmentGT:
    id: int
    center: np.ndarray                 # (N, 3) local metric
    left: np.ndarray
    right: np.ndarray
    is_intersection: bool
    left_mark: str
    right_mark: str
    successors: list[int] = field(default_factory=list)
    left_neighbor: int | None = None
    right_neighbor: int | None = None

    @property
    def heading(self) -> float:
        d = self.center[-1, :2] - self.center[0, :2]
        return float(np.arctan2(d[1], d[0]))


@dataclass
class AV2GroundTruth:
    lanes: dict[int, LaneSegmentGT]
    carriageways: list[list[int]]      # lane ids grouped into physical carriageways
    crossings: list[np.ndarray]
    drivable: list[np.ndarray]

    def lane_count_at(self, xy: np.ndarray, max_dist: float = 12.0) -> int | None:
        """True number of lanes in the carriageway nearest ``xy``."""
        best, best_d = None, max_dist
        for group in self.carriageways:
            for lid in group:
                c = self.lanes[lid].center[:, :2]
                d = float(np.min(np.hypot(*(c - np.asarray(xy)[:2]).T)))
                if d < best_d:
                    best_d, best = d, group
        return len(best) if best else None

    def stats(self) -> dict:
        return {"lane_segments": len(self.lanes),
                "intersection_segments": sum(1 for l in self.lanes.values() if l.is_intersection),
                "carriageways": len(self.carriageways),
                "crossings": len(self.crossings)}


def _poly(points: list[dict]) -> np.ndarray:
    return np.array([[p["x"], p["y"], p.get("z", 0.0)] for p in points], dtype=float)


def load_map(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def _to_local(arr: np.ndarray, georef, frame: LocalFrame) -> np.ndarray:
    return georef.city_to_local(arr, frame)


def ground_truth(raw: dict, georef, frame: LocalFrame) -> AV2GroundTruth:
    """Full lane-level truth, in the local metric frame."""
    lanes: dict[int, LaneSegmentGT] = {}
    for v in raw.get("lane_segments", {}).values():
        left = _to_local(_poly(v["left_lane_boundary"]), georef, frame)
        right = _to_local(_poly(v["right_lane_boundary"]), georef, frame)
        n = max(len(left), len(right), 2)
        li = resample_polyline(left[:, :2], max(polyline_length(left[:, :2]) / n, 0.5))
        ri = resample_polyline(right[:, :2], max(polyline_length(right[:, :2]) / n, 0.5))
        m = min(len(li), len(ri))
        if m < 2:
            continue
        center = 0.5 * (li[:m] + ri[:m])
        lanes[int(v["id"])] = LaneSegmentGT(
            id=int(v["id"]),
            center=np.column_stack([center, np.zeros(len(center))]),
            left=left, right=right,
            is_intersection=bool(v.get("is_intersection", False)),
            left_mark=str(v.get("left_lane_mark_type", "NONE")),
            right_mark=str(v.get("right_lane_mark_type", "NONE")),
            successors=[int(s) for s in v.get("successors", []) or []],
            left_neighbor=v.get("left_neighbor_id"),
            right_neighbor=v.get("right_neighbor_id"),
        )

    crossings = [_to_local(_poly(v["edge1"]), georef, frame) for v in
                 raw.get("pedestrian_crossings", {}).values()]
    drivable = [_to_local(_poly(v["area_boundary"]), georef, frame) for v in
                raw.get("drivable_areas", {}).values()]
    gt = AV2GroundTruth(lanes, _carriageways(lanes), crossings, drivable)
    log.info("av2 ground truth: %s", gt.stats())
    return gt


def _carriageways(lanes: dict[int, LaneSegmentGT]) -> list[list[int]]:
    """Group laterally-adjacent lanes (excluding junction interiors)."""
    ids = [i for i, l in lanes.items() if not l.is_intersection]
    parent = {i: i for i in ids}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in ids:
        for nb in (lanes[i].left_neighbor, lanes[i].right_neighbor):
            if nb is not None and int(nb) in parent:
                ra, rb = find(i), find(int(nb))
                if ra != rb:
                    parent[ra] = rb

    groups: dict[int, list[int]] = {}
    for i in ids:
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


# --------------------------------------------------------------------------- #
# prior
# --------------------------------------------------------------------------- #


def build_prior(gt: AV2GroundTruth, frame: LocalFrame, *, lanes_tag: str = "none",
                simplify_m: float = 2.0, jitter_m: float = 0.0,
                seed: int = 0) -> PriorData:
    """Degrade the HD map into something with OSM's information content.

    ``lanes_tag`` controls the one piece of metadata the experiment turns on
    and off:

    * ``none``   -- no ``lanes`` tag at all (the common OSM case: the pipeline
      must derive the count from imagery or fall back to a class default);
    * ``class``  -- a nominal count from the road class, as if a mapper had
      guessed;
    * ``true``   -- the real count, as an upper bound on what a perfect map
      would give.
    """
    rng = np.random.default_rng(seed)
    ways: list[PriorWay] = []

    inter_clusters = _intersection_clusters(gt)
    for gi, group in enumerate(gt.carriageways):
        centre = _carriageway_centreline(gt, group)
        if centre is None or polyline_length(centre) < 8.0:
            continue
        centre = _extend_to_junctions(gt, group, centre, inter_clusters)
        centre = simplify_polyline(centre, simplify_m)
        if jitter_m > 0:
            centre = centre + rng.normal(0, jitter_m, size=(1, 2))

        n_true = len(group)
        headings = np.array([gt.lanes[l].heading for l in group])
        spread = np.abs(np.angle(np.exp(1j * (headings - headings[0]))))
        oneway = bool(np.all(spread < np.radians(60)))
        highway = _class_for(n_true, oneway)

        tags = {"highway": highway, "oneway": "yes" if oneway else "no"}
        if lanes_tag == "true":
            tags["lanes"] = str(n_true)
        elif lanes_tag == "class":
            from ...prior.osm_tags import CLASS_DEFAULTS
            tags["lanes"] = str(CLASS_DEFAULTS.get(highway, (2,))[0])

        ways.append(PriorWay(f"av2_{gi}", _to_wgs84(centre, frame), tags))

    log.info("av2 prior: %s ways (lanes_tag=%s, simplify=%.1f m, jitter=%.1f m)",
             len(ways), lanes_tag, simplify_m, jitter_m)
    return PriorData(ways=ways,
                     attribution="Argoverse 2 HD map, degraded to OSM information content",
                     kind="osm-like",
                     detail={"lanes_tag": lanes_tag, "simplify_m": simplify_m,
                             "jitter_m": jitter_m})


def _to_wgs84(xy: np.ndarray, frame: LocalFrame) -> np.ndarray:
    """``PriorWay`` carries WGS84, but the prior is assembled in metres."""
    lon, lat = frame.to_wgs84(xy[:, 0], xy[:, 1])
    return np.column_stack([lon, lat])


def _class_for(n_lanes: int, oneway: bool) -> str:
    total = n_lanes if not oneway else n_lanes * 2
    if total >= 6:
        return "primary"
    if total >= 4:
        return "secondary"
    if total >= 3:
        return "tertiary"
    return "residential"


def _carriageway_centreline(gt: AV2GroundTruth, group: list[int]) -> np.ndarray | None:
    """Average the lanes of a carriageway into a single road centreline.

    Lanes of one carriageway rarely span the same stations -- a right-turn bay
    starts halfway along, the kerb lane ends at a bus stop.  Sampling every lane
    by nearest point would then *clamp* the short ones to their own end and drag
    the average sideways, producing a hook and a run of identical points at the
    tip.  A clamped sample carries no information about that station, so it is
    simply not counted.
    """
    lanes = [gt.lanes[i] for i in group]
    ref = max(lanes, key=lambda l: polyline_length(l.center[:, :2]))
    ref_c = resample_polyline(ref.center[:, :2], 2.0)
    if len(ref_c) < 2:
        return None

    acc = np.zeros_like(ref_c)
    cnt = np.zeros(len(ref_c))
    for ln in lanes:
        c = ln.center[:, :2]
        if len(c) < 2:
            continue
        idx = np.argmin(np.linalg.norm(c[None, :, :] - ref_c[:, None, :], axis=2), axis=1)
        interior = (idx > 0) & (idx < len(c) - 1)
        acc[interior] += c[idx[interior]]
        cnt[interior] += 1
    missing = cnt < 1
    acc[missing] = ref_c[missing]
    cnt[missing] = 1
    out = acc / cnt[:, None]

    keep = np.concatenate([[True], np.hypot(*np.diff(out, axis=0).T) > 1e-3])
    out = out[keep]
    return out if len(out) >= 2 else None


@dataclass
class JunctionGT:
    """A physical junction: its interior lanes and the roads that meet there."""

    id: int
    lane_ids: set[int]
    center: np.ndarray
    entries: set[int] = field(default_factory=set)     # non-intersection lanes feeding in
    exits: set[int] = field(default_factory=set)       # non-intersection lanes fed


def _intersection_clusters(gt: AV2GroundTruth, link_m: float = 10.0) -> list[JunctionGT]:
    """Group interior lanes into junctions by *connectivity*, not proximity.

    Proximity alone splits a wide junction into several clumps -- the left-turn
    and right-turn paths of one intersection can be 40 m apart -- and then every
    arm attaches to a different phantom junction, which is exactly the failure
    that leaves the exported map with no shared nodes and no routable topology.

    Two interior lanes belong to the same junction when they share a road *in
    the same role*: both are movements **out of** the same approach lane, or
    both are movements **into** the same exit lane.  Sharing a road in opposite
    roles is the signature of two *different* junctions joined by a block -- the
    lane leaving one is the lane entering the next -- so treating "shares a road"
    as one relation merges a whole corridor into a single phantom intersection.

    Proximity is kept only as a weak extra link, for junctions whose interior
    lanes are not connected through a common road at all.
    """
    inter = [i for i, l in gt.lanes.items() if l.is_intersection]
    if not inter:
        return []

    preds: dict[int, list[int]] = {}
    for i, l in gt.lanes.items():
        for s in l.successors:
            preds.setdefault(int(s), []).append(i)

    parent = {i: i for i in inter}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    by_entry: dict[int, list[int]] = {}
    by_exit: dict[int, list[int]] = {}
    for i in inter:
        for road in gt.lanes[i].successors:
            road = int(road)
            if road in gt.lanes and not gt.lanes[road].is_intersection:
                by_exit.setdefault(road, []).append(i)
        for road in preds.get(i, []):
            if road in gt.lanes and not gt.lanes[road].is_intersection:
                by_entry.setdefault(road, []).append(i)
    for table in (by_entry, by_exit):
        for group in table.values():
            for j in group[1:]:
                union(group[0], j)

    # an interior lane's own successors inside the junction (multi-stage crossings)
    for i in inter:
        for s in gt.lanes[i].successors:
            if int(s) in parent:
                union(i, int(s))

    cent = {i: gt.lanes[i].center[:, :2].mean(axis=0) for i in inter}
    for a_i, a in enumerate(inter):
        for b in inter[a_i + 1:]:
            if float(np.hypot(*(cent[a] - cent[b]))) <= link_m:
                union(a, b)

    groups: dict[int, list[int]] = {}
    for i in inter:
        groups.setdefault(find(i), []).append(i)

    out: list[JunctionGT] = []
    for k, (_, ids) in enumerate(sorted(groups.items())):
        pts = np.vstack([gt.lanes[i].center[:, :2] for i in ids])
        j = JunctionGT(id=k, lane_ids=set(ids), center=pts.mean(axis=0))
        for i in ids:
            for s in gt.lanes[i].successors:
                if int(s) in gt.lanes and not gt.lanes[int(s)].is_intersection:
                    j.exits.add(int(s))
            for p in preds.get(i, []):
                if p in gt.lanes and not gt.lanes[p].is_intersection:
                    j.entries.add(p)
        out.append(j)
    log.info("av2 junctions: %s from %s interior lanes", len(out), len(inter))
    return out


def _extend_to_junctions(gt: AV2GroundTruth, group: list[int], centre: np.ndarray,
                         junctions: list[JunctionGT], reach: float = 45.0) -> np.ndarray:
    """Run the centreline into the junction centres this carriageway meets.

    OSM ways meet *at a shared node* in the middle of a junction; a road that
    merely stops at the stop bar produces no junction at all downstream, and the
    exported lanelets are then a set of disconnected stubs.  Which junction an
    arm belongs to is read off the HD map's own lane connectivity rather than
    guessed from distance, so two arms of the same intersection always land on
    literally the same point.
    """
    members = set(group)
    linked = [j for j in junctions if (members & j.entries) or (members & j.exits)]
    if not linked:
        return centre

    out = centre
    for end in (0, -1):
        p, q = out[end], (out[-1] if end == 0 else out[0])
        # the junction this end faces: nearer to it than to the other end, and
        # ahead of it, so the line is lengthened rather than folded back
        cands = []
        for j in linked:
            d = float(np.hypot(*(j.center - p)))
            if d > reach or d >= float(np.hypot(*(j.center - q))):
                continue
            k = min(4, len(out) - 1)
            tangent = out[end] - (out[-1 - k] if end == -1 else out[k])
            if np.dot(j.center - p, tangent) <= 0:
                continue
            cands.append((d, j))
        if not cands:
            continue
        j = min(cands, key=lambda t: t[0])[1]
        out = np.vstack([out, j.center]) if end == -1 else np.vstack([j.center, out])
    return out


def aoi_for_ground_truth(gt: AV2GroundTruth, frame: LocalFrame, margin: float = 30.0) -> AOI:
    pts = np.vstack([l.center[:, :2] for l in gt.lanes.values()])
    lon, lat = frame.to_wgs84(
        [pts[:, 0].min() - margin, pts[:, 0].max() + margin],
        [pts[:, 1].min() - margin, pts[:, 1].max() + margin])
    return AOI("av2_aoi", float(lon[0]), float(lat[0]), float(lon[1]), float(lat[1]))
