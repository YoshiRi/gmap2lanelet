"""Intersection topology.

This is the part of the problem where public data runs out, and the PoC is
explicit about it.

* Aerial imagery shows the *shape* of the junction (the paved area) and,
  sometimes, stop bars and turn arrows -- but at 0.3 m/px arrows are a few
  pixels and unreadable, and the interior of a junction carries no lane
  markings at all by design.
* OSM knows the junction *exists* and which ways meet there, but the tag that
  would answer "which lane may turn where" (``turn:lanes``) is essentially
  never present: in the AOIs used here it is absent on 100% of edges.

So connectivity inside a junction is **inferred from a rule**, not observed.
Every lane produced here is marked ``Source.INFERRED``, given a low confidence,
and raised as a review item.  Getting this wrong quietly would be the worst
possible outcome for a map that a vehicle drives on.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np

from ..config import PipelineConfig
from ..geo import angle_diff, hermite, offset_polyline, resample_polyline
from ..prior import osm_tags
from ..prior.road_graph import RoadPrior
from ..types import Intersection, MarkingType, Provenance, Source
from .segments import SegmentResult
from .strips import LaneStrip, StripBoundary, slice_group

log = logging.getLogger(__name__)


@dataclass
class Approach:
    """One direction of one road arm at a junction."""

    edge_id: str
    node_id: str
    incoming: list[LaneStrip] = field(default_factory=list)   # travel into the junction
    outgoing: list[LaneStrip] = field(default_factory=list)   # travel out of it
    heading_in: float = 0.0        # heading of traffic arriving (rad)
    heading_out: float = 0.0
    tags: dict = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.edge_id}@{self.node_id}"


def resolve_intersections(prior: RoadPrior, segments: dict[str, SegmentResult],
                          cfg: PipelineConfig) -> tuple[dict[str, Intersection], list[LaneStrip],
                                                        list[tuple[str, str]]]:
    """Trim strips back from junctions and connect them with turn lanes.

    Returns ``(intersections, turn_strips, connections)`` where a connection is
    an ``(upstream_strip_id, downstream_strip_id)`` pair.
    """
    by_id = {s.id: s for r in segments.values() for s in r.strips}
    intersections: dict[str, Intersection] = {}
    turns: list[LaneStrip] = []
    connections: list[tuple[str, str]] = []

    # Pass 1: work out every junction's extent, and how far back each group of
    # lanes has to be cut.  Trims are accumulated first and applied once,
    # because a segment can be trimmed at both ends by two different junctions.
    plans: dict[str, tuple] = {}
    trims: dict[str, list[int]] = {}
    groups: dict[str, list[LaneStrip]] = {}
    for r in segments.values():
        for s in r.strips:
            groups.setdefault(s.group, []).append(s)
    for g, ss in groups.items():
        trims[g] = [0, len(ss[0].center)]

    for cid, cluster in sorted(prior.clusters.items()):
        approaches = _collect_approaches(prior, cluster, segments, by_id)
        if len(approaches) < 2:
            continue
        radius = _radius(cluster, approaches, cfg)
        _plan_trims(approaches, cluster.center, radius, groups, trims)
        plans[cid] = (cluster, approaches, radius)

    # Pass 2: apply, once per group.
    for g, (i0, i1) in trims.items():
        slice_group(groups[g], i0, i1)

    # Pass 3: build the connectivity on the trimmed geometry.
    for cid, (cluster, approaches, radius) in plans.items():
        inter = Intersection(
            id=cid, center=cluster.center.copy(), radius=radius,
            prior_node_ids=list(cluster.node_ids), approach_count=len(approaches),
            provenance=Provenance(Source.INFERRED, {
                "rule": "geometric manoeuvre model",
                "turn_lanes_tagged": any(_turn_spec(a) for a in approaches),
            }),
        )
        _update_headings(approaches, cluster.center)

        new_turns, new_conns, notes = _connect(approaches, inter, cfg)
        turns.extend(new_turns)
        connections.extend(new_conns)
        inter.turn_lane_ids = [t.id for t in new_turns]
        inter.notes = notes
        inter.confidence = _intersection_confidence(approaches, new_turns, notes)
        intersections[cid] = inter

    log.info("intersections: %s resolved, %s turn lanes", len(intersections), len(turns))
    return intersections, turns, connections


# --------------------------------------------------------------------------- #


def _collect_approaches(prior, cluster, segments, by_id) -> list[Approach]:
    node_set = set(cluster.node_ids)
    out: dict[str, Approach] = {}
    for nid in cluster.node_ids:
        for eid in prior.nodes[nid].edge_ids:
            res = segments.get(eid)
            if res is None:
                continue
            # An edge with both ends in the cluster is internal to it; ignore.
            e = prior.edges[eid]
            if e.from_node in node_set and e.to_node in node_set:
                continue
            key = f"{eid}@{nid}"
            ap = out.setdefault(key, Approach(eid, nid, tags=dict(e.tags)))
            for s in res.strips:
                if s.end_node == nid:
                    ap.incoming.append(s)
                elif s.start_node == nid:
                    ap.outgoing.append(s)
    approaches = [a for a in out.values() if a.incoming or a.outgoing]
    for a in approaches:
        a.incoming = _sort_left_to_right(a.incoming, at_end=True)
        a.outgoing = _sort_left_to_right(a.outgoing, at_end=False)
    return approaches


def _sort_left_to_right(strips: list[LaneStrip], at_end: bool) -> list[LaneStrip]:
    """Order lanes left-to-right *as the driver sees them*.

    Lane slot indices are left-to-right in the prior edge's own direction, so
    for lanes travelling against that direction the order is inverted.  Sorting
    by lateral position in the travel frame is direction-agnostic.
    """
    if len(strips) < 2:
        return strips
    poses = [s.exit_pose() if at_end else s.entry_pose() for s in strips]
    h = np.mean([p[1] for p in poses])
    n = np.array([-math.sin(h), math.cos(h)])          # left-hand normal
    return [s for _, s in sorted(zip([float(np.dot(p[0], n)) for p in poses], strips),
                                 key=lambda t: -t[0])]


def _radius(cluster, approaches: list[Approach], cfg: PipelineConfig) -> float:
    """Big enough to clear the widest arm crossing the junction."""
    half = 0.0
    for a in approaches:
        strips = a.incoming + a.outgoing
        if not strips:
            continue
        w = sum(s.width for s in strips)
        half = max(half, 0.5 * w)
    r = half * 1.25 + cfg.intersection_margin + cluster.radius
    return float(np.clip(r, cfg.intersection_min_radius, cfg.intersection_max_radius))


def _plan_trims(approaches: list[Approach], center: np.ndarray, radius: float,
                groups: dict[str, list[LaneStrip]], trims: dict[str, list[int]]) -> None:
    """Record how far back each lane group must be cut to clear the junction.

    A whole carriageway is cut at one station so that its shared boundaries stay
    aligned with its centrelines; the station chosen is the most conservative
    over the lanes of the group, i.e. every lane of the group ends outside the
    junction circle.
    """
    for a in approaches:
        for s in a.incoming:
            d = np.hypot(*(s.center - center).T)
            keep = np.flatnonzero(d > radius)
            cut = int(keep[-1]) + 1 if len(keep) >= 2 else 2
            trims[s.group][1] = min(trims[s.group][1], cut)
            s.trimmed_end = True
        for s in a.outgoing:
            d = np.hypot(*(s.center - center).T)
            keep = np.flatnonzero(d > radius)
            cut = int(keep[0]) if len(keep) >= 2 else max(0, len(s.center) - 2)
            trims[s.group][0] = max(trims[s.group][0], cut)
            s.trimmed_start = True


def _update_headings(approaches: list[Approach], center: np.ndarray) -> None:
    for a in approaches:
        if a.incoming:
            _, h = a.incoming[0].exit_pose()
            a.heading_in = h
        if a.outgoing:
            _, h = a.outgoing[0].entry_pose()
            a.heading_out = h
        if not a.incoming and a.outgoing:
            a.heading_in = a.heading_out
        if not a.outgoing and a.incoming:
            a.heading_out = a.heading_in


def _turn_spec(a: Approach):
    return osm_tags.turn_lanes(a.tags)


def _manoeuvre(h_in: float, h_out: float, cfg: PipelineConfig) -> str | None:
    d = math.degrees(angle_diff(h_out, h_in))
    if abs(d) <= cfg.turn_through_deg:
        return "through"
    if abs(d) >= cfg.turn_max_deg:
        return "u_turn" if cfg.allow_u_turn else None
    return "left" if d > 0 else "right"


def _allowed_sets(n: int, targets: set[str], spec) -> list[set[str]]:
    """Which manoeuvres each incoming lane may make (lanes ordered left->right).

    With ``turn:lanes`` present this is read off the map.  Without it -- the
    normal case -- we apply the near-universal convention: the leftmost lane
    turns left, the rightmost turns right, everything goes through.
    """
    if spec is not None:
        lanes = list(spec.value)
        if len(lanes) == n:
            norm = []
            for toks in lanes:
                s = set()
                for t in toks:
                    if "left" in t:
                        s.add("left")
                    elif "right" in t:
                        s.add("right")
                    elif t in {"through", "merge_to_left", "merge_to_right"}:
                        s.add("through")
                    elif t == "reverse":
                        s.add("u_turn")
                norm.append(s or {"through"})
            return norm

    if n <= 1:
        return [set(targets)]
    out = [{"through"} for _ in range(n)]
    if "left" in targets:
        out[0].add("left")
    if "right" in targets:
        out[-1].add("right")
    if "through" not in targets:
        for s in out:
            s.discard("through")
        if "left" in targets:
            for s in out:
                s.add("left")
        if "right" in targets:
            for s in out:
                s.add("right")
    return out


def _pair(src: list[LaneStrip], dst: list[LaneStrip], manoeuvre: str
          ) -> list[tuple[LaneStrip, LaneStrip]]:
    """Match incoming to outgoing lanes, preserving lane order.

    Left turns keep left-alignment, right turns and through movements keep
    right-alignment -- the convention that makes turns not cross each other.
    """
    if not src or not dst:
        return []
    # Every incoming lane must get somewhere.  When the two sides have different
    # lane counts the surplus merges into (or diverges from) the outermost lane
    # rather than being silently dropped, which would strand it in the graph.
    n = max(len(src), len(dst))
    pairs, seen = [], set()
    for i in range(n):
        si, di = min(i, len(src) - 1), min(i, len(dst) - 1)
        if manoeuvre == "left":
            u, v = src[si], dst[di]
        else:
            u, v = src[len(src) - 1 - si], dst[len(dst) - 1 - di]
        if (u.id, v.id) in seen:
            continue
        seen.add((u.id, v.id))
        pairs.append((u, v))
    return pairs


def _connect(approaches: list[Approach], inter: Intersection, cfg: PipelineConfig):
    turns: list[LaneStrip] = []
    conns: list[tuple[str, str]] = []
    notes: list[str] = []

    tagged = 0
    for a in approaches:
        if not a.incoming:
            continue
        spec = _turn_spec(a)
        tagged += spec is not None

        targets: dict[str, list[Approach]] = {}
        for b in approaches:
            if b is a or not b.outgoing:
                continue
            if b.edge_id == a.edge_id and b.node_id == a.node_id:
                continue
            m = _manoeuvre(a.heading_in, b.heading_out, cfg)
            if m is None:
                continue
            targets.setdefault(m, []).append(b)

        if not targets:
            notes.append(f"approach {a.key}: no reachable exit")
            continue

        allowed = _allowed_sets(len(a.incoming), set(targets), spec)
        for m, bs in targets.items():
            if len(bs) > 1:
                notes.append(f"approach {a.key}: {len(bs)} candidate '{m}' exits, "
                             "picked the best-aligned one")
                bs = [min(bs, key=lambda b: abs(angle_diff(b.heading_out, a.heading_in)))]
            b = bs[0]
            src = [s for s, al in zip(a.incoming, allowed) if m in al]
            for u, v in _pair(src, b.outgoing, m):
                strip = _turn_strip(u, v, m, inter, spec is not None, cfg)
                if strip is None:
                    continue
                turns.append(strip)
                conns.append((u.id, strip.id))
                conns.append((strip.id, v.id))

    if tagged == 0:
        notes.append("no turn:lanes tags on any approach - connectivity is inferred "
                     "from the default manoeuvre convention")
    if len(approaches) > 4:
        notes.append(f"{len(approaches)} approaches: unusual junction, verify manually")
    return turns, conns, notes


def _turn_strip(u: LaneStrip, v: LaneStrip, manoeuvre: str, inter: Intersection,
                tagged: bool, cfg: PipelineConfig) -> LaneStrip | None:
    p0, h0 = u.exit_pose()
    p1, h1 = v.entry_pose()
    d = float(np.hypot(*(p1 - p0)))
    if d < 0.5:
        return None

    n = int(np.clip(round(d / 1.5), 6, 60))
    center = hermite(p0, h0, p1, h1, n=n)
    center = resample_polyline(center, 1.0)
    if len(center) < 2:
        return None

    # Width tapers from the entry lane to the exit lane.
    t = np.linspace(0, 1, len(center))
    half = 0.5 * (u.width * (1 - t) + v.width * t)
    left_pts = offset_polyline(center, half)
    right_pts = offset_polyline(center, -half)

    # Stitch the ends onto the neighbours so the exported lanelets literally
    # share their boundary end points (that is what makes them routable).
    left_pts[0] = u.left.points[-1]
    right_pts[0] = u.right.points[-1]
    left_pts[-1] = v.left.points[0]
    right_pts[-1] = v.right.points[0]

    conf = 0.30 + (0.22 if tagged else 0.0) + (0.08 if manoeuvre == "through" else 0.0)
    conf = min(conf, 0.75) * min(1.0, 0.6 + 0.4 * min(u.confidence, v.confidence) / 0.9)

    sid = f"{inter.id}_{u.id}__{v.id}"
    return LaneStrip(
        id=sid, segment_id=inter.id, group=sid, center=center,
        left=StripBoundary(left_pts, MarkingType.VIRTUAL, Source.INFERRED,
                           key=f"{sid}_L"),
        right=StripBoundary(right_pts, MarkingType.VIRTUAL, Source.INFERRED,
                            key=f"{sid}_R"),
        start_node=u.end_node, end_node=v.start_node,
        index_from_left=u.index_from_left,
        width=float(0.5 * (u.width + v.width)),
        speed_limit_kph=min(x for x in (u.speed_limit_kph, v.speed_limit_kph) if x) if
        (u.speed_limit_kph or v.speed_limit_kph) else None,
        one_way=True, confidence=float(conf),
        provenance=Provenance(Source.INFERRED, {
            "manoeuvre": manoeuvre,
            "from": u.id,
            "to": v.id,
            "turn_lanes_tag": tagged,
            "rule": "turn:lanes" if tagged else "default manoeuvre convention",
        }),
        flags=["intersection_topology_inferred"] + ([] if tagged else ["turn_lanes_tag_missing"]),
        detail={"intersection": inter.id, "manoeuvre": manoeuvre},
    )


def _intersection_confidence(approaches, turns, notes) -> float:
    if not turns:
        return 0.15
    c = float(np.mean([t.confidence for t in turns]))
    if len(approaches) > 4:
        c *= 0.8
    if any("no reachable exit" in n for n in notes):
        c *= 0.9
    return float(np.clip(c, 0.05, 0.9))
