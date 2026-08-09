"""Assemble the lane graph: segments + junctions + through-connections."""

from __future__ import annotations

import logging
import math

import numpy as np

from ..config import PipelineConfig
from ..geo import AOI, LocalFrame, angle_diff
from ..observation.base import Evidence
from ..prior.road_graph import RoadPrior
from ..types import Boundary, Lane, LaneGraph, Provenance, Source
from .intersection import resolve_intersections
from .segments import SegmentResult, build_all_segments
from .strips import LaneStrip

log = logging.getLogger(__name__)


def build_lane_graph(prior: RoadPrior, evidence: Evidence, aoi: AOI, frame: LocalFrame,
                     cfg: PipelineConfig) -> tuple[LaneGraph, dict[str, SegmentResult]]:
    segments = build_all_segments(prior, evidence, cfg)
    intersections, turn_strips, connections = resolve_intersections(prior, segments, cfg)

    road_strips = [s for r in segments.values() for s in r.strips]
    all_strips = road_strips + turn_strips
    by_id = {s.id: s for s in all_strips}

    simple = _connect_simple_nodes(prior, road_strips, cfg)
    connections = connections + simple

    graph = LaneGraph(aoi=aoi, frame=frame)
    graph.intersections = intersections
    _emit(graph, all_strips, turn_ids={s.id for s in turn_strips})

    gaps = 0
    for u, v in connections:
        if u in graph.lanes and v in graph.lanes:
            graph.connect(u, v)
            gaps += _stitch(by_id[u], by_id[v], cfg)

    graph.meta.update({
        "prior": prior.stats(),
        "observation": evidence.detail | {"backend": evidence.backend},
        "config": cfg.to_dict(),
        "attribution": {"prior": prior.attribution},
        "connections": len(connections),
        "stitch_gaps": gaps,
    })
    log.info("lane graph: %s", graph.stats())
    return graph, segments


# --------------------------------------------------------------------------- #


def _emit(graph: LaneGraph, strips: list[LaneStrip], turn_ids: set[str]) -> None:
    """Strips -> Lane / Boundary objects, sharing boundaries where they are shared."""
    bmap: dict[int, str] = {}

    def boundary_id(sb) -> str:
        if id(sb) in bmap:
            return bmap[id(sb)]
        bid = f"b{len(graph.boundaries)}"
        graph.add_boundary(Boundary(
            id=bid, points=np.asarray(sb.points, dtype=float), marking=sb.marking,
            confidence=_boundary_confidence(sb),
            provenance=Provenance(sb.source, {
                "coverage": round(sb.coverage, 3),
                "strength": round(sb.strength, 3),
                **{k: v for k, v in sb.detail.items() if k != "v0"},
            }),
        ))
        bmap[id(sb)] = bid
        return bid

    for s in strips:
        if len(s.center) < 2:
            continue
        graph.add_lane(Lane(
            id=s.id, left_id=boundary_id(s.left), right_id=boundary_id(s.right),
            centerline=np.asarray(s.center, dtype=float), segment_id=s.segment_id,
            kind="turn" if s.id in turn_ids else "road",
            index_from_left=s.index_from_left, width=float(s.width),
            speed_limit_kph=s.speed_limit_kph, one_way=s.one_way,
            confidence=float(s.confidence), provenance=s.provenance,
            turn_direction=s.detail.get("manoeuvre"),
            attributes={"flags": list(s.flags), **{k: v for k, v in s.detail.items()
                                                   if k not in {"manoeuvre"}}},
        ))


def _boundary_confidence(sb) -> float:
    from ..types import MarkingType

    if sb.marking is MarkingType.VIRTUAL:
        return 0.25
    if sb.marking is MarkingType.ROAD_EDGE:
        return float(np.clip(0.35 + 0.4 * sb.coverage, 0.2, 0.8))
    return float(np.clip(0.45 + 0.5 * sb.coverage, 0.3, 0.95))


def _connect_simple_nodes(prior: RoadPrior, strips: list[LaneStrip],
                          cfg: PipelineConfig) -> list[tuple[str, str]]:
    """Join lanes across nodes that are not junctions (a way split, a curve)."""
    clustered = {n for c in prior.clusters.values() for n in c.node_ids}
    arriving: dict[str, list[LaneStrip]] = {}
    leaving: dict[str, list[LaneStrip]] = {}
    for s in strips:
        arriving.setdefault(s.end_node, []).append(s)
        leaving.setdefault(s.start_node, []).append(s)

    out: list[tuple[str, str]] = []
    for nid, node in prior.nodes.items():
        if nid in clustered:
            continue
        ins = arriving.get(nid, [])
        outs = leaving.get(nid, [])
        if not ins or not outs:
            continue
        for u in ins:
            # never turn back onto the edge we came from
            cands = [v for v in outs if v.segment_id != u.segment_id]
            if not cands:
                continue
            _, h_in = u.exit_pose()
            scored = []
            for v in cands:
                p, h_out = v.entry_pose()
                dh = abs(math.degrees(angle_diff(h_out, h_in)))
                d = float(np.hypot(*(p - u.exit_pose()[0])))
                if dh > 75.0:
                    continue
                scored.append((dh * 0.15 + d, v))
            if not scored:
                continue
            scored.sort(key=lambda t: t[0])
            out.append((u.id, scored[0][1].id))
    return out


def _stitch(u: LaneStrip, v: LaneStrip, cfg: PipelineConfig) -> int:
    """Make ``u``'s end and ``v``'s start literally the same points.

    Lanelet2 derives successor/predecessor relations from *shared boundary end
    points*, so this is what turns a pile of lanelets into a routable map.
    Returns 1 if the two ends were too far apart to be joined.
    """
    tol = 4.0
    gap = 0
    for a, b in ((u.left, v.left), (u.right, v.right)):
        if len(a.points) < 2 or len(b.points) < 2:
            continue
        d = float(np.hypot(*(b.points[0] - a.points[-1])))
        if d > tol:
            gap = 1
            continue
        m = 0.5 * (a.points[-1] + b.points[0])
        a.points[-1] = m
        b.points[0] = m
    # keep centrelines consistent with the boundaries we just moved
    if len(u.center) >= 2:
        u.center[-1] = 0.5 * (u.left.points[-1] + u.right.points[-1])
    if len(v.center) >= 2:
        v.center[0] = 0.5 * (v.left.points[0] + v.right.points[0])
    return gap
