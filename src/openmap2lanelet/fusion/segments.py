"""Turn each prior edge into lane strips."""

from __future__ import annotations

import logging

import numpy as np

from ..config import PipelineConfig
from ..observation.base import Evidence
from ..prior import osm_tags
from ..prior.road_graph import RoadEdge, RoadPrior
from ..types import MarkingType, Provenance, Source
from .corridor import extract_corridors
from .lanes import CarriagewaySolution, solve_lanes
from .profile import Profile, build_profile
from .strips import LaneStrip, StripBoundary

log = logging.getLogger(__name__)


class SegmentResult:
    """Everything produced for one prior edge (kept for reporting / debug)."""

    def __init__(self, edge: RoadEdge):
        self.edge = edge
        self.strips: list[LaneStrip] = []
        self.solutions: list[CarriagewaySolution] = []
        self.profiles: list[Profile] = []
        self.corridors: list = []
        self.flags: list[str] = []
        self.lateral_shift: float = 0.0
        self.max_lateral_shift: float = 0.0


def build_segment(edge: RoadEdge, evidence: Evidence, cfg: PipelineConfig) -> SegmentResult:
    """Corridor -> lane structure -> strips, for a single prior edge."""
    res = SegmentResult(edge)
    tags = edge.tags
    expected_w = osm_tags.expected_carriageway_width(tags)
    oneway = bool(osm_tags.is_oneway(tags).value)
    speed = osm_tags.speed_limit_kph(tags)

    half_width = max(cfg.profile_half_width, 0.5 * expected_w * 2.0 + cfg.max_lateral_shift)
    prof = build_profile(edge.points, evidence, half_width=half_width,
                         ds=cfg.station_step, du=cfg.offset_step)

    corridors = extract_corridors(
        prof, expected_width=expected_w, oneway=oneway,
        road_thr=cfg.road_threshold, max_shift=cfg.max_lateral_shift,
        bleed_factor=min(cfg.corridor_bleed_factor, osm_tags.bleed_factor(tags)),
    )

    for ci, cor in enumerate(corridors):
        if not cor.valid.any():
            res.flags.append("corridor_not_found")
            continue

        sol = solve_lanes(prof, cor, tags, drive_on_right=cfg.drive_on_right,
                          marking_thr=cfg.marking_threshold)
        res.solutions.append(sol)
        res.corridors.append(cor)
        res.profiles.append(prof)

        shift = float(np.nanmedian(np.abs(cor.center_offset[cor.valid])))
        res.lateral_shift = max(res.lateral_shift, shift)
        res.max_lateral_shift = max(
            res.max_lateral_shift,
            float(np.nanmax(np.abs(cor.center_offset[cor.valid]))) if cor.valid.any() else 0.0,
        )

        # Boundary polylines in world coordinates.  Neighbouring lanes share the
        # boundary *object* between them, which is what lets a whole
        # carriageway be trimmed consistently at a junction later.
        idx = np.arange(prof.n_stations)
        world = []
        for bi, b in enumerate(sol.boundaries):
            pts = _unfold(prof.to_world(idx, b.offsets))
            world.append(StripBoundary(pts, b.marking, b.source, b.coverage, b.strength,
                                       dict(b.detail), key=f"{edge.id}_c{ci}_b{bi}"))
        # One reversed copy per boundary, shared by all backward lanes.
        rev = [StripBoundary(b.points[::-1].copy(), b.marking, b.source, b.coverage,
                             b.strength, dict(b.detail), key=b.key + "r")
               for b in world]

        for slot in sol.lanes:
            lb, rb = world[slot.left_idx], world[slot.right_idx]
            start_node, end_node = edge.from_node, edge.to_node
            l_out, r_out = lb, rb
            if slot.forward:
                center = 0.5 * (lb.points + rb.points)
            else:
                # travelling the other way: order reverses and left/right swap
                l_out, r_out = rev[slot.right_idx], rev[slot.left_idx]
                center = 0.5 * (l_out.points + r_out.points)
                start_node, end_node = edge.to_node, edge.from_node

            sid = f"{edge.id}_c{ci}_l{slot.left_idx}{'f' if slot.forward else 'b'}"
            group = f"{edge.id}_c{ci}_{'f' if slot.forward else 'b'}"
            bidirectional = (not oneway) and sol.n_lanes == 1 and len(corridors) == 1
            res.strips.append(LaneStrip(
                id=sid, segment_id=edge.id, group=group, center=center,
                left=l_out, right=r_out,
                start_node=start_node, end_node=end_node,
                index_from_left=slot.left_idx, width=slot.width,
                speed_limit_kph=float(speed.value), one_way=not bidirectional,
                confidence=_lane_confidence(sol, slot, l_out, r_out),
                provenance=Provenance(sol.count_source, {
                    "lane_count": sol.n_lanes,
                    "lane_count_osm": sol.n_osm,
                    "lane_count_image": sol.n_image,
                    "speed_source": speed.source.value,
                    "highway": edge.highway,
                    "carriageway": cor.side,
                }),
                flags=list(sol.flags),
                detail=dict(sol.detail),
            ))

    if not res.strips and "corridor_not_found" not in res.flags:
        res.flags.append("no_lanes_generated")
    return res


def _unfold(pts: np.ndarray, iters: int = 4) -> np.ndarray:
    """Remove folds from an offset polyline, keeping one point per station.

    Where a boundary is offset further than the reference line's radius of
    curvature the offset curve doubles back on itself.  Folded vertices are
    replaced by a straight interpolation across the fold, which keeps the array
    length -- and therefore the correspondence between a lane's centreline and
    its two boundaries -- intact.
    """
    from scipy.ndimage import binary_dilation

    p = np.asarray(pts, dtype=float).copy()
    if len(p) < 4:
        return p
    idx = np.arange(len(p))
    for _ in range(iters):
        d = np.diff(p, axis=0)
        n = np.hypot(d[:, 0], d[:, 1])
        n[n < 1e-9] = 1e-9
        u = d / n[:, None]
        reversed_at = np.flatnonzero((u[:-1] * u[1:]).sum(axis=1) < 0.0) + 1
        if not len(reversed_at):
            break
        mask = np.zeros(len(p), dtype=bool)
        mask[reversed_at] = True
        mask = binary_dilation(mask, np.ones(3, dtype=bool))
        mask[0] = mask[-1] = False
        if mask.all() or not mask.any():
            break
        good = ~mask
        p[mask, 0] = np.interp(idx[mask], idx[good], p[good, 0])
        p[mask, 1] = np.interp(idx[mask], idx[good], p[good, 1])
    return p


def _lane_confidence(sol: CarriagewaySolution, slot, left: StripBoundary,
                     right: StripBoundary) -> float:
    """Lane confidence = count confidence, tempered by boundary evidence."""
    c = sol.confidence
    obs = sum(1 for b in (left, right) if b.source is Source.IMAGE
              and b.marking is not MarkingType.ROAD_EDGE)
    edges = sum(1 for b in (left, right) if b.marking is MarkingType.ROAD_EDGE)
    c += 0.06 * obs + 0.02 * edges
    if any(b.marking is MarkingType.VIRTUAL for b in (left, right)):
        c -= 0.12
    if not (2.4 <= slot.width <= 5.0):
        c -= 0.15
    return float(np.clip(c, 0.05, 0.95))


def build_all_segments(prior: RoadPrior, evidence: Evidence, cfg: PipelineConfig
                       ) -> dict[str, SegmentResult]:
    out: dict[str, SegmentResult] = {}
    internal = _internal_to_junction(prior)
    for i, (eid, edge) in enumerate(sorted(prior.edges.items())):
        if edge.length < cfg.min_segment_length:
            continue
        if edge.highway in cfg.skip_classes:
            continue
        if eid in internal:
            # Both ends sit inside one junction cluster: this is the inside of a
            # big intersection (the short link between the two carriageways of a
            # divided road, say).  The junction's own turn lanes cover that area,
            # so building road lanes here would only strand them.
            continue
        try:
            out[eid] = build_segment(edge, evidence, cfg)
        except Exception as exc:                       # noqa: BLE001 - one bad edge must not kill the run
            log.exception("segment %s failed: %s", eid, exc)
        if (i + 1) % 25 == 0:
            log.info("segments %s/%s", i + 1, len(prior.edges))
    n = sum(len(r.strips) for r in out.values())
    log.info("built %s lane strips over %s segments (%s absorbed into junctions)",
             n, len(out), len(internal))
    return out


def _internal_to_junction(prior: RoadPrior) -> set[str]:
    """Edges whose two ends belong to the same junction cluster."""
    cluster_of = {n: cid for cid, c in prior.clusters.items() for n in c.node_ids}
    return {eid for eid, e in prior.edges.items()
            if cluster_of.get(e.from_node) is not None
            and cluster_of.get(e.from_node) == cluster_of.get(e.to_node)}
