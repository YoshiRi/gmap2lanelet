"""Failure detection.

The PoC is judged on *failure modes*, not on absolute accuracy, so this module
is a first-class part of the pipeline rather than an afterthought.  Each
detector turns an internal signal that something went wrong into a georeferenced
review item a human can be sent to.

The catalogue below maps one-to-one onto the questions the brief asks:

===========================  =================================================
code                         question it answers
===========================  =================================================
``prior_geometry_shift``     where does the map's geometry disagree with the image
``lane_count_conflict``      where do we get the number of lanes wrong
``markings_not_observed``    where are lane markings invisible
``corridor_not_found``       where does the map claim a road the image denies
``pavement_without_prior``   where does the image show road the map omits
``corridor_width_clamped``   where does pavement bleed (parking lots, forecourts)
``intersection_unresolved``  where can intersection topology not be decided
``lane_disconnected``        where does the lane graph fall apart
``implausible_geometry``     where is the output self-evidently wrong
===========================  =================================================
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from ..config import PipelineConfig
from ..observation.base import Evidence
from ..prior import osm_tags
from ..prior.road_graph import RoadPrior
from ..types import LaneGraph, MarkingType, ReviewItem, Source

log = logging.getLogger(__name__)

SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}

CATALOGUE: dict[str, str] = {
    "prior_geometry_shift": "OSM centreline is laterally displaced from the observed pavement",
    "lane_count_conflict": "lane count from markings disagrees with the lane count from OSM",
    "lane_count_unverified": "no marking evidence to confirm the OSM lane count",
    "markings_not_observed": "no lane markings detectable on a road class that should have them",
    "corridor_not_found": "OSM claims a road but no drivable surface was observed",
    "pavement_without_prior": "large paved area with no road in OSM (parking, forecourt, new road)",
    "corridor_width_clamped": "observed pavement is far wider than the road class allows",
    "intersection_unresolved": "intersection connectivity could not be determined from public data",
    "lane_disconnected": "lane has neither predecessor nor successor",
    "implausible_geometry": "generated lane geometry is self-evidently wrong",
    "low_confidence": "element confidence below the review threshold",
}


@dataclass
class FailureReport:
    items: list[ReviewItem]
    counts: dict[str, int]
    stats: dict

    def by_kind(self) -> dict[str, list[ReviewItem]]:
        out: dict[str, list[ReviewItem]] = {}
        for it in self.items:
            out.setdefault(it.kind, []).append(it)
        return out


def analyse(graph: LaneGraph, prior: RoadPrior, segments: dict, evidence: Evidence,
            cfg: PipelineConfig, validation: dict | None = None) -> FailureReport:
    items: list[ReviewItem] = []
    n = [0]

    def add(_kind: str, _severity: str, _message: str, _position, _elements=None, **detail):
        n[0] += 1
        items.append(ReviewItem(f"r{n[0]:04d}", _kind, _severity, _message,
                                np.asarray(_position, dtype=float), _elements or [], detail))

    _segment_failures(add, segments, cfg)
    _pavement_without_prior(add, graph, prior, evidence, cfg)
    _intersection_failures(add, graph)
    _lane_failures(add, graph, cfg)

    items.sort(key=lambda it: (SEVERITY_ORDER.get(it.severity, 3), it.kind))
    counts: dict[str, int] = {}
    for it in items:
        counts[it.kind] = counts.get(it.kind, 0) + 1

    stats = _stats(graph, segments, counts, validation)
    log.info("failure analysis: %s review items %s", len(items), counts)
    return FailureReport(items, counts, stats)


# --------------------------------------------------------------------------- #


def _mid(pts: np.ndarray) -> np.ndarray:
    return np.asarray(pts[len(pts) // 2], dtype=float)


def _segment_failures(add, segments: dict, cfg: PipelineConfig) -> None:
    for eid, res in segments.items():
        edge = res.edge
        mid = _mid(edge.points)
        marked_class = edge.highway not in osm_tags.UNMARKED_CLASSES

        if "corridor_not_found" in res.flags:
            add("corridor_not_found", "high",
                f"{edge.highway} way {edge.way_id}: OSM has a road here but no drivable "
                f"surface was observed in the imagery", mid, [eid],
                highway=edge.highway, length_m=round(edge.length, 1))
            continue

        if res.lateral_shift > 2.0:
            sev = "high" if res.lateral_shift > 5.0 else "medium"
            add("prior_geometry_shift", sev,
                f"prior centreline is {res.lateral_shift:.1f} m "
                f"(max {res.max_lateral_shift:.1f} m) off the observed pavement centre",
                mid, [eid], median_shift_m=round(res.lateral_shift, 2),
                max_shift_m=round(res.max_lateral_shift, 2), highway=edge.highway)

        for sol in res.solutions:
            flags = set(sol.flags)
            if "lane_count_conflict" in flags:
                add("lane_count_conflict", "high" if abs((sol.n_image or 0) - sol.n_osm) > 1
                    else "medium",
                    f"markings imply {sol.detail['n_geometry']} lanes, OSM says {sol.n_osm} "
                    f"(kept {sol.n_lanes}, source={sol.count_source.value})",
                    mid, [eid], n_osm=sol.n_osm, n_image=sol.n_image,
                    n_geometry=sol.detail["n_geometry"], chosen=sol.n_lanes,
                    corridor_width_m=sol.detail["corridor_width"])
            if "no_lane_markings_observed" in flags and marked_class:
                add("markings_not_observed", "medium",
                    f"no lane markings detected on a {edge.highway} road; lane division "
                    f"is an even split of the {sol.detail['corridor_width']:.1f} m corridor",
                    mid, [eid], highway=edge.highway,
                    corridor_width_m=sol.detail["corridor_width"], n_lanes=sol.n_lanes)
            elif "lane_count_unverified" in flags:
                add("lane_count_unverified", "low",
                    f"lane count {sol.n_lanes} taken from OSM with no imagery confirmation",
                    mid, [eid], n_lanes=sol.n_lanes, highway=edge.highway)
            if "corridor_width_clamped" in flags:
                add("corridor_width_clamped", "medium",
                    f"observed pavement much wider than a {edge.highway} carriageway; "
                    f"corridor was capped (adjacent parking or forecourt is likely)",
                    mid, [eid], corridor_width_m=sol.detail["corridor_width"],
                    highway=edge.highway)
            if "corridor_partially_unobserved" in flags:
                add("markings_not_observed", "low",
                    f"pavement observed on only {100 * sol.detail['corridor_coverage']:.0f}% "
                    f"of the segment (occlusion by trees, shadow or vehicles)",
                    mid, [eid], coverage=sol.detail["corridor_coverage"])
            if "centre_divider_unobserved" in flags:
                add("lane_count_conflict", "medium",
                    "no centre line observed on a two-way road; the direction split is "
                    "an assumption", mid, [eid], n_lanes=sol.n_lanes)


def _pavement_without_prior(add, graph: LaneGraph, prior: RoadPrior, evidence: Evidence,
                            cfg: PipelineConfig, min_area_m2: float = 1500.0) -> None:
    """Paved areas the map never mentions.

    These are usually parking lots -- which is the *right* answer, and evidence
    that the topology prior is doing useful work by suppressing them -- but the
    same signal catches a genuinely missing road, so it is worth reporting.
    """
    import cv2
    from scipy import ndimage as ndi

    road = evidence.road_prob
    step = max(1, int(round(1.0 / road.gsd)))              # ~1 m working grid
    mask = (road.data[::step, ::step] > 0.6) & (evidence.vegetation.data[::step, ::step] < 0.5)
    h, w = mask.shape
    cell = road.gsd * step

    covered = np.zeros((h, w), dtype=np.uint8)
    for ln in graph.lanes.values():
        lb = graph.boundaries.get(ln.left_id)
        rb = graph.boundaries.get(ln.right_id)
        if lb is None or rb is None or len(lb.points) < 2:
            continue
        poly = np.vstack([lb.points, rb.points[::-1]])
        c, r = road.world_to_pixel(poly[:, 0], poly[:, 1])
        pts = np.column_stack([c / step, r / step]).astype(np.int32)
        cv2.fillPoly(covered, [pts], 1)
    # be generous: anything within ~6 m of a generated lane counts as covered
    covered = ndi.binary_dilation(covered.astype(bool), np.ones((13, 13)))

    orphan = mask & ~covered
    orphan = ndi.binary_opening(orphan, np.ones((5, 5)))
    lab, n = ndi.label(orphan)
    if not n:
        return
    sizes = ndi.sum(orphan, lab, index=np.arange(1, n + 1)) * cell * cell
    for i in np.argsort(-sizes)[:12]:
        area = float(sizes[i])
        if area < min_area_m2:
            break
        cy, cx = ndi.center_of_mass(lab == i + 1)
        x, y = road.pixel_to_world(cx * step, cy * step)
        add("pavement_without_prior", "low" if area < 6000 else "medium",
            f"{area:,.0f} m2 of pavement with no road in the prior "
            f"(parking lot, forecourt, or a road missing from the map)",
            [float(x), float(y)], [], area_m2=round(area, 1))


def _intersection_failures(add, graph: LaneGraph) -> None:
    for iid, it in graph.intersections.items():
        untagged = not it.provenance.detail.get("turn_lanes_tagged", False)
        sev = "high" if (untagged and it.approach_count > 4) else "medium"
        msgs = list(it.notes)
        if untagged:
            msgs.insert(0, "no turn:lanes information in the prior")
        if not it.turn_lane_ids:
            add("intersection_unresolved", "high",
                f"junction with {it.approach_count} approaches produced no turn lanes",
                it.center, [iid], approaches=it.approach_count, notes=msgs)
            continue
        add("intersection_unresolved", sev,
            f"connectivity for {it.approach_count} approaches ({len(it.turn_lane_ids)} turn "
            f"lanes) was inferred, not observed: " + "; ".join(msgs[:3]),
            it.center, [iid] + it.turn_lane_ids[:8], approaches=it.approach_count,
            turn_lanes=len(it.turn_lane_ids), confidence=round(it.confidence, 2), notes=msgs)


def _lane_failures(add, graph: LaneGraph, cfg: PipelineConfig) -> None:
    for lid, ln in graph.lanes.items():
        if not ln.predecessors and not ln.successors:
            add("lane_disconnected", "medium",
                f"lane {lid} has no predecessor and no successor", _mid(ln.centerline), [lid],
                lane_kind=ln.kind, segment=ln.segment_id)
        if ln.width < 2.0 or ln.width > 6.0:
            add("implausible_geometry", "medium",
                f"lane width {ln.width:.1f} m is outside any plausible range",
                _mid(ln.centerline), [lid], width_m=round(ln.width, 2))
        elif _self_intersects(ln.centerline):
            add("implausible_geometry", "high",
                f"lane {lid} centreline crosses itself", _mid(ln.centerline), [lid])
        if ln.confidence < cfg.review_confidence_threshold and ln.kind == "road":
            add("low_confidence", "low",
                f"lane confidence {ln.confidence:.2f} below the review threshold",
                _mid(ln.centerline), [lid], confidence=round(ln.confidence, 2),
                flags=ln.attributes.get("flags", []))


def _self_intersects(pts: np.ndarray) -> bool:
    if len(pts) < 4:
        return False
    try:
        from shapely.geometry import LineString

        return not LineString(pts).is_simple
    except Exception:                                        # noqa: BLE001
        return False


def _by_group(segments: dict) -> dict:
    """Per-road-class results.

    Aggregating over every prior edge is misleading: in a typical US suburban
    tile most edges by count are parking aisles, where "lane markings" are
    parking-bay stripes and the OSM lane count is nominal.  What a reader
    actually wants to know is how well this works on roads.
    """
    groups: dict[str, dict] = {}
    for res in segments.values():
        g = osm_tags.road_group(res.edge.tags)
        d = groups.setdefault(g, {"carriageways": 0, "with_markings": 0, "conflicts": 0,
                                  "agree": 0, "km": 0.0, "shift": []})
        d["km"] += res.edge.length / 1000.0
        if res.corridors:
            d["shift"].append(res.lateral_shift)
        for sol in res.solutions:
            d["carriageways"] += 1
            seen = sol.detail["observed_boundaries"] > 0
            d["with_markings"] += seen
            if seen:
                d["conflicts"] += "lane_count_conflict" in sol.flags
                d["agree"] += "lane_count_conflict" not in sol.flags

    out = {}
    for g, d in sorted(groups.items()):
        n, m = d["carriageways"], d["with_markings"]
        out[g] = {
            "km": round(d["km"], 2),
            "carriageways": n,
            "markings_observed": round(m / n, 3) if n else 0.0,
            # of the carriageways where markings *were* observed, how often the
            # image-derived lane count matched OSM
            "lane_count_agreement_when_observed": round(d["agree"] / m, 3) if m else None,
            "prior_shift_median_m": round(float(np.median(d["shift"])), 2) if d["shift"] else 0.0,
        }
    return out


def _stats(graph: LaneGraph, segments: dict, counts: dict, validation: dict | None) -> dict:
    sols = [s for r in segments.values() for s in r.solutions]
    lanes = list(graph.lanes.values())
    road_lanes = [l for l in lanes if l.kind == "road"]

    def frac(pred, seq) -> float:
        seq = list(seq)
        return round(sum(1 for x in seq if pred(x)) / len(seq), 4) if seq else 0.0

    shifts = [r.lateral_shift for r in segments.values() if r.corridors]
    by_group = _by_group(segments)
    bnds = list(graph.boundaries.values())
    return {
        "segments": len(segments),
        "carriageways": len(sols),
        "lanes_total": len(lanes),
        "lanes_road": len(road_lanes),
        "lanes_turn": len(lanes) - len(road_lanes),
        "lane_count_source": {
            s.value: frac(lambda x, s=s: x.count_source is s, sols)
            for s in (Source.OSM, Source.IMAGE, Source.FUSED)
        },
        "carriageways_with_observed_markings":
            frac(lambda s: s.detail["observed_boundaries"] > 0, sols),
        "carriageways_with_lane_count_conflict":
            frac(lambda s: "lane_count_conflict" in s.flags, sols),
        "dual_carriageways_detected": sum(1 for s in sols if s.detail.get("dual_carriageway")),
        "boundaries_observed_fraction": frac(
            lambda b: b.marking in (MarkingType.SOLID, MarkingType.DASHED,
                                    MarkingType.DOUBLE_SOLID), bnds),
        "boundaries_virtual_fraction": frac(lambda b: b.marking is MarkingType.VIRTUAL, bnds),
        "prior_shift_median_m": round(float(np.median(shifts)), 2) if shifts else 0.0,
        "prior_shift_p90_m": round(float(np.percentile(shifts, 90)), 2) if shifts else 0.0,
        "mean_lane_confidence": round(float(np.mean([l.confidence for l in lanes])), 3)
        if lanes else 0.0,
        "by_road_group": by_group,
        "review_items": sum(counts.values()),
        "review_items_by_kind": counts,
        "lanelet2": (validation or {}).get("routing", {}),
    }
