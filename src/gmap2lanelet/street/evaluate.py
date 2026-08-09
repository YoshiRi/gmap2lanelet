"""Evaluation against the Argoverse 2 HD map held back as ground truth.

The experiment is set up so that the *inputs* only ever carry what public data
carries.  The AV2 map is degraded to OSM's information content (centrelines,
a road class, no lane counts, no turn tags) and used as the topology prior; the
full lane-level map is never shown to the pipeline and is opened only here.

What can honestly be scored, and what cannot:

============================  ==========================================
quantity                      ground truth
============================  ==========================================
lane geometry                 AV2 lane centrelines
lane count                    AV2 carriageway membership
turn permissions per lane     AV2 intersection connectivity
stop line position            AV2 pedestrian-crossing polygons (proxy)
traffic light position        **none** -- AV2 has no signal annotations
============================  ==========================================

The last row is not a gap in the evaluation, it is a finding: the open datasets
that ship posed street imagery generally do not annotate signal heads, so a
traffic-light position recovered this way can be checked for *self-consistency*
(reprojection error, multi-view agreement, physical height) but not against a
survey.  Reporting a made-up accuracy figure there would be worse than
reporting none, so the numbers below stop where the truth stops.
"""

from __future__ import annotations

import logging
import math

import numpy as np

from ..geo import angle_diff, polyline_length, resample_polyline
from ..types import LaneGraph
from .sources.av2_map import AV2GroundTruth
from .types import LandmarkKind, SemanticLayer

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# geometry and lane count
# --------------------------------------------------------------------------- #


def lane_geometry_error(graph: LaneGraph, gt: AV2GroundTruth, *,
                        max_match_m: float = 6.0, step: float = 2.0) -> dict:
    """Lateral distance from each generated centreline to the nearest true lane.

    Matching is per *sample*, not per lane: a generated lane that drifts off one
    true lane and onto its neighbour should be penalised for the drift, not
    rewarded for still being near something.  Samples with no true lane inside
    ``max_match_m`` are reported separately as unmatched rather than silently
    dropped, because they are where the map claims a lane that is not there.
    """
    truth = [l.center[:, :2] for l in gt.lanes.values() if not l.is_intersection]
    if not truth:
        return {}
    pool = np.vstack([resample_polyline(t, 1.0) for t in truth if len(t) >= 2])

    errs, unmatched, total = [], 0, 0
    per_lane = {}
    for ln in graph.lanes.values():
        if ln.kind != "road" or len(ln.centerline) < 2:
            continue
        pts = resample_polyline(np.asarray(ln.centerline)[:, :2], step)
        d = np.min(np.linalg.norm(pool[None, :, :] - pts[:, None, :], axis=2), axis=1)
        total += len(d)
        ok = d <= max_match_m
        unmatched += int((~ok).sum())
        if ok.any():
            errs.extend(d[ok].tolist())
            per_lane[ln.id] = round(float(np.mean(d[ok])), 3)

    if not errs:
        return {"samples": total, "unmatched_fraction": 1.0}
    e = np.array(errs)
    return {
        "samples": total,
        "mean_lateral_error_m": round(float(e.mean()), 3),
        "median_lateral_error_m": round(float(np.median(e)), 3),
        "p90_lateral_error_m": round(float(np.percentile(e, 90)), 3),
        "within_0_5m": round(float((e <= 0.5).mean()), 3),
        "within_1m": round(float((e <= 1.0).mean()), 3),
        "unmatched_fraction": round(unmatched / max(total, 1), 3),
        "per_lane_mean_m": per_lane,
    }


def lane_count_error(graph: LaneGraph, gt: AV2GroundTruth) -> dict:
    """Per generated carriageway: how many lanes were produced vs how many exist."""
    groups: dict[str, list] = {}
    for ln in graph.lanes.values():
        if ln.kind != "road":
            continue
        key = f"{ln.segment_id}|{ln.attributes.get('carriageway', 0)}"
        groups.setdefault(key, []).append(ln)

    rows, diffs = [], []
    for key, lanes in groups.items():
        mid = lanes[len(lanes) // 2].centerline
        true = gt.lane_count_at(mid[len(mid) // 2])
        if true is None:
            continue
        rows.append({"group": key, "estimated": len(lanes), "true": true})
        diffs.append(len(lanes) - true)
    if not diffs:
        return {}
    d = np.array(diffs)
    return {
        "carriageways_scored": len(d),
        "exact": round(float((d == 0).mean()), 3),
        "within_1": round(float((np.abs(d) <= 1).mean()), 3),
        "mean_signed_error": round(float(d.mean()), 3),
        "mean_absolute_error": round(float(np.abs(d).mean()), 3),
        "rows": rows,
    }


# --------------------------------------------------------------------------- #
# turn semantics -- the question street imagery is supposed to answer
# --------------------------------------------------------------------------- #


def _true_manoeuvres(gt: AV2GroundTruth, lane_id: int, through_deg: float = 30.0,
                     u_turn_deg: float = 150.0) -> set[str]:
    """Manoeuvres actually available from one true lane, via junction interiors."""
    out: set[str] = set()
    src = gt.lanes.get(lane_id)
    if src is None:
        return out
    h_in = src.heading
    for s in src.successors:
        nxt = gt.lanes.get(int(s))
        if nxt is None:
            continue
        if not nxt.is_intersection:
            out.add("through")
            continue
        # the manoeuvre is the heading change across the interior lane
        d = math.degrees(angle_diff(nxt.heading, h_in))
        # a short interior stub understates the turn; use its exit where possible
        for e in nxt.successors:
            ex = gt.lanes.get(int(e))
            if ex is not None and not ex.is_intersection:
                d = math.degrees(angle_diff(ex.heading, h_in))
                break
        if abs(d) <= through_deg:
            out.add("through")
        elif abs(d) >= u_turn_deg:
            out.add("u_turn")
        else:
            out.add("left" if d > 0 else "right")
    return out


def _nearest_true_lane(gt: AV2GroundTruth, pts: np.ndarray, heading_rad: float,
                       max_m: float = 4.0, max_heading_deg: float = 45.0):
    """Match a generated lane to the true lane it represents (position + heading)."""
    tail = np.asarray(pts)[-max(2, len(pts) // 3):, :2]
    best, best_d = None, max_m
    for lid, l in gt.lanes.items():
        if l.is_intersection:
            continue
        if abs(angle_diff(l.heading, heading_rad)) > math.radians(max_heading_deg):
            continue
        c = l.center[:, :2]
        d = float(np.mean(np.min(np.linalg.norm(c[None, :, :] - tail[:, None, :], axis=2),
                                 axis=1)))
        if d < best_d:
            best_d, best = d, lid
    return best, best_d


def turn_semantics_error(graph: LaneGraph, layer: SemanticLayer,
                         gt: AV2GroundTruth) -> dict:
    """Score the inferred turn convention against the paint, and both against truth.

    Three sets are compared per lane:

    * **inferred** -- what the intersection model assumed from lane order alone
      (the phase-1 answer, using no observation at all);
    * **observed** -- what the painted arrow says (the phase-2 answer);
    * **true**     -- what the HD map's connectivity actually allows.

    The number that matters is whether the arrow moves the answer towards truth.
    """
    from ..geo import heading as poly_heading

    rows = []
    for a in layer.arrows.values():
        ln = graph.lanes.get(a.lane_id)
        if ln is None:
            continue
        h = poly_heading(ln.centerline, at_start=False)
        tid, d = _nearest_true_lane(gt, ln.centerline, h)
        if tid is None:
            continue
        true = _true_manoeuvres(gt, tid)
        if not true:
            continue
        observed = set(a.manoeuvres)
        inferred = set(ln.attributes.get("turn_manoeuvres_inferred") or [])
        union = observed | inferred
        rows.append({
            "lane": ln.id, "true_lane": tid, "match_m": round(d, 2),
            "true": sorted(true), "observed": sorted(observed),
            "inferred": sorted(inferred), "union": sorted(union),
            "observed_exact": observed == true,
            "inferred_exact": bool(inferred) and inferred == true,
            "union_exact": union == true,
            "observed_iou": round(_iou(observed, true), 3),
            "inferred_iou": round(_iou(inferred, true), 3),
            "union_iou": round(_iou(union, true), 3),
            # precision: does the paint ever claim a manoeuvre that is illegal?
            "observed_precision": round(len(observed & true) / max(len(observed), 1), 3),
            "observed_recall": round(len(observed & true) / max(len(true), 1), 3),
            "inferred_precision": round(len(inferred & true) / max(len(inferred), 1), 3)
            if inferred else None,
            "inferred_recall": round(len(inferred & true) / max(len(true), 1), 3),
        })

    if not rows:
        return {"lanes_scored": 0}

    def m(k):
        vals = [r[k] for r in rows if r[k] is not None]
        return round(float(np.mean(vals)), 3) if vals else None

    return {
        "lanes_scored": len(rows),
        "observed_exact": m("observed_exact"),
        "inferred_exact": m("inferred_exact"),
        "union_exact": m("union_exact"),
        "observed_mean_iou": m("observed_iou"),
        "inferred_mean_iou": m("inferred_iou"),
        "union_mean_iou": m("union_iou"),
        "observed_precision": m("observed_precision"),
        "observed_recall": m("observed_recall"),
        "inferred_precision": m("inferred_precision"),
        "inferred_recall": m("inferred_recall"),
        "rows": rows,
    }


def _iou(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(len(a | b), 1)


# --------------------------------------------------------------------------- #
# stop lines
# --------------------------------------------------------------------------- #


def stop_line_error(layer: SemanticLayer, gt: AV2GroundTruth, *,
                    max_m: float = 12.0) -> dict:
    """Distance from each observed stop bar to the nearest crosswalk.

    AV2 annotates pedestrian crossings but not stop bars, so the crossing is the
    only available anchor.  A correctly placed stop line sits a metre or two
    *upstream* of the crossing it precedes, so the score to read is not "error"
    but whether the offset is small, positive and consistent.
    """
    if not gt.crossings:
        return {"crossings": 0}
    cross = np.vstack([resample_polyline(c[:, :2], 0.5) for c in gt.crossings
                       if len(c) >= 2])

    rows, ds = [], []
    for sl in layer.stop_lines.values():
        if not sl.observed:
            continue
        mid = np.asarray(sl.points).mean(axis=0)
        d = float(np.min(np.linalg.norm(cross - mid[None, :], axis=1)))
        rows.append({"stop_line": sl.id, "nearest_crossing_m": round(d, 2),
                     "confidence": round(sl.confidence, 2)})
        if d <= max_m:
            ds.append(d)
    return {
        "crossings": len(gt.crossings),
        "observed_stop_lines": len(rows),
        "near_a_crossing": len(ds),
        "median_offset_m": round(float(np.median(ds)), 2) if ds else None,
        "rows": rows,
    }


# --------------------------------------------------------------------------- #
# traffic lights -- self-consistency only
# --------------------------------------------------------------------------- #


def traffic_light_quality(layer: SemanticLayer) -> dict:
    """No positional ground truth exists, so report the evidence instead."""
    tls = [l for l in layer.landmarks.values() if l.kind is LandmarkKind.TRAFFIC_LIGHT]
    if not tls:
        return {"traffic_lights": 0}
    sig = np.array([l.position_sigma_m for l in tls])
    rep = np.array([l.residual_px for l in tls])
    views = np.array([l.n_views for l in tls])
    heights = np.array([l.height_above_ground for l in tls
                        if l.height_above_ground is not None])
    assigned = [a for a in layer.assignments.values() if a.lane_ids]

    # signal heads on one mast arm should be a metre or two apart and at the same
    # height: agreement between neighbours is independent evidence of the scale
    pos = np.array([l.position for l in tls])
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    nn = d.min(axis=1)

    return {
        "traffic_lights": len(tls),
        "median_position_sigma_m": round(float(np.median(sig)), 2),
        "median_reprojection_px": round(float(np.median(rep)), 2),
        "median_views": int(np.median(views)),
        "height_median_m": round(float(np.median(heights)), 2) if len(heights) else None,
        "height_p10_p90_m": [round(float(np.percentile(heights, 10)), 2),
                             round(float(np.percentile(heights, 90)), 2)]
        if len(heights) else None,
        "mast_arm_plausible_fraction": round(float(np.mean((heights >= 4.0) &
                                                           (heights <= 8.0))), 3)
        if len(heights) else None,
        "nearest_neighbour_median_m": round(float(np.median(nn)), 2),
        "assigned": len(assigned),
        "assignment_rate": round(len(assigned) / len(tls), 3),
        "ground_truth": "none available: AV2 does not annotate traffic lights",
    }


# --------------------------------------------------------------------------- #


def evaluate(graph: LaneGraph, layer: SemanticLayer, gt: AV2GroundTruth) -> dict:
    out = {
        "lane_geometry": lane_geometry_error(graph, gt),
        "lane_count": lane_count_error(graph, gt),
        "turn_semantics": turn_semantics_error(graph, layer, gt),
        "stop_lines": stop_line_error(layer, gt),
        "traffic_lights": traffic_light_quality(layer),
        "ground_truth": gt.stats(),
    }
    log.info("evaluation: geometry median %.2f m, turn IoU inferred %.2f -> observed %.2f",
             out["lane_geometry"].get("median_lateral_error_m", float("nan")),
             out["turn_semantics"].get("inferred_mean_iou", float("nan")),
             out["turn_semantics"].get("observed_mean_iou", float("nan")))
    return out


def total_lane_km(gt: AV2GroundTruth) -> float:
    return round(sum(polyline_length(l.center[:, :2]) for l in gt.lanes.values()) / 1000, 3)
