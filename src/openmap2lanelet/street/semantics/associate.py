"""Traffic light -> stop line -> controlled lanelets.

A 3-D position for a signal head is not yet a map.  What a vehicle needs is the
*relation*: this light governs this stop line, which is crossed by these lanes.
That relation is what this module recovers, and it is the part that street-level
imagery alone cannot supply -- it only exists once the lane graph from the
OSM+aerial stage is on the table.

The association is deliberately posed in the **approach frame** rather than as a
nearest-neighbour search in the plane.  At a four-way junction every arm's stop
line is within 30 m of every other arm's signals, so distance alone assigns
about half of them to the wrong approach.  Three constraints break the tie:

* **Longitudinal**: a signal sits at, or beyond, the stop line it governs, never
  far behind it.  In the approach frame that is a one-sided window.
* **Facing**: a signal is aimed back down the approach it controls.  Because the
  lights were triangulated from the frames that saw them, the mean bearing from
  the light to those cameras *is* the direction it faces, to within the spread
  of the observing trajectory.
* **Height**: a mast-arm head is 4-7 m up, a pedestrian head 2.5-3.5 m.  A
  detection triangulated to 15 m is not a traffic light.

Anything that fails to clear the threshold is left unassigned and flagged, which
is the whole point: an unassigned light is a review item, a wrongly assigned one
is a map defect that no downstream check will catch.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np

from ...geo import angle_diff, heading
from ...types import LaneGraph
from ..types import Landmark, LandmarkKind, SemanticLayer, StopLine, TrafficLightAssignment
from .stopline import detect_stop_line, make_stop_line, stop_line_geometry

log = logging.getLogger(__name__)


@dataclass
class Approach:
    """One carriageway arriving at one junction: the unit signals control."""

    id: str
    intersection_id: str
    segment_id: str
    lane_ids: list[str] = field(default_factory=list)
    #: representative centreline of the approach, travel direction, ending at
    #: the junction; used as the ``s`` axis of the approach frame
    axis: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    stop_point: np.ndarray = field(default_factory=lambda: np.zeros(2))
    heading: float = 0.0           # rad, direction of travel at the stop line
    half_width: float = 5.0
    stop_line_id: str | None = None

    def frame(self, p: np.ndarray) -> tuple[float, float]:
        """(longitudinal, lateral) of a point relative to the stop line.

        Longitudinal is positive *downstream* -- into the junction, the side a
        signal is on.  Lateral is positive to the driver's left.
        """
        d = np.asarray(p, dtype=float)[:2] - self.stop_point
        c, s = math.cos(self.heading), math.sin(self.heading)
        return float(d[0] * c + d[1] * s), float(-d[0] * s + d[1] * c)


# --------------------------------------------------------------------------- #
# approaches
# --------------------------------------------------------------------------- #


def build_approaches(graph: LaneGraph, *, gather_m: float = 12.0,
                     extend_m: float = 18.0) -> list[Approach]:
    """Group incoming road lanes by (junction, prior edge).

    A lane is *incoming* to a junction when its downstream end is inside the
    junction's circle-plus-margin and it is travelling towards the centre.  That
    is exactly the set the intersection stage trimmed back, so no extra geometry
    is invented here.
    """
    out: dict[str, Approach] = {}
    for inter in graph.intersections.values():
        for ln in graph.lanes.values():
            if ln.kind != "road" or len(ln.centerline) < 3:
                continue
            end = ln.centerline[-1]
            d = float(np.hypot(*(end - inter.center)))
            if d > inter.radius + gather_m:
                continue
            h = heading(ln.centerline, at_start=False)
            to_centre = inter.center - end
            if float(np.hypot(*to_centre)) < 1e-6:
                continue
            bearing = math.atan2(to_centre[1], to_centre[0])
            if abs(angle_diff(bearing, h)) > math.radians(75):
                continue                       # leaving, or passing by
            key = f"{inter.id}|{ln.segment_id}"
            ap = out.setdefault(key, Approach(id=key.replace("|", "@"),
                                              intersection_id=inter.id,
                                              segment_id=ln.segment_id))
            ap.lane_ids.append(ln.id)

    for ap in out.values():
        _finalise(ap, graph, extend_m)
    approaches = [a for a in out.values() if len(a.axis) >= 3]
    log.info("approaches: %s across %s intersections",
             len(approaches), len({a.intersection_id for a in approaches}))
    return approaches


def _finalise(ap: Approach, graph: LaneGraph, extend_m: float = 18.0) -> None:
    lanes = [graph.lanes[i] for i in ap.lane_ids]
    # order left-to-right in the travel frame so the axis is the middle lane
    h = float(np.mean([heading(l.centerline, at_start=False) for l in lanes]))
    n = np.array([-math.sin(h), math.cos(h)])
    lanes.sort(key=lambda l: -float(np.dot(l.centerline[-1], n)))
    ap.lane_ids = [l.id for l in lanes]

    mid = lanes[len(lanes) // 2]
    axis = np.asarray(mid.centerline, dtype=float)[:, :2]
    # The lane was trimmed back to clear the junction circle, but the stop bar
    # is painted *at* the junction, i.e. beyond that cut.  Searching only the
    # trimmed lane would look everywhere except where the marking is, so the
    # axis is run forward along its final heading to cover the gap.
    if extend_m > 0 and len(axis) >= 2:
        h = heading(axis, at_start=False)
        axis = np.vstack([axis, axis[-1] + extend_m * np.array([math.cos(h), math.sin(h)])])
    ap.axis = axis
    ap.stop_point = ap.axis[-1].copy()
    ap.heading = heading(ap.axis, at_start=False)
    # half-width spans the whole carriageway, not one lane: a signal is aimed at
    # the approach, and the lateral test must not reject the outer lanes
    lat = [abs(float(np.dot(l.centerline[-1] - ap.stop_point, n))) for l in lanes]
    ap.half_width = float(max(max(lat) + 0.5 * lanes[0].width, 2.0))


# --------------------------------------------------------------------------- #
# stop lines
# --------------------------------------------------------------------------- #


def build_stop_lines(approaches: list[Approach], marking, *,
                     search_from: float = 0.0, search_to: float = 32.0,
                     min_score: float = 0.30) -> dict[str, StopLine]:
    """Observe a stop bar per approach, or place one at the junction edge.

    ``marking`` is the marking-response raster (BEV or aerial).  When no bar is
    visible the stop line is still emitted -- a junction has one whether or not
    it is painted -- but as ``Source.INFERRED`` with a low confidence and a flag,
    so that the difference between *measured* and *assumed* survives into the
    exported map.
    """
    out: dict[str, StopLine] = {}
    for i, ap in enumerate(approaches):
        lid = f"sl{i:03d}"
        found = None
        if marking is not None:
            found = detect_stop_line(ap.axis, ap.half_width, marking,
                                     search_from=search_from, search_to=search_to)
        if found is not None and found[1] >= min_score:
            station, score, detail = found
            pts = stop_line_geometry(ap.axis, station, ap.half_width)
            sl = make_stop_line(lid, list(ap.lane_ids), pts, ap.segment_id,
                                True, score, {**detail, "approach": ap.id,
                                              "station_from_junction_m": round(station, 2)})
            # the stop line, not the trimmed lane end, is the datum for signals
            ap.stop_point = pts.mean(axis=0)
        else:
            pts = stop_line_geometry(ap.axis, 0.0, ap.half_width)
            why = "no transverse marking above threshold" if marking is not None \
                else "no marking raster available"
            sl = make_stop_line(lid, list(ap.lane_ids), pts, ap.segment_id,
                                False, 0.0, {"approach": ap.id, "reason": why,
                                             "best_score": None if found is None
                                             else round(found[1], 3)})
        ap.stop_line_id = lid
        out[lid] = sl

    obs = sum(1 for s in out.values() if s.observed)
    log.info("stop lines: %s (%s observed, %s inferred from the junction edge)",
             len(out), obs, len(out) - obs)
    return out


# --------------------------------------------------------------------------- #
# traffic lights
# --------------------------------------------------------------------------- #


class TrafficLightAssociator:
    """Score every (light, approach) pair; keep the ones that make sense."""

    def __init__(self, *, long_window: tuple[float, float] = (-14.0, 55.0),
                 max_lateral_m: float = 24.0, height_range: tuple[float, float] = (2.0, 9.0),
                 facing_tol_deg: float = 85.0, sigma_long_m: float = 22.0,
                 sigma_lat_m: float = 11.0, min_score: float = 0.28,
                 max_range_m: float = 70.0):
        self.long_window = long_window
        self.max_lateral_m = max_lateral_m
        self.height_range = height_range
        self.facing_tol = math.radians(facing_tol_deg)
        self.sigma_long = sigma_long_m
        self.sigma_lat = sigma_lat_m
        self.min_score = min_score
        self.max_range_m = max_range_m

    def run(self, landmarks: list[Landmark], approaches: list[Approach],
            stop_lines: dict[str, StopLine]) -> list[TrafficLightAssignment]:
        assignments: list[TrafficLightAssignment] = []
        n = 0
        for lm in landmarks:
            if lm.kind is not LandmarkKind.TRAFFIC_LIGHT:
                continue
            n += 1
            scored = [(s, a, why) for s, a, why in
                      (self._score(lm, a) for a in approaches) if s > 0]
            scored.sort(key=lambda t: -t[0])
            aid = f"tla{len(assignments):03d}"
            if not scored or scored[0][0] < self.min_score:
                assignments.append(TrafficLightAssignment(
                    id=aid, landmark_id=lm.id, stop_line_id=None, lane_ids=[],
                    intersection_id=None, confidence=0.0,
                    reason=self._why_not(lm, approaches, scored),
                    flags=["traffic_light_unassigned"] + self._unassigned_flags(lm)))
                continue

            score, ap, why = scored[0]
            flags: list[str] = []
            # a second approach almost as good means the geometry did not decide
            if len(scored) > 1 and scored[1][0] > 0.85 * score \
                    and scored[1][1].id != ap.id:
                flags.append("traffic_light_approach_ambiguous")
            sl = stop_lines.get(ap.stop_line_id or "")
            if sl is not None and not sl.observed:
                flags.append("stop_line_inferred")

            conf = float(np.clip(score * (0.55 + 0.45 * lm.confidence), 0.05, 0.95))
            assignments.append(TrafficLightAssignment(
                id=aid, landmark_id=lm.id, stop_line_id=ap.stop_line_id,
                lane_ids=list(ap.lane_ids), intersection_id=ap.intersection_id,
                confidence=conf,
                reason=why["why"], flags=flags))

        ok = sum(1 for a in assignments if a.lane_ids)
        log.info("traffic light association: %s/%s assigned to an approach", ok, n)
        return assignments

    # -- why a light stayed unassigned --------------------------------------

    def _why_not(self, lm: Landmark, approaches: list[Approach], scored) -> str:
        """Say which stage ran out, not just that nothing matched.

        "unassigned" is useless to a reviewer; "the nearest modelled approach is
        94 m away" points straight at the junction the geometry stage failed to
        resolve, and "faces away from every approach" points at the light
        instead.  The distinction decides which half of the pipeline to fix.
        """
        if scored:
            best = scored[0]
            return f"best approach scored {best[0]:.2f} (below threshold): {best[2]['why']}"
        if not approaches:
            return "the lane graph contains no modelled approach at all"
        near = min(approaches,
                   key=lambda a: float(np.hypot(*(a.stop_point - np.asarray(lm.position)[:2]))))
        d = float(np.hypot(*(near.stop_point - np.asarray(lm.position)[:2])))
        if d > self.max_range_m:
            return (f"nearest modelled approach is {d:.0f} m away: the junction this "
                    "signal belongs to was not resolved by the geometry stage")
        s, _ = near.frame(lm.position)
        if s < self.long_window[0]:
            return (f"{-s:.0f} m *behind* the only approach modelled at junction "
                    f"{near.intersection_id}: it is the far-side head of the opposing "
                    "arm, and that arm produced no lanes")
        h = lm.height_above_ground
        if h is not None and h < 3.6:
            return (f"{d:.0f} m from the nearest approach, mounted at {h:.1f} m: "
                    "probably a pedestrian or near-side head, not a lane signal")
        return f"no approach within {self.max_range_m:.0f} m passed the geometry tests"

    @staticmethod
    def _unassigned_flags(lm: Landmark) -> list[str]:
        h = lm.height_above_ground
        return ["likely_pedestrian_signal"] if h is not None and h < 3.6 else []

    # -- scoring ------------------------------------------------------------

    def _score(self, lm: Landmark, ap: Approach):
        s, u = ap.frame(lm.position)
        rng = math.hypot(s, u)
        why = {"approach": ap.id, "long_m": round(s, 1), "lat_m": round(u, 1)}

        if rng > self.max_range_m:
            return 0.0, ap, {**why, "why": "out of range"}
        if not (self.long_window[0] <= s <= self.long_window[1]):
            return 0.0, ap, {**why, "why": "behind the stop line or past the junction"}
        if abs(u) > self.max_lateral_m:
            return 0.0, ap, {**why, "why": "too far off the approach axis"}

        h = lm.height_above_ground
        if h is not None and not (self.height_range[0] <= h <= self.height_range[1]):
            return 0.0, ap, {**why, "why": f"implausible height {h:.1f} m"}

        # a signal governing this approach faces back along it
        face_cos = 1.0
        if lm.facing is not None:
            err = abs(angle_diff(lm.facing, ap.heading + math.pi))
            if err > self.facing_tol:
                return 0.0, ap, {**why, "why": f"faces away ({math.degrees(err):.0f} deg)"}
            face_cos = float(0.5 + 0.5 * math.cos(err))
            why["facing_err_deg"] = round(math.degrees(err), 1)

        # The window opens a little way *behind* the stop line: a near-side head
        # is legal, and both the triangulated position and the painted bar carry
        # a couple of metres of error.  What keeps that from stealing the
        # opposing approach's signals is the facing test, not the window --
        # a signal 12 m behind A's stop line is 12 m past B's, but it looks the
        # wrong way for B.
        g_long = math.exp(-0.5 * ((s - 10.0) / self.sigma_long) ** 2)
        g_lat = math.exp(-0.5 * (u / self.sigma_lat) ** 2)
        h_term = 1.0 if h is None else float(np.clip(1.2 - abs(h - 5.0) / 7.0, 0.4, 1.0))
        score = float(g_long * g_lat * face_cos * h_term)
        why["why"] = (f"{s:.0f} m past the stop line, {abs(u):.0f} m "
                      f"{'left' if u > 0 else 'right'} of the axis")
        return score, ap, why


# --------------------------------------------------------------------------- #
# arrows -> per-lane turn permission
# --------------------------------------------------------------------------- #


def apply_arrows(graph: LaneGraph, arrows: dict, *, min_confidence: float = 0.35
                 ) -> dict:
    """Record what the paint says next to what the intersection model assumed.

    The intersection stage guesses "leftmost turns left"; an arrow observed in a
    lane is a measurement of the same fact.  Neither is overwritten here, and
    that is deliberate: one readable arrow is evidence that a manoeuvre *is*
    allowed, not evidence that the others are forbidden -- a lane can carry a
    combined symbol, or a second symbol further back that this pass missed.  So
    both sets are attached to the lane and a disagreement becomes a review item
    rather than a silent edit to the map's connectivity.
    """
    changed, agreed, conflict = 0, 0, []
    for a in arrows.values():
        ln = graph.lanes.get(a.lane_id)
        if ln is None or a.confidence < min_confidence:
            continue
        observed = sorted(a.manoeuvres)
        inferred = sorted(_inferred_manoeuvres(graph, ln))
        ln.attributes["turn_manoeuvres_observed"] = observed
        ln.attributes["turn_manoeuvres_inferred"] = inferred
        ln.attributes["arrow_id"] = a.id
        ln.attributes["arrow_confidence"] = round(a.confidence, 3)
        if inferred and set(observed) != set(inferred):
            conflict.append({"lane": ln.id, "observed": observed, "inferred": inferred})
            changed += 1
        elif inferred:
            agreed += 1
            # the convention was right: say so, and raise the lane's confidence
            ln.confidence = float(np.clip(ln.confidence + 0.05, 0, 0.95))
            ln.provenance.detail["turn_confirmed_by_arrow"] = a.id
    log.info("arrows: %s lanes confirmed the inferred turn set, %s contradicted it",
             agreed, changed)
    return {"arrow_lanes": len(arrows), "arrow_agrees": agreed,
            "arrow_conflicts": conflict}


def _inferred_manoeuvres(graph: LaneGraph, ln) -> set[str]:
    out: set[str] = set()
    for sid in ln.successors:
        s = graph.lanes.get(sid)
        if s is None:
            continue
        m = s.turn_direction or s.provenance.detail.get("manoeuvre")
        if m == "through":
            m = "through"
        if m:
            out.add(m)
    return out


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #


def associate(graph: LaneGraph, landmarks: list[Landmark], marking=None,
              arrows: dict | None = None, *,
              associator: TrafficLightAssociator | None = None) -> SemanticLayer:
    """Run the whole semantic association chain over one lane graph."""
    approaches = build_approaches(graph)
    stop_lines = build_stop_lines(approaches, marking)
    assoc = associator or TrafficLightAssociator()
    assignments = assoc.run(landmarks, approaches, stop_lines)

    # a stop line only governs the lanes of *its* approach; record back-links so
    # the export can attach the regulatory element to the right lanelets
    for a in assignments:
        if a.stop_line_id:
            stop_lines[a.stop_line_id].provenance.detail.setdefault(
                "traffic_lights", []).append(a.landmark_id)

    layer = SemanticLayer(
        landmarks={l.id: l for l in landmarks},
        stop_lines=stop_lines,
        arrows=dict(arrows or {}),
        assignments={a.id: a for a in assignments},
    )
    layer.detail["approaches"] = len(approaches)
    layer.detail["approaches_with_signals"] = len(
        {a.intersection_id for a in assignments if a.lane_ids})
    if arrows:
        layer.detail.update(apply_arrows(graph, arrows))

    # keep only stop lines that either were observed or govern something: an
    # inferred stop line on an unsignalised arm is noise, not evidence
    for sid, sl in list(stop_lines.items()):
        if not sl.observed and not sl.provenance.detail.get("traffic_lights"):
            stop_lines.pop(sid)
    layer.detail["unassigned_traffic_lights"] = sum(
        1 for a in assignments if not a.lane_ids)
    return layer
