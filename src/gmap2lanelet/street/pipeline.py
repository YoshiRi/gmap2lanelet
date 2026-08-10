"""The street-level stage, end to end.

    OSM prior ──┐
                ├─► lane graph (phase 1)  ──┐
    aerial   ───┘                           │
                                            ├─► lane graph + semantics ─► Lanelet2
    street imagery ─► detect ─► triangulate ┤        (regulatory elements)
                   └─► IPM ─► BEV ─► stop lines / arrows

Two things are worth stating plainly about this design.

**Street imagery is evidence, not geometry.**  The lane graph is still produced
by the OSM prior and the overhead view.  The street stage only adds meaning:
where the signals are, which stop line they govern, what the arrows say.  The
one exception is deliberate -- the BEV mosaic is offered to the *same* geometry
backend as an alternative raster, because at 5 cm/px it can resolve markings
that 27 cm satellite imagery cannot, and showing that difference is part of
answering "where is public information insufficient".

**Nothing is forced.**  A signal whose approach cannot be decided stays
unassigned; a stop line that was never seen is marked inferred; an arrow that
does not classify cleanly is dropped rather than guessed.  The counts of each
are the result.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..config import PipelineConfig
from ..export.lanelet2_osm import write_lanelet2
from ..geo import LocalFrame
from ..qa.validate import validate_lanelet2
from ..types import LaneGraph
from .geo.ipm import BevMosaic
from .semantics.arrows import find_arrows, merge_arrows
from .semantics.associate import TrafficLightAssociator, associate
from .types import LandmarkKind, SemanticLayer

log = logging.getLogger(__name__)


@dataclass
class StreetResult:
    frame: LocalFrame
    graph: LaneGraph
    semantics: SemanticLayer
    validation: dict = field(default_factory=dict)
    paths: dict[str, str] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)
    detail: dict = field(default_factory=dict)

    def summary(self) -> dict:
        return {
            "graph": self.graph.stats(),
            "semantics": self.semantics.stats(),
            "lanelet2": self.validation,
            "timings_s": {k: round(v, 2) for k, v in self.timings.items()},
            **self.detail,
        }


def build_bev(sequence, prior, frame: LocalFrame, *, resolution: float = 0.05,
              roi_halfwidth_m: float = 16.0, cache: str | Path | None = None):
    """Rectify the sequence into an overhead raster over the prior's corridor."""
    cache = Path(cache) if cache else None
    if cache and (cache.with_suffix(".npy")).exists():
        from ..raster import GeoRaster
        meta = json.loads(cache.with_suffix(".json").read_text())
        data = np.load(cache.with_suffix(".npy"))
        log.info("bev: reusing cached mosaic %s", cache.with_suffix(".npy"))
        return GeoRaster(data, meta["x0"], meta["y0"], meta["dx"], meta["dy"], "bev")

    roi = [e.points for e in prior.edges.values()]
    rgb, _ = BevMosaic(resolution=resolution).build(
        sequence.frames, sequence.ground, frame,
        roi_polylines=roi, roi_halfwidth_m=roi_halfwidth_m)
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache.with_suffix(".npy"), rgb.data)
        cache.with_suffix(".json").write_text(json.dumps(
            {"x0": rgb.x0, "y0": rgb.y0, "dx": rgb.dx, "dy": rgb.dy}))
    return rgb


def extract_arrows(graph: LaneGraph, marking, *, min_lane_length: float = 12.0,
                   search_from: float = 2.0, search_to: float = 45.0) -> dict:
    """Read painted arrows out of every lane that reaches a junction.

    Arrows only exist on approaches, so only lanes with a successor inside an
    intersection are worth searching -- which also keeps the false-positive rate
    down, since a random blob mid-block cannot be mistaken for a turn permission.
    """
    turn_lane_ids = {i for ln in graph.lanes.values() if ln.kind == "turn"
                     for i in [ln.id]}
    found = []
    for ln in graph.lanes.values():
        if ln.kind != "road" or len(ln.centerline) < 5:
            continue
        if not any(s in turn_lane_ids for s in ln.successors):
            continue
        from ..geo import polyline_length
        if polyline_length(ln.centerline) < min_lane_length:
            continue
        found.extend(find_arrows(ln.id, ln.centerline, 0.5 * ln.width, marking,
                                 search_from=search_from, search_to=search_to))
    merged = merge_arrows(found)
    return {a.id: a for a in merged}


def run_semantics(graph: LaneGraph, frame: LocalFrame, landmarks, *,
                  marking=None, cfg: PipelineConfig | None = None,
                  out_dir: str | Path = "outputs/street",
                  associator: TrafficLightAssociator | None = None,
                  read_arrows: bool = True, imagery=None, frames=None) -> StreetResult:
    """Associate street-level evidence to an existing lane graph and export."""
    cfg = cfg or PipelineConfig()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t: dict[str, float] = {}

    arrows: dict = {}
    if read_arrows and marking is not None:
        t0 = time.time()
        arrows = extract_arrows(graph, marking)
        t["arrows"] = time.time() - t0

    t0 = time.time()
    layer = associate(graph, landmarks, marking=marking, arrows=arrows,
                      associator=associator)
    t["association"] = time.time() - t0

    t0 = time.time()
    map_path = out / "lanelet2_map_semantic.osm"
    _, counts = write_lanelet2(graph, map_path, cfg, semantics=layer)
    validation = validate_lanelet2(map_path, frame.lat0, frame.lon0)
    validation["export_counts"] = counts
    t["export"] = time.time() - t0

    (out / "semantics.json").write_text(json.dumps(layer.to_dict(), indent=1,
                                                   default=_default))
    paths = {"lanelet2": str(map_path), "semantics": str(out / "semantics.json")}

    items = street_review_items(layer)
    (out / "street_review.json").write_text(json.dumps(
        {"catalogue": {k: {"severity": s, "description": d}
                       for k, (s, d) in STREET_FAILURES.items()},
         "items": [i.to_dict(frame) for i in items]}, indent=1, default=_default))
    paths["street_review"] = str(out / "street_review.json")

    if imagery is not None or frames is not None:
        t0 = time.time()
        paths.update(_visuals(out / "viz", imagery, graph, layer, frames))
        t["visuals"] = time.time() - t0

    res = StreetResult(frame=frame, graph=graph, semantics=layer,
                       validation=validation, timings=t, paths=paths)
    res.detail["review"] = {"items": len(items),
                            "by_kind": _count([i.kind for i in items])}
    log.info("street stage: %s", json.dumps(layer.stats()))
    return res


def _count(xs) -> dict:
    out: dict[str, int] = {}
    for x in xs:
        out[x] = out.get(x, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def _visuals(vd: Path, imagery, graph, layer, frames) -> dict:
    from ..viz.semantics import render_landmark_check, render_semantics

    paths: dict[str, str] = {}
    if imagery is not None:
        paths["semantics_overlay"] = str(render_semantics(
            imagery, graph, layer, vd / "semantics.png",
            title="street-level semantics over the road raster"))
    if frames:
        p = render_landmark_check(frames, layer, vd / "landmark_check.png", n=5)
        if p is not None:
            paths["landmark_check"] = str(p)
    return paths


# --------------------------------------------------------------------------- #
# review items
# --------------------------------------------------------------------------- #


STREET_FAILURES = {
    "traffic_light_unassigned": (
        "high", "A signal head was located in 3-D but no approach could be shown to be "
        "the one it governs. Usually a junction arm the vehicle never drove, or a light "
        "for a movement the lane graph does not contain."),
    "traffic_light_approach_ambiguous": (
        "high", "Two approaches score almost equally. The geometry does not decide it; "
        "a human must."),
    "stop_line_inferred": (
        "medium", "No transverse marking was visible, so the stop line was placed at the "
        "junction edge. Position is a convention, not a measurement."),
    "arrow_repeats_disagree": (
        "medium", "The same lane carries arrows that classify differently. One of the "
        "blobs is probably not an arrow."),
    "arrow_contradicts_inferred_turns": (
        "high", "The painted arrow permits a different set of manoeuvres than the "
        "intersection model assumed. The paint is the authority."),
    "traffic_light_weak_geometry": (
        "medium", "Triangulated from a short baseline or few views: the position may be "
        "several metres out even though the association is plausible."),
}


def street_review_items(layer: SemanticLayer, start: int = 0) -> list:
    """Turn everything the street stage refused to decide into review items."""
    from ..types import ReviewItem

    items: list[ReviewItem] = []

    def add(code, msg, pos, elements, detail=None):
        sev = STREET_FAILURES.get(code, ("medium", ""))[0]
        items.append(ReviewItem(id=f"S{start + len(items):03d}", kind=code, severity=sev,
                                message=msg, position=np.asarray(pos, dtype=float)[:2],
                                element_ids=list(elements), detail=detail or {}))

    for a in layer.assignments.values():
        lm = layer.landmarks.get(a.landmark_id)
        if lm is None:
            continue
        if not a.lane_ids:
            add("traffic_light_unassigned",
                f"traffic light {lm.id} could not be tied to an approach ({a.reason})",
                lm.position, [lm.id], {"reason": a.reason, "confidence": lm.confidence})
        for f in a.flags:
            if f in STREET_FAILURES and f not in {"stop_line_inferred",
                                                  "traffic_light_unassigned"}:
                add(f, f"traffic light {lm.id}: {STREET_FAILURES[f][1]}",
                    lm.position, [lm.id, a.stop_line_id or ""], {"reason": a.reason})
        if a.lane_ids and ("weak_geometry" in lm.flags or "few_views" in lm.flags):
            add("traffic_light_weak_geometry",
                f"traffic light {lm.id}: sigma {lm.position_sigma_m:.1f} m over "
                f"{lm.n_views} views, baseline {lm.baseline_m:.0f} m",
                lm.position, [lm.id], {"flags": lm.flags})

    for sl in layer.stop_lines.values():
        if not sl.observed:
            add("stop_line_inferred",
                f"stop line {sl.id} placed at the junction edge: no marking observed",
                sl.points.mean(axis=0), [sl.id] + list(sl.lane_ids),
                dict(sl.provenance.detail))

    for c in layer.detail.get("arrow_conflicts", []):
        arrow = next((a for a in layer.arrows.values() if a.lane_id == c["lane"]), None)
        if arrow is None:
            continue
        add("arrow_contradicts_inferred_turns",
            f"lane {c['lane']}: paint says {'+'.join(c['observed'])}, the intersection "
            f"model assumed {'+'.join(c['inferred'])}",
            arrow.position, [c["lane"], arrow.id], c)

    for a in layer.arrows.values():
        if "arrow_repeats_disagree" in a.flags:
            add("arrow_repeats_disagree",
                f"lane {a.lane_id}: repeated arrows classify differently",
                a.position, [a.lane_id, a.id], dict(a.provenance.detail))

    log.info("street review: %s items", len(items))
    return items


def semantic_stats(layer: SemanticLayer) -> dict:
    tls = [l for l in layer.landmarks.values() if l.kind is LandmarkKind.TRAFFIC_LIGHT]
    assigned = [a for a in layer.assignments.values() if a.lane_ids]
    return {
        "traffic_lights_located": len(tls),
        "traffic_lights_assigned": len(assigned),
        "assignment_rate": round(len(assigned) / len(tls), 3) if tls else 0.0,
        "median_position_sigma_m": round(float(np.median([l.position_sigma_m for l in tls])), 2)
        if tls else None,
        "stop_lines_observed": sum(1 for s in layer.stop_lines.values() if s.observed),
        "stop_lines_inferred": sum(1 for s in layer.stop_lines.values() if not s.observed),
        "lanes_with_arrow": len(layer.arrows),
    }


def _default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, set):
        return sorted(o)
    return str(o)
