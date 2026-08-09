"""End-to-end pipeline.

    public data  ->  topology prior  +  geometry observation
                              \\      /
                            lane graph
                                 |
                         Lanelet2 + review report
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import PipelineConfig
from .export.lanelet2_osm import write_lanelet2
from .fusion.builder import build_lane_graph
from .geo import AOI, LocalFrame
from .observation.base import Evidence
from .observation.classical import ClassicalBackend
from .prior.road_graph import RoadPrior, build_road_prior
from .qa.failures import FailureReport, analyse
from .qa.validate import validate_lanelet2
from .sources.base import ImageryData, PriorData
from .types import LaneGraph

log = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    aoi: AOI
    frame: LocalFrame
    imagery: ImageryData
    prior_data: PriorData
    prior: RoadPrior
    evidence: Evidence
    graph: LaneGraph
    segments: dict
    failures: FailureReport
    validation: dict
    paths: dict[str, str] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)

    def summary(self) -> dict:
        return {
            "aoi": self.aoi.to_dict(),
            "frame": self.frame.to_dict(),
            "imagery": {"gsd": round(self.imagery.gsd, 3),
                        "size_px": list(self.imagery.raster.shape),
                        "attribution": self.imagery.attribution,
                        **self.imagery.detail},
            "prior": {"attribution": self.prior_data.attribution,
                      "kind": self.prior_data.kind, **self.prior.stats()},
            "observation": {"backend": self.evidence.backend, **self.evidence.detail},
            "graph": self.graph.stats(),
            "lanelet2": self.validation,
            "failures": self.failures.stats,
            "timings_s": {k: round(v, 2) for k, v in self.timings.items()},
            "paths": self.paths,
        }


def run(*, aoi: AOI, imagery: ImageryData, prior_data: PriorData, frame: LocalFrame,
        cfg: PipelineConfig | None = None, out_dir: str | Path = "outputs/run",
        backend=None, make_visuals: bool = True) -> PipelineResult:
    cfg = cfg or PipelineConfig()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t: dict[str, float] = {}

    t0 = time.time()
    prior = build_road_prior(prior_data, aoi, frame, snap=cfg.node_snap,
                             min_stub=cfg.min_stub,
                             junction_radius=cfg.junction_cluster_radius)
    t["prior"] = time.time() - t0

    t0 = time.time()
    backend = backend or ClassicalBackend()
    evidence = backend.run(imagery.raster, prior)
    t["observation"] = time.time() - t0

    t0 = time.time()
    graph, segments = build_lane_graph(prior, evidence, aoi, frame, cfg)
    t["fusion"] = time.time() - t0

    t0 = time.time()
    map_path = out / "lanelet2_map.osm"
    write_lanelet2(graph, map_path, cfg)
    validation = validate_lanelet2(map_path, frame.lat0, frame.lon0)
    t["export"] = time.time() - t0

    t0 = time.time()
    failures = analyse(graph, prior, segments, evidence, cfg, validation)
    t["failure_analysis"] = time.time() - t0

    paths = {"lanelet2": str(map_path)}
    _write_json(out / "lane_graph.json", _graph_json(graph))
    paths["lane_graph"] = str(out / "lane_graph.json")
    _write_json(out / "review_items.json",
                {"catalogue": _catalogue(), "stats": failures.stats,
                 "items": [i.to_dict(frame) for i in failures.items]})
    paths["review_items"] = str(out / "review_items.json")

    if make_visuals:
        t0 = time.time()
        paths.update(_visuals(out, imagery, prior, graph, evidence, segments, failures))
        t["visuals"] = time.time() - t0

    result = PipelineResult(aoi, frame, imagery, prior_data, prior, evidence, graph,
                            segments, failures, validation, paths, t)

    from .qa.report import write_report
    paths["report"] = str(write_report(result, out / "report.md"))
    _write_json(out / "summary.json", result.summary())
    paths["summary"] = str(out / "summary.json")
    result.paths = paths

    log.info("done in %.1fs -> %s", sum(t.values()), out)
    return result


# --------------------------------------------------------------------------- #


def _catalogue() -> dict:
    from .qa.failures import CATALOGUE

    return CATALOGUE


def _visuals(out: Path, imagery, prior, graph, evidence, segments, failures) -> dict:
    from .viz.overlay import render_crop, render_evidence, render_overlay, render_profile_debug
    from .viz.viewer import write_viewer

    paths = {}
    vd = out / "viz"
    paths["overlay"] = str(render_overlay(imagery.raster, prior, graph, vd / "overlay.png",
                                          title="imagery + OSM prior + generated lanes"))
    paths["prior_only"] = str(render_overlay(
        imagery.raster, prior, graph, vd / "prior_only.png", show_boundaries=False,
        show_centerlines=False, show_intersections=False, dim=0.2,
        title="imagery + OSM prior (topology input)"))
    paths["evidence"] = str(render_evidence(imagery.raster, evidence, vd / "evidence.png"))

    # a road-aligned debug view of the longest segment that produced lanes
    cands = [r for r in segments.values() if r.corridors and r.strips]
    if cands:
        best = max(cands, key=lambda r: r.edge.length)
        p = render_profile_debug(best, vd / "profile_debug.png")
        if p:
            paths["profile_debug"] = str(p)

    # crops for the most severe review items
    crops = []
    for it in failures.items[:8]:
        p = render_crop(imagery.raster, prior, graph, it.position, 55,
                        vd / f"review_{it.id}.png", title=f"{it.id} {it.kind}")
        crops.append({"id": it.id, "kind": it.kind, "path": str(p)})
    paths["review_crops"] = crops

    paths["viewer"] = str(write_viewer(out / "viewer.html", imagery, prior, graph,
                                       evidence, failures))
    return paths


def _graph_json(graph: LaneGraph) -> dict:
    return {
        "aoi": graph.aoi.to_dict(),
        "frame": graph.frame.to_dict(),
        "meta": graph.meta,
        "stats": graph.stats(),
        "lanes": [l.to_dict() for l in graph.lanes.values()],
        "boundaries": [b.to_dict() for b in graph.boundaries.values()],
        "intersections": [i.to_dict() for i in graph.intersections.values()],
    }


def _write_json(path: Path, data: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1, default=_default))
    return path


def _default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, Path):
        return str(o)
    return str(o)
