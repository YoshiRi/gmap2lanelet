"""The reproducible phase-2 experiment on one Argoverse 2 log.

The design of the experiment matters as much as the code:

* the **inputs** are an OSM-equivalent prior (centrelines and a road class, with
  the lane counts and turn tags stripped) and the log's posed camera imagery;
* the **held-back truth** is the log's full HD map -- lane-level geometry,
  carriageway membership and junction connectivity -- which the pipeline never
  sees and which is opened only by :mod:`gmap2lanelet.street.evaluate`.

That split is what makes the numbers mean anything.  It also means the run is
reproducible from public data alone: the log, its map and the detector weights
are all fetched from public endpoints and cached.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np

from ..config import PipelineConfig
from ..geo import LocalFrame
from ..pipeline import run as run_base
from ..sources.base import ImageryData
from . import evaluate as ev
from . import pipeline as spipe
from .sources import av2_map
from .sources.av2 import AV2LogSource

log = logging.getLogger(__name__)


def run_av2_experiment(log_id: str, city: str, *, out_dir: str | Path,
                       stride: int = 3, max_frames: int = 140,
                       bev_resolution: float = 0.05, lanes_tag: str = "none",
                       weights: str = "yolov8m.pt", imgsz: int = 1280,
                       conf: float = 0.25, cfg: PipelineConfig | None = None,
                       make_visuals: bool = True) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = cfg or PipelineConfig()
    t: dict[str, float] = {}

    # -- inputs -------------------------------------------------------------
    t0 = time.time()
    src = AV2LogSource(log_id, city, stride=stride, max_frames=max_frames)
    raw = av2_map.load_map(src.dir / "map/log_map.json")
    frame = _frame_for(raw, src)
    gt = av2_map.ground_truth(raw, src.georef, frame)
    aoi = av2_map.aoi_for_ground_truth(gt, frame)
    prior_data = av2_map.build_prior(gt, frame, lanes_tag=lanes_tag)
    seq = src.fetch(aoi, frame)
    t["inputs"] = time.time() - t0

    # -- street imagery -> overhead raster ----------------------------------
    from ..prior.road_graph import build_road_prior

    t0 = time.time()
    prior = build_road_prior(prior_data, aoi, frame, snap=cfg.node_snap,
                             min_stub=cfg.min_stub,
                             junction_radius=cfg.junction_cluster_radius)
    bev = spipe.build_bev(seq, prior, frame, resolution=bev_resolution,
                          cache=out / "cache" / "bev")
    t["bev"] = time.time() - t0

    imagery = ImageryData(
        raster=bev, gsd=bev.gsd,
        attribution=seq.attribution + ", inverse-perspective rectified",
        detail={"kind": "street-derived BEV mosaic", "frames": len(seq.frames)})

    # -- lane graph (phase 1, on the rectified raster) -----------------------
    t0 = time.time()
    base = run_base(aoi=aoi, imagery=imagery, prior_data=prior_data, frame=frame,
                    cfg=cfg, out_dir=out, make_visuals=make_visuals)
    t["lane_graph"] = time.time() - t0

    # -- street semantics ----------------------------------------------------
    t0 = time.time()
    landmarks = _landmarks(seq, out, weights=weights, imgsz=imgsz, conf=conf)
    t["detect_and_triangulate"] = time.time() - t0

    street = spipe.run_semantics(
        base.graph, frame, landmarks, marking=base.evidence.marking, cfg=cfg,
        out_dir=out, imagery=bev if make_visuals else None,
        frames=seq.frames if make_visuals else None)
    t.update(street.timings)

    # -- evaluation ----------------------------------------------------------
    t0 = time.time()
    scores = ev.evaluate(base.graph, street.semantics, gt)
    t["evaluation"] = time.time() - t0

    summary = {
        "log": log_id, "city": city,
        "inputs": {"prior": prior_data.detail | {"ways": len(prior_data.ways)},
                   "frames": len(seq.frames), "bev_gsd_m": round(bev.gsd, 3),
                   "attribution": [prior_data.attribution, seq.attribution]},
        "lane_graph": base.graph.stats(),
        "semantics": street.semantics.stats(),
        "lanelet2": street.validation,
        "evaluation": scores,
        "review": {"phase1": base.failures.stats, "phase2": street.detail.get("review")},
        "paths": {**base.paths, **street.paths},
        "timings_s": {k: round(v, 2) for k, v in t.items()},
    }
    (out / "experiment.json").write_text(json.dumps(summary, indent=1, default=_default))
    log.info("experiment done in %.0fs -> %s", sum(t.values()), out)
    return summary


# --------------------------------------------------------------------------- #


def _frame_for(raw: dict, src: AV2LogSource) -> LocalFrame:
    pts = np.array([[p["x"], p["y"]] for v in raw["lane_segments"].values()
                    for p in v["left_lane_boundary"]])
    lon, lat = src.georef.city_to_wgs84(pts[:, 0], pts[:, 1])
    return LocalFrame(float((lat.min() + lat.max()) / 2),
                      float((lon.min() + lon.max()) / 2))


def _landmarks(seq, out: Path, *, weights: str, imgsz: int, conf: float):
    """Detect + triangulate, caching the per-frame boxes (the expensive part)."""
    from .detect.yolo import YoloDetector
    from .semantics.landmarks import LandmarkBuilder
    from .types import Detection, LandmarkKind

    cache = out / "cache" / "detections.json"
    if cache.exists():
        rows = json.loads(cache.read_text())
        dets = [Detection(frame_id=d["frame"], kind=LandmarkKind(d["kind"]),
                          bbox=tuple(d["bbox"]), score=d["score"], camera=d["camera"])
                for d in rows]
        log.info("reusing %s cached detections", len(dets))
    else:
        dets = YoloDetector(weights=weights, imgsz=imgsz, conf=conf).detect(seq.frames)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps([d.to_dict() for d in dets]))
    return LandmarkBuilder().build(dets, seq.frames, seq.ground)


def _default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, set):
        return sorted(o)
    return str(o)
