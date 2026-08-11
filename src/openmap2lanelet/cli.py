"""Command line interface.

    # public SpaceNet tile (imagery + OSM-like prior, no credentials needed)
    openmap2lanelet run --source spacenet --image-id 93 --out outputs/vegas93

    # arbitrary place on Earth: OSM via Overpass + your own aerial tile service
    openmap2lanelet run --source live \\
        --bbox 139.7601 35.6801 139.7649 35.6841 \\
        --tiles "https://.../{z}/{y}/{x}" --tiles-attribution "..." \\
        --drive-on left --out outputs/tokyo

    # batch several AOIs and compare failure modes
    openmap2lanelet batch --image-ids 93,162,48,10,100 --out outputs/batch

    # street-level semantics: traffic lights, stop lines, lane arrows, and a
    # Lanelet2 map with regulatory elements, scored against a held-back HD map
    openmap2lanelet street --log 20dd185d-b4eb-3024-a17a-b4e5d8b15b65 --city DTW \\
        --out outputs/street_detroit
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .config import PipelineConfig
from .geo import AOI, LocalFrame

log = logging.getLogger("openmap2lanelet")


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--out", default="outputs/run", help="output directory")
    p.add_argument("--config", help="JSON file of PipelineConfig overrides")
    p.add_argument("--drive-on", choices=["right", "left"], default="right",
                   help="side of the road traffic drives on")
    p.add_argument("--no-visuals", action="store_true", help="skip figure/viewer rendering")
    p.add_argument("-v", "--verbose", action="store_true")


def _config(args) -> PipelineConfig:
    d = {}
    if getattr(args, "config", None):
        d = json.loads(Path(args.config).read_text())
    cfg = PipelineConfig.from_dict(d)
    cfg.drive_on_right = args.drive_on == "right"
    return cfg


def _spacenet_inputs(args):
    from .sources.spacenet import SpaceNetSource

    src = SpaceNetSource(args.aoi_name, args.image_id)
    aoi = src.aoi()
    frame = LocalFrame.for_aoi(aoi)
    return aoi, frame, src.fetch_imagery(aoi, frame), src.fetch_prior(aoi)


def _live_inputs(args):
    from .sources.osm_overpass import OsmFileSource, OverpassSource
    from .sources.xyz_tiles import XYZTileSource

    if not args.bbox:
        raise SystemExit("--bbox W S E N is required for --source live")
    w, s, e, n = args.bbox
    aoi = AOI(args.name or "aoi", w, s, e, n)
    frame = LocalFrame.for_aoi(aoi)

    if not args.tiles:
        raise SystemExit("--tiles URL template is required for --source live")
    imagery = XYZTileSource(args.tiles, args.tiles_attribution or args.tiles,
                            zoom=args.zoom).fetch(aoi, frame)

    prior_src = OsmFileSource(args.osm_file) if args.osm_file else \
        OverpassSource(args.overpass)
    return aoi, frame, imagery, prior_src.fetch(aoi)


def cmd_run(args) -> int:
    from .pipeline import run

    aoi, frame, imagery, prior = (_spacenet_inputs(args) if args.source == "spacenet"
                                  else _live_inputs(args))
    res = run(aoi=aoi, imagery=imagery, prior_data=prior, frame=frame, cfg=_config(args),
              out_dir=args.out, make_visuals=not args.no_visuals)
    print(json.dumps(res.summary(), indent=1, default=str))
    return 0


def cmd_street(args) -> int:
    from .street.experiment import run_av2_experiment

    summary = run_av2_experiment(
        args.log, args.city, out_dir=args.out, stride=args.stride,
        max_frames=args.max_frames, bev_resolution=args.bev_resolution,
        lanes_tag=args.lanes_tag, weights=args.weights, imgsz=args.imgsz,
        cfg=_config(args), make_visuals=not args.no_visuals)
    print(json.dumps({k: v for k, v in summary.items() if k != "evaluation"},
                     indent=1, default=str))
    print(json.dumps(_eval_headline(summary["evaluation"]), indent=1, default=str))
    return 0


def _eval_headline(e: dict) -> dict:
    return {
        "lane_geometry_median_m": e.get("lane_geometry", {}).get("median_lateral_error_m"),
        "lane_count_within_1": e.get("lane_count", {}).get("within_1"),
        "turn_iou_inferred": e.get("turn_semantics", {}).get("inferred_mean_iou"),
        "turn_iou_observed": e.get("turn_semantics", {}).get("observed_mean_iou"),
        "traffic_lights": e.get("traffic_lights", {}).get("traffic_lights"),
        "traffic_light_assignment_rate": e.get("traffic_lights", {}).get("assignment_rate"),
        "stop_line_median_offset_m": e.get("stop_lines", {}).get("median_offset_m"),
    }


def cmd_batch(args) -> int:
    from .pipeline import run
    from .sources.spacenet import SpaceNetSource

    out_root = Path(args.out)
    rows = []
    for image_id in [int(x) for x in args.image_ids.split(",")]:
        try:
            src = SpaceNetSource(args.aoi_name, image_id)
            aoi = src.aoi()
            frame = LocalFrame.for_aoi(aoi)
            res = run(aoi=aoi, imagery=src.fetch_imagery(aoi, frame),
                      prior_data=src.fetch_prior(aoi), frame=frame, cfg=_config(args),
                      out_dir=out_root / f"img{image_id}",
                      make_visuals=not args.no_visuals)
            rows.append({"image_id": image_id, **_row(res)})
            log.info("img%s done", image_id)
        except Exception as exc:                            # noqa: BLE001
            log.exception("img%s failed: %s", image_id, exc)
            rows.append({"image_id": image_id, "error": str(exc)})

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "batch_summary.json").write_text(json.dumps(rows, indent=1, default=str))
    _print_table(rows)
    return 0


def _row(res) -> dict:
    st = res.failures.stats
    g = res.graph.stats()
    r = (res.validation or {}).get("routing", {})
    return {
        "lanes": g["lanes"], "lane_km": g["lane_km"],
        "markings_seen": st["carriageways_with_observed_markings"],
        "count_conflict": st["carriageways_with_lane_count_conflict"],
        "obs_boundaries": st["boundaries_observed_fraction"],
        "shift_median_m": st["prior_shift_median_m"],
        "confidence": g["mean_confidence"],
        "connected": r.get("connected_fraction", 0.0),
        "review_items": st["review_items"],
    }


def _print_table(rows: list[dict]) -> None:
    if not rows:
        return
    cols = [k for k in rows[0] if k != "error"]
    widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in cols}
    print("  ".join(c.ljust(widths[c]) for c in cols))
    print("  ".join("-" * widths[c] for c in cols))
    for r in rows:
        print("  ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="openmap2lanelet", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="build a lane-level map for one AOI")
    r.add_argument("--source", choices=["spacenet", "live"], default="spacenet")
    r.add_argument("--aoi-name", default="AOI_2_Vegas")
    r.add_argument("--image-id", type=int, default=93)
    r.add_argument("--bbox", nargs=4, type=float, metavar=("W", "S", "E", "N"))
    r.add_argument("--name", help="AOI name for --source live")
    r.add_argument("--tiles", help="XYZ tile URL template with {z}/{x}/{y}")
    r.add_argument("--tiles-attribution")
    r.add_argument("--zoom", type=int, default=19)
    r.add_argument("--overpass", default="https://overpass-api.de/api/interpreter")
    r.add_argument("--osm-file", help="use a local .osm extract instead of Overpass")
    _add_common(r)
    r.set_defaults(func=cmd_run)

    s = sub.add_parser("street", help="add street-level semantics (signals, stop lines, "
                                      "arrows) on an Argoverse 2 log")
    s.add_argument("--log", default="20dd185d-b4eb-3024-a17a-b4e5d8b15b65",
                   help="Argoverse 2 sensor log id")
    s.add_argument("--city", default="DTW", help="AV2 city code (DTW, ATX, MIA, PIT, PAO, WDC)")
    s.add_argument("--stride", type=int, default=3, help="use every Nth frame")
    s.add_argument("--max-frames", type=int, default=140)
    s.add_argument("--bev-resolution", type=float, default=0.05,
                   help="metres per pixel of the rectified overhead mosaic")
    s.add_argument("--lanes-tag", choices=["none", "class", "true"], default="none",
                   help="how much lane-count information the prior is allowed to carry")
    s.add_argument("--weights", default="yolov8m.pt")
    s.add_argument("--imgsz", type=int, default=1280)
    _add_common(s)
    s.set_defaults(func=cmd_street)

    b = sub.add_parser("batch", help="run several SpaceNet AOIs and compare")
    b.add_argument("--aoi-name", default="AOI_2_Vegas")
    b.add_argument("--image-ids", default="93,162,48,10,100")
    _add_common(b)
    b.set_defaults(func=cmd_batch)

    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
