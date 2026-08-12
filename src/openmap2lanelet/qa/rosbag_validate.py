"""Validate an exported map's traffic lights against a real driven rosbag.

Two ways to validate an exported map exist in this repo: structurally, via
the real Lanelet2 library (``qa/validate.py``), and now empirically, against
what a real drive actually saw. The bag's own traffic-light landmarks (built
by the *same*, unmodified ``LandmarkBuilder`` used to build maps in the first
place -- see ``street/sources/rosbag.py``) are compared against the map's
already-exported ``type=traffic_light`` ways to find matches, map entries the
drive did not confirm, and -- the point of this module -- signals the drive
saw that the map does not have at all.
"""

from __future__ import annotations

import json
import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..geo import LocalFrame
from ..street.sources.base import StreetFrame
from ..street.types import Landmark, LandmarkKind
from ..types import ReviewItem

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class MapTrafficLight:
    id: str
    lat: float
    lon: float
    ele_m: float
    aspect: str | None
    confidence: float


def read_map_traffic_lights(osm_path: str | Path) -> list[MapTrafficLight]:
    """Every ``type=traffic_light`` way in an exported map, no ``lanelet2``
    dependency -- a plain XML walk, the same pattern the test suite already
    uses to assert on exported maps."""
    root = ET.parse(osm_path).getroot()
    nodes = {n.get("id"): n for n in root.findall("node")}

    out: list[MapTrafficLight] = []
    for way in root.findall("way"):
        tags = {t.get("k"): t.get("v") for t in way.findall("tag")}
        if tags.get("type") != "traffic_light":
            continue
        pts = [nodes[nd.get("ref")] for nd in way.findall("nd") if nd.get("ref") in nodes]
        if not pts:
            continue
        lat = sum(float(n.get("lat")) for n in pts) / len(pts)
        lon = sum(float(n.get("lon")) for n in pts) / len(pts)
        eles = [float(t.get("v")) for n in pts for t in n.findall("tag") if t.get("k") == "ele"]
        out.append(MapTrafficLight(
            id=tags.get("gm2ll:id", way.get("id")), lat=lat, lon=lon,
            ele_m=sum(eles) / len(eles) if eles else 0.0,
            aspect=tags.get("gm2ll:aspect"),
            confidence=float(tags.get("gm2ll:confidence", 0.0))))
    log.info("%s: %s traffic lights in the map", Path(osm_path).name, len(out))
    return out


def _in_view(pos3: np.ndarray, bag_frames: list[StreetFrame], range_m: float) -> bool:
    for f in bag_frames:
        if float(np.linalg.norm(f.camera.center[:2] - pos3[:2])) > range_m:
            continue
        _, ok = f.camera.project(pos3[None, :])
        if bool(ok[0]):
            return True
    return False


def validate_against_rosbag(map_path: str | Path, bag_landmarks: list[Landmark],
                             bag_frames: list[StreetFrame], map_frame: LocalFrame, *,
                             match_radius_m: float = 5.0, in_view_range_m: float = 60.0,
                             out_dir: str | Path | None = None) -> dict:
    """Confirm/contradict a map's traffic lights against a real drive.

    ``map_frame`` must be the *same* ``LocalFrame`` the caller threaded
    through ``RosbagStreetSource.fetch``/``LandmarkBuilder`` -- the map's own
    ``local_x``/``local_y`` tags are in an origin the caller doesn't
    otherwise know, so matching is always done by re-projecting the map's
    WGS84 lat/lon into this frame, never by trusting those tags.
    """
    map_lights = read_map_traffic_lights(map_path)
    map_xy = [np.asarray(map_frame.to_local(ml.lon, ml.lat), dtype=float).reshape(2)
             for ml in map_lights]

    bag_lights = [lm for lm in bag_landmarks if lm.kind is LandmarkKind.TRAFFIC_LIGHT]

    # greedy 1:1 nearest-neighbour matching, closest pairs first
    candidates = [(float(np.hypot(*(map_xy[i] - bag_lights[j].position[:2]))), i, j)
                  for i in range(len(map_lights)) for j in range(len(bag_lights))]
    candidates = [c for c in candidates if c[0] <= match_radius_m]
    candidates.sort(key=lambda c: c[0])

    matched_map: set[int] = set()
    matched_bag: set[int] = set()
    items: list[ReviewItem] = []

    def add(kind: str, severity: str, message: str, position: np.ndarray,
            element_ids: list[str], detail: dict) -> None:
        items.append(ReviewItem(id=f"RB{len(items):03d}", kind=kind, severity=severity,
                                message=message, position=position,
                                element_ids=element_ids, detail=detail))

    for d, i, j in candidates:
        if i in matched_map or j in matched_bag:
            continue
        matched_map.add(i)
        matched_bag.add(j)
        add("traffic_light_confirmed", "low",
            f"map light {map_lights[i].id} confirmed by the drive "
            f"({d:.1f} m from landmark {bag_lights[j].id})",
            map_xy[i], [map_lights[i].id, bag_lights[j].id],
            {"map_id": map_lights[i].id, "bag_landmark_id": bag_lights[j].id,
             "distance_m": round(d, 2)})

    for i, ml in enumerate(map_lights):
        if i in matched_map:
            continue
        pos3 = np.array([map_xy[i][0], map_xy[i][1], ml.ele_m])
        in_view = _in_view(pos3, bag_frames, in_view_range_m)
        # absence off the driven corridor is not evidence of a map error --
        # street imagery is a linear sample of a planar problem (README) --
        # so severity only escalates when the drive actually looked there.
        add("traffic_light_map_only", "medium" if in_view else "low",
            f"map light {ml.id} was not confirmed by the drive" +
            (" despite being in view" if in_view else " (never in the driven corridor)"),
            map_xy[i], [ml.id], {"map_id": ml.id, "in_view": in_view})

    for j, lm in enumerate(bag_lights):
        if j in matched_bag:
            continue
        add("traffic_light_undocumented", "high",
            f"landmark {lm.id} triangulated from the drive has no nearby map entry",
            lm.position[:2].copy(), [lm.id],
            {"bag_landmark_id": lm.id, "n_views": lm.n_views, "confidence": lm.confidence})

    result = {
        "map_lights": len(map_lights), "bag_lights": len(bag_lights),
        "confirmed": len(matched_map),
        "map_only": len(map_lights) - len(matched_map),
        "undocumented": len(bag_lights) - len(matched_bag),
        "items": [it.to_dict(map_frame) for it in items],
    }
    if out_dir:
        path = Path(out_dir) / "rosbag_validation.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=1))

    log.info("rosbag validation: %s confirmed, %s map-only, %s undocumented",
             result["confirmed"], result["map_only"], result["undocumented"])
    return result
