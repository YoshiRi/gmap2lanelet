"""Lanelet2 OSM-XML export.

Lanelet2 stores maps as OSM XML with a fixed vocabulary:

* **Point**  -> ``<node>`` with ``lat``/``lon`` (and ``ele``, ``local_x``,
  ``local_y`` as tags);
* **LineString** -> ``<way>`` tagged ``type=`` (``line_thin``, ``road_border``,
  ``virtual``, ...) and, for painted lines, ``subtype=`` (``solid``,
  ``dashed``, ``solid_solid``);
* **Lanelet** -> ``<relation type=lanelet>`` with exactly one ``left`` and one
  ``right`` way member (extra members are an error in Lanelet2's parser).

The single most important detail is that **successor relations are implicit**:
Lanelet2 decides that lanelet B follows A because they *share the same
boundary end points*.  So the exporter maintains one node registry keyed on
rounded coordinates, and the fusion stage takes care to make joined lanes end
at literally identical coordinates.

Provenance and confidence are written as extra ``gm2ll:*`` tags.  Lanelet2
preserves unknown attributes, so a reviewer opening the map in JOSM sees which
source produced each element and how much to trust it.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from ..config import PipelineConfig
from ..geo import simplify_polyline
from ..types import LaneGraph, MarkingType, Source

log = logging.getLogger(__name__)

# marking -> (type, subtype)
LINE_TYPES: dict[MarkingType, tuple[str, str | None]] = {
    MarkingType.SOLID: ("line_thin", "solid"),
    MarkingType.DASHED: ("line_thin", "dashed"),
    MarkingType.DOUBLE_SOLID: ("line_thin", "solid_solid"),
    MarkingType.ROAD_EDGE: ("road_border", None),
    MarkingType.VIRTUAL: ("virtual", None),
    MarkingType.UNKNOWN: ("virtual", None),
}


class Lanelet2Writer:
    def __init__(self, graph: LaneGraph, cfg: PipelineConfig, start_id: int = 1000):
        self.graph = graph
        self.cfg = cfg
        self._next = start_id
        self._nodes: dict[tuple[int, int], int] = {}
        self.root = ET.Element("osm", {"version": "0.6", "generator": "gmap2lanelet"})
        self.counts = {"nodes": 0, "ways": 0, "lanelets": 0}

    # -- ids ----------------------------------------------------------------

    def _id(self) -> int:
        self._next += 1
        return self._next

    def _node(self, x: float, y: float) -> int:
        q = self.cfg.node_merge_tolerance
        key = (int(round(x / q)), int(round(y / q)))
        if key in self._nodes:
            return self._nodes[key]

        nid = self._id()
        lon, lat = self.graph.frame.to_wgs84(x, y)
        el = ET.SubElement(self.root, "node", {
            "id": str(nid), "visible": "true", "version": "1",
            "lat": f"{float(lat):.9f}", "lon": f"{float(lon):.9f}",
        })
        _tag(el, "ele", f"{self.cfg.elevation:.2f}")
        _tag(el, "local_x", f"{x:.3f}")
        _tag(el, "local_y", f"{y:.3f}")
        self._nodes[key] = nid
        self.counts["nodes"] += 1
        return nid

    # -- primitives ---------------------------------------------------------

    def _way(self, pts: np.ndarray, tags: dict[str, str]) -> int:
        wid = self._id()
        el = ET.SubElement(self.root, "way", {"id": str(wid), "visible": "true", "version": "1"})
        last = None
        for x, y in pts:
            nid = self._node(float(x), float(y))
            if nid == last:                       # collapse duplicate consecutive nodes
                continue
            ET.SubElement(el, "nd", {"ref": str(nid)})
            last = nid
        for k, v in tags.items():
            _tag(el, k, v)
        self.counts["ways"] += 1
        return wid

    def _lanelet(self, left: int, right: int, tags: dict[str, str]) -> int:
        rid = self._id()
        el = ET.SubElement(self.root, "relation", {"id": str(rid), "visible": "true",
                                                   "version": "1"})
        ET.SubElement(el, "member", {"type": "way", "ref": str(left), "role": "left"})
        ET.SubElement(el, "member", {"type": "way", "ref": str(right), "role": "right"})
        for k, v in tags.items():
            _tag(el, k, v)
        self.counts["lanelets"] += 1
        return rid

    # -- driver -------------------------------------------------------------

    def build(self) -> ET.ElementTree:
        ways: dict[str, int] = {}
        for bid, b in self.graph.boundaries.items():
            if len(b.points) < 2:
                continue
            t, sub = LINE_TYPES.get(b.marking, ("virtual", None))
            tags = {"type": t}
            if sub:
                tags["subtype"] = sub
            tags["gm2ll:source"] = b.provenance.source.value
            tags["gm2ll:confidence"] = f"{b.confidence:.2f}"
            tags["gm2ll:id"] = bid
            pts = _keep_ends(b.points, simplify_polyline(b.points, self.cfg.export_simplify_m))
            ways[bid] = self._way(pts, tags)

        for lid, ln in self.graph.lanes.items():
            lw, rw = ways.get(ln.left_id), ways.get(ln.right_id)
            if lw is None or rw is None:
                log.warning("lane %s dropped: missing boundary way", lid)
                continue
            tags = {
                "type": "lanelet",
                "subtype": "road",
                "location": "urban",
                "one_way": "yes" if ln.one_way else "no",
                "gm2ll:id": lid,
                "gm2ll:source": ln.provenance.source.value,
                "gm2ll:confidence": f"{ln.confidence:.2f}",
                "gm2ll:kind": ln.kind,
                "gm2ll:segment": ln.segment_id,
            }
            if ln.speed_limit_kph:
                tags["speed_limit"] = f"{ln.speed_limit_kph:.0f} kmh"
            if ln.turn_direction:
                tags["turn_direction"] = ln.turn_direction
            flags = ln.attributes.get("flags") or []
            if flags:
                tags["gm2ll:flags"] = ";".join(sorted(set(flags)))
            if ln.provenance.detail.get("lane_count_osm") is not None:
                tags["gm2ll:lanes_osm"] = str(ln.provenance.detail["lane_count_osm"])
            if ln.provenance.detail.get("lane_count_image") is not None:
                tags["gm2ll:lanes_image"] = str(ln.provenance.detail["lane_count_image"])
            if ln.confidence < self.cfg.review_confidence_threshold:
                tags["gm2ll:review"] = "yes"
            self._lanelet(lw, rw, tags)

        return ET.ElementTree(self.root)

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tree = self.build()
        ET.indent(tree, space="  ")
        tree.write(path, encoding="utf-8", xml_declaration=True)
        log.info("wrote %s (%s nodes, %s ways, %s lanelets)", path, self.counts["nodes"],
                 self.counts["ways"], self.counts["lanelets"])
        return path


def _keep_ends(original: np.ndarray, simplified: np.ndarray) -> np.ndarray:
    """Douglas-Peucker may move nothing, but never let it move the end points.

    Shared end points are what Lanelet2 uses to derive successor relations, so
    they must survive simplification bit-for-bit.
    """
    out = np.asarray(simplified, dtype=float).copy()
    out[0] = original[0]
    out[-1] = original[-1]
    return out


def _tag(parent: ET.Element, k: str, v: str) -> None:
    ET.SubElement(parent, "tag", {"k": k, "v": str(v)})


def write_lanelet2(graph: LaneGraph, path: str | Path, cfg: PipelineConfig) -> tuple[Path, dict]:
    w = Lanelet2Writer(graph, cfg)
    p = w.write(path)
    return p, dict(w.counts)
