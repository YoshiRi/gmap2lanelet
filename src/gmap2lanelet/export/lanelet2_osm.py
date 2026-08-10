"""Lanelet2 OSM-XML export.

Lanelet2 stores maps as OSM XML with a fixed vocabulary:

* **Point**  -> ``<node>`` with ``lat``/``lon`` (and ``ele``, ``local_x``,
  ``local_y`` as tags);
* **LineString** -> ``<way>`` tagged ``type=`` (``line_thin``, ``road_border``,
  ``virtual``, ...) and, for painted lines, ``subtype=`` (``solid``,
  ``dashed``, ``solid_solid``);
* **Lanelet** -> ``<relation type=lanelet>`` with exactly one ``left`` and one
  ``right`` way member, plus any number of ``regulatory_element`` members;
* **Regulatory element** -> ``<relation type=regulatory_element>``, here always
  ``subtype=traffic_light``, holding the signal itself (``refers``) and the line
  where a vehicle must stop (``ref_line``).

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
from ..types import LaneGraph, MarkingType

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
    def __init__(self, graph: LaneGraph, cfg: PipelineConfig, start_id: int = 1000,
                 semantics=None):
        self.graph = graph
        self.cfg = cfg
        self.semantics = semantics
        self._next = start_id
        self._nodes: dict[tuple[int, int, int], int] = {}
        self._cache: dict[tuple[str, str], int] = {}
        self.root = ET.Element("osm", {"version": "0.6", "generator": "gmap2lanelet"})
        self.counts = {"nodes": 0, "ways": 0, "lanelets": 0,
                       "regulatory_elements": 0, "traffic_lights": 0, "stop_lines": 0}

    # -- ids ----------------------------------------------------------------

    def _id(self) -> int:
        self._next += 1
        return self._next

    def _node(self, x: float, y: float, ele: float | None = None) -> int:
        q = self.cfg.node_merge_tolerance
        z = self.cfg.elevation if ele is None else float(ele)
        # Elevation is part of the identity: a signal head 5 m above a stop line
        # must not collapse onto it, and shared *ground* end points -- which are
        # what carries successor relations -- all sit at the same z.
        key = (int(round(x / q)), int(round(y / q)), int(round(z / q)))
        if key in self._nodes:
            return self._nodes[key]

        nid = self._id()
        lon, lat = self.graph.frame.to_wgs84(x, y)
        el = ET.SubElement(self.root, "node", {
            "id": str(nid), "visible": "true", "version": "1",
            "lat": f"{float(lat):.9f}", "lon": f"{float(lon):.9f}",
        })
        _tag(el, "ele", f"{z:.2f}")
        _tag(el, "local_x", f"{x:.3f}")
        _tag(el, "local_y", f"{y:.3f}")
        self._nodes[key] = nid
        self.counts["nodes"] += 1
        return nid

    # -- primitives ---------------------------------------------------------

    def _way(self, pts: np.ndarray, tags: dict[str, str], ele: float | None = None) -> int:
        wid = self._id()
        el = ET.SubElement(self.root, "way", {"id": str(wid), "visible": "true", "version": "1"})
        last = None
        for x, y in pts:
            nid = self._node(float(x), float(y), ele)
            if nid == last:                       # collapse duplicate consecutive nodes
                continue
            ET.SubElement(el, "nd", {"ref": str(nid)})
            last = nid
        for k, v in tags.items():
            _tag(el, k, v)
        self.counts["ways"] += 1
        return wid

    def _lanelet(self, left: int, right: int, tags: dict[str, str],
                 regulatory: list[int] | None = None) -> int:
        rid = self._id()
        el = ET.SubElement(self.root, "relation", {"id": str(rid), "visible": "true",
                                                   "version": "1"})
        ET.SubElement(el, "member", {"type": "way", "ref": str(left), "role": "left"})
        ET.SubElement(el, "member", {"type": "way", "ref": str(right), "role": "right"})
        for r in regulatory or []:
            ET.SubElement(el, "member", {"type": "relation", "ref": str(r),
                                         "role": "regulatory_element"})
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

        regs = self._regulatory_elements()

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
            for k in ("turn_manoeuvres_observed", "arrow_id", "arrow_confidence"):
                if k in ln.attributes:
                    v = ln.attributes[k]
                    tags[f"gm2ll:{k}"] = ";".join(v) if isinstance(v, list) else str(v)
            self._lanelet(lw, rw, tags, regs.get(lid))

        return ET.ElementTree(self.root)

    # -- regulatory elements ------------------------------------------------

    def _regulatory_elements(self) -> dict[str, list[int]]:
        """Emit traffic-light regulatory elements; return lane -> relation ids.

        Lanelet2's model is a three-way relation: the *regulatory element* holds
        the physical signal (``refers``) and the line where a vehicle must stop
        (``ref_line``), and every lanelet it governs carries it as a member with
        role ``regulatory_element``.  That is precisely the chain the street
        layer recovers, so it maps across without invention.

        Only assignments that survived association are written.  A signal whose
        controlled lanes could not be decided is *not* exported as a regulatory
        element -- it would silently claim knowledge the pipeline does not have.
        It stays in the review report instead.
        """
        out: dict[str, list[int]] = {}
        sem = self.semantics
        if sem is None:
            return out

        for a in sem.assignments.values():
            lm = sem.landmarks.get(a.landmark_id)
            if lm is None or not a.lane_ids or a.stop_line_id not in sem.stop_lines:
                continue
            sl = sem.stop_lines[a.stop_line_id]

            tl_way = self._traffic_light_way(lm)
            sl_way = self._stop_line_way(sl)

            rid = self._id()
            el = ET.SubElement(self.root, "relation",
                               {"id": str(rid), "visible": "true", "version": "1"})
            ET.SubElement(el, "member", {"type": "way", "ref": str(tl_way), "role": "refers"})
            ET.SubElement(el, "member", {"type": "way", "ref": str(sl_way), "role": "ref_line"})
            _tag(el, "type", "regulatory_element")
            _tag(el, "subtype", "traffic_light")
            _tag(el, "gm2ll:id", a.id)
            _tag(el, "gm2ll:source", "street_imagery")
            _tag(el, "gm2ll:confidence", f"{a.confidence:.2f}")
            _tag(el, "gm2ll:reason", a.reason)
            _tag(el, "gm2ll:landmark", a.landmark_id)
            if a.flags:
                _tag(el, "gm2ll:flags", ";".join(sorted(set(a.flags))))
            if a.confidence < self.cfg.review_confidence_threshold:
                _tag(el, "gm2ll:review", "yes")
            self.counts["regulatory_elements"] += 1

            for lid in a.lane_ids:
                if lid in self.graph.lanes:
                    out.setdefault(lid, []).append(rid)
                    # the manoeuvre lanes through the junction inherit the signal
                    for succ in self.graph.lanes[lid].successors:
                        if self.graph.lanes.get(succ) is not None and \
                                self.graph.lanes[succ].kind == "turn":
                            out.setdefault(succ, []).append(rid)
        for lid in out:
            out[lid] = sorted(set(out[lid]))
        return out

    def _traffic_light_way(self, lm) -> int:
        """A signal head as a short horizontal line at its measured height."""
        key = ("tl", lm.id)
        if key in self._cache:
            return self._cache[key]
        p = np.asarray(lm.position, dtype=float)
        # span the face perpendicular to the direction it looks
        face = lm.facing if lm.facing is not None else 0.0
        n = np.array([-np.sin(face), np.cos(face)])
        pts = np.array([p[:2] - 0.35 * n, p[:2] + 0.35 * n])
        tags = {
            "type": "traffic_light",
            "subtype": "red_yellow_green",
            "height": f"{0.9:.2f}",
            "gm2ll:id": lm.id,
            "gm2ll:source": lm.provenance.source.value,
            "gm2ll:confidence": f"{lm.confidence:.2f}",
            "gm2ll:n_views": str(lm.n_views),
            "gm2ll:position_sigma_m": f"{lm.position_sigma_m:.2f}",
            "gm2ll:baseline_m": f"{lm.baseline_m:.1f}",
        }
        if lm.height_above_ground is not None:
            tags["gm2ll:height_above_ground_m"] = f"{lm.height_above_ground:.2f}"
        if lm.flags:
            tags["gm2ll:flags"] = ";".join(sorted(set(lm.flags)))
        wid = self._way(pts, tags, ele=self.cfg.elevation + float(p[2] - self._datum(lm)))
        self._cache[key] = wid
        self.counts["traffic_lights"] += 1
        return wid

    def _datum(self, lm) -> float:
        """Local ground level under a landmark, so ``ele`` stays a height."""
        if lm.height_above_ground is None:
            return float(lm.position[2])
        return float(lm.position[2]) - float(lm.height_above_ground)

    def _stop_line_way(self, sl) -> int:
        key = ("sl", sl.id)
        if key in self._cache:
            return self._cache[key]
        tags = {
            "type": "stop_line",
            "gm2ll:id": sl.id,
            "gm2ll:source": sl.provenance.source.value,
            "gm2ll:confidence": f"{sl.confidence:.2f}",
            "gm2ll:observed": "yes" if sl.observed else "no",
        }
        if sl.flags:
            tags["gm2ll:flags"] = ";".join(sorted(set(sl.flags)))
        wid = self._way(np.asarray(sl.points, dtype=float), tags)
        self._cache[key] = wid
        self.counts["stop_lines"] += 1
        return wid

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tree = self.build()
        ET.indent(tree, space="  ")
        tree.write(path, encoding="utf-8", xml_declaration=True)
        log.info("wrote %s (%s nodes, %s ways, %s lanelets, %s regulatory elements)",
                 path, self.counts["nodes"], self.counts["ways"],
                 self.counts["lanelets"], self.counts["regulatory_elements"])
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


def write_lanelet2(graph: LaneGraph, path: str | Path, cfg: PipelineConfig,
                   semantics=None) -> tuple[Path, dict]:
    w = Lanelet2Writer(graph, cfg, semantics=semantics)
    p = w.write(path)
    return p, dict(w.counts)
