"""The topology prior: a cleaned road-centreline graph in the local frame.

This is the "OSM half" of the PoC.  We take the map at its word about
*connectivity* (what joins what, one-way-ness, road class) and treat its
*geometry* as a rough guess to be corrected against the imagery later.

Cleaning steps
--------------
1. project way vertices into the local metric frame;
2. split every way at vertices it shares with another way, so that graph nodes
   are real junctions or way ends;
3. drop dangling stubs shorter than ``min_stub``;
4. cluster junction nodes that are within ``junction_radius`` of each other --
   a divided road crossing another divided road yields four OSM nodes but one
   physical intersection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from ..geo import AOI, LocalFrame, polyline_length, resample_polyline
from ..sources.base import PriorData
from . import osm_tags

log = logging.getLogger(__name__)


@dataclass
class RoadNode:
    id: str
    xy: np.ndarray
    edge_ids: list[str] = field(default_factory=list)

    @property
    def degree(self) -> int:
        return len(self.edge_ids)


@dataclass
class RoadEdge:
    id: str
    points: np.ndarray                       # (N, 2) metres, from_node -> to_node
    tags: dict[str, str]
    from_node: str
    to_node: str
    way_id: str

    @property
    def length(self) -> float:
        return polyline_length(self.points)

    @property
    def highway(self) -> str:
        return osm_tags.highway_class(self.tags)


@dataclass
class JunctionCluster:
    id: str
    node_ids: list[str]
    center: np.ndarray
    radius: float


@dataclass
class RoadPrior:
    frame: LocalFrame
    aoi: AOI
    nodes: dict[str, RoadNode]
    edges: dict[str, RoadEdge]
    clusters: dict[str, JunctionCluster] = field(default_factory=dict)
    attribution: str = ""
    kind: str = "osm"

    def node_cluster(self, node_id: str) -> str | None:
        for cid, cl in self.clusters.items():
            if node_id in cl.node_ids:
                return cid
        return None

    def stats(self) -> dict:
        return {
            "nodes": len(self.nodes),
            "edges": len(self.edges),
            "junction_clusters": len(self.clusters),
            "edge_km": round(sum(e.length for e in self.edges.values()) / 1000, 3),
            "classes": _counter([e.highway for e in self.edges.values()]),
            "with_lanes_tag": sum(1 for e in self.edges.values() if "lanes" in e.tags),
            "with_turn_lanes_tag": sum(1 for e in self.edges.values()
                                       if any(k.startswith("turn:lanes") for k in e.tags)),
            "with_maxspeed_tag": sum(1 for e in self.edges.values() if "maxspeed" in e.tags),
        }


def _counter(vals) -> dict[str, int]:
    out: dict[str, int] = {}
    for v in vals:
        out[v] = out.get(v, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def build_road_prior(prior: PriorData, aoi: AOI, frame: LocalFrame, *,
                     snap: float = 1.0, min_stub: float = 6.0,
                     junction_radius: float = 22.0,
                     vertex_step: float = 2.0) -> RoadPrior:
    """Turn raw ways into a cleaned metric road graph."""
    # 1. project into the metric frame.
    #
    # Do NOT resample yet.  In OSM a junction *is* a vertex shared by two ways,
    # and resampling moves vertices off it -- which silently turns a crossroads
    # into two roads that never meet.  Resampling happens after the split, per
    # edge, where it is harmless because `resample_polyline` pins the ends.
    ways: list[tuple[str, np.ndarray, dict]] = []
    for w in prior.ways:
        x, y = frame.to_local(w.coords[:, 0], w.coords[:, 1])
        pts = _dedupe(np.column_stack([x, y]), tol=0.05)
        if len(pts) < 2 or polyline_length(pts) < 1.0:
            continue
        ways.append((w.id, pts, dict(w.tags)))

    # 2. find split vertices: any quantised position used by >1 way, plus ends
    def key(p) -> tuple[int, int]:
        return (int(round(p[0] / snap)), int(round(p[1] / snap)))

    usage: dict[tuple[int, int], set[str]] = {}
    for wid, pts, _ in ways:
        for p in pts:
            usage.setdefault(key(p), set()).add(wid)
    shared = {k for k, s in usage.items() if len(s) > 1}

    nodes: dict[str, RoadNode] = {}
    edges: dict[str, RoadEdge] = {}

    def node_at(p: np.ndarray) -> str:
        k = key(p)
        nid = f"n{k[0]}_{k[1]}"
        if nid not in nodes:
            nodes[nid] = RoadNode(nid, np.asarray(p, dtype=float))
        return nid

    # 3. split
    for wid, pts, tags in ways:
        cut = [0]
        for i in range(1, len(pts) - 1):
            if key(pts[i]) in shared:
                cut.append(i)
        cut.append(len(pts) - 1)
        cut = sorted(set(cut))
        for a, b in zip(cut, cut[1:]):
            seg = pts[a:b + 1]
            if len(seg) < 2 or polyline_length(seg) < 0.5:
                continue
            n0, n1 = node_at(seg[0]), node_at(seg[-1])
            if n0 == n1 and polyline_length(seg) < 5.0:
                continue
            eid = f"e{len(edges)}"
            # Resample now that the topology is fixed: `resample_polyline` keeps
            # the first and last vertex, so the shared junction points survive.
            if polyline_length(seg) > vertex_step * 2:
                seg = resample_polyline(seg, vertex_step)
            # snap the ends onto the (quantised) node positions for exact topology
            seg = seg.copy()
            seg[0] = nodes[n0].xy
            seg[-1] = nodes[n1].xy
            edges[eid] = RoadEdge(eid, seg, tags, n0, n1, wid)
            nodes[n0].edge_ids.append(eid)
            nodes[n1].edge_ids.append(eid)

    _drop_stubs(nodes, edges, min_stub)
    clusters = _cluster_junctions(nodes, junction_radius)

    rp = RoadPrior(frame=frame, aoi=aoi, nodes=nodes, edges=edges, clusters=clusters,
                   attribution=prior.attribution, kind=prior.kind)
    log.info("road prior: %s", rp.stats())
    return rp


def _dedupe(pts: np.ndarray, tol: float) -> np.ndarray:
    keep = [0]
    for i in range(1, len(pts)):
        if np.hypot(*(pts[i] - pts[keep[-1]])) > tol:
            keep.append(i)
    return pts[keep]


def _drop_stubs(nodes: dict[str, RoadNode], edges: dict[str, RoadEdge], min_stub: float) -> None:
    changed = True
    while changed:
        changed = False
        for eid, e in list(edges.items()):
            if e.length >= min_stub:
                continue
            deg0 = nodes[e.from_node].degree
            deg1 = nodes[e.to_node].degree
            if min(deg0, deg1) == 1 and max(deg0, deg1) >= 2:
                for nid in (e.from_node, e.to_node):
                    nodes[nid].edge_ids.remove(eid)
                del edges[eid]
                changed = True
        for nid, n in list(nodes.items()):
            if n.degree == 0:
                del nodes[nid]


def _cluster_junctions(nodes: dict[str, RoadNode], radius: float) -> dict[str, JunctionCluster]:
    """Single-link clustering of degree>=3 nodes."""
    js = [n for n in nodes.values() if n.degree >= 3]
    if not js:
        return {}
    parent = {n.id: n.id for n in js}

    def find(a: str) -> str:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, a in enumerate(js):
        for b in js[i + 1:]:
            if np.hypot(*(a.xy - b.xy)) <= radius:
                ra, rb = find(a.id), find(b.id)
                if ra != rb:
                    parent[ra] = rb

    groups: dict[str, list[str]] = {}
    for n in js:
        groups.setdefault(find(n.id), []).append(n.id)

    out: dict[str, JunctionCluster] = {}
    for i, (_, members) in enumerate(sorted(groups.items())):
        pts = np.array([nodes[m].xy for m in members])
        c = pts.mean(axis=0)
        r = float(np.max(np.hypot(*(pts - c).T))) if len(pts) > 1 else 0.0
        out[f"j{i}"] = JunctionCluster(f"j{i}", sorted(members), c, r)
    return out
