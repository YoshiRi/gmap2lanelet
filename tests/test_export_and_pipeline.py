"""Export, Lanelet2 validation and the end-to-end pipeline (no network)."""

import importlib.util
import xml.etree.ElementTree as ET

import pytest
from conftest import build_cross_scene

from openmap2lanelet.config import PipelineConfig
from openmap2lanelet.export.lanelet2_osm import write_lanelet2
from openmap2lanelet.fusion.builder import build_lane_graph
from openmap2lanelet.observation.classical import ClassicalBackend
from openmap2lanelet.pipeline import run
from openmap2lanelet.prior.road_graph import build_road_prior
from openmap2lanelet.qa.validate import validate_lanelet2

# Only the two tests that *load* the map need the official bindings.  Gating the
# whole module on them would silently drop the export-structure and end-to-end
# tests on any machine without Lanelet2 installed -- which is most of them, and
# which is exactly where a regression would go unnoticed.
needs_lanelet2 = pytest.mark.skipif(
    importlib.util.find_spec("lanelet2") is None,
    reason="official Lanelet2 bindings not installed")


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    from conftest import build_scene

    aoi, frame, imagery, prior, truth = build_scene()
    rp = build_road_prior(prior, aoi, frame)
    ev = ClassicalBackend().run(imagery.raster, rp)
    cfg = PipelineConfig()
    graph, segments = build_lane_graph(rp, ev, aoi, frame, cfg)
    path = tmp_path_factory.mktemp("ll2") / "map.osm"
    write_lanelet2(graph, path, cfg)
    return graph, frame, path, cfg


def test_export_structure(built):
    graph, _, path, _ = built
    root = ET.parse(path).getroot()
    nodes = root.findall("node")
    ways = root.findall("way")
    rels = root.findall("relation")

    assert nodes and ways and rels
    assert len(rels) == len(graph.lanes)

    for n in nodes:
        assert -90 <= float(n.get("lat")) <= 90
        assert -180 <= float(n.get("lon")) <= 180
        keys = {t.get("k") for t in n.findall("tag")}
        assert {"ele", "local_x", "local_y"} <= keys

    for w in ways:
        tags = {t.get("k"): t.get("v") for t in w.findall("tag")}
        assert tags["type"] in {"line_thin", "road_border", "virtual"}
        if tags["type"] == "line_thin":
            assert tags["subtype"] in {"solid", "dashed", "solid_solid"}
        assert "gm2ll:source" in tags and "gm2ll:confidence" in tags
        assert len(w.findall("nd")) >= 2

    for r in rels:
        roles = [m.get("role") for m in r.findall("member")]
        assert roles.count("left") == 1 and roles.count("right") == 1
        assert len(roles) == 2, "Lanelet2 errors on extra lanelet members"
        tags = {t.get("k"): t.get("v") for t in r.findall("tag")}
        assert tags["type"] == "lanelet" and tags["subtype"] == "road"
        assert tags["one_way"] in {"yes", "no"}


@needs_lanelet2
def test_lanelet2_loads_without_errors(built):
    _, frame, path, _ = built
    rep = validate_lanelet2(path, frame.lat0, frame.lon0)
    assert rep["available"] and rep["parsed"]
    assert rep["parse_error_count"] == 0
    assert rep["lanelets"] > 0
    assert rep["routing"]["passable_lanelets"] == rep["lanelets"]


def test_adjacent_lanes_share_their_boundary(built):
    """Lane k's left boundary must *be* lane k+1's right boundary, not a copy."""
    graph, _, path, _ = built
    for direction in ("f", "b"):
        lanes = sorted((l for l in graph.lanes.values() if l.id.endswith(direction)),
                       key=lambda l: l.index_from_left)
        assert len(lanes) >= 2
        for a, b in zip(lanes, lanes[1:]):
            shared = {a.left_id, a.right_id} & {b.left_id, b.right_id}
            assert shared, f"{a.id} and {b.id} do not share a boundary"
    # n lanes in a carriageway need n+1 boundaries, not 2n
    assert len(graph.boundaries) < 2 * len(graph.lanes)


@needs_lanelet2
def test_intersection_produces_routable_turns(tmp_path):
    aoi, frame, imagery, prior, _ = build_cross_scene()
    cfg = PipelineConfig()
    rp = build_road_prior(prior, aoi, frame)
    ev = ClassicalBackend().run(imagery.raster, rp)
    graph, _ = build_lane_graph(rp, ev, aoi, frame, cfg)

    assert len(graph.intersections) == 1
    inter = next(iter(graph.intersections.values()))
    assert inter.approach_count == 4
    assert inter.turn_lane_ids, "a four-way junction must produce turn lanes"
    # connectivity is inferred, so it must be reported as such
    assert any("turn:lanes" in n for n in inter.notes)
    assert inter.confidence < 0.6

    turns = [l for l in graph.lanes.values() if l.kind == "turn"]
    assert {t.turn_direction for t in turns} <= {"left", "right", "through", "u_turn"}
    assert "through" in {t.turn_direction for t in turns}
    assert all(t.predecessors and t.successors for t in turns)

    path = tmp_path / "cross.osm"
    write_lanelet2(graph, path, cfg)
    rep = validate_lanelet2(path, frame.lat0, frame.lon0)
    assert rep["parsed"] and rep["parse_error_count"] == 0
    # the junction is the whole point: most lanelets must reach a successor
    assert rep["routing"]["connected_fraction"] > 0.8


def test_end_to_end_pipeline(tmp_path, scene):
    aoi, frame, imagery, prior, truth = scene
    res = run(aoi=aoi, imagery=imagery, prior_data=prior, frame=frame,
              out_dir=tmp_path / "run", make_visuals=False)

    for key in ("lanelet2", "lane_graph", "review_items", "report", "summary"):
        assert (tmp_path / "run").joinpath(
            {"lanelet2": "lanelet2_map.osm", "lane_graph": "lane_graph.json",
             "review_items": "review_items.json", "report": "report.md",
             "summary": "summary.json"}[key]).exists()

    # the map is only *loaded* where the bindings exist; everything else about
    # the run is checked either way
    if res.validation.get("available"):
        assert res.validation["parsed"]
    assert res.graph.stats()["road_lanes"] == truth["lanes"]
    # every lane carries provenance and a confidence
    for ln in res.graph.lanes.values():
        assert 0.0 <= ln.confidence <= 1.0
        assert ln.provenance.source is not None
    assert "by_road_group" in res.failures.stats


def test_review_items_are_georeferenced(tmp_path, scene):
    aoi, frame, imagery, prior, _ = scene
    res = run(aoi=aoi, imagery=imagery, prior_data=prior, frame=frame,
              out_dir=tmp_path / "run2", make_visuals=False)
    for item in res.failures.items:
        d = item.to_dict(frame)
        assert aoi.south - 0.01 <= d["lat"] <= aoi.north + 0.01
        assert aoi.west - 0.01 <= d["lon"] <= aoi.east + 0.01
        assert d["severity"] in {"high", "medium", "low"}
        assert d["message"]
