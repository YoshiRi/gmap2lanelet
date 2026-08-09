"""Fusion tests on the synthetic scene, where ground truth is known."""

import numpy as np
import pytest

from gmap2lanelet.config import PipelineConfig
from gmap2lanelet.fusion.builder import build_lane_graph
from gmap2lanelet.fusion.corridor import extract_corridors
from gmap2lanelet.fusion.lanes import solve_lanes
from gmap2lanelet.fusion.profile import build_profile
from gmap2lanelet.observation.classical import ClassicalBackend
from gmap2lanelet.prior.road_graph import build_road_prior
from gmap2lanelet.types import MarkingType, Source

from conftest import build_scene


@pytest.fixture(scope="module")
def solved():
    aoi, frame, imagery, prior, truth = build_scene()
    rp = build_road_prior(prior, aoi, frame)
    ev = ClassicalBackend().run(imagery.raster, rp)
    edge = next(iter(rp.edges.values()))
    prof = build_profile(edge.points, ev, half_width=25.0)
    cors = extract_corridors(prof, expected_width=truth["lanes"] * truth["lane_w"] + 1.6,
                             oneway=False)
    sol = solve_lanes(prof, cors[0], edge.tags)
    return aoi, frame, imagery, prior, truth, rp, ev, prof, cors, sol


def test_observation_finds_road_and_markings(solved):
    *_, ev, prof, cors, sol = solved
    road = ev.road_prob.data
    assert 0.05 < float((road > 0.5).mean()) < 0.6
    assert float(ev.marking.data.max()) > 0.5


def test_corridor_recovers_true_edges_despite_shifted_prior(solved):
    _, _, _, _, truth, _, _, prof, cors, _ = solved
    assert len(cors) == 1
    cor = cors[0]
    assert cor.coverage > 0.85

    # The prior sits `prior_shift` m to the left of the true centre, so the
    # corridor centre must come back by about that much (offsets are +left).
    shift = float(np.median(cor.center_offset[cor.valid]))
    assert shift == pytest.approx(-truth["prior_shift"], abs=0.8)

    width = cor.mean_width()
    assert width == pytest.approx(2 * truth["road_half"], abs=1.2)


def test_lane_solution_matches_truth(solved):
    *_, truth, _, _, prof, cors, sol = solved
    assert sol.n_lanes == truth["lanes"]
    assert sol.n_image == truth["lanes"]
    assert sol.count_source is Source.FUSED           # image and OSM agreed
    assert sol.confidence >= 0.8
    assert "lane_count_conflict" not in sol.flags

    # Boundary positions are lateral offsets from the (misplaced) prior, so the
    # truth in that frame is the true world offset minus the prior's own shift.
    offsets = sorted(float(np.median(b.offsets)) for b in sol.boundaries)
    true_off = sorted(o - truth["prior_shift"] for o in truth["boundaries"])
    assert np.allclose(offsets, true_off, atol=0.7), f"{offsets} vs {true_off}"

    markings = [b.marking for b in sol.boundaries]
    assert markings[0] is MarkingType.ROAD_EDGE and markings[-1] is MarkingType.ROAD_EDGE
    assert MarkingType.SOLID in markings                # the centre line
    assert MarkingType.DASHED in markings               # the lane dividers


def test_directions_split_for_two_way_road(solved):
    *_, sol = solved
    fwd = [s for s in sol.lanes if s.forward]
    assert len(fwd) == sol.n_lanes // 2
    # right-hand traffic: forward lanes are the right-hand ones
    assert all(s.left_idx >= sol.n_lanes // 2 for s in fwd)


def test_lane_count_conflict_is_flagged():
    """OSM claiming 2 lanes on a road painted with 4 must raise a conflict."""
    aoi, frame, imagery, prior, truth = build_scene()
    prior.ways[0].tags["lanes"] = "2"
    rp = build_road_prior(prior, aoi, frame)
    ev = ClassicalBackend().run(imagery.raster, rp)
    edge = next(iter(rp.edges.values()))
    prof = build_profile(edge.points, ev, half_width=25.0)
    cor = extract_corridors(prof, expected_width=16.0, oneway=False)[0]
    sol = solve_lanes(prof, cor, edge.tags)

    assert "lane_count_conflict" in sol.flags
    assert sol.n_osm == 2 and sol.detail["n_geometry"] == 4


def test_unmarked_road_falls_back_to_prior():
    """With no paint at all the lane count must come from OSM, at low confidence."""
    aoi, frame, imagery, prior, truth = build_scene(paint=False)
    rp = build_road_prior(prior, aoi, frame)
    ev = ClassicalBackend().run(imagery.raster, rp)
    edge = next(iter(rp.edges.values()))
    prof = build_profile(edge.points, ev, half_width=25.0)
    cor = extract_corridors(prof, expected_width=16.0, oneway=False)[0]
    sol = solve_lanes(prof, cor, edge.tags)

    assert sol.count_source is Source.OSM
    assert sol.n_lanes == truth["lanes"]
    assert sol.confidence <= 0.5
    assert "no_lane_markings_observed" in sol.flags
    assert all(b.marking in (MarkingType.ROAD_EDGE, MarkingType.VIRTUAL)
               for b in sol.boundaries)


def test_lane_graph_is_built_and_connected(scene):
    aoi, frame, imagery, prior, truth = scene
    rp = build_road_prior(prior, aoi, frame)
    ev = ClassicalBackend().run(imagery.raster, rp)
    graph, segments = build_lane_graph(rp, ev, aoi, frame, PipelineConfig())

    assert graph.stats()["road_lanes"] == truth["lanes"]
    for ln in graph.lanes.values():
        assert 2.5 <= ln.width <= 4.5
        assert len(ln.centerline) >= 2
    # every lane's boundaries exist and are the same length as its centreline
    for ln in graph.lanes.values():
        for bid in (ln.left_id, ln.right_id):
            assert len(graph.boundaries[bid].points) == len(ln.centerline)
