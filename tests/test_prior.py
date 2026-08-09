import numpy as np

from gmap2lanelet.geo import AOI, LocalFrame
from gmap2lanelet.prior import osm_tags
from gmap2lanelet.prior.road_graph import build_road_prior
from gmap2lanelet.sources.base import PriorData, PriorWay
from gmap2lanelet.types import Source


def test_lane_count_prefers_tag_then_default():
    v = osm_tags.lane_count({"highway": "primary", "lanes": "5"})
    assert v.value == 5 and v.source is Source.OSM

    v = osm_tags.lane_count({"highway": "primary", "lanes:forward": "2",
                             "lanes:backward": "3"})
    assert v.value == 5 and v.source is Source.OSM

    v = osm_tags.lane_count({"highway": "residential"})
    assert v.value == 2 and v.source is Source.DEFAULT

    # a one-way way carries half the class default
    v = osm_tags.lane_count({"highway": "primary", "oneway": "yes"})
    assert v.value == 2 and v.source is Source.DEFAULT


def test_oneway_and_direction_split():
    assert osm_tags.is_oneway({"highway": "primary", "oneway": "yes"}).value is True
    assert osm_tags.is_oneway({"highway": "residential"}).value is False
    assert osm_tags.is_oneway({"highway": "primary", "junction": "roundabout"}).value is True

    assert osm_tags.directional_split({"highway": "primary", "oneway": "yes"}, 3) == (3, 0)
    assert osm_tags.directional_split({"highway": "primary"}, 4) == (2, 2)
    assert osm_tags.directional_split(
        {"highway": "primary", "lanes:forward": "3"}, 5) == (3, 2)


def test_maxspeed_units():
    assert osm_tags.speed_limit_kph({"maxspeed": "50"}).value == 50
    assert osm_tags.speed_limit_kph({"maxspeed": "35 mph"}).value == 56.3
    v = osm_tags.speed_limit_kph({"highway": "residential"})
    assert v.source is Source.DEFAULT


def test_turn_lanes_parsing():
    assert osm_tags.turn_lanes({"highway": "primary"}) is None
    v = osm_tags.turn_lanes({"turn:lanes": "left|through;right|"})
    assert v.value == [{"left"}, {"through", "right"}, {"through"}]


def _way(coords, tags, frame, wid="w"):
    lon, lat = frame.to_wgs84(coords[:, 0], coords[:, 1])
    return PriorWay(wid, np.column_stack([lon, lat]), tags)


def test_road_graph_splits_at_shared_vertices():
    aoi = AOI("t", -0.002, -0.002, 0.002, 0.002)
    frame = LocalFrame.for_aoi(aoi)
    ew = np.column_stack([np.linspace(-100, 100, 41), np.zeros(41)])
    ns = np.column_stack([np.zeros(41), np.linspace(-100, 100, 41)])
    prior = PriorData([_way(ew, {"highway": "primary"}, frame, "a"),
                       _way(ns, {"highway": "primary"}, frame, "b")], "t", "osm")

    rp = build_road_prior(prior, aoi, frame)
    # each way is cut at the crossing, giving four edges around one junction
    assert len(rp.edges) == 4
    assert len(rp.clusters) == 1
    junction = next(iter(rp.clusters.values()))
    assert sum(rp.nodes[n].degree for n in junction.node_ids) == 4


def test_road_graph_clusters_nearby_junctions():
    """A dual carriageway crossing a road makes two OSM nodes, one intersection."""
    aoi = AOI("t", -0.002, -0.002, 0.002, 0.002)
    frame = LocalFrame.for_aoi(aoi)
    ways = []
    for i, y in enumerate((-6.0, 6.0)):                     # two carriageways
        pts = np.column_stack([np.linspace(-100, 100, 41), np.full(41, y)])
        ways.append(_way(pts, {"highway": "primary", "oneway": "yes"}, frame, f"c{i}"))
    # As OSM would have it, the crossing way carries a vertex on each
    # carriageway -- that shared node is what makes it a junction at all.
    ys = np.unique(np.concatenate([np.linspace(-100, 100, 41), [-6.0, 6.0]]))
    ns = np.column_stack([np.zeros_like(ys), ys])
    ways.append(_way(ns, {"highway": "secondary"}, frame, "x"))

    rp = build_road_prior(PriorData(ways, "t", "osm"), aoi, frame)
    assert len(rp.clusters) == 1                            # 12 m apart -> one junction
    assert len(next(iter(rp.clusters.values())).node_ids) == 2
