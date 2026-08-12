"""Comparing an exported map's traffic lights against a real drive.

Reuses the synthetic four-way junction and ``_light``/``_painted`` helpers
already built for the association/export tests (``test_street_semantics.py``)
rather than re-deriving a scene -- a real exported ``.osm`` still has to come
from ``Lanelet2Writer``, so building one is the only realistic fixture.
"""

from __future__ import annotations

import numpy as np
import pytest
from test_street_semantics import ARMS, STOP_R, _light, _painted, junction  # noqa: F401

from openmap2lanelet.config import PipelineConfig
from openmap2lanelet.export.lanelet2_osm import Lanelet2Writer
from openmap2lanelet.geo import AOI, LocalFrame
from openmap2lanelet.qa.rosbag_validate import read_map_traffic_lights, validate_against_rosbag
from openmap2lanelet.street.geo.camera import Camera, Intrinsics
from openmap2lanelet.street.semantics.associate import associate
from openmap2lanelet.street.sources.base import StreetFrame
from openmap2lanelet.street.types import Landmark, LandmarkKind


@pytest.fixture
def exported_map(junction, tmp_path):  # noqa: F811 -- pytest fixture-name convention
    """Two real map traffic lights, on arms E and N."""
    lights = [_light("t0", "E", ahead=20.0), _light("t1", "N", ahead=20.0)]
    layer = associate(junction, lights, marking=_painted(bar_at=2.0))
    path = Lanelet2Writer(junction, PipelineConfig(), semantics=layer).write(
        tmp_path / "map.osm")
    return path, junction.frame


def _bag_landmark(lid: str, position_xy: tuple[float, float], *, n_views: int = 10,
                  confidence: float = 0.8) -> Landmark:
    return Landmark(id=lid, kind=LandmarkKind.TRAFFIC_LIGHT,
                    position=np.array([position_xy[0], position_xy[1], 5.0]),
                    n_views=n_views, confidence=confidence)


def _looking_camera(at_xy: tuple[float, float], from_xy: tuple[float, float]) -> Camera:
    """A camera at ``from_xy`` pointed at ``at_xy``, optical +z forward."""
    d = np.array([at_xy[0] - from_xy[0], at_xy[1] - from_xy[1], 0.0])
    d = d / np.linalg.norm(d)
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(d, up)
    right = right / np.linalg.norm(right)
    down = np.cross(d, right)
    rot = np.column_stack([right, down, d])
    t = np.eye(4)
    t[:3, :3] = rot
    t[:3, 3] = [from_xy[0], from_xy[1], 1.5]
    intr = Intrinsics(fx=800, fy=800, cx=640, cy=360, width=1280, height=720)
    return Camera("front", intr, t)


def test_read_map_traffic_lights_recovers_both(exported_map):
    path, _frame = exported_map
    lights = read_map_traffic_lights(path)
    assert len(lights) == 2
    assert {round(l.confidence, 1) for l in lights} == {0.8}


def test_map_light_confirmed_by_a_nearby_bag_landmark(exported_map):
    path, frame = exported_map
    map_lights = read_map_traffic_lights(path)
    # place a bag landmark right on top of the first map light's local position
    x, y = frame.to_local(map_lights[0].lon, map_lights[0].lat)
    bag_landmarks = [_bag_landmark("bl0", (float(x), float(y)))]

    result = validate_against_rosbag(path, bag_landmarks, bag_frames=[], map_frame=frame,
                                     match_radius_m=5.0)

    assert result["confirmed"] == 1
    assert result["map_only"] == 1                  # the other map light, unmatched
    assert result["undocumented"] == 0
    confirmed = [it for it in result["items"] if it["kind"] == "traffic_light_confirmed"]
    assert len(confirmed) == 1
    assert confirmed[0]["detail"]["map_id"] == map_lights[0].id


def test_undocumented_signal_is_the_high_severity_case(exported_map):
    path, frame = exported_map
    map_lights = read_map_traffic_lights(path)
    # two bag landmarks confirming both map lights, plus one nowhere near either
    bag_landmarks = []
    for ml in map_lights:
        x, y = frame.to_local(ml.lon, ml.lat)
        bag_landmarks.append(_bag_landmark(f"bl_{ml.id}", (float(x), float(y))))
    bag_landmarks.append(_bag_landmark("bl_extra", (500.0, 500.0)))

    result = validate_against_rosbag(path, bag_landmarks, bag_frames=[], map_frame=frame)

    assert result["confirmed"] == 2
    assert result["undocumented"] == 1
    undoc = [it for it in result["items"] if it["kind"] == "traffic_light_undocumented"]
    assert len(undoc) == 1
    assert undoc[0]["severity"] == "high"
    assert undoc[0]["detail"]["bag_landmark_id"] == "bl_extra"


def test_map_only_severity_escalates_when_the_drive_looked_there(exported_map):
    path, frame = exported_map
    map_lights = read_map_traffic_lights(path)
    x, y = frame.to_local(map_lights[0].lon, map_lights[0].lat)
    unseen_x, unseen_y = frame.to_local(map_lights[1].lon, map_lights[1].lat)

    # a camera positioned to look straight at the *first* light only
    cam = _looking_camera((float(x), float(y)), (float(x) - 30.0, float(y)))
    intr = cam.intrinsics
    frame_obj = StreetFrame(id="cam/0", camera=cam, image_path="unused.png", timestamp_ns=0)

    # both lights sit near the small synthetic junction's centre, so the range
    # gate -- not the camera FOV -- is what keeps the unseen light excluded.
    result = validate_against_rosbag(path, bag_landmarks=[], bag_frames=[frame_obj],
                                     map_frame=frame, in_view_range_m=32.0)

    by_id = {it["detail"]["map_id"]: it for it in result["items"]
            if it["kind"] == "traffic_light_map_only"}
    assert by_id[map_lights[0].id]["severity"] == "medium"     # in view, not confirmed
    assert by_id[map_lights[1].id]["severity"] == "low"        # never in view
    assert intr.width == 1280                                  # sanity: fixture camera is sane


def test_no_map_lights_reports_bag_only_as_undocumented(tmp_path):
    aoi = AOI("empty", -0.001, -0.001, 0.001, 0.001)
    frame = LocalFrame.for_aoi(aoi)
    from openmap2lanelet.types import Boundary, Lane, LaneGraph, MarkingType, Provenance, Source

    g = LaneGraph(aoi=aoi, frame=frame)
    c = np.array([[0.0, 0.0], [10.0, 0.0]])
    g.add_boundary(Boundary(id="bL", points=c + [0, 1.5], marking=MarkingType.SOLID,
                            confidence=0.8, provenance=Provenance(Source.IMAGE)))
    g.add_boundary(Boundary(id="bR", points=c + [0, -1.5], marking=MarkingType.SOLID,
                            confidence=0.8, provenance=Provenance(Source.IMAGE)))
    g.add_lane(Lane(id="l0", left_id="bL", right_id="bR", centerline=c, segment_id="e0",
                    kind="road", index_from_left=0, width=3.0, confidence=0.7,
                    provenance=Provenance(Source.FUSED)))
    path = Lanelet2Writer(g, PipelineConfig()).write(tmp_path / "empty.osm")

    assert read_map_traffic_lights(path) == []
    result = validate_against_rosbag(path, [_bag_landmark("bl0", (5.0, 0.0))],
                                     bag_frames=[], map_frame=frame)
    assert result["map_lights"] == 0
    assert result["undocumented"] == 1
