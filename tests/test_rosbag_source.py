"""Pure helpers in street/sources/rosbag.py -- no real rosbag/rosbags object
ever touches these tests, matching how YoloDetector itself is never invoked
in the existing detector tests."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from openmap2lanelet.street.geo.camera import quat_to_rot, se3
from openmap2lanelet.street.sources.base import StreetFrame
from openmap2lanelet.street.sources.rosbag import (
    RawBox,
    _decode_common_pose,
    _nearest,
    boxes_to_detections,
    compose_camera_pose,
    lookup_static_transform,
)
from openmap2lanelet.street.types import LandmarkKind


def _quat(w, x, y, z):
    return SimpleNamespace(w=w, x=x, y=y, z=z)


def _point(x, y, z):
    return SimpleNamespace(x=x, y=y, z=z)


# -- pose decoding ------------------------------------------------------------


def test_decode_pose_stamped():
    msg = SimpleNamespace(pose=SimpleNamespace(
        position=_point(1.0, 2.0, 3.0), orientation=_quat(1.0, 0.0, 0.0, 0.0)))
    xyz, quat_wxyz = _decode_common_pose(msg)
    assert np.allclose(xyz, [1.0, 2.0, 3.0])
    assert np.allclose(quat_wxyz, [1.0, 0.0, 0.0, 0.0])


def test_decode_odometry_unwraps_extra_pose_nesting():
    msg = SimpleNamespace(pose=SimpleNamespace(pose=SimpleNamespace(
        position=_point(4.0, 5.0, 6.0), orientation=_quat(0.7071, 0.0, 0.0, 0.7071))))
    xyz, quat_wxyz = _decode_common_pose(msg)
    assert np.allclose(xyz, [4.0, 5.0, 6.0])
    assert np.allclose(quat_wxyz, [0.7071, 0.0, 0.0, 0.7071])


def test_decode_pose_with_covariance_stamped_same_shape_as_odometry():
    msg = SimpleNamespace(pose=SimpleNamespace(pose=SimpleNamespace(
        position=_point(0.0, 0.0, 0.0), orientation=_quat(1.0, 0.0, 0.0, 0.0))))
    xyz, quat_wxyz = _decode_common_pose(msg)
    assert np.allclose(xyz, [0.0, 0.0, 0.0])
    assert np.allclose(quat_wxyz, [1.0, 0.0, 0.0, 0.0])


def test_decode_pose_reorders_ros_xyzw_into_wxyz():
    # A 90deg yaw in ROS's (x, y, z, w) convention.
    msg = SimpleNamespace(pose=SimpleNamespace(
        position=_point(0.0, 0.0, 0.0), orientation=_quat(w=0.7071068, x=0.0, y=0.0, z=0.7071068)))
    _, quat_wxyz = _decode_common_pose(msg)
    rot = quat_to_rot(*quat_wxyz)
    # rotates +x to +y
    assert np.allclose(rot @ [1, 0, 0], [0, 1, 0], atol=1e-4)


# -- SE3 composition ------------------------------------------------------------


def test_compose_camera_pose_is_matrix_product():
    a = se3(np.eye(3), np.array([1.0, 0.0, 0.0]))
    b = se3(np.eye(3), np.array([0.0, 2.0, 0.0]))
    out = compose_camera_pose(a, b)
    assert np.allclose(out[:3, 3], [1.0, 2.0, 0.0])


# -- static TF lookup ------------------------------------------------------------


def test_lookup_static_transform_direct_edge():
    t = se3(np.eye(3), np.array([0.1, 0.2, 0.3]))
    transforms = {("base_link", "camera_front"): t}
    out = lookup_static_transform(transforms, "base_link", "camera_front")
    assert np.allclose(out, t)


def test_lookup_static_transform_through_intermediate_frame():
    a = se3(np.eye(3), np.array([1.0, 0.0, 0.0]))       # base_link -> sensor_mount
    b = se3(np.eye(3), np.array([0.0, 0.5, 0.0]))        # sensor_mount -> camera_front
    transforms = {("base_link", "sensor_mount"): a, ("sensor_mount", "camera_front"): b}
    out = lookup_static_transform(transforms, "base_link", "camera_front")
    assert np.allclose(out[:3, 3], [1.0, 0.5, 0.0])


def test_lookup_static_transform_inverts_reversed_edges():
    t = se3(np.eye(3), np.array([2.0, 0.0, 0.0]))
    transforms = {("camera_front", "base_link"): t}      # edge recorded the other way round
    out = lookup_static_transform(transforms, "base_link", "camera_front")
    assert np.allclose(out[:3, 3], [-2.0, 0.0, 0.0])


def test_lookup_static_transform_same_frame_is_identity():
    out = lookup_static_transform({}, "base_link", "base_link")
    assert np.allclose(out, np.eye(4))


def test_lookup_static_transform_missing_chain_returns_none():
    out = lookup_static_transform({}, "base_link", "camera_front")
    assert out is None


# -- detection conversion ------------------------------------------------------------


def test_boxes_to_detections():
    from openmap2lanelet.street.geo.camera import Camera, Intrinsics

    intr = Intrinsics(fx=500, fy=500, cx=320, cy=240, width=640, height=480)
    frame = StreetFrame(id="cam/123", camera=Camera("cam", intr, np.eye(4)),
                        image_path="unused.png", timestamp_ns=123)
    boxes = [RawBox(kind=LandmarkKind.TRAFFIC_LIGHT, bbox=(1.0, 2.0, 3.0, 4.0), score=0.9)]
    dets = boxes_to_detections(frame, boxes)
    assert len(dets) == 1
    assert dets[0].frame_id == "cam/123"
    assert dets[0].bbox == (1.0, 2.0, 3.0, 4.0)
    assert dets[0].camera == "cam"


# -- nearest-timestamp matching ------------------------------------------------------------


def test_nearest_within_tolerance():
    ts = np.array([100, 200, 300], dtype=np.int64)
    assert _nearest(ts, 190, tolerance_ns=50) == 1
    assert _nearest(ts, 260, tolerance_ns=50) == 2


def test_nearest_outside_tolerance_is_none():
    ts = np.array([100, 200, 300], dtype=np.int64)
    assert _nearest(ts, 500, tolerance_ns=50) is None


def test_nearest_empty_is_none():
    assert _nearest(np.array([], dtype=np.int64), 100, tolerance_ns=50) is None


@pytest.mark.parametrize("t,expected", [(100, 0), (300, 2), (150, 0), (151, 1)])
def test_nearest_picks_closer_neighbour(t, expected):
    ts = np.array([100, 200, 300], dtype=np.int64)
    assert _nearest(ts, t, tolerance_ns=100) == expected
