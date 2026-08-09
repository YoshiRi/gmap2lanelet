"""Camera, triangulation and ground-surface maths.

These run without network or model weights: the geometry is exercised against
synthetic cameras whose answers are known analytically, which is the only way
to tell a genuine bug from a plausible-looking 3-D point.
"""

from __future__ import annotations

import numpy as np
import pytest

from gmap2lanelet.street.geo.camera import Camera, Intrinsics, quat_to_rot, se3, se3_inv
from gmap2lanelet.street.geo.ground import GroundSurface
from gmap2lanelet.street.geo.triangulate import (closest_approach, perpendicular_distances,
                                                 reprojection_errors, triangulate)


def _cam(x: float, y: float, z: float = 1.5, yaw: float = 0.0) -> Camera:
    """A camera at (x, y, z) looking along ``yaw``, optical axis = +z_cam."""
    k = Intrinsics(fx=1000.0, fy=1000.0, cx=960.0, cy=600.0, width=1920, height=1200)
    c, s = np.cos(yaw), np.sin(yaw)
    fwd = np.array([c, s, 0.0])                 # camera +z
    right = np.array([s, -c, 0.0])              # camera +x
    down = np.array([0.0, 0.0, -1.0])           # camera +y
    rot = np.column_stack([right, down, fwd])   # columns are camera axes in world
    return Camera(name="test", intrinsics=k, world_T_cam=se3(rot, np.array([x, y, z])))


def test_se3_roundtrip_and_quaternion():
    r = quat_to_rot(np.cos(0.3), 0.0, 0.0, np.sin(0.3))
    assert np.allclose(r @ r.T, np.eye(3), atol=1e-9)
    assert np.isclose(np.linalg.det(r), 1.0)
    t = se3(r, np.array([3.0, -2.0, 1.0]))
    assert np.allclose(se3_inv(t) @ t, np.eye(4), atol=1e-9)


def test_project_and_ray_are_inverse():
    cam = _cam(0.0, 0.0, 1.5, yaw=0.0)
    p = np.array([[20.0, 1.5, 5.0], [35.0, -3.0, 4.0]])
    uv, ok = cam.project(p)
    assert ok.all()
    d = cam.ray(uv)
    to_p = p - cam.center[None, :]
    to_p /= np.linalg.norm(to_p, axis=1, keepdims=True)
    assert np.allclose(d, to_p, atol=1e-6)


def test_points_behind_the_camera_do_not_project():
    cam = _cam(0.0, 0.0, yaw=0.0)
    _, ok = cam.project(np.array([[-10.0, 0.0, 2.0]]))
    assert not ok[0]


def test_distortion_roundtrip():
    k = Intrinsics(fx=900.0, fy=900.0, cx=800.0, cy=500.0, width=1600, height=1000,
                   k1=-0.28, k2=0.09, k3=-0.01)
    xn = np.array([0.0, 0.35, -0.5])
    yn = np.array([0.0, -0.22, 0.4])
    bx, by = k.undistort(*k.distort(xn, yn))
    assert np.allclose(bx, xn, atol=1e-6)
    assert np.allclose(by, yn, atol=1e-6)


def test_triangulation_recovers_a_known_point():
    target = np.array([30.0, 2.0, 5.2])
    cams = [_cam(x, 0.0) for x in (0.0, 6.0, 12.0, 18.0)]
    origins = np.array([c.center for c in cams])
    dirs = np.array([(target - c.center) / np.linalg.norm(target - c.center) for c in cams])

    tri = triangulate(origins, dirs, inlier_m=0.5)
    assert tri is not None
    assert np.allclose(tri.point, target, atol=1e-6)
    assert tri.n_inliers == 4
    assert tri.baseline_m == pytest.approx(18.0, abs=1e-6)
    assert tri.residual_m < 1e-6


def test_triangulation_rejects_an_outlier_ray():
    target = np.array([25.0, -1.0, 5.0])
    cams = [_cam(x, 0.0) for x in (0.0, 5.0, 10.0, 15.0, 20.0)]
    origins = np.array([c.center for c in cams])
    dirs = []
    for i, c in enumerate(cams):
        aim = target if i != 2 else np.array([25.0, 14.0, 5.0])   # a mismatched head
        v = aim - c.center
        dirs.append(v / np.linalg.norm(v))
    tri = triangulate(origins, np.array(dirs), inlier_m=0.6)
    assert tri is not None
    assert not tri.inliers[2]
    assert np.linalg.norm(tri.point - target) < 0.2


def test_parallel_rays_carry_no_range():
    o1, o2 = np.array([0.0, 0.0, 0.0]), np.array([0.0, 3.0, 0.0])
    d = np.array([1.0, 0.0, 0.0])
    p, gap, _ = closest_approach(o1, d, o2, d)
    assert p is None and not np.isfinite(gap)


def test_perpendicular_distance_and_reprojection():
    cams = [_cam(0.0, 0.0), _cam(10.0, 0.0)]
    target = np.array([28.0, 3.0, 5.0])
    origins = np.array([c.center for c in cams])
    dirs = np.array([(target - o) / np.linalg.norm(target - o) for o in origins])
    assert np.allclose(perpendicular_distances(target, origins, dirs), 0.0, atol=1e-9)

    uvs = np.array([c.project(target[None, :])[0][0] for c in cams])
    assert np.allclose(reprojection_errors(target, cams, uvs), 0.0, atol=1e-6)
    off = reprojection_errors(target + np.array([0.0, 0.6, 0.0]), cams, uvs)
    assert (off > 5.0).all()


def test_ground_surface_elevation_and_ray_intersection():
    # a plane tilted 5% in x, sampled on a 0.5 m grid
    n = 60
    xs = np.arange(n) * 0.5
    z = 0.05 * xs[None, :] * np.ones((n, 1))
    g = GroundSurface(heights=z, x0=0.0, y0=0.0, res=0.5)

    assert g.elevation(np.array([10.0]), np.array([5.0]))[0] == pytest.approx(0.5, abs=1e-6)
    assert g.height_above(np.array([[10.0, 5.0, 4.5]]))[0] == pytest.approx(4.0, abs=1e-6)

    origin = np.array([2.0, 5.0, 3.0])
    d = np.array([1.0, 0.0, -0.5])
    d /= np.linalg.norm(d)
    hit = g.intersect_ray(origin, d)
    assert hit is not None
    assert abs(hit[2] - float(g.elevation(hit[:1], hit[1:2])[0])) < 0.05


def test_ground_surface_reports_unmapped_cells_as_nan():
    g = GroundSurface(heights=np.full((10, 10), np.nan), x0=0.0, y0=0.0, res=1.0)
    assert not np.isfinite(g.elevation(np.array([5.0]), np.array([5.0]))[0])
    assert not np.isfinite(g.height_above(np.array([[5.0, 5.0, 3.0]]))[0])
