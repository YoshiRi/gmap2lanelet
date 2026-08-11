import numpy as np
import pytest

from openmap2lanelet.geo import (
    AOI,
    LocalFrame,
    hermite,
    offset_polyline,
    polyline_length,
    resample_polyline,
    simplify_polyline,
)
from openmap2lanelet.raster import GeoRaster


def test_local_frame_roundtrip():
    aoi = AOI("t", 139.760, 35.680, 139.765, 35.684)
    f = LocalFrame.for_aoi(aoi)
    lon, lat = np.array([139.761, 139.7645]), np.array([35.6812, 35.6836])
    x, y = f.to_local(lon, lat)
    lon2, lat2 = f.to_wgs84(x, y)
    assert np.allclose(lon, lon2, atol=1e-12)
    assert np.allclose(lat, lat2, atol=1e-12)


def test_local_frame_scale_is_metric():
    f = LocalFrame(35.68, 139.76)
    # 0.001 deg of latitude is ~111.3 m anywhere
    _, y = f.to_local(139.76, 35.681)
    assert 110.0 < float(y) < 112.5
    # longitude shrinks by cos(lat)
    x, _ = f.to_local(139.761, 35.68)
    assert 89.0 < float(x) < 91.5


def test_resample_and_length():
    pts = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]])
    assert polyline_length(pts) == pytest.approx(20.0)
    r = resample_polyline(pts, 1.0)
    steps = np.hypot(*np.diff(r, axis=0).T)
    assert np.allclose(steps, steps[0], atol=1e-6)
    assert polyline_length(r) == pytest.approx(20.0, abs=0.2)


def test_offset_polyline_is_parallel():
    pts = np.column_stack([np.linspace(0, 50, 51), np.zeros(51)])
    left = offset_polyline(pts, 3.0)
    assert np.allclose(left[:, 1], 3.0, atol=1e-9)   # +offset is to the left
    right = offset_polyline(pts, -3.0)
    assert np.allclose(right[:, 1], -3.0, atol=1e-9)


def test_hermite_matches_end_poses():
    p0, p1 = np.array([0.0, 0.0]), np.array([20.0, 20.0])
    c = hermite(p0, 0.0, p1, np.pi / 2, n=2001)
    assert np.allclose(c[0], p0)
    assert np.allclose(c[-1], p1)
    # finite differences approach the analytic end tangents as n grows
    assert np.arctan2(*(c[1] - c[0])[::-1]) == pytest.approx(0.0, abs=2e-3)
    assert np.arctan2(*(c[-1] - c[-2])[::-1]) == pytest.approx(np.pi / 2, abs=2e-3)


def test_simplify_keeps_shape():
    pts = np.column_stack([np.linspace(0, 100, 401), np.zeros(401)])
    s = simplify_polyline(pts, 0.1)
    assert len(s) == 2
    assert np.allclose(s[0], pts[0]) and np.allclose(s[-1], pts[-1])


def test_raster_pixel_world_roundtrip_and_sampling():
    data = np.zeros((40, 60), dtype=np.float32)
    data[10, 20] = 1.0
    r = GeoRaster(data, x0=-30.0, y0=20.0, dx=0.5, dy=0.5)
    x, y = r.pixel_to_world(20.0, 10.0)
    c, row = r.world_to_pixel(x, y)
    assert c == pytest.approx(20.0) and row == pytest.approx(10.0)
    assert float(r.sample(x, y)) == pytest.approx(1.0)
    # halfway to a zero neighbour -> half the value
    x2, _ = r.pixel_to_world(20.5, 10.0)
    assert float(r.sample(x2, y)) == pytest.approx(0.5, abs=1e-6)
    assert bool(r.contains(x, y)) and not bool(r.contains(x - 1000, y))
    assert r.gsd == pytest.approx(0.5)
