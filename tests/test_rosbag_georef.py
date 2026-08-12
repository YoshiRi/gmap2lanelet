"""Georeference strategies for a rosbag's self-localization pose frame."""

from __future__ import annotations

import importlib.util
import math

import numpy as np
import pytest

from openmap2lanelet.geo import LocalFrame
from openmap2lanelet.street.sources.rosbag import LocalCartesianGeoref, MgrsGeoref

needs_mgrs = pytest.mark.skipif(
    importlib.util.find_spec("mgrs") is None, reason="mgrs package not installed")


def test_local_cartesian_to_wgs84_matches_local_frame_when_unrotated():
    georef = LocalCartesianGeoref(lat0=35.6271, lon0=139.7781)
    ref = LocalFrame(35.6271, 139.7781)
    lon, lat = georef.to_wgs84(np.array([100.0]), np.array([50.0]))
    exp_lon, exp_lat = ref.to_wgs84(100.0, 50.0)
    assert lon[0] == pytest.approx(float(exp_lon), abs=1e-9)
    assert lat[0] == pytest.approx(float(exp_lat), abs=1e-9)


def test_local_cartesian_round_trips_through_to_local():
    georef = LocalCartesianGeoref(lat0=35.6271, lon0=139.7781, rotation_rad=math.radians(15))
    frame = LocalFrame(35.6271, 139.7781)
    lon, lat = georef.to_wgs84(np.array([40.0]), np.array([-15.0]))
    xyz = georef.to_local(np.array([[40.0, -15.0, 3.0]]), frame)
    back_x, back_y = frame.to_local(lon, lat)
    assert xyz[0, 0] == pytest.approx(float(back_x[0]), abs=1e-6)
    assert xyz[0, 1] == pytest.approx(float(back_y[0]), abs=1e-6)
    assert xyz[0, 2] == pytest.approx(3.0)


def test_local_cartesian_rotation_changes_the_bearing():
    unrotated = LocalCartesianGeoref(lat0=35.0, lon0=139.0)
    rotated = LocalCartesianGeoref(lat0=35.0, lon0=139.0, rotation_rad=math.pi / 2)
    lon_u, lat_u = unrotated.to_wgs84(np.array([100.0]), np.array([0.0]))
    lon_r, lat_r = rotated.to_wgs84(np.array([100.0]), np.array([0.0]))
    assert not (lon_u[0] == pytest.approx(lon_r[0]) and lat_u[0] == pytest.approx(lat_r[0]))


@needs_mgrs
def test_mgrs_georef_round_trips():
    import mgrs as mgrs_lib

    # A real point near Tokyo Teleport station, converted to MGRS so the test
    # doesn't have to hand-derive a grid zone.
    lat0, lon0 = 35.6271, 139.7781
    mgrs_str = mgrs_lib.MGRS().toMGRS(lat0, lon0, MGRSPrecision=5)
    grid_zone, easting, northing = mgrs_str[:5], mgrs_str[5:10], mgrs_str[10:15]

    georef = MgrsGeoref(grid_zone)
    lon, lat = georef.to_wgs84(np.array([float(easting)]), np.array([float(northing)]))
    assert lat[0] == pytest.approx(lat0, abs=1e-3)
    assert lon[0] == pytest.approx(lon0, abs=1e-3)


@needs_mgrs
def test_mgrs_to_local_matches_frame_to_local():
    lat0, lon0 = 35.6271, 139.7781
    import mgrs as mgrs_lib

    mgrs_str = mgrs_lib.MGRS().toMGRS(lat0, lon0, MGRSPrecision=5)
    grid_zone, easting, northing = mgrs_str[:5], mgrs_str[5:10], mgrs_str[10:15]

    georef = MgrsGeoref(grid_zone)
    frame = LocalFrame(lat0, lon0)
    xyz = georef.to_local(np.array([[float(easting), float(northing), 5.0]]), frame)
    assert xyz[0, 0] == pytest.approx(0.0, abs=2.0)     # within MGRS's ~1m rounding
    assert xyz[0, 1] == pytest.approx(0.0, abs=2.0)
    assert xyz[0, 2] == pytest.approx(5.0)
