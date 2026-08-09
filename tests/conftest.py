"""Synthetic scene used by the network-free tests.

We render a small aerial-like image of a known road (asphalt, painted lines,
kerbs, some occlusion) together with a *deliberately wrong* prior, so the tests
can assert that the pipeline recovers the geometry the prior did not have.
"""

from __future__ import annotations

import numpy as np
import pytest

from gmap2lanelet.geo import AOI, LocalFrame
from gmap2lanelet.raster import GeoRaster
from gmap2lanelet.sources.base import ImageryData, PriorData, PriorWay

GSD = 0.25


@pytest.fixture(scope="session")
def scene():
    return build_scene()


def build_scene(*, lanes: int = 4, lane_w: float = 3.5, prior_shift: float = 3.0,
                length: float = 200.0, occlude: bool = True, paint: bool = True,
                seed: int = 7):
    """A straight east-west road with ``lanes`` lanes and a shifted prior.

    Returns ``(aoi, frame, imagery, prior, truth)``.
    """
    rng = np.random.default_rng(seed)
    aoi = AOI("synthetic", -0.0015, -0.0008, 0.0015, 0.0008)
    frame = LocalFrame.for_aoi(aoi)

    half_w, half_h = length / 2, 45.0
    w = int(2 * half_w / GSD)
    h = int(2 * half_h / GSD)
    xs = np.linspace(-half_w, half_w, w)
    ys = np.linspace(half_h, -half_h, h)
    X, Y = np.meshgrid(xs, ys)

    road_half = lanes * lane_w / 2
    on_road = np.abs(Y) <= road_half

    # Colours matter: the pavement model separates surfaces mainly by
    # chromaticity, so the scene has to be chromatically realistic -- warm dry
    # ground, neutral-to-cool asphalt -- and not a grey-on-grey cartoon.
    img = np.zeros((h, w, 3), dtype=np.float32)
    img[:] = np.array([150.0, 128.0, 100.0])                        # dry ground
    img += rng.normal(0, 6, (h, w, 1))
    img[on_road] = np.array([52.0, 54.0, 60.0])                     # asphalt
    img[on_road] += rng.normal(0, 3.5, (h, w, 3))[on_road]

    # Paint is 15 cm wide on a 25 cm grid, i.e. *sub-pixel* -- exactly the
    # regime of 0.3 m satellite imagery.  Rendering it as a hard mask would
    # either miss it entirely (if no sample falls inside the stripe) or make it
    # a full-brightness line the detector could never fail on, so it is
    # rasterised by area coverage.
    PAINT = np.array([208.0, 205.0, 196.0])
    truth_boundaries = [-road_half + i * lane_w for i in range(lanes + 1)]
    for i, off in enumerate(truth_boundaries):
        if i in (0, lanes) or not paint:
            continue                                               # kerb, not paint
        cov = np.clip((0.075 + GSD / 2 - np.abs(Y - off)) / GSD, 0.0, 1.0)
        if i != lanes // 2:                                         # dashed, 3 m in 12 m
            cov = cov * ((np.abs(X) % 12.0) < 3.0)
        img = img * (1 - cov[:, :, None]) + PAINT * cov[:, :, None]

    if occlude:                                                     # a tree shadow
        blob = ((X - 40) ** 2 / 90 + (Y - 4) ** 2 / 26) < 1
        img[blob] *= 0.45
        img[blob, 1] += 25                                          # greenish canopy

    img = np.clip(img, 0, 255).astype(np.uint8)
    raster = GeoRaster.from_bounds(img, frame, aoi.west, aoi.south, aoi.east, aoi.north,
                                   name="aerial")
    # from_bounds derives the scale from the AOI; rebuild it on the exact metric
    # grid the scene was drawn on so the test geometry is unambiguous.
    raster = GeoRaster(img, -half_w, half_h, GSD, GSD, "aerial")

    imagery = ImageryData(raster=raster, attribution="synthetic", gsd=GSD,
                          detail={"synthetic": True})

    # the prior: right shape, wrong position, and only a nominal lane count
    px = np.linspace(-half_w + 5, half_w - 5, 12)
    py = np.full_like(px, prior_shift)
    lon, lat = frame.to_wgs84(px, py)
    prior = PriorData(
        ways=[PriorWay("w1", np.column_stack([lon, lat]),
                       {"highway": "secondary", "lanes": str(lanes), "oneway": "no"})],
        attribution="synthetic", kind="osm-like")

    truth = {"lanes": lanes, "lane_w": lane_w, "road_half": road_half,
             "boundaries": truth_boundaries, "prior_shift": prior_shift,
             "center_y": 0.0}
    return aoi, frame, imagery, prior, truth


def build_cross_scene(seed: int = 3):
    """Two roads crossing at right angles, for intersection tests."""
    aoi = AOI("cross", -0.0012, -0.0012, 0.0012, 0.0012)
    frame = LocalFrame.for_aoi(aoi)
    rng = np.random.default_rng(seed)

    half = 60.0
    n = int(2 * half / GSD)
    xs = np.linspace(-half, half, n)
    ys = np.linspace(half, -half, n)
    X, Y = np.meshgrid(xs, ys)

    road = (np.abs(Y) <= 7.0) | (np.abs(X) <= 7.0)
    img = np.zeros((n, n, 3), dtype=np.float32)
    img[:] = np.array([150.0, 128.0, 100.0])
    img += rng.normal(0, 5, (n, n, 1))
    img[road] = np.array([52.0, 54.0, 60.0]) + rng.normal(0, 3, (n, n, 3))[road]
    inside_junction = (np.abs(X) < 8) & (np.abs(Y) < 8)
    for d in (Y, X):
        cov = np.clip((0.075 + GSD / 2 - np.abs(d)) / GSD, 0.0, 1.0)
        cov = cov * road * ~inside_junction
        img = img * (1 - cov[:, :, None]) + np.array([208.0, 205.0, 196.0]) * cov[:, :, None]
    img = np.clip(img, 0, 255).astype(np.uint8)

    raster = GeoRaster(img, -half, half, GSD, GSD, "aerial")
    imagery = ImageryData(raster=raster, attribution="synthetic", gsd=GSD, detail={})

    ways = []
    for i, (a, b) in enumerate(([(-55, 0), (55, 0)], [(0, -55), (0, 55)])):
        # 23 vertices puts one exactly on the crossing, which is how OSM
        # represents a junction: two ways sharing a node.
        pts = np.linspace(a, b, 23)
        lon, lat = frame.to_wgs84(pts[:, 0], pts[:, 1])
        ways.append(PriorWay(f"w{i}", np.column_stack([lon, lat]),
                             {"highway": "tertiary", "lanes": "2", "oneway": "no"}))
    return aoi, frame, imagery, PriorData(ways, "synthetic", "osm-like"), {}
