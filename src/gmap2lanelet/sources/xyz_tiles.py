"""Aerial/satellite imagery from a public XYZ tile service.

This is the reference *live* imagery source: point it at any slippy-map tile
endpoint you are licensed to use and it mosaics the tiles covering the AOI into
a single georeferenced raster in the local metric frame.

No endpoint is hard-coded to a default that would imply permission you may not
have -- pass the template (and honour its terms) explicitly, e.g.::

    XYZTileSource(
        "https://server.arcgisonline.com/ArcGIS/rest/services/"
        "World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attribution="Esri World Imagery",
    )

Zoom 19 is ~0.3 m/px at the equator and ~0.22 m/px at 45 deg latitude, i.e. the
same order as the SpaceNet imagery used by the offline configuration.
"""

from __future__ import annotations

import io
import logging
import math

import numpy as np

from ..geo import AOI, LocalFrame
from ..raster import GeoRaster
from .base import ImageryData
from .cache import fetch_bytes

log = logging.getLogger(__name__)

TILE = 256


def deg2tile(lon: float, lat: float, z: int) -> tuple[float, float]:
    n = 2.0**z
    x = (lon + 180.0) / 360.0 * n
    lat_r = math.radians(lat)
    y = (1.0 - math.asinh(math.tan(lat_r)) / math.pi) / 2.0 * n
    return x, y


def tile2deg(x: float, y: float, z: int) -> tuple[float, float]:
    n = 2.0**z
    lon = x / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    return lon, lat


class XYZTileSource:
    name = "xyz"

    def __init__(self, url_template: str, attribution: str, zoom: int = 19,
                 user_agent: str = "gmap2lanelet/0.1 (research PoC)",
                 use_cache: bool = True, max_tiles: int = 400):
        self.url_template = url_template
        self.attribution = attribution
        self.zoom = int(zoom)
        self.headers = {"User-Agent": user_agent}
        self.use_cache = use_cache
        self.max_tiles = max_tiles

    def fetch(self, aoi: AOI, frame: LocalFrame) -> ImageryData:
        from PIL import Image

        z = self.zoom
        x0f, y0f = deg2tile(aoi.west, aoi.north, z)
        x1f, y1f = deg2tile(aoi.east, aoi.south, z)
        tx0, ty0 = int(math.floor(x0f)), int(math.floor(y0f))
        tx1, ty1 = int(math.floor(x1f)), int(math.floor(y1f))
        nx, ny = tx1 - tx0 + 1, ty1 - ty0 + 1
        if nx * ny > self.max_tiles:
            raise ValueError(f"AOI needs {nx * ny} tiles at z={z}; raise max_tiles or lower zoom")

        mosaic = np.zeros((ny * TILE, nx * TILE, 3), dtype=np.uint8)
        missing = 0
        for j, ty in enumerate(range(ty0, ty1 + 1)):
            for i, tx in enumerate(range(tx0, tx1 + 1)):
                url = self.url_template.format(z=z, x=tx, y=ty)
                try:
                    img = Image.open(io.BytesIO(
                        fetch_bytes(url, headers=self.headers, use_cache=self.use_cache)
                    )).convert("RGB")
                    mosaic[j * TILE:(j + 1) * TILE, i * TILE:(i + 1) * TILE] = np.asarray(img)
                except Exception as exc:                        # noqa: BLE001
                    missing += 1
                    log.warning("tile %s/%s/%s missing: %s", z, tx, ty, exc)

        # Mosaic footprint (whole tiles), then crop to the AOI.
        w_lon, n_lat = tile2deg(tx0, ty0, z)
        e_lon, s_lat = tile2deg(tx1 + 1, ty1 + 1, z)
        raster = GeoRaster.from_bounds(mosaic, frame, w_lon, s_lat, e_lon, n_lat, name="aerial")
        raster = _crop_to_aoi(raster, aoi, frame)

        log.info("xyz mosaic z=%s: %sx%s px, gsd=%.2f m (%s tiles, %s missing)",
                 z, raster.shape[1], raster.shape[0], raster.gsd, nx * ny, missing)
        return ImageryData(raster=raster, attribution=self.attribution, gsd=raster.gsd,
                           detail={"zoom": z, "tiles": nx * ny, "missing_tiles": missing,
                                   "template": self.url_template})


def _crop_to_aoi(raster: GeoRaster, aoi: AOI, frame: LocalFrame) -> GeoRaster:
    (xw, xe), (ys, yn) = frame.to_local([aoi.west, aoi.east], [aoi.south, aoi.north])
    c0, r0 = raster.world_to_pixel(xw, yn)
    c1, r1 = raster.world_to_pixel(xe, ys)
    h, w = raster.shape
    c0, r0 = max(0, int(math.floor(c0))), max(0, int(math.floor(r0)))
    c1, r1 = min(w, int(math.ceil(c1))), min(h, int(math.ceil(r1)))
    if c1 - c0 < 2 or r1 - r0 < 2:
        return raster
    x0, y0 = raster.pixel_to_world(c0, r0)
    return GeoRaster(raster.data[r0:r1, c0:c1], float(x0), float(y0),
                     raster.dx, raster.dy, raster.name)
