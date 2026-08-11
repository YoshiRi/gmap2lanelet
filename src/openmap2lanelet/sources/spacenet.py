"""SpaceNet-3 (roads) as a public imagery + OSM-like prior source.

Why this source exists
----------------------
The reference "live" configuration of this PoC is *Overpass + XYZ aerial
tiles* (see ``osm_overpass.py`` / ``xyz_tiles.py``).  SpaceNet is a second,
fully public, **anonymously downloadable** source that provides both halves of
the problem for the same footprint:

* ``PS-RGB``    - pansharpened WorldView-3, ~0.3 m GSD, 1300x1300 px
                  (~300 x 390 m -- exactly the "few hundred metres" scope);
* ``geojson_roads`` - hand-digitised road *centrelines* with OSM-equivalent
                  attributes (road class, lane count, one-way, paved).

The centrelines play exactly the role OSM plays: a topologically sound
centreline graph with coarse geometry and sparse lane metadata, and no lane
geometry at all.  Attribute names are translated to real OSM tag keys on
import, so everything downstream is written against OSM semantics only.

Licence: SpaceNet imagery is CC-BY-SA 4.0 (Maxar / SpaceNet LLC).
"""

from __future__ import annotations

import json
import logging

import numpy as np

from ..geo import AOI, LocalFrame
from ..raster import GeoRaster
from .base import ImageryData, PriorData, PriorWay
from .cache import cached_path, fetch_bytes

log = logging.getLogger(__name__)

S3_ROOT = "https://s3.amazonaws.com/spacenet-dataset/spacenet/SN3_roads/train"

# SpaceNet road_type code -> OSM highway value
ROAD_TYPE_TO_HIGHWAY = {
    "1": "motorway",
    "2": "primary",
    "3": "secondary",
    "4": "tertiary",
    "5": "residential",
    "6": "unclassified",
    "7": "track",
}


class SpaceNetSource:
    """Imagery *and* prior for one SpaceNet tile."""

    name = "spacenet"

    def __init__(self, aoi_name: str = "AOI_2_Vegas", image_id: int = 93,
                 use_cache: bool = True):
        self.aoi_name = aoi_name
        self.image_id = int(image_id)
        self.use_cache = use_cache
        self._tif: bytes | None = None

    # -- urls ---------------------------------------------------------------

    @property
    def _stem(self) -> str:
        return f"SN3_roads_train_{self.aoi_name}"

    @property
    def image_url(self) -> str:
        return f"{S3_ROOT}/{self.aoi_name}/PS-RGB/{self._stem}_PS-RGB_img{self.image_id}.tif"

    @property
    def roads_url(self) -> str:
        return (f"{S3_ROOT}/{self.aoi_name}/geojson_roads/"
                f"{self._stem}_geojson_roads_img{self.image_id}.geojson")

    @property
    def speed_url(self) -> str:
        return (f"{S3_ROOT}/{self.aoi_name}/geojson_roads_speed/"
                f"{self._stem}_geojson_roads_speed_img{self.image_id}.geojson")

    # -- fetching -----------------------------------------------------------

    def _tif_path(self):
        p = cached_path(self.image_url)
        if not (self.use_cache and p.exists() and p.stat().st_size > 0):
            data = fetch_bytes(self.image_url, use_cache=self.use_cache)
            p.write_bytes(data)
        return p

    def aoi(self) -> AOI:
        """AOI = the footprint of the imagery tile."""
        import rasterio

        with rasterio.open(self._tif_path()) as src:
            b = src.bounds
            if src.crs is not None and src.crs.to_epsg() != 4326:
                from rasterio.warp import transform_bounds
                w, s, e, n = transform_bounds(src.crs, "EPSG:4326", *b)
            else:
                w, s, e, n = b.left, b.bottom, b.right, b.top
        return AOI(f"{self.aoi_name}_img{self.image_id}", w, s, e, n)

    def fetch_imagery(self, aoi: AOI, frame: LocalFrame) -> ImageryData:
        import rasterio
        from rasterio.warp import transform_bounds

        with rasterio.open(self._tif_path()) as src:
            arr = src.read()                                    # (C, H, W)
            b = src.bounds
            if src.crs is not None and src.crs.to_epsg() != 4326:
                w, s, e, n = transform_bounds(src.crs, "EPSG:4326", *b)
            else:
                w, s, e, n = b.left, b.bottom, b.right, b.top

        rgb = _stretch(arr)
        raster = GeoRaster.from_bounds(rgb, frame, w, s, e, n, name="aerial")
        valid = float((arr.sum(axis=0) > 0).mean())
        log.info("spacenet imagery %s: %sx%s px, gsd=%.2f m, valid=%.1f%%",
                 self.image_id, raster.shape[1], raster.shape[0], raster.gsd, 100 * valid)
        return ImageryData(
            raster=raster,
            attribution="SpaceNet / Maxar WorldView-3 (CC BY-SA 4.0)",
            gsd=raster.gsd,
            detail={"url": self.image_url, "valid_fraction": round(valid, 4),
                    "sensor": "WorldView-3 pansharpened RGB"},
        )

    def fetch_prior(self, aoi: AOI) -> PriorData:
        gj = json.loads(fetch_bytes(self.roads_url, use_cache=self.use_cache))
        speeds = self._speed_lookup()
        ways: list[PriorWay] = []
        for i, feat in enumerate(gj.get("features", [])):
            geom = feat.get("geometry") or {}
            if geom.get("type") != "LineString":
                continue
            coords = np.asarray(geom["coordinates"], dtype=float)[:, :2]
            if len(coords) < 2:
                continue
            props = feat.get("properties", {})
            ways.append(PriorWay(id=f"w{props.get('road_id', i)}_{i}", coords=coords,
                                 tags=_to_osm_tags(props, speeds)))
        log.info("spacenet prior: %s ways", len(ways))
        return PriorData(ways=ways, attribution="SpaceNet road labels (CC BY-SA 4.0)",
                         kind="osm-like", detail={"url": self.roads_url})

    def _speed_lookup(self) -> dict[int, float]:
        """road_id -> speed limit (km/h) from the SpaceNet-5 style speed labels."""
        try:
            gj = json.loads(fetch_bytes(self.speed_url, use_cache=self.use_cache))
        except Exception as exc:                                # noqa: BLE001
            log.debug("no speed labels for img%s (%s)", self.image_id, exc)
            return {}
        out: dict[int, float] = {}
        for feat in gj.get("features", []):
            p = feat.get("properties", {})
            rid = p.get("road_id")
            v = p.get("inferred_speed_kph", p.get("speed_m_s"))
            if rid is None or v is None:
                continue
            v = float(v)
            if "speed_m_s" in p and "inferred_speed_kph" not in p:
                v *= 3.6
            out[int(rid)] = v
        return out


def _to_osm_tags(props: dict, speeds: dict[int, float]) -> dict[str, str]:
    """Translate SpaceNet road attributes into real OSM tag keys."""
    tags: dict[str, str] = {}
    tags["highway"] = ROAD_TYPE_TO_HIGHWAY.get(str(props.get("road_type", "")), "unclassified")

    lanes = props.get("lane_number", props.get("lane_numbe"))
    if lanes not in (None, "", "0"):
        try:
            n = int(float(lanes))
            if n > 0:
                tags["lanes"] = str(n)
        except (TypeError, ValueError):
            pass

    # SpaceNet uses 1 = yes, 2 = no for boolean-ish codes.
    if str(props.get("one_way_ty", "")) == "1":
        tags["oneway"] = "yes"
    if str(props.get("paved", "")) == "2":
        tags["surface"] = "unpaved"
    if str(props.get("bridge_typ", "")) == "1":
        tags["bridge"] = "yes"

    rid = props.get("road_id")
    if rid is not None and int(rid) in speeds:
        tags["maxspeed"] = f"{speeds[int(rid)]:.0f}"
    return tags


def _stretch(arr: np.ndarray) -> np.ndarray:
    """uint16 multispectral -> display uint8 RGB with a robust per-band stretch."""
    arr = arr.astype(np.float32)
    if arr.shape[0] < 3:
        arr = np.repeat(arr[:1], 3, axis=0)
    bands = []
    for b in arr[:3]:
        valid = b[b > 0]
        if valid.size < 100:
            bands.append(np.zeros_like(b))
            continue
        lo, hi = np.percentile(valid, (1.0, 99.0))
        if hi - lo < 1e-6:
            hi = lo + 1.0
        bands.append(np.clip((b - lo) / (hi - lo), 0, 1))
    return (np.stack(bands, axis=-1) * 255).astype(np.uint8)
