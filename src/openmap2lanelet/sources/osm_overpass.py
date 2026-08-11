"""OpenStreetMap road prior via the Overpass API (live) or a local .osm file.

This is the reference prior source: it is what you use for an arbitrary place
on Earth.  ``OverpassSource`` needs outbound access to an Overpass endpoint;
``OsmFileSource`` reads the same data from a downloaded ``.osm`` XML extract
(``https://www.openstreetmap.org/api/0.6/map?bbox=...`` or a JOSM export), so
the pipeline stays usable on machines without direct Overpass access.
"""

from __future__ import annotations

import json
import logging
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from ..geo import AOI
from .base import PriorData, PriorWay
from .cache import post_form

log = logging.getLogger(__name__)

DEFAULT_ENDPOINT = "https://overpass-api.de/api/interpreter"
USER_AGENT = "openmap2lanelet/0.1 (research PoC)"

# Road classes we treat as drivable.  Deliberately excludes footway/cycleway.
DRIVABLE = (
    "motorway|trunk|primary|secondary|tertiary|unclassified|residential|"
    "living_street|service|motorway_link|trunk_link|primary_link|secondary_link|"
    "tertiary_link"
)


class OverpassSource:
    """Fetch drivable highways in a bbox from Overpass."""

    name = "overpass"

    def __init__(self, endpoint: str = DEFAULT_ENDPOINT, use_cache: bool = True,
                 drivable: str = DRIVABLE):
        self.endpoint = endpoint
        self.use_cache = use_cache
        self.drivable = drivable

    def query(self, aoi: AOI) -> str:
        s, w, n, e = aoi.south, aoi.west, aoi.north, aoi.east
        return (
            "[out:json][timeout:120];"
            f'way["highway"~"^({self.drivable})$"]({s},{w},{n},{e});'
            "out body geom;"
        )

    def fetch(self, aoi: AOI) -> PriorData:
        raw = post_form(self.endpoint, {"data": self.query(aoi)},
                        headers={"User-Agent": USER_AGENT}, use_cache=self.use_cache)
        doc = json.loads(raw)
        ways: list[PriorWay] = []
        for el in doc.get("elements", []):
            if el.get("type") != "way" or "geometry" not in el:
                continue
            coords = np.asarray([[p["lon"], p["lat"]] for p in el["geometry"]], dtype=float)
            if len(coords) < 2:
                continue
            ways.append(PriorWay(id=f"osm_w{el['id']}", coords=coords,
                                 tags={str(k): str(v) for k, v in el.get("tags", {}).items()}))
        log.info("overpass: %s drivable ways", len(ways))
        return PriorData(ways=ways, attribution="© OpenStreetMap contributors (ODbL)",
                         kind="osm", detail={"endpoint": self.endpoint})


class OsmFileSource:
    """Read the same prior from an ``.osm`` XML file."""

    name = "osm-file"

    def __init__(self, path: str | Path, drivable: str = DRIVABLE):
        self.path = Path(path)
        self.allowed = set(drivable.split("|"))

    def fetch(self, aoi: AOI) -> PriorData:
        root = ET.parse(self.path).getroot()
        nodes = {
            n.get("id"): (float(n.get("lon")), float(n.get("lat")))
            for n in root.findall("node")
            if n.get("lon") is not None and n.get("lat") is not None
        }
        ways: list[PriorWay] = []
        for w in root.findall("way"):
            tags = {t.get("k"): t.get("v") for t in w.findall("tag")}
            if tags.get("highway") not in self.allowed:
                continue
            refs = [nd.get("ref") for nd in w.findall("nd")]
            coords = np.asarray([nodes[r] for r in refs if r in nodes], dtype=float)
            if len(coords) < 2:
                continue
            ways.append(PriorWay(id=f"osm_w{w.get('id')}", coords=coords, tags=tags))
        log.info("osm file %s: %s drivable ways", self.path.name, len(ways))
        return PriorData(ways=ways, attribution="© OpenStreetMap contributors (ODbL)",
                         kind="osm", detail={"path": str(self.path)})
