"""Argoverse 2 sensor logs as a posed street-level imagery source.

Why this source
---------------
The reference street-level source for an arbitrary place is Mapillary (see
``mapillary.py``), but it needs an API token and its poses come from SfM with
no published uncertainty.  Argoverse 2's sensor dataset is **anonymously
downloadable from S3** and gives, for the same drive:

* seven calibrated ring cameras with intrinsics *and* extrinsics,
* 6-DoF ego poses in a metric city frame at high rate,
* a per-log **ground-height raster** -- real 3-D geometry, not a flat plane,
* an HD map (lane segments with boundaries, mark types and intersection flags)
  which this project uses **only** as evaluation ground truth and, in degraded
  form, as an OSM stand-in prior.

City coordinates are metric offsets from a published per-city origin, so they
convert exactly to WGS84 through UTM.

Licence: Argoverse 2 is released under CC BY-NC-SA 4.0 (Argo AI / Argoverse).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from ...geo import AOI, LocalFrame
from ..geo.camera import Camera, Intrinsics, pose_to_local_frame, quat_to_rot, se3
from ..geo.ground import GroundSurface
from .base import StreetFrame, StreetSequence

log = logging.getLogger(__name__)

S3_ROOT = "https://s3.amazonaws.com/argoverse/datasets/av2/sensor"

# (latitude, longitude) of each city's coordinate-frame origin, and its UTM zone
CITY_ORIGIN = {
    "ATX": (30.27464237939507, -97.7404457407424, 14),
    "DTW": (42.29993066912924, -83.17555750783717, 17),
    "MIA": (25.77452579915163, -80.19656914449405, 17),
    "PAO": (37.416065, -122.13571963362166, 10),
    "PIT": (40.44177902989321, -80.01294377242584, 17),
    "WDC": (38.889377, -77.0355047439081, 18),
}

RING_CAMERAS = ("ring_front_center", "ring_front_left", "ring_front_right",
                "ring_side_left", "ring_side_right",
                "ring_rear_left", "ring_rear_right")


class CityGeoref:
    """AV2 city metric frame <-> WGS84 <-> the pipeline's local metric frame."""

    def __init__(self, city: str):
        from pyproj import Proj

        if city not in CITY_ORIGIN:
            raise ValueError(f"unknown AV2 city {city!r}")
        lat0, lon0, zone = CITY_ORIGIN[city]
        self.city = city
        self.proj = Proj(proj="utm", zone=zone, ellps="WGS84", datum="WGS84",
                         units="m", south=False)
        self.origin_utm = np.array(self.proj(lon0, lat0), dtype=float)

    def city_to_wgs84(self, x, y):
        e = np.asarray(x, dtype=float) + self.origin_utm[0]
        n = np.asarray(y, dtype=float) + self.origin_utm[1]
        lon, lat = self.proj(e, n, inverse=True)
        return np.asarray(lon), np.asarray(lat)

    def city_to_local(self, xyz: np.ndarray, frame: LocalFrame) -> np.ndarray:
        """City (N, 3) -> local metric (N, 3); z is passed through unchanged."""
        p = np.atleast_2d(np.asarray(xyz, dtype=float))
        lon, lat = self.city_to_wgs84(p[:, 0], p[:, 1])
        lx, ly = frame.to_local(lon, lat)
        z = p[:, 2] if p.shape[1] > 2 else np.zeros(len(p))
        return np.column_stack([lx, ly, z])

    def aoi_for(self, xy: np.ndarray, margin: float = 40.0) -> AOI:
        p = np.atleast_2d(np.asarray(xy, dtype=float))
        lon, lat = self.city_to_wgs84(
            [p[:, 0].min() - margin, p[:, 0].max() + margin],
            [p[:, 1].min() - margin, p[:, 1].max() + margin])
        return AOI(f"av2_{self.city}", float(lon[0]), float(lat[0]),
                   float(lon[1]), float(lat[1]))


class AV2LogSource:
    """One Argoverse 2 sensor log."""

    name = "argoverse2"

    def __init__(self, log_id: str, city: str, split: str = "val",
                 cameras: tuple[str, ...] = ("ring_front_center",),
                 stride: int = 3, max_frames: int = 200, cache_dir: str | None = None):
        self.log_id = log_id
        self.city = city
        self.split = split
        self.cameras = cameras
        self.stride = stride
        self.max_frames = max_frames
        self.georef = CityGeoref(city)
        base = Path(cache_dir) if cache_dir else \
            Path(__file__).resolve().parents[4] / "data" / "cache" / "av2"
        self.dir = base / log_id
        self.dir.mkdir(parents=True, exist_ok=True)

    # -- download -----------------------------------------------------------

    @property
    def _base_url(self) -> str:
        return f"{S3_ROOT}/{self.split}/{self.log_id}"

    def _get(self, rel: str, local: str | None = None) -> Path:
        from ...sources.cache import fetch_bytes

        p = self.dir / (local or rel)
        if p.exists() and p.stat().st_size > 0:
            return p
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(fetch_bytes(f"{self._base_url}/{rel}", use_cache=False))
        return p

    def _list(self, prefix: str) -> list[str]:
        import re

        from ...sources.cache import fetch_bytes

        url = ("https://s3.amazonaws.com/argoverse/?list-type=2&max-keys=1000&prefix="
               f"datasets/av2/sensor/{self.split}/{self.log_id}/{prefix}")
        doc = fetch_bytes(url, use_cache=True).decode()
        return re.findall(r"<Key>([^<]+)</Key>", doc)

    # -- calibration and poses ---------------------------------------------

    def _calibration(self):
        import pandas as pd

        intr = pd.read_feather(self._get("calibration/intrinsics.feather"))
        extr = pd.read_feather(self._get("calibration/egovehicle_SE3_sensor.feather"))
        cams = {}
        for name in self.cameras:
            i = intr[intr.sensor_name == name]
            e = extr[extr.sensor_name == name]
            if i.empty or e.empty:
                log.warning("no calibration for camera %s", name)
                continue
            i = i.iloc[0]
            e = e.iloc[0]
            cams[name] = (
                Intrinsics(float(i.fx_px), float(i.fy_px), float(i.cx_px), float(i.cy_px),
                           int(i.width_px), int(i.height_px),
                           float(i.k1), float(i.k2), float(i.k3)),
                se3(quat_to_rot(e.qw, e.qx, e.qy, e.qz),
                    np.array([e.tx_m, e.ty_m, e.tz_m], dtype=float)),
            )
        return cams

    def _poses(self):
        import pandas as pd

        p = pd.read_feather(self._get("city_SE3_egovehicle.feather")).sort_values("timestamp_ns")
        return (p.timestamp_ns.to_numpy(dtype=np.int64),
                p[["qw", "qx", "qy", "qz"]].to_numpy(dtype=float),
                p[["tx_m", "ty_m", "tz_m"]].to_numpy(dtype=float))

    @staticmethod
    def _interp_pose(ts, quats, trans, t: int) -> np.ndarray | None:
        """Nearest-neighbour rotation with linear translation.

        Frame rate is ~150 Hz against 20 Hz imagery, so the residual rotation
        error from not slerping is well under the calibration uncertainty.
        """
        if t < ts[0] - 5e7 or t > ts[-1] + 5e7:
            return None
        i = int(np.searchsorted(ts, t))
        i0 = max(0, min(i - 1, len(ts) - 1))
        i1 = max(0, min(i, len(ts) - 1))
        if i0 == i1:
            q, tr = quats[i0], trans[i0]
        else:
            span = max(ts[i1] - ts[i0], 1)
            a = float(np.clip((t - ts[i0]) / span, 0, 1))
            q = quats[i1] if a > 0.5 else quats[i0]
            tr = (1 - a) * trans[i0] + a * trans[i1]
        return se3(quat_to_rot(*q), tr)

    # -- ground surface -----------------------------------------------------

    def ground_surface(self, frame: LocalFrame) -> GroundSurface | None:
        keys = self._list("map/")
        gk = [k for k in keys if "ground_height_surface" in k]
        sk = [k for k in keys if "img_Sim2_city" in k]
        if not gk or not sk:
            log.warning("log %s has no ground-height surface", self.log_id)
            return None
        gp = self._get(gk[0].split(f"{self.log_id}/")[-1], "map/ground_height.npy")
        sp = self._get(sk[0].split(f"{self.log_id}/")[-1], "map/img_Sim2_city.json")
        h = np.load(gp).astype(np.float32)
        s2 = json.loads(sp.read_text())
        s = float(s2["s"])
        tx, ty = float(s2["t"][0]), float(s2["t"][1])

        # AV2 stores img = s * (city + t); sample the grid in city coords and
        # convert the whole surface into the local metric frame in one go.
        rows, cols = h.shape
        cx = np.arange(cols) / s - tx
        cy = np.arange(rows) / s - ty
        gx, gy = np.meshgrid(cx, cy)
        pts = self.georef.city_to_local(
            np.column_stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)]), frame)
        x0, y0 = float(pts[0, 0]), float(pts[0, 1])
        res = float(np.hypot(pts[1, 0] - pts[0, 0], pts[1, 1] - pts[0, 1]))
        surf = GroundSurface(h, x0, y0, res, name=f"av2_{self.city}")
        log.info("ground surface %s: %sx%s @ %.2f m, %.0f%% covered",
                 self.log_id[:8], rows, cols, res, 100 * surf.coverage)
        return surf

    # -- main ---------------------------------------------------------------

    def fetch(self, aoi: AOI | None, frame: LocalFrame) -> StreetSequence:
        cams = self._calibration()
        ts, quats, trans = self._poses()

        frames: list[StreetFrame] = []
        for cam_name, (intr, ego_T_cam) in cams.items():
            keys = [k for k in self._list(f"sensors/cameras/{cam_name}/") if k.endswith(".jpg")]
            keys.sort()
            keys = keys[::self.stride][:self.max_frames]
            for k in keys:
                stamp = int(Path(k).stem)
                city_T_ego = self._interp_pose(ts, quats, trans, stamp)
                if city_T_ego is None:
                    continue
                city_T_cam = city_T_ego @ ego_T_cam
                world_T_cam = self._city_pose_to_local(city_T_cam, frame)
                img = self._get(f"sensors/cameras/{cam_name}/{Path(k).name}",
                                f"cam/{cam_name}/{Path(k).name}")
                frames.append(StreetFrame(
                    id=f"{cam_name}/{stamp}", camera=Camera(cam_name, intr, world_T_cam),
                    image_path=str(img), timestamp_ns=stamp,
                    detail={"log": self.log_id, "camera": cam_name}))

        frames.sort(key=lambda f: (f.timestamp_ns, f.id))
        log.info("av2 %s: %s posed frames from %s camera(s)",
                 self.log_id[:8], len(frames), len(cams))
        return StreetSequence(
            frames=frames,
            attribution="Argoverse 2 Sensor Dataset (CC BY-NC-SA 4.0, Argo AI)",
            ground=self.ground_surface(frame),
            detail={"log_id": self.log_id, "city": self.city, "split": self.split,
                    "cameras": list(cams), "stride": self.stride})

    def _city_pose_to_local(self, city_T_cam: np.ndarray, frame: LocalFrame) -> np.ndarray:
        """Re-express a pose from the AV2 city frame in the local metric frame.

        The two frames differ by a translation and a small rotation (UTM grid
        convergence plus the equirectangular tangent) -- see
        ``pose_to_local_frame`` for how the rotation is recovered.
        """
        return pose_to_local_frame(city_T_cam, lambda pts: self.georef.city_to_local(pts, frame))


def find_logs(split: str = "val", limit: int = 40) -> list[tuple[str, str]]:
    """List ``(log_id, city)`` pairs available in a split."""
    import re

    from ...sources.cache import fetch_bytes

    url = ("https://s3.amazonaws.com/argoverse/?list-type=2&delimiter=/&max-keys=1000"
           f"&prefix=datasets/av2/sensor/{split}/")
    doc = fetch_bytes(url, use_cache=True).decode()
    ids = [p.split("/")[-2] for p in re.findall(r"<Prefix>([^<]+)</Prefix>", doc)
           if p.count("/") == 5]
    out = []
    for lid in ids[:limit]:
        murl = ("https://s3.amazonaws.com/argoverse/?list-type=2&max-keys=20&prefix="
                f"datasets/av2/sensor/{split}/{lid}/map/")
        keys = re.findall(r"<Key>([^<]+)</Key>", fetch_bytes(murl, use_cache=True).decode())
        arc = [k for k in keys if "log_map_archive" in k]
        if arc:
            out.append((lid, arc[0].split("____")[1].split("_")[0]))
    return out
