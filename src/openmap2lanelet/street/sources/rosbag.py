"""Posed frames + detections from a ROS2 rosbag (a real driven log).

This is not a source for *building* a map -- it is a source for **checking**
one: the bag supplies posed camera frames and the vehicle's own onboard
detections (e.g. a full-frame YOLOX pass), which ``LandmarkBuilder``
triangulates into 3-D ``Landmark``s exactly as it does for Argoverse 2
(``av2.py``). Those landmarks are then compared against an already-exported
Lanelet2 map by ``qa/rosbag_validate.py``.

Nothing about the bag's topic names or message shapes is assumed. Every
platform names its topics differently and this project's own convention
(``sources/xyz_tiles.py``, ``sources/osm_overpass.py``) is to never hard-code
an external endpoint -- so topic names are a required constructor argument,
and how to decode the pose/detection messages is a required callable. The
image, camera_info and ``/tf_static`` message *shapes*, by contrast, are
standard and stable across ROS distributions, so those are decoded directly.

Reading a bag needs the ``rosbags`` package (``pip install -e ".[rosbag]"``),
a pure-Python library that reads a ROS2 bag's own embedded message
definitions -- no ROS installation required. The import is lazy so this
module loads fine without the extra installed.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from ...geo import AOI, LocalFrame
from ..geo.camera import Camera, Intrinsics, pose_to_local_frame, quat_to_rot, se3
from ..types import Detection, LandmarkKind
from .base import StreetFrame, StreetSequence

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RosbagTopics:
    """Every topic this reader needs, named explicitly -- no defaults assumed
    for anything platform-specific."""

    image: str
    camera_info: str
    pose: str
    detections: str
    tf_static: str = "/tf_static"          # near-universal, still overridable


@dataclass(frozen=True)
class RawBox:
    """One 2-D detection decoded from the bag's own detection topic."""

    kind: LandmarkKind
    bbox: tuple[float, float, float, float]        # x0, y0, x1, y1 in pixels
    score: float


DecodePoseFn = Callable[[object], tuple[np.ndarray, np.ndarray]]
"""``msg -> (xyz (3,), quat_wxyz (4,))``."""

DecodeDetectionsFn = Callable[[object], list[RawBox]]
"""``msg -> list[RawBox]``.

No default is provided: the real message type (Autoware/tier4 YOLOX output)
is platform-specific and unconfirmed. Write one against a sample from the
actual bag before using this reader -- e.g. for a
``tier4_perception_msgs/msg/DetectedObjectsWithFeature``-shaped topic, iterate
``msg.feature_objects``, read each ``.feature.roi`` (x_offset, y_offset,
width, height) and ``.object.classification`` for the label/score.
"""


class BagGeoref(Protocol):
    """Converts the bag's self-localization frame to WGS84 and to the
    pipeline's local metric frame. See ``MgrsGeoref``/``LocalCartesianGeoref``."""

    def to_wgs84(self, x, y) -> tuple[np.ndarray, np.ndarray]: ...

    def to_local(self, xyz: np.ndarray, frame: LocalFrame) -> np.ndarray: ...


class MgrsGeoref:
    """The bag's pose x/y are MGRS easting/northing within one 100 km square.

    ``grid_zone`` is the MGRS grid-zone-designator + 100km-square id (e.g.
    ``"54SUE"`` for central Tokyo) that the bag's local map frame was built
    against -- this project has no way to discover it on its own (our own
    exported ``.osm`` files carry no MGRS metadata, see
    ``export/lanelet2_osm.py``), so it must be supplied explicitly, e.g. from
    the vehicle stack's ``map_projector_info``.
    """

    def __init__(self, grid_zone: str):
        self.grid_zone = grid_zone

    def to_wgs84(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        import mgrs

        m = mgrs.MGRS()
        x = np.atleast_1d(np.asarray(x, dtype=float))
        y = np.atleast_1d(np.asarray(y, dtype=float))
        lats = np.empty(len(x))
        lons = np.empty(len(x))
        for i, (xi, yi) in enumerate(zip(x, y, strict=True)):
            # MGRS strings are metre-precision integers; 1m precision (5+5
            # digits) is well below the triangulation's own uncertainty.
            s = f"{self.grid_zone}{int(round(xi)):05d}{int(round(yi)):05d}"
            lat, lon = m.toLatLon(s)
            lats[i], lons[i] = lat, lon
        return lons, lats

    def to_local(self, xyz: np.ndarray, frame: LocalFrame) -> np.ndarray:
        p = np.atleast_2d(np.asarray(xyz, dtype=float))
        lon, lat = self.to_wgs84(p[:, 0], p[:, 1])
        lx, ly = frame.to_local(lon, lat)
        z = p[:, 2] if p.shape[1] > 2 else np.zeros(len(p))
        return np.column_stack([lx, ly, z])


class LocalCartesianGeoref:
    """The bag's pose x/y is a tangent plane about ``(lat0, lon0)``, optionally
    rotated ``rotation_rad`` off true north (some local map frames are laid
    out along a road axis rather than east/north)."""

    def __init__(self, lat0: float, lon0: float, rotation_rad: float = 0.0):
        self._frame = LocalFrame(lat0, lon0)
        self._rot = float(rotation_rad)

    def _rotate(self, x: np.ndarray, y: np.ndarray, inverse: bool = False):
        a = -self._rot if inverse else self._rot
        c, s = np.cos(a), np.sin(a)
        return c * x - s * y, s * x + c * y

    def to_wgs84(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        ex, ny = self._rotate(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        return self._frame.to_wgs84(ex, ny)

    def to_local(self, xyz: np.ndarray, frame: LocalFrame) -> np.ndarray:
        p = np.atleast_2d(np.asarray(xyz, dtype=float))
        lon, lat = self._frame.to_wgs84(*self._rotate(p[:, 0], p[:, 1]))
        lx, ly = frame.to_local(lon, lat)
        z = p[:, 2] if p.shape[1] > 2 else np.zeros(len(p))
        return np.column_stack([lx, ly, z])


# --------------------------------------------------------------------------- #
# pure helpers -- no rosbags import, exercised directly by tests
# --------------------------------------------------------------------------- #


def _decode_common_pose(msg: object) -> tuple[np.ndarray, np.ndarray]:
    """Default ``DecodePoseFn``: covers PoseStamped, Odometry and
    PoseWithCovarianceStamped, whose ``pose``/``pose.pose`` fields all carry
    the same ``position``/``orientation`` shape."""
    p = msg.pose
    p = getattr(p, "pose", p)                  # unwrap Odometry/WithCovariance's extra nesting
    pos = p.position
    o = p.orientation
    xyz = np.array([pos.x, pos.y, pos.z], dtype=float)
    # ROS quaternions are (x, y, z, w); quat_to_rot wants Hamilton (w, x, y, z).
    quat_wxyz = np.array([o.w, o.x, o.y, o.z], dtype=float)
    return xyz, quat_wxyz


def compose_camera_pose(map_T_base: np.ndarray, base_T_cam: np.ndarray) -> np.ndarray:
    """``map_T_cam``, from a self-localization pose and a static extrinsic."""
    return map_T_base @ base_T_cam


def lookup_static_transform(transforms: dict[tuple[str, str], np.ndarray],
                            parent: str, child: str) -> np.ndarray | None:
    """Breadth-first search over ``/tf_static``'s (parent, child) -> SE3 edges.

    ``camera_frame_id`` is rarely a *direct* child of ``base_frame_id`` in a
    real TF tree (there's usually a sensor mount / lidar frame in between), so
    this composes whatever chain connects them, inverting edges walked
    parent<-child.
    """
    if parent == child:
        return np.eye(4)
    adjacency: dict[str, list[tuple[str, np.ndarray]]] = {}
    for (p, c), t in transforms.items():
        adjacency.setdefault(p, []).append((c, t))
        adjacency.setdefault(c, []).append((p, np.linalg.inv(t)))

    from collections import deque

    seen = {parent}
    queue = deque([(parent, np.eye(4))])
    while queue:
        frame, acc = queue.popleft()
        for nxt, t in adjacency.get(frame, []):
            if nxt in seen:
                continue
            reached = acc @ t
            if nxt == child:
                return reached
            seen.add(nxt)
            queue.append((nxt, reached))
    return None


def boxes_to_detections(frame: StreetFrame, boxes: list[RawBox]) -> list[Detection]:
    return [Detection(frame_id=frame.id, kind=b.kind, bbox=b.bbox, score=b.score,
                      camera=frame.camera.name)
            for b in boxes]


def _nearest(timestamps_ns: np.ndarray, t: int, tolerance_ns: int) -> int | None:
    if len(timestamps_ns) == 0:
        return None
    i = int(np.searchsorted(timestamps_ns, t))
    candidates = [j for j in (i - 1, i) if 0 <= j < len(timestamps_ns)]
    if not candidates:
        return None
    best = min(candidates, key=lambda j: abs(int(timestamps_ns[j]) - t))
    return best if abs(int(timestamps_ns[best]) - t) <= tolerance_ns else None


# --------------------------------------------------------------------------- #


class RosbagStreetSource:
    """Posed frames (``StreetImagerySource``) and detections (``DetectorBackend``)
    read from one ROS2 rosbag."""

    name = "rosbag"

    def __init__(self, bag_path: str | Path, topics: RosbagTopics, georef: BagGeoref, *,
                 camera_frame_id: str, base_frame_id: str,
                 decode_pose: DecodePoseFn = _decode_common_pose,
                 decode_detections: DecodeDetectionsFn,
                 pose_sync_tolerance_ns: int = 50_000_000,
                 stride: int = 1, max_frames: int = 0,
                 cache_dir: str | Path | None = None):
        self.bag_path = Path(bag_path)
        self.topics = topics
        self.georef = georef
        self.camera_frame_id = camera_frame_id
        self.base_frame_id = base_frame_id
        self.decode_pose = decode_pose
        self.decode_detections = decode_detections
        self.pose_sync_tolerance_ns = pose_sync_tolerance_ns
        self.stride = stride
        self.max_frames = max_frames
        self.cache_dir = Path(cache_dir) if cache_dir else \
            self.bag_path.parent / f"{self.bag_path.stem}_cache"
        self._frames_cache: list[StreetFrame] | None = None

    # -- bag reading (lazy `rosbags` import) --------------------------------

    def _reader(self):
        from rosbags.highlevel import AnyReader

        return AnyReader([self.bag_path])

    def _read_topic(self, reader, topic: str):
        conns = [c for c in reader.connections if c.topic == topic]
        if not conns:
            log.warning("topic %s not present in %s", topic, self.bag_path)
        for connection, timestamp, rawdata in reader.messages(connections=conns):
            yield timestamp, reader.deserialize(rawdata, connection.msgtype)

    def _static_transforms(self, reader) -> dict[tuple[str, str], np.ndarray]:
        out: dict[tuple[str, str], np.ndarray] = {}
        for _, msg in self._read_topic(reader, self.topics.tf_static):
            for t in msg.transforms:
                trans = t.transform.translation
                rot = t.transform.rotation
                xyz = np.array([trans.x, trans.y, trans.z], dtype=float)
                m = se3(quat_to_rot(rot.w, rot.x, rot.y, rot.z), xyz)
                out[(t.header.frame_id, t.child_frame_id)] = m
        return out

    def _intrinsics(self, reader) -> Intrinsics | None:
        for _, msg in self._read_topic(reader, self.topics.camera_info):
            k = np.asarray(msg.k, dtype=float).reshape(3, 3)
            d = list(msg.d) if len(msg.d) else [0.0, 0.0, 0.0]
            # Assumes a plumb_bob-style [k1, k2, p1, p2, k3] layout, matching
            # `Intrinsics.distort`'s pure-radial model; if the bag's
            # `distortion_model` is fisheye/equidistant this needs revisiting.
            k1 = d[0] if len(d) > 0 else 0.0
            k2 = d[1] if len(d) > 1 else 0.0
            k3 = d[4] if len(d) > 4 else 0.0
            return Intrinsics(fx=float(k[0, 0]), fy=float(k[1, 1]),
                              cx=float(k[0, 2]), cy=float(k[1, 2]),
                              width=int(msg.width), height=int(msg.height),
                              k1=k1, k2=k2, k3=k3)
        return None

    def _decode_image(self, msg) -> np.ndarray:
        import io

        from PIL import Image

        if hasattr(msg, "format"):                      # sensor_msgs/CompressedImage
            return np.asarray(Image.open(io.BytesIO(bytes(msg.data))).convert("RGB"))
        # sensor_msgs/Image: reshape raw bytes by encoding.
        h, w = int(msg.height), int(msg.width)
        buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)
        if msg.encoding in ("rgb8", "bgr8"):
            arr = buf.reshape(h, w, 3)
            return arr[:, :, ::-1] if msg.encoding == "bgr8" else arr
        if msg.encoding == "mono8":
            return np.repeat(buf.reshape(h, w, 1), 3, axis=2)
        raise ValueError(f"unsupported image encoding {msg.encoding!r}")

    # -- StreetImagerySource -------------------------------------------------

    def fetch(self, aoi: AOI | None, frame: LocalFrame) -> StreetSequence:
        with self._reader() as reader:
            static_tf = self._static_transforms(reader)
            base_T_cam = lookup_static_transform(static_tf, self.base_frame_id,
                                                 self.camera_frame_id)
            if base_T_cam is None:
                raise ValueError(
                    f"no /tf_static chain from {self.base_frame_id!r} to "
                    f"{self.camera_frame_id!r} in {self.bag_path}")
            intr = self._intrinsics(reader)
            if intr is None:
                raise ValueError(f"no camera_info on {self.topics.camera_info!r}")

            poses = sorted(self._read_topic(reader, self.topics.pose), key=lambda p: p[0])
            pose_ts = np.array([t for t, _ in poses], dtype=np.int64)

            images = sorted(self._read_topic(reader, self.topics.image), key=lambda p: p[0])
            images = images[::self.stride]
            if self.max_frames:
                images = images[:self.max_frames]

            self.cache_dir.mkdir(parents=True, exist_ok=True)
            frames: list[StreetFrame] = []
            for ts, img_msg in images:
                i = _nearest(pose_ts, ts, self.pose_sync_tolerance_ns)
                if i is None:
                    continue
                xyz, quat_wxyz = self.decode_pose(poses[i][1])
                map_T_base = se3(quat_to_rot(*quat_wxyz), xyz)
                map_T_cam = compose_camera_pose(map_T_base, base_T_cam)
                world_T_cam = pose_to_local_frame(
                    map_T_cam, lambda pts: self.georef.to_local(pts, frame))

                img_path = self.cache_dir / f"{ts}.png"
                if not img_path.exists():
                    from PIL import Image

                    Image.fromarray(self._decode_image(img_msg)).save(img_path)

                frames.append(StreetFrame(
                    id=f"{self.camera_frame_id}/{ts}",
                    camera=Camera(self.camera_frame_id, intr, world_T_cam),
                    image_path=str(img_path), timestamp_ns=ts,
                    detail={"bag": str(self.bag_path)}))

        log.info("rosbag %s: %s posed frames", self.bag_path.name, len(frames))
        self._frames_cache = frames
        return StreetSequence(
            frames=frames, attribution=f"rosbag {self.bag_path.name} (user-supplied drive)",
            ground=None, detail={"bag": str(self.bag_path), "topics": vars(self.topics)})

    # -- DetectorBackend ------------------------------------------------------

    def detect(self, frames: list[StreetFrame]) -> list[Detection]:
        ts = np.array([f.timestamp_ns for f in frames], dtype=np.int64)
        out: list[Detection] = []
        with self._reader() as reader:
            for t, msg in self._read_topic(reader, self.topics.detections):
                i = _nearest(ts, t, self.pose_sync_tolerance_ns)
                if i is None:
                    continue
                out.extend(boxes_to_detections(frames[i], self.decode_detections(msg)))
        log.info("rosbag %s: %s detections matched to frames", self.bag_path.name, len(out))
        return out
