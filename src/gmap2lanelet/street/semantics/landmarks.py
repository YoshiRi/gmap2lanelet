"""From per-frame boxes to 3-D landmarks.

The hard part is not triangulation, it is **data association**: a mast arm
carries three or four signal heads a few metres apart, each seen in a hundred
frames, and the wrong pairing puts a light in the middle of the sky.

The approach here is seed-and-grow rather than tracking:

1. triangulate every plausible *pair* of detections taken from frames a useful
   baseline apart, keeping only pairs that meet in front of both cameras, at a
   sane range, and at a sane height above the ground surface;
2. cluster those pair estimates in 3-D -- a real object produces a tight knot of
   them, a mismatched pair produces a scattered one;
3. re-collect every detection whose ray passes near a cluster and re-triangulate
   from all of them, with RANSAC to drop the strays.

Nothing is promoted to a landmark on a single view: a bearing with no baseline
is a direction, not a position.
"""

from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np

from ...types import Provenance, Source
from ..geo.triangulate import (closest_approach, perpendicular_distances,
                               reprojection_errors, triangulate)
from ..types import Detection, Landmark, LandmarkKind

log = logging.getLogger(__name__)


class LandmarkBuilder:
    def __init__(self, *, min_baseline_m: float = 3.0, max_range_m: float = 90.0,
                 min_range_m: float = 4.0, pair_gap_m: float = 1.0,
                 cluster_m: float = 1.5, inlier_m: float = 1.0,
                 height_range: tuple[float, float] = (1.2, 10.0),
                 min_views: int = 3, max_reproj_px: float = 40.0):
        self.min_baseline_m = min_baseline_m
        self.max_range_m = max_range_m
        self.min_range_m = min_range_m
        self.pair_gap_m = pair_gap_m
        self.cluster_m = cluster_m
        self.inlier_m = inlier_m
        self.height_range = height_range
        self.min_views = min_views
        self.max_reproj_px = max_reproj_px

    # -- main ---------------------------------------------------------------

    def build(self, detections: list[Detection], frames, ground=None) -> list[Landmark]:
        by_frame = {f.id: f for f in frames}
        out: list[Landmark] = []
        for kind in (LandmarkKind.TRAFFIC_LIGHT, LandmarkKind.STOP_SIGN):
            dets = [d for d in detections if d.kind is kind and d.frame_id in by_frame]
            out.extend(self._build_kind(kind, dets, by_frame, ground))
        for i, lm in enumerate(out):
            lm.id = f"{lm.kind.value[:2]}{i:03d}"
        log.info("landmarks: %s (%s traffic lights) from %s detections",
                 len(out), sum(1 for l in out if l.kind is LandmarkKind.TRAFFIC_LIGHT),
                 len(detections))
        return out

    def _build_kind(self, kind, dets, by_frame, ground) -> list[Landmark]:
        if len(dets) < 2:
            return []
        rays = self._rays(dets, by_frame)
        seeds = self._seed_points(dets, rays, ground)
        if not seeds:
            log.info("no %s seeds survived the geometry checks", kind.value)
            return []
        clusters = self._cluster(np.array([s[0] for s in seeds]))
        log.info("%s: %s pair seeds -> %s clusters", kind.value, len(seeds), len(clusters))

        landmarks: list[Landmark] = []
        for c in clusters:
            lm = self._fit(kind, c, dets, rays, by_frame, ground)
            if lm is not None:
                landmarks.append(lm)
        return self._merge(landmarks)

    # -- steps --------------------------------------------------------------

    @staticmethod
    def _rays(dets, by_frame):
        origins, dirs = [], []
        for d in dets:
            cam = by_frame[d.frame_id].camera
            origins.append(cam.center)
            dirs.append(cam.ray(np.array([d.center]))[0])
        return np.array(origins), np.array(dirs)

    def _seed_points(self, dets, rays, ground):
        origins, dirs = rays
        # bucket detections by camera position so we only pair views that are
        # actually far enough apart to carry range information
        order = np.argsort([d.frame_id for d in dets])
        seeds = []
        n = len(dets)
        for ii in range(n):
            i = int(order[ii])
            for jj in range(ii + 1, n):
                j = int(order[jj])
                base = float(np.linalg.norm(origins[i] - origins[j]))
                if base < self.min_baseline_m or base > self.max_range_m:
                    continue
                p, gap, (s, t) = closest_approach(origins[i], dirs[i], origins[j], dirs[j])
                if p is None or gap > self.pair_gap_m or s <= 0 or t <= 0:
                    continue
                if not (self.min_range_m <= min(s, t) <= self.max_range_m):
                    continue
                if not self._height_ok(p, ground):
                    continue
                seeds.append((p, i, j))
        return seeds

    def _height_ok(self, p, ground) -> bool:
        if ground is None:
            return True
        h = float(ground.height_above(np.asarray(p)[None, :])[0])
        if not np.isfinite(h):
            return True                      # unmapped ground: cannot judge
        return self.height_range[0] <= h <= self.height_range[1]

    def _cluster(self, pts: np.ndarray) -> list[np.ndarray]:
        """Single-link clustering with a metric radius."""
        if len(pts) == 0:
            return []
        parent = list(range(len(pts)))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
        for i, j in zip(*np.nonzero(d <= self.cluster_m)):
            if i < j:
                ri, rj = find(int(i)), find(int(j))
                if ri != rj:
                    parent[ri] = rj
        groups = defaultdict(list)
        for i in range(len(pts)):
            groups[find(i)].append(i)
        # a real object yields many mutually-consistent pair estimates
        return [pts[idx] for idx in groups.values() if len(idx) >= 2]

    def _fit(self, kind, cluster_pts, dets, rays, by_frame, ground) -> Landmark | None:
        origins, dirs = rays
        centre = cluster_pts.mean(axis=0)
        near = perpendicular_distances(centre, origins, dirs) <= self.inlier_m * 2.0
        # only rays that actually point at it, not ones that merely pass nearby
        ahead = np.einsum("ij,ij->i", centre[None, :] - origins, dirs) > 0
        sel = np.flatnonzero(near & ahead)
        if len(sel) < 2:
            return None

        tri = triangulate(origins[sel], dirs[sel], inlier_m=self.inlier_m,
                          min_rays=2)
        if tri is None:
            return None
        keep = sel[tri.inliers]
        if len(keep) < 2:
            return None

        cams = [by_frame[dets[k].frame_id].camera for k in keep]
        uvs = np.array([dets[k].center for k in keep])
        rep = reprojection_errors(tri.point, cams, uvs)
        rep_ok = np.isfinite(rep) & (rep <= self.max_reproj_px)
        if rep_ok.sum() < 2:
            return None
        keep = keep[rep_ok]
        mean_rep = float(np.mean(rep[rep_ok]))

        height = None
        if ground is not None:
            h = float(ground.height_above(tri.point[None, :])[0])
            height = h if np.isfinite(h) else None

        obs = [dets[k] for k in keep]
        cam_centres = np.array([by_frame[d.frame_id].camera.center for d in obs])
        to_cams = cam_centres - tri.point[None, :]
        mean_dir = to_cams[:, :2].mean(axis=0)
        facing = float(np.arctan2(mean_dir[1], mean_dir[0])) if np.linalg.norm(mean_dir) > 1e-6 else None

        lm = Landmark(
            id="", kind=kind, position=tri.point, detections=obs, n_views=len(obs),
            baseline_m=tri.baseline_m, residual_px=mean_rep,
            position_sigma_m=float(min(tri.sigma_m, 99.0)),
            height_above_ground=height, facing=facing,
            provenance=Provenance(Source.IMAGE, {
                "detector": "yolov8-coco", "n_views": len(obs),
                "baseline_m": round(tri.baseline_m, 2),
                "triangulation_residual_m": round(tri.residual_m, 3),
            }),
        )
        lm.confidence, lm.flags = self._score(lm, obs)
        return lm

    def _score(self, lm: Landmark, obs) -> tuple[float, list[str]]:
        flags: list[str] = []
        c = 0.25
        c += 0.30 * min(lm.n_views / 12.0, 1.0)
        c += 0.20 * min(lm.baseline_m / 25.0, 1.0)
        c += 0.15 * float(np.clip(1.0 - lm.residual_px / self.max_reproj_px, 0, 1))
        c += 0.10 * float(np.mean([d.score for d in obs]))

        if lm.n_views < self.min_views:
            flags.append("few_views")
            c -= 0.15
        if lm.baseline_m < self.min_baseline_m * 2:
            flags.append("short_baseline")
            c -= 0.10
        if lm.position_sigma_m > 1.5:
            flags.append("weak_geometry")
            c -= 0.10
        if lm.height_above_ground is None:
            flags.append("height_unverified")
            c -= 0.05
        elif lm.kind is LandmarkKind.TRAFFIC_LIGHT and not (2.0 <= lm.height_above_ground <= 8.5):
            flags.append("implausible_height")
            c -= 0.20
        return float(np.clip(c, 0.05, 0.95)), flags

    def _merge(self, landmarks: list[Landmark]) -> list[Landmark]:
        """Fuse duplicates that different seed clusters found separately."""
        out: list[Landmark] = []
        for lm in sorted(landmarks, key=lambda l: -l.n_views):
            dup = None
            for kept in out:
                if kept.kind is lm.kind and \
                        float(np.linalg.norm(kept.position - lm.position)) < self.cluster_m:
                    dup = kept
                    break
            if dup is None:
                out.append(lm)
                continue
            seen = {(d.frame_id, d.bbox) for d in dup.detections}
            for d in lm.detections:
                if (d.frame_id, d.bbox) not in seen:
                    dup.detections.append(d)
            dup.n_views = len(dup.detections)
        return out
