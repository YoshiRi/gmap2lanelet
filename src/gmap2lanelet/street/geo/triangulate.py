"""Multi-view triangulation of bearing rays.

Detections give bearings, not range.  Two bearings from different places give a
position; more give a position *and* a way to tell whether to believe it.

Everything here reports its own uncertainty, because a traffic light
triangulated from a 2 m baseline at 60 m range is not the same claim as one
triangulated from a 40 m baseline at 25 m range even when both "converge".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TriangulationResult:
    point: np.ndarray                 # (3,)
    inliers: np.ndarray               # bool mask over the input rays
    residual_m: float                 # RMS perpendicular distance of inlier rays
    sigma_m: float                    # 1-sigma positional uncertainty
    baseline_m: float                 # max separation of inlier ray origins
    condition: float                  # conditioning of the normal matrix

    @property
    def n_inliers(self) -> int:
        return int(self.inliers.sum())


def closest_approach(o1: np.ndarray, d1: np.ndarray, o2: np.ndarray, d2: np.ndarray):
    """Midpoint of the shortest segment between two rays, and its length."""
    d1 = d1 / max(np.linalg.norm(d1), 1e-12)
    d2 = d2 / max(np.linalg.norm(d2), 1e-12)
    w0 = o1 - o2
    a, b, c = 1.0, float(d1 @ d2), 1.0
    d, e = float(d1 @ w0), float(d2 @ w0)
    den = a * c - b * b
    if abs(den) < 1e-9:                       # parallel rays carry no range
        return None, np.inf, (0.0, 0.0)
    s = (b * e - c * d) / den
    t = (a * e - b * d) / den
    p1, p2 = o1 + s * d1, o2 + t * d2
    return 0.5 * (p1 + p2), float(np.linalg.norm(p1 - p2)), (s, t)


def _lsq(origins: np.ndarray, dirs: np.ndarray):
    """Least-squares point minimising perpendicular distance to all rays."""
    a = np.zeros((3, 3))
    b = np.zeros(3)
    for o, d in zip(origins, dirs):
        d = d / max(np.linalg.norm(d), 1e-12)
        proj = np.eye(3) - np.outer(d, d)
        a += proj
        b += proj @ o
    w = np.linalg.eigvalsh(a)
    cond = float(w.max() / max(w.min(), 1e-12))
    if w.min() < 1e-8:
        return None, cond, None
    ainv = np.linalg.inv(a)
    return ainv @ b, cond, ainv


def perpendicular_distances(point: np.ndarray, origins: np.ndarray,
                            dirs: np.ndarray) -> np.ndarray:
    v = point[None, :] - origins
    dn = dirs / np.maximum(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-12)
    along = np.sum(v * dn, axis=1, keepdims=True)
    return np.linalg.norm(v - along * dn, axis=1)


def triangulate(origins: np.ndarray, dirs: np.ndarray, *, inlier_m: float = 1.2,
                min_rays: int = 2, iters: int = 60, seed: int = 0
                ) -> TriangulationResult | None:
    """RANSAC over ray pairs, then a least-squares fit to the inliers."""
    origins = np.atleast_2d(np.asarray(origins, dtype=float))
    dirs = np.atleast_2d(np.asarray(dirs, dtype=float))
    n = len(origins)
    if n < max(2, min_rays):
        return None

    rng = np.random.default_rng(seed)
    best_mask = None
    best_score = -1.0

    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    if len(pairs) > iters:
        pairs = [pairs[k] for k in rng.choice(len(pairs), iters, replace=False)]
    for i, j in pairs:
        p, gap, (s, t) = closest_approach(origins[i], dirs[i], origins[j], dirs[j])
        if p is None or gap > inlier_m * 2 or s <= 0 or t <= 0:
            continue
        mask = perpendicular_distances(p, origins, dirs) <= inlier_m
        score = mask.sum() - 0.01 * gap
        if score > best_score:
            best_score, best_mask = score, mask

    if best_mask is None or best_mask.sum() < max(2, min_rays):
        best_mask = np.ones(n, dtype=bool)

    point, cond, ainv = _lsq(origins[best_mask], dirs[best_mask])
    if point is None:
        return None
    # one refinement pass now that the estimate is better than the seed pair
    mask = perpendicular_distances(point, origins, dirs) <= inlier_m
    if mask.sum() >= max(2, min_rays):
        p2, cond2, ainv2 = _lsq(origins[mask], dirs[mask])
        if p2 is not None:
            point, cond, ainv, best_mask = p2, cond2, ainv2, mask

    d = perpendicular_distances(point, origins[best_mask], dirs[best_mask])
    resid = float(np.sqrt(np.mean(d**2))) if len(d) else 0.0
    oc = origins[best_mask]
    baseline = 0.0
    if len(oc) > 1:
        baseline = float(np.max(np.linalg.norm(oc[:, None, :] - oc[None, :, :], axis=2)))
    # positional sigma from the ray geometry, scaled by the observed residual
    sigma = float(np.sqrt(np.trace(ainv)) * max(resid, 0.05)) if ainv is not None else np.inf

    return TriangulationResult(point=point, inliers=best_mask, residual_m=resid,
                               sigma_m=sigma, baseline_m=baseline, condition=cond)


def reprojection_errors(point: np.ndarray, cameras, uvs: np.ndarray) -> np.ndarray:
    """Pixel error of a 3-D point against the detections that produced it."""
    out = []
    for cam, uv in zip(cameras, uvs):
        p, ok = cam.project(point[None, :])
        out.append(float(np.linalg.norm(p[0] - uv)) if ok[0] else np.nan)
    return np.array(out)
