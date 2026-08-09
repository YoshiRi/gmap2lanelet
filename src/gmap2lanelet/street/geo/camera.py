"""Pinhole camera with radial distortion, and rigid transforms.

Conventions
-----------
* ``SE3`` is a 4x4 matrix mapping *from* the child frame *to* the parent frame.
* Camera frame is optical: +x right, +y down, +z forward.
* World here is the pipeline's **local metric frame** (east, north, up), so a
  camera pose is ``world_T_cam``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def quat_to_rot(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    """Hamilton quaternion (w, x, y, z) -> 3x3 rotation."""
    n = np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if n < 1e-12:
        return np.eye(3)
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ])


def se3(rot: np.ndarray, t: np.ndarray) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = rot
    m[:3, 3] = t
    return m


def se3_inv(m: np.ndarray) -> np.ndarray:
    r = m[:3, :3]
    out = np.eye(4)
    out[:3, :3] = r.T
    out[:3, 3] = -r.T @ m[:3, 3]
    return out


@dataclass
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    k1: float = 0.0
    k2: float = 0.0
    k3: float = 0.0

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]])

    # -- distortion ---------------------------------------------------------

    def distort(self, xn: np.ndarray, yn: np.ndarray):
        """Apply radial distortion to normalised image coordinates."""
        r2 = xn * xn + yn * yn
        s = 1.0 + self.k1 * r2 + self.k2 * r2 * r2 + self.k3 * r2 * r2 * r2
        return xn * s, yn * s

    def undistort(self, xd: np.ndarray, yd: np.ndarray, iters: int = 12):
        """Invert the radial model by fixed-point iteration.

        There is no closed form; iterating ``xn = xd / s(xn)`` converges in a
        handful of steps for the distortion magnitudes real cameras have.
        """
        xn, yn = np.array(xd, dtype=float), np.array(yd, dtype=float)
        for _ in range(iters):
            r2 = xn * xn + yn * yn
            s = 1.0 + self.k1 * r2 + self.k2 * r2 * r2 + self.k3 * r2 * r2 * r2
            s = np.where(np.abs(s) < 1e-6, 1e-6, s)
            xn, yn = xd / s, yd / s
        return xn, yn


@dataclass
class Camera:
    """A posed camera: intrinsics plus ``world_T_cam``."""

    name: str
    intrinsics: Intrinsics
    world_T_cam: np.ndarray                       # 4x4

    @property
    def center(self) -> np.ndarray:
        return self.world_T_cam[:3, 3].copy()

    @property
    def forward(self) -> np.ndarray:
        """Optical axis (+z of the camera) expressed in world coordinates."""
        return self.world_T_cam[:3, 2].copy()

    @property
    def yaw(self) -> float:
        f = self.forward
        return float(np.arctan2(f[1], f[0]))

    # -- projection ---------------------------------------------------------

    def project(self, pts_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """World points (N, 3) -> pixels (N, 2) and a validity mask (in front, in frame)."""
        pts = np.atleast_2d(np.asarray(pts_world, dtype=float))
        cam_T_world = se3_inv(self.world_T_cam)
        pc = (cam_T_world[:3, :3] @ pts.T).T + cam_T_world[:3, 3]
        z = pc[:, 2]
        ok = z > 0.1
        zz = np.where(ok, z, 1.0)
        xn, yn = pc[:, 0] / zz, pc[:, 1] / zz
        xd, yd = self.intrinsics.distort(xn, yn)
        u = self.intrinsics.fx * xd + self.intrinsics.cx
        v = self.intrinsics.fy * yd + self.intrinsics.cy
        ok &= (u >= 0) & (u < self.intrinsics.width) & (v >= 0) & (v < self.intrinsics.height)
        return np.column_stack([u, v]), ok

    def ray(self, uv: np.ndarray) -> np.ndarray:
        """Pixels (N, 2) -> unit ray directions (N, 3) in world coordinates."""
        uv = np.atleast_2d(np.asarray(uv, dtype=float))
        xd = (uv[:, 0] - self.intrinsics.cx) / self.intrinsics.fx
        yd = (uv[:, 1] - self.intrinsics.cy) / self.intrinsics.fy
        xn, yn = self.intrinsics.undistort(xd, yd)
        d = np.column_stack([xn, yn, np.ones(len(xn))])
        d = (self.world_T_cam[:3, :3] @ d.T).T
        n = np.linalg.norm(d, axis=1, keepdims=True)
        return d / np.maximum(n, 1e-12)


def look_direction_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Bearing from ``a`` to ``b`` in degrees, x=east y=north."""
    d = np.asarray(b)[:2] - np.asarray(a)[:2]
    return float(np.degrees(np.arctan2(d[1], d[0])))
