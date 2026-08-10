"""Pictures of the street-level semantic layer.

The claim being illustrated is a *chain*, not a set of points, so the figures
draw the chain: the signal head, the line from it to the stop line it was
assigned to, and the lanes behind that stop line.  A wrong association is then
visible at a glance -- the tie line crosses the junction diagonally -- which is
the only practical way to audit an association that has no ground truth.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from ..raster import GeoRaster
from ..types import LaneGraph
from .overlay import _fig, _save

log = logging.getLogger(__name__)

TL_COLOR = "#ffd400"
TL_UNASSIGNED = "#ff4d4d"
STOP_OBSERVED = "#00ffa8"
STOP_INFERRED = "#7a7a7a"
ARROW_COLOR = "#ff7bf5"


def render_semantics(raster: GeoRaster, graph: LaneGraph, layer, path: str | Path, *,
                     dim: float = 0.45, scale: float = 1.0,
                     title: str | None = None) -> Path:
    """Overhead view: lanes, stop lines, signal heads and their assignments."""
    from matplotlib.patches import Circle

    fig, ax = _fig(raster, scale)
    img = raster.data
    if dim > 0:
        img = (img.astype(np.float32) * (1 - dim)).astype(np.uint8)
    ax.imshow(img, extent=raster.extent, origin="upper", interpolation="bilinear")

    for ln in graph.lanes.values():
        col = "#4d7fff" if ln.kind == "turn" else "#dddddd"
        ax.plot(ln.centerline[:, 0], ln.centerline[:, 1], color=col, lw=0.9,
                alpha=0.30 + 0.5 * float(np.clip(ln.confidence, 0, 1)), zorder=3)

    lanes_by_id = graph.lanes
    for a in layer.assignments.values():
        if not a.lane_ids:
            continue
        for lid in a.lane_ids:
            ln = lanes_by_id.get(lid)
            if ln is not None:
                ax.plot(ln.centerline[:, 0], ln.centerline[:, 1], color=TL_COLOR,
                        lw=2.2, alpha=0.55, zorder=4)

    for sl in layer.stop_lines.values():
        p = np.asarray(sl.points)
        col = STOP_OBSERVED if sl.observed else STOP_INFERRED
        ax.plot(p[:, 0], p[:, 1], color=col, lw=2.6,
                ls="-" if sl.observed else (0, (3, 3)), alpha=0.95, zorder=6)

    # the association itself: a line from the head to the stop line it governs
    for a in layer.assignments.values():
        lm = layer.landmarks.get(a.landmark_id)
        if lm is None:
            continue
        assigned = bool(a.lane_ids)
        col = TL_COLOR if assigned else TL_UNASSIGNED
        if assigned and a.stop_line_id in layer.stop_lines:
            mid = np.asarray(layer.stop_lines[a.stop_line_id].points).mean(axis=0)
            ax.plot([lm.position[0], mid[0]], [lm.position[1], mid[1]], color=col,
                    lw=0.9, alpha=0.65, ls=(0, (2, 2)), zorder=7)
        ax.add_patch(Circle(lm.position[:2], 1.4, fc=col, ec="black", lw=0.5,
                            alpha=0.95, zorder=8))
        if lm.facing is not None:
            d = 4.0 * np.array([np.cos(lm.facing), np.sin(lm.facing)])
            ax.plot([lm.position[0], lm.position[0] + d[0]],
                    [lm.position[1], lm.position[1] + d[1]], color=col, lw=1.4,
                    alpha=0.9, zorder=8)

    for ar in layer.arrows.values():
        ax.plot(*ar.position[:2], marker="^", color=ARROW_COLOR, ms=7, mec="black",
                mew=0.5, zorder=9)
        ax.text(ar.position[0] + 1.5, ar.position[1], "/".join(sorted(ar.manoeuvres)),
                fontsize=6, color=ARROW_COLOR, zorder=9)

    _legend(ax, title)
    return _save(fig, path)


def _legend(ax, title: str | None) -> None:
    lines = [
        ("traffic light, assigned", TL_COLOR),
        ("traffic light, unassigned", TL_UNASSIGNED),
        ("stop line, observed", STOP_OBSERVED),
        ("stop line, inferred", STOP_INFERRED),
        ("lane arrow", ARROW_COLOR),
    ]
    txt = (title + "\n" if title else "") + "\n".join(f"■ {n}" for n, _ in lines)
    t = ax.text(0.01, 0.99, txt, transform=ax.transAxes, va="top", ha="left",
                fontsize=8, color="white", linespacing=1.5,
                bbox=dict(facecolor="black", alpha=0.6, pad=5, edgecolor="none"))
    t.set_zorder(20)


def render_landmark_check(frames, layer, path: str | Path, *, n: int = 6,
                          landmark_ids: list[str] | None = None) -> Path | None:
    """Reproject each 3-D landmark into the frames that saw it.

    With no positional ground truth this is the audit that is actually
    available: if the recovered point lands back inside its own detections in
    views taken tens of metres apart, the triangulation is consistent; if it
    drifts across the image as the vehicle moves, it is not.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    by_frame = {f.id: f for f in frames}
    lms = [layer.landmarks[i] for i in landmark_ids] if landmark_ids else \
        sorted(layer.landmarks.values(), key=lambda l: -l.n_views)[:n]
    lms = [l for l in lms if l.detections]
    if not lms:
        return None

    cols = 3
    fig, axes = plt.subplots(len(lms), cols, figsize=(3.2 * cols, 2.4 * len(lms)),
                             dpi=110, squeeze=False)
    for r, lm in enumerate(lms):
        picks = _spread(lm.detections, cols)
        for c in range(cols):
            ax = axes[r][c]
            ax.set_axis_off()
            if c >= len(picks):
                continue
            det = picks[c]
            f = by_frame.get(det.frame_id)
            if f is None:
                continue
            img = f.load()
            x0, y0, x1, y1 = det.bbox
            pad = max(60.0, 2.0 * (x1 - x0))
            cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
            ax.imshow(img)
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                   ec="#00ff88", lw=1.2))
            uv, ok = f.camera.project(lm.position[None, :])
            if ok[0]:
                ax.plot(uv[0, 0], uv[0, 1], marker="+", color="#ffd400", ms=12, mew=2)
            ax.set_xlim(cx - pad, cx + pad)
            ax.set_ylim(cy + pad, cy - pad)
            if c == 0:
                ax.set_title(f"{lm.id}  {lm.n_views} views  "
                             f"sigma {lm.position_sigma_m:.2f} m  "
                             f"h {lm.height_above_ground:.1f} m"
                             if lm.height_above_ground is not None else lm.id,
                             fontsize=7, loc="left")
    fig.suptitle("green = detection, yellow + = the triangulated 3-D point reprojected",
                 fontsize=8)
    fig.tight_layout()
    return _save(fig, path)


def _spread(items, k: int):
    if len(items) <= k:
        return list(items)
    idx = np.linspace(0, len(items) - 1, k).round().astype(int)
    return [items[int(i)] for i in idx]
