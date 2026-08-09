"""Static overlays: imagery + prior + result, on one set of axes."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from ..observation.base import Evidence
from ..prior.road_graph import RoadPrior
from ..raster import GeoRaster
from ..types import LaneGraph, MarkingType, Source

log = logging.getLogger(__name__)

MARKING_STYLE: dict[MarkingType, tuple[str, str, float]] = {
    #                       colour     linestyle  width
    MarkingType.SOLID:        ("#ffd400", "-", 1.4),
    MarkingType.DASHED:       ("#ffd400", (0, (4, 3)), 1.2),
    MarkingType.DOUBLE_SOLID: ("#ff8c00", "-", 1.8),
    MarkingType.ROAD_EDGE:    ("#00e5ff", "-", 1.0),
    MarkingType.VIRTUAL:      ("#ff4dd2", (0, (1, 2)), 0.9),
    MarkingType.UNKNOWN:      ("#999999", (0, (1, 3)), 0.8),
}

SOURCE_COLOR = {
    Source.OSM: "#4c9aff",
    Source.IMAGE: "#36d399",
    Source.FUSED: "#a78bfa",
    Source.INFERRED: "#f87272",
    Source.DEFAULT: "#9ca3af",
}


def _fig(raster: GeoRaster, scale: float = 1.0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h, w = raster.shape
    dpi = 100
    fig = plt.figure(figsize=(w / dpi * scale, h / dpi * scale), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    x0, x1, y0, y1 = raster.extent
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    return fig, ax


def render_overlay(raster: GeoRaster, prior: RoadPrior, graph: LaneGraph, path: str | Path,
                   *, show_prior: bool = True, show_boundaries: bool = True,
                   show_centerlines: bool = True, show_intersections: bool = True,
                   dim: float = 0.35, scale: float = 1.0, title: str | None = None) -> Path:
    """The headline picture: imagery under, prior and result over."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    fig, ax = _fig(raster, scale)
    img = raster.data
    if dim > 0:
        img = (img.astype(np.float32) * (1 - dim)).astype(np.uint8)
    ax.imshow(img, extent=raster.extent, origin="upper", interpolation="bilinear")

    if show_prior:
        for e in prior.edges.values():
            ax.plot(e.points[:, 0], e.points[:, 1], color="#00b3ff", lw=1.6, alpha=0.55,
                    ls=(0, (6, 4)), zorder=2)

    if show_intersections:
        for it in graph.intersections.values():
            ax.add_patch(Circle(it.center, it.radius, fill=False, ec="#ff6b6b",
                                lw=1.0, ls=(0, (3, 3)), alpha=0.7, zorder=3))

    if show_boundaries:
        for b in graph.boundaries.values():
            c, ls, lw = MARKING_STYLE.get(b.marking, ("#999", "-", 0.8))
            ax.plot(b.points[:, 0], b.points[:, 1], color=c, ls=ls, lw=lw,
                    alpha=0.9, zorder=4)

    if show_centerlines:
        for ln in graph.lanes.values():
            col = SOURCE_COLOR.get(ln.provenance.source, "#ffffff")
            alpha = 0.35 + 0.6 * float(np.clip(ln.confidence, 0, 1))
            lw = 1.0 if ln.kind == "turn" else 1.6
            ls = (0, (5, 3)) if ln.kind == "turn" else "-"
            ax.plot(ln.centerline[:, 0], ln.centerline[:, 1], color=col, lw=lw, ls=ls,
                    alpha=alpha, zorder=5)
            _arrow(ax, ln.centerline, col, alpha)

    if title:
        ax.text(0.01, 0.99, title, transform=ax.transAxes, va="top", ha="left",
                fontsize=11, color="white",
                bbox=dict(facecolor="black", alpha=0.55, pad=4, edgecolor="none"))

    return _save(fig, path)


def _arrow(ax, pts: np.ndarray, color: str, alpha: float) -> None:
    if len(pts) < 4:
        return
    i = len(pts) // 2
    d = pts[min(i + 1, len(pts) - 1)] - pts[i]
    n = float(np.hypot(*d))
    if n < 1e-6:
        return
    d = d / n * 2.2
    ax.annotate("", xy=tuple(pts[i] + d), xytext=tuple(pts[i]),
                arrowprops=dict(arrowstyle="-|>", color=color, alpha=alpha, lw=0.9,
                                shrinkA=0, shrinkB=0), zorder=6)


def render_evidence(raster: GeoRaster, evidence: Evidence, path: str | Path,
                    scale: float = 1.0) -> Path:
    """Two-panel debug view of what the imagery backend actually saw."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h, w = raster.shape
    dpi = 100
    fig, axes = plt.subplots(1, 2, figsize=(2 * w / dpi * scale, h / dpi * scale), dpi=dpi)
    for ax in axes:
        ax.set_axis_off()

    axes[0].imshow(raster.data, extent=raster.extent, origin="upper")
    axes[0].imshow(evidence.road_prob.data, extent=raster.extent, origin="upper",
                   cmap="Blues", alpha=0.55, vmin=0, vmax=1)
    axes[0].set_title("road surface probability", fontsize=9)

    axes[1].imshow(raster.data, extent=raster.extent, origin="upper")
    m = np.ma.masked_less(evidence.marking.data, 0.2)
    axes[1].imshow(m, extent=raster.extent, origin="upper", cmap="autumn",
                   alpha=0.95, vmin=0.2, vmax=1)
    axes[1].set_title("lane-marking response", fontsize=9)

    fig.tight_layout(pad=0.3)
    return _save(fig, path)


def render_crop(raster: GeoRaster, prior: RoadPrior, graph: LaneGraph, center: np.ndarray,
                half: float, path: str | Path, *, title: str | None = None,
                px: int = 520) -> Path:
    """A zoomed review crop around a point of interest."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    dpi = 100
    fig = plt.figure(figsize=(px / dpi, px / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.imshow(raster.data, extent=raster.extent, origin="upper", interpolation="bilinear")
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)

    for e in prior.edges.values():
        ax.plot(e.points[:, 0], e.points[:, 1], color="#00b3ff", lw=1.8, alpha=0.7,
                ls=(0, (6, 4)))
    for b in graph.boundaries.values():
        c, ls, lw = MARKING_STYLE.get(b.marking, ("#999", "-", 0.8))
        ax.plot(b.points[:, 0], b.points[:, 1], color=c, ls=ls, lw=lw * 1.4, alpha=0.95)
    for ln in graph.lanes.values():
        col = SOURCE_COLOR.get(ln.provenance.source, "#fff")
        ax.plot(ln.centerline[:, 0], ln.centerline[:, 1], color=col, lw=1.6,
                alpha=0.35 + 0.6 * ln.confidence)
    for it in graph.intersections.values():
        ax.add_patch(Circle(it.center, it.radius, fill=False, ec="#ff6b6b", lw=1.2,
                            ls=(0, (3, 3)), alpha=0.8))
    if title:
        ax.text(0.02, 0.98, title, transform=ax.transAxes, va="top", fontsize=9,
                color="white", bbox=dict(facecolor="black", alpha=0.6, pad=3,
                                         edgecolor="none"))
    return _save(fig, path)


def render_profile_debug(seg_result, path: str | Path) -> Path | None:
    """The road-aligned view of one segment: the pipeline's inner workspace."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not seg_result.corridors:
        return None
    prof = seg_result.profiles[0]
    fig, axes = plt.subplots(1, 2, figsize=(9, 5), dpi=110, sharey=True)

    for ax, data, name, cmap in ((axes[0], prof.road, "road probability", "Blues"),
                                 (axes[1], prof.mark, "marking response", "inferno")):
        ax.imshow(data, aspect="auto", cmap=cmap, origin="lower",
                  extent=(prof.offsets[0], prof.offsets[-1],
                          prof.stations[0], prof.stations[-1]), vmin=0, vmax=1)
        ax.set_title(f"{name}", fontsize=9)
        ax.set_xlabel("lateral offset from prior centreline [m]", fontsize=8)
        for cor in seg_result.corridors:
            ax.plot(cor.left, prof.stations[:len(cor.left)], color="#00e5ff", lw=1.2)
            ax.plot(cor.right, prof.stations[:len(cor.right)], color="#00e5ff", lw=1.2)
            ax.plot(cor.center_offset, prof.stations[:len(cor.left)], color="#ff4dd2",
                    lw=1.0, ls=(0, (4, 3)))
        ax.axvline(0.0, color="w", lw=0.8, alpha=0.6)
    axes[0].set_ylabel("station along prior edge [m]", fontsize=8)
    fig.suptitle(f"segment {seg_result.edge.id} ({seg_result.edge.highway}) - "
                 f"road-aligned evidence; magenta = corrected centreline", fontsize=9)
    fig.tight_layout()
    return _save(fig, path)


def _save(fig, path: str | Path) -> Path:
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02, facecolor="#101014")
    plt.close(fig)
    return path
