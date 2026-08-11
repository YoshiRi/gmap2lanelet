"""Painted lane arrows -> per-lane turn permissions.

This is the piece that replaces a guess with an observation.  Without it,
intersection connectivity comes from the "leftmost turns left, rightmost turns
right" convention; with it, the lane that is actually painted with a left arrow
is the lane allowed to turn left.

Classification is deliberately geometric rather than learned.  An arrow, seen
from above and rotated into its lane's frame, is a shaft with a head; the head
of a turn arrow is displaced to one side of the shaft, the head of a through
arrow is not.  Measuring that displacement is interpretable, needs no training
data, and degrades into "unknown" rather than into a confident wrong answer.
"""

from __future__ import annotations

import logging

import numpy as np

from ...geo import cumulative_length, normals, resample_polyline
from ...types import Provenance, Source
from ..types import LaneArrow

log = logging.getLogger(__name__)


def find_arrows(lane_id: str, lane_center: np.ndarray, half_width: float, marking,
                *, search_from: float = 3.0, search_to: float = 45.0,
                res: float = 0.06, thr: float = 0.35,
                area_range: tuple[float, float] = (0.35, 6.0),
                length_range: tuple[float, float] = (1.0, 7.5),
                width_range: tuple[float, float] = (0.35, 2.6)) -> list[LaneArrow]:
    """Rasterise the lane in its own frame and look for arrow-shaped blobs."""
    from scipy import ndimage as ndi

    c = resample_polyline(np.asarray(lane_center, dtype=float)[:, :2], res)
    if len(c) < 8:
        return []
    n = normals(c)
    s = cumulative_length(c)
    total = float(s[-1])
    lo, hi = max(0.0, total - search_to), max(0.0, total - search_from)
    sel = (s >= lo) & (s <= hi)
    if sel.sum() < 20:
        return []

    us = np.arange(-half_width + 0.15, half_width - 0.15 + 1e-9, res)
    pts = c[sel][:, None, :] + n[sel][:, None, :] * us[None, :, None]
    resp = marking.sample(pts[:, :, 0], pts[:, :, 1], fill=0.0)
    mask = resp >= thr
    if mask.sum() < 20:
        return []

    mask = ndi.binary_closing(mask, np.ones((3, 3)))
    lab, n_lab = ndi.label(mask)
    stations = s[sel]
    out: list[LaneArrow] = []

    for k in range(1, n_lab + 1):
        rows, cols = np.nonzero(lab == k)
        area = len(rows) * res * res
        if not (area_range[0] <= area <= area_range[1]):
            continue
        s_span = (rows.max() - rows.min() + 1) * res
        u_span = (cols.max() - cols.min() + 1) * res
        if not (length_range[0] <= s_span <= length_range[1]):
            continue
        if not (width_range[0] <= u_span <= width_range[1]):
            continue
        # an arrow is mostly empty box: solid rectangles are text or patches
        fill = area / max(s_span * u_span, 1e-6)
        if fill > 0.72:
            continue

        manoeuvres, score, detail = _classify(rows, cols, us, res)
        if not manoeuvres:
            continue
        i = int(np.clip(rows.mean(), 0, sel.sum() - 1))
        u = float(us[int(np.clip(cols.mean(), 0, len(us) - 1))])
        pos = c[sel][i] + n[sel][i] * u
        out.append(LaneArrow(
            id="", lane_id=lane_id, position=pos, manoeuvres=manoeuvres,
            score=score,
            confidence=float(np.clip(0.30 + 0.5 * score, 0.1, 0.85)),
            provenance=Provenance(Source.IMAGE, {
                "area_m2": round(area, 2), "length_m": round(s_span, 2),
                "width_m": round(u_span, 2), "fill": round(fill, 2),
                "station_from_end_m": round(total - float(stations[i]), 1), **detail}),
            flags=["arrow_shape_classified_geometrically"]))
    return out


def _classify(rows: np.ndarray, cols: np.ndarray, us: np.ndarray, res: float):
    """Compare the lateral extent of the head against the shaft.

    Rows increase along the direction of travel, so the *head* of the arrow is
    at high row indices and the shaft at low ones.
    """
    r0, r1 = rows.min(), rows.max()
    length = r1 - r0 + 1
    if length < 6:
        return set(), 0.0, {}

    head = rows >= r0 + 0.62 * length
    shaft = rows <= r0 + 0.38 * length
    if head.sum() < 4 or shaft.sum() < 4:
        return set(), 0.0, {}

    shaft_c = float(np.median(us[cols[shaft]]))
    head_u = us[cols[head]]
    left = float(np.percentile(head_u, 97)) - shaft_c        # +u is left
    right = shaft_c - float(np.percentile(head_u, 3))
    head_w = float(np.percentile(head_u, 97) - np.percentile(head_u, 3))
    shaft_w = float(np.percentile(us[cols[shaft]], 97) - np.percentile(us[cols[shaft]], 3))

    manoeuvres: set[str] = set()
    turn_thr = 0.40
    if left > turn_thr and left > 1.5 * max(right, 1e-3):
        manoeuvres.add("left")
    if right > turn_thr and right > 1.5 * max(left, 1e-3):
        manoeuvres.add("right")
    if not manoeuvres or (head_w > 0.5 and abs(left - right) < 0.30):
        manoeuvres.add("through")
    # a wide symmetric head on a narrow shaft is a plain through arrow
    if head_w < shaft_w * 1.15 and not manoeuvres:
        manoeuvres.add("through")

    asym = abs(left - right) / max(head_w, 1e-3)
    score = float(np.clip(0.35 + 0.65 * min(asym * 1.4, 1.0), 0, 1)) \
        if manoeuvres != {"through"} else float(np.clip(0.8 - asym, 0.2, 0.8))
    detail = {"head_left_m": round(left, 2), "head_right_m": round(right, 2),
              "head_width_m": round(head_w, 2), "shaft_width_m": round(shaft_w, 2)}
    return manoeuvres, score, detail


def merge_arrows(arrows: list[LaneArrow]) -> list[LaneArrow]:
    """One lane usually carries one arrow repeated; combine what they say."""
    by_lane: dict[str, list[LaneArrow]] = {}
    for a in arrows:
        by_lane.setdefault(a.lane_id, []).append(a)

    out: list[LaneArrow] = []
    for group in by_lane.values():
        group.sort(key=lambda a: -a.score)
        best = group[0]
        if len(group) > 1:
            agree = sum(1 for a in group if a.manoeuvres == best.manoeuvres)
            best.confidence = float(np.clip(best.confidence + 0.08 * (agree - 1), 0.1, 0.9))
            best.provenance.detail["repeats"] = len(group)
            best.provenance.detail["agreeing_repeats"] = agree
            if agree < len(group):
                best.flags.append("arrow_repeats_disagree")
        out.append(best)
    for i, a in enumerate(out):
        a.id = f"ar{i:03d}"
    log.info("arrows: %s lanes carry a readable arrow", len(out))
    return out
