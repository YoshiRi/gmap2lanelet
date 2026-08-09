"""Stop lines: transverse markings across an approach.

Longitudinal markings (lane lines) and transverse ones (stop bars, crosswalks)
live in the same raster and are separated by *orientation in the road-aligned
frame*: a lane line is a ridge in ``u`` that persists over ``s``; a stop bar is a
ridge in ``s`` that spans most of ``u``.  Sampling the marking response in that
frame therefore turns the problem into finding a narrow, wide-spanning peak.

A crosswalk is the classic false positive -- it is also transverse, also spans
the road, and sits within a few metres of the stop bar.  It is rejected by
periodicity: a zebra is several bars in a row at ~1 m pitch, a stop bar is one.
"""

from __future__ import annotations

import logging

import numpy as np

from ...geo import cumulative_length, normals, resample_polyline
from ...types import Provenance, Source
from ..types import StopLine

log = logging.getLogger(__name__)


def detect_stop_line(lane_center: np.ndarray, half_width: float, marking,
                     *, search_from: float, search_to: float, ds: float = 0.05,
                     du: float = 0.05, min_span: float = 0.55,
                     thickness_range: tuple[float, float] = (0.15, 0.9),
                     thr: float = 0.35, zebra_pitch: tuple[float, float] = (0.6, 2.2)
                     ) -> tuple[float, float, dict] | None:
    """Find a stop bar on ``lane_center``, measured from its downstream end.

    Returns ``(station_from_end, score, detail)`` or ``None``.  Stations are in
    metres back from the end of the lane, which is where the junction is.
    """
    c = resample_polyline(np.asarray(lane_center, dtype=float)[:, :2], ds)
    if len(c) < 4:
        return None
    n = normals(c)
    s = cumulative_length(c)
    total = float(s[-1])
    lo = max(0.0, total - search_to)
    hi = max(0.0, total - search_from)
    sel = (s >= lo) & (s <= hi)
    if sel.sum() < 6:
        return None

    us = np.arange(-half_width, half_width + 1e-9, du)
    pts = c[sel][:, None, :] + n[sel][:, None, :] * us[None, :, None]
    resp = marking.sample(pts[:, :, 0], pts[:, :, 1], fill=0.0)

    span = (resp >= thr).mean(axis=1)          # fraction of the width marked
    if span.max() < min_span:
        return None

    stations = s[sel]
    peaks = _peaks(span, min_span)
    if not peaks:
        return None

    best = None
    for a, b in peaks:
        thick = float(stations[b] - stations[a]) + ds
        if not (thickness_range[0] <= thick <= thickness_range[1]):
            continue
        strength = float(span[a:b + 1].max())
        pos = float(stations[a:b + 1].mean())
        score = strength * (1.0 - abs(thick - 0.45) / 1.2)
        if best is None or score > best[1]:
            best = (pos, score, thick)
    if best is None:
        return None

    pos, score, thick = best
    zebra = _looks_like_zebra([(float(stations[a]), float(stations[b])) for a, b in peaks],
                              zebra_pitch)
    detail = {"thickness_m": round(thick, 2), "span_fraction": round(float(span.max()), 2),
              "n_transverse_bars": len(peaks), "zebra_rejected": zebra}
    if zebra:
        # keep the *most upstream* bar: a stop line sits behind the crossing
        pos = min(p for p, _ in [(float(stations[a:b + 1].mean()), 0) for a, b in peaks])
        score *= 0.6
        detail["note"] = "several transverse bars: crosswalk assumed, upstream bar kept"
    return total - pos, float(np.clip(score, 0, 1)), detail


def _peaks(span: np.ndarray, thr: float) -> list[tuple[int, int]]:
    out, start = [], None
    for i, v in enumerate(span):
        if v >= thr and start is None:
            start = i
        elif v < thr and start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, len(span) - 1))
    return out


def _looks_like_zebra(bars: list[tuple[float, float]], pitch: tuple[float, float]) -> bool:
    if len(bars) < 3:
        return False
    centres = sorted(0.5 * (a + b) for a, b in bars)
    gaps = np.diff(centres)
    return bool(len(gaps) >= 2 and pitch[0] <= float(np.median(gaps)) <= pitch[1]
                and float(np.std(gaps)) < 0.6)


def stop_line_geometry(lane_center: np.ndarray, station_from_end: float,
                       half_width: float) -> np.ndarray:
    """Build the two-point transverse line at a station back from the lane end."""
    c = resample_polyline(np.asarray(lane_center, dtype=float)[:, :2], 0.25)
    s = cumulative_length(c)
    target = max(0.0, float(s[-1]) - station_from_end)
    i = int(np.clip(np.searchsorted(s, target), 1, len(c) - 1))
    n = normals(c)[i]
    p = c[i]
    return np.array([p + n * half_width, p - n * half_width])


def make_stop_line(lid: str, lane_ids: list[str], points: np.ndarray, segment_id: str,
                   observed: bool, score: float, detail: dict) -> StopLine:
    conf = float(np.clip(0.35 + 0.55 * score, 0.1, 0.92)) if observed else 0.3
    flags = [] if observed else ["stop_line_inferred_from_junction_edge"]
    return StopLine(id=lid, points=points, segment_id=segment_id, lane_ids=lane_ids,
                    observed=observed, confidence=conf,
                    provenance=Provenance(Source.IMAGE if observed else Source.INFERRED,
                                          detail),
                    flags=flags)
