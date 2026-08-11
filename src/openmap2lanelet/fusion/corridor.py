"""Corridor extraction: where the pavement actually is.

Given a prior centreline and the rectified evidence, decide for every station
which lateral interval is drivable surface.  Three things make this more than a
threshold:

* **the prior is misplaced.**  OSM centrelines are routinely 2-10 m off, and on
  divided roads they are drawn down the middle of *both* carriageways.  So the
  corridor is searched in a window around the prior, and the resulting lateral
  offset is the geometry correction the whole PoC is about.
* **pavement does not stop at the road.**  Parking aisles, forecourts and
  driveways are the same asphalt.  The prior's road class bounds how wide the
  corridor is allowed to get, and exceeding that bound is *reported* rather
  than silently accepted.
* **divided roads have two corridors.**  A single non-oneway OSM way whose
  pavement comes in two runs separated by a median is a dual carriageway, and
  is emitted as two corridors with opposite directions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from .profile import Profile, median_along, smooth_along

log = logging.getLogger(__name__)


@dataclass
class Corridor:
    """One paved carriageway tracked along a prior edge."""

    left: np.ndarray            # (S,) lateral offset of the left edge (+left)
    right: np.ndarray           # (S,) lateral offset of the right edge
    valid: np.ndarray           # (S,) bool
    side: str = "single"        # "single" | "left" | "right" (of the prior line)
    clamped: np.ndarray | None = None   # (S,) bool: width hit the prior-derived cap
    detail: dict = field(default_factory=dict)

    @property
    def width(self) -> np.ndarray:
        return self.left - self.right

    @property
    def center_offset(self) -> np.ndarray:
        return 0.5 * (self.left + self.right)

    @property
    def coverage(self) -> float:
        return float(self.valid.mean()) if len(self.valid) else 0.0

    def mean_width(self) -> float:
        w = self.width[self.valid]
        return float(np.median(w)) if w.size else 0.0


def extract_corridors(prof: Profile, *, expected_width: float, oneway: bool,
                      road_thr: float = 0.45, max_shift: float = 12.0,
                      bleed_factor: float = 1.7, min_width: float = 2.6,
                      gap_tolerance: float = 0.8, smooth_m: float = 9.0,
                      median_gap_max: float = 30.0) -> list[Corridor]:
    """Return one corridor, or two for a detected dual carriageway."""
    road = smooth_along(prof.road, max(3, int(round(0.75 / _du(prof)))), axis=1)
    S, U = road.shape
    du = _du(prof)
    max_half = 0.5 * expected_width * bleed_factor
    gap_px = max(1, int(round(gap_tolerance / du)))

    prim_l = np.full(S, np.nan)
    prim_r = np.full(S, np.nan)
    sec_l = np.full(S, np.nan)
    sec_r = np.full(S, np.nan)

    for i in range(S):
        runs = _runs(road[i] > road_thr, gap_px)
        runs = [(a, b) for a, b in runs if (b - a + 1) * du >= min_width]
        if not runs:
            continue
        # offsets ascend with index, so run (a, b) spans [offsets[a], offsets[b]]
        cand = []
        for a, b in runs:
            lo, hi = float(prof.offsets[a]), float(prof.offsets[b])
            if lo > max_shift or hi < -max_shift:
                continue
            dist = 0.0 if lo <= 0 <= hi else min(abs(lo), abs(hi))
            cand.append((dist, lo, hi, float(road[i, a:b + 1].mean())))
        if not cand:
            continue
        cand.sort(key=lambda c: (c[0], -c[3]))
        _, lo, hi, _ = cand[0]
        prim_r[i], prim_l[i] = lo, hi

        # A run on the far side of the prior line, separated by a median-sized
        # gap, is the other carriageway.  If the primary run already straddles
        # the prior line the road is undivided and there is nothing to pair.
        if lo <= 0 <= hi:
            continue
        want_left = hi < 0          # primary lies right of the prior line
        best: tuple[float, float, float] | None = None
        for _, lo2, hi2, _score in cand[1:]:
            gap = (lo2 - hi) if want_left else (lo - hi2)
            if not (want_left and lo2 > hi) and not (not want_left and hi2 < lo):
                continue
            if 0 <= gap <= median_gap_max and (best is None or gap < best[0]):
                best = (gap, lo2, hi2)
        if best is not None:
            sec_r[i], sec_l[i] = best[1], best[2]

    dual = _looks_dual(prim_l, prim_r, sec_l, sec_r, expected_width, oneway)
    if dual:
        # Each carriageway carries about half the expected total width.
        half_cap = 0.25 * expected_width * bleed_factor
        a = _finalise(prim_l, prim_r, max_half=half_cap, du=du, smooth_m=smooth_m,
                      ds=_ds(prof), kappa=prof.curvature)
        b = _finalise(sec_l, sec_r, max_half=half_cap, du=du, smooth_m=smooth_m,
                      ds=_ds(prof), kappa=prof.curvature)
        # label which carriageway sits left / right of the prior line
        ca = np.nanmedian(a.center_offset[a.valid]) if a.valid.any() else 0.0
        cb = np.nanmedian(b.center_offset[b.valid]) if b.valid.any() else 0.0
        a.side, b.side = ("left", "right") if ca > cb else ("right", "left")
        for c in (a, b):
            c.detail["dual_carriageway"] = True
        log.debug("dual carriageway detected (offsets %.1f / %.1f m)", ca, cb)
        return [a, b]

    c = _finalise(prim_l, prim_r, max_half=max_half, du=du, smooth_m=smooth_m,
                  ds=_ds(prof), kappa=prof.curvature)
    c.detail["dual_carriageway"] = False
    return [c]


# --------------------------------------------------------------------------- #


def _du(prof: Profile) -> float:
    return float(prof.offsets[1] - prof.offsets[0])


def _ds(prof: Profile) -> float:
    return float(prof.stations[1] - prof.stations[0]) if len(prof.stations) > 1 else 1.0


def _runs(mask: np.ndarray, gap_px: int) -> list[tuple[int, int]]:
    """Index runs of True, bridging gaps up to ``gap_px`` (markings dim the
    road model, and a painted line must not split a carriageway)."""
    m = mask.copy()
    if gap_px > 0:
        idx = np.flatnonzero(m)
        for a, b in zip(idx, idx[1:]):
            if 1 < b - a <= gap_px + 1:
                m[a:b] = True
    out = []
    start = None
    for i, v in enumerate(m):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, len(m) - 1))
    return out


def _looks_dual(pl, pr, sl, sr, expected_width: float, oneway: bool) -> bool:
    """Decide at segment level whether the two runs are two carriageways."""
    if oneway:
        return False
    have = np.isfinite(sl) & np.isfinite(pl)
    if have.mean() < 0.55:
        return False
    w1 = np.nanmedian((pl - pr)[have])
    w2 = np.nanmedian((sl - sr)[have])
    if not np.isfinite(w1) or not np.isfinite(w2):
        return False
    # Both runs must look like carriageways, and neither alone should already
    # account for the whole expected width.
    if min(w1, w2) < 4.5:
        return False
    if max(w1, w2) > 0.95 * expected_width:
        return False
    return (w1 + w2) > 0.7 * expected_width


def _finalise(l: np.ndarray, r: np.ndarray, *, max_half: float, du: float,
              smooth_m: float, ds: float, kappa: np.ndarray | None = None,
              fold_margin: float = 0.75) -> Corridor:
    """Clamp, fill gaps and smooth the raw per-station edges.

    ``kappa`` is the signed curvature of the reference line.  Offsetting a
    polyline further than its radius of curvature folds it into a loop -- which
    is how a lane centreline ends up crossing itself at a tight corner -- so
    each side is additionally capped at ``fold_margin / |kappa|`` on whichever
    side is the inside of the bend."""
    valid = np.isfinite(l) & np.isfinite(r)
    if not valid.any():
        S = len(l)
        return Corridor(np.zeros(S), np.zeros(S), np.zeros(S, bool))

    c = 0.5 * (l + r)
    half = 0.5 * (l - r)
    clamped = np.zeros(len(l), dtype=bool)
    over = valid & (half > max_half)
    clamped[over] = True
    half = np.where(over, max_half, half)

    if kappa is not None and len(kappa) == len(l):
        # Offsetting past the radius of curvature folds the offset curve into a
        # loop.  It is the *total* offset that matters, not the half width, so
        # clamp whichever boundary lies on the inside of the bend and rebuild
        # the centre and half width from the clamped pair.
        with np.errstate(divide="ignore"):
            lim = np.where(np.abs(kappa) > 1e-6,
                           fold_margin / np.maximum(np.abs(kappa), 1e-9), np.inf)
        li = np.where(kappa > 0, np.minimum(np.nan_to_num(l, nan=0.0), lim), l)
        ri = np.where(kappa < 0, np.maximum(np.nan_to_num(r, nan=0.0), -lim), r)
        li = np.maximum(li, ri + 1.0)
        c = 0.5 * (li + ri)
        half = np.minimum(half, 0.5 * (li - ri))

    win = max(3, int(round(smooth_m / max(ds, 1e-6))) | 1)
    c_f = _fill(c, valid)
    h_f = _fill(half, valid)
    c_s = smooth_along(median_along(c_f, win), max(3, win // 2))
    h_s = smooth_along(median_along(h_f, win), max(3, win // 2))

    return Corridor(left=c_s + h_s, right=c_s - h_s, valid=valid, clamped=clamped,
                    detail={"clamped_fraction": round(float(clamped[valid].mean()), 3)
                            if valid.any() else 0.0})


def _fill(a: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Linear interpolation across invalid stations (edge-held at the ends)."""
    idx = np.arange(len(a))
    if not valid.any():
        return np.zeros_like(a)
    return np.interp(idx, idx[valid], a[valid])
