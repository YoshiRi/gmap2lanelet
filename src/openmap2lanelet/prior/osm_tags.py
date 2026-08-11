"""Interpretation of OSM road tags.

Everything here is *prior* knowledge: what the map says, plus the modelling
defaults we fall back on when it says nothing.  Each accessor returns the value
together with the :class:`Source` that produced it, so the fusion stage can
tell "OSM asserted 4 lanes" apart from "we assumed 2 because it is residential".
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..types import Source

# highway class -> (default total lanes, default lane width m, default speed km/h)
CLASS_DEFAULTS: dict[str, tuple[int, float, float]] = {
    "motorway": (4, 3.65, 110),
    "motorway_link": (1, 4.0, 60),
    "trunk": (4, 3.5, 90),
    "trunk_link": (1, 4.0, 50),
    "primary": (4, 3.5, 60),
    "primary_link": (1, 4.0, 40),
    "secondary": (2, 3.4, 50),
    "secondary_link": (1, 3.8, 40),
    "tertiary": (2, 3.3, 40),
    "tertiary_link": (1, 3.8, 30),
    "unclassified": (2, 3.2, 30),
    "residential": (2, 3.0, 30),
    "living_street": (1, 3.5, 15),
    "service": (1, 3.0, 15),
    "track": (1, 3.0, 15),
}
FALLBACK = (2, 3.25, 30)

# Classes that are unlikely to carry painted lane markings at all.  Used to
# suppress the "no markings visible" failure flag where absence is expected.
UNMARKED_CLASSES = {"service", "track", "living_street", "residential", "unclassified"}


@dataclass
class Valued:
    """A value with its provenance."""

    value: object
    source: Source
    detail: dict | None = None

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Valued({self.value!r}, {self.source.value})"


def highway_class(tags: dict[str, str]) -> str:
    return tags.get("highway", "unclassified")


def defaults_for(tags: dict[str, str]) -> tuple[int, float, float]:
    return CLASS_DEFAULTS.get(highway_class(tags), FALLBACK)


def is_oneway(tags: dict[str, str]) -> Valued:
    v = (tags.get("oneway") or "").lower()
    if v in {"yes", "true", "1", "-1"}:
        return Valued(True, Source.OSM, {"oneway": v})
    if v in {"no", "false", "0"}:
        return Valued(False, Source.OSM, {"oneway": v})
    if highway_class(tags) in {"motorway", "motorway_link", "trunk_link",
                               "primary_link", "secondary_link", "tertiary_link"}:
        return Valued(True, Source.DEFAULT, {"reason": "link/motorway default"})
    if (tags.get("junction") or "").lower() in {"roundabout", "circular"}:
        return Valued(True, Source.DEFAULT, {"reason": "roundabout"})
    return Valued(False, Source.DEFAULT, {"reason": "class default"})


def _int(v: str | None) -> int | None:
    if v is None:
        return None
    m = re.match(r"^\s*(\d+)", str(v))
    return int(m.group(1)) if m else None


def lane_count(tags: dict[str, str]) -> Valued:
    """Total lane count across the carriageway described by this way."""
    n = _int(tags.get("lanes"))
    if n and n > 0:
        return Valued(n, Source.OSM, {"tag": "lanes"})

    fwd = _int(tags.get("lanes:forward"))
    bwd = _int(tags.get("lanes:backward"))
    if fwd or bwd:
        return Valued((fwd or 0) + (bwd or 0), Source.OSM,
                      {"tag": "lanes:forward/backward", "forward": fwd, "backward": bwd})

    n_def = defaults_for(tags)[0]
    if is_oneway(tags).value:
        n_def = max(1, n_def // 2)
    return Valued(n_def, Source.DEFAULT, {"reason": f"class={highway_class(tags)}"})


def directional_split(tags: dict[str, str], total: int) -> tuple[int, int]:
    """Split ``total`` lanes into (forward, backward)."""
    if is_oneway(tags).value:
        return total, 0
    fwd = _int(tags.get("lanes:forward"))
    bwd = _int(tags.get("lanes:backward"))
    if fwd is not None and bwd is not None and fwd + bwd == total:
        return fwd, bwd
    if fwd is not None and 0 < fwd < total:
        return fwd, total - fwd
    if bwd is not None and 0 < bwd < total:
        return total - bwd, bwd
    if total == 1:
        # A single-lane two-way road: model it as one bidirectional lane.
        return 1, 0
    fwd = total // 2
    return fwd, total - fwd


def lane_width(tags: dict[str, str]) -> Valued:
    w = tags.get("width:lanes") or tags.get("lane_width")
    if w:
        try:
            return Valued(float(str(w).split("|")[0]), Source.OSM, {"tag": "width:lanes"})
        except ValueError:
            pass
    return Valued(defaults_for(tags)[1], Source.DEFAULT, {"reason": highway_class(tags)})


def speed_limit_kph(tags: dict[str, str]) -> Valued:
    v = tags.get("maxspeed")
    if v:
        s = str(v).strip().lower()
        m = re.match(r"^(\d+(?:\.\d+)?)\s*(mph)?$", s)
        if m:
            val = float(m.group(1))
            if m.group(2) == "mph":
                val *= 1.609344
            return Valued(round(val, 1), Source.OSM, {"tag": "maxspeed", "raw": v})
    return Valued(defaults_for(tags)[2], Source.DEFAULT, {"reason": highway_class(tags)})


TURN_TOKENS = {"left", "slight_left", "sharp_left", "through", "right",
               "slight_right", "sharp_right", "merge_to_left", "merge_to_right",
               "reverse", "none", ""}


def turn_lanes(tags: dict[str, str], key: str = "turn:lanes") -> Valued | None:
    """Parse ``turn:lanes`` style tags into a per-lane list of manoeuvre sets.

    Returns ``None`` when the tag is absent -- which, in practice, is almost
    always: this is one of the concrete public-data gaps the PoC reports.
    """
    raw = tags.get(key)
    if not raw:
        return None
    lanes = []
    for spec in str(raw).split("|"):
        toks = {t for t in spec.split(";") if t in TURN_TOKENS} - {"", "none"}
        lanes.append(toks or {"through"})
    return Valued(lanes, Source.OSM, {"tag": key, "raw": raw})


# How far past the class-expected width the observed pavement may extend before
# we stop believing it is carriageway.  A parking aisle is genuinely one lane
# wide and everything beyond it is bays, so minor classes get a tight bound;
# an arterial may legitimately have turn pockets and shoulders.
BLEED_BY_CLASS: dict[str, float] = {
    "service": 1.20, "track": 1.20, "living_street": 1.30,
    "unclassified": 1.35, "residential": 1.40,
}
DEFAULT_BLEED = 1.65


def bleed_factor(tags: dict[str, str]) -> float:
    return BLEED_BY_CLASS.get(highway_class(tags), DEFAULT_BLEED)


def road_group(tags: dict[str, str]) -> str:
    """Coarse grouping used for reporting: results differ hugely between them."""
    h = highway_class(tags)
    if h in {"motorway", "trunk", "primary", "secondary", "tertiary",
             "motorway_link", "trunk_link", "primary_link", "secondary_link",
             "tertiary_link"}:
        return "major"
    if h in {"residential", "living_street"}:
        return "residential"
    return "minor"


def expected_carriageway_width(tags: dict[str, str]) -> float:
    """Rough expected paved width (m), used to bound the corridor search."""
    n = int(lane_count(tags).value)
    w = float(lane_width(tags).value)
    shoulder = 1.5 if highway_class(tags) in {"motorway", "trunk", "primary"} else 0.8
    return n * w + 2 * shoulder
