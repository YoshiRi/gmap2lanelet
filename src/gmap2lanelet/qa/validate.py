"""Validate the exported map with the real Lanelet2 library.

"Can it be converted to Lanelet2?" is only answered honestly by loading the
file with Lanelet2 itself and building a routing graph from it -- a file that
parses but whose lanelets have no successors is not a map anyone can drive.
"""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)


def validate_lanelet2(path: str | Path, lat0: float, lon0: float) -> dict:
    """Parse + route-check the exported map.  Returns a report dict."""
    path = Path(path)
    out: dict = {"file": str(path), "available": False}
    try:
        import lanelet2  # noqa: F401  -- imported to probe availability
        from lanelet2.io import Origin, loadRobust
        from lanelet2.projection import UtmProjector
    except ImportError as exc:
        out["error"] = f"lanelet2 python bindings not installed ({exc})"
        out["hint"] = "pip install lanelet2"
        return out

    out["available"] = True
    projector = UtmProjector(Origin(lat0, lon0))
    try:
        lmap, errors = loadRobust(str(path), projector)
    except Exception as exc:                                  # noqa: BLE001
        out["parsed"] = False
        out["error"] = str(exc)
        return out

    out["parsed"] = True
    out["parse_errors"] = [str(e) for e in errors][:50]
    out["parse_error_count"] = len(errors)
    out["points"] = len(lmap.pointLayer)
    out["linestrings"] = len(lmap.lineStringLayer)
    out["lanelets"] = len(lmap.laneletLayer)
    out["regulatory_elements"] = len(lmap.regulatoryElementLayer)

    # A regulatory element that parses but that no lanelet references governs
    # nothing.  Counting the round trip -- lanelet -> regelem -> stop line -- is
    # the only way to know the semantic layer actually landed in the map.
    try:
        n_tl, n_ref_line, governed = 0, 0, set()
        for ll in lmap.laneletLayer:
            for re in ll.trafficLights():
                n_tl += 1
                governed.add(int(re.id))
                try:
                    if re.stopLine is not None and len(re.stopLine) >= 2:
                        n_ref_line += 1
                except Exception:                              # noqa: BLE001, S110
                    pass
        out["traffic_lights"] = {
            "regulatory_elements": len(lmap.regulatoryElementLayer),
            "lanelet_links": n_tl,
            "distinct_governing": len(governed),
            "with_stop_line": n_ref_line,
        }
    except Exception as exc:                                  # noqa: BLE001
        out["traffic_light_check_error"] = str(exc)

    try:
        from lanelet2.routing import RoutingGraph
        from lanelet2.traffic_rules import Locations, Participants, create

        rules = create(Locations.Germany, Participants.Vehicle)
        graph = RoutingGraph(lmap, rules)

        passable = 0
        with_succ = 0
        with_pred = 0
        isolated = 0
        for ll in lmap.laneletLayer:
            if not rules.canPass(ll):
                continue
            passable += 1
            ns = len(graph.following(ll))
            np_ = len(graph.previous(ll))
            with_succ += ns > 0
            with_pred += np_ > 0
            isolated += (ns == 0 and np_ == 0)

        out["routing"] = {
            "passable_lanelets": passable,
            "with_successor": with_succ,
            "with_predecessor": with_pred,
            "isolated": isolated,
            "connected_fraction": round(1 - isolated / passable, 4) if passable else 0.0,
        }
    except Exception as exc:                                  # noqa: BLE001
        out["routing_error"] = str(exc)

    log.info("lanelet2 validation: parsed=%s lanelets=%s routing=%s",
             out.get("parsed"), out.get("lanelets"), out.get("routing"))
    return out
