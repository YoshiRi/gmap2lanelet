"""Markdown run report."""

from __future__ import annotations

from pathlib import Path

from .failures import CATALOGUE, SEVERITY_ORDER


def write_report(result, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    s = result.summary()
    st = result.failures.stats
    L: list[str] = []

    L.append(f"# openmap2lanelet run report - {result.aoi.name}\n")
    L.append(f"AOI `{result.aoi.west:.6f}, {result.aoi.south:.6f}, "
             f"{result.aoi.east:.6f}, {result.aoi.north:.6f}` (WGS84)\n")

    L.append("## Inputs\n")
    L.append("| | |\n|---|---|")
    L.append(f"| imagery | {s['imagery']['attribution']} |")
    L.append(f"| imagery size | {s['imagery']['size_px'][1]} x {s['imagery']['size_px'][0]} px "
             f"@ {s['imagery']['gsd']} m/px |")
    L.append(f"| prior | {s['prior']['attribution']} ({s['prior']['kind']}) |")
    L.append(f"| prior graph | {s['prior']['edges']} edges / {s['prior']['edge_km']} km, "
             f"{s['prior']['junction_clusters']} junction clusters |")
    L.append(f"| prior lane tags | `lanes` on {s['prior']['with_lanes_tag']}/"
             f"{s['prior']['edges']}, `turn:lanes` on {s['prior']['with_turn_lanes_tag']}/"
             f"{s['prior']['edges']}, `maxspeed` on {s['prior']['with_maxspeed_tag']}/"
             f"{s['prior']['edges']} |")
    L.append(f"| observation backend | `{s['observation']['backend']}` |\n")

    L.append("## Output\n")
    g = s["graph"]
    L.append("| | |\n|---|---|")
    L.append(f"| lanes | {g['lanes']} ({g['road_lanes']} road + {g['turn_lanes']} turn) |")
    L.append(f"| lane length | {g['lane_km']} km |")
    L.append(f"| lane boundaries | {g['boundaries']} |")
    L.append(f"| intersections | {g['intersections']} |")
    L.append(f"| mean lane confidence | {g['mean_confidence']} |\n")

    v = result.validation
    L.append("## Lanelet2 validation\n")
    if not v.get("available"):
        L.append(f"Lanelet2 bindings unavailable: {v.get('error')}\n")
    elif not v.get("parsed"):
        L.append(f"**Failed to parse**: {v.get('error')}\n")
    else:
        r = v.get("routing", {})
        L.append(f"Loaded by the official Lanelet2 library with "
                 f"**{v['parse_error_count']} parse errors**: "
                 f"{v['points']} points, {v['linestrings']} linestrings, "
                 f"{v['lanelets']} lanelets.\n")
        if r:
            L.append(f"Routing graph (vehicle rules): {r['passable_lanelets']} passable "
                     f"lanelets, {r['with_successor']} with a successor, "
                     f"{r['with_predecessor']} with a predecessor, {r['isolated']} isolated "
                     f"-> **{100 * r['connected_fraction']:.1f}% connected**.\n")

    L.append("## What each source decided\n")
    src = st["lane_count_source"]
    L.append("Lane count provenance (share of carriageways):\n")
    L.append("| source | share |\n|---|---|")
    for k, val in src.items():
        L.append(f"| {k} | {100 * val:.1f}% |")
    L.append("")
    L.append(f"- carriageways where lane markings were observed at all: "
             f"**{100 * st['carriageways_with_observed_markings']:.1f}%**")
    L.append(f"- carriageways where image and OSM disagreed on lane count: "
             f"**{100 * st['carriageways_with_lane_count_conflict']:.1f}%**")
    L.append(f"- lane boundaries backed by an observed painted line: "
             f"**{100 * st['boundaries_observed_fraction']:.1f}%** "
             f"(virtual: {100 * st['boundaries_virtual_fraction']:.1f}%)")
    L.append(f"- dual carriageways detected from the imagery: "
             f"**{st['dual_carriageways_detected']}**")
    L.append(f"- geometry correction applied to the prior: median "
             f"**{st['prior_shift_median_m']} m**, p90 **{st['prior_shift_p90_m']} m**\n")

    L.append("## Failure modes observed\n")
    if not result.failures.items:
        L.append("No review items were raised.\n")
    else:
        L.append("| code | n | what it means |\n|---|---|---|")
        for kind, n in sorted(result.failures.counts.items(), key=lambda kv: -kv[1]):
            L.append(f"| `{kind}` | {n} | {CATALOGUE.get(kind, '')} |")
        L.append("")

        L.append("### Highest-severity items\n")
        for it in sorted(result.failures.items,
                         key=lambda i: SEVERITY_ORDER.get(i.severity, 3))[:15]:
            lon, lat = result.frame.to_wgs84(it.position[0], it.position[1])
            L.append(f"- **{it.id}** `{it.kind}` ({it.severity}) at "
                     f"{float(lat):.6f}, {float(lon):.6f} - {it.message}")
        L.append("")

    L.append("## Timings\n")
    L.append("| stage | s |\n|---|---|")
    for k, val in s["timings_s"].items():
        L.append(f"| {k} | {val} |")
    L.append("")

    L.append("## Files\n")
    for k, val in result.paths.items():
        if isinstance(val, str):
            L.append(f"- `{k}`: {val}")
    L.append("")

    path.write_text("\n".join(L), encoding="utf-8")
    return path
