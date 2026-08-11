# openmap2lanelet

[![CI](https://github.com/YoshiRi/gmap2lanelet/actions/workflows/ci.yml/badge.svg)](https://github.com/YoshiRi/gmap2lanelet/actions/workflows/ci.yml)

**Lane-level vector maps from public map data + public aerial imagery, exported as Lanelet2.**

A proof of concept for one hypothesis:

> Using existing road data (OSM) as a **topology prior** and correcting it with
> **geometry observed in aerial/satellite imagery**, a usable lane-level map can be
> semi-automatically generated from public information alone.

The point of this PoC is **not** accuracy. It is to establish, with numbers,
*what can be automated and what cannot* — and to make everything that cannot be
automated visible, georeferenced and reviewable.

The name is literal: every source this project touches (SpaceNet, OpenStreetMap,
Argoverse 2, and whatever XYZ tile service you point it at) is openly licensed.
No Google product is used or required anywhere in this repository.

Give it a place, get back a Lanelet2 map candidate plus a list of the places a
human has to look at.

```bash
pip install -e ".[lanelet2]"

# a public SpaceNet tile: 0.27 m/px imagery + OSM-like road vectors, no credentials
openmap2lanelet run --image-id 38 --out outputs/vegas38

# anywhere on Earth: OSM via Overpass + an aerial tile service you are licensed to use
openmap2lanelet run --source live \
    --bbox 139.7601 35.6801 139.7649 35.6841 --drive-on left \
    --tiles "https://<your-imagery-host>/{z}/{y}/{x}" --tiles-attribution "..." \
    --out outputs/tokyo

# compare failure modes across many AOIs
openmap2lanelet batch --image-ids 93,162,48,10,100,151,89,38,124,160 --out outputs/batch

# add the semantics that overhead imagery cannot see: traffic lights, the stop
# lines they govern, the lanes behind them, and painted lane arrows
openmap2lanelet street --log 20dd185d-b4eb-3024-a17a-b4e5d8b15b65 --city DTW \
    --out outputs/street_detroit
```

![imagery + OSM prior + generated lanes](docs/figures/overlay_img38.jpg)

*Cyan dashed: OSM prior. Yellow: painted lines detected in the imagery (solid /
dashed). Cyan solid: observed road edge. Magenta dotted: virtual boundary, i.e.
no paint was visible and the split is inferred. Green/blue/purple: lane
centrelines coloured by which source decided the lane. Red circles: junctions,
whose interior connectivity is inferred rather than observed.*

---

## Inputs and outputs

### What you have to supply

| | required? | what it is | where it plugs in |
|---|---|---|---|
| **AOI** | yes | a lon/lat bounding box, a few hundred metres square | `AOI(name, west, south, east, north)` |
| **road prior** | yes | road **centrelines** in WGS84 with OSM-style tags. Only `highway=` is really needed; `oneway`, `lanes`, `maxspeed`, `turn:lanes` are used when present and their absence is reported, not papered over | `PriorData(ways=[PriorWay(id, coords, tags)], …)` |
| **overhead raster** | yes | RGB, north-up, georeferenced, ideally ≤ 0.3 m/px. Anything coarser than ~0.5 m/px will not resolve lane paint | `ImageryData(raster=GeoRaster(...), gsd, attribution)` |
| **posed street frames** | only for `street` | images **with camera poses** in a known frame plus intrinsics. Pose is the whole point: an image without one can be classified but not mapped | `StreetSequence(frames=[StreetFrame(id, camera, image_path)], …)` |
| **ground-height surface** | optional | any raster DEM/DTM. Without it, IPM assumes a plane and heights above ground cannot be checked | `GroundSurface(heights, x0, y0, res)` |

Three loaders ship in the box (`sources/`): SpaceNet-3 (imagery **and** an
OSM-shaped centreline graph, no credentials), Overpass or a local `.osm` file,
and any XYZ tile template. `street/sources/` adds Argoverse 2. All four are
thin adapters onto the dataclasses above — writing a fifth is ~50 lines.

The pipeline itself is source-agnostic: it never touches the network, never
assumes a projection beyond "north-up lon/lat", and works entirely in a local
metric frame it builds from the AOI.

### What comes out

| file | contents |
|---|---|
| `lanelet2_map.osm` | the map. Lanelet2 OSM-XML: nodes, `line_thin`/`road_border`/`virtual` ways, `type=lanelet` relations. Every element tagged `gm2ll:source`, `gm2ll:confidence`, and `gm2ll:review=yes` where a human should look |
| `lanelet2_map_semantic.osm` | *(street stage)* the same, plus `type=regulatory_element` / `subtype=traffic_light` relations linking signal → stop line → governed lanelets |
| `viewer.html` | self-contained offline viewer: imagery, prior, lanes, evidence layers and review pins, no server and no CDN |
| `report.md` | human-readable run report |
| `review_items.json` | **the work queue** — every place the pipeline was unsure, with a WGS84 position, a failure code, a severity and an explanation |
| `street_review.json` | the same for the street stage (unassigned signals, ambiguous approaches, inferred stop lines, arrow conflicts) |
| `lane_graph.json`, `semantics.json` | the format-neutral intermediate representation, if you want to render or re-export it yourself |
| `summary.json`, `evaluation.json` | all statistics, per element |
| `viz/*.png` | overlays, evidence layers, the road-aligned debug view, semantic overlay, signal reprojection audit, and a crop per review item |

**Nothing is emitted without provenance.** `Source.OSM` = the map decided it,
`IMAGE` = the imagery measured it, `FUSED` = they agreed, `INFERRED` = a rule
invented it, `DEFAULT` = neither source said anything. That distinction is the
deliverable as much as the geometry is.

## The idea

The public sources fail in *opposite* ways, which is what makes fusing them
worthwhile:

| | OSM / map data | Aerial imagery | Street-level imagery |
|---|---|---|---|
| connectivity, one-way, road class | **reliable** | absent | absent |
| lane count | present but often nominal | measurable where paint is visible | measurable, but only where driven |
| geometry (position, width, shape) | 1–2 m off, routinely worse | **reliable** | high resolution, corridor only |
| lane-level turn permissions | essentially never tagged | not visible at 0.3 m/px | **readable from paint** |
| traffic lights, signs, stop lines | not mapped to lanes | invisible | **the only source** |

So the pipeline never asks the imagery what connects to what, never asks the map
where anything is, and never asks the overhead view what a lane *means*:

```
OSM / map data ────► topology prior ──┐
                                      ├──► lane graph ──┐
aerial imagery ────► geometry evidence┘                 ├──► Lanelet2
                                                        │    (+ regulatory
street imagery ────► semantic evidence ─────────────────┘      elements)
                     (signals, stop lines, arrows)      │
                                                        └──► review items
```

The mechanism that makes this work is a **road-aligned frame**. Every prior edge
defines a curvilinear coordinate system (arc length `s` along it, lateral offset
`u` across it) and all evidence is rectified into it. Finding lane boundaries
then stops being 2-D curve tracing and becomes 1-D peak finding, and the prior
only has to be right enough to point the cross-section in the right direction —
its position can be, and is, wrong.

## Pipeline

| stage | module | what it does |
|---|---|---|
| acquisition | `sources/` | Overpass / `.osm` file / SpaceNet for the prior; XYZ tiles / SpaceNet for imagery |
| topology prior | `prior/` | project, split ways at shared junction nodes, drop stubs, cluster junctions |
| observation | `observation/` | road-surface probability + lane-marking response (pluggable backend) |
| profiling | `fusion/profile.py` | rectify evidence into the road-aligned frame |
| corridor | `fusion/corridor.py` | find the pavement, correct the prior's lateral error, detect dual carriageways |
| lane structure | `fusion/lanes.py` | lane boundaries from markings, lane count arbitrated against OSM |
| intersections | `fusion/intersection.py` | trim approaches, infer turn connectivity, generate turn lanes |
| assembly | `fusion/builder.py` | stitch lanes into a connected graph |
| export | `export/lanelet2_osm.py` | Lanelet2 OSM-XML with provenance tags and regulatory elements |
| QA | `qa/` | failure detection, review items, Lanelet2 round-trip validation |
| viz | `viz/` | static overlays + a self-contained interactive viewer |

Street-level stage (phase 2), which consumes the lane graph above:

| stage | module | what it does |
|---|---|---|
| posed imagery | `street/sources/` | frames + calibrated poses + a ground-height surface (Argoverse 2; the interface takes any posed source) |
| detection | `street/detect/` | signal heads and signs per frame (off-the-shelf YOLOv8, no fine-tuning) |
| geolocation | `street/semantics/landmarks.py` | seed-and-grow data association, multi-view triangulation with RANSAC, per-object σ |
| rectification | `street/geo/ipm.py` | inverse perspective mapping onto the real ground surface → a 5 cm/px overhead mosaic |
| stop lines, arrows | `street/semantics/` | transverse-bar detection and geometric arrow classification in the road-aligned frame |
| association | `street/semantics/associate.py` | signal → approach → stop line → controlled lanelets, decided in the approach frame using facing |
| evaluation | `street/evaluate.py` | scores against a held-back HD map |

Every stage attaches `source` and `confidence` to what it produces, and every
intersection interior is `Source.INFERRED` by construction — see
[what comes out](#what-comes-out).

## Results

10 AOIs in Las Vegas (SpaceNet-3, 0.27 m/px, ~316 × 390 m each), 35.8 km of
prior road → **3 118 lanelets / 133 lane-km**.

| | major roads | residential | minor (service, parking aisles) |
|---|---|---|---|
| prior length | 6.0 km | 8.7 km | 17.8 km |
| lane markings observed | **88.7 %** | 62.1 % | 51.8 % |
| lane count agrees with OSM (when markings seen) | 39.4 % | 27.0 % | 9.7 % |
| median geometry correction applied to the prior | 1.41 m | 1.27 m | 1.40 m |

Lanelet2 export: **0 parse errors on all 10 AOIs**, routing graph builds,
84–96 % of lanelets reach a successor or predecessor (median 91 %).

Public-data gaps found: `turn:lanes` present on **0 / 715** prior edges,
`maxspeed` on **0 / 715**.

Full analysis, including every failure mode with worked examples:
**[docs/RESULTS.md](docs/RESULTS.md)**.

### Phase 2 — street-level semantics

One Detroit log (Argoverse 2, 133 posed frames), with the HD map degraded to
OSM information content as the input prior and held back as ground truth:

| | |
|---|---|
| signal heads located in 3-D | **32**, median 22 views, median σ 2.24 m |
| recovered height distribution | median 4.79 m; a mast-arm mode near 5 m and a pedestrian mode near 2.7 m |
| signals tied to stop line + controlled lanelets | **17 (53 %)**, governing 121 lanelets |
| stop bars observed | 6 of 14 approaches, **2.48 m** median offset from the crosswalk they precede |
| Lanelet2 export | 17 `regulatory_element` relations, 0 parse errors, all 121 links resolve to a stop line |
| painted arrows vs the HD map's real connectivity | precision **1.00**, recall 0.53 |

The arrow result is the interesting one: **paint never claims an illegal
manoeuvre but reports only about half the legal ones**, so it is treated as
evidence that a manoeuvre is permitted — not that the others are forbidden. The
union of paint and the phase-1 convention beats either alone (IoU 0.83 vs 0.77
and 0.53).

The dominant cause of an unassigned signal is not recognition: 7 of 15 are
unassigned because their junction arm was never driven, so the geometry stage
produced no lanes to attach them to. Street imagery is a *linear* sample of a
*planar* problem.

**[docs/STREET_RESULTS.md](docs/STREET_RESULTS.md)** — full analysis, including
what cannot be scored at all (no open dataset with posed street imagery
annotates traffic lights, so no positional accuracy figure is reported, only the
evidence).

## What works today, and what does not

The whole point of the PoC. Every claim below is backed by a number in
[RESULTS](docs/RESULTS.md) or [STREET_RESULTS](docs/STREET_RESULTS.md).

### ✅ Works — automatable end to end

| capability | evidence |
|---|---|
| specify a place → a Lanelet2 map candidate comes out | 11 AOIs, **0 parse errors**, the real Lanelet2 library loads every one and builds a routing graph |
| find the drivable surface and correct the prior's lateral error | median correction applied 1.3–1.4 m; the prior is used for *direction*, never for position |
| lane boundary geometry where paint is visible | markings observed on **88.7 %** of major carriageways (62 % residential, 52 % service) |
| dual-carriageway detection | separates the two directions of a divided road from the single OSM way |
| routable topology | 84–96 % of lanelets reach a successor or predecessor (median 91 %) |
| locate traffic lights in 3-D from posed street imagery | **32** heads, median 22 views, median σ 2.24 m; the recovered height distribution independently splits into a ~5 m mast-arm mode and a ~2.7 m pedestrian mode |
| place stop lines where a bar is painted and visible | **2.48 m** median offset from the crosswalk they precede — physically correct |
| tie signal → stop line → controlled lanelets, and export it | **17** Lanelet2 `regulatory_element` relations, 121 lanelet links, every one resolving to a stop line |
| read painted lane arrows without training data | **precision 1.00** against real junction connectivity — the paint never claimed an illegal manoeuvre |
| say where it is unsure | 153 + 48 georeferenced review items with severity and an explanation; `gm2ll:review=yes` in the map so JOSM can filter for them |

### ⚠️ Partly works — usable, but do not trust it unattended

| capability | state |
|---|---|
| **lane count** | ±1 lane on **87 %** of carriageways, exactly right on 39 %. Image and prior disagree on ~60 % of major carriageways and **nothing in public data breaks the tie** |
| **intersection connectivity** | every turn lane is a *rule* ("leftmost turns left"), not a measurement. Arrows add evidence — precision 1.00, **recall 0.53** — so they confirm but do not complete it |
| **signal → lane assignment** | **53 %** assigned. Of the 15 unassigned, 7 fail because their junction arm was never driven and the geometry stage produced no lanes to attach to, 6 are correctly-excluded pedestrian heads, 2 are genuinely ambiguous |
| **speed limits** | `maxspeed` was absent on **0 / 715** edges, so every value is a class default |

### ❌ Does not work / out of scope today

| gap | why |
|---|---|
| **signal aspect** (arrow signal vs ball signal) | the lens state is never classified, so a left-turn signal and a through signal on the same mast arm are indistinguishable. This is the single biggest missing piece |
| **signal grouping** | four heads on one arm displaying the same aspect are one regulatory element in reality and four here — grouping needs the aspect |
| **sign semantics** | detection only (COCO stop signs). No speed-limit, no-turn or regulatory sign is read or attached |
| **anything outside the driven corridor** | street imagery is a *linear* sample of a *planar* problem: 33 % of generated geometry there has no true lane near it, and 2 of 6 junctions produced no turn lanes at all |
| **elevation** | `PipelineConfig.elevation` is a single constant. The street stage measures real per-object heights, but the lane graph is flat |
| **crosswalks, cycle lanes, bus stops, parking** | not modelled; the drivable network only |
| **OpenDRIVE** | Lanelet2 only, though `LaneGraph` is format-neutral by construction |
| **traffic-light positional accuracy** | **unmeasurable**: no openly-licensed dataset with posed street imagery annotates traffic lights. Evidence is reported; a fabricated accuracy figure is not |

### The one-line answer

> **Geometry is largely automatable. Meaning is partly automatable. Neither
> lane counts nor intersection connectivity can be *closed* from public data
> alone — but both can be measured, bounded and handed to a human as a ranked,
> georeferenced work queue.**

## Documentation

- [docs/RESULTS.md](docs/RESULTS.md) — evaluation, failure catalogue, what to fix next
- [docs/STREET_RESULTS.md](docs/STREET_RESULTS.md) — street-level semantics: signals, stop lines, arrows
- [docs/DESIGN.md](docs/DESIGN.md) — architecture and algorithms
- [docs/PRIOR_WORK.md](docs/PRIOR_WORK.md) — DeepAerialMapper, SIO-Mapper, Lanelet2 and what was reused
- [docs/CALIBRATION.md](docs/CALIBRATION.md) — how the two detection thresholds were set

## Install

```bash
pip install -e ".[lanelet2,dev]"
pytest                       # 50 tests, no network required
ruff check src tests         # the lint gate CI runs
```

The `street` subcommand additionally needs `ultralytics` (detector); it is in
the `street` extra. Everything else, including the street stage's geometry,
association, export and evaluation code, runs on the base install.

The `lanelet2` extra installs the official Lanelet2 Python bindings, which the
QA stage uses to *actually load* the exported map and build a routing graph from
it. Without it 48 of the 50 tests still run and the validation section reports
that it was unavailable — but the question this project exists to answer stops
being checked, so CI installs it.

### CI

[`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs on every push to
`main` and every pull request:

| job | what it checks |
|---|---|
| `lint` | `ruff check` — a deliberately narrow rule set (`E,F,W,I,UP,B`) that catches undefined names, unused imports, shadowed builtins and mutable defaults, not house style |
| `test` (3.10 / 3.11 / 3.12) | the full suite **with** the Lanelet2 bindings, so the export really is loaded and routed; then that the CLI entry point and every subcommand parse; then that the street modules import **without** the `street` extra installed |

The suite is offline by design — no S3, no Overpass, no model weights — so
nothing is mocked and nothing is skipped. Anything that needs the network
belongs in a manual run, not in CI.

## Data sources and licensing

- **SpaceNet-3** imagery and road labels — CC BY-SA 4.0 (Maxar / SpaceNet LLC),
  anonymously downloadable from `s3://spacenet-dataset`. Used as the default
  source because it provides high-resolution imagery *and* an OSM-shaped
  centreline graph for the same footprint with no credentials.
- **OpenStreetMap** via Overpass — ODbL. The reference prior for arbitrary AOIs.
- **Aerial imagery** for `--source live` is whatever tile service you pass;
  no endpoint is hard-coded, because using one would imply a licence you may not
  have.
- **Argoverse 2 Sensor Dataset** — CC BY-NC-SA 4.0 (Argo AI), anonymously
  downloadable from `s3://argoverse`. Used for the street-level stage because it
  ships accurate camera poses, calibration and a ground-height surface, plus an
  HD map that can be held back as ground truth. **Non-commercial**: it is the
  evaluation harness, not a production input.
- **YOLOv8** weights from the Ultralytics GitHub release assets — AGPL-3.0.
  Used unmodified for detection; swap `street/detect/` for any other detector.

## TODO

Ordered by *how much of the remaining uncertainty each one removes*, not by how
interesting it is. Every item names the gap it closes from the table above and
where it attaches — the PoC was built so none of them requires restructuring it.

### Tier 1 — closes a gap that currently blocks correctness

**1. Signal aspect classification** (`street/detect/`) — crop each triangulated
head from its highest-resolution view and classify the lens layout: ball, left
arrow, right arrow, pedestrian. This is the highest-value single addition in the
project. It closes three gaps at once — *which movement* a signal controls,
signal **grouping** (heads showing the same aspect on one arm become one
regulatory element), and the pedestrian-head exclusion that is currently a
height heuristic. It attaches to the existing `Landmark` objects and inherits
association, export and review unchanged.

**2. Read every arrow in a lane, not the best one**
(`street/semantics/arrows.py`) — arrow recall is 0.53 purely because
`merge_arrows` keeps one symbol per lane and a combined symbol (`↰↑`) classifies
as one manoeuvre. Detect symbol *repeats* along the lane and decompose combined
heads into their components. Recall should follow precision upward without
touching the 1.00 precision that makes the result usable.

**3. Break the lane-count tie** — the largest unresolved quantity in phase 1.
Three candidate signals, in increasing cost: (a) carriageway **width** against a
class-conditional lane-width prior, already computed in `fusion/corridor.py`;
(b) counting lanes in the 5 cm BEV mosaic where street imagery exists — it
resolves paint that 27 cm satellite cannot; (c) run `--lanes-tag class|true` to
first isolate how much of the residual error is the prior's rather than the
observation's. Do (c) first: it is one command and it says whether (a) and (b)
are worth building.

**4. Coverage-aware confidence** — 33 % of street-stage geometry has no true
lane near it, all of it outside the driven corridor. The mosaic already produces
a coverage raster; feed it into `Lane.confidence` so extrapolated geometry is
*marked* as extrapolated instead of merely being wrong quietly.

### Tier 2 — broadens what the map contains

**5. Sign semantics** (`street/semantics/`) — a classifier over detected sign
crops (speed limit, no-turn, one-way, yield), geolocated by the same
triangulation and attached as `subtype=traffic_sign` regulatory elements. The
`ref_line`/`refers` machinery is already there; only the classifier is missing.

**6. Crosswalks, cycle lanes, bus stops** — the BEV mosaic already resolves
zebra ladders clearly enough that `stopline.py`'s periodicity test *rejects*
them; the same measurement could emit them as Lanelet2 `crosswalk` lanelets
instead of discarding them.

**7. Elevation** — replace the constant `PipelineConfig.elevation` with a
sampled `GroundSurface`. `GeoRaster` and the local metric frame are already
z-agnostic and the street stage already carries real per-object heights, so this
is plumbing rather than design.

**8. OpenDRIVE writer** — a second writer alongside `export/lanelet2_osm.py`.
`LaneGraph` is format-neutral by construction; the work is in the format, not in
the pipeline.

### Tier 3 — scale, validation and workflow

**9. More street-level sources** (`street/sources/`) — `StreetImagerySource`
needs only posed frames, so **Mapillary** or KartaView would replace the
non-commercially-licensed Argoverse 2 with a genuinely open, worldwide input.
Their poses are far noisier, which makes this a real experiment rather than an
adapter: it directly tests how much pose quality the triangulation needs.

**10. Multi-log / multi-city runs** — everything so far is one Detroit log and
ten Las Vegas tiles. `n = 1` for the street stage. Left/right-hand traffic,
European and Japanese junction geometry, and a second signal-mounting convention
(vertical pole vs span wire) are all untested.

**11. Human-in-the-loop correction loop** — `review_items.json` is already the
work queue and `gm2ll:review=yes` already filters in JOSM, but there is no path
for a correction to flow *back* in. A round trip — export, correct in JOSM,
re-import as high-confidence evidence — would turn the PoC into something
operable.

**12. Gaussian splatting / photorealistic reconstruction** — consumes the same
local metric frame and AOI definition; useful both as a denser geometry source
and as a way to render validation views.

### Known issues

- `docs/RESULTS.md` figures come from SpaceNet AOIs and
  `docs/STREET_RESULTS.md` from Argoverse 2; the two have never been run on the
  **same** ground, so aerial-vs-BEV geometry is compared only indirectly.
- Junction clustering in `street/sources/av2_map.py` merges intersections whose
  interior lanes share a road in the same role. One Detroit junction with 17
  entries is almost certainly two junctions.
- The BEV mosaic canvas covers only the camera trajectory ± 34 m, so lanes in
  the AOI but off the driven route are exported with no imagery evidence at all.
