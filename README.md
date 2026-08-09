# gmap2lanelet

**Lane-level vector maps from public map data + public aerial imagery, exported as Lanelet2.**

A proof of concept for one hypothesis:

> Using existing road data (OSM) as a **topology prior** and correcting it with
> **geometry observed in aerial/satellite imagery**, a usable lane-level map can be
> semi-automatically generated from public information alone.

The point of this PoC is **not** accuracy. It is to establish, with numbers,
*what can be automated and what cannot* — and to make everything that cannot be
automated visible, georeferenced and reviewable.

Give it a place, get back a Lanelet2 map candidate plus a list of the places a
human has to look at.

```bash
pip install -e ".[lanelet2]"

# a public SpaceNet tile: 0.27 m/px imagery + OSM-like road vectors, no credentials
gmap2lanelet run --image-id 38 --out outputs/vegas38

# anywhere on Earth: OSM via Overpass + an aerial tile service you are licensed to use
gmap2lanelet run --source live \
    --bbox 139.7601 35.6801 139.7649 35.6841 --drive-on left \
    --tiles "https://<your-imagery-host>/{z}/{y}/{x}" --tiles-attribution "..." \
    --out outputs/tokyo

# compare failure modes across many AOIs
gmap2lanelet batch --image-ids 93,162,48,10,100,151,89,38,124,160 --out outputs/batch

# add the semantics that overhead imagery cannot see: traffic lights, the stop
# lines they govern, the lanes behind them, and painted lane arrows
gmap2lanelet street --log 20dd185d-b4eb-3024-a17a-b4e5d8b15b65 --city DTW \
    --out outputs/street_detroit
```

Each run writes `lanelet2_map.osm`, `viewer.html` (self-contained, offline),
`report.md`, `review_items.json`, `lane_graph.json` and a set of figures.

![imagery + OSM prior + generated lanes](docs/figures/overlay_img38.jpg)

*Cyan dashed: OSM prior. Yellow: painted lines detected in the imagery (solid /
dashed). Cyan solid: observed road edge. Magenta dotted: virtual boundary, i.e.
no paint was visible and the split is inferred. Green/blue/purple: lane
centrelines coloured by which source decided the lane. Red circles: junctions,
whose interior connectivity is inferred rather than observed.*

---

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

**Every element carries `source` and `confidence`.** `Source.OSM` means the map
decided it, `Source.IMAGE` means the imagery measured it, `Source.FUSED` means
they agreed, `Source.INFERRED` means a rule invented it (all intersection
interiors), `Source.DEFAULT` means neither source said anything. These are
written into the exported map as `gm2ll:*` tags, so a reviewer opening it in
JOSM sees where each number came from.

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

**What is automatable:** road-surface geometry, the lateral correction of the
prior, lane-boundary geometry where paint is visible, dual-carriageway
detection, and the entire Lanelet2 conversion.

**What is not:** the number of lanes (image and prior disagree on ~60 % of major
carriageways and nothing in public data breaks the tie), and intersection
connectivity (never observable, never tagged — every turn lane in this PoC is
a rule, not a measurement).

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
```

The `street` subcommand additionally needs `ultralytics` (detector) and
`opencv-python`; both are in the `street` extra.

The `lanelet2` extra installs the official Lanelet2 Python bindings, which the
QA stage uses to *actually load* the exported map and build a routing graph from
it. Without it everything still runs; the validation section just reports that
it was unavailable.

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

## Extending

The PoC was built so the planned next steps do not require restructuring it:

- **more street-level sources** → `StreetImagerySource` needs only posed frames,
  so Mapillary, KartaView or a dashcam with GNSS/INS drop in; `GroundSurface`
  takes any raster DEM and `Detector` any box detector.
- **signal aspect / sign classification** → attaches to the existing `Landmark`
  objects and immediately inherits association, export and review. This is the
  single highest-value addition: it is what separates a left-turn signal from a
  through signal, which is the main thing the current stage cannot decide.
- **3-D / depth** → `GeoRaster` and the local metric frame are already
  z-agnostic; `PipelineConfig.elevation` is the single place elevation is
  assumed flat, and the street stage already carries real per-object heights.
- **Gaussian splatting / photorealistic environments** → consumes the same
  local metric frame and AOI definition.
- **OpenDRIVE overlay** → a second writer next to `export/lanelet2_osm.py`; the
  `LaneGraph` intermediate representation is format-neutral by construction.
- **human-in-the-loop correction** → `review_items.json` is already the work
  queue, and the exported map carries `gm2ll:review=yes` on low-confidence
  lanelets so JOSM can filter for them.
