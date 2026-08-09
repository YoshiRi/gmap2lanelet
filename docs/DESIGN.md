# Design

## The central idea: a road-aligned frame

Everything in this PoC follows from one decision. Each prior edge defines a
curvilinear coordinate system — arc length `s` along its centreline, lateral
offset `u` across it — and all image evidence is resampled into that frame
before any decision is made (`fusion/profile.py`).

Two consequences:

1. **Finding lane boundaries becomes 1-D.** In the rectified `(s, u)` image, a
   lane boundary is a vertical stripe. Instead of tracing curves in 2-D, we
   aggregate along `s` and find peaks in `u`.
2. **The prior only has to be approximately right.** Its job is to point the
   cross-section roughly across the road. Its *position* is then measured and
   corrected — an OSM way 4 m off produces exactly the same cross-sections as
   one that is 0 m off, just shifted, and the shift is the output.

This is why the topology prior earns its place even though its geometry is poor,
and it is the difference between this and running a segmentation model over a
tile and vectorising whatever comes out.

## Stages

### 1. Sources (`sources/`)

`PriorSource` yields road centrelines with **OSM tag keys**; `ImagerySource`
yields a georeferenced RGB raster. Everything downstream is written against OSM
semantics only, so swapping the prior source changes nothing.

- `osm_overpass.OverpassSource` — live Overpass query for drivable highways.
- `osm_overpass.OsmFileSource` — the same from a downloaded `.osm` extract.
- `xyz_tiles.XYZTileSource` — mosaics slippy-map tiles over the AOI. No default
  endpoint is hard-coded: using one would imply a licence the user may not hold.
- `spacenet.SpaceNetSource` — public imagery **and** an OSM-shaped centreline
  graph for the same footprint, anonymously downloadable. This is the default
  because it makes the PoC reproducible by anyone with no credentials.

### 2. Coordinate frames (`geo.py`, `raster.py`)

A local equirectangular tangent plane at the AOI centre. Over a few hundred
metres the error is sub-centimetre, and — more usefully — it keeps the mapping
from a north-up lat/lon raster to metres *exactly affine*, so no imagery is ever
resampled. `GeoRaster` carries that affine and does bilinear sampling at metric
coordinates.

### 3. Topology prior (`prior/`)

Project → split each way at vertices it shares with another way → drop dangling
stubs → cluster junction nodes within 22 m.

The ordering matters and was a real bug: **resampling before the split destroys
topology**, because in OSM a junction *is* a vertex shared by two ways, and
moving vertices moves them off it. Ways are split on their original vertices;
resampling happens afterwards, per edge, where `resample_polyline` pins the
endpoints.

Junction clustering is what turns the four OSM nodes of a divided-road crossing
into one physical intersection.

`prior/osm_tags.py` interprets tags and — importantly — reports *where each
value came from*: `lanes=4` from the map is not the same fact as "4 because it
is a primary road", and the fusion stage arbitrates differently for each.

### 4. Observation (`observation/`)

Produces per-pixel evidence: road-surface probability, lane-marking response,
vegetation and shadow. It knows nothing about lanes or connectivity.

**Road surface** is weakly supervised *by the prior*: colour statistics are
sampled in a thin buffer along the prior centrelines, a small mixture is fitted
in illumination-normalised chromaticity space, and the model is then evaluated
everywhere. This is the map teaching the classifier, not the map deciding where
roads are — the model happily finds pavement OSM never mentions, which is
exactly the conflict we want to detect.

Pavement is not unimodal (fresh asphalt, weathered asphalt, concrete differ 2× in
brightness while sharing a near-neutral chromaticity), so a single Gaussian
loses whichever mode is in the minority. A 3-component mixture over luminance
with chromaticity-dominant distance fixes it.

**Markings** are white top-hat at the paint scale × a Sato ridge filter (so
*linear* bright structures beat compact ones — vehicles, roof furniture),
normalised **locally** and gated by an absolute contrast floor. See
[CALIBRATION.md](CALIBRATION.md) for why both halves of that are necessary.

#### Why not a pretrained segmentation model?

The brief allows one, and the interface (`ObservationBackend`) is designed for
it — `observation/pretrained.py` implements the same contract. But the published
aerial lane-marking segmenters (SkyScapes, the one behind DeepAerialMapper) are
trained on 5–13 cm orthophotos, where a lane marking is 2–3 px wide. At 27 cm it
is *sub-pixel*, visible only through partial pixel coverage. Running such a model
out of domain would produce confident, unverifiable output — the opposite of
what a failure-focused PoC needs. The classical backend is self-calibrating,
has two interpretable thresholds, and its failures are legible.

### 5. Corridor (`fusion/corridor.py`)

Per station, find runs of road-probability above threshold within a search
window, pick the one nearest the prior line, and track its edges along `s` with
median + moving-average smoothing.

Three things it deliberately does:

- **returns the lateral correction** — `center_offset(s)` is the geometry fix;
- **caps width by road class** — pavement does not stop at the road (parking
  aisles, forecourts, driveways are the same asphalt), so the prior's class
  bounds how wide a corridor may get, and hitting that bound is *reported*
  rather than silently accepted;
- **detects dual carriageways** — a non-oneway way whose pavement comes in two
  runs separated by a median-sized gap becomes two corridors with opposite
  directions.

It also caps the total lateral offset by the reference line's radius of
curvature, because offsetting a polyline past its radius folds it into a loop.

### 6. Lane structure (`fusion/lanes.py`)

In corridor-relative coordinates, score each lateral offset by **how often** a
marking is present there, not by its mean response — a dashed line is painted
about a quarter of the time, so averaging amplitude buries it under the solid
centre line, while a hit rate separates "dashed" (~0.25) from "noise" (~0.02)
and is invariant to how bright the paint is. Peaks are then tracked along `s`
into boundary polylines, and their hit rate classifies them solid / dashed.

The boundary sequence is regularised with two edits that encode road knowledge
rather than map data: a gap wide enough for *k* lanes with no marking in it gets
*k−1* **virtual** boundaries (paint occluded or worn), and two boundaries closer
together than a lane are one boundary (a double line).

Then the arbitration, which is deliberately explicit and logged because "which
source won, and why" is an output of this PoC:

| condition | lane count from | confidence |
|---|---|---|
| markings self-consistent and count matches OSM | fused | 0.85 |
| markings disagree by 1 and corridor width agrees with markings | image | 0.60 |
| markings disagree by > 1, OSM explicitly tagged, width supports OSM | OSM | 0.40 |
| no markings observed, width supports OSM | OSM | 0.45 |
| no markings observed, width does not | OSM | 0.30 |

### 7. Intersections (`fusion/intersection.py`)

Trim every approach back to the junction boundary, then connect. Trimming is
planned for all junctions first and applied once per lane group, because a
segment can be trimmed at both ends by two different junctions, and because
lanes of one carriageway *share* their boundary objects — a shared boundary must
be sliced exactly once, at one station, or it stops corresponding to the
centrelines.

Connectivity is a rule (see [RESULTS](RESULTS.md#q2-how-far-can-intersection-topology-be-recovered)).
Turn paths are cubic Hermite curves between the measured entry and exit poses,
with their boundary end points snapped to the neighbours' — which is what makes
them routable in Lanelet2.

### 8. Assembly and export

`fusion/builder.py` connects lanes across non-junction nodes and stitches
matched ends to identical coordinates. `export/lanelet2_osm.py` writes Lanelet2
OSM-XML: points → nodes with `lat`/`lon`/`ele`/`local_x`/`local_y`, boundaries →
ways tagged `line_thin`/`road_border`/`virtual`, lanes → relations with exactly
one `left` and one `right` member (extra members are a Lanelet2 parse error).

Provenance rides along as `gm2ll:source`, `gm2ll:confidence`, `gm2ll:flags`,
`gm2ll:lanes_osm`, `gm2ll:lanes_image`, `gm2ll:review`. Lanelet2 preserves
unknown attributes, so this survives a JOSM round trip.

## Data model

`LaneGraph` is the format-neutral intermediate representation: `Lane`,
`Boundary`, `Intersection`, each with `Provenance` and `confidence`. Adding
OpenDRIVE export means adding a writer, not touching the pipeline.

## Testing

26 tests, **no network access required**. The interesting ones run on a
synthetic scene (`tests/conftest.py`) that renders a road with known lane
geometry, sub-pixel paint rasterised by area coverage, tree occlusion, and a
prior deliberately displaced by 3 m — so the tests can assert that the pipeline
recovers geometry the prior did not have, to 0.25 m. A companion scene with the
paint deleted asserts the opposite property: that nothing is invented.
