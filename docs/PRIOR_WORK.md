# Prior work

What was reviewed before building, what was reused, and where this PoC
deliberately diverges.

## DeepAerialMapper

*Krajewski & Kim, "DeepAerialMapper: Deep Learning-based Semi-automatic HD Map
Creation for Highly Automated Vehicles", arXiv:2410.00769 —
<https://github.com/RobertKrajewski/DeepAerialMapper>*

Two-stage: a segmentation network labels an aerial image into `ROAD`,
`LANEMARKING`, `SYMBOL`, `SIDEWALK`, `PARKING`, `TRAFFICISLAND`, …; then a
classical stage locates, classifies and groups lane boundaries and symbols into
lanes and exports Lanelet2 for editing in JOSM. Reported > 96 % recall and
precision on lane markings and road borders. The repository contains the second
stage; the segmentation model and its weights are not part of it.

**Reused conceptually:**

- the overall shape — *segment, then vectorise, then export Lanelet2 so a human
  finishes the job in JOSM*;
- Lanelet2 as the output format for exactly that reason;
- the semantic vocabulary (road / marking / parking / sidewalk) as the set of
  things worth distinguishing.

**Where this diverges:** DeepAerialMapper derives the map *from the mask alone*.
Its inputs are 5–13 cm orthophotos, where that is reasonable. This PoC works at
27 cm, where lane markings are sub-pixel and a mask-only approach cannot recover
connectivity or road semantics — so a topology prior carries the connectivity
and the imagery is used only for geometry. That inversion is the point of the
brief, and it is what makes the two approaches complementary rather than
competing: with 10 cm imagery, DeepAerialMapper's observation stage would drop
straight into this pipeline's `ObservationBackend` interface.

## SIO-Mapper

*"SIO-Mapper: A Framework for Lane-Level HD Map Construction Using Satellite
Images and OpenStreetMap with No On-Site Visits", arXiv:2504.09882*

The closest published work to this brief: city-scale lane-level HD maps from
satellite imagery **plus OSM**, with no site visits. SIO-Net fuses satellite and
OSM features with a transformer encoder that approximates road shape with a
quadratic function to guide lane extraction, plus a cluster- and graph-based
method for merging lane segments over large areas. Validated on NAVER Labs Open
Dataset and nuScenes across Korea, the US and Singapore.

**Reused conceptually:**

- the central premise — satellite + OSM is sufficient to attempt lane-level
  mapping without site visits;
- *road-shape-guided lane extraction*: SIO-Net encodes road shape to steer the
  network; this PoC does the geometric equivalent by rectifying evidence into
  the road-aligned frame of the prior, which achieves the same "look for lanes
  along the road, not anywhere" bias without training anything;
- the need for an explicit lane-integration step, here the corridor/junction
  stitching in `fusion/builder.py`.

**Where this diverges:** SIO-Mapper learns the fusion. This PoC keeps it
explicit and rule-based, because the deliverable is a failure analysis — a
learned fusion gives one number per lane and no account of *why*, whereas an
explicit arbitration can report "markings said 5, OSM said 4, I kept 5 because
the corridor is 18 m wide, flag it". For a PoC whose stated success criterion is
knowing what cannot be automated, legibility beats accuracy.

## Lanelet2

*Poggenhans et al., "Lanelet2: A High-Definition Map Framework for the Future of
Automated Driving", ITSC 2018 —
<https://github.com/fzi-forschungszentrum-informatik/Lanelet2>*

**Reused directly**, in two ways:

1. as the output format — OSM-XML with `type=lanelet` relations carrying `left`
   and `right` way members, linestrings typed `line_thin`/`road_border`/
   `virtual` with `subtype=solid|dashed|solid_solid`;
2. as the **validator** — the official Python bindings (`pip install lanelet2`)
   load every exported map and build a `RoutingGraph` from it under vehicle
   traffic rules. "Can it be converted to Lanelet2?" is not answered by writing
   syntactically valid XML; it is answered by the library accepting the file and
   the routing graph being connected.

The critical property learned from the format: **successor relations are
implicit in shared boundary end points**. A map whose lanelets do not share
points parses cleanly and routes nowhere. That single fact drives the node
registry in the exporter, the endpoint stitching in the builder, and the
end-point-preserving simplification.

## Aerial road / lane segmentation

Surveyed: SpaceNet-3/5 road extraction (centreline graphs from 30 cm satellite
imagery), SkyScapes (dense aerial semantic segmentation including lane markings,
13 cm), and the "Lane Boundary Geometry Extraction from Satellite Imagery"
line of work (arXiv:2002.02362).

**Conclusion that shaped the build:** every published lane-*marking* segmenter
operates at 5–13 cm ground sampling. At the 25–30 cm available from public
satellite sources, markings are sub-pixel and out of domain for those models.
Road-*extent* models (SpaceNet) do transfer, but road extent is the part this
PoC can already obtain classically with prior-supervised colour statistics.
Hence: a weight-free default backend with an interface ready for a learned one,
rather than an out-of-domain model producing unverifiable confidence. See
[DESIGN](DESIGN.md#why-not-a-pretrained-segmentation-model).

**Reused directly:** the SpaceNet-3 dataset itself — imagery *and* road-centreline
labels for the same footprint, anonymously downloadable, which is what makes
this PoC reproducible without credentials or a mapping-service licence.

## OSM lane information

Surveyed the tagging schema: `lanes`, `lanes:forward`, `lanes:backward`,
`oneway`, `turn:lanes`, `turn:lanes:forward`, `width`, `maxspeed`, `highway`
classes, `junction=roundabout`, `dual_carriageway` conventions.

**Reused directly** in `prior/osm_tags.py`, including the class-based defaults
used when tags are absent — with the source of every value tracked, so
"the map says 4 lanes" is never confused with "primary roads usually have 4".

**Finding:** the tags that matter most for lane-level mapping are the ones that
are not there. Across 715 prior edges, `turn:lanes` appeared on 0 and `maxspeed`
on 0. `lanes` was present on all of them but disagreed with the imagery on about
60 % of major carriageways. This is the concrete answer to "where is public
information insufficient" and is quantified in [RESULTS](RESULTS.md#q4-where-is-public-information-insufficient).
