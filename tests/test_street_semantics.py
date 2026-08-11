"""Association, stop lines, arrows and the regulatory-element export.

The scene is a synthetic four-way junction built directly as a lane graph, so
the tests can state the right answer: this signal head is on this arm, facing
this way, and no other arm may claim it.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from gmap2lanelet.config import PipelineConfig
from gmap2lanelet.export.lanelet2_osm import Lanelet2Writer
from gmap2lanelet.geo import AOI, LocalFrame
from gmap2lanelet.raster import GeoRaster
from gmap2lanelet.street.semantics.arrows import find_arrows, merge_arrows
from gmap2lanelet.street.semantics.associate import (
    TrafficLightAssociator,
    associate,
    build_approaches,
    build_stop_lines,
)
from gmap2lanelet.street.semantics.stopline import detect_stop_line
from gmap2lanelet.street.types import Landmark, LandmarkKind, SignalAspect
from gmap2lanelet.types import (
    Boundary,
    Intersection,
    Lane,
    LaneGraph,
    MarkingType,
    Provenance,
    Source,
)

LANE_W = 3.5
ARMS = {"E": 0.0, "N": math.pi / 2, "W": math.pi, "S": -math.pi / 2}
STOP_R = 14.0                     # arms are trimmed back to this radius


@pytest.fixture(scope="module")
def junction() -> LaneGraph:
    """Four arms, two incoming lanes each, all ending 14 m from the centre."""
    aoi = AOI("t", -0.001, -0.001, 0.001, 0.001)
    frame = LocalFrame.for_aoi(aoi)
    g = LaneGraph(aoi=aoi, frame=frame)
    g.intersections["j0"] = Intersection(id="j0", center=np.zeros(2), radius=STOP_R,
                                         prior_node_ids=["n0"], approach_count=4)

    for arm, bearing in ARMS.items():
        # traffic on this arm travels *towards* the centre, i.e. heading = bearing + pi
        h = bearing + math.pi
        d = np.array([math.cos(h), math.sin(h)])
        n = np.array([-d[1], d[0]])
        start = -d * (STOP_R + 60.0)
        for k in range(2):                     # k=0 is the left lane
            offset = (0.5 - k) * LANE_W
            s = np.linspace(0, 60.0, 31)
            c = start[None, :] + s[:, None] * d[None, :] + offset * n[None, :]
            lid = f"{arm}{k}"
            for side, sgn in (("L", +1), ("R", -1)):
                g.add_boundary(Boundary(id=f"b{lid}{side}",
                                        points=c + sgn * 0.5 * LANE_W * n[None, :],
                                        marking=MarkingType.SOLID, confidence=0.8,
                                        provenance=Provenance(Source.IMAGE)))
            g.add_lane(Lane(id=lid, left_id=f"b{lid}L", right_id=f"b{lid}R",
                            centerline=c, segment_id=f"e{arm}", kind="road",
                            index_from_left=k, width=LANE_W, confidence=0.7,
                            provenance=Provenance(Source.FUSED)))
    return g


def _light(lid: str, arm: str, *, ahead: float, lateral: float = 0.0,
           height: float = 5.5, facing_arm: str | None = None) -> Landmark:
    """A signal for ``arm``, ``ahead`` metres past that arm's stop line."""
    bearing = ARMS[arm]
    h = bearing + math.pi                       # travel direction on the arm
    d = np.array([math.cos(h), math.sin(h)])
    n = np.array([-d[1], d[0]])
    stop = -d * STOP_R
    p = stop + ahead * d + lateral * n
    face = ARMS[facing_arm or arm]              # points back down its own arm
    return Landmark(id=lid, kind=LandmarkKind.TRAFFIC_LIGHT,
                    position=np.array([p[0], p[1], height]), n_views=12,
                    baseline_m=30.0, residual_px=4.0, position_sigma_m=0.4,
                    height_above_ground=height, facing=face, confidence=0.8)


# --------------------------------------------------------------------------- #


def test_approaches_are_one_per_arm(junction):
    aps = build_approaches(junction)
    assert len(aps) == 4
    assert {a.intersection_id for a in aps} == {"j0"}
    assert all(len(a.lane_ids) == 2 for a in aps)
    # the axis is run forward past the trimmed lane end so a stop bar at the
    # junction edge is inside the search window
    for a in aps:
        assert float(np.hypot(*a.stop_point)) < STOP_R


def test_far_side_signal_is_assigned_to_its_own_arm(junction):
    aps = build_approaches(junction)
    sls = build_stop_lines(aps, marking=None)
    lm = _light("t0", "E", ahead=22.0)

    out = TrafficLightAssociator().run([lm], aps, sls)
    assert len(out) == 1
    a = out[0]
    assert a.lane_ids
    assigned = next(x for x in aps if x.stop_line_id == a.stop_line_id)
    assert set(a.lane_ids) == {"E0", "E1"}
    assert assigned.segment_id == "eE"


def test_facing_decides_between_two_co_located_approaches(junction):
    """The far-side head of one arm sits over the stop line of the opposite arm.

    Position alone cannot tell them apart -- it is the same point in space.  The
    only thing that separates "the signal E traffic obeys" from "the signal W
    traffic obeys" is which way the head looks, so flipping the facing must flip
    the assignment and nothing else.
    """
    aps = build_approaches(junction)
    sls = build_stop_lines(aps, marking=None)

    east = TrafficLightAssociator().run([_light("t1", "E", ahead=22.0)], aps, sls)[0]
    west = TrafficLightAssociator().run(
        [_light("t1", "E", ahead=22.0, facing_arm="W")], aps, sls)[0]

    assert set(east.lane_ids) == {"E0", "E1"}
    assert set(west.lane_ids) == {"W0", "W1"}


def test_a_signal_no_approach_faces_is_left_unassigned(junction):
    """A head aimed across the junction belongs to no arm modelled here."""
    aps = build_approaches(junction)
    sls = build_stop_lines(aps, marking=None)
    lm = _light("t1", "E", ahead=22.0, lateral=18.0, facing_arm="N")
    lm.position[:2] += np.array([0.0, 40.0])          # well off every arm's axis

    out = TrafficLightAssociator().run([lm], aps, sls)
    assert not out[0].lane_ids
    assert "traffic_light_unassigned" in out[0].flags


def test_implausible_height_is_rejected(junction):
    aps = build_approaches(junction)
    sls = build_stop_lines(aps, marking=None)
    out = TrafficLightAssociator().run([_light("t2", "N", ahead=15.0, height=17.0)],
                                       aps, sls)
    assert not out[0].lane_ids


def test_each_arm_keeps_its_own_signals(junction):
    aps = build_approaches(junction)
    sls = build_stop_lines(aps, marking=None)
    lights = [_light(f"t{i}", arm, ahead=18.0, lateral=lat)
              for i, (arm, lat) in enumerate(
                  [(a, l) for a in ARMS for l in (-3.0, 3.0)])]
    out = TrafficLightAssociator().run(lights, aps, sls)

    assert all(a.lane_ids for a in out), [a.reason for a in out if not a.lane_ids]
    for lm, a in zip(lights, out):
        arm = next(k for k in ARMS if lm.id in
                   {f"t{i}" for i, (k2, _) in enumerate(
                       [(x, l) for x in ARMS for l in (-3.0, 3.0)]) if k2 == k})
        assert {x[0] for x in a.lane_ids} == {arm}


def test_unobserved_stop_lines_are_flagged_not_invented(junction):
    aps = build_approaches(junction)
    sls = build_stop_lines(aps, marking=None)
    assert len(sls) == 4
    assert all(not s.observed for s in sls.values())
    assert all("stop_line_inferred_from_junction_edge" in s.flags for s in sls.values())
    assert all(s.provenance.source is Source.INFERRED for s in sls.values())


# --------------------------------------------------------------------------- #
# marking-raster readers
# --------------------------------------------------------------------------- #


def _painted(bar_at: float | None = None, zebra: bool = False,
             arrow: str | None = None, lane: str = "E0", res: float = 0.05) -> GeoRaster:
    """A marking-response raster for the eastern arm.

    The eastern arm carries west-bound traffic, so travel is towards -x and its
    lanes are trimmed at ``x = +STOP_R``.  Everything below is placed in that
    arm's own frame: ``s`` is metres *upstream* of the trimmed end (positive =
    away from the junction) and ``u`` is metres to the driver's left, which for
    this arm is -y.
    """
    x0, y0 = -40.0, 20.0
    w, h = int(100 / res), int(40 / res)
    data = np.zeros((h, w), dtype=np.float32)
    lane_y = -1.75 if lane == "E0" else 1.75

    def box(s0, s1, u0, u1, y_center=0.0):
        c0, c1 = int((STOP_R + min(s0, s1) - x0) / res), int((STOP_R + max(s0, s1) - x0) / res)
        r0 = int((y0 - (y_center - min(u0, u1))) / res)
        r1 = int((y0 - (y_center - max(u0, u1))) / res)
        data[min(r0, r1):max(r0, r1), c0:c1] = 1.0

    if bar_at is not None:                      # a stop bar past the trimmed end
        box(-bar_at - 0.2, -bar_at + 0.2, -4.0, 4.0)
    if zebra:
        for k in range(4):
            s = -3.0 - 1.2 * k
            box(s - 0.25, s + 0.25, -4.0, 4.0)
    if arrow is not None:
        # rows increase along travel, so the head is *downstream*: smaller s
        box(9.6, 14.0, -0.15, 0.15, lane_y)                        # shaft
        if arrow == "left":
            box(9.6, 10.0, 0.0, 1.0, lane_y)                       # bend, to the left
            box(8.8, 9.8, 0.55, 1.45, lane_y)                      # head
        else:
            box(8.6, 9.8, -0.5, 0.5, lane_y)                       # symmetric head
    return GeoRaster(data, x0, y0, res, res, "marking")


def test_stop_bar_is_found_at_the_right_station(junction):
    ap = next(a for a in build_approaches(junction) if a.segment_id == "eE")
    found = detect_stop_line(ap.axis, ap.half_width, _painted(bar_at=2.0),
                             search_from=0.0, search_to=32.0)
    assert found is not None
    station, score, detail = found
    # the axis ends 18 m past the trimmed lane end, i.e. 16 m past the bar
    assert station == pytest.approx(16.0, abs=1.0)
    assert score > 0.3
    assert detail["thickness_m"] == pytest.approx(0.4, abs=0.25)


def test_a_crosswalk_alone_does_not_become_a_stop_line(junction):
    ap = next(a for a in build_approaches(junction) if a.segment_id == "eE")
    found = detect_stop_line(ap.axis, ap.half_width, _painted(zebra=True),
                             search_from=0.0, search_to=32.0)
    assert found is None or found[2]["zebra_rejected"]


def test_stop_lines_are_marked_observed_when_seen(junction):
    aps = build_approaches(junction)
    sls = build_stop_lines(aps, _painted(bar_at=2.0))
    observed = [s for s in sls.values() if s.observed]
    assert len(observed) == 1
    assert observed[0].provenance.source is Source.IMAGE
    assert observed[0].confidence > 0.4


def test_arrow_classification_separates_left_from_through(junction):
    lane = junction.lanes["E0"]
    left = find_arrows(lane.id, lane.centerline, 0.5 * LANE_W, _painted(arrow="left"))
    through = find_arrows(lane.id, lane.centerline, 0.5 * LANE_W, _painted(arrow="through"))
    assert left and "left" in merge_arrows(left)[0].manoeuvres
    assert through and merge_arrows(through)[0].manoeuvres == {"through"}


def test_no_paint_yields_no_arrows(junction):
    lane = junction.lanes["E0"]
    assert find_arrows(lane.id, lane.centerline, 0.5 * LANE_W, _painted()) == []


# --------------------------------------------------------------------------- #
# export
# --------------------------------------------------------------------------- #


def test_regulatory_elements_reach_the_exported_map(junction, tmp_path):
    lights = [_light("t0", "E", ahead=20.0), _light("t1", "N", ahead=20.0)]
    layer = associate(junction, lights, marking=_painted(bar_at=2.0))
    assert sum(1 for a in layer.assignments.values() if a.lane_ids) == 2

    w = Lanelet2Writer(junction, PipelineConfig(), semantics=layer)
    path = w.write(tmp_path / "map.osm")
    root = ET.parse(path).getroot()

    regs = [r for r in root.findall("relation")
            if any(t.get("k") == "type" and t.get("v") == "regulatory_element"
                   for t in r.findall("tag"))]
    assert len(regs) == 2
    for r in regs:
        roles = {m.get("role") for m in r.findall("member")}
        assert roles == {"refers", "ref_line"}
        subtype = next(t.get("v") for t in r.findall("tag") if t.get("k") == "subtype")
        assert subtype == "traffic_light"

    ways = {wy.get("id"): {t.get("k"): t.get("v") for t in wy.findall("tag")}
            for wy in root.findall("way")}
    types = [ways[m.get("ref")]["type"] for r in regs for m in r.findall("member")]
    assert sorted(set(types)) == ["stop_line", "traffic_light"]

    reg_ids = {r.get("id") for r in regs}
    governed = [r for r in root.findall("relation")
                if any(m.get("role") == "regulatory_element" and m.get("ref") in reg_ids
                       for m in r.findall("member"))]
    assert len(governed) == 4                     # two lanes per assigned arm
    for ll in governed:
        tags = {t.get("k") for t in ll.findall("tag")}
        assert "gm2ll:confidence" in tags


def test_traffic_light_aspect_is_exported_additively(junction, tmp_path):
    light = _light("t0", "E", ahead=20.0)
    light.aspect = SignalAspect.ARROW_LEFT
    light.aspect_confidence = 0.8
    layer = associate(junction, [light], marking=_painted(bar_at=2.0))

    w = Lanelet2Writer(junction, PipelineConfig(), semantics=layer)
    root = ET.parse(w.write(tmp_path / "map.osm")).getroot()

    tl_way = next(wy for wy in root.findall("way")
                  if any(t.get("k") == "type" and t.get("v") == "traffic_light"
                         for t in wy.findall("tag")))
    tags = {t.get("k"): t.get("v") for t in tl_way.findall("tag")}
    assert tags["subtype"] == "red_yellow_green"
    assert tags["gm2ll:aspect"] == "arrow_left"
    assert tags["gm2ll:aspect_confidence"] == "0.80"


def test_a_map_without_semantics_is_unchanged(junction, tmp_path):
    plain = Lanelet2Writer(junction, PipelineConfig()).write(tmp_path / "plain.osm")
    root = ET.parse(plain).getroot()
    assert not [r for r in root.findall("relation")
                if any(t.get("k") == "type" and t.get("v") == "regulatory_element"
                       for t in r.findall("tag"))]
    for r in root.findall("relation"):
        assert len(r.findall("member")) == 2
