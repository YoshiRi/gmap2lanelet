"""Pipeline configuration.

Defaults are tuned for ~0.25-0.35 m/px imagery of an urban area.  Everything
that encodes a modelling assumption (rather than a fact from the data) lives
here so it is easy to see what the PoC is assuming.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PipelineConfig:
    # -- prior cleaning -----------------------------------------------------
    node_snap: float = 1.0                  # m, vertices closer than this are one node
    min_stub: float = 6.0                   # m, dangling edges shorter than this go
    junction_cluster_radius: float = 22.0   # m, nodes within this are one intersection
    min_segment_length: float = 8.0         # m, edges shorter than this are skipped
    skip_classes: tuple[str, ...] = ()      # highway values to ignore entirely

    # -- profiling ----------------------------------------------------------
    station_step: float = 1.0               # m along the road
    offset_step: float = 0.25               # m across the road
    profile_half_width: float = 22.0        # m, minimum half-extent of a cross-section
    max_lateral_shift: float = 12.0         # m, how far the prior may be wrong

    # -- corridor -----------------------------------------------------------
    road_threshold: float = 0.45            # road-probability cut
    corridor_bleed_factor: float = 1.7      # cap on width vs the class-expected width
    marking_threshold: float = 0.30

    # -- lanes --------------------------------------------------------------
    drive_on_right: bool = True             # False for Japan / UK / ...
    default_lane_width: float = 3.5

    # -- intersections ------------------------------------------------------
    intersection_margin: float = 2.5        # m added to the junction radius
    intersection_min_radius: float = 8.0
    intersection_max_radius: float = 45.0
    allow_u_turn: bool = False
    turn_through_deg: float = 40.0          # |heading change| below this is "through"
    turn_max_deg: float = 160.0             # above this it is a U-turn

    # -- export -------------------------------------------------------------
    node_merge_tolerance: float = 0.05      # m, points closer than this share an OSM node
    export_simplify_m: float = 0.10         # m, Douglas-Peucker on exported linestrings
    elevation: float = 0.0

    # -- reporting ----------------------------------------------------------
    review_confidence_threshold: float = 0.45

    extra: dict = field(default_factory=dict)

    @staticmethod
    def from_dict(d: dict | None) -> PipelineConfig:
        d = dict(d or {})
        known = set(PipelineConfig.__dataclass_fields__)
        extra = {k: v for k, v in d.items() if k not in known}
        cfg = PipelineConfig(**{k: v for k, v in d.items() if k in known})
        cfg.extra.update(extra)
        return cfg

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__ if k != "extra"}
