"""Self-contained HTML viewer.

One file, no network: the aerial image is embedded as a JPEG data URI and every
vector layer is inline SVG in the local metric frame.  Layers can be toggled and
any element can be clicked to see where it came from and how much it is trusted
-- which is the point of keeping provenance on every element in the first place.
"""

from __future__ import annotations

import base64
import io
import json
import logging
from pathlib import Path

import numpy as np

from ..geo import simplify_polyline
from ..types import LaneGraph, MarkingType, Source

log = logging.getLogger(__name__)

MARKING_COLOR = {
    MarkingType.SOLID: "#ffd400",
    MarkingType.DASHED: "#ffd400",
    MarkingType.DOUBLE_SOLID: "#ff8c00",
    MarkingType.ROAD_EDGE: "#00e5ff",
    MarkingType.VIRTUAL: "#ff4dd2",
    MarkingType.UNKNOWN: "#9ca3af",
}
SOURCE_COLOR = {
    Source.OSM: "#4c9aff", Source.IMAGE: "#36d399", Source.FUSED: "#a78bfa",
    Source.INFERRED: "#f87272", Source.DEFAULT: "#9ca3af",
}
SEVERITY_COLOR = {"high": "#ef4444", "medium": "#f59e0b", "low": "#38bdf8"}


def write_viewer(path: str | Path, imagery, prior, graph: LaneGraph, evidence,
                 failures, simplify: float = 0.2) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raster = imagery.raster
    x0, x1, y0, y1 = raster.extent
    w, h = x1 - x0, y1 - y0

    def poly(pts: np.ndarray) -> str:
        p = simplify_polyline(np.asarray(pts, float), simplify)
        return " ".join(f"{x - x0:.2f},{y1 - y:.2f}" for x, y in p)

    layers: dict[str, list[str]] = {k: [] for k in
                                    ("prior", "boundaries", "centerlines", "intersections",
                                     "review")}
    meta: dict[str, dict] = {}

    for e in prior.edges.values():
        eid = f"prior-{e.id}"
        meta[eid] = {"kind": "OSM prior edge", "id": e.id, "highway": e.highway,
                     "length_m": round(e.length, 1), "tags": e.tags}
        layers["prior"].append(
            f'<polyline id="{eid}" class="hit prior" points="{poly(e.points)}" />')

    for bid, b in graph.boundaries.items():
        col = MARKING_COLOR.get(b.marking, "#9ca3af")
        dash = ' stroke-dasharray="1.6 1.4"' if b.marking in (
            MarkingType.DASHED, MarkingType.VIRTUAL) else ""
        eid = f"bnd-{bid}"
        meta[eid] = {"kind": "lane boundary", "id": bid, "marking": b.marking.value,
                     "source": b.provenance.source.value,
                     "confidence": round(b.confidence, 2), **b.provenance.detail}
        layers["boundaries"].append(
            f'<polyline id="{eid}" class="hit bnd" points="{poly(b.points)}" '
            f'stroke="{col}"{dash} />')

    for lid, ln in graph.lanes.items():
        col = SOURCE_COLOR.get(ln.provenance.source, "#fff")
        eid = f"lane-{lid}"
        meta[eid] = {
            "kind": f"{ln.kind} lane", "id": lid, "segment": ln.segment_id,
            "source": ln.provenance.source.value, "confidence": round(ln.confidence, 2),
            "width_m": round(ln.width, 2), "one_way": ln.one_way,
            "speed_limit_kph": ln.speed_limit_kph, "turn": ln.turn_direction,
            "predecessors": ln.predecessors, "successors": ln.successors,
            "flags": ln.attributes.get("flags", []),
            **{k: v for k, v in ln.provenance.detail.items() if k != "flags"},
        }
        op = 0.35 + 0.6 * float(np.clip(ln.confidence, 0, 1))
        dash = ' stroke-dasharray="3 2"' if ln.kind == "turn" else ""
        layers["centerlines"].append(
            f'<polyline id="{eid}" class="hit lane" points="{poly(ln.centerline)}" '
            f'stroke="{col}" opacity="{op:.2f}"{dash} />')

    for iid, it in graph.intersections.items():
        eid = f"int-{iid}"
        meta[eid] = {"kind": "intersection", "id": iid, "approaches": it.approach_count,
                     "turn_lanes": len(it.turn_lane_ids),
                     "confidence": round(it.confidence, 2), "notes": it.notes,
                     **it.provenance.detail}
        layers["intersections"].append(
            f'<circle id="{eid}" class="hit inter" cx="{it.center[0] - x0:.2f}" '
            f'cy="{y1 - it.center[1]:.2f}" r="{it.radius:.2f}" />')

    for item in failures.items:
        eid = f"rev-{item.id}"
        meta[eid] = {"kind": f"review: {item.kind}", "id": item.id,
                     "severity": item.severity, "message": item.message,
                     "elements": item.element_ids, **item.detail}
        layers["review"].append(
            f'<circle id="{eid}" class="hit rev" cx="{item.position[0] - x0:.2f}" '
            f'cy="{y1 - item.position[1]:.2f}" r="4" '
            f'fill="{SEVERITY_COLOR.get(item.severity, "#38bdf8")}" />')

    html = _TEMPLATE.format(
        title=graph.aoi.name,
        img=_data_uri(raster.data),
        overlay_road=_data_uri(_heat(evidence.road_prob.data, (60, 130, 255))),
        overlay_mark=_data_uri(_heat(evidence.marking.data, (255, 90, 0), thr=0.3)),
        w=f"{w:.2f}", h=f"{h:.2f}",
        prior="\n".join(layers["prior"]),
        boundaries="\n".join(layers["boundaries"]),
        centerlines="\n".join(layers["centerlines"]),
        intersections="\n".join(layers["intersections"]),
        review="\n".join(layers["review"]),
        meta=json.dumps(meta, default=str),
        stats=json.dumps(failures.stats, default=str),
        counts=json.dumps(failures.counts),
        attribution=f"{imagery.attribution} &middot; {prior.attribution}",
    )
    path.write_text(html, encoding="utf-8")
    log.info("wrote viewer %s (%.1f MB)", path, path.stat().st_size / 1e6)
    return path


def _data_uri(arr: np.ndarray) -> str:
    from PIL import Image

    buf = io.BytesIO()
    if arr.ndim == 3 and arr.shape[2] == 4:
        Image.fromarray(arr, "RGBA").save(buf, format="PNG", optimize=True)
        mime = "png"
    else:
        Image.fromarray(arr).convert("RGB").save(buf, format="JPEG", quality=82)
        mime = "jpeg"
    return f"data:image/{mime};base64," + base64.b64encode(buf.getvalue()).decode()


def _heat(data: np.ndarray, rgb: tuple[int, int, int], thr: float = 0.5) -> np.ndarray:
    """Single-colour RGBA overlay from a [0, 1] raster."""
    a = np.clip((data - thr) / max(1e-6, 1 - thr), 0, 1)
    out = np.zeros((*data.shape, 4), dtype=np.uint8)
    for i, c in enumerate(rgb):
        out[:, :, i] = c
    out[:, :, 3] = (a * 190).astype(np.uint8)
    return out


_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>gmap2lanelet - {title}</title>
<style>
 :root {{ color-scheme: dark; }}
 * {{ box-sizing: border-box; }}
 body {{ margin:0; background:#0b0c10; color:#e6e8ee;
        font:13px/1.45 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif; }}
 #wrap {{ display:flex; height:100vh; }}
 #stage {{ flex:1; overflow:hidden; position:relative; cursor:grab; }}
 #stage.drag {{ cursor:grabbing; }}
 #canvas {{ position:absolute; transform-origin:0 0; }}
 #canvas img {{ position:absolute; left:0; top:0; width:100%; height:100%;
                image-rendering:auto; }}
 svg {{ position:absolute; left:0; top:0; width:100%; height:100%; overflow:visible; }}
 polyline {{ fill:none; vector-effect:non-scaling-stroke; }}
 .prior {{ stroke:#00b3ff; stroke-width:2.2; stroke-dasharray:6 4; opacity:.7; }}
 .bnd {{ stroke-width:1.6; }}
 .lane {{ stroke-width:2; }}
 .inter {{ fill:none; stroke:#ff6b6b; stroke-width:1.4; stroke-dasharray:4 4;
           opacity:.75; vector-effect:non-scaling-stroke; }}
 .rev {{ stroke:#0b0c10; stroke-width:1; vector-effect:non-scaling-stroke; }}
 .hit {{ cursor:pointer; }}
 .hit:hover {{ filter:brightness(1.6); }}
 .sel {{ stroke:#fff !important; stroke-width:4 !important; }}
 #side {{ width:340px; flex:none; background:#12141b; border-left:1px solid #232735;
          overflow:auto; padding:14px; }}
 h1 {{ font-size:15px; margin:0 0 4px; }}
 h2 {{ font-size:12px; text-transform:uppercase; letter-spacing:.06em;
       color:#8b93a7; margin:16px 0 6px; }}
 label {{ display:flex; align-items:center; gap:7px; padding:2px 0; }}
 .legend i {{ display:inline-block; width:16px; height:3px; margin-right:7px;
              vertical-align:middle; }}
 .legend div {{ padding:1px 0; color:#c3c9d8; }}
 table {{ width:100%; border-collapse:collapse; font-size:12px; }}
 td {{ padding:2px 4px; border-bottom:1px solid #1c202b; vertical-align:top;
       word-break:break-word; }}
 td:first-child {{ color:#8b93a7; width:44%; }}
 #info {{ background:#0e1017; border:1px solid #232735; border-radius:6px; padding:8px;
          min-height:80px; }}
 .muted {{ color:#8b93a7; }}
 .pill {{ display:inline-block; padding:1px 6px; border-radius:999px; font-size:11px;
          background:#232735; margin:1px 2px 1px 0; }}
 #hint {{ position:absolute; left:10px; bottom:10px; background:#0b0c10cc; padding:6px 9px;
          border-radius:6px; font-size:11px; color:#98a0b3; }}
</style></head><body>
<div id="wrap">
 <div id="stage">
  <div id="canvas">
   <img src="{img}" alt="aerial imagery">
   <img id="ovRoad" src="{overlay_road}" style="display:none" alt="">
   <img id="ovMark" src="{overlay_mark}" style="display:none" alt="">
   <svg viewBox="0 0 {w} {h}" preserveAspectRatio="none">
    <g id="g-prior">{prior}</g>
    <g id="g-boundaries">{boundaries}</g>
    <g id="g-centerlines">{centerlines}</g>
    <g id="g-intersections">{intersections}</g>
    <g id="g-review">{review}</g>
   </svg>
  </div>
  <div id="hint">scroll = zoom &middot; drag = pan &middot; click an element for provenance</div>
 </div>
 <div id="side">
  <h1>gmap2lanelet</h1>
  <div class="muted">{title}</div>
  <h2>layers</h2>
  <label><input type="checkbox" id="l-img" checked> aerial imagery</label>
  <label><input type="checkbox" id="l-ovRoad"> road-surface evidence</label>
  <label><input type="checkbox" id="l-ovMark"> lane-marking evidence</label>
  <label><input type="checkbox" id="l-prior" checked> OSM prior centrelines</label>
  <label><input type="checkbox" id="l-boundaries" checked> lane boundaries</label>
  <label><input type="checkbox" id="l-centerlines" checked> lane centrelines</label>
  <label><input type="checkbox" id="l-intersections" checked> intersections</label>
  <label><input type="checkbox" id="l-review" checked> review items</label>
  <h2>legend</h2>
  <div class="legend">
   <div><i style="background:#00b3ff"></i>OSM prior centreline</div>
   <div><i style="background:#ffd400"></i>painted line (solid / dashed)</div>
   <div><i style="background:#00e5ff"></i>road edge (observed)</div>
   <div><i style="background:#ff4dd2"></i>virtual boundary (inferred)</div>
   <div><i style="background:#36d399"></i>lane: geometry from image</div>
   <div><i style="background:#a78bfa"></i>lane: fused</div>
   <div><i style="background:#4c9aff"></i>lane: count from OSM</div>
   <div><i style="background:#f87272"></i>lane: inferred (intersection)</div>
  </div>
  <h2>selection</h2>
  <div id="info" class="muted">nothing selected</div>
  <h2>review items</h2>
  <div id="counts"></div>
  <h2>run statistics</h2>
  <table id="stats"></table>
  <h2>attribution</h2>
  <div class="muted">{attribution}</div>
 </div>
</div>
<script>
const META = {meta}, STATS = {stats}, COUNTS = {counts};
const stage = document.getElementById('stage'), canvas = document.getElementById('canvas');
const W = {w}, H = {h};
let z = Math.min(stage.clientWidth / W, stage.clientHeight / H), ox = 0, oy = 0;
function apply() {{
  canvas.style.width = W + 'px'; canvas.style.height = H + 'px';
  canvas.style.transform = `translate(${{ox}}px,${{oy}}px) scale(${{z}})`;
}}
apply();
stage.addEventListener('wheel', e => {{
  e.preventDefault();
  const r = stage.getBoundingClientRect(), mx = e.clientX - r.left, my = e.clientY - r.top;
  const k = Math.exp(-e.deltaY * 0.0016), nz = Math.min(60, Math.max(0.05, z * k));
  ox = mx - (mx - ox) * (nz / z); oy = my - (my - oy) * (nz / z); z = nz; apply();
}}, {{passive: false}});
let drag = null;
stage.addEventListener('pointerdown', e => {{
  drag = {{x: e.clientX - ox, y: e.clientY - oy}}; stage.classList.add('drag');
  stage.setPointerCapture(e.pointerId);
}});
stage.addEventListener('pointermove', e => {{
  if (!drag) return; ox = e.clientX - drag.x; oy = e.clientY - drag.y; apply();
}});
stage.addEventListener('pointerup', () => {{ drag = null; stage.classList.remove('drag'); }});

for (const id of ['prior','boundaries','centerlines','intersections','review']) {{
  document.getElementById('l-' + id).onchange = e =>
    document.getElementById('g-' + id).style.display = e.target.checked ? '' : 'none';
}}
document.getElementById('l-img').onchange = e =>
  canvas.querySelector('img').style.display = e.target.checked ? '' : 'none';
for (const id of ['ovRoad','ovMark']) {{
  document.getElementById('l-' + id).onchange = e =>
    document.getElementById(id).style.display = e.target.checked ? '' : 'none';
}}

let sel = null;
function esc(s) {{ return String(s).replace(/[<>&]/g, c => ({{'<':'&lt;','>':'&gt;','&':'&amp;'}})[c]); }}
function show(id) {{
  const m = META[id]; if (!m) return;
  if (sel) sel.classList.remove('sel');
  sel = document.getElementById(id); sel.classList.add('sel');
  let rows = '';
  for (const [k, v] of Object.entries(m)) {{
    let val = Array.isArray(v)
      ? (v.length ? v.map(x => `<span class="pill">${{esc(x)}}</span>`).join(' ') : '<span class="muted">none</span>')
      : (v && typeof v === 'object' ? esc(JSON.stringify(v)) : esc(v));
    rows += `<tr><td>${{esc(k)}}</td><td>${{val}}</td></tr>`;
  }}
  document.getElementById('info').className = '';
  document.getElementById('info').innerHTML = `<table>${{rows}}</table>`;
}}
document.querySelectorAll('.hit').forEach(el => el.onclick = ev => {{
  ev.stopPropagation(); show(el.id);
}});

document.getElementById('counts').innerHTML = Object.keys(COUNTS).length
  ? Object.entries(COUNTS).sort((a,b) => b[1]-a[1])
      .map(([k,v]) => `<span class="pill">${{esc(k)}} &middot; ${{v}}</span>`).join(' ')
  : '<span class="muted">none</span>';
document.getElementById('stats').innerHTML = Object.entries(STATS)
  .filter(([k]) => k !== 'review_items_by_kind')
  .map(([k,v]) => `<tr><td>${{esc(k)}}</td><td>${{typeof v === 'object'
      ? esc(JSON.stringify(v)) : esc(v)}}</td></tr>`).join('');
</script></body></html>
"""
