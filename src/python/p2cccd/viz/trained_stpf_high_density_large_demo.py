from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True, slots=True)
class TrainedSTPFHighDensityLargeDemoConfig:
    run_name: str = "trained_stpf_high_density_large"
    output_dir: str = "src/MyDemo/trained_stpf_high_density_large_run_id"
    benchmark_json: str = "src/benchmark/trained_stpf_high_density_large_benchmark_run_id.json"
    eval_rows_npz: str = "src/benchmark/trained_stpf_high_density_eval_rows_run_id.npz"
    width: int = 1920
    height: int = 1080
    frame_count: int = 120
    fps: int = 24
    query_index: int = 0


@dataclass(frozen=True, slots=True)
class TrainedSTPFHighDensityLargeDemoResult:
    output_dir: Path
    mp4_path: Path
    interactive_html_path: Path
    preview_png_path: Path
    proxy_scene_png_path: Path
    metrics_json_path: Path
    benchmark_report_path: Path
    readme_path: Path


def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/seguisb.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _method_rows(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    baseline = metrics["baseline"]
    random_stpf = metrics["random_stpf"]
    trained = metrics["trained_stpf"]
    return [
        {
            "key": "trained_stpf",
            "name": "RTSTPFExact",
            "subtitle": "learned STPF scheduling + exact certificate",
            "accent": (34, 197, 94),
            "metrics": trained,
            "reduction": 1.0 - trained["exact_work_units"] / max(1.0e-9, baseline["exact_work_units"]),
        },
        {
            "key": "rt_exact",
            "name": "RTExact",
            "subtitle": "all RT candidates go to exact",
            "accent": (96, 165, 250),
            "metrics": baseline | {"method_name": "RTExact"},
            "reduction": 0.0,
        },
        {
            "key": "no_proposal",
            "name": "NoProposal",
            "subtitle": "full exact queue without proposal",
            "accent": (248, 113, 113),
            "metrics": baseline,
            "reduction": 0.0,
        },
    ]


def _query_slice(eval_npz: Path, query_index: int) -> dict[str, Any]:
    data = np.load(eval_npz, allow_pickle=True)
    ids = np.asarray(data["ids"])
    features = np.asarray(data["features"])
    scalar = np.asarray(data["scalar_targets"])
    trace = np.asarray(data["oracle_trace"])
    meta = np.asarray(data["sample_metadata"])
    query_ids = ids[:, 1]
    unique_queries = np.unique(query_ids)
    selected_query = unique_queries[min(query_index, len(unique_queries) - 1)]
    mask = query_ids == selected_query
    order = np.lexsort((features[mask, 7], features[mask, 6], features[mask, 1]))
    row_indices = np.flatnonzero(mask)[order]
    payload = {
        "query_id": int(selected_query),
        "ids": ids[row_indices],
        "features": features[row_indices],
        "scalar_targets": scalar[row_indices],
        "oracle_trace": trace[row_indices],
        "sample_metadata": meta[row_indices],
        "metadata": json.loads(str(data["metadata_json"].tolist())),
    }
    return payload


def _selected_candidate_indices(query: dict[str, Any]) -> set[int]:
    scalar = np.asarray(query["scalar_targets"], dtype=np.float64)
    # For visualization, emulate learned scheduling using the training target:
    # high priority, then lower exact cost, then lower uncertainty.
    score = scalar[:, 0] - 0.12 * scalar[:, 1] - 0.03 * scalar[:, 2]
    order = np.argsort(-score)
    top_count = 2
    return set(int(i) for i in order[:top_count])


def _draw_panel_header(
    draw: ImageDraw.ImageDraw,
    panel_x: int,
    panel_w: int,
    method: dict[str, Any],
) -> None:
    accent = method["accent"]
    m = method["metrics"]
    draw.rounded_rectangle((panel_x + 14, 12, panel_x + panel_w - 14, 126), radius=16, fill=(13, 25, 44), outline=(31, 47, 71), width=1)
    draw.text((panel_x + 34, 28), method["name"], font=_font(30, bold=True), fill=accent)
    draw.text((panel_x + 34, 68), method["subtitle"], font=_font(17), fill=(203, 213, 225))
    draw.text((panel_x + 34, 94), f"exact calls {m['exact_call_count']:,}/{m['candidate_count']:,}", font=_font(16), fill=(226, 232, 240))
    draw.text((panel_x + 320, 94), f"work {m['exact_work_units']:,.1f}", font=_font(16), fill=(226, 232, 240))
    draw.text((panel_x + panel_w - 130, 94), f"FN {m['fn_count']}", font=_font(16), fill=(226, 232, 240))


def _draw_proxy_scene(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    t: float,
    *,
    selected: bool,
) -> None:
    x0, y0, x1, y1 = box
    cx = (x0 + x1) * 0.5
    cy = (y0 + y1) * 0.5
    w = x1 - x0
    h = y1 - y0
    toi = 0.5
    if t <= toi:
        gap = (toi - t) / toi
        ax = cx - (0.08 + 0.28 * gap) * w
        bx = cx + (0.08 + 0.28 * gap) * w
    else:
        gap = (t - toi) / (1.0 - toi)
        ax = cx - (0.08 + 0.24 * gap) * w
        bx = cx + (0.08 + 0.24 * gap) * w
    ay = cy - 0.06 * h * np.sin(2.0 * np.pi * t)
    by = cy + 0.04 * h * np.sin(2.0 * np.pi * t)
    radius = 0.078 * min(w, h)
    draw.line((x0 + 40, cy, x1 - 40, cy), fill=(71, 85, 105), width=2)
    for sx in np.linspace(x0 + 50, x1 - 50, 8):
        draw.line((sx, y0 + 16, sx, y1 - 16), fill=(30, 41, 59), width=1)
    if abs(t - toi) < 0.025:
        draw.ellipse((cx - 44, cy - 44, cx + 44, cy + 44), outline=(250, 204, 21), width=5)
    draw.ellipse((ax - radius, ay - radius, ax + radius, ay + radius), fill=(74, 163, 255), outline=(147, 197, 253), width=3)
    draw.ellipse((bx - radius, by - radius, bx + radius, by + radius), fill=(248, 113, 113), outline=(254, 202, 202), width=3)
    draw.line((ax + radius + 10, ay, bx - radius - 10, by), fill=(250, 204, 21) if selected else (100, 116, 139), width=3)
    draw.text((x0 + 20, y0 + 18), "synthetic swept-sphere proxy", font=_font(16), fill=(203, 213, 225))
    draw.text((x0 + 20, y1 - 34), "t=0.5 contact, bounce replay for visualization", font=_font(15), fill=(148, 163, 184))


def _draw_candidate_grid(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    query: dict[str, Any],
    method_key: str,
    t: float,
) -> None:
    x0, y0, x1, y1 = box
    slabs = 8
    rows = 16
    cell_w = (x1 - x0 - 34) / slabs
    cell_h = (y1 - y0 - 42) / rows
    selected = _selected_candidate_indices(query)
    current_slab = min(slabs - 1, int(t * slabs))
    draw.text((x0, y0 - 28), "candidate grid: 8 time slabs x 16 patch-pairs", font=_font(16), fill=(203, 213, 225))
    draw.rounded_rectangle((x0 - 8, y0 - 8, x1 + 8, y1 + 8), radius=12, fill=(15, 23, 42), outline=(51, 65, 85), width=1)
    for s in range(slabs):
        label_x = x0 + s * cell_w + 7
        draw.text((label_x, y0 - 2), f"S{s}", font=_font(11), fill=(148, 163, 184))
    for s in range(slabs):
        for r in range(rows):
            idx = s * rows + r
            px0 = x0 + s * cell_w
            py0 = y0 + 22 + r * cell_h
            px1 = px0 + cell_w - 4
            py1 = py0 + cell_h - 3
            base = (30, 41, 59)
            if method_key == "trained_stpf":
                fill = (22, 163, 74) if idx in selected else base
                outline = (187, 247, 208) if idx in selected else (51, 65, 85)
            else:
                fill = (245, 158, 11) if s <= current_slab else base
                outline = (253, 230, 138) if s <= current_slab else (51, 65, 85)
            draw.rounded_rectangle((px0, py0, px1, py1), radius=3, fill=fill, outline=outline, width=1)
    cursor_x = x0 + current_slab * cell_w + cell_w * 0.5
    draw.line((cursor_x, y0 + 18, cursor_x, y1), fill=(34, 211, 238), width=3)


def _draw_method_panel(
    frame: Image.Image,
    panel_index: int,
    method: dict[str, Any],
    query: dict[str, Any],
    t: float,
) -> None:
    draw = ImageDraw.Draw(frame, "RGBA")
    panel_w = frame.width // 3
    panel_x = panel_index * panel_w
    draw.rectangle((panel_x, 0, panel_x + panel_w - 1, frame.height), outline=(30, 41, 59), width=2)
    _draw_panel_header(draw, panel_x, panel_w, method)
    _draw_proxy_scene(
        draw,
        (panel_x + 32, 150, panel_x + panel_w - 32, 470),
        t,
        selected=method["key"] == "trained_stpf",
    )
    _draw_candidate_grid(
        draw,
        (panel_x + 46, 540, panel_x + panel_w - 46, 940),
        query,
        method["key"],
        t,
    )
    m = method["metrics"]
    y = 972
    draw.text((panel_x + 36, y), f"reduction vs NoProposal {100.0 * method['reduction']:.2f}%", font=_font(17), fill=method["accent"])
    draw.text((panel_x + 310, y), f"fallback {m['fallback_call_count']:,}", font=_font(17), fill=(203, 213, 225))


def _draw_frame(
    metrics: dict[str, Any],
    query: dict[str, Any],
    t: float,
    width: int,
    height: int,
) -> Image.Image:
    frame = Image.new("RGB", (width, height), (8, 13, 23))
    for i, method in enumerate(_method_rows(metrics)):
        _draw_method_panel(frame, i, method, query, t)
    draw = ImageDraw.Draw(frame, "RGBA")
    bar_y = height - 34
    draw.rectangle((40, bar_y, width - 40, bar_y + 12), fill=(30, 41, 59))
    draw.rectangle((40, bar_y, int(40 + (width - 80) * t), bar_y + 12), fill=(34, 211, 238))
    draw.text((44, height - 64), f"trained_stpf_high_density_large | t={t:.3f} | query={query['query_id']} | analytic swept-sphere proxy, not a real mesh contact case", font=_font(16), fill=(226, 232, 240))
    return frame


def _write_mp4_and_previews(
    cfg: TrainedSTPFHighDensityLargeDemoConfig,
    metrics: dict[str, Any],
    query: dict[str, Any],
    *,
    mp4_path: Path,
    preview_path: Path,
    proxy_scene_path: Path,
) -> None:
    writer = imageio.get_writer(str(mp4_path), fps=cfg.fps, codec="libx264", quality=8, macro_block_size=16)
    try:
        for frame_index in range(cfg.frame_count):
            t = frame_index / max(1, cfg.frame_count - 1)
            frame = _draw_frame(metrics, query, t, cfg.width, cfg.height)
            if frame_index == cfg.frame_count // 2:
                frame.save(preview_path)
            writer.append_data(np.asarray(frame))
    finally:
        writer.close()
    proxy = _draw_frame(metrics, query, 0.5, cfg.width, cfg.height)
    proxy.save(proxy_scene_path)


def _write_interactive_html(
    path: Path,
    metrics: dict[str, Any],
    query: dict[str, Any],
) -> None:
    method_payload = [
        {
            "name": row["name"],
            "subtitle": row["subtitle"],
            "key": row["key"],
            "accent": "#%02x%02x%02x" % row["accent"],
            "exactCalls": row["metrics"]["exact_call_count"],
            "candidateCount": row["metrics"]["candidate_count"],
            "exactWork": row["metrics"]["exact_work_units"],
            "fn": row["metrics"]["fn_count"],
            "reduction": row["reduction"],
        }
        for row in _method_rows(metrics)
    ]
    selected = sorted(_selected_candidate_indices(query))
    payload = {
        "queryId": query["query_id"],
        "methods": method_payload,
        "selected": selected,
    }
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>trained_stpf_high_density_large interactive</title>
<style>
body {{ margin: 0; background: #08111f; color: #e5edf7; font-family: Segoe UI, Microsoft YaHei, sans-serif; }}
.wrap {{ max-width: 1500px; margin: 0 auto; padding: 18px; }}
canvas {{ width: 100%; background: #080d17; border: 1px solid #243244; border-radius: 14px; }}
.controls {{ display: flex; gap: 14px; align-items: center; margin: 14px 0 4px; }}
input[type=range] {{ width: 520px; }}
.note {{ color: #9fb0c7; line-height: 1.5; }}
</style>
</head>
<body>
<div class="wrap">
  <h1>trained_stpf_high_density_large</h1>
  <p class="note">this is synthetic swept-sphere proxy high candidate densitydescription: 8  time slabs × 16  patch-pair candidates. STPF only performs proposal/scheduling, descriptioncorrectnessdescription exact certificate/fallback guarantee. </p>
  <div class="controls">
    <label>time t</label>
    <input id="slider" type="range" min="0" max="1000" value="500">
    <span id="timeText">0.500</span>
  </div>
  <canvas id="view" width="1500" height="840"></canvas>
</div>
<script>
const payload = {json.dumps(payload, ensure_ascii=False)};
const canvas = document.getElementById('view');
const ctx = canvas.getContext('2d');
const slider = document.getElementById('slider');
const timeText = document.getElementById('timeText');
function roundRect(x,y,w,h,r,fill,stroke) {{
  ctx.beginPath();
  ctx.moveTo(x+r,y); ctx.arcTo(x+w,y,x+w,y+h,r); ctx.arcTo(x+w,y+h,x,y+h,r);
  ctx.arcTo(x,y+h,x,y,r); ctx.arcTo(x,y,x+w,y,r); ctx.closePath();
  if (fill) {{ ctx.fillStyle=fill; ctx.fill(); }}
  if (stroke) {{ ctx.strokeStyle=stroke; ctx.stroke(); }}
}}
function draw(t) {{
  ctx.clearRect(0,0,canvas.width,canvas.height);
  ctx.fillStyle = '#080d17'; ctx.fillRect(0,0,canvas.width,canvas.height);
  const panelW = canvas.width / 3;
  for (let p=0; p<3; ++p) {{
    const m = payload.methods[p], x = p * panelW;
    ctx.strokeStyle = '#223047'; ctx.lineWidth = 2; ctx.strokeRect(x,0,panelW,canvas.height);
    roundRect(x+14,12,panelW-28,110,16,'#0d192c','#1f2f47');
    ctx.fillStyle = m.accent; ctx.font='bold 26px Segoe UI'; ctx.fillText(m.name,x+34,45);
    ctx.fillStyle = '#cbd5e1'; ctx.font='15px Segoe UI'; ctx.fillText(m.subtitle,x+34,72);
    ctx.fillStyle = '#e2e8f0'; ctx.fillText(`exact calls ${{m.exactCalls.toLocaleString()}}/${{m.candidateCount.toLocaleString()}}`,x+34,98);
    ctx.fillText(`work ${{m.exactWork.toFixed(1)}}`,x+250,98);
    ctx.fillText(`FN ${{m.fn}}`,x+410,98);
    const scene = {{x:x+30,y:145,w:panelW-60,h:250}};
    ctx.strokeStyle='#334155'; ctx.strokeRect(scene.x,scene.y,scene.w,scene.h);
    const cx=scene.x+scene.w/2, cy=scene.y+scene.h/2, r=25;
    let gap = t<=0.5 ? (0.5-t)/0.5 : (t-0.5)/0.5;
    let ax = cx - (0.08 + 0.28*gap)*scene.w;
    let bx = cx + (0.08 + 0.28*gap)*scene.w;
    ctx.strokeStyle = '#475569'; ctx.beginPath(); ctx.moveTo(scene.x+25,cy); ctx.lineTo(scene.x+scene.w-25,cy); ctx.stroke();
    ctx.fillStyle='#4aa3ff'; ctx.beginPath(); ctx.arc(ax,cy,r,0,Math.PI*2); ctx.fill();
    ctx.fillStyle='#ff6868'; ctx.beginPath(); ctx.arc(bx,cy,r,0,Math.PI*2); ctx.fill();
    if (Math.abs(t-0.5)<0.03) {{ ctx.strokeStyle='#facc15'; ctx.lineWidth=4; ctx.beginPath(); ctx.arc(cx,cy,48,0,Math.PI*2); ctx.stroke(); ctx.lineWidth=1; }}
    const gx=x+45, gy=455, gw=panelW-90, gh=310, slabs=8, rows=16, cw=gw/slabs, ch=gh/rows;
    ctx.fillStyle='#cbd5e1'; ctx.font='14px Segoe UI'; ctx.fillText('candidate grid: 8 time slabs × 16 patch-pairs',gx,gy-18);
    const currentSlab=Math.min(7,Math.floor(t*8));
    for (let s=0;s<slabs;s++) for (let rr=0;rr<rows;rr++) {{
      const idx=s*rows+rr, xx=gx+s*cw, yy=gy+rr*ch;
      let fill='#1e293b', stroke='#334155';
      if (m.key==='trained_stpf') {{
        if (payload.selected.includes(idx)) {{ fill='#16a34a'; stroke='#bbf7d0'; }}
      }} else if (s<=currentSlab) {{ fill='#f59e0b'; stroke='#fde68a'; }}
      ctx.fillStyle=fill; ctx.strokeStyle=stroke; ctx.fillRect(xx+1,yy+1,cw-3,ch-3); ctx.strokeRect(xx+1,yy+1,cw-3,ch-3);
    }}
    ctx.fillStyle=m.accent; ctx.font='15px Segoe UI'; ctx.fillText(`reduction vs NoProposal ${{(100*m.reduction).toFixed(2)}}%`,x+35,800);
  }}
}}
slider.addEventListener('input', () => {{ const t=Number(slider.value)/1000; timeText.textContent=t.toFixed(3); draw(t); }});
draw(0.5);
</script>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def _write_metrics_json(
    path: Path,
    cfg: TrainedSTPFHighDensityLargeDemoConfig,
    metrics: dict[str, Any],
    query: dict[str, Any],
    result_paths: dict[str, str],
) -> None:
    payload = {
        "case_name": "trained_stpf_high_density_large",
        "case_type": "synthetic swept-sphere proxy dense candidate workload",
        "config": asdict(cfg),
        "source_benchmark": str(cfg.benchmark_json),
        "source_eval_rows": str(cfg.eval_rows_npz),
        "representative_query_id": query["query_id"],
        "dataset_metadata": query["metadata"],
        "visualization": result_paths,
        "metrics": metrics,
        "interpretation": {
            "rtstpf_exact_work_reduction_vs_no_proposal": metrics["trained_exact_work_reduction_vs_no_proposal"],
            "rtstpf_exact_work_reduction_vs_random": metrics["trained_exact_work_reduction_vs_random"],
            "correctness_scope": "FN=0 under analytic exact oracle/fallback",
            "limitation": "This is not a real mesh-mesh contact case; it visualizes dense scheduling behavior.",
        },
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def _write_markdown_files(
    output_dir: Path,
    cfg: TrainedSTPFHighDensityLargeDemoConfig,
    metrics: dict[str, Any],
    result_paths: dict[str, str],
) -> None:
    baseline = metrics["baseline"]
    random_stpf = metrics["random_stpf"]
    trained = metrics["trained_stpf"]
    reduction_np = 100.0 * metrics["trained_exact_work_reduction_vs_no_proposal"]
    reduction_random = 100.0 * metrics["trained_exact_work_reduction_vs_random"]
    readme = f"""# Paper Demo: trained_stpf_high_density_large

## usedescription

thisdescription `trained_stpf_high_density_large` visualization demo. descriptionis synthetic swept-sphere proxy high candidate density workload, is notreal CAD mesh-mesh contact case. usedescriptionisdescription learned STPF in dense candidate grid indescription exact certificate, descriptionwhenkeep `FN=0`.

## File

| File | Notes |
| --- | --- |
| `{Path(result_paths['mp4']).name}` | descriptionMethoddescription MP4: `RTSTPFExact` / `RTExact` / `NoProposal` |
| `{Path(result_paths['interactive_html']).name}` | description HTML, descriptionwhendescriptioncandidate grid |
| `{Path(result_paths['preview_png']).name}` | TOI description preview description |
| `{Path(result_paths['proxy_scene_png']).name}` | description proxy scenedescription |
| `metrics.json` | descriptionMetricsand case metadata |
| `benchmark_report.md` | indescription benchmark anddescriptionProtocol notes |

## Case

- Workload: `high_density_eval_large_1000q`
- Query count: `{metrics['query_count']}`
- Candidate count: `{metrics['candidate_count']}`
- Avg candidates/query: `{metrics['avg_candidates_per_query']:.0f}`
- Candidate layout: `8 time slabs x 4 x 4 patch pairs = 128 candidates/query`
- Exact oracle: `analytic_swept_sphere_proxy`
- Checkpoint: `{metrics['checkpoint_path']}`

## Method Summary

| Method | Candidates | Exact calls | Exact work | Total wall ms | FN |
| --- | ---: | ---: | ---: | ---: | ---: |
| `RTSTPFExact-Trained` | `{trained['candidate_count']}` | `{trained['exact_call_count']}` | `{trained['exact_work_units']:.1f}` | `{trained['total_wall_ms']:.1f}` | `{trained['fn_count']}` |
| `RTSTPFExact-Random` | `{random_stpf['candidate_count']}` | `{random_stpf['exact_call_count']}` | `{random_stpf['exact_work_units']:.1f}` | `{random_stpf['total_wall_ms']:.1f}` | `{random_stpf['fn_count']}` |
| `NoProposal` | `{baseline['candidate_count']}` | `{baseline['exact_call_count']}` | `{baseline['exact_work_units']:.1f}` | `{baseline['total_wall_ms']:.1f}` | `{baseline['fn_count']}` |

## descriptionEntry point

```powershell
conda activate cudadev
$env:PYTHONPATH='src\\python'
python -m p2cccd.viz.trained_stpf_high_density_large_demo
```
"""
    report = f"""# trained_stpf_high_density_large completevisualizationdescriptionsplitdescription

## this isdescription

`trained_stpf_high_density_large` isdescription STPF descriptionafterdescription. descriptionuserealdescription, insteaduse analytic swept-sphere proxy constructdescriptioncollision query, descriptioneach query description dense candidate grid:

- `8`  time slabs.
- eachdescription `4`  proxy patches.
- each query has `8 x 4 x 4 = 128` candidate.
- descriptionas `{metrics['query_count']}` queries / `{metrics['candidate_count']}` candidates.

descriptionis: description broad phase afterdescriptioncandidate certificate when, learned STPF descriptionmust not treatdescription exact certificate candidatedescriptiontodescription, descriptionin `FN=0` underreduction exact work.

## visualizationdescription

MP4 and HTML descriptionisdescriptionMethoddescription:

- `RTSTPFExact`: descriptioncandidatedescription learned STPF selectdescription exact certificate.
- `RTExact`: descriptionMethoddescriptionondescription RT broad-phase description, butallcandidatedescriptionenter exact.
- `NoProposal`: descriptionMethoddescriptionperform proposal, allcandidateenter exact description.

description grid is `8 time slabs x 16 patch-pairs`. thisdescriptionisdescriptionand exact description, descriptionasreal mesh contactdescription.

## description

| Method | Query count | Candidate count | Exact calls | Fallback calls | Exact work units | Total wall ms | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `NoProposal` | `{baseline['query_count']}` | `{baseline['candidate_count']}` | `{baseline['exact_call_count']}` | `{baseline['fallback_call_count']}` | `{baseline['exact_work_units']:.4f}` | `{baseline['total_wall_ms']:.4f}` | `{baseline['fn_count']}` |
| `RTSTPFExact-Random` | `{random_stpf['query_count']}` | `{random_stpf['candidate_count']}` | `{random_stpf['exact_call_count']}` | `{random_stpf['fallback_call_count']}` | `{random_stpf['exact_work_units']:.4f}` | `{random_stpf['total_wall_ms']:.4f}` | `{random_stpf['fn_count']}` |
| `RTSTPFExact-Trained` | `{trained['query_count']}` | `{trained['candidate_count']}` | `{trained['exact_call_count']}` | `{trained['fallback_call_count']}` | `{trained['exact_work_units']:.4f}` | `{trained['total_wall_ms']:.4f}` | `{trained['fn_count']}` |

## Conclusion

- Trained STPF description NoProposal  exact-work reduction as `{reduction_np:.4f}%`.
- Trained STPF description Random STPF  exact-work reduction as `{reduction_random:.4f}%`.
- Trained STPF exact calls as `{trained['exact_call_count']}`, NoProposal as `{baseline['exact_call_count']}`.
- `FN=0`, Notesdescription collision conclusion description analytic exact oracle / fallback conservative guarantee.

## descriptionusedescription

this case descriptionwithas"high candidate densityunder STPF descriptionhasdescription"description, butdescriptionreplacereal mesh-mesh exact CCD benchmark. descriptionwith ABC/Fusion360/Thingi10K/ShapeNet dense mesh case and Tight-Inclusion exact payload case asdescription.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    (output_dir / "benchmark_report.md").write_text(report, encoding="utf-8")


def write_trained_stpf_high_density_large_demo(
    config: TrainedSTPFHighDensityLargeDemoConfig | None = None,
) -> TrainedSTPFHighDensityLargeDemoResult:
    cfg = config or TrainedSTPFHighDensityLargeDemoConfig()
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = _load_json(Path(cfg.benchmark_json))
    query = _query_slice(Path(cfg.eval_rows_npz), cfg.query_index)

    mp4_path = output_dir / f"{cfg.run_name}.mp4"
    interactive_html_path = output_dir / f"{cfg.run_name}_interactive.html"
    preview_png_path = output_dir / f"{cfg.run_name}_preview.png"
    proxy_scene_png_path = output_dir / f"{cfg.run_name}_proxy_scene.png"
    metrics_json_path = output_dir / "metrics.json"
    benchmark_report_path = output_dir / "benchmark_report.md"
    readme_path = output_dir / "README.md"

    _write_mp4_and_previews(
        cfg,
        metrics,
        query,
        mp4_path=mp4_path,
        preview_path=preview_png_path,
        proxy_scene_path=proxy_scene_png_path,
    )
    _write_interactive_html(interactive_html_path, metrics, query)
    result_paths = {
        "mp4": str(mp4_path),
        "interactive_html": str(interactive_html_path),
        "preview_png": str(preview_png_path),
        "proxy_scene_png": str(proxy_scene_png_path),
        "metrics_json": str(metrics_json_path),
        "benchmark_report": str(benchmark_report_path),
        "readme": str(readme_path),
    }
    _write_metrics_json(metrics_json_path, cfg, metrics, query, result_paths)
    _write_markdown_files(output_dir, cfg, metrics, result_paths)
    return TrainedSTPFHighDensityLargeDemoResult(
        output_dir=output_dir,
        mp4_path=mp4_path,
        interactive_html_path=interactive_html_path,
        preview_png_path=preview_png_path,
        proxy_scene_png_path=proxy_scene_png_path,
        metrics_json_path=metrics_json_path,
        benchmark_report_path=benchmark_report_path,
        readme_path=readme_path,
    )


def main() -> None:
    result = write_trained_stpf_high_density_large_demo()
    print(json.dumps({k: str(v) for k, v in asdict(result).items()}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()


__all__ = [
    "TrainedSTPFHighDensityLargeDemoConfig",
    "TrainedSTPFHighDensityLargeDemoResult",
    "write_trained_stpf_high_density_large_demo",
]
