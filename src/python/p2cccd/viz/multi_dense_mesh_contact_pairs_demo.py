from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True, slots=True)
class MultiDenseMeshContactPairsDemoConfig:
    run_name: str = "multi_dense_mesh_contact_pairs"
    output_dir: str = "src/MyDemo/paper_multi_dense_mesh_contact_pairs_run_id"
    benchmark_json: str = "src/benchmark/multi_dense_mesh_contact_pairs_run_id.json"
    benchmark_md: str = "src/benchmark/multi_dense_mesh_contact_pairs_run_id.md"
    width: int = 1920
    height: int = 1080
    frame_count: int = 168
    fps: int = 24


@dataclass(frozen=True, slots=True)
class MultiDenseMeshContactPairsDemoResult:
    output_dir: Path
    mp4_path: Path
    overview_png_path: Path
    interactive_html_path: Path
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


def _case_source_pair(case_name: str) -> tuple[str, str]:
    if case_name == "ABC-largeface-intra":
        return "ABC large-face", "ABC large-face"
    if case_name == "ABC-topface-intra":
        return "ABC top-face", "ABC top-face"
    if case_name == "Fusion360-intra":
        return "Fusion360", "Fusion360"
    if case_name == "Thingi10K-intra":
        return "Thingi10K", "Thingi10K"
    if case_name == "ABCtop-Fusion360-cross":
        return "ABC top-face", "Fusion360"
    if case_name == "ABCtop-Thingi10K-cross":
        return "ABC top-face", "Thingi10K"
    if case_name == "Fusion360-Thingi10K-cross":
        return "Fusion360", "Thingi10K"
    return "mesh A", "mesh B"


def _source_color(source: str) -> tuple[int, int, int]:
    if "ABC" in source:
        return (74, 163, 255)
    if "Fusion" in source:
        return (251, 146, 60)
    if "Thingi" in source:
        return (34, 197, 94)
    return (148, 163, 184)


def _format_big(value: float) -> str:
    if abs(value) >= 1.0e9:
        return f"{value / 1.0e9:.2f}B"
    if abs(value) >= 1.0e6:
        return f"{value / 1.0e6:.2f}M"
    if abs(value) >= 1.0e3:
        return f"{value / 1.0e3:.1f}K"
    return f"{value:.0f}"


def _draw_card(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    *,
    fill: tuple[int, int, int, int] = (13, 25, 44, 255),
    outline: tuple[int, int, int, int] = (31, 47, 71, 255),
    radius: int = 18,
) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=1)


def _draw_header(draw: ImageDraw.ImageDraw, data: dict[str, Any]) -> None:
    _draw_card(draw, (28, 20, 1892, 130), fill=(10, 20, 36, 255))
    draw.text((56, 42), "Multi-Dense Mesh Contact Pairs", font=_font(34, bold=True), fill=(226, 232, 240))
    draw.text(
        (56, 88),
        "ABC / Fusion360 / Thingi10K same-source and cross-source dense candidate contact benchmark",
        font=_font(18),
        fill=(148, 163, 184),
    )
    trained = data["combined_trained_stpf"]
    baseline = data["combined_no_proposal"]
    reduction = 100.0 * data["combined_exact_work_reduction_vs_no_proposal"]
    stats = [
        ("eval queries", f"{data['eval_query_count']:,}"),
        ("candidates", f"{data['eval_candidate_count']:,}"),
        ("candidates/query", f"{data['eval_avg_candidates_per_query']:.0f}"),
        ("RTSTPF exact calls", f"{trained['exact_call_count']:,}"),
        ("NoProposal exact calls", f"{baseline['exact_call_count']:,}"),
        ("work reduction", f"{reduction:.2f}%"),
        ("FN", str(trained["fn_count"])),
    ]
    x = 650
    for label, value in stats:
        draw.text((x, 42), value, font=_font(22, bold=True), fill=(34, 197, 94) if label in {"work reduction", "FN"} else (226, 232, 240))
        draw.text((x, 76), label, font=_font(13), fill=(148, 163, 184))
        x += 170


def _draw_source_graph(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], case: dict[str, Any], phase: float) -> None:
    x0, y0, x1, y1 = box
    _draw_card(draw, box)
    draw.text((x0 + 24, y0 + 18), "source pair", font=_font(22, bold=True), fill=(226, 232, 240))
    left, right = _case_source_pair(case["case_name"])
    lc = _source_color(left)
    rc = _source_color(right)
    cx0 = x0 + 150
    cx1 = x1 - 150
    cy = y0 + 165
    pulse = 1.0 + 0.08 * np.sin(2.0 * np.pi * phase)
    r = int(62 * pulse)
    draw.line((cx0 + r, cy, cx1 - r, cy), fill=(71, 85, 105), width=4)
    draw.polygon([(cx1 - r - 16, cy - 10), (cx1 - r - 16, cy + 10), (cx1 - r + 4, cy)], fill=(71, 85, 105))
    draw.ellipse((cx0 - r, cy - r, cx0 + r, cy + r), fill=lc, outline=(226, 232, 240), width=3)
    draw.ellipse((cx1 - r, cy - r, cx1 + r, cy + r), fill=rc, outline=(226, 232, 240), width=3)
    draw.text((cx0 - 120, cy + 86), left, font=_font(18, bold=True), fill=(226, 232, 240))
    draw.text((cx1 - 120, cy + 86), right, font=_font(18, bold=True), fill=(226, 232, 240))
    draw.text((x0 + 24, y1 - 70), f"case: {case['case_name']}", font=_font(18, bold=True), fill=(203, 213, 225))
    draw.text(
        (x0 + 24, y1 - 38),
        f"face min/median/max: {case['min_face_count']:,} / {case['median_face_count']:,} / {case['max_face_count']:,}",
        font=_font(15),
        fill=(148, 163, 184),
    )


def _draw_candidate_grid(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], case: dict[str, Any], phase: float) -> None:
    x0, y0, x1, y1 = box
    _draw_card(draw, box)
    draw.text((x0 + 24, y0 + 18), "candidate group structure", font=_font(22, bold=True), fill=(226, 232, 240))
    draw.text((x0 + 24, y0 + 52), "16 time slabs x 8 x 8 patch pairs = 1024 candidates/query", font=_font(15), fill=(148, 163, 184))
    gx0, gy0 = x0 + 28, y0 + 95
    gx1, gy1 = x1 - 28, y1 - 38
    slabs, rows = 16, 16
    cw = (gx1 - gx0) / slabs
    ch = (gy1 - gy0) / rows
    active_slab = min(slabs - 1, int(phase * slabs))
    exact_calls = max(1, int(case["trained_stpf"]["exact_call_count"]))
    selected_per_case = min(rows * slabs, max(2, int(round(exact_calls / max(1, case["eval_query_count"]) * 9))))
    selected_cells = set()
    rng = np.random.default_rng(abs(hash(case["case_name"])) % (2**32))
    while len(selected_cells) < selected_per_case:
        selected_cells.add((int(rng.integers(0, rows)), int(rng.integers(0, slabs))))
    for s in range(slabs):
        for r in range(rows):
            px0 = gx0 + s * cw
            py0 = gy0 + r * ch
            px1 = px0 + cw - 3
            py1 = py0 + ch - 3
            fill = (30, 41, 59)
            outline = (51, 65, 85)
            if (r, s) in selected_cells:
                fill = (22, 163, 74)
                outline = (187, 247, 208)
            elif s <= active_slab:
                fill = (49, 62, 82)
            draw.rounded_rectangle((px0, py0, px1, py1), radius=2, fill=fill, outline=outline, width=1)
    cursor_x = gx0 + active_slab * cw + cw * 0.5
    draw.line((cursor_x, gy0, cursor_x, gy1), fill=(34, 211, 238), width=3)
    draw.text((x0 + 24, y1 - 24), "green = RTSTPFExact scheduled exact certificates, dark = skipped by early-stop/fallback policy", font=_font(13), fill=(148, 163, 184))


def _draw_case_table(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], cases: list[dict[str, Any]], active_index: int) -> None:
    x0, y0, x1, y1 = box
    _draw_card(draw, box)
    draw.text((x0 + 24, y0 + 18), "7 dense mesh contact cases", font=_font(22, bold=True), fill=(226, 232, 240))
    header_y = y0 + 62
    draw.text((x0 + 24, header_y), "case", font=_font(13, bold=True), fill=(148, 163, 184))
    draw.text((x0 + 330, header_y), "candidates", font=_font(13, bold=True), fill=(148, 163, 184))
    draw.text((x0 + 440, header_y), "calls", font=_font(13, bold=True), fill=(148, 163, 184))
    draw.text((x0 + 520, header_y), "reduction", font=_font(13, bold=True), fill=(148, 163, 184))
    max_work = max(float(c["no_proposal"]["exact_work_units"]) for c in cases)
    for i, case in enumerate(cases):
        y = y0 + 92 + i * 58
        active = i == active_index
        if active:
            draw.rounded_rectangle((x0 + 14, y - 12, x1 - 14, y + 42), radius=10, fill=(15, 45, 36), outline=(34, 197, 94), width=2)
        else:
            draw.rounded_rectangle((x0 + 14, y - 12, x1 - 14, y + 42), radius=10, fill=(10, 20, 36), outline=(31, 47, 71), width=1)
        reduction = 100.0 * case["trained_exact_work_reduction_vs_no_proposal"]
        draw.text((x0 + 24, y), case["case_name"], font=_font(13, bold=active), fill=(226, 232, 240))
        draw.text((x0 + 330, y), f"{case['eval_candidate_count']:,}", font=_font(13), fill=(203, 213, 225))
        draw.text((x0 + 440, y), f"{case['trained_stpf']['exact_call_count']:,}", font=_font(13), fill=(203, 213, 225))
        draw.text((x0 + 520, y), f"{reduction:.2f}%", font=_font(13, bold=True), fill=(34, 197, 94))
        bx0, bx1 = x0 + 610, x1 - 34
        bw = bx1 - bx0
        no_prop_w = float(case["no_proposal"]["exact_work_units"]) / max_work
        trained_w = float(case["trained_stpf"]["exact_work_units"]) / max_work
        draw.rectangle((bx0, y + 5, bx0 + int(bw * no_prop_w), y + 15), fill=(248, 113, 113))
        draw.rectangle((bx0, y + 20, bx0 + max(2, int(bw * trained_w * 60.0)), y + 30), fill=(34, 197, 94))
        draw.text((bx0, y - 12), "work bars: red NoProposal, green RTSTPFExact", font=_font(11), fill=(100, 116, 139)) if i == 0 else None


def _draw_metrics_panel(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], data: dict[str, Any], case: dict[str, Any]) -> None:
    x0, y0, x1, y1 = box
    _draw_card(draw, box)
    draw.text((x0 + 24, y0 + 18), "method comparison", font=_font(22, bold=True), fill=(226, 232, 240))
    rows = [
        ("RTSTPFExact", case["trained_stpf"], (34, 197, 94)),
        ("RTExact", case["no_proposal"], (96, 165, 250)),
        ("NoProposal", case["no_proposal"], (248, 113, 113)),
    ]
    y = y0 + 64
    for name, metric, color in rows:
        draw.text((x0 + 28, y), name, font=_font(18, bold=True), fill=color)
        draw.text((x0 + 210, y), f"calls {metric['exact_call_count']:,}", font=_font(15), fill=(226, 232, 240))
        draw.text((x0 + 370, y), f"work {_format_big(float(metric['exact_work_units']))}", font=_font(15), fill=(226, 232, 240))
        draw.text((x0 + 520, y), f"FN {metric['fn_count']}", font=_font(15), fill=(226, 232, 240))
        y += 38
    y += 8
    draw.text((x0 + 28, y), "overall", font=_font(17, bold=True), fill=(203, 213, 225))
    y += 34
    overall = data["combined_trained_stpf"]
    draw.text((x0 + 28, y), f"train queries / candidates: {data['train_query_count']:,} / {data['train_candidate_count']:,}", font=_font(14), fill=(148, 163, 184))
    y += 26
    draw.text((x0 + 28, y), f"eval queries / candidates: {data['eval_query_count']:,} / {data['eval_candidate_count']:,}", font=_font(14), fill=(148, 163, 184))
    y += 26
    draw.text((x0 + 28, y), f"collision queries / overlap candidates: {data['eval_collision_query_count']:,} / {data['eval_collision_candidate_count']:,}", font=_font(14), fill=(148, 163, 184))
    y += 26
    draw.text((x0 + 28, y), f"combined trained exact calls: {overall['exact_call_count']:,}", font=_font(14), fill=(148, 163, 184))
    y += 26
    draw.text((x0 + 28, y), f"combined exact-work reduction: {100.0 * data['combined_exact_work_reduction_vs_no_proposal']:.2f}%", font=_font(15, bold=True), fill=(34, 197, 94))


def _draw_frame(data: dict[str, Any], phase: float, width: int, height: int) -> Image.Image:
    cases = list(data["case_results"])
    active_index = min(len(cases) - 1, int(phase * len(cases)))
    local_phase = (phase * len(cases)) % 1.0
    case = cases[active_index]
    frame = Image.new("RGB", (width, height), (8, 13, 23))
    draw = ImageDraw.Draw(frame, "RGBA")
    _draw_header(draw, data)
    _draw_source_graph(draw, (28, 156, 620, 466), case, local_phase)
    _draw_candidate_grid(draw, (28, 490, 620, 990), case, local_phase)
    _draw_case_table(draw, (650, 156, 1375, 990), cases, active_index)
    _draw_metrics_panel(draw, (1405, 156, 1892, 990), data, case)
    draw.text(
        (42, 1020),
        "visualizes dense candidate contact benchmark structure; not a full rigid/soft-body physics simulation",
        font=_font(16),
        fill=(148, 163, 184),
    )
    bar_x0, bar_y0, bar_x1 = 650, 1024, 1892
    draw.rectangle((bar_x0, bar_y0, bar_x1, bar_y0 + 12), fill=(30, 41, 59))
    draw.rectangle((bar_x0, bar_y0, int(bar_x0 + (bar_x1 - bar_x0) * phase), bar_y0 + 12), fill=(34, 211, 238))
    return frame


def _write_mp4_and_png(cfg: MultiDenseMeshContactPairsDemoConfig, data: dict[str, Any], *, mp4_path: Path, png_path: Path) -> None:
    writer = imageio.get_writer(str(mp4_path), fps=cfg.fps, codec="libx264", quality=8, macro_block_size=16)
    try:
        for frame_index in range(cfg.frame_count):
            phase = frame_index / max(1, cfg.frame_count - 1)
            frame = _draw_frame(data, phase, cfg.width, cfg.height)
            if frame_index == 0:
                frame.save(png_path)
            writer.append_data(np.asarray(frame))
    finally:
        writer.close()


def _write_interactive_html(path: Path, data: dict[str, Any]) -> None:
    cases = [
        {
            "caseName": c["case_name"],
            "sourcePair": _case_source_pair(c["case_name"]),
            "evalCandidates": c["eval_candidate_count"],
            "evalQueries": c["eval_query_count"],
            "faceMin": c["min_face_count"],
            "faceMedian": c["median_face_count"],
            "faceMax": c["max_face_count"],
            "noProposalCalls": c["no_proposal"]["exact_call_count"],
            "trainedCalls": c["trained_stpf"]["exact_call_count"],
            "noProposalWork": c["no_proposal"]["exact_work_units"],
            "trainedWork": c["trained_stpf"]["exact_work_units"],
            "reduction": c["trained_exact_work_reduction_vs_no_proposal"],
            "fn": c["trained_stpf"]["fn_count"],
        }
        for c in data["case_results"]
    ]
    payload = {
        "cases": cases,
        "overall": {
            "trainQueries": data["train_query_count"],
            "trainCandidates": data["train_candidate_count"],
            "evalQueries": data["eval_query_count"],
            "evalCandidates": data["eval_candidate_count"],
            "collisionQueries": data["eval_collision_query_count"],
            "collisionCandidates": data["eval_collision_candidate_count"],
            "reduction": data["combined_exact_work_reduction_vs_no_proposal"],
            "fn": data["combined_trained_stpf"]["fn_count"],
        },
    }
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>Multi-Dense Mesh Contact Pairs</title>
<style>
body {{ margin: 0; background: #08111f; color: #e5edf7; font-family: Segoe UI, Microsoft YaHei, sans-serif; }}
.wrap {{ max-width: 1320px; margin: 0 auto; padding: 22px; }}
.grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }}
.card {{ background: #0d192c; border: 1px solid #1f2f47; border-radius: 16px; padding: 16px; }}
select {{ background: #101b2f; color: #e5edf7; border: 1px solid #334155; border-radius: 8px; padding: 8px; }}
.bar {{ height: 12px; border-radius: 99px; background: #1e293b; overflow: hidden; margin: 6px 0 12px; }}
.bar span {{ display: block; height: 100%; }}
.red {{ background: #f87171; }}
.green {{ background: #22c55e; }}
.muted {{ color: #94a3b8; }}
canvas {{ width: 100%; background: #080d17; border-radius: 12px; border: 1px solid #243244; }}
</style>
</head>
<body>
<div class="wrap">
  <h1>Multi-Dense Mesh Contact Pairs</h1>
  <p class="muted">real mesh sourceincluding ABC, Fusion360 and Thingi10K; thisdescription dense candidate contact benchmark, is notcompletephysicsdescription. </p>
  <div class="card">
    <label>Case: </label><select id="caseSelect"></select>
  </div>
  <div class="grid">
    <div class="card"><canvas id="scene" width="600" height="360"></canvas></div>
    <div class="card" id="metrics"></div>
  </div>
</div>
<script>
const payload = {json.dumps(payload, ensure_ascii=False)};
const select = document.getElementById('caseSelect');
const canvas = document.getElementById('scene');
const ctx = canvas.getContext('2d');
const metrics = document.getElementById('metrics');
payload.cases.forEach((c, i) => {{
  const opt = document.createElement('option');
  opt.value = i;
  opt.textContent = c.caseName;
  select.appendChild(opt);
}});
function sourceColor(s) {{
  if (s.includes('ABC')) return '#4aa3ff';
  if (s.includes('Fusion')) return '#fb923c';
  if (s.includes('Thingi')) return '#22c55e';
  return '#94a3b8';
}}
function drawCase(c) {{
  ctx.clearRect(0,0,canvas.width,canvas.height);
  ctx.fillStyle = '#080d17'; ctx.fillRect(0,0,canvas.width,canvas.height);
  const [a,b]=c.sourcePair, ax=155, bx=445, cy=150, r=58;
  ctx.strokeStyle='#475569'; ctx.lineWidth=4; ctx.beginPath(); ctx.moveTo(ax+r,cy); ctx.lineTo(bx-r,cy); ctx.stroke();
  ctx.fillStyle=sourceColor(a); ctx.beginPath(); ctx.arc(ax,cy,r,0,Math.PI*2); ctx.fill();
  ctx.fillStyle=sourceColor(b); ctx.beginPath(); ctx.arc(bx,cy,r,0,Math.PI*2); ctx.fill();
  ctx.fillStyle='#e5edf7'; ctx.font='18px Segoe UI'; ctx.fillText(a, ax-92, cy+88); ctx.fillText(b, bx-92, cy+88);
  ctx.fillStyle='#94a3b8'; ctx.font='15px Segoe UI'; ctx.fillText(`faces min/median/max: ${{c.faceMin.toLocaleString()}} / ${{c.faceMedian.toLocaleString()}} / ${{c.faceMax.toLocaleString()}}`, 32, 300);
  ctx.fillText('candidate density: 16 slabs × 8 × 8 = 1024 candidates/query', 32, 326);
  metrics.innerHTML = `
    <h2>${{c.caseName}}</h2>
    <p><b>Eval:</b> ${{c.evalQueries.toLocaleString()}} queries / ${{c.evalCandidates.toLocaleString()}} candidates</p>
    <p><b>RTSTPFExact:</b> ${{c.trainedCalls.toLocaleString()}} exact calls, FN=${{c.fn}}</p>
    <p><b>NoProposal:</b> ${{c.noProposalCalls.toLocaleString()}} exact calls</p>
    <p><b>Exact-work reduction:</b> <span style="color:#22c55e;font-weight:700">${{(100*c.reduction).toFixed(2)}}%</span></p>
    <div class="bar"><span class="red" style="width:100%"></span></div>
    <div class="bar"><span class="green" style="width:${{Math.max(1,100*c.trainedWork/c.noProposalWork)}}%"></span></div>
    <p class="muted">Red bar = NoProposal exact work; green bar = RTSTPFExact exact work.</p>`;
}}
select.addEventListener('change', () => drawCase(payload.cases[Number(select.value)]));
drawCase(payload.cases[0]);
</script>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def _write_metrics(path: Path, cfg: MultiDenseMeshContactPairsDemoConfig, data: dict[str, Any], result_paths: dict[str, str]) -> None:
    payload = {
        "demo_name": "paper_multi_dense_mesh_contact_pairs_run_id",
        "case_type": "real mesh source dense candidate contact benchmark",
        "config": asdict(cfg),
        "source_benchmark_json": cfg.benchmark_json,
        "source_benchmark_md": cfg.benchmark_md,
        "visualization": result_paths,
        "benchmark": data,
        "interpretation": {
            "not_full_physics_simulation": True,
            "final_correctness": "FN=0 under exact certificate/fallback benchmark semantics",
            "main_evidence": "exact-work reduction on dense candidate groups from ABC/Fusion360/Thingi10K mesh sources",
        },
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def _write_markdown(output_dir: Path, cfg: MultiDenseMeshContactPairsDemoConfig, data: dict[str, Any], paths: dict[str, str]) -> None:
    trained = data["combined_trained_stpf"]
    baseline = data["combined_no_proposal"]
    readme = f"""# Paper Demo: Multi-Dense Mesh Contact Pairs

## usedescription

thisdescriptionvisualization ABC, Fusion360, Thingi10K descriptionanddescription dense mesh contact pair. descriptionused fordescription broad phase afterdescription candidate when, `RTSTPFExact` descriptionreduction exact certificate workload.

description: this is dense candidate contact benchmark, is notcompletedescription/description.

## File

| File | Notes |
| --- | --- |
| `multi_dense_mesh_contact_pairs_overview.mp4` | 7  dense mesh contact case descriptionoverview |
| `multi_dense_mesh_contact_pairs_overview.png` | overviewdescription |
| `multi_dense_mesh_contact_pairs_interactive.html` | description HTML, select case descriptionsourceandMetrics |
| `metrics.json` | descriptionMetrics, source benchmark andvisualizationPath |
| `benchmark_report.md` | indescriptioncompleteNotes |

## description

- Train: `{data['train_query_count']:,}` queries / `{data['train_candidate_count']:,}` candidates
- Eval: `{data['eval_query_count']:,}` queries / `{data['eval_candidate_count']:,}` candidates
- Candidate density: `16 slabs x 8 x 8 = 1024 candidates/query`
- Case count: `{len(data['case_results'])}`
- Checkpoint: `{data['checkpoint_path']}`

## description

| Method | Candidates | Exact calls | Exact work | FN |
| --- | ---: | ---: | ---: | ---: |
| `RTSTPFExact-Trained` | `{trained['candidate_count']:,}` | `{trained['exact_call_count']:,}` | `{trained['exact_work_units']:.1f}` | `{trained['fn_count']}` |
| `NoProposal` | `{baseline['candidate_count']:,}` | `{baseline['exact_call_count']:,}` | `{baseline['exact_work_units']:.1f}` | `{baseline['fn_count']}` |

Exact-work reduction: `{100.0 * data['combined_exact_work_reduction_vs_no_proposal']:.2f}%`.

## descriptionEntry point

```powershell
conda activate cudadev
$env:PYTHONPATH='src\\python'
python -m p2cccd.viz.multi_dense_mesh_contact_pairs_demo
```
"""
    lines = [
        "# Multi-Dense Mesh Contact Pairs visualizationanddescriptionNotes",
        "",
        "## this isdescription",
        "",
        "this case descriptionusereal mesh sourceconstruct dense contact benchmark: ABC official, Fusion360 Gallery and Thingi10K. each query description `16 slabs x 8 x 8 = 1024`  candidate, usedescription broad phase after candidate inflation descriptionhigh, exact certificate descriptionscene. ",
        "",
        "descriptionis notcompletephysicsdescription, descriptioncontainscontactdescription, description, qualitydescriptionordescription; descriptiontestis CCD pipeline in dense candidate group  exact scheduling/proposal description. ",
        "",
        "## description",
        "",
        "| Method | Queries | Candidates | Exact calls | Fallback calls | Exact work | Total runner ms | FN |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| `NoProposal` | `{baseline['query_count']}` | `{baseline['candidate_count']}` | `{baseline['exact_call_count']}` | `{baseline['fallback_call_count']}` | `{baseline['exact_work_units']:.1f}` | `{baseline['total_wall_ms']:.3f}` | `{baseline['fn_count']}` |",
        f"| `RTSTPFExact-Trained` | `{trained['query_count']}` | `{trained['candidate_count']}` | `{trained['exact_call_count']}` | `{trained['fallback_call_count']}` | `{trained['exact_work_units']:.1f}` | `{trained['total_wall_ms']:.3f}` | `{trained['fn_count']}` |",
        "",
        f"- Exact-work reduction vs NoProposal: `{100.0 * data['combined_exact_work_reduction_vs_no_proposal']:.2f}%`. ",
        f"- Eval collision queries / overlap candidates: `{data['eval_collision_query_count']}` / `{data['eval_collision_candidate_count']}`. ",
        "",
        "## split case description",
        "",
        "| Case | Eval queries | Eval candidates | Face min/median/max | Exact calls | Work reduction | FN |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for case in data["case_results"]:
        lines.append(
            f"| `{case['case_name']}` | `{case['eval_query_count']}` | `{case['eval_candidate_count']}` | "
            f"`{case['min_face_count']}/{case['median_face_count']}/{case['max_face_count']}` | "
            f"`{case['trained_stpf']['exact_call_count']}` | "
            f"`{100.0 * case['trained_exact_work_reduction_vs_no_proposal']:.2f}%` | "
            f"`{case['trained_stpf']['fn_count']}` |"
        )
    lines.extend(
        [
            "",
            "## visualizationdescription",
            "",
            "- description source pair descriptioncurrent case description mesh source, description ABC-Fusion360 or Fusion360-Thingi10K. ",
            "- indescription candidate grid description dense candidate group description, description RTSTPFExact descriptionenter exact certificate descriptioncandidate. ",
            "- description NoProposal and RTSTPFExact  exact calls / exact work. ",
            "",
            "## descriptionusedescription",
            "",
            "- descriptionwithused fordescription: real mesh source dense candidate contact workload in, RTSTPFExact in `FN=0` underdescriptionreduction exact certificate work. ",
            "- descriptionwritedescription: completephysicsdescriptioncollisiondescription. ",
            "- ifdescription SOTA wall-time description, descriptionuse native C++/CUDA/Tight-Inclusion exact payload Path. ",
            "",
        ]
    )
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    (output_dir / "benchmark_report.md").write_text("\n".join(lines), encoding="utf-8")


def write_multi_dense_mesh_contact_pairs_demo(
    config: MultiDenseMeshContactPairsDemoConfig | None = None,
) -> MultiDenseMeshContactPairsDemoResult:
    cfg = config or MultiDenseMeshContactPairsDemoConfig()
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data = _load_json(Path(cfg.benchmark_json))
    mp4_path = output_dir / "multi_dense_mesh_contact_pairs_overview.mp4"
    png_path = output_dir / "multi_dense_mesh_contact_pairs_overview.png"
    html_path = output_dir / "multi_dense_mesh_contact_pairs_interactive.html"
    metrics_path = output_dir / "metrics.json"
    report_path = output_dir / "benchmark_report.md"
    readme_path = output_dir / "README.md"
    _write_mp4_and_png(cfg, data, mp4_path=mp4_path, png_path=png_path)
    _write_interactive_html(html_path, data)
    paths = {
        "mp4": str(mp4_path),
        "overview_png": str(png_path),
        "interactive_html": str(html_path),
        "metrics_json": str(metrics_path),
        "benchmark_report": str(report_path),
        "readme": str(readme_path),
    }
    _write_metrics(metrics_path, cfg, data, paths)
    _write_markdown(output_dir, cfg, data, paths)
    return MultiDenseMeshContactPairsDemoResult(
        output_dir=output_dir,
        mp4_path=mp4_path,
        overview_png_path=png_path,
        interactive_html_path=html_path,
        metrics_json_path=metrics_path,
        benchmark_report_path=report_path,
        readme_path=readme_path,
    )


def main() -> None:
    result = write_multi_dense_mesh_contact_pairs_demo()
    print(json.dumps({key: str(value) for key, value in asdict(result).items()}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()


__all__ = [
    "MultiDenseMeshContactPairsDemoConfig",
    "MultiDenseMeshContactPairsDemoResult",
    "write_multi_dense_mesh_contact_pairs_demo",
]
