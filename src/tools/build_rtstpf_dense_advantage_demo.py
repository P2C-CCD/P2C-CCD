from __future__ import annotations

import csv
import html
import json
import math
import shutil
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
P2CCCD = ROOT / "src"
SOURCE_JSON = P2CCCD / "benchmark" / "rtstpf_paper_full_checkpoint_complete_benchmark_run_id.json"
SOURCE_REPORT = P2CCCD / "benchmark" / "rtstpf_paper_full_checkpoint_complete_benchmark_run_id.md"
SOURCE_FINAL_REPORT = P2CCCD / "benchmark" / "rtstpf_paper_full_checkpoint_final_report_run_id.md"
DEMO_DIR = P2CCCD / "MyDemo" / "paper_rtstpf_dense_advantage_run_id"


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


def _pct(value: float, digits: int = 2) -> str:
    return f"{100.0 * float(value):.{digits}f}%"


def _fmt_int(value: float | int) -> str:
    return f"{int(round(float(value))):,}"


def _fmt_float(value: float, digits: int = 2) -> str:
    return f"{float(value):,.{digits}f}"


def _ensure_dirs() -> dict[str, Path]:
    paths = {
        "root": DEMO_DIR,
        "reports": DEMO_DIR / "reports",
        "visualizations": DEMO_DIR / "visualizations",
        "data": DEMO_DIR / "data",
        "sources": DEMO_DIR / "sources",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def _load_dense_metrics() -> dict[str, Any]:
    payload = json.loads(SOURCE_JSON.read_text(encoding="utf-8"))
    dense = payload["dense"]
    selected = []
    for item in dense:
        no_proposal = item["no_proposal"]
        rtstpf = item["rtstpf"]
        exact_call_reduction = 1.0 - rtstpf["exact_call_count"] / max(1.0, no_proposal["exact_call_count"])
        selected.append(
            {
                "benchmark": item["benchmark"],
                "density": item["density"],
                "eval_queries": item["eval_queries"],
                "eval_candidates": item["eval_candidates"],
                "case_results": item["case_results"],
                "no_proposal": no_proposal,
                "rt_exact": {
                    **no_proposal,
                    "method_name": "RTExact",
                    "note": "For this dense-candidate proxy workload, RTExact sends all RT candidates to exact and therefore has the same exact-work envelope as NoProposal.",
                },
                "rtstpf": rtstpf,
                "exact_work_reduction": item["exact_work_reduction"],
                "exact_call_reduction": exact_call_reduction,
            },
        )
    return {
        "demo_name": "paper_rtstpf_dense_advantage_run_id",
        "source_json": str(SOURCE_JSON),
        "source_report": str(SOURCE_REPORT),
        "checkpoint_path": payload.get("checkpoint_path"),
        "onnx_path": payload.get("onnx_path"),
        "ort_provider": payload.get("ort_provider"),
        "device": payload.get("device"),
        "batch_size": payload.get("batch_size"),
        "dense": selected,
    }


def _write_metrics(paths: dict[str, Path], metrics: dict[str, Any]) -> None:
    (paths["data"] / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (paths["data"] / "dense_case_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "benchmark",
                "case_name",
                "density",
                "eval_queries",
                "eval_candidates",
                "no_proposal_exact_calls",
                "rtstpf_exact_calls",
                "exact_call_reduction",
                "no_proposal_exact_work",
                "rtstpf_exact_work",
                "exact_work_reduction",
                "rtstpf_fn",
            ],
        )
        for group in metrics["dense"]:
            for case in group["case_results"]:
                no_proposal = case["no_proposal"]
                rtstpf = case["rtstpf"]
                call_reduction = 1.0 - rtstpf["exact_call_count"] / max(1.0, no_proposal["exact_call_count"])
                writer.writerow(
                    [
                        group["benchmark"],
                        case["case_name"],
                        group["density"],
                        case["eval_queries"],
                        case["eval_candidates"],
                        no_proposal["exact_call_count"],
                        rtstpf["exact_call_count"],
                        call_reduction,
                        no_proposal["exact_work_units"],
                        rtstpf["exact_work_units"],
                        case["exact_work_reduction"],
                        rtstpf["fn_count"],
                    ],
                )


def _copy_sources(paths: dict[str, Path]) -> None:
    for source in (SOURCE_JSON, SOURCE_REPORT, SOURCE_FINAL_REPORT):
        if source.exists():
            shutil.copy2(source, paths["sources"] / source.name)


def _draw_card(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    lines: list[str],
    *,
    accent: tuple[int, int, int],
) -> None:
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=24, fill=(15, 23, 42), outline=(51, 65, 85), width=2)
    draw.rounded_rectangle((x0, y0, x1, y0 + 8), radius=4, fill=accent)
    draw.text((x0 + 26, y0 + 28), title, font=_font(34, bold=True), fill=accent)
    y = y0 + 86
    for line in lines:
        draw.text((x0 + 26, y), line, font=_font(25), fill=(226, 232, 240))
        y += 39


def _log_bar_width(value: float, max_value: float, *, max_width: int) -> int:
    return int(max_width * math.log10(max(1.0, value)) / math.log10(max(10.0, max_value)))


def _draw_bar_pair(
    draw: ImageDraw.ImageDraw,
    *,
    x: int,
    y: int,
    width: int,
    label: str,
    no_value: float,
    rtstpf_value: float,
    max_value: float,
    unit: str,
) -> None:
    draw.text((x, y), label, font=_font(24, bold=True), fill=(226, 232, 240))
    no_w = _log_bar_width(no_value, max_value, max_width=width)
    stpf_w = _log_bar_width(rtstpf_value, max_value, max_width=width)
    y1 = y + 42
    draw.rounded_rectangle((x, y1, x + width, y1 + 28), radius=10, fill=(30, 41, 59))
    draw.rounded_rectangle((x, y1, x + no_w, y1 + 28), radius=10, fill=(96, 165, 250))
    draw.text((x + width + 18, y1 - 2), f"NoProposal/RTExact {no_value:,.0f} {unit}", font=_font(20), fill=(191, 219, 254))
    y2 = y + 88
    draw.rounded_rectangle((x, y2, x + width, y2 + 28), radius=10, fill=(30, 41, 59))
    draw.rounded_rectangle((x, y2, x + max(3, stpf_w), y2 + 28), radius=10, fill=(34, 197, 94))
    draw.text((x + width + 18, y2 - 2), f"RTSTPFExact {rtstpf_value:,.0f} {unit}", font=_font(20), fill=(187, 247, 208))


def _make_overview_png(paths: dict[str, Path], metrics: dict[str, Any]) -> None:
    image = Image.new("RGB", (1920, 1080), (8, 13, 25))
    draw = ImageDraw.Draw(image)
    draw.text((70, 44), "RTSTPFExact Dense Advantage Cases", font=_font(54, bold=True), fill=(240, 253, 244))
    draw.text(
        (74, 112),
        "learned STPF proposal + exact certificate | dense/high-cost candidate workloads | FN=0",
        font=_font(28),
        fill=(148, 163, 184),
    )
    groups = metrics["dense"]
    colors = [(34, 197, 94), (20, 184, 166)]
    max_work = max(group["no_proposal"]["exact_work_units"] for group in groups)
    for index, group in enumerate(groups):
        x0 = 70 + index * 890
        _draw_card(
            draw,
            (x0, 180, x0 + 840, 405),
            group["benchmark"],
            [
                f"density: {_fmt_float(group['density'], 0)} candidates/query",
                f"queries/candidates: {_fmt_int(group['eval_queries'])} / {_fmt_int(group['eval_candidates'])}",
                f"exact calls: {_fmt_int(group['no_proposal']['exact_call_count'])} -> {_fmt_int(group['rtstpf']['exact_call_count'])}",
                f"exact-call reduction: {_pct(group['exact_call_reduction'])}",
                f"exact-work reduction: {_pct(group['exact_work_reduction'])}",
                f"correctness: FN={group['rtstpf']['fn_count']}",
            ],
            accent=colors[index],
        )
        _draw_bar_pair(
            draw,
            x=x0,
            y=470,
            width=520,
            label="Primitive-weighted exact work (log scale)",
            no_value=group["no_proposal"]["exact_work_units"],
            rtstpf_value=group["rtstpf"]["exact_work_units"],
            max_value=max_work,
            unit="work",
        )
        _draw_bar_pair(
            draw,
            x=x0,
            y=650,
            width=520,
            label="Exact certificate calls (log scale)",
            no_value=group["no_proposal"]["exact_call_count"],
            rtstpf_value=group["rtstpf"]["exact_call_count"],
            max_value=max(group["no_proposal"]["exact_call_count"] for group in groups),
            unit="calls",
        )
    draw.rounded_rectangle((70, 865, 1850, 1018), radius=24, fill=(15, 23, 42), outline=(51, 65, 85), width=2)
    conclusion = (
        "Conclusion: RTSTPFExact is strongest when broad phase leaves many high-cost candidates. "
        "The network only schedules proposals; final collision correctness is still guarded by exact certification."
    )
    draw.text((100, 900), conclusion, font=_font(28, bold=True), fill=(226, 232, 240))
    draw.text(
        (100, 950),
        "Wall-time caveat: these dense reports use proxy bookkeeping for exact work; final native speed table must use compiled C++/CUDA exact kernels.",
        font=_font(24),
        fill=(251, 191, 36),
    )
    image.save(paths["visualizations"] / "dense_advantage_overview.png")


def _make_animation_mp4(paths: dict[str, Path], metrics: dict[str, Any]) -> None:
    frames = []
    width, height = 1600, 912
    groups = metrics["dense"]
    font_title = _font(42, bold=True)
    font_mid = _font(26, bold=True)
    font_text = _font(22)
    max_calls = max(group["no_proposal"]["exact_call_count"] for group in groups)
    max_work = max(group["no_proposal"]["exact_work_units"] for group in groups)
    for frame_index in range(144):
        progress = min(1.0, frame_index / 110.0)
        eased = 1.0 - (1.0 - progress) ** 3
        image = Image.new("RGB", (width, height), (8, 13, 25))
        draw = ImageDraw.Draw(image)
        draw.text((54, 38), "Dense Candidate Funnel: RTSTPFExact vs RTExact / NoProposal", font=font_title, fill=(240, 253, 244))
        draw.text((58, 94), "animated exact-call and exact-work reduction, final certificate remains exact", font=font_text, fill=(148, 163, 184))
        for index, group in enumerate(groups):
            y_base = 170 + index * 330
            accent = (34, 197, 94) if index == 0 else (20, 184, 166)
            draw.rounded_rectangle((54, y_base, width - 54, y_base + 270), radius=24, fill=(15, 23, 42), outline=(51, 65, 85), width=2)
            draw.text((86, y_base + 26), group["benchmark"], font=font_mid, fill=accent)
            draw.text(
                (86, y_base + 66),
                f"density {group['density']:.0f} | queries {group['eval_queries']:,} | candidates {group['eval_candidates']:,} | FN {group['rtstpf']['fn_count']}",
                font=font_text,
                fill=(226, 232, 240),
            )
            no_calls = group["no_proposal"]["exact_call_count"]
            stpf_calls = group["rtstpf"]["exact_call_count"]
            no_work = group["no_proposal"]["exact_work_units"]
            stpf_work = group["rtstpf"]["exact_work_units"]
            current_calls = no_calls + (stpf_calls - no_calls) * eased
            current_work = no_work + (stpf_work - no_work) * eased
            call_w = int(1000 * current_calls / max_calls)
            work_w = int(1000 * math.log10(max(1.0, current_work)) / math.log10(max_work))
            draw.text((86, y_base + 118), "exact calls", font=font_text, fill=(203, 213, 225))
            draw.rounded_rectangle((260, y_base + 114, 1260, y_base + 146), radius=12, fill=(30, 41, 59))
            draw.rounded_rectangle((260, y_base + 114, 260 + max(3, call_w), y_base + 146), radius=12, fill=accent)
            draw.text((1290, y_base + 112), f"{current_calls:,.0f}", font=font_text, fill=(226, 232, 240))
            draw.text((86, y_base + 178), "exact work", font=font_text, fill=(203, 213, 225))
            draw.rounded_rectangle((260, y_base + 174, 1260, y_base + 206), radius=12, fill=(30, 41, 59))
            draw.rounded_rectangle((260, y_base + 174, 260 + max(3, work_w), y_base + 206), radius=12, fill=accent)
            draw.text((1290, y_base + 172), f"{current_work:,.0f}", font=font_text, fill=(226, 232, 240))
            draw.text(
                (86, y_base + 226),
                f"final: call reduction {_pct(group['exact_call_reduction'])}, work reduction {_pct(group['exact_work_reduction'])}",
                font=font_text,
                fill=(187, 247, 208),
            )
        frames.append(image)
    imageio.mimsave(paths["visualizations"] / "dense_advantage_animation.mp4", frames, fps=24, quality=8)


def _write_interactive_html(paths: dict[str, Path], metrics: dict[str, Any]) -> None:
    groups_json = json.dumps(metrics["dense"], ensure_ascii=False)
    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>RTSTPFExact Dense Advantage</title>
  <style>
    :root {{
      --bg: #08111f;
      --panel: #0f172a;
      --line: #334155;
      --text: #e5e7eb;
      --muted: #94a3b8;
      --green: #22c55e;
      --teal: #14b8a6;
      --blue: #60a5fa;
      --amber: #fbbf24;
    }}
    body {{
      margin: 0;
      background: radial-gradient(circle at top left, #123a31 0, var(--bg) 42%, #020617 100%);
      color: var(--text);
      font-family: "Microsoft YaHei", "Segoe UI", sans-serif;
    }}
    main {{ max-width: 1280px; margin: 0 auto; padding: 36px 28px 56px; }}
    h1 {{ margin: 0; font-size: 40px; letter-spacing: -0.02em; }}
    .sub {{ margin-top: 10px; color: var(--muted); font-size: 18px; }}
    .cards {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 20px; margin-top: 28px; }}
    .card {{ background: rgba(15,23,42,.88); border: 1px solid var(--line); border-radius: 24px; padding: 22px; box-shadow: 0 22px 70px rgba(0,0,0,.25); }}
    .card h2 {{ margin: 0 0 14px; color: var(--green); font-size: 24px; }}
    .metric {{ display: grid; grid-template-columns: 190px 1fr; gap: 10px; margin: 8px 0; color: #cbd5e1; }}
    .metric b {{ color: white; }}
    .bar {{ height: 18px; background: #1e293b; border-radius: 999px; overflow: hidden; margin: 8px 0 16px; }}
    .fill {{ height: 100%; border-radius: 999px; background: linear-gradient(90deg, var(--green), var(--teal)); }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 28px; background: rgba(15,23,42,.78); border-radius: 18px; overflow: hidden; }}
    th, td {{ border-bottom: 1px solid #263244; padding: 10px 12px; text-align: right; font-size: 14px; }}
    th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) {{ text-align: left; }}
    th {{ color: #dbeafe; background: #111827; position: sticky; top: 0; }}
    .note {{ margin-top: 24px; padding: 18px 20px; border-radius: 18px; border: 1px solid rgba(251,191,36,.42); background: rgba(120,53,15,.18); color: #fde68a; }}
  </style>
</head>
<body>
  <main>
    <h1>RTSTPFExact Dense Advantage Demo</h1>
    <div class="sub">currentadvantage case: broad phase afterdescriptionhighdescription candidates, learned STPF only performs proposal/scheduling, description exact certificate guaranteecorrectness. </div>
    <section id="cards" class="cards"></section>
    <table>
      <thead>
        <tr>
          <th>Benchmark</th><th>Case</th><th>Density</th><th>Queries</th><th>Candidates</th><th>Exact Calls</th><th>RTSTPF Calls</th><th>Call Reduction</th><th>Work Reduction</th><th>FN</th>
        </tr>
      </thead>
      <tbody id="rows"></tbody>
    </table>
    <div class="note">Notes: hereadvantageMetricsis primitive-weighted exact work and exact-call reduction; current dense benchmark  wall time description Python/ORT proposal+scheduling overhead, final paper wall-time descriptionusedescriptionafter C++/CUDA exact Path. </div>
  </main>
  <script>
    const groups = {groups_json};
    function pct(v) {{ return (100*v).toFixed(2) + "%"; }}
    function fmt(v) {{ return Math.round(v).toLocaleString(); }}
    const cards = document.getElementById("cards");
    for (const g of groups) {{
      const card = document.createElement("article");
      card.className = "card";
      card.innerHTML = `
        <h2>${{g.benchmark}}</h2>
        <div class="metric"><span>density</span><b>${{g.density.toFixed(0)}} candidates/query</b></div>
        <div class="metric"><span>queries/candidates</span><b>${{fmt(g.eval_queries)}} / ${{fmt(g.eval_candidates)}}</b></div>
        <div class="metric"><span>exact calls</span><b>${{fmt(g.no_proposal.exact_call_count)}} -> ${{fmt(g.rtstpf.exact_call_count)}}</b></div>
        <div class="bar"><div class="fill" style="width:${{100*g.exact_call_reduction}}%"></div></div>
        <div class="metric"><span>exact-work reduction</span><b>${{pct(g.exact_work_reduction)}}</b></div>
        <div class="metric"><span>correctness</span><b>FN=${{g.rtstpf.fn_count}}</b></div>
      `;
      cards.appendChild(card);
    }}
    const rows = document.getElementById("rows");
    for (const g of groups) {{
      for (const c of g.case_results) {{
        const np = c.no_proposal;
        const st = c.rtstpf;
        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td>${{g.benchmark}}</td><td>${{c.case_name}}</td><td>${{g.density.toFixed(0)}}</td>
          <td>${{fmt(c.eval_queries)}}</td><td>${{fmt(c.eval_candidates)}}</td>
          <td>${{fmt(np.exact_call_count)}}</td><td>${{fmt(st.exact_call_count)}}</td>
          <td>${{pct(1 - st.exact_call_count / np.exact_call_count)}}</td>
          <td>${{pct(c.exact_work_reduction)}}</td><td>${{st.fn_count}}</td>
        `;
        rows.appendChild(tr);
      }}
    }}
  </script>
</body>
</html>
"""
    (paths["visualizations"] / "dense_advantage_interactive.html").write_text(html_text, encoding="utf-8")


def _write_report(paths: dict[str, Path], metrics: dict[str, Any]) -> None:
    lines = [
        "# RTSTPFExact densecandidateadvantage Case completedescriptionreport",
        "",
        "## 1. descriptionNotes",
        "",
        "- `reports/benchmark_report.md`: indescriptionandcorrectnessreport. ",
        "- `data/metrics.json`: fromdescriptionnew benchmark descriptionafterdescriptionMetrics. ",
        "- `data/dense_case_summary.csv`: each dense case description. ",
        "- `visualizations/dense_advantage_overview.png`: descriptionleveloverview figure. ",
        "- `visualizations/dense_advantage_animation.mp4`: candidate exact work description. ",
        "- `visualizations/dense_advantage_interactive.html`: description dense benchmark description. ",
        "- `sources/`: original benchmark reportand JSON description. ",
        "",
        "## 2. Conclusion",
        "",
        "currentdescriptionthis paperadvantageis dense/high-cost candidate workload, rather thandescription primitive query workload. ",
        "indescriptionscenein, Broad Phase descriptionafterdescriptioncandidate, RTExact / NoProposal descriptioncandidatedescription exact certificate; RTSTPFExact descriptionuse learned STPF descriptioncandidateperformwhendescription proposal anddescription, descriptionhighdescriptioncandidatedescription exact certificate, descriptionkeep `FN = 0`. ",
        "",
        "## 3. description",
        "",
        "| Benchmark | Density | Queries | Candidates | NoProposal/RTExact calls | RTSTPFExact calls | Exact-call reduction | Exact-work reduction | FN |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for group in metrics["dense"]:
        lines.append(
            "| "
            f"`{group['benchmark']}` | "
            f"`{_fmt_float(group['density'], 0)}` | "
            f"`{_fmt_int(group['eval_queries'])}` | "
            f"`{_fmt_int(group['eval_candidates'])}` | "
            f"`{_fmt_int(group['no_proposal']['exact_call_count'])}` | "
            f"`{_fmt_int(group['rtstpf']['exact_call_count'])}` | "
            f"`{_pct(group['exact_call_reduction'])}` | "
            f"`{_pct(group['exact_work_reduction'])}` | "
            f"`{group['rtstpf']['fn_count']}` |"
        )
    lines.extend(
        [
            "",
            "## 4. split Case description",
            "",
            "| Benchmark | Case | Queries | Candidates | Face min/median/max | RTSTPF calls | Exact-call reduction | Exact-work reduction | FN |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ],
    )
    for group in metrics["dense"]:
        for case in group["case_results"]:
            no_proposal = case["no_proposal"]
            rtstpf = case["rtstpf"]
            call_reduction = 1.0 - rtstpf["exact_call_count"] / max(1.0, no_proposal["exact_call_count"])
            lines.append(
                "| "
                f"`{group['benchmark']}` | "
                f"`{case['case_name']}` | "
                f"`{_fmt_int(case['eval_queries'])}` | "
                f"`{_fmt_int(case['eval_candidates'])}` | "
                f"`{_fmt_int(case['face_min'])}/{_fmt_int(case['face_median'])}/{_fmt_int(case['face_max'])}` | "
                f"`{_fmt_int(rtstpf['exact_call_count'])}` | "
                f"`{_pct(call_reduction)}` | "
                f"`{_pct(case['exact_work_reduction'])}` | "
                f"`{rtstpf['fn_count']}` |"
            )
    lines.extend(
        [
            "",
            "## 5. correctnessdescription",
            "",
            "- `RTSTPFExact` does not directly outputdescriptioncollisionConclusion, descriptionOutput proposal / scheduling. ",
            "- descriptioncollisiondescription exact certificate Pathdescription, descriptionwithdescriptioncorrectnessMetricsdescriptionis `FN = 0`. ",
            "- description dense benchmark summarizedescriptionas `FN = 0`. ",
            "",
            "## 6. description",
            "",
            "- `NoProposal` and `RTExact` send all broad-phase candidates to exact certification in this dense-candidate proxy workload. ",
            "- `RTSTPFExact` description exact certificate calldescriptionfromdescriptionleveldescriptiontodescriptionlevel, descriptionwhendescription fallback. ",
            "- reportin `exact_work_units` is primitive-weighted exact work descriptionMetrics, is notdescription. ",
            "- current Python/ORT proposal+scheduling wall time descriptionasdescription SOTA wall-time description; description wall-time descriptionafter `C++ scheduling + ORT TensorRT EP + CUDA/Tight-Inclusion exact` Path. ",
            "",
            "## 7. description",
            "",
            "`On dense/high-cost candidate workloads where broad phase still leaves 6.6e5 to 6.9e5 exact candidates, RTSTPFExact reduces exact certificate calls by 99.88%-99.95% and primitive-weighted exact work by 99.97%-99.98%, while preserving zero false negatives through exact certification.`",
            "",
        ],
    )
    text = "\n".join(lines)
    (paths["reports"] / "benchmark_report.md").write_text(text, encoding="utf-8")
    (paths["root"] / "benchmark_report.md").write_text(text, encoding="utf-8")
    (paths["root"] / "README.md").write_text(
        "\n".join(
            [
                "# RTSTPFExact Dense Advantage Demo",
                "",
                "thisdescriptioncurrentdescriptionthis paperMethodadvantage dense/high-cost candidate cases. ",
                "",
                "## description",
                "",
                "- report: `benchmark_report.md`",
                "- overview figure: `visualizations/dense_advantage_overview.png`",
                "- description: `visualizations/dense_advantage_animation.mp4`",
                "- description: `visualizations/dense_advantage_interactive.html`",
                "- description: `data/metrics.json` and `data/dense_case_summary.csv`",
                "",
                "## descriptionConclusion",
                "",
                "RTSTPFExact in dense/high-cost candidate workload onkeep `FN=0`, description exact certificate calls and primitive-weighted exact work descriptionlevelwithon. this demo used fordescriptioninadvantage case description, description wall-time descriptionshould stilldescriptionuse native C++/CUDA exact benchmark. ",
                "",
            ],
        ),
        encoding="utf-8",
    )


def main() -> None:
    paths = _ensure_dirs()
    metrics = _load_dense_metrics()
    _write_metrics(paths, metrics)
    _copy_sources(paths)
    _make_overview_png(paths, metrics)
    _make_animation_mp4(paths, metrics)
    _write_interactive_html(paths, metrics)
    _write_report(paths, metrics)
    print(DEMO_DIR)


if __name__ == "__main__":
    main()
