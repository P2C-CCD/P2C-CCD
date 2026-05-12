from __future__ import annotations

from html import escape
import json
from pathlib import Path
from typing import Any

from p2cccd.bench.bvh_exact import _try_load_p2cccd_cpp
from p2cccd.data.response import (
    build_elastic_impact_response,
    proxy_mass_from_radius,
    replay_positions_at_time,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[5]


def _resolve_repo_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (_repo_root() / path).resolve()


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _cpp_mesh_module() -> Any:
    cpp = _try_load_p2cccd_cpp()
    required = ("load_triangle_mesh", "validate_triangle_mesh", "center_mesh_at_aabb_center")
    if cpp is None or any(not hasattr(cpp, name) for name in required):
        raise RuntimeError("p2cccd_cpp mesh bindings are unavailable for visualization")
    return cpp


def _downsample_edges(edges: list[tuple[int, int]], max_edge_count: int) -> list[tuple[int, int]]:
    if len(edges) <= max_edge_count:
        return edges
    stride = max(1, len(edges) // max_edge_count)
    return edges[::stride][:max_edge_count]


def _load_projected_mesh(path: str | Path, *, max_edge_count: int = 1800) -> dict[str, Any]:
    cpp = _cpp_mesh_module()
    mesh = cpp.load_triangle_mesh(str(path))
    cpp.validate_triangle_mesh(mesh)
    centered_mesh, _ = cpp.center_mesh_at_aabb_center(mesh)
    vertices = [tuple(float(value) for value in vertex[:2]) for vertex in centered_mesh.vertices_ref]
    bounding_radius = max(
        (
            (float(vertex[0]) * float(vertex[0]) + float(vertex[1]) * float(vertex[1]) + float(vertex[2]) * float(vertex[2]))
            ** 0.5
            for vertex in centered_mesh.vertices_ref
        ),
        default=1.0e-3,
    )
    edges: set[tuple[int, int]] = set()
    for triangle in centered_mesh.triangles:
        i0, i1, i2 = (int(triangle[0]), int(triangle[1]), int(triangle[2]))
        for a, b in ((i0, i1), (i1, i2), (i2, i0)):
            edge = (a, b) if a < b else (b, a)
            edges.add(edge)
    sorted_edges = _downsample_edges(sorted(edges), max_edge_count)
    return {
        "vertices": vertices,
        "edges": sorted_edges,
        "bounding_radius": float(max(1.0e-6, bounding_radius)),
    }


def _lerp3(a: tuple[float, float, float], b: tuple[float, float, float], t: float) -> tuple[float, float, float]:
    return (
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    )


def _response_payload(
    selected_query: dict[str, Any],
    selected_result: dict[str, Any],
    mesh_a: dict[str, Any],
    mesh_b: dict[str, Any],
) -> dict[str, Any]:
    response = build_elastic_impact_response(
        center_a_t0=tuple(float(v) for v in selected_query["translation_a_t0"]),
        center_a_t1=tuple(float(v) for v in selected_query["translation_a_t1"]),
        center_b_t0=tuple(float(v) for v in selected_query["translation_b_t0"]),
        center_b_t1=tuple(float(v) for v in selected_query["translation_b_t1"]),
        toi=float(selected_result["toi_upper"]),
        collided=bool(selected_result["predicted_collision"]) and str(selected_result["status"]) == "collision",
        mass_a=proxy_mass_from_radius(float(mesh_a["bounding_radius"])),
        mass_b=proxy_mass_from_radius(float(mesh_b["bounding_radius"])),
        restitution=1.0,
    )
    return {
        "collided": response.collided,
        "toi": response.toi,
        "mass_a": response.mass_a,
        "mass_b": response.mass_b,
        "restitution": response.restitution,
        "normal": list(response.normal),
        "center_a_toi": list(response.center_a_toi),
        "center_b_toi": list(response.center_b_toi),
        "velocity_a_pre": list(response.velocity_a_pre),
        "velocity_b_pre": list(response.velocity_b_pre),
        "velocity_a_post": list(response.velocity_a_post),
        "velocity_b_post": list(response.velocity_b_post),
    }


def _animation_payload(ground_truth_json_path: str | Path, *, query_id: int | None = None) -> dict[str, Any]:
    summary = _load_json(ground_truth_json_path)
    queries_path = _resolve_repo_path(summary["dataset"]["queries_jsonl_path"])
    queries = [
        json.loads(line)
        for line in queries_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    results = {int(row["query_id"]): row for row in summary["query_results"]}
    selected_query: dict[str, Any] | None = None
    selected_result: dict[str, Any] | None = None
    if query_id is not None:
        for query in queries:
            if int(query["query_id"]) == int(query_id):
                selected_query = query
                selected_result = results[int(query_id)]
                break
    if selected_query is None:
        ordered = sorted(results.values(), key=lambda row: (row["status"] != "collision", row["query_id"]))
        selected_result = ordered[0]
        selected_query = next(query for query in queries if int(query["query_id"]) == int(selected_result["query_id"]))

    mesh_a = _load_projected_mesh(selected_query["mesh_a_path"])
    mesh_b = _load_projected_mesh(selected_query["mesh_b_path"])
    ta0 = tuple(float(v) for v in selected_query["translation_a_t0"])
    ta1 = tuple(float(v) for v in selected_query["translation_a_t1"])
    tb0 = tuple(float(v) for v in selected_query["translation_b_t0"])
    tb1 = tuple(float(v) for v in selected_query["translation_b_t1"])
    response = _response_payload(selected_query, selected_result, mesh_a, mesh_b)
    bounce_a_t1, bounce_b_t1 = replay_positions_at_time(
        build_elastic_impact_response(
            center_a_t0=ta0,
            center_a_t1=ta1,
            center_b_t0=tb0,
            center_b_t1=tb1,
            toi=float(response["toi"]),
            collided=bool(response["collided"]),
            mass_a=float(response["mass_a"]),
            mass_b=float(response["mass_b"]),
            restitution=float(response["restitution"]),
        ),
        1.0,
        mode="bounce",
    )

    min_x = min(
        min(vertex[0] + t[0] for vertex in mesh_a["vertices"])
        for t in (ta0, ta1, bounce_a_t1)
    )
    max_x = max(
        max(vertex[0] + t[0] for vertex in mesh_a["vertices"])
        for t in (ta0, ta1, bounce_a_t1)
    )
    min_y = min(
        min(vertex[1] + t[1] for vertex in mesh_a["vertices"])
        for t in (ta0, ta1, bounce_a_t1)
    )
    max_y = max(
        max(vertex[1] + t[1] for vertex in mesh_a["vertices"])
        for t in (ta0, ta1, bounce_a_t1)
    )
    min_x = min(min_x, *(min(vertex[0] + t[0] for vertex in mesh_b["vertices"]) for t in (tb0, tb1, bounce_b_t1)))
    max_x = max(max_x, *(max(vertex[0] + t[0] for vertex in mesh_b["vertices"]) for t in (tb0, tb1, bounce_b_t1)))
    min_y = min(min_y, *(min(vertex[1] + t[1] for vertex in mesh_b["vertices"]) for t in (tb0, tb1, bounce_b_t1)))
    max_y = max(max_y, *(max(vertex[1] + t[1] for vertex in mesh_b["vertices"]) for t in (tb0, tb1, bounce_b_t1)))
    pad_x = max(1.0e-3, (max_x - min_x) * 0.1)
    pad_y = max(1.0e-3, (max_y - min_y) * 0.1)

    return {
        "title": f"{Path(ground_truth_json_path).stem} mesh animation",
        "query": selected_query,
        "result": selected_result,
        "mesh_a": mesh_a,
        "mesh_b": mesh_b,
        "response": response,
        "bounds": {
            "min_x": min_x - pad_x,
            "max_x": max_x + pad_x,
            "min_y": min_y - pad_y,
            "max_y": max_y + pad_y,
        },
    }


def write_abc_mesh_exact_comparison_html(summary_json_path: str | Path, output_path: str | Path) -> Path:
    summary = _load_json(summary_json_path)
    methods = summary["method_summaries"]
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "\n".join(
            (
                "<!doctype html>",
                "<html><head><meta charset=\"utf-8\">",
                f"<title>{escape(Path(summary_json_path).stem)} overview</title>",
                "<style>body{font-family:Segoe UI,Arial,sans-serif;margin:20px;background:#0f172a;color:#e2e8f0}"
                "main{display:grid;grid-template-columns:420px 1fr;gap:20px}section{background:#111827;border:1px solid #334155;border-radius:14px;padding:16px}"
                "table{width:100%;border-collapse:collapse;font-size:13px}th,td{padding:8px;border-bottom:1px solid #334155;text-align:right}th:first-child,td:first-child{text-align:left}"
                ".bar{height:20px;border-radius:8px}.small{color:#94a3b8;font-size:12px}</style></head><body>",
                f"<h1>{escape(Path(summary_json_path).stem)} overview</h1>",
                "<main>",
                "<section>",
                f"<p><strong>dataset</strong><br><code>{escape(summary['dataset']['source_root'])}</code></p>",
                f"<p><strong>queries</strong>: {int(summary['dataset']['query_count'])}</p>",
                f"<p><strong>pairs</strong>: {int(summary['dataset']['pair_count'])}</p>",
                "<p class=\"small\">Metrics are from real mesh-mesh exact benchmark description JSON. descriptionsplitdescriptionas RT / proposal / exact. </p>",
                "</section>",
                "<section>",
                "<table><thead><tr><th>Method</th><th>FN</th><th>Recall</th><th>rt ms</th><th>proposal ms</th><th>exact ms</th><th>total ms</th><th>qps</th></tr></thead><tbody>",
                *[
                    "<tr>"
                    f"<td>{escape(row['method'])}</td>"
                    f"<td>{int(row['fn_count'])}</td>"
                    f"<td>{float(row['candidate_recall']):.4f}</td>"
                    f"<td>{float(row['rt_ms']):.4f}</td>"
                    f"<td>{float(row['proposal_ms']):.4f}</td>"
                    f"<td>{float(row['exact_ms']):.4f}</td>"
                    f"<td>{float(row['total_ms']):.4f}</td>"
                    f"<td>{float(row['qps']):.2f}</td>"
                    "</tr>"
                    for row in methods
                ],
                "</tbody></table><div id=\"bars\" style=\"margin-top:18px\"></div></section></main>",
                f"<script>const methods={json.dumps(methods, ensure_ascii=False)};"
                "const bars=document.getElementById('bars');"
                "const maxTotal=Math.max(...methods.map(r=>r.total_ms),1e-6);"
                "for(const row of methods){"
                "const wrap=document.createElement('div');wrap.style.margin='12px 0';"
                "const label=document.createElement('div');label.textContent=`${row.method}  total=${row.total_ms.toFixed(4)} ms`;label.style.marginBottom='6px';wrap.appendChild(label);"
                "const track=document.createElement('div');track.style.background='#1f2937';track.style.border='1px solid #374151';track.style.borderRadius='10px';track.style.height='22px';track.style.display='flex';track.style.overflow='hidden';"
                "for(const seg of [{k:'rt_ms',c:'#2563eb'},{k:'proposal_ms',c:'#f59e0b'},{k:'exact_ms',c:'#dc2626'}]){const v=Math.max(0,row[seg.k]);const part=document.createElement('div');part.className='bar';part.style.background=seg.c;part.style.width=`${100*v/maxTotal}%`;track.appendChild(part);}"
                "wrap.appendChild(track);bars.appendChild(wrap);}"
                "</script></body></html>",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    return output


def write_abc_mesh_exact_animation_html(
    ground_truth_json_path: str | Path,
    output_path: str | Path,
    *,
    query_id: int | None = None,
) -> Path:
    data = _animation_payload(ground_truth_json_path, query_id=query_id)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "\n".join(
            (
                "<!doctype html>",
                "<html><head><meta charset='utf-8'>",
                f"<title>{escape(data['title'])}</title>",
                "<style>body{font-family:Segoe UI,Arial,sans-serif;margin:20px;background:#0f172a;color:#e2e8f0}"
                "main{display:grid;grid-template-columns:980px 320px;gap:20px}canvas{background:#f8fafc;border-radius:12px;box-shadow:0 4px 16px rgba(0,0,0,0.25)}"
                ".card{background:#111827;border:1px solid #334155;border-radius:12px;padding:14px}.small{color:#94a3b8;font-size:12px}button,input[type=range]{width:100%}select{width:100%;padding:8px;border-radius:8px;background:#0f172a;color:#e2e8f0;border:1px solid #334155}</style></head><body>",
                f"<h1>{escape(data['title'])}</h1><main><div><canvas id='view' width='960' height='640'></canvas>"
                "<div style='margin-top:12px;display:grid;grid-template-columns:140px 1fr 220px;gap:12px;'><button id='play'>Play / Pause</button><input id='slider' type='range' min='0' max='120' step='1' value='0'/><select id='mode'><option value='bounce'>BounceReplay</option><option value='stop_at_toi'>StopAtTOI</option><option value='raw'>RawMotion</option></select></div></div>",
                "<div class='card'><div id='info'></div><p class='small'>default mode is `BounceReplay`. descriptionindescription TOI descriptionbydescription, descriptionmomentumdescription, coefficient of restitution 1.0 descriptioncollisionperform post-impact replay; `RawMotion` descriptionoriginal benchmark Inputtrajectory. </p></div></main>",
                f"<script>const data={json.dumps(data, ensure_ascii=False)};"
                "const canvas=document.getElementById('view');const ctx=canvas.getContext('2d');const slider=document.getElementById('slider');const play=document.getElementById('play');const info=document.getElementById('info');const modeSelect=document.getElementById('mode');"
                "let running=false;let frame=0;let mode=data.response.collided?'bounce':'raw';modeSelect.value=mode;"
                "function add(a,b){return [a[0]+b[0],a[1]+b[1],a[2]+b[2]];} function scale(a,s){return [a[0]*s,a[1]*s,a[2]*s];} function lerp(a,b,t){return [a[0]+(b[0]-a[0])*t,a[1]+(b[1]-a[1])*t,a[2]+(b[2]-a[2])*t];}"
                "function mapX(x){const b=data.bounds;return 60+(x-b.min_x)/(b.max_x-b.min_x)*(canvas.width-120);}"
                "function mapY(y){const b=data.bounds;return canvas.height-120-(y-b.min_y)/(b.max_y-b.min_y)*(canvas.height-220);}"
                "function drawMesh(mesh,center,stroke){ctx.strokeStyle=stroke;ctx.lineWidth=1.2;ctx.beginPath();for(const edge of mesh.edges){const a=mesh.vertices[edge[0]],b=mesh.vertices[edge[1]];ctx.moveTo(mapX(a[0]+center[0]),mapY(a[1]+center[1]));ctx.lineTo(mapX(b[0]+center[0]),mapY(b[1]+center[1]));}ctx.stroke();}"
                "function drawTimeline(t){const x0=80,x1=canvas.width-80,y=canvas.height-60;ctx.strokeStyle='#334155';ctx.lineWidth=8;ctx.beginPath();ctx.moveTo(x0,y);ctx.lineTo(x1,y);ctx.stroke();"
                "const toi=data.result.toi_upper;const xt=x0+(x1-x0)*t;const xToi=x0+(x1-x0)*toi;ctx.strokeStyle='#16a34a';ctx.lineWidth=4;ctx.beginPath();ctx.moveTo(xToi,y-18);ctx.lineTo(xToi,y+18);ctx.stroke();"
                "ctx.strokeStyle='#f59e0b';ctx.beginPath();ctx.moveTo(xt,y-18);ctx.lineTo(xt,y+18);ctx.stroke();ctx.fillStyle='#334155';ctx.font='14px Segoe UI';ctx.fillText('t=0',x0-10,y+34);ctx.fillText('t=1',x1-10,y+34);ctx.fillStyle='#16a34a';ctx.fillText('TOI upper',xToi-18,y-22);}"
                "function pairCenters(t){if(mode==='raw'){return [lerp(data.query.translation_a_t0,data.query.translation_a_t1,t),lerp(data.query.translation_b_t0,data.query.translation_b_t1,t)];}"
                "const toi=Math.max(0,Math.min(1,data.response.toi)); if(mode==='stop_at_toi' || !data.response.collided){const s=Math.min(t,toi);return [lerp(data.query.translation_a_t0,data.query.translation_a_t1,s),lerp(data.query.translation_b_t0,data.query.translation_b_t1,s)];}"
                "if(t<=toi){return [lerp(data.query.translation_a_t0,data.query.translation_a_t1,t),lerp(data.query.translation_b_t0,data.query.translation_b_t1,t)];}"
                "const dt=t-toi; return [add(data.response.center_a_toi,scale(data.response.velocity_a_post,dt)),add(data.response.center_b_toi,scale(data.response.velocity_b_post,dt))];}"
                "function render(idx){const t=idx/120.0;const pair=pairCenters(t);const ca=pair[0];const cb=pair[1];"
                "ctx.clearRect(0,0,canvas.width,canvas.height);ctx.fillStyle='#f8fafc';ctx.fillRect(0,0,canvas.width,canvas.height);ctx.strokeStyle='#cbd5e1';ctx.strokeRect(60,40,canvas.width-120,canvas.height-180);drawMesh(data.mesh_a,ca,'rgba(37,99,235,0.9)');drawMesh(data.mesh_b,cb,'rgba(220,38,38,0.9)');drawTimeline(t);"
                "info.innerHTML=`<p><strong>pair_id</strong><br><code>${data.query.pair_id}</code></p><p><strong>query_id</strong>: ${data.query.query_id}</p><p><strong>status</strong>: ${data.result.status}</p><p><strong>predicted collision</strong>: ${data.result.predicted_collision}</p><p><strong>TOI upper</strong>: ${data.result.toi_upper.toFixed(4)}</p><p><strong>safe margin lb</strong>: ${data.result.safe_margin_lb.toFixed(4)}</p><p><strong>PT kept / total</strong>: ${data.result.point_triangle_kept_pairs} / ${data.result.point_triangle_total_pairs}</p><p><strong>EE kept / total</strong>: ${data.result.edge_edge_kept_pairs} / ${data.result.edge_edge_total_pairs}</p><p><strong>mass a / b</strong>: ${data.response.mass_a.toFixed(4)} / ${data.response.mass_b.toFixed(4)}</p><p><strong>restitution</strong>: ${data.response.restitution.toFixed(2)}</p><p><strong>normal</strong>: [${data.response.normal.map(v=>v.toFixed(3)).join(', ')}]</p><p><strong>mode</strong>: ${mode==='bounce'?'BounceReplay':(mode==='stop_at_toi'?'StopAtTOI':'RawMotion')}</p>`;}"
                "play.onclick=()=>{running=!running;}; slider.oninput=()=>{frame=Number(slider.value);render(frame);}; modeSelect.onchange=()=>{mode=modeSelect.value;render(frame);}; window.addEventListener('keydown',e=>{if(e.key==='m'){const order=['bounce','stop_at_toi','raw'];const next=(order.indexOf(mode)+1)%order.length;mode=order[next];modeSelect.value=mode;render(frame);}});"
                "function tick(){if(running){frame=(frame+1)%121;slider.value=String(frame);render(frame);}requestAnimationFrame(tick);}render(0);tick();</script></body></html>",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    return output


def write_abc_mesh_exact_visual_bundle(summary_json_path: str | Path) -> tuple[Path, Path]:
    summary_path = Path(summary_json_path)
    stem = summary_path.stem
    overview_path = summary_path.with_name(f"{stem}_overview.html")
    write_abc_mesh_exact_comparison_html(summary_path, overview_path)
    summary = _load_json(summary_path)
    ground_truth_json = _resolve_repo_path(summary["ground_truth_report"]).with_suffix(".json")
    animation_path = summary_path.with_name(f"{stem}_animation.html")
    write_abc_mesh_exact_animation_html(ground_truth_json, animation_path)
    return overview_path, animation_path


__all__ = [
    "write_abc_mesh_exact_animation_html",
    "write_abc_mesh_exact_comparison_html",
    "write_abc_mesh_exact_visual_bundle",
]
