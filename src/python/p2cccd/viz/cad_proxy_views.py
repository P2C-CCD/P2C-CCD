from __future__ import annotations

from html import escape
import json
from pathlib import Path
from typing import Any

from p2cccd.data.oracle import evaluate_swept_sphere_oracle
from p2cccd.data.response import build_sample_elastic_impact_response
from p2cccd.datasets.cad.abc_adapter import ABCDatasetAdapter
from p2cccd.datasets.cad.abc_training import _samples_from_pairs


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[5]


def _cpp_mesh_module() -> Any:
    from p2cccd.bench.bvh_exact import _try_load_p2cccd_cpp

    cpp = _try_load_p2cccd_cpp()
    required = ("load_triangle_mesh", "validate_triangle_mesh", "center_mesh_at_aabb_center")
    if cpp is None or any(not hasattr(cpp, name) for name in required):
        raise RuntimeError("p2cccd_cpp mesh bindings are unavailable for CAD visualization")
    return cpp


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_projected_mesh(path: str | Path, *, max_edge_count: int = 1800) -> dict[str, Any]:
    cpp = _cpp_mesh_module()
    mesh = cpp.load_triangle_mesh(str(path))
    cpp.validate_triangle_mesh(mesh)
    centered_mesh, _ = cpp.center_mesh_at_aabb_center(mesh)
    vertices = [tuple(float(value) for value in vertex[:2]) for vertex in centered_mesh.vertices_ref]
    edges: set[tuple[int, int]] = set()
    for triangle in centered_mesh.triangles:
        i0, i1, i2 = (int(triangle[0]), int(triangle[1]), int(triangle[2]))
        for a, b in ((i0, i1), (i1, i2), (i2, i0)):
            edges.add((a, b) if a < b else (b, a))
    sorted_edges = sorted(edges)
    if len(sorted_edges) > max_edge_count:
        stride = max(1, len(sorted_edges) // max_edge_count)
        sorted_edges = sorted_edges[::stride][:max_edge_count]
    return {
        "vertices": vertices,
        "edges": sorted_edges,
    }


def _select_pair_sample(
    dataset_manifest_path: str | Path,
    *,
    pair_id: str,
    split: str,
) -> tuple[dict[str, Any], Any]:
    manifest = _load_json(dataset_manifest_path)
    adapter = ABCDatasetAdapter(Path(manifest["source_root"]))
    assets = tuple(adapter.load_asset(path) for path in manifest["asset_paths"])
    pairs = {pair.pair_id: pair for pair in adapter.generate_mesh_pairs(assets=assets, limit=None)}
    if pair_id not in pairs:
        raise KeyError(f"pair_id {pair_id} is not part of benchmark manifest")
    pair = pairs[pair_id]
    sample = next((candidate for candidate in _samples_from_pairs((pair,), first_sample_id=1) if candidate.split == split), None)
    if sample is None:
        raise KeyError(f"split {split} is not available for pair {pair_id}")
    return manifest, pair, sample


def write_abc_cad_collision_animation_html(
    dataset_manifest_path: str | Path,
    output_path: str | Path,
    *,
    pair_id: str,
    split: str,
) -> Path:
    manifest, pair, sample = _select_pair_sample(dataset_manifest_path, pair_id=pair_id, split=split)
    trace = evaluate_swept_sphere_oracle(sample)
    response = build_sample_elastic_impact_response(
        sample,
        toi=trace.toi,
        collided=trace.collided,
    )
    mesh_a = _load_projected_mesh(pair.asset_a.asset_path)
    mesh_b = _load_projected_mesh(pair.asset_b.asset_path)
    positions = (
        sample.center_a_t0,
        sample.center_a_t1,
        sample.center_b_t0,
        sample.center_b_t1,
        response.center_a_toi,
        response.center_b_toi,
    )
    min_x = min(
        *(min(vertex[0] + pos[0] for vertex in mesh_a["vertices"]) for pos in (sample.center_a_t0, sample.center_a_t1, response.center_a_toi)),
        *(min(vertex[0] + pos[0] for vertex in mesh_b["vertices"]) for pos in (sample.center_b_t0, sample.center_b_t1, response.center_b_toi)),
        min(pos[0] - radius for pos, radius in ((sample.center_a_t0, sample.radius_a), (sample.center_a_t1, sample.radius_a), (sample.center_b_t0, sample.radius_b), (sample.center_b_t1, sample.radius_b))),
    )
    max_x = max(
        *(max(vertex[0] + pos[0] for vertex in mesh_a["vertices"]) for pos in (sample.center_a_t0, sample.center_a_t1, response.center_a_toi)),
        *(max(vertex[0] + pos[0] for vertex in mesh_b["vertices"]) for pos in (sample.center_b_t0, sample.center_b_t1, response.center_b_toi)),
        max(pos[0] + radius for pos, radius in ((sample.center_a_t0, sample.radius_a), (sample.center_a_t1, sample.radius_a), (sample.center_b_t0, sample.radius_b), (sample.center_b_t1, sample.radius_b))),
    )
    min_y = min(
        *(min(vertex[1] + pos[1] for vertex in mesh_a["vertices"]) for pos in (sample.center_a_t0, sample.center_a_t1, response.center_a_toi)),
        *(min(vertex[1] + pos[1] for vertex in mesh_b["vertices"]) for pos in (sample.center_b_t0, sample.center_b_t1, response.center_b_toi)),
        min(pos[1] - radius for pos, radius in ((sample.center_a_t0, sample.radius_a), (sample.center_a_t1, sample.radius_a), (sample.center_b_t0, sample.radius_b), (sample.center_b_t1, sample.radius_b))),
    )
    max_y = max(
        *(max(vertex[1] + pos[1] for vertex in mesh_a["vertices"]) for pos in (sample.center_a_t0, sample.center_a_t1, response.center_a_toi)),
        *(max(vertex[1] + pos[1] for vertex in mesh_b["vertices"]) for pos in (sample.center_b_t0, sample.center_b_t1, response.center_b_toi)),
        max(pos[1] + radius for pos, radius in ((sample.center_a_t0, sample.radius_a), (sample.center_a_t1, sample.radius_a), (sample.center_b_t0, sample.radius_b), (sample.center_b_t1, sample.radius_b))),
    )
    pad_x = max(1.0e-3, (max_x - min_x) * 0.12)
    pad_y = max(1.0e-3, (max_y - min_y) * 0.12)
    payload = {
        "title": f"{Path(output_path).stem} proxy collision animation",
        "manifest": {
            "run_name": manifest["run_name"],
            "source_root": manifest["source_root"],
        },
        "pair_id": pair_id,
        "split": split,
        "mesh_a": mesh_a,
        "mesh_b": mesh_b,
        "radius_a": sample.radius_a,
        "radius_b": sample.radius_b,
        "center_a_t0": list(sample.center_a_t0),
        "center_a_t1": list(sample.center_a_t1),
        "center_b_t0": list(sample.center_b_t0),
        "center_b_t1": list(sample.center_b_t1),
        "toi": trace.toi,
        "contact_interval": [trace.contact_interval_t0, trace.contact_interval_t1],
        "safe_margin": trace.safe_margin,
        "response": {
            "collided": response.collided,
            "toi": response.toi,
            "mass_a": response.mass_a,
            "mass_b": response.mass_b,
            "restitution": response.restitution,
            "normal": list(response.normal),
            "center_a_toi": list(response.center_a_toi),
            "center_b_toi": list(response.center_b_toi),
            "velocity_a_post": list(response.velocity_a_post),
            "velocity_b_post": list(response.velocity_b_post),
        },
        "bounds": {
            "min_x": min_x - pad_x,
            "max_x": max_x + pad_x,
            "min_y": min_y - pad_y,
            "max_y": max_y + pad_y,
        },
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "\n".join(
            (
                "<!doctype html>",
                "<html><head><meta charset='utf-8'>",
                f"<title>{escape(payload['title'])}</title>",
                "<style>body{font-family:Segoe UI,Arial,sans-serif;margin:20px;background:#0f172a;color:#e2e8f0}"
                "main{display:grid;grid-template-columns:980px 320px;gap:20px}canvas{background:#f8fafc;border-radius:12px;box-shadow:0 4px 16px rgba(0,0,0,0.25)}"
                ".card{background:#111827;border:1px solid #334155;border-radius:12px;padding:14px}.small{color:#94a3b8;font-size:12px}button,input[type=range],select{width:100%}select{padding:8px;border-radius:8px;background:#0f172a;color:#e2e8f0;border:1px solid #334155}</style></head><body>",
                f"<h1>{escape(payload['title'])}</h1><main><div><canvas id='view' width='960' height='640'></canvas>",
                "<div style='margin-top:12px;display:grid;grid-template-columns:140px 1fr 220px;gap:12px;'><button id='play'>Play / Pause</button><input id='slider' type='range' min='0' max='120' step='1' value='0'/><select id='mode'><option value='bounce'>BounceReplay</option><option value='stop_at_toi'>StopAtTOI</option><option value='raw'>RawMotion</option></select></div></div>",
                "<div class='card'><div id='info'></div><p class='small'>this is CAD proxy benchmark description. description swept-sphere proxy, description centered CAD mesh  XY description, defaultbymomentumdescriptionandcoefficient of restitution 1.0 performdescription replay. </p></div></main>",
                f"<script>const data={json.dumps(payload, ensure_ascii=False)};"
                "const canvas=document.getElementById('view');const ctx=canvas.getContext('2d');const slider=document.getElementById('slider');const play=document.getElementById('play');const info=document.getElementById('info');const modeSelect=document.getElementById('mode');"
                "let running=false;let frame=0;let mode=data.response.collided?'bounce':'raw';modeSelect.value=mode;"
                "function add(a,b){return [a[0]+b[0],a[1]+b[1],a[2]+b[2]];} function scale(a,s){return [a[0]*s,a[1]*s,a[2]*s];} function lerp(a,b,t){return [a[0]+(b[0]-a[0])*t,a[1]+(b[1]-a[1])*t,a[2]+(b[2]-a[2])*t];}"
                "function mapX(x){const b=data.bounds;return 60+(x-b.min_x)/(b.max_x-b.min_x)*(canvas.width-120);} function mapY(y){const b=data.bounds;return canvas.height-120-(y-b.min_y)/(b.max_y-b.min_y)*(canvas.height-220);}"
                "function drawMesh(mesh,center,stroke){ctx.strokeStyle=stroke;ctx.lineWidth=1.2;ctx.beginPath();for(const edge of mesh.edges){const a=mesh.vertices[edge[0]],b=mesh.vertices[edge[1]];ctx.moveTo(mapX(a[0]+center[0]),mapY(a[1]+center[1]));ctx.lineTo(mapX(b[0]+center[0]),mapY(b[1]+center[1]));}ctx.stroke();}"
                "function drawProxy(center,radius,fill,stroke){ctx.fillStyle=fill;ctx.strokeStyle=stroke;ctx.lineWidth=1.8;ctx.beginPath();ctx.arc(mapX(center[0]),mapY(center[1]),Math.max(4.0,radius*(canvas.width-120)/(data.bounds.max_x-data.bounds.min_x)),0,Math.PI*2);ctx.fill();ctx.stroke();}"
                "function centersAt(t){if(mode==='raw'){return [lerp(data.center_a_t0,data.center_a_t1,t),lerp(data.center_b_t0,data.center_b_t1,t)];}const toi=Math.max(0,Math.min(1,data.response.toi));if(mode==='stop_at_toi' || !data.response.collided){const s=Math.min(t,toi);return [lerp(data.center_a_t0,data.center_a_t1,s),lerp(data.center_b_t0,data.center_b_t1,s)];}if(t<=toi){return [lerp(data.center_a_t0,data.center_a_t1,t),lerp(data.center_b_t0,data.center_b_t1,t)];}const dt=t-toi;return [add(data.response.center_a_toi,scale(data.response.velocity_a_post,dt)),add(data.response.center_b_toi,scale(data.response.velocity_b_post,dt))];}"
                "function drawTimeline(t){const x0=80,x1=canvas.width-80,y=canvas.height-60;ctx.strokeStyle='#334155';ctx.lineWidth=8;ctx.beginPath();ctx.moveTo(x0,y);ctx.lineTo(x1,y);ctx.stroke();const toi=data.toi;const interval0=x0+(x1-x0)*data.contact_interval[0];const interval1=x0+(x1-x0)*data.contact_interval[1];ctx.strokeStyle='#22c55e';ctx.lineWidth=6;ctx.beginPath();ctx.moveTo(interval0,y);ctx.lineTo(interval1,y);ctx.stroke();const xt=x0+(x1-x0)*t;const xToi=x0+(x1-x0)*toi;ctx.strokeStyle='#16a34a';ctx.lineWidth=4;ctx.beginPath();ctx.moveTo(xToi,y-18);ctx.lineTo(xToi,y+18);ctx.stroke();ctx.strokeStyle='#f59e0b';ctx.beginPath();ctx.moveTo(xt,y-18);ctx.lineTo(xt,y+18);ctx.stroke();ctx.fillStyle='#334155';ctx.font='14px Segoe UI';ctx.fillText('t=0',x0-10,y+34);ctx.fillText('t=1',x1-10,y+34);}"
                "function render(idx){const t=idx/120.0;const pair=centersAt(t);const ca=pair[0];const cb=pair[1];ctx.clearRect(0,0,canvas.width,canvas.height);ctx.fillStyle='#f8fafc';ctx.fillRect(0,0,canvas.width,canvas.height);ctx.strokeStyle='#cbd5e1';ctx.strokeRect(60,40,canvas.width-120,canvas.height-180);drawProxy(ca,data.radius_a,'rgba(96,165,250,0.18)','rgba(59,130,246,0.85)');drawProxy(cb,data.radius_b,'rgba(248,113,113,0.18)','rgba(220,38,38,0.85)');drawMesh(data.mesh_a,ca,'rgba(37,99,235,0.9)');drawMesh(data.mesh_b,cb,'rgba(220,38,38,0.9)');drawTimeline(t);info.innerHTML=`<p><strong>run</strong><br><code>${data.manifest.run_name}</code></p><p><strong>pair_id</strong><br><code>${data.pair_id}</code></p><p><strong>split</strong>: ${data.split}</p><p><strong>TOI</strong>: ${data.toi.toFixed(4)}</p><p><strong>contact interval</strong>: [${data.contact_interval[0].toFixed(4)}, ${data.contact_interval[1].toFixed(4)}]</p><p><strong>safe margin</strong>: ${data.safe_margin.toFixed(4)}</p><p><strong>mass a / b</strong>: ${data.response.mass_a.toFixed(4)} / ${data.response.mass_b.toFixed(4)}</p><p><strong>mode</strong>: ${mode==='bounce'?'BounceReplay':(mode==='stop_at_toi'?'StopAtTOI':'RawMotion')}</p>`;}"
                "play.onclick=()=>{running=!running;}; slider.oninput=()=>{frame=Number(slider.value);render(frame);}; modeSelect.onchange=()=>{mode=modeSelect.value;render(frame);}; window.addEventListener('keydown',e=>{if(e.key==='m'){const order=['bounce','stop_at_toi','raw'];const next=(order.indexOf(mode)+1)%order.length;mode=order[next];modeSelect.value=mode;render(frame);}});"
                "function tick(){if(running){frame=(frame+1)%121;slider.value=String(frame);render(frame);}requestAnimationFrame(tick);}render(0);tick();</script></body></html>",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    return output


__all__ = ["write_abc_cad_collision_animation_html"]
