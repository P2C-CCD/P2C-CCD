from __future__ import annotations

from dataclasses import asdict
import json
import math
from pathlib import Path
import shutil
from typing import Any

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import sys


ROOT = Path(__file__).resolve().parents[2]
P2CCCD = ROOT / "src"
sys.path.insert(0, str(P2CCCD / "python"))

from p2cccd.bench.high_density_mesh_training_benchmark import (  # noqa: E402
    _dataset_from_samples,
    _load_abc_assets,
    _load_fusion360_assets,
    _load_thingi10k_assets,
    _scale_workload_costs,
)
from p2cccd.bench.large_dense_complex_mesh_cases import (  # noqa: E402
    LargeDenseComplexMeshCasesConfig,
    _make_heavy_cross_pairs,
    _make_heavy_intra_pairs,
    _rename_assets,
    _split_pairs,
)
from p2cccd.bench.trained_stpf_high_density import (  # noqa: E402
    HighDensityMethodMetrics,
    benchmark_no_proposal_on_high_density_workload,
    benchmark_stpf_on_high_density_workload,
    build_high_density_stpf_workload,
)
from p2cccd.data.oracle import evaluate_swept_sphere_oracle  # noqa: E402
from p2cccd.contracts import ProxyType  # noqa: E402
from p2cccd.data.response import proxy_mass_from_radius  # noqa: E402
from p2cccd.data.response import build_sample_elastic_impact_response, replay_positions_at_time  # noqa: E402
from p2cccd.data.samplers import MotionDiscPairSample, PairFamily  # noqa: E402
from p2cccd.viz.high_density_collision_mp4 import (  # noqa: E402
    _RenderMesh,
    _load_model,
    _load_render_mesh,
    _pyvista_polydata,
    _rt_exact_metrics_from_no_proposal,
    _selected_candidate_ids_for_stpf,
)


DEMO_ROOT = P2CCCD / "MyDemo"
RUN_ROOT = DEMO_ROOT / "paper_large_dense_complex_mesh_cases_run_id"
CHECKPOINT = P2CCCD / "outputs" / "stpf_training" / "rtstpf_paper_dataset_v2_paper_full_run_id" / "model_state.pt"
SOURCE_JSON = P2CCCD / "benchmark" / "rtstpf_paper_full_checkpoint_complete_benchmark_run_id.json"
SOURCE_MD = P2CCCD / "benchmark" / "rtstpf_paper_full_checkpoint_complete_benchmark_run_id.md"
CASE_ORDER = (
    "L1-ABC-megaface-intra",
    "L2-ABCmegaface-Fusiondense-cross",
    "L3-ABCmegaface-Thingi10Kdirty-cross",
)


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


def _safe_name(name: str) -> str:
    return name.lower().replace("+", "plus").replace("/", "_").replace(" ", "_")


def _pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def _fmt(value: float | int, digits: int = 1) -> str:
    if isinstance(value, int):
        return f"{value:,}"
    return f"{value:,.{digits}f}"


def _agg_queries(aggregate: dict[str, Any]) -> int:
    return int(aggregate.get("eval_query_count", aggregate.get("eval_queries", 0)))


def _agg_candidates(aggregate: dict[str, Any]) -> int:
    return int(aggregate.get("eval_candidate_count", aggregate.get("eval_candidates", 0)))


def _load_case_aggregate_metrics() -> dict[str, dict[str, Any]]:
    payload = json.loads(SOURCE_JSON.read_text(encoding="utf-8"))
    case_metrics = {}
    for group in payload["dense"]:
        if group["benchmark"] != "large_dense_complex_mesh_cases":
            continue
        for item in group["case_results"]:
            case_metrics[item["case_name"]] = item
    return case_metrics


def _reconstruct_eval_pairs() -> dict[str, tuple[Any, ...]]:
    cfg = LargeDenseComplexMeshCasesConfig()
    abc = _rename_assets(_load_abc_assets(Path(cfg.abc_root), cfg.asset_limit_per_source), "ABC megaface")
    fusion = _rename_assets(_load_fusion360_assets(Path(cfg.fusion360_root), cfg.asset_limit_per_source), "Fusion 360 dense")
    thingi = tuple(
        sorted(
            _rename_assets(_load_thingi10k_assets(Path(cfg.thingi10k_root), cfg.asset_limit_per_source), "Thingi10K dirty"),
            key=lambda asset: (-asset.dirty_score, -asset.face_count, asset.asset_id),
        ),
    )
    pairs_by_case = {
        "L1-ABC-megaface-intra": _make_heavy_intra_pairs(
            "L1-ABC-megaface-intra",
            abc,
            limit=cfg.pair_limit_per_case,
        ),
        "L2-ABCmegaface-Fusiondense-cross": _make_heavy_cross_pairs(
            "L2-ABCmegaface-Fusiondense-cross",
            abc,
            fusion,
            limit=cfg.pair_limit_per_case,
        ),
        "L3-ABCmegaface-Thingi10Kdirty-cross": _make_heavy_cross_pairs(
            "L3-ABCmegaface-Thingi10Kdirty-cross",
            abc,
            thingi,
            limit=cfg.pair_limit_per_case,
        ),
    }
    eval_pairs_by_case = {}
    for offset, (case_name, pairs) in enumerate(pairs_by_case.items()):
        _, eval_pairs = _split_pairs(pairs, train_fraction=cfg.train_fraction, seed=cfg.seed + offset)
        eval_pairs_by_case[case_name] = eval_pairs
    return eval_pairs_by_case


def _build_true_surface_contact_sample(
    *,
    mesh_a: _RenderMesh,
    mesh_b: _RenderMesh,
    pair: Any,
    sample_id: int,
) -> tuple[Any, Any]:
    a_min = mesh_a.vertices.min(axis=0)
    a_max = mesh_a.vertices.max(axis=0)
    b_min = mesh_b.vertices.min(axis=0)
    contact_center_x = float(a_max[0] - b_min[0])
    object_width = max(1.0e-6, contact_center_x)
    radius = 0.5 * object_width
    delta = 1.15 * object_width
    half_delta = 0.5 * delta
    center_a_t0 = (-half_delta, 0.0, 0.0)
    center_a_t1 = (half_delta, 0.0, 0.0)
    center_b_t0 = (contact_center_x + half_delta, 0.0, 0.0)
    center_b_t1 = (contact_center_x - half_delta, 0.0, 0.0)
    mass_a = proxy_mass_from_radius(radius)
    mass_b = proxy_mass_from_radius(radius)
    sample = MotionDiscPairSample(
        sample_id=sample_id,
        query_id=sample_id + 8_700_000,
        candidate_id=sample_id + 8_800_000,
        split="true_large_dense_mesh_surface_contact",
        family=PairFamily.MESH_PAIR,
        object_a_id=700_000 + (abs(hash(pair.asset_a.asset_id)) % 200_000),
        patch_a_id=1,
        object_b_id=700_000 + (abs(hash(pair.asset_b.asset_id)) % 200_000),
        patch_b_id=1,
        slab_id=8,
        center_a_t0=center_a_t0,
        center_a_t1=center_a_t1,
        center_b_t0=center_b_t0,
        center_b_t1=center_b_t1,
        radius_a=radius,
        radius_b=radius,
        proxy_type_a=ProxyType.SWEPT_AABB,
        proxy_type_b=ProxyType.SWEPT_AABB,
        hardness=1.0,
        ood=False,
        mass_a=mass_a,
        mass_b=mass_b,
        restitution=1.0,
    )
    trace = evaluate_swept_sphere_oracle(sample)
    if not trace.collided:
        raise RuntimeError(f"constructed true surface contact sample did not collide for {pair.source_name}")
    return sample, trace


def _build_single_workload(pair: Any, sample: Any, cfg: LargeDenseComplexMeshCasesConfig):
    dataset = _dataset_from_samples([sample])
    workload = build_high_density_stpf_workload(dataset, cfg.high_density, name=f"{pair.source_name}_demo")
    return _scale_workload_costs(workload, {sample.query_id: pair.cost_scale})


def _mesh_points_at(mesh: _RenderMesh, center: tuple[float, float, float], scale: float) -> np.ndarray:
    return (mesh.vertices + np.asarray(center, dtype=np.float32)) * np.float32(scale)


def _camera_from_points(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    target = 0.5 * (points.min(axis=0) + points.max(axis=0))
    diagonal = max(1.0e-6, float(np.linalg.norm(points.max(axis=0) - points.min(axis=0))))
    camera = target + np.asarray([0.92 * diagonal, -1.18 * diagonal, 0.76 * diagonal], dtype=np.float32)
    return target, camera, diagonal


def _contact_camera(
    *,
    mesh_a: _RenderMesh,
    mesh_b: _RenderMesh,
    response: Any,
    trace: Any,
    scene_scale: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    center_a, center_b = replay_positions_at_time(response, float(trace.toi), mode="bounce")
    vertices_a = mesh_a.vertices + np.asarray(center_a, dtype=np.float32)
    vertices_b = mesh_b.vertices + np.asarray(center_b, dtype=np.float32)
    point_a = vertices_a[int(np.argmax(vertices_a[:, 0]))]
    point_b = vertices_b[int(np.argmin(vertices_b[:, 0]))]
    target = 0.5 * (point_a + point_b) * np.float32(scene_scale)
    diag_a = float(np.linalg.norm(mesh_a.vertices.max(axis=0) - mesh_a.vertices.min(axis=0)))
    diag_b = float(np.linalg.norm(mesh_b.vertices.max(axis=0) - mesh_b.vertices.min(axis=0)))
    local_scale = max(0.60 * min(diag_a, diag_b), 0.14 * max(diag_a, diag_b), 1.0e-4) * scene_scale
    camera = target + np.asarray([1.15 * local_scale, -1.34 * local_scale, 0.82 * local_scale], dtype=np.float32)
    return target, camera, local_scale


def _draw_method_card(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    name: str,
    subtitle: str,
    metric: HighDensityMethodMetrics,
    baseline_work: float,
    *,
    accent: tuple[int, int, int],
) -> None:
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=18, fill=(15, 23, 42, 228), outline=(51, 65, 85, 255), width=2)
    draw.text((x0 + 22, y0 + 18), name, font=_font(29, bold=True), fill=accent + (255,))
    draw.text((x0 + 22, y0 + 55), subtitle, font=_font(17), fill=(226, 232, 240, 255))
    reduction = 1.0 - metric.exact_work_units / max(1.0e-9, baseline_work)
    lines = [
        f"exact calls {metric.exact_call_count:,}/{metric.candidate_count:,}",
        f"exact work {metric.exact_work_units:,.1f}",
        f"reduction {_pct(max(0.0, reduction))}",
        f"FN {metric.fn_count}",
    ]
    for index, line in enumerate(lines):
        draw.text(
            (x0 + 22 + (index % 2) * 275, y0 + 90 + (index // 2) * 28),
            line,
            font=_font(17),
            fill=(241, 245, 249, 255),
        )
    bar_x0, bar_y0, bar_x1 = x0 + 22, y1 - 20, x1 - 22
    draw.rounded_rectangle((bar_x0, bar_y0, bar_x1, bar_y0 + 10), radius=5, fill=(30, 41, 59, 255))
    frac = min(1.0, metric.exact_call_count / max(1.0, metric.candidate_count))
    draw.rounded_rectangle(
        (bar_x0, bar_y0, bar_x0 + max(3, int((bar_x1 - bar_x0) * frac)), bar_y0 + 10),
        radius=5,
        fill=accent + (255,),
    )


def _draw_overlay(
    image: Image.Image,
    *,
    case_name: str,
    sample: Any,
    trace: Any,
    t: float,
    progress: float,
    view_label: str,
    view_note: str,
    no_proposal: HighDensityMethodMetrics,
    rt_exact: HighDensityMethodMetrics,
    rtstpf: HighDensityMethodMetrics,
    aggregate: dict[str, Any],
) -> None:
    draw = ImageDraw.Draw(image, "RGBA")
    draw.rounded_rectangle((28, 24, image.width - 28, 170), radius=22, fill=(8, 13, 25, 210))
    draw.text((54, 44), case_name, font=_font(38, bold=True), fill=(240, 253, 244, 255))
    draw.text(
        (56, 94),
        (
            f"true mesh dense contact | t={t:.3f} | TOI={float(trace.toi):.4f} | "
            f"{view_label} | density={_agg_candidates(aggregate) // max(1, _agg_queries(aggregate))} candidates/query"
        ),
        font=_font(22),
        fill=(203, 213, 225, 255),
    )
    draw.text(
        (56, 127),
        (
            f"aggregate: {_fmt(_agg_queries(aggregate), 0)} queries, "
            f"{_fmt(_agg_candidates(aggregate), 0)} candidates, "
            f"work reduction {_pct(aggregate['exact_work_reduction'])}, FN={aggregate['rtstpf']['fn_count']}"
        ),
        font=_font(20),
        fill=(187, 247, 208, 255),
    )
    draw.text((image.width - 560, 127), view_note, font=_font(18), fill=(226, 232, 240, 230))
    panel_w = (image.width - 80) // 3
    y0 = image.height - 180
    _draw_method_card(
        draw,
        (28, y0, 28 + panel_w - 12, image.height - 24),
        "RTSTPFExact",
        "learned proposal + exact certificate",
        rtstpf,
        no_proposal.exact_work_units,
        accent=(34, 197, 94),
    )
    _draw_method_card(
        draw,
        (40 + panel_w, y0, 40 + 2 * panel_w - 12, image.height - 24),
        "RTExact",
        "RT candidates direct to exact",
        rt_exact,
        no_proposal.exact_work_units,
        accent=(96, 165, 250),
    )
    _draw_method_card(
        draw,
        (52 + 2 * panel_w, y0, 52 + 3 * panel_w - 12, image.height - 24),
        "NoProposal",
        "fallback exact queue",
        no_proposal,
        no_proposal.exact_work_units,
        accent=(248, 113, 113),
    )
    progress_x = int(54 + (image.width - 108) * progress)
    draw.rectangle((54, image.height - 214, image.width - 54, image.height - 202), fill=(30, 41, 59, 255))
    draw.rectangle((54, image.height - 214, progress_x, image.height - 202), fill=(34, 211, 238, 255))


def _render_case_video(
    *,
    case_dir: Path,
    case_name: str,
    pair: Any,
    sample: Any,
    trace: Any,
    no_proposal: HighDensityMethodMetrics,
    rt_exact: HighDensityMethodMetrics,
    rtstpf: HighDensityMethodMetrics,
    aggregate: dict[str, Any],
    frame_count: int,
    fps: int,
    max_faces: int,
    scene_scale: float,
    view_mode: str,
) -> dict[str, str]:
    import pyvista as pv

    if view_mode not in {"global", "local_zoom"}:
        raise ValueError(f"unsupported view_mode: {view_mode}")
    vis_dir = case_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    mesh_a = _load_render_mesh(pair.asset_a.asset_path, max_faces=max_faces, seed=sample.sample_id + 11)
    mesh_b = _load_render_mesh(pair.asset_b.asset_path, max_faces=max_faces, seed=sample.sample_id + 12)
    response = build_sample_elastic_impact_response(sample, toi=trace.toi, collided=trace.collided)

    is_global = view_mode == "global"
    start_t = 0.0 if is_global else max(0.0, float(trace.toi) - 0.24)
    end_t = 1.0 if is_global else min(1.0, float(trace.toi) + 0.24)
    initial_a, initial_b = replay_positions_at_time(response, start_t, mode="bounce")
    poly_a = _pyvista_polydata(mesh_a)
    poly_b = _pyvista_polydata(mesh_b)
    poly_a.points = _mesh_points_at(mesh_a, initial_a, scene_scale)
    poly_b.points = _mesh_points_at(mesh_b, initial_b, scene_scale)
    if is_global:
        all_points = []
        for sample_t in (0.0, float(trace.toi), 1.0):
            ca, cb = replay_positions_at_time(response, sample_t, mode="bounce")
            all_points.append(_mesh_points_at(mesh_a, ca, scene_scale))
            all_points.append(_mesh_points_at(mesh_b, cb, scene_scale))
        target, camera, global_diagonal = _camera_from_points(np.vstack(all_points))
        parallel_scale = max(0.01, 0.40 * global_diagonal)
        view_label = "GLOBAL"
        view_note = "global approach/contact/separation replay"
    else:
        target, camera, local_scale = _contact_camera(
            mesh_a=mesh_a,
            mesh_b=mesh_b,
            response=response,
            trace=trace,
            scene_scale=scene_scale,
        )
        parallel_scale = max(0.01, 1.00 * local_scale)
        view_label = "LOCAL ZOOM"
        view_note = "TOI-local contact-window zoom"

    width, height = 1920, 1088
    render_h = height - 230
    pv.OFF_SCREEN = True
    plotter = pv.Plotter(off_screen=True, window_size=(width, render_h))
    plotter.set_background("#414b59")
    plotter.enable_anti_aliasing("ssaa")
    plotter.add_mesh(poly_a, color="#4aa3ff", smooth_shading=True, specular=0.36, roughness=0.56, show_edges=False)
    plotter.add_mesh(poly_b, color="#ff6868", smooth_shading=True, specular=0.36, roughness=0.56, show_edges=False)
    plotter.add_light(pv.Light(position=tuple(camera), focal_point=tuple(target), color="white", intensity=0.85))
    plotter.camera_position = (tuple(camera.tolist()), tuple(target.tolist()), (0.0, 0.0, 1.0))
    plotter.camera.parallel_projection = True
    plotter.camera.parallel_scale = parallel_scale
    plotter.camera.clipping_range = (0.01, 10000.0)

    mp4_path = vis_dir / f"{_safe_name(case_name)}_{view_mode}.mp4"
    preview_path = vis_dir / f"{_safe_name(case_name)}_{view_mode}_preview.png"
    before_path = vis_dir / f"{_safe_name(case_name)}_{view_mode}_before.png"
    toi_path = vis_dir / f"{_safe_name(case_name)}_{view_mode}_toi.png"
    after_path = vis_dir / f"{_safe_name(case_name)}_{view_mode}_after.png"
    writer = imageio.get_writer(str(mp4_path), fps=fps, codec="libx264", quality=8, macro_block_size=16)
    frame_paths = {}
    try:
        for frame_index in range(frame_count):
            progress = frame_index / max(1, frame_count - 1)
            t_value = start_t + (end_t - start_t) * progress
            ca, cb = replay_positions_at_time(response, float(t_value), mode="bounce")
            poly_a.points = _mesh_points_at(mesh_a, ca, scene_scale)
            poly_b.points = _mesh_points_at(mesh_b, cb, scene_scale)
            plotter.render()
            rgb = np.asarray(plotter.screenshot(return_img=True)[:, :, :3], dtype=np.uint8)
            frame = Image.new("RGB", (width, height), (8, 13, 25))
            frame.paste(Image.fromarray(rgb, mode="RGB"), (0, 170))
            _draw_overlay(
                frame,
                case_name=case_name,
                sample=sample,
                trace=trace,
                t=float(t_value),
                progress=float(progress),
                view_label=view_label,
                view_note=view_note,
                no_proposal=no_proposal,
                rt_exact=rt_exact,
                rtstpf=rtstpf,
                aggregate=aggregate,
            )
            if frame_index == 0:
                frame.save(before_path)
                frame_paths["before"] = str(before_path)
            toi_progress = 0.5 if not is_global else float(trace.toi)
            if frame_index == int(round(toi_progress * (frame_count - 1))):
                frame.save(toi_path)
                frame.save(preview_path)
                frame_paths["toi"] = str(toi_path)
                frame_paths["preview"] = str(preview_path)
            if frame_index == frame_count - 1:
                frame.save(after_path)
                frame_paths["after"] = str(after_path)
            writer.append_data(np.asarray(frame))
    finally:
        writer.close()
        plotter.close()

    return {
        "mp4": str(mp4_path),
        "preview": str(preview_path),
        "before": str(before_path),
        "toi": str(toi_path),
        "after": str(after_path),
    }


def _write_interactive_html(
    *,
    case_dir: Path,
    case_name: str,
    pair: Any,
    sample: Any,
    trace: Any,
    max_faces: int,
    scene_scale: float,
) -> str:
    import plotly.graph_objects as go

    mesh_a = _load_render_mesh(pair.asset_a.asset_path, max_faces=max_faces, seed=sample.sample_id + 101)
    mesh_b = _load_render_mesh(pair.asset_b.asset_path, max_faces=max_faces, seed=sample.sample_id + 102)
    response = build_sample_elastic_impact_response(sample, toi=trace.toi, collided=trace.collided)
    times = np.linspace(max(0.0, float(trace.toi) - 0.22), min(1.0, float(trace.toi) + 0.22), 9)
    toi_index = int(np.argmin(np.abs(times - float(trace.toi))))
    times[toi_index] = float(trace.toi)

    def verts(t_value: float) -> tuple[np.ndarray, np.ndarray]:
        ca, cb = replay_positions_at_time(response, float(t_value), mode="bounce")
        return _mesh_points_at(mesh_a, ca, scene_scale), _mesh_points_at(mesh_b, cb, scene_scale)

    va, vb = verts(float(times[toi_index]))
    target, camera, local_scale = _contact_camera(
        mesh_a=mesh_a,
        mesh_b=mesh_b,
        response=response,
        trace=trace,
        scene_scale=scene_scale,
    )
    eye = camera - target
    eye_unit = eye / max(1.0e-6, float(np.linalg.norm(eye)))
    data = [
        go.Mesh3d(
            x=va[:, 0],
            y=va[:, 1],
            z=va[:, 2],
            i=mesh_a.faces[:, 0],
            j=mesh_a.faces[:, 1],
            k=mesh_a.faces[:, 2],
            name="blue mesh A",
            color="#46a3ff",
            opacity=0.68,
            flatshading=False,
            hoverinfo="skip",
        ),
        go.Mesh3d(
            x=vb[:, 0],
            y=vb[:, 1],
            z=vb[:, 2],
            i=mesh_b.faces[:, 0],
            j=mesh_b.faces[:, 1],
            k=mesh_b.faces[:, 2],
            name="red mesh B",
            color="#ff6868",
            opacity=0.68,
            flatshading=False,
            hoverinfo="skip",
        ),
        go.Scatter3d(
            x=[target[0]],
            y=[target[1]],
            z=[target[2]],
            mode="markers+text",
            marker={"size": 7, "color": "#facc15"},
            text=["TOI"],
            textposition="top center",
            hoverinfo="skip",
        ),
    ]
    frames = []
    for index, t_value in enumerate(times):
        fa, fb = verts(float(t_value))
        frames.append(
            go.Frame(
                name=f"{index:02d} | t={float(t_value):.4f}",
                traces=[0, 1],
                data=[go.Mesh3d(x=fa[:, 0], y=fa[:, 1], z=fa[:, 2]), go.Mesh3d(x=fb[:, 0], y=fb[:, 1], z=fb[:, 2])],
            )
        )
    fig = go.Figure(data=data, frames=frames)
    fig.update_layout(
        title=f"{case_name} | interactive dense mesh contact | TOI={float(trace.toi):.6f}",
        paper_bgcolor="#0b1120",
        plot_bgcolor="#0b1120",
        font={"color": "#e5e7eb"},
        margin={"l": 0, "r": 0, "t": 58, "b": 0},
        scene={
            "bgcolor": "#202936",
            "aspectmode": "cube",
            "camera": {
                "eye": {"x": float(2.2 * eye_unit[0]), "y": float(2.2 * eye_unit[1]), "z": float(2.2 * eye_unit[2])}
            },
            "xaxis": {"visible": False},
            "yaxis": {"visible": False},
            "zaxis": {"visible": False},
        },
        sliders=[
            {
                "active": toi_index,
                "steps": [
                    {
                        "label": f"t={float(t):.3f}" + (" TOI" if i == toi_index else ""),
                        "method": "animate",
                        "args": [[frames[i].name], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}}],
                    }
                    for i, t in enumerate(times)
                ],
            }
        ],
        updatemenus=[
            {
                "type": "buttons",
                "buttons": [
                    {"label": "Play", "method": "animate", "args": [None, {"frame": {"duration": 220, "redraw": True}}]},
                    {"label": "Pause", "method": "animate", "args": [[None], {"mode": "immediate"}]},
                ],
            }
        ],
    )
    html_path = case_dir / "visualizations" / f"{_safe_name(case_name)}_interactive.html"
    fig.write_html(str(html_path), include_plotlyjs=True, full_html=True, auto_play=False)
    return str(html_path)


def _write_case_report(
    *,
    case_dir: Path,
    case_name: str,
    pair: Any,
    sample: Any,
    trace: Any,
    no_proposal: HighDensityMethodMetrics,
    rt_exact: HighDensityMethodMetrics,
    rtstpf: HighDensityMethodMetrics,
    aggregate: dict[str, Any],
    assets: dict[str, str],
) -> None:
    call_reduction = 1.0 - rtstpf.exact_call_count / max(1.0, no_proposal.exact_call_count)
    work_reduction = 1.0 - rtstpf.exact_work_units / max(1.0e-9, no_proposal.exact_work_units)
    aggregate_call_reduction = 1.0 - aggregate["rtstpf"]["exact_call_count"] / max(1.0, aggregate["no_proposal"]["exact_call_count"])
    lines = [
        f"# {case_name} complete visualization and performance report",
        "",
        "## File",
        "",
        f"- Global MP4: `{assets['global_mp4']}`",
        f"- Local zoom MP4: `{assets['local_zoom_mp4']}`",
        f"- Interactive HTML: `{assets['interactive_html']}`",
        f"- Global preview PNG: `{assets['global_preview']}`",
        f"- Local zoom preview PNG: `{assets['local_zoom_preview']}`",
        f"- Global before / TOI / after PNG: `{assets['global_before']}`, `{assets['global_toi']}`, `{assets['global_after']}`",
        f"- Local zoom before / TOI / after PNG: `{assets['local_zoom_before']}`, `{assets['local_zoom_toi']}`, `{assets['local_zoom_after']}`",
        f"- Metrics JSON: `{case_dir / 'metrics.json'}`",
        "",
        "## Case",
        "",
        f"- Case: `{case_name}`",
        f"- Asset A: `{pair.asset_a.asset_id}`",
        f"- Asset B: `{pair.asset_b.asset_id}`",
        f"- Asset A faces: `{pair.asset_a.face_count:,}`",
        f"- Asset B faces: `{pair.asset_b.face_count:,}`",
        f"- Pair cost scale: `{pair.cost_scale:.3f}`",
        f"- Query id: `{sample.query_id}`",
        f"- Split tag: `{sample.split}`",
        f"- TOI: `{float(trace.toi):.6f}`",
        f"- Contact interval: `[{float(trace.contact_interval_t0):.6f}, {float(trace.contact_interval_t1):.6f}]`",
        "",
        "## description Query visualizationMetrics",
        "",
        "| Method | Candidates | Exact calls | Exact work | Call reduction | Work reduction | FN |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| `RTSTPFExact` | `{rtstpf.candidate_count:,}` | `{rtstpf.exact_call_count:,}` | `{rtstpf.exact_work_units:,.1f}` | `{_pct(call_reduction)}` | `{_pct(work_reduction)}` | `{rtstpf.fn_count}` |",
        f"| `RTExact` | `{rt_exact.candidate_count:,}` | `{rt_exact.exact_call_count:,}` | `{rt_exact.exact_work_units:,.1f}` | `0.00%` | `0.00%` | `{rt_exact.fn_count}` |",
        f"| `NoProposal` | `{no_proposal.candidate_count:,}` | `{no_proposal.exact_call_count:,}` | `{no_proposal.exact_work_units:,.1f}` | `0.00%` | `0.00%` | `{no_proposal.fn_count}` |",
        "",
        "## Aggregate Benchmark Metrics",
        "",
        "| Scope | Queries | Candidates | NoProposal/RTExact calls | RTSTPFExact calls | Call reduction | Work reduction | FN |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| `{case_name}` | `{_agg_queries(aggregate):,}` | `{_agg_candidates(aggregate):,}` | `{aggregate['no_proposal']['exact_call_count']:,}` | `{aggregate['rtstpf']['exact_call_count']:,}` | `{_pct(aggregate_call_reduction)}` | `{_pct(aggregate['exact_work_reduction'])}` | `{aggregate['rtstpf']['fn_count']}` |",
        "",
        "## correctnessNotes",
        "",
        "- `RTSTPFExact` only performs proposal/scheduling, descriptionconnectdescriptioncollisionanddescription. ",
        "- descriptioncollisionConclusiondescription exact certificate description, thereforedescriptioncorrectnessdescriptionis `FN = 0`. ",
        "- this case description query visualizationand aggregate benchmark descriptionkeep `FN = 0`. ",
        "",
        "## description",
        "",
        "this case description Broad Phase afterdescriptioncandidatehigh-densitycontactscene. this paperMethodadvantageis notreplace Broad Phase, insteadin Broad Phase descriptionafterdescriptionenter exact certificate candidatedescription. thisdescription case descriptionasdescriptionindescription `learned STPF + exact certificate` advantagevisualizationdescription. ",
        "",
    ]
    report = "\n".join(lines)
    (case_dir / "benchmark_report.md").write_text(report, encoding="utf-8")
    (case_dir / f"{_safe_name(case_name)}.md").write_text(report, encoding="utf-8")


def _write_metrics(
    *,
    case_dir: Path,
    case_name: str,
    pair: Any,
    sample: Any,
    trace: Any,
    no_proposal: HighDensityMethodMetrics,
    rt_exact: HighDensityMethodMetrics,
    rtstpf: HighDensityMethodMetrics,
    aggregate: dict[str, Any],
    selected_candidate_ids: tuple[int, ...],
    assets: dict[str, str],
) -> None:
    payload = {
        "case_name": case_name,
        "pair": {
            "source_name": pair.source_name,
            "asset_a": asdict(pair.asset_a),
            "asset_b": asdict(pair.asset_b),
            "pair_score": pair.pair_score,
            "cost_scale": pair.cost_scale,
        },
        "sample": {
            "sample_id": sample.sample_id,
            "query_id": sample.query_id,
            "split": sample.split,
        },
        "trace": {
            "collided": bool(trace.collided),
            "toi": float(trace.toi),
            "contact_interval_t0": float(trace.contact_interval_t0),
            "contact_interval_t1": float(trace.contact_interval_t1),
        },
        "methods": {
            "RTSTPFExact": asdict(rtstpf),
            "RTExact": asdict(rt_exact),
            "NoProposal": asdict(no_proposal),
        },
        "aggregate": aggregate,
        "selected_candidate_ids": list(selected_candidate_ids),
        "assets": assets,
    }
    (case_dir / "metrics.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def _write_contact_sheet(case_dirs: list[Path], *, view_mode: str) -> Path:
    tile_w, tile_h = 560, 318
    margin = 24
    title_h = 94
    sheet = Image.new("RGB", (3 * tile_w + 4 * margin, title_h + tile_h + 2 * margin), (8, 13, 25))
    draw = ImageDraw.Draw(sheet, "RGBA")
    title = "Large Dense Complex Mesh Cases - Global View" if view_mode == "global" else "Large Dense Complex Mesh Cases - Local Zoom View"
    subtitle = "ABC megaface + Fusion dense + Thingi10K dirty, real mesh replay"
    draw.text((margin, 24), title, font=_font(30, bold=True), fill=(226, 232, 240, 255))
    draw.text((margin, 62), subtitle, font=_font(17), fill=(148, 163, 184, 255))
    suffix = "_global_preview.png" if view_mode == "global" else "_local_zoom_preview.png"
    for idx, case_dir in enumerate(case_dirs):
        preview = next((case_dir / "visualizations").glob(f"*{suffix}"))
        image = Image.open(preview).convert("RGB").resize((tile_w, tile_h), Image.Resampling.LANCZOS)
        x = margin + idx * (tile_w + margin)
        y = title_h + margin
        sheet.paste(image, (x, y))
        draw.rounded_rectangle((x, y, x + tile_w, y + 42), radius=10, fill=(8, 13, 25, 210))
        draw.text((x + 14, y + 9), case_dir.name, font=_font(15, bold=True), fill=(226, 232, 240, 255))
    path = RUN_ROOT / f"large_dense_complex_mesh_cases_{view_mode}_contact_sheet.png"
    sheet.save(path)
    return path


def _write_root_readme(case_dirs: list[Path], *, global_sheet: Path, local_sheet: Path) -> None:
    lines = [
        "# Paper Demo: Large Dense Complex Mesh Cases",
        "",
        "thisdescriptioncontains ABC megaface, Fusion dense, Thingi10K dirty description, description dense case. each case descriptioncontains global MP4, local zoom MP4, description HTML, TOI descriptionafter PNG, metrics JSON andindescriptionperformance report. ",
        "",
        "## descriptionvisualizationdescription",
        "",
        "- `global`: completedescriptionconnectdescription, collision, separation, used fordescriptionanddescriptionscenedescription. ",
        "- `local_zoom`: TOI description, used fordescriptionindescriptioncontactdescriptionandcandidatedescription. ",
        "",
        "## overview figure",
        "",
        f"- Global sheet: `{global_sheet.name}`",
        f"- Local zoom sheet: `{local_sheet.name}`",
        "",
        "| Case | Directory |",
        "| --- | --- |",
    ]
    for case_dir in case_dirs:
        lines.append(f"| `{case_dir.name}` | `{case_dir}` |")
    lines.extend(
        [
            "",
            "description: description case description Broad Phase afterdescriptionindescriptioncandidate dense/high-cost workload. RTSTPFExact through learned proposal/scheduling description exact certificate call, butdescriptioncollisionConclusiondescription exact certificate description. ",
            "",
        ],
    )
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    (RUN_ROOT / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    cfg = LargeDenseComplexMeshCasesConfig()
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"checkpoint not found: {CHECKPOINT}")
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    if SOURCE_JSON.exists():
        shutil.copy2(SOURCE_JSON, RUN_ROOT / "source_complete_benchmark.json")
    if SOURCE_MD.exists():
        shutil.copy2(SOURCE_MD, RUN_ROOT / "source_complete_benchmark.md")

    aggregate_by_case = _load_case_aggregate_metrics()
    eval_pairs_by_case = _reconstruct_eval_pairs()
    model = _load_model(CHECKPOINT, device="cuda")
    case_dirs: list[Path] = []

    for case_index, case_name in enumerate(CASE_ORDER):
        case_dir = RUN_ROOT / _safe_name(case_name)
        case_dir.mkdir(parents=True, exist_ok=True)
        (case_dir / "visualizations").mkdir(exist_ok=True)
        aggregate = aggregate_by_case[case_name]
        pair = eval_pairs_by_case[case_name][0]
        probe_mesh_a = _load_render_mesh(pair.asset_a.asset_path, max_faces=120_000, seed=15_000_000 + case_index * 1000 + 11)
        probe_mesh_b = _load_render_mesh(pair.asset_b.asset_path, max_faces=120_000, seed=15_000_000 + case_index * 1000 + 12)
        sample, trace = _build_true_surface_contact_sample(
            mesh_a=probe_mesh_a,
            mesh_b=probe_mesh_b,
            pair=pair,
            sample_id=15_000_000 + case_index * 1000,
        )
        workload = _build_single_workload(pair, sample, cfg)
        no_proposal = benchmark_no_proposal_on_high_density_workload(workload)
        rt_exact = _rt_exact_metrics_from_no_proposal(no_proposal)
        rtstpf = benchmark_stpf_on_high_density_workload(
            workload,
            model=model,
            device="cuda",
            proposal_batch_size=8192,
            method_name=f"{case_name}-RTSTPFExact",
        )
        selected = _selected_candidate_ids_for_stpf(workload, model=model, device="cuda", batch_size=8192)
        global_assets = _render_case_video(
            case_dir=case_dir,
            case_name=case_name,
            pair=pair,
            sample=sample,
            trace=trace,
            no_proposal=no_proposal,
            rt_exact=rt_exact,
            rtstpf=rtstpf,
            aggregate=aggregate,
            frame_count=72,
            fps=24,
            max_faces=120_000,
            scene_scale=18.0,
            view_mode="global",
        )
        local_assets = _render_case_video(
            case_dir=case_dir,
            case_name=case_name,
            pair=pair,
            sample=sample,
            trace=trace,
            no_proposal=no_proposal,
            rt_exact=rt_exact,
            rtstpf=rtstpf,
            aggregate=aggregate,
            frame_count=72,
            fps=24,
            max_faces=120_000,
            scene_scale=18.0,
            view_mode="local_zoom",
        )
        assets = {
            "global_mp4": global_assets["mp4"],
            "global_preview": global_assets["preview"],
            "global_before": global_assets["before"],
            "global_toi": global_assets["toi"],
            "global_after": global_assets["after"],
            "local_zoom_mp4": local_assets["mp4"],
            "local_zoom_preview": local_assets["preview"],
            "local_zoom_before": local_assets["before"],
            "local_zoom_toi": local_assets["toi"],
            "local_zoom_after": local_assets["after"],
        }
        assets["interactive_html"] = _write_interactive_html(
            case_dir=case_dir,
            case_name=case_name,
            pair=pair,
            sample=sample,
            trace=trace,
            max_faces=12_000,
            scene_scale=18.0,
        )
        _write_metrics(
            case_dir=case_dir,
            case_name=case_name,
            pair=pair,
            sample=sample,
            trace=trace,
            no_proposal=no_proposal,
            rt_exact=rt_exact,
            rtstpf=rtstpf,
            aggregate=aggregate,
            selected_candidate_ids=selected,
            assets=assets,
        )
        _write_case_report(
            case_dir=case_dir,
            case_name=case_name,
            pair=pair,
            sample=sample,
            trace=trace,
            no_proposal=no_proposal,
            rt_exact=rt_exact,
            rtstpf=rtstpf,
            aggregate=aggregate,
            assets=assets,
        )
        case_dirs.append(case_dir)
        print(f"built {case_name}: {case_dir} toi={float(trace.toi):.6f}")

    global_sheet = _write_contact_sheet(case_dirs, view_mode="global")
    local_sheet = _write_contact_sheet(case_dirs, view_mode="local_zoom")
    _write_root_readme(case_dirs, global_sheet=global_sheet, local_sheet=local_sheet)
    print(RUN_ROOT)


if __name__ == "__main__":
    main()
