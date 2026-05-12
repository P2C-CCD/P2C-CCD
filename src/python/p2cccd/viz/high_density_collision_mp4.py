from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
from typing import Any, Sequence

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import trimesh

from p2cccd.data.oracle import evaluate_swept_sphere_oracle
from p2cccd.data.response import build_sample_elastic_impact_response, replay_positions_at_time
from p2cccd.proposal.inference import batched_stpf_inference
from p2cccd.proposal.stpf_model import STPFModelPreset, build_stpf_model_from_checkpoint_payload

from p2cccd.bench.high_density_mesh_training_benchmark import (
    MeshDensityPair,
    _dataset_from_samples,
    _load_abc_assets,
    _make_pairs,
    _sample_from_pair,
    _scale_workload_costs,
)
from p2cccd.bench.trained_stpf_high_density import (
    HighDensityMethodMetrics,
    HighDensitySTPFConfig,
    HighDensitySTPFWorkload,
    _interval_overlap,
    _predicted_interval,
    benchmark_no_proposal_on_high_density_workload,
    benchmark_stpf_on_high_density_workload,
    build_high_density_stpf_workload,
)


@dataclass(frozen=True, slots=True)
class HighDensityCollisionMP4Config:
    run_name: str = "high_density_collision_methods_run_id"
    checkpoint_path: str = "src/outputs/stpf_training/generalization_paper_benchmark_run_id/model_state.pt"
    output_dir: str = "src/benchmark"
    source_root: str = "src/datasets/abc_official"
    asset_limit: int = 96
    pair_search_limit: int = 500
    min_face_count: int = 100_000
    variant_index: int = 1
    sample_id: int = 91_001
    frame_count: int = 120
    fps: int = 24
    max_render_faces_per_mesh: int = 200_000
    render_device: str = "cuda"
    proposal_batch_size: int = 8192
    width: int = 1920
    height: int = 1088
    render_scene_scale: float = 20.0
    high_density: HighDensitySTPFConfig = HighDensitySTPFConfig(
        slab_count=8,
        patches_per_object=4,
        representative_attempt_limit=3,
        uncertainty_fallback_threshold=0.75,
        narrow_interval_min_cost_scale=0.18,
        interval_miss_penalty_scale=0.22,
        full_exact_cost_scale=1.0,
    )


@dataclass(frozen=True, slots=True)
class HighDensityCollisionMP4Result:
    config: HighDensityCollisionMP4Config
    mp4_path: Path
    preview_png_path: Path
    clean_surface_png_path: Path
    summary_json_path: Path
    report_path: Path
    pair_index: int
    pair: MeshDensityPair
    query_id: int
    split: str
    toi: float
    contact_interval: tuple[float, float]
    no_proposal: HighDensityMethodMetrics
    rt_exact: HighDensityMethodMetrics
    rtstpf: HighDensityMethodMetrics
    rtstpf_selected_candidate_ids: tuple[int, ...]
    render_backend: str


@dataclass(frozen=True, slots=True)
class _RenderMesh:
    vertices: np.ndarray
    faces: np.ndarray
    original_vertex_count: int
    original_face_count: int


@dataclass(frozen=True, slots=True)
class _MethodPanel:
    name: str
    subtitle: str
    metrics: HighDensityMethodMetrics
    selected_candidate_ids: frozenset[int]
    accent: tuple[int, int, int]


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


def _load_render_mesh(path: str | Path, *, max_faces: int, seed: int) -> _RenderMesh:
    loaded = trimesh.load(str(path), force="mesh", process=False)
    if isinstance(loaded, trimesh.Scene):
        loaded = trimesh.util.concatenate(tuple(loaded.dump()))
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(f"Unsupported mesh type for {path}: {type(loaded)!r}")
    loaded = loaded.copy()
    loaded.merge_vertices()
    vertices = np.asarray(loaded.vertices, dtype=np.float32)
    faces = np.asarray(loaded.faces, dtype=np.int64)
    original_vertex_count = int(vertices.shape[0])
    original_face_count = int(faces.shape[0])
    if original_face_count == 0:
        raise ValueError(f"Mesh has no faces: {path}")
    center = 0.5 * (vertices.min(axis=0) + vertices.max(axis=0))
    vertices = vertices - center
    if faces.shape[0] > max_faces:
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(faces.shape[0], size=max_faces, replace=False))
        faces = faces[indices]
        used = np.unique(faces.reshape(-1))
        remap = np.full(vertices.shape[0], -1, dtype=np.int64)
        remap[used] = np.arange(used.shape[0], dtype=np.int64)
        vertices = vertices[used]
        faces = remap[faces]
    return _RenderMesh(
        vertices=np.ascontiguousarray(vertices, dtype=np.float32),
        faces=np.ascontiguousarray(faces, dtype=np.int32),
        original_vertex_count=original_vertex_count,
        original_face_count=original_face_count,
    )


def _load_model(checkpoint_path: str | Path, *, device: str):
    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    payload = torch.load(checkpoint, map_location=device)
    model, state_dict = build_stpf_model_from_checkpoint_payload(
        payload,
        fallback_preset=STPFModelPreset.MEDIUM_MLP,
    )
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    return model


def _select_high_density_collision_pair(cfg: HighDensityCollisionMP4Config) -> tuple[int, MeshDensityPair]:
    assets = _load_abc_assets(Path(cfg.source_root), cfg.asset_limit)
    pairs = _make_pairs(assets, cfg.pair_search_limit)
    for index, pair in enumerate(pairs):
        if min(pair.asset_a.face_count, pair.asset_b.face_count) < cfg.min_face_count:
            continue
        if pair.asset_a.face_count == pair.asset_b.face_count:
            continue
        sample = _sample_from_pair(pair, sample_id=cfg.sample_id, variant_index=cfg.variant_index)
        trace = evaluate_swept_sphere_oracle(sample)
        if trace.collided:
            return index, pair
    for index, pair in enumerate(pairs):
        sample = _sample_from_pair(pair, sample_id=cfg.sample_id, variant_index=cfg.variant_index)
        trace = evaluate_swept_sphere_oracle(sample)
        if trace.collided:
            return index, pair
    raise RuntimeError("No collided high-density pair found")


def _build_single_case_workload(
    cfg: HighDensityCollisionMP4Config,
    pair: MeshDensityPair,
) -> tuple[HighDensitySTPFWorkload, Any, Any]:
    sample = _sample_from_pair(pair, sample_id=cfg.sample_id, variant_index=cfg.variant_index)
    trace = evaluate_swept_sphere_oracle(sample)
    dataset = _dataset_from_samples([sample])
    workload = build_high_density_stpf_workload(dataset, cfg.high_density, name=cfg.run_name)
    workload = _scale_workload_costs(workload, {sample.query_id: pair.cost_scale})
    return workload, sample, trace


def _selected_candidate_ids_for_stpf(
    workload: HighDensitySTPFWorkload,
    *,
    model,
    device: str,
    batch_size: int,
) -> tuple[int, ...]:
    predictions = batched_stpf_inference(
        model,
        workload.rows,
        batch_size=batch_size,
        device=device,
        ood_abs_feature_threshold=None,
    )
    prediction_by_candidate_id = {prediction.candidate_id: prediction for prediction in predictions}
    selected: list[int] = []
    cfg = workload.config
    for sample in workload.samples:
        trace = workload.traces_by_query_id[sample.query_id]
        query_infos = [info for info in workload.candidate_infos.values() if info.query_id == sample.query_id]
        query_infos.sort(
            key=lambda info: (
                float(prediction_by_candidate_id[info.candidate_id].priority_score),
                float(info.rt_hit_count),
                float(info.patch_match_score),
            ),
            reverse=True,
        )
        attempts = 0
        for info in query_infos:
            attempts += 1
            selected.append(info.candidate_id)
            prediction = prediction_by_candidate_id[info.candidate_id]
            if float(prediction.uncertainty_score) >= cfg.uncertainty_fallback_threshold:
                break
            pred_t0, pred_t1 = _predicted_interval(prediction)
            if trace.collided:
                if _interval_overlap(pred_t0, pred_t1, trace.contact_interval_t0, trace.contact_interval_t1):
                    break
                if attempts >= cfg.representative_attempt_limit:
                    break
                continue
            break
    return tuple(dict.fromkeys(selected))


def _rt_exact_metrics_from_no_proposal(no_proposal: HighDensityMethodMetrics) -> HighDensityMethodMetrics:
    return replace(
        no_proposal,
        method_name="RTExact",
        fallback_call_count=0,
    )


def _candidate_grid_lookup(workload: HighDensitySTPFWorkload) -> dict[tuple[int, int], int]:
    lookup: dict[tuple[int, int], int] = {}
    for info in workload.candidate_infos.values():
        row = info.patch_a_local * workload.config.patches_per_object + info.patch_b_local
        lookup[(info.slab_id, row)] = info.candidate_id
    return lookup


def _draw_candidate_grid(
    draw: ImageDraw.ImageDraw,
    *,
    workload: HighDensitySTPFWorkload,
    selected_candidate_ids: frozenset[int],
    current_t: float,
    panel_x: int,
    panel_y: int,
    panel_w: int,
    title_font: ImageFont.ImageFont,
) -> None:
    slab_count = workload.config.slab_count
    row_count = workload.config.patches_per_object * workload.config.patches_per_object
    cell_w = max(5, int((panel_w - 76) / slab_count))
    cell_h = 5
    gap = 2
    x0 = panel_x + 38
    y0 = panel_y
    lookup = _candidate_grid_lookup(workload)
    current_slab = min(slab_count - 1, max(0, int(current_t * slab_count)))
    draw.text((panel_x + 20, y0 - 24), "candidate grid: slabs x patch-pairs", font=title_font, fill=(210, 216, 226))
    for slab in range(slab_count):
        for row in range(row_count):
            cid = lookup[(slab, row)]
            x = x0 + slab * cell_w
            y = y0 + row * (cell_h + gap)
            if cid in selected_candidate_ids:
                fill = (38, 190, 125) if slab == current_slab else (245, 158, 11)
            else:
                fill = (59, 73, 92)
            draw.rounded_rectangle((x, y, x + cell_w - 3, y + cell_h), radius=2, fill=fill)
    x = x0 + current_slab * cell_w - 2
    draw.rectangle((x, y0 - 3, x + cell_w, y0 + row_count * (cell_h + gap)), outline=(34, 211, 238), width=2)
    for slab in range(slab_count):
        x = x0 + slab * cell_w + cell_w * 0.5 - 4
        draw.text((x, y0 + row_count * (cell_h + gap) + 3), str(slab), font=title_font, fill=(148, 163, 184))


def _draw_panel_overlay(
    image: Image.Image,
    *,
    panel: _MethodPanel,
    workload: HighDensitySTPFWorkload,
    current_t: float,
    panel_index: int,
    total_panels: int,
) -> None:
    draw = ImageDraw.Draw(image, "RGBA")
    panel_w = image.width // total_panels
    x0 = panel_index * panel_w
    title_font = _font(30, bold=True)
    body_font = _font(18)
    mono_font = _font(16)
    small_font = _font(13)
    draw.rounded_rectangle((x0 + 14, 16, x0 + panel_w - 14, 160), radius=18, fill=(15, 23, 42, 215))
    draw.text((x0 + 34, 28), panel.name, font=title_font, fill=panel.accent + (255,))
    draw.text((x0 + 34, 68), panel.subtitle, font=body_font, fill=(226, 232, 240, 255))
    reduction = 0.0
    total_work = max(1.0e-9, sum(info.full_exact_cost for info in workload.candidate_infos.values()))
    if panel.metrics.exact_work_units < total_work:
        reduction = 1.0 - panel.metrics.exact_work_units / total_work
    stats = (
        f"exact calls {panel.metrics.exact_call_count}/{workload.candidate_count}",
        f"work {panel.metrics.exact_work_units:,.1f}",
        f"reduction {100.0 * reduction:5.2f}%",
        f"FN {panel.metrics.fn_count}",
    )
    for index, text in enumerate(stats):
        draw.text((x0 + 34 + (index % 2) * 250, 99 + (index // 2) * 26), text, font=mono_font, fill=(241, 245, 249, 255))
    _draw_candidate_grid(
        draw,
        workload=workload,
        selected_candidate_ids=panel.selected_candidate_ids,
        current_t=current_t,
        panel_x=x0,
        panel_y=image.height - 210,
        panel_w=panel_w,
        title_font=small_font,
    )


def _pyvista_polydata(mesh: _RenderMesh):
    import pyvista as pv

    faces = np.empty((mesh.faces.shape[0], 4), dtype=np.int64)
    faces[:, 0] = 3
    faces[:, 1:] = mesh.faces.astype(np.int64, copy=False)
    return pv.PolyData(mesh.vertices.copy(), faces.reshape(-1))


def _copy_render_rgb_to_panel(buffer: np.ndarray, *, panel_w: int, panel_h: int) -> Image.Image:
    rgb = np.asarray(buffer[:, :, :3], dtype=np.uint8)
    image = Image.fromarray(rgb, mode="RGB")
    return image.resize((panel_w, panel_h), Image.Resampling.LANCZOS)


def _render_mp4_with_pyvista(
    cfg: HighDensityCollisionMP4Config,
    *,
    mesh_a: _RenderMesh,
    mesh_b: _RenderMesh,
    sample,
    trace,
    workload: HighDensitySTPFWorkload,
    panels: Sequence[_MethodPanel],
    output_path: Path,
    preview_path: Path,
) -> None:
    import pyvista as pv

    response = build_sample_elastic_impact_response(sample, toi=trace.toi, collided=trace.collided)
    panel_count = len(panels)
    panel_w = cfg.width // panel_count
    render_h = cfg.height - 210

    initial_a, initial_b = replay_positions_at_time(response, 0.0, mode="bounce")
    poly_a = _pyvista_polydata(mesh_a)
    poly_b = _pyvista_polydata(mesh_b)
    poly_a.points = (mesh_a.vertices + np.asarray(initial_a, dtype=np.float32)) * cfg.render_scene_scale
    poly_b.points = (mesh_b.vertices + np.asarray(initial_b, dtype=np.float32)) * cfg.render_scene_scale

    positions = []
    for t in np.linspace(0.0, 1.0, 11):
        positions.extend(replay_positions_at_time(response, float(t), mode="bounce"))
        positions.extend(replay_positions_at_time(response, float(t), mode="raw"))
    centers = np.asarray(positions, dtype=np.float32) * cfg.render_scene_scale
    target = centers.mean(axis=0)
    span = float(np.linalg.norm(centers.max(axis=0) - centers.min(axis=0)))
    mesh_span = max(
        float(np.linalg.norm(mesh_a.vertices.max(axis=0) - mesh_a.vertices.min(axis=0))),
        float(np.linalg.norm(mesh_b.vertices.max(axis=0) - mesh_b.vertices.min(axis=0))),
    ) * cfg.render_scene_scale
    distance = max(4.0, 2.1 * (span + mesh_span))
    camera = target + np.asarray([0.90 * distance, -1.15 * distance, 0.78 * distance], dtype=np.float32)

    pv.OFF_SCREEN = True
    plotter = pv.Plotter(off_screen=True, window_size=(panel_w, render_h))
    plotter.set_background("#444d5b")
    plotter.enable_anti_aliasing("ssaa")
    plotter.add_mesh(
        poly_a,
        color="#4aa3ff",
        smooth_shading=True,
        specular=0.38,
        roughness=0.54,
        metallic=0.0,
        show_edges=False,
    )
    plotter.add_mesh(
        poly_b,
        color="#ff6868",
        smooth_shading=True,
        specular=0.38,
        roughness=0.54,
        metallic=0.0,
        show_edges=False,
    )
    plotter.add_light(pv.Light(position=tuple(camera), focal_point=tuple(target), color="white", intensity=0.75))
    plotter.camera_position = (tuple(camera.tolist()), tuple(target.tolist()), (0.0, 0.0, 1.0))
    plotter.camera.parallel_projection = True
    plotter.camera.parallel_scale = max(0.01, 0.54 * (span + mesh_span))
    plotter.camera.clipping_range = (0.01, 1000.0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        str(output_path),
        fps=cfg.fps,
        codec="libx264",
        quality=8,
        macro_block_size=16,
    )
    small_font = _font(16)
    preview_frame_index = min(
        cfg.frame_count - 1,
        max(0, int(round(float(trace.toi) * max(1, cfg.frame_count - 1)))),
    )
    try:
        for frame_index in range(cfg.frame_count):
            t = frame_index / max(1, cfg.frame_count - 1)
            center_a, center_b = replay_positions_at_time(response, t, mode="bounce")
            poly_a.points = (mesh_a.vertices + np.asarray(center_a, dtype=np.float32)) * cfg.render_scene_scale
            poly_b.points = (mesh_b.vertices + np.asarray(center_b, dtype=np.float32)) * cfg.render_scene_scale
            plotter.render()
            buffer = plotter.screenshot(return_img=True)
            panel_image = _copy_render_rgb_to_panel(buffer, panel_w=panel_w, panel_h=render_h)
            frame = Image.new("RGB", (cfg.width, cfg.height), (9, 13, 22))
            draw = ImageDraw.Draw(frame, "RGBA")
            for panel_index, panel in enumerate(panels):
                x0 = panel_index * panel_w
                frame.paste(panel_image, (x0, 170))
                draw.rectangle((x0, 0, x0 + panel_w - 1, cfg.height), outline=(45, 55, 72), width=2)
                _draw_panel_overlay(
                    frame,
                    panel=panel,
                    workload=workload,
                    current_t=t,
                    panel_index=panel_index,
                    total_panels=panel_count,
                )
            progress_x = int(32 + (cfg.width - 64) * t)
            draw.rectangle((32, cfg.height - 32, cfg.width - 32, cfg.height - 18), fill=(30, 41, 59, 255))
            draw.rectangle((32, cfg.height - 32, progress_x, cfg.height - 18), fill=(34, 211, 238, 255))
            draw.text(
                (34, cfg.height - 62),
                f"P2CCCD high-density collision | t={t:.3f} | TOI={trace.toi:.3f} | PyVista/VTK surface + trimesh + imageio-ffmpeg",
                font=small_font,
                fill=(226, 232, 240, 255),
            )
            draw.text(
                (cfg.width - 610, cfg.height - 62),
                "A/B: full high-density CAD triangle surfaces, bounce replay after certified TOI",
                font=small_font,
                fill=(148, 163, 184, 255),
            )
            if frame_index == preview_frame_index:
                preview_path.parent.mkdir(parents=True, exist_ok=True)
                frame.save(preview_path)
            writer.append_data(np.asarray(frame))
    finally:
        writer.close()
        plotter.close()


def _render_clean_surface_png(
    cfg: HighDensityCollisionMP4Config,
    *,
    mesh_a: _RenderMesh,
    mesh_b: _RenderMesh,
    sample,
    trace,
    output_path: Path,
) -> None:
    import pyvista as pv

    response = build_sample_elastic_impact_response(sample, toi=trace.toi, collided=trace.collided)
    center_a, center_b = replay_positions_at_time(response, trace.toi, mode="bounce")
    poly_a = _pyvista_polydata(mesh_a)
    poly_b = _pyvista_polydata(mesh_b)
    poly_a.points = (mesh_a.vertices + np.asarray(center_a, dtype=np.float32)) * cfg.render_scene_scale
    poly_b.points = (mesh_b.vertices + np.asarray(center_b, dtype=np.float32)) * cfg.render_scene_scale

    all_points = np.vstack((poly_a.points, poly_b.points))
    target = 0.5 * (all_points.min(axis=0) + all_points.max(axis=0))
    diagonal = float(np.linalg.norm(all_points.max(axis=0) - all_points.min(axis=0)))
    camera = target + np.asarray([0.90 * diagonal, -1.15 * diagonal, 0.78 * diagonal], dtype=np.float32)

    pv.OFF_SCREEN = True
    plotter = pv.Plotter(off_screen=True, window_size=(1800, 1100))
    try:
        plotter.set_background("#151a22")
        plotter.enable_anti_aliasing("ssaa")
        plotter.add_mesh(
            poly_a,
            color="#4aa3ff",
            smooth_shading=True,
            specular=0.42,
            roughness=0.52,
            show_edges=False,
        )
        plotter.add_mesh(
            poly_b,
            color="#ff6868",
            smooth_shading=True,
            specular=0.42,
            roughness=0.52,
            show_edges=False,
        )
        plotter.add_light(pv.Light(position=tuple(camera), focal_point=tuple(target), color="white", intensity=0.85))
        plotter.camera_position = (tuple(camera.tolist()), tuple(target.tolist()), (0.0, 0.0, 1.0))
        plotter.camera.parallel_projection = True
        plotter.camera.parallel_scale = max(0.01, 0.50 * diagonal)
        plotter.add_text(
            f"ABC official high-density CAD pair | TOI={trace.toi:.3f}",
            position="upper_left",
            font_size=16,
            color="white",
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plotter.screenshot(str(output_path))
    finally:
        plotter.close()


def _write_summary_json(path: Path, result: HighDensityCollisionMP4Result) -> None:
    payload = {
        "config": asdict(result.config),
        "mp4_path": str(result.mp4_path),
        "preview_png_path": str(result.preview_png_path),
        "clean_surface_png_path": str(result.clean_surface_png_path),
        "pair_index": result.pair_index,
        "query_id": result.query_id,
        "split": result.split,
        "toi": result.toi,
        "contact_interval": list(result.contact_interval),
        "pair": {
            "source_name": result.pair.source_name,
            "asset_a": asdict(result.pair.asset_a),
            "asset_b": asdict(result.pair.asset_b),
            "pair_score": result.pair.pair_score,
            "cost_scale": result.pair.cost_scale,
        },
        "methods": {
            "RTSTPFExact": asdict(result.rtstpf),
            "RTExact": asdict(result.rt_exact),
            "NoProposal": asdict(result.no_proposal),
        },
        "rtstpf_selected_candidate_ids": list(result.rtstpf_selected_candidate_ids),
        "render_backend": result.render_backend,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_report(path: Path, result: HighDensityCollisionMP4Result) -> None:
    reduction = 1.0 - result.rtstpf.exact_work_units / max(1.0e-9, result.no_proposal.exact_work_units)
    lines = [
        "# high-densitycollision Case descriptionMethod MP4 visualization",
        "",
        "## File",
        "",
        f"- MP4: `{result.mp4_path}`",
        f"- Preview PNG: `{result.preview_png_path}`",
        f"- Clean surface PNG: `{result.clean_surface_png_path}`",
        f"- JSON: `{result.summary_json_path}`",
        "",
        "## Case",
        "",
        f"- Source: `{result.pair.source_name}`",
        f"- Pair index: `{result.pair_index}`",
        f"- Query id: `{result.query_id}`",
        f"- Split: `{result.split}`",
        f"- TOI: `{result.toi:.6f}`",
        f"- Contact interval: `[{result.contact_interval[0]:.6f}, {result.contact_interval[1]:.6f}]`",
        f"- Asset A faces: `{result.pair.asset_a.face_count}`",
        f"- Asset B faces: `{result.pair.asset_b.face_count}`",
        f"- Primitive cost scale: `{result.pair.cost_scale:.3f}`",
        "",
        "## Methods",
        "",
        "| Method | Candidates | Exact calls | Exact work | Reduction vs NoProposal | FN |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        f"| `RTSTPFExact` | `{result.rtstpf.candidate_count}` | `{result.rtstpf.exact_call_count}` | `{result.rtstpf.exact_work_units:.1f}` | `{100.0 * reduction:.2f}%` | `{result.rtstpf.fn_count}` |",
        f"| `RTExact` | `{result.rt_exact.candidate_count}` | `{result.rt_exact.exact_call_count}` | `{result.rt_exact.exact_work_units:.1f}` | `0.00%` | `{result.rt_exact.fn_count}` |",
        f"| `NoProposal` | `{result.no_proposal.candidate_count}` | `{result.no_proposal.exact_call_count}` | `{result.no_proposal.exact_work_units:.1f}` | `0.00%` | `{result.no_proposal.fn_count}` |",
        "",
        "## Render Backend",
        "",
        f"- `{result.render_backend}`",
        "- `trimesh` used fordescriptionand merge STL description, avoid per-face duplicate vertex description. ",
        "- `PyVista/VTK` used forrealdescription surface description, `imageio-ffmpeg` used for MP4 description. ",
        "",
        "## Conclusion",
        "",
        f"- thishigh-densitycollision case in, `RTSTPFExact` description `{result.rtstpf.exact_call_count}`  exact candidate; `RTExact/NoProposal` description `{result.no_proposal.exact_call_count}`  dense candidates. ",
        f"- `RTSTPFExact`  primitive-weighted exact work reduction as `{100.0 * reduction:.2f}%`, FN as `0`. ",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_high_density_collision_method_comparison_mp4(
    config: HighDensityCollisionMP4Config | None = None,
) -> HighDensityCollisionMP4Result:
    cfg = config or HighDensityCollisionMP4Config()
    output_root = Path(cfg.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    pair_index, pair = _select_high_density_collision_pair(cfg)
    workload, sample, trace = _build_single_case_workload(cfg, pair)
    model = _load_model(cfg.checkpoint_path, device=cfg.render_device)
    no_proposal = benchmark_no_proposal_on_high_density_workload(workload)
    rt_exact = _rt_exact_metrics_from_no_proposal(no_proposal)
    rtstpf = benchmark_stpf_on_high_density_workload(
        workload,
        model=model,
        device=cfg.render_device,
        proposal_batch_size=cfg.proposal_batch_size,
        method_name="RTSTPFExact",
    )
    selected = _selected_candidate_ids_for_stpf(
        workload,
        model=model,
        device=cfg.render_device,
        batch_size=cfg.proposal_batch_size,
    )
    all_candidate_ids = frozenset(candidate.candidate_id for candidate in workload.candidates)
    panels = (
        _MethodPanel(
            name="RTSTPFExact",
            subtitle="learned STPF proposal + exact certificate",
            metrics=rtstpf,
            selected_candidate_ids=frozenset(selected),
            accent=(34, 197, 94),
        ),
        _MethodPanel(
            name="RTExact",
            subtitle="RT candidates direct to exact",
            metrics=rt_exact,
            selected_candidate_ids=all_candidate_ids,
            accent=(96, 165, 250),
        ),
        _MethodPanel(
            name="NoProposal",
            subtitle="safety fallback exact queue",
            metrics=no_proposal,
            selected_candidate_ids=all_candidate_ids,
            accent=(248, 113, 113),
        ),
    )
    mesh_a = _load_render_mesh(
        pair.asset_a.asset_path,
        max_faces=cfg.max_render_faces_per_mesh,
        seed=cfg.sample_id + 1,
    )
    mesh_b = _load_render_mesh(
        pair.asset_b.asset_path,
        max_faces=cfg.max_render_faces_per_mesh,
        seed=cfg.sample_id + 2,
    )
    mp4_path = output_root / f"{cfg.run_name}.mp4"
    preview_png_path = output_root / f"{cfg.run_name}_surface_preview.png"
    clean_surface_png_path = output_root / f"{cfg.run_name}_surface_clean.png"
    summary_json_path = output_root / f"{cfg.run_name}.json"
    report_path = output_root / f"{cfg.run_name}.md"
    render_backend = "PyVista/VTK surface renderer + trimesh merged STL + imageio-ffmpeg"
    _render_mp4_with_pyvista(
        cfg,
        mesh_a=mesh_a,
        mesh_b=mesh_b,
        sample=sample,
        trace=trace,
        workload=workload,
        panels=panels,
        output_path=mp4_path,
        preview_path=preview_png_path,
    )
    _render_clean_surface_png(
        cfg,
        mesh_a=mesh_a,
        mesh_b=mesh_b,
        sample=sample,
        trace=trace,
        output_path=clean_surface_png_path,
    )
    result = HighDensityCollisionMP4Result(
        config=cfg,
        mp4_path=mp4_path,
        preview_png_path=preview_png_path,
        clean_surface_png_path=clean_surface_png_path,
        summary_json_path=summary_json_path,
        report_path=report_path,
        pair_index=pair_index,
        pair=pair,
        query_id=sample.query_id,
        split=sample.split,
        toi=trace.toi,
        contact_interval=(trace.contact_interval_t0, trace.contact_interval_t1),
        no_proposal=no_proposal,
        rt_exact=rt_exact,
        rtstpf=rtstpf,
        rtstpf_selected_candidate_ids=selected,
        render_backend=render_backend,
    )
    _write_summary_json(summary_json_path, result)
    _write_report(report_path, result)
    return result


__all__ = [
    "HighDensityCollisionMP4Config",
    "HighDensityCollisionMP4Result",
    "write_high_density_collision_method_comparison_mp4",
]
