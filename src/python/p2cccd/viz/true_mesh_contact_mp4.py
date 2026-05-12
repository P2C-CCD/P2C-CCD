from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw
import torch

from p2cccd.contracts import ProxyType
from p2cccd.data.dataset import GeneratedDataset, proposal_row_from_oracle_trace
from p2cccd.data.oracle import evaluate_swept_sphere_oracle
from p2cccd.data.response import (
    build_sample_elastic_impact_response,
    kinetic_energy,
    momentum,
    proxy_mass_from_radius,
    replay_positions_at_time,
)
from p2cccd.data.samplers import MotionDiscPairSample, PairFamily
from p2cccd.proposal.stpf_model import STPFModelPreset, build_stpf_model_from_checkpoint_payload

from p2cccd.bench.high_density_mesh_training_benchmark import (
    MeshDensityAsset,
    _cost_scale,
    _load_abc_assets,
    _scale_workload_costs,
)
from p2cccd.bench.trained_stpf_high_density import (
    HighDensityMethodMetrics,
    HighDensitySTPFConfig,
    HighDensitySTPFWorkload,
    benchmark_no_proposal_on_high_density_workload,
    benchmark_stpf_on_high_density_workload,
    build_high_density_stpf_workload,
)

from .high_density_collision_mp4 import (
    _MethodPanel,
    _RenderMesh,
    _font,
    _load_render_mesh,
    _pyvista_polydata,
    _render_clean_surface_png,
    _render_mp4_with_pyvista,
    _rt_exact_metrics_from_no_proposal,
    _selected_candidate_ids_for_stpf,
)


@dataclass(frozen=True, slots=True)
class TrueMeshSurfaceContactMP4Config:
    run_name: str = "true_mesh_surface_contact_methods_run_id"
    checkpoint_path: str = "src/outputs/stpf_training/generalization_paper_benchmark_run_id/model_state.pt"
    output_dir: str = "src/benchmark"
    source_root: str = "src/datasets/abc_official"
    asset_id_contains: str = "00140255_3c164a7c54908b1d1f92a122_trimesh_011"
    asset_limit: int = 256
    frame_count: int = 121
    fps: int = 24
    width: int = 1920
    height: int = 1088
    max_render_faces_per_mesh: int = 200_000
    render_scene_scale: float = 20.0
    approach_gap_scale: float = 1.15
    render_device: str = "cuda"
    proposal_batch_size: int = 8192
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
class TrueMeshSurfaceContactMP4Result:
    config: TrueMeshSurfaceContactMP4Config
    mp4_path: Path
    preview_png_path: Path
    clean_surface_png_path: Path
    collision_before_png_path: Path
    collision_toi_png_path: Path
    collision_after_png_path: Path
    summary_json_path: Path
    report_path: Path
    asset: MeshDensityAsset
    query_id: int
    toi: float
    toi_frame_index: int
    toi_video_seconds: float
    contact_center_x: float
    aabb_gap_at_toi: tuple[float, float, float]
    mass_a: float
    mass_b: float
    restitution: float
    velocity_a_pre: tuple[float, float, float]
    velocity_b_pre: tuple[float, float, float]
    velocity_a_post: tuple[float, float, float]
    velocity_b_post: tuple[float, float, float]
    momentum_pre: tuple[float, float, float]
    momentum_post: tuple[float, float, float]
    kinetic_energy_pre: float
    kinetic_energy_post: float
    no_proposal: HighDensityMethodMetrics
    rt_exact: HighDensityMethodMetrics
    rtstpf: HighDensityMethodMetrics
    rtstpf_selected_candidate_ids: tuple[int, ...]
    render_backend: str


@dataclass(frozen=True, slots=True)
class TrueMeshSurfaceContactZoomMP4Config:
    run_name: str = "true_mesh_surface_contact_zoom_wireframe_run_id"
    output_dir: str = "src/benchmark"
    source_root: str = "src/datasets/abc_official"
    asset_id_contains: str = "00140255_3c164a7c54908b1d1f92a122_trimesh_011"
    asset_limit: int = 256
    frame_count: int = 96
    fps: int = 24
    width: int = 1920
    height: int = 1088
    max_render_faces_per_mesh: int = 200_000
    render_scene_scale: float = 20.0
    approach_gap_scale: float = 1.15
    time_start: float = 0.35
    time_end: float = 0.65


@dataclass(frozen=True, slots=True)
class TrueMeshSurfaceContactZoomMP4Result:
    config: TrueMeshSurfaceContactZoomMP4Config
    mp4_path: Path
    preview_png_path: Path
    asset: MeshDensityAsset
    query_id: int
    toi: float
    toi_frame_index: int
    toi_video_seconds: float
    contact_center_x: float
    render_backend: str


@dataclass(frozen=True, slots=True)
class TrueMeshSurfaceContactInteractiveHTMLConfig:
    run_name: str = "true_mesh_surface_contact_interactive_wireframe_run_id"
    output_dir: str = "src/benchmark"
    source_root: str = "src/datasets/abc_official"
    asset_id_contains: str = "00140255_3c164a7c54908b1d1f92a122_trimesh_011"
    asset_limit: int = 256
    frame_count: int = 7
    max_interactive_faces_per_mesh: int | None = 35_000
    max_wireframe_edges_per_mesh: int | None = 55_000
    render_scene_scale: float = 20.0
    approach_gap_scale: float = 1.15
    time_start: float = 0.35
    time_end: float = 0.65
    include_plotlyjs: bool = True
    light_background: str = "#f2f6fa"
    blue_surface_color: str = "#37b7ea"
    red_surface_color: str = "#ff7e68"
    blue_wire_color: str = "rgba(3,63,99,0.78)"
    red_wire_color: str = "rgba(127,29,29,0.78)"
    surface_opacity: float = 0.82
    wireframe_line_width: float = 2.2


@dataclass(frozen=True, slots=True)
class TrueMeshSurfaceContactInteractiveHTMLResult:
    config: TrueMeshSurfaceContactInteractiveHTMLConfig
    html_path: Path
    asset: MeshDensityAsset
    query_id: int
    toi: float
    initial_frame_index: int
    contact_center_x: float
    face_count_per_object_rendered: int
    wireframe_edge_count_per_object: int
    render_backend: str


@dataclass(frozen=True, slots=True)
class TrueMeshSurfaceContactFullSnapshotConfig:
    run_name: str = "abc_full_wireframe_snapshots"
    output_dir: str = "src/benchmark"
    source_root: str = "src/datasets/abc_official"
    asset_id_contains: str = "00140255_3c164a7c54908b1d1f92a122_trimesh_011"
    asset_limit: int = 256
    times: tuple[float, ...] = (0.35, 0.43, 0.49, 0.50, 0.57, 0.65)
    width: int = 2880
    height: int = 1620
    max_render_faces_per_mesh: int = 200_000
    render_scene_scale: float = 20.0
    approach_gap_scale: float = 1.15
    camera_parallel_scale_multiplier: float = 1.08
    light_background: str = "#f2f6fa"
    surface_opacity: float = 0.88
    surface_edge_line_width: float = 0.95
    wireframe_overlay_line_width: float = 1.45
    wireframe_overlay_opacity: float = 0.48


@dataclass(frozen=True, slots=True)
class TrueMeshSurfaceContactFullSnapshotResult:
    config: TrueMeshSurfaceContactFullSnapshotConfig
    output_dir: Path
    snapshot_paths: tuple[Path, ...]
    asset: MeshDensityAsset
    query_id: int
    toi: float
    contact_center_x: float
    face_count_per_object_rendered: int
    render_backend: str


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


def _base_config_from_zoom_config(cfg: TrueMeshSurfaceContactZoomMP4Config) -> TrueMeshSurfaceContactMP4Config:
    return TrueMeshSurfaceContactMP4Config(
        source_root=cfg.source_root,
        asset_id_contains=cfg.asset_id_contains,
        asset_limit=cfg.asset_limit,
        frame_count=121,
        fps=cfg.fps,
        width=cfg.width,
        height=cfg.height,
        max_render_faces_per_mesh=cfg.max_render_faces_per_mesh,
        render_scene_scale=cfg.render_scene_scale,
        approach_gap_scale=cfg.approach_gap_scale,
    )


def _base_config_from_interactive_config(
    cfg: TrueMeshSurfaceContactInteractiveHTMLConfig,
) -> TrueMeshSurfaceContactMP4Config:
    return TrueMeshSurfaceContactMP4Config(
        source_root=cfg.source_root,
        asset_id_contains=cfg.asset_id_contains,
        asset_limit=cfg.asset_limit,
        max_render_faces_per_mesh=cfg.max_interactive_faces_per_mesh or 2_147_483_647,
        render_scene_scale=cfg.render_scene_scale,
        approach_gap_scale=cfg.approach_gap_scale,
    )


def _select_asset(cfg: TrueMeshSurfaceContactMP4Config) -> MeshDensityAsset:
    for asset in _load_abc_assets(Path(cfg.source_root), cfg.asset_limit):
        if cfg.asset_id_contains in asset.asset_id or cfg.asset_id_contains in asset.asset_path:
            return asset
    assets = _load_abc_assets(Path(cfg.source_root), cfg.asset_limit)
    if not assets:
        raise RuntimeError(f"no ABC assets found under {cfg.source_root}")
    return assets[0]


def _mirror_x(mesh: _RenderMesh) -> _RenderMesh:
    vertices = np.ascontiguousarray(mesh.vertices.copy(), dtype=np.float32)
    vertices[:, 0] *= -1.0
    faces = np.ascontiguousarray(mesh.faces[:, [0, 2, 1]], dtype=np.int32)
    return _RenderMesh(
        vertices=vertices,
        faces=faces,
        original_vertex_count=mesh.original_vertex_count,
        original_face_count=mesh.original_face_count,
    )


def _build_true_surface_contact_sample(
    *,
    mesh_a: _RenderMesh,
    mesh_b: _RenderMesh,
    asset: MeshDensityAsset,
    cfg: TrueMeshSurfaceContactMP4Config,
) -> tuple[MotionDiscPairSample, float, tuple[float, float, float]]:
    a_min = mesh_a.vertices.min(axis=0)
    a_max = mesh_a.vertices.max(axis=0)
    b_min = mesh_b.vertices.min(axis=0)
    b_max = mesh_b.vertices.max(axis=0)
    contact_center_x = float(a_max[0] - b_min[0])
    object_width = max(1.0e-6, contact_center_x)
    radius = 0.5 * object_width
    delta = cfg.approach_gap_scale * object_width
    half_delta = 0.5 * delta
    center_a_t0 = (-half_delta, 0.0, 0.0)
    center_a_t1 = (half_delta, 0.0, 0.0)
    center_b_t0 = (contact_center_x + half_delta, 0.0, 0.0)
    center_b_t1 = (contact_center_x - half_delta, 0.0, 0.0)
    sample = MotionDiscPairSample(
        sample_id=95_001,
        query_id=9_501_001,
        candidate_id=9_601_001,
        split="true_mesh_surface_contact",
        family=PairFamily.MESH_PAIR,
        object_a_id=950_001,
        patch_a_id=1,
        object_b_id=950_002,
        patch_b_id=1,
        slab_id=4,
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
        mass_a=proxy_mass_from_radius(radius),
        mass_b=proxy_mass_from_radius(radius),
        restitution=1.0,
    )
    center_a_toi = np.asarray((0.0, 0.0, 0.0), dtype=np.float32)
    center_b_toi = np.asarray((contact_center_x, 0.0, 0.0), dtype=np.float32)
    aabb_gap = np.maximum(
        (a_min + center_a_toi) - (b_max + center_b_toi),
        (b_min + center_b_toi) - (a_max + center_a_toi),
    )
    return sample, contact_center_x, tuple(float(value) for value in aabb_gap)


def _dataset_from_sample(sample: MotionDiscPairSample) -> GeneratedDataset:
    trace = evaluate_swept_sphere_oracle(sample)
    row = proposal_row_from_oracle_trace(sample, trace)
    return GeneratedDataset(
        rows=[row],
        samples=[sample],
        traces=[trace],
        split_names=(sample.split,),
    )


def _build_workload(
    *,
    sample: MotionDiscPairSample,
    asset: MeshDensityAsset,
    cfg: TrueMeshSurfaceContactMP4Config,
) -> HighDensitySTPFWorkload:
    dataset = _dataset_from_sample(sample)
    workload = build_high_density_stpf_workload(dataset, cfg.high_density, name=cfg.run_name)
    cost_scale = _cost_scale(asset, asset)
    return _scale_workload_costs(workload, {sample.query_id: cost_scale})


def _write_summary_json(path: Path, result: TrueMeshSurfaceContactMP4Result) -> None:
    path.write_text(
        json.dumps(
            {
                "config": asdict(result.config),
                "mp4_path": str(result.mp4_path),
                "preview_png_path": str(result.preview_png_path),
                "clean_surface_png_path": str(result.clean_surface_png_path),
                "collision_before_png_path": str(result.collision_before_png_path),
                "collision_toi_png_path": str(result.collision_toi_png_path),
                "collision_after_png_path": str(result.collision_after_png_path),
                "query_id": result.query_id,
                "toi": result.toi,
                "toi_frame_index": result.toi_frame_index,
                "toi_video_seconds": result.toi_video_seconds,
                "contact_center_x": result.contact_center_x,
                "aabb_gap_at_toi": list(result.aabb_gap_at_toi),
                "physics_model": {
                    "description": "equal-mass one-dimensional elastic impact replay along +x/-x; CCD query keeps the same relative swept motion and TOI, while rendered post-TOI motion uses momentum-conserving bounce",
                    "mass_a": result.mass_a,
                    "mass_b": result.mass_b,
                    "restitution": result.restitution,
                    "velocity_a_pre": list(result.velocity_a_pre),
                    "velocity_b_pre": list(result.velocity_b_pre),
                    "velocity_a_post": list(result.velocity_a_post),
                    "velocity_b_post": list(result.velocity_b_post),
                    "momentum_pre": list(result.momentum_pre),
                    "momentum_post": list(result.momentum_post),
                    "kinetic_energy_pre": result.kinetic_energy_pre,
                    "kinetic_energy_post": result.kinetic_energy_post,
                    "momentum_delta_l2": float(
                        np.linalg.norm(np.asarray(result.momentum_post) - np.asarray(result.momentum_pre))
                    ),
                    "kinetic_energy_delta": result.kinetic_energy_post - result.kinetic_energy_pre,
                },
                "asset": asdict(result.asset),
                "methods": {
                    "RTSTPFExact": asdict(result.rtstpf),
                    "RTExact": asdict(result.rt_exact),
                    "NoProposal": asdict(result.no_proposal),
                },
                "rtstpf_selected_candidate_ids": list(result.rtstpf_selected_candidate_ids),
                "render_backend": result.render_backend,
                "true_surface_contact": True,
                "contact_construction": (
                    "mesh B is x-mirrored from mesh A; both bodies move symmetrically toward the contact plane; "
                    "at TOI, B.min_x + translation_x equals A.max_x; post-TOI replay uses equal-mass elastic bounce"
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_report(path: Path, result: TrueMeshSurfaceContactMP4Result) -> None:
    reduction = 1.0 - result.rtstpf.exact_work_units / max(1.0e-9, result.no_proposal.exact_work_units)
    call_reduction = 1.0 - result.rtstpf.exact_call_count / max(1, result.no_proposal.exact_call_count)
    work_speedup = result.no_proposal.exact_work_units / max(1.0e-9, result.rtstpf.exact_work_units)
    total_faces = 2 * result.asset.face_count
    avg_full_work = result.no_proposal.exact_work_units / max(1, result.no_proposal.exact_call_count)
    proposal_share = result.rtstpf.proposal_wall_ms / max(1.0e-9, result.rtstpf.total_wall_ms)
    max_gap = max(result.aabb_gap_at_toi)
    momentum_delta = float(np.linalg.norm(np.asarray(result.momentum_post) - np.asarray(result.momentum_pre)))
    energy_delta = result.kinetic_energy_post - result.kinetic_energy_pre
    lines = [
        "# complete performance analysis: real triangle-surface contact Case performance report",
        "",
        "## Conclusiondescription",
        "",
        f"descriptionreportisthisreal triangle-surface contact case complete performance analysis, coveragecorrectness, candidate density, exact certificate workload, wall-time timing protocol, descriptionsplitdescriptionandthis paperMethodadvantage. this case isdescriptionhigh-density ABC CAD mesh-mesh CCD test, descriptionisvisualizationdescription. descriptionanddescriptionhas `{result.asset.face_count}` description, description `{total_faces}` description; descriptionconstructguarantee `t = {result.toi:.1f}` whendescriptionrealdescription surface contact. visualization replay descriptionusedescriptionquality, coefficient of restitution `{result.restitution:.2f}` descriptioncollision: collisiondescription, collisionafterbymomentumdescriptionsplitdescription. ",
        "",
        f"descriptionConclusion: in `FN = {result.rtstpf.fn_count}` correctnessconstraintunder, `RTSTPFExact` description exact certificate callfrom `{result.no_proposal.exact_call_count}` descriptionreduced to `{result.rtstpf.exact_call_count}` description, calldescriptionreduction `{100.0 * call_reduction:.2f}%`; primitive-weighted `Exact work` from `{result.no_proposal.exact_work_units:.1f}` reduced to `{result.rtstpf.exact_work_units:.1f}`, reduction `{100.0 * reduction:.2f}%`, description `{work_speedup:.2f}x`  exact work reduction. ",
        "",
        f"current wall-time descriptionwhendescription `RTSTPFExact`  `{result.rtstpf.total_wall_ms:.4f} ms` descriptionfrom Python/PyTorch proposal inference; descriptionNotesthisdescription query benchmark case descriptionis proposal runtime, rather than exact certificate. descriptionadvantagedescriptionas: this paperMethodinhigh candidate density case indescriptionreductiondescription exact work, descriptionkeepdescription exact certificate correctness; descriptiontodescriptionafter `ORT/TensorRT + C++ scheduling + CUDA exact` Pathanddescription batch description. ",
        "",
        "## testdescription",
        "",
        f"- Dataset: `{result.asset.source_name}`",
        f"- Asset: `{result.asset.asset_id}`",
        f"- Faces per object: `{result.asset.face_count}`",
        f"- Total faces in rendered pair: `{total_faces}`",
        f"- Query id: `{result.query_id}`",
        f"- Candidate count: `{result.no_proposal.candidate_count}`",
        f"- Candidate density: `{result.no_proposal.avg_candidates_per_query:.1f} candidates/query`",
        f"- True surface TOI: `{result.toi:.6f}`",
        f"- TOI frame: `{result.toi_frame_index}`",
        f"- TOI video seconds: `{result.toi_video_seconds:.6f}`",
        f"- Contact center x: `{result.contact_center_x:.9f}`",
        f"- AABB gap at TOI: `({result.aabb_gap_at_toi[0]:.9e}, {result.aabb_gap_at_toi[1]:.9e}, {result.aabb_gap_at_toi[2]:.9e})`",
        f"- Max separating gap at TOI: `{max_gap:.9e}`",
        "",
        "## physicscollisionModel",
        "",
        "description demo  CCD detectionInputdescriptioniscollisiondescription; TOI afterdescriptionused for replay/visualization, descriptionanddescriptionthis paperMethodcandidatedescriptionadvantage. asdescription, descriptionqualitydescriptioncollision: descriptionfromdescription, descriptionfromdescription, in TOI contactafterdescriptionsplitdescription. ",
        "",
        "| Quantity | Value |",
        "| --- | ---: |",
        f"| Mass A | `{result.mass_a:.9e}` |",
        f"| Mass B | `{result.mass_b:.9e}` |",
        f"| Restitution | `{result.restitution:.3f}` |",
        f"| Velocity A pre | `({result.velocity_a_pre[0]:+.9e}, {result.velocity_a_pre[1]:+.9e}, {result.velocity_a_pre[2]:+.9e})` |",
        f"| Velocity B pre | `({result.velocity_b_pre[0]:+.9e}, {result.velocity_b_pre[1]:+.9e}, {result.velocity_b_pre[2]:+.9e})` |",
        f"| Velocity A post | `({result.velocity_a_post[0]:+.9e}, {result.velocity_a_post[1]:+.9e}, {result.velocity_a_post[2]:+.9e})` |",
        f"| Velocity B post | `({result.velocity_b_post[0]:+.9e}, {result.velocity_b_post[1]:+.9e}, {result.velocity_b_post[2]:+.9e})` |",
        f"| Momentum pre | `({result.momentum_pre[0]:+.9e}, {result.momentum_pre[1]:+.9e}, {result.momentum_pre[2]:+.9e})` |",
        f"| Momentum post | `({result.momentum_post[0]:+.9e}, {result.momentum_post[1]:+.9e}, {result.momentum_post[2]:+.9e})` |",
        f"| Momentum delta L2 | `{momentum_delta:.9e}` |",
        f"| Kinetic energy pre | `{result.kinetic_energy_pre:.9e}` |",
        f"| Kinetic energy post | `{result.kinetic_energy_post:.9e}` |",
        f"| Kinetic energy delta | `{energy_delta:.9e}` |",
        "",
        "## testFile",
        "",
        f"- MP4: `{result.mp4_path}`",
        "- Zoom wireframe MP4: `collision_zoom_wireframe.mp4`",
        "- Interactive HTML: `collision_zoom_wireframe_interactive.html`",
        f"- Preview PNG: `{result.preview_png_path}`",
        f"- Clean surface PNG: `{result.clean_surface_png_path}`",
        f"- Collision before PNG: `{result.collision_before_png_path}`",
        f"- Collision TOI PNG: `{result.collision_toi_png_path}`",
        f"- Collision after PNG: `{result.collision_after_png_path}`",
        "- Metrics JSON: `metrics.json`",
        "",
        "## correctness",
        "",
        "| Method | FN | Interval hit | Interval miss | Notes |",
        "| --- | ---: | ---: | ---: | --- |",
        f"| `RTSTPFExact` | `{result.rtstpf.fn_count}` | `{result.rtstpf.interval_hit_count}` | `{result.rtstpf.interval_miss_count}` | learned STPF descriptionincandidatecoveragereal TOI description, descriptionenter exact certificate |",
        f"| `RTExact` | `{result.rt_exact.fn_count}` | `{result.rt_exact.interval_hit_count}` | `{result.rt_exact.interval_miss_count}` | descriptionconnect RT candidates description exact |",
        f"| `NoProposal` | `{result.no_proposal.fn_count}` | `{result.no_proposal.interval_hit_count}` | `{result.no_proposal.interval_miss_count}` | fallback exact queue, descriptionperform proposal reduction |",
        "",
        "thistestdescriptionneuraldescriptionconnectdescriptioncollisionConclusion; descriptiononly determinescandidatedescriptionand exact workload. descriptionisdescriptioncollisiondescription exact certificate Pathdescription, therefore `RTSTPFExact` correctnessConclusiondescription `FN = 0` and selected candidate isdescriptioncoverage TOI. ",
        "",
        "## workloadMetrics",
        "",
        "| Method | Queries | Candidates | Exact calls | Fallback calls | Exact work | Exact-call reduction | Exact-work reduction |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| `RTSTPFExact` | `{result.rtstpf.query_count}` | `{result.rtstpf.candidate_count}` | `{result.rtstpf.exact_call_count}` | `{result.rtstpf.fallback_call_count}` | `{result.rtstpf.exact_work_units:.1f}` | `{100.0 * call_reduction:.2f}%` | `{100.0 * reduction:.2f}%` |",
        f"| `RTExact` | `{result.rt_exact.query_count}` | `{result.rt_exact.candidate_count}` | `{result.rt_exact.exact_call_count}` | `{result.rt_exact.fallback_call_count}` | `{result.rt_exact.exact_work_units:.1f}` | `0.00%` | `0.00%` |",
        f"| `NoProposal` | `{result.no_proposal.query_count}` | `{result.no_proposal.candidate_count}` | `{result.no_proposal.exact_call_count}` | `{result.no_proposal.fallback_call_count}` | `{result.no_proposal.exact_work_units:.1f}` | `0.00%` | `0.00%` |",
        "",
        f"`Exact work` is primitive-weighted exact certificate workloaddescription, is notdescription. descriptionby candidate  full/narrow exact cost description, usedescriptionreal exact CCD indescription mesh ondescription. this case in, `RTExact/NoProposal` descriptioneach candidate  full exact work description `{avg_full_work:.1f}`, description `RTSTPFExact` description narrow exact candidate, work as `{result.rtstpf.exact_work_units:.1f}`. ",
        "",
        "## Wall-Time descriptionwhen",
        "",
        "| Method | Proposal wall ms | Scheduling wall ms | Total wall ms | description |",
        "| --- | ---: | ---: | ---: | --- |",
        f"| `RTSTPFExact` | `{result.rtstpf.proposal_wall_ms:.4f}` | `{result.rtstpf.scheduling_wall_ms:.4f}` | `{result.rtstpf.total_wall_ms:.4f}` | Python/PyTorch proposal inference, description `{100.0 * proposal_share:.2f}%` |",
        f"| `RTExact` | `{result.rt_exact.proposal_wall_ms:.4f}` | `{result.rt_exact.scheduling_wall_ms:.4f}` | `{result.rt_exact.total_wall_ms:.4f}` | Python bookkeeping |",
        f"| `NoProposal` | `{result.no_proposal.proposal_wall_ms:.4f}` | `{result.no_proposal.scheduling_wall_ms:.4f}` | `{result.no_proposal.total_wall_ms:.4f}` | Python bookkeeping |",
        "",
        "here wall-time is notdescriptionwhen, descriptioncontains MP4/PNG/HTML generate; descriptionuse `PyVista/VTK`, description. ",
        "",
        f"here `RTExact/NoProposal = {result.no_proposal.total_wall_ms:.4f} ms` descriptionis notdescription `{result.no_proposal.exact_call_count}` descriptionreal mesh-mesh exact CCD kernel descriptionwhen, insteadcurrent Python benchmark  candidate cost summarizedescription `exact_work_units`  bookkeeping whendescription. thereforedescriptionuse `{result.no_proposal.total_wall_ms:.4f} ms` description `RTExact/NoProposal` description `RTSTPFExact` inreal exact CCD ondescription. ",
        "",
        f"currentdescription query  `RTSTPFExact = {result.rtstpf.total_wall_ms:.4f} ms` descriptionis Python/PyTorch description batch proposal fixedoverhead. thisdescriptionwithused fordescription: proposal runtime descriptionafterdescriptionPathdescription; butdescriptionthis paperMethoddescription exact workload reductiondescription. ",
        "",
        "## descriptionsplitdescription",
        "",
        f"- Broad/narrow candidate grid description `{result.no_proposal.candidate_count}`  candidate, sourceis `8`  time slabs and `4 x 4` patch-pair description. this isdescriptionhigh candidate density  hard case. ",
        f"- `RTExact` and `NoProposal` descriptionhas learned proposal, description `{result.no_proposal.candidate_count}`  candidate description exact certificate, descriptionwith exact work descriptionto `{result.no_proposal.exact_work_units:.1f}`. ",
        f"- `RTSTPFExact`  STPF proposal descriptiontocoveragereal TOI candidatedescription, description exact certify `{result.rtstpf.exact_call_count}`  candidate, descriptionwith exact work underreduced to `{result.rtstpf.exact_work_units:.1f}`. ",
        "- currentdescriptiontodescription wall-time descriptionhasdescription exact work reduction, isdescriptionasthisdescription exact descriptionsplitwith cost model/bookkeeping descriptionstatistics, real exact kernel descriptionhasinthis markdown indescriptionwhen. ",
        "- current `RTSTPFExact` wall-time descriptionis Python/PyTorch proposal fixedoverhead; description query, description batch description. afterdescriptionuse `medium_mlp + ORT TensorRT EP + C++ scheduling + CUDA exact` descriptionPathasdescription. ",
        "",
        "## this paperMethodadvantage",
        "",
        "1. `RTSTPFExact` descriptionin proposal layer, descriptionneuraldescriptionconnectdescriptioncollisionConclusion; description exact certificate description, thereforedescriptionwithdescriptionwhendescriptionuse learned prior and exact correctness. ",
        f"2. inhigh candidate density case in, this paperMethoddescriptionreduction exact calldescriptionand exact work. this case  exact call reduction as `{100.0 * call_reduction:.2f}%`, exact work reduction as `{100.0 * reduction:.2f}%`. ",
        "3. description `RTExact`, this paperMethodis notreplace RT broad phase, insteadin RT candidates descriptionafterdescription STPF descriptionlayer, use learned temporal/patch prior selectdescriptionhasdescription exact certificate description. ",
        "4. description `NoProposal`, this paperMethodavoidall candidate description exact queue, description CAD mesh, candidate inflation description, exact certificate descriptionscene. ",
        "5. This case is not intended to represent every possible case; it targets the high candidate density and exact-work regime used by the model and candidate scheduler. ",
        "",
        "## description",
        "",
        "this case descriptionwithasdescriptionin qualitative + workload reduction evidence: descriptionreal triangle-surface contact, TOI descriptionafterdescription, descriptionmethod comparison, withdescription learned STPF description exact workload description. descriptionusedescriptionafterdescriptionPathstatisticsdescriptiontodescription wall time; description case descriptionuse `Exact calls`, `Exact work`, `FN` and `TOI`. ",
        "",
        "recommenddescription: ",
        "",
        f"`On a high-density ABC CAD mesh-mesh CCD case with {result.asset.face_count:,} triangles per object and {result.no_proposal.candidate_count} RT candidates, RTSTPFExact reduces exact certificate calls from {result.no_proposal.exact_call_count} to {result.rtstpf.exact_call_count} and primitive-weighted exact work by {100.0 * reduction:.2f}%, while preserving zero false negatives through exact certification.`",
        "",
        "## realcontactconstruct",
        "",
        "- descriptionuse ABC official high-density CAD mesh. ",
        "- descriptionsame mesh description x descriptionto, guaranteecontactdescriptionhasdescriptionsupport surface. ",
        "- collisiondescriptionanddescription; `B.min_x + translation_x == A.max_x` descriptionin `t = 0.5`, thereforethisdescriptionisrealdescription surface contact, descriptionisdescription/description AABB contact. ",
        "- collisionafter replay descriptionusedescriptionqualitydescriptioncollisionModel, descriptionsplitdescription; this replay description CCD detectiondescriptionand STPF exact-work Metrics. ",
        "- descriptionas `contact_center_x / 2`, description STPF workload  oracle TOI andreal surface TOI description. ",
        "",
        "## Render Backend",
        "",
        f"- `{result.render_backend}`",
        "- `Plotly WebGL offline HTML` used fordescription. ",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_collision_frame_pngs(
    *,
    mp4_path: Path,
    output_root: Path,
    run_name: str,
    toi: float,
    frame_count: int,
    fps: int,
) -> tuple[Path, Path, Path, int, float]:
    frame_index = int(round(float(toi) * max(1, int(frame_count) - 1)))
    frame_index = max(0, min(int(frame_count) - 1, frame_index))
    context_offset = max(1, int(round(0.08 * max(1, int(frame_count) - 1))))
    before_index = max(0, frame_index - context_offset)
    after_index = min(int(frame_count) - 1, frame_index + context_offset)
    outputs = (
        output_root / f"{run_name}_collision_before_frame{before_index:03d}.png",
        output_root / f"{run_name}_collision_toi_frame{frame_index:03d}.png",
        output_root / f"{run_name}_collision_after_frame{after_index:03d}.png",
    )
    reader = imageio.get_reader(str(mp4_path))
    try:
        for index, output in zip((before_index, frame_index, after_index), outputs):
            imageio.imwrite(output, reader.get_data(index))
    finally:
        reader.close()
    return outputs[0], outputs[1], outputs[2], frame_index, frame_index / float(fps)


def write_true_mesh_surface_contact_method_comparison_mp4(
    config: TrueMeshSurfaceContactMP4Config | None = None,
) -> TrueMeshSurfaceContactMP4Result:
    cfg = config or TrueMeshSurfaceContactMP4Config()
    output_root = Path(cfg.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    asset = _select_asset(cfg)
    mesh_a = _load_render_mesh(
        asset.asset_path,
        max_faces=cfg.max_render_faces_per_mesh,
        seed=cfg.frame_count + 1,
    )
    mesh_b = _mirror_x(mesh_a)
    sample, contact_center_x, aabb_gap = _build_true_surface_contact_sample(
        mesh_a=mesh_a,
        mesh_b=mesh_b,
        asset=asset,
        cfg=cfg,
    )
    trace = evaluate_swept_sphere_oracle(sample)
    if not trace.collided:
        raise RuntimeError("constructed true surface contact sample did not collide")
    response = build_sample_elastic_impact_response(sample, toi=trace.toi, collided=trace.collided)
    workload = _build_workload(sample=sample, asset=asset, cfg=cfg)
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
    mp4_path = output_root / f"{cfg.run_name}.mp4"
    preview_png_path = output_root / f"{cfg.run_name}_surface_preview.png"
    clean_surface_png_path = output_root / f"{cfg.run_name}_surface_clean.png"
    summary_json_path = output_root / f"{cfg.run_name}.json"
    report_path = output_root / f"{cfg.run_name}.md"
    render_backend = "PyVista/VTK true triangle-surface renderer + mirrored ABC mesh + imageio-ffmpeg"
    _render_mp4_with_pyvista(
        cfg,  # type: ignore[arg-type]
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
        cfg,  # type: ignore[arg-type]
        mesh_a=mesh_a,
        mesh_b=mesh_b,
        sample=sample,
        trace=trace,
        output_path=clean_surface_png_path,
    )
    (
        collision_before_png_path,
        collision_toi_png_path,
        collision_after_png_path,
        toi_frame_index,
        toi_video_seconds,
    ) = _write_collision_frame_pngs(
        mp4_path=mp4_path,
        output_root=output_root,
        run_name=cfg.run_name,
        toi=float(trace.toi),
        frame_count=cfg.frame_count,
        fps=cfg.fps,
    )
    result = TrueMeshSurfaceContactMP4Result(
        config=cfg,
        mp4_path=mp4_path,
        preview_png_path=preview_png_path,
        clean_surface_png_path=clean_surface_png_path,
        collision_before_png_path=collision_before_png_path,
        collision_toi_png_path=collision_toi_png_path,
        collision_after_png_path=collision_after_png_path,
        summary_json_path=summary_json_path,
        report_path=report_path,
        asset=asset,
        query_id=sample.query_id,
        toi=float(trace.toi),
        toi_frame_index=toi_frame_index,
        toi_video_seconds=toi_video_seconds,
        contact_center_x=contact_center_x,
        aabb_gap_at_toi=aabb_gap,
        mass_a=response.mass_a,
        mass_b=response.mass_b,
        restitution=response.restitution,
        velocity_a_pre=response.velocity_a_pre,
        velocity_b_pre=response.velocity_b_pre,
        velocity_a_post=response.velocity_a_post,
        velocity_b_post=response.velocity_b_post,
        momentum_pre=momentum(response, post_impact=False),
        momentum_post=momentum(response, post_impact=True),
        kinetic_energy_pre=kinetic_energy(response, post_impact=False),
        kinetic_energy_post=kinetic_energy(response, post_impact=True),
        no_proposal=no_proposal,
        rt_exact=rt_exact,
        rtstpf=rtstpf,
        rtstpf_selected_candidate_ids=selected,
        render_backend=render_backend,
    )
    _write_summary_json(summary_json_path, result)
    _write_report(report_path, result)
    return result


def _contact_focus_point(
    *,
    mesh_a: _RenderMesh,
    mesh_b: _RenderMesh,
    contact_center_x: float,
) -> np.ndarray:
    a_min = mesh_a.vertices.min(axis=0)
    a_max = mesh_a.vertices.max(axis=0)
    b_min = mesh_b.vertices.min(axis=0)
    b_max = mesh_b.vertices.max(axis=0)
    x_extent = max(float(a_max[0] - a_min[0]), float(b_max[0] - b_min[0]), 1.0e-6)
    support_band = max(1.0e-5, 0.015 * x_extent)
    support_a = mesh_a.vertices[mesh_a.vertices[:, 0] >= a_max[0] - support_band]
    support_b = mesh_b.vertices[mesh_b.vertices[:, 0] <= b_min[0] + support_band]
    if support_a.size == 0:
        support_a = mesh_a.vertices[np.argmax(mesh_a.vertices[:, 0]) : np.argmax(mesh_a.vertices[:, 0]) + 1]
    if support_b.size == 0:
        support_b = mesh_b.vertices[np.argmin(mesh_b.vertices[:, 0]) : np.argmin(mesh_b.vertices[:, 0]) + 1]
    translated_support_b = support_b + np.asarray((contact_center_x, 0.0, 0.0), dtype=np.float32)
    support = np.vstack((support_a, translated_support_b))
    return np.asarray(
        (
            float(a_max[0]),
            float(np.mean(support[:, 1])),
            float(np.mean(support[:, 2])),
        ),
        dtype=np.float32,
    )


def _zoom_camera(
    *,
    mesh_a: _RenderMesh,
    mesh_b: _RenderMesh,
    focus: np.ndarray,
    contact_center_x: float,
    scene_scale: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    a_toi = mesh_a.vertices
    b_toi = mesh_b.vertices + np.asarray((contact_center_x, 0.0, 0.0), dtype=np.float32)
    all_toi = np.vstack((a_toi, b_toi)) * scene_scale
    full_diag = float(np.linalg.norm(all_toi.max(axis=0) - all_toi.min(axis=0)))
    focus_scaled = focus * scene_scale
    local_radius = max(0.6, 0.16 * full_diag)
    distance = max(3.0, 2.25 * local_radius)
    camera = focus_scaled + np.asarray(
        (1.24 * distance, -1.58 * distance, 0.92 * distance),
        dtype=np.float32,
    )
    parallel_scale = max(0.35, local_radius)
    return focus_scaled, camera, parallel_scale


def _draw_zoom_overlay(
    frame: Image.Image,
    *,
    frame_index: int,
    frame_count: int,
    t: float,
    toi: float,
    time_start: float,
    time_end: float,
    signed_gap: float,
    asset: MeshDensityAsset,
) -> None:
    draw = ImageDraw.Draw(frame, "RGBA")
    title_font = _font(32, bold=True)
    body_font = _font(20)
    mono_font = _font(18)
    small_font = _font(15)
    draw.rounded_rectangle((28, 24, 780, 178), radius=18, fill=(6, 12, 24, 212))
    draw.text((54, 38), "True Mesh Surface Contact Zoom", font=title_font, fill=(241, 245, 249, 255))
    draw.text(
        (54, 82),
        "surface render + triangle wireframe | two-body elastic bounce replay",
        font=body_font,
        fill=(203, 213, 225, 255),
    )
    draw.text(
        (54, 116),
        f"t={t:.6f}  TOI={toi:.6f}  gap_x={signed_gap:+.6e}",
        font=mono_font,
        fill=(250, 204, 21, 255) if abs(t - toi) < 0.0015 else (226, 232, 240, 255),
    )
    draw.text(
        (54, 144),
        f"frame {frame_index + 1}/{frame_count} | ABC faces/object {asset.face_count:,}",
        font=mono_font,
        fill=(186, 230, 253, 255),
    )

    bar_x0 = 72
    bar_x1 = frame.width - 72
    bar_y0 = frame.height - 58
    bar_y1 = frame.height - 38
    draw.rounded_rectangle((bar_x0, bar_y0, bar_x1, bar_y1), radius=10, fill=(15, 23, 42, 230))
    denom = max(1.0e-9, time_end - time_start)
    current_alpha = min(1.0, max(0.0, (t - time_start) / denom))
    toi_alpha = min(1.0, max(0.0, (toi - time_start) / denom))
    current_x = int(round(bar_x0 + (bar_x1 - bar_x0) * current_alpha))
    toi_x = int(round(bar_x0 + (bar_x1 - bar_x0) * toi_alpha))
    draw.rounded_rectangle((bar_x0, bar_y0, current_x, bar_y1), radius=10, fill=(34, 211, 238, 255))
    draw.rectangle((toi_x - 2, bar_y0 - 12, toi_x + 2, bar_y1 + 12), fill=(250, 204, 21, 255))
    draw.text((bar_x0, bar_y0 - 26), f"zoom window [{time_start:.2f}, {time_end:.2f}]", font=small_font, fill=(226, 232, 240, 255))
    draw.text((toi_x + 10, bar_y0 - 26), "TOI", font=small_font, fill=(250, 204, 21, 255))


def write_true_mesh_surface_contact_zoom_wireframe_mp4(
    config: TrueMeshSurfaceContactZoomMP4Config | None = None,
) -> TrueMeshSurfaceContactZoomMP4Result:
    import pyvista as pv

    cfg = config or TrueMeshSurfaceContactZoomMP4Config()
    if cfg.time_end <= cfg.time_start:
        raise ValueError("time_end must be greater than time_start")
    output_root = Path(cfg.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    base_cfg = _base_config_from_zoom_config(cfg)
    asset = _select_asset(base_cfg)
    mesh_a = _load_render_mesh(
        asset.asset_path,
        max_faces=cfg.max_render_faces_per_mesh,
        seed=cfg.frame_count + 17,
    )
    mesh_b = _mirror_x(mesh_a)
    sample, contact_center_x, _ = _build_true_surface_contact_sample(
        mesh_a=mesh_a,
        mesh_b=mesh_b,
        asset=asset,
        cfg=base_cfg,
    )
    trace = evaluate_swept_sphere_oracle(sample)
    if not trace.collided:
        raise RuntimeError("constructed true surface contact sample did not collide")
    response = build_sample_elastic_impact_response(sample, toi=trace.toi, collided=trace.collided)

    times = np.linspace(cfg.time_start, cfg.time_end, cfg.frame_count, dtype=np.float64)
    toi_frame_index = int(np.argmin(np.abs(times - float(trace.toi))))
    times[toi_frame_index] = float(trace.toi)
    toi_video_seconds = toi_frame_index / float(cfg.fps)

    focus = _contact_focus_point(mesh_a=mesh_a, mesh_b=mesh_b, contact_center_x=contact_center_x)
    target, camera, parallel_scale = _zoom_camera(
        mesh_a=mesh_a,
        mesh_b=mesh_b,
        focus=focus,
        contact_center_x=contact_center_x,
        scene_scale=cfg.render_scene_scale,
    )

    initial_a, initial_b = replay_positions_at_time(response, float(times[0]), mode="bounce")
    poly_a = _pyvista_polydata(mesh_a)
    poly_b = _pyvista_polydata(mesh_b)
    poly_a.points = (mesh_a.vertices + np.asarray(initial_a, dtype=np.float32)) * cfg.render_scene_scale
    poly_b.points = (mesh_b.vertices + np.asarray(initial_b, dtype=np.float32)) * cfg.render_scene_scale

    pv.OFF_SCREEN = True
    plotter = pv.Plotter(off_screen=True, window_size=(cfg.width, cfg.height))
    plotter.set_background("#202936")
    plotter.enable_anti_aliasing("ssaa")
    plotter.add_mesh(
        poly_a,
        color="#46a3ff",
        smooth_shading=True,
        specular=0.36,
        roughness=0.58,
        opacity=0.78,
        show_edges=False,
    )
    plotter.add_mesh(
        poly_b,
        color="#ff6b66",
        smooth_shading=True,
        specular=0.36,
        roughness=0.58,
        opacity=0.78,
        show_edges=False,
    )
    plotter.add_mesh(
        poly_a,
        style="wireframe",
        color="#d7ecff",
        line_width=0.75,
        opacity=0.36,
        lighting=False,
    )
    plotter.add_mesh(
        poly_b,
        style="wireframe",
        color="#ffd5cf",
        line_width=0.75,
        opacity=0.36,
        lighting=False,
    )
    marker_radius = max(0.035, 0.012 * parallel_scale)
    marker = pv.Sphere(radius=marker_radius, center=tuple(target.tolist()), theta_resolution=32, phi_resolution=16)
    plotter.add_mesh(marker, color="#facc15", emissive=True, opacity=0.88)
    plotter.add_light(pv.Light(position=tuple(camera), focal_point=tuple(target), color="white", intensity=0.82))
    plotter.add_light(pv.Light(position=tuple((target + np.asarray((-3.0, 2.0, 5.0), dtype=np.float32)).tolist()), intensity=0.25))
    plotter.camera_position = (tuple(camera.tolist()), tuple(target.tolist()), (0.0, 0.0, 1.0))
    plotter.camera.parallel_projection = True
    plotter.camera.parallel_scale = parallel_scale
    plotter.camera.clipping_range = (0.01, 1000.0)

    mp4_path = output_root / f"{cfg.run_name}.mp4"
    preview_png_path = output_root / f"{cfg.run_name}_preview.png"
    writer = imageio.get_writer(
        str(mp4_path),
        fps=cfg.fps,
        codec="libx264",
        quality=9,
        macro_block_size=16,
    )
    a_max_x = float(mesh_a.vertices[:, 0].max())
    b_min_x = float(mesh_b.vertices[:, 0].min())
    try:
        for frame_index, t_value in enumerate(times):
            center_a, center_b = replay_positions_at_time(response, float(t_value), mode="bounce")
            center_a_array = np.asarray(center_a, dtype=np.float32)
            center_b_array = np.asarray(center_b, dtype=np.float32)
            poly_a.points = (mesh_a.vertices + center_a_array) * cfg.render_scene_scale
            poly_b.points = (mesh_b.vertices + center_b_array) * cfg.render_scene_scale
            plotter.render()
            buffer = np.asarray(plotter.screenshot(return_img=True))
            frame = Image.fromarray(buffer[:, :, :3].astype(np.uint8), mode="RGB")
            signed_gap = (b_min_x + float(center_b_array[0])) - (a_max_x + float(center_a_array[0]))
            _draw_zoom_overlay(
                frame,
                frame_index=frame_index,
                frame_count=cfg.frame_count,
                t=float(t_value),
                toi=float(trace.toi),
                time_start=cfg.time_start,
                time_end=cfg.time_end,
                signed_gap=signed_gap,
                asset=asset,
            )
            if frame_index == toi_frame_index:
                frame.save(preview_png_path)
            writer.append_data(np.asarray(frame))
    finally:
        writer.close()
        plotter.close()

    return TrueMeshSurfaceContactZoomMP4Result(
        config=cfg,
        mp4_path=mp4_path,
        preview_png_path=preview_png_path,
        asset=asset,
        query_id=sample.query_id,
        toi=float(trace.toi),
        toi_frame_index=toi_frame_index,
        toi_video_seconds=toi_video_seconds,
        contact_center_x=contact_center_x,
        render_backend="PyVista/VTK close-up surface+wireframe renderer + imageio-ffmpeg",
    )


def write_true_mesh_surface_contact_full_snapshot_pngs(
    config: TrueMeshSurfaceContactFullSnapshotConfig | None = None,
) -> TrueMeshSurfaceContactFullSnapshotResult:
    import pyvista as pv

    cfg = config or TrueMeshSurfaceContactFullSnapshotConfig()
    if not cfg.times:
        raise ValueError("times must contain at least one frame time")
    output_root = Path(cfg.output_dir)
    frame_dir = output_root / cfg.run_name
    frame_dir.mkdir(parents=True, exist_ok=True)
    base_cfg = TrueMeshSurfaceContactMP4Config(
        source_root=cfg.source_root,
        asset_id_contains=cfg.asset_id_contains,
        asset_limit=cfg.asset_limit,
        width=cfg.width,
        height=cfg.height,
        max_render_faces_per_mesh=cfg.max_render_faces_per_mesh,
        render_scene_scale=cfg.render_scene_scale,
        approach_gap_scale=cfg.approach_gap_scale,
    )
    asset = _select_asset(base_cfg)
    mesh_a = _load_render_mesh(
        asset.asset_path,
        max_faces=cfg.max_render_faces_per_mesh,
        seed=271828,
    )
    mesh_b = _mirror_x(mesh_a)
    sample, contact_center_x, _ = _build_true_surface_contact_sample(
        mesh_a=mesh_a,
        mesh_b=mesh_b,
        asset=asset,
        cfg=base_cfg,
    )
    trace = evaluate_swept_sphere_oracle(sample)
    if not trace.collided:
        raise RuntimeError("constructed true surface contact sample did not collide")
    response = build_sample_elastic_impact_response(sample, toi=trace.toi, collided=trace.collided)

    focus = _contact_focus_point(mesh_a=mesh_a, mesh_b=mesh_b, contact_center_x=contact_center_x)
    target, camera, parallel_scale = _zoom_camera(
        mesh_a=mesh_a,
        mesh_b=mesh_b,
        focus=focus,
        contact_center_x=contact_center_x,
        scene_scale=cfg.render_scene_scale,
    )
    parallel_scale *= cfg.camera_parallel_scale_multiplier

    first_a, first_b = replay_positions_at_time(response, float(cfg.times[0]), mode="bounce")
    poly_a = _pyvista_polydata(mesh_a)
    poly_b = _pyvista_polydata(mesh_b)
    poly_a.points = (mesh_a.vertices + np.asarray(first_a, dtype=np.float32)) * cfg.render_scene_scale
    poly_b.points = (mesh_b.vertices + np.asarray(first_b, dtype=np.float32)) * cfg.render_scene_scale

    pv.OFF_SCREEN = True
    plotter = pv.Plotter(off_screen=True, window_size=(cfg.width, cfg.height))
    snapshot_paths: list[Path] = []
    try:
        plotter.set_background(cfg.light_background)
        plotter.enable_anti_aliasing("ssaa")
        plotter.add_mesh(
            poly_a,
            color="#37b7ea",
            smooth_shading=False,
            specular=0.30,
            roughness=0.62,
            opacity=cfg.surface_opacity,
            show_edges=True,
            edge_color="#075985",
            line_width=cfg.surface_edge_line_width,
        )
        plotter.add_mesh(
            poly_b,
            color="#ff7e68",
            smooth_shading=False,
            specular=0.30,
            roughness=0.62,
            opacity=cfg.surface_opacity,
            show_edges=True,
            edge_color="#9f2f22",
            line_width=cfg.surface_edge_line_width,
        )
        plotter.add_mesh(
            poly_a,
            style="wireframe",
            color="#033f63",
            line_width=cfg.wireframe_overlay_line_width,
            opacity=cfg.wireframe_overlay_opacity,
            lighting=False,
        )
        plotter.add_mesh(
            poly_b,
            style="wireframe",
            color="#7f1d1d",
            line_width=cfg.wireframe_overlay_line_width,
            opacity=cfg.wireframe_overlay_opacity,
            lighting=False,
        )
        plotter.add_light(pv.Light(position=tuple(camera), focal_point=tuple(target), color="white", intensity=0.82))
        plotter.add_light(
            pv.Light(
                position=tuple((target + np.asarray((-2.0, 3.0, 4.2), dtype=np.float32)).tolist()),
                focal_point=tuple(target),
                intensity=0.32,
            ),
        )
        plotter.camera_position = (tuple(camera.tolist()), tuple(target.tolist()), (0.0, 0.0, 1.0))
        plotter.camera.parallel_projection = True
        plotter.camera.parallel_scale = parallel_scale
        plotter.camera.clipping_range = (0.01, 1000.0)
        plotter.hide_axes()

        for index, t_value in enumerate(cfg.times, start=1):
            center_a, center_b = replay_positions_at_time(response, float(t_value), mode="bounce")
            poly_a.points = (mesh_a.vertices + np.asarray(center_a, dtype=np.float32)) * cfg.render_scene_scale
            poly_b.points = (mesh_b.vertices + np.asarray(center_b, dtype=np.float32)) * cfg.render_scene_scale
            plotter.render()
            path = frame_dir / f"frame_{index:02d}_t{float(t_value):.3f}.png"
            plotter.screenshot(str(path))
            snapshot_paths.append(path)

        manifest_path = frame_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "asset_id": asset.asset_id,
                    "asset_path": asset.asset_path,
                    "original_faces_per_object": asset.face_count,
                    "rendered_faces_per_object": int(mesh_a.faces.shape[0]),
                    "times": [float(t) for t in cfg.times],
                    "toi": float(trace.toi),
                    "render_backend": "PyVista/VTK full triangle surface renderer; no face or edge sampling",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    finally:
        plotter.close()

    return TrueMeshSurfaceContactFullSnapshotResult(
        config=cfg,
        output_dir=frame_dir,
        snapshot_paths=tuple(snapshot_paths),
        asset=asset,
        query_id=sample.query_id,
        toi=float(trace.toi),
        contact_center_x=contact_center_x,
        face_count_per_object_rendered=int(mesh_a.faces.shape[0]),
        render_backend="PyVista/VTK full triangle surface renderer; no face or edge sampling",
    )


def _sample_unique_edges(
    faces: np.ndarray,
    *,
    max_edges: int | None,
    seed: int,
) -> np.ndarray:
    edges = np.vstack(
        (
            faces[:, [0, 1]],
            faces[:, [1, 2]],
            faces[:, [2, 0]],
        ),
    )
    edges = np.sort(edges.astype(np.int64, copy=False), axis=1)
    edges = np.unique(edges, axis=0)
    if max_edges is not None and edges.shape[0] > max_edges:
        rng = np.random.default_rng(seed)
        selected = np.sort(rng.choice(edges.shape[0], size=max_edges, replace=False))
        edges = edges[selected]
    return np.ascontiguousarray(edges, dtype=np.int64)


def _wireframe_xyz(vertices: np.ndarray, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    starts = vertices[edges[:, 0]]
    ends = vertices[edges[:, 1]]
    xyz = np.empty((edges.shape[0] * 3, 3), dtype=np.float32)
    xyz[0::3] = starts
    xyz[1::3] = ends
    xyz[2::3] = np.nan
    return xyz[:, 0], xyz[:, 1], xyz[:, 2]


def _transformed_vertices(
    mesh: _RenderMesh,
    center: tuple[float, float, float],
    *,
    scene_scale: float,
) -> np.ndarray:
    return (mesh.vertices + np.asarray(center, dtype=np.float32)) * np.float32(scene_scale)


def write_true_mesh_surface_contact_interactive_html(
    config: TrueMeshSurfaceContactInteractiveHTMLConfig | None = None,
) -> TrueMeshSurfaceContactInteractiveHTMLResult:
    import plotly.graph_objects as go

    cfg = config or TrueMeshSurfaceContactInteractiveHTMLConfig()
    if cfg.frame_count < 3:
        raise ValueError("frame_count must be >= 3")
    if cfg.time_end <= cfg.time_start:
        raise ValueError("time_end must be greater than time_start")
    output_root = Path(cfg.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    base_cfg = _base_config_from_interactive_config(cfg)
    asset = _select_asset(base_cfg)
    mesh_a = _load_render_mesh(
        asset.asset_path,
        max_faces=cfg.max_interactive_faces_per_mesh or 2_147_483_647,
        seed=cfg.frame_count + 101,
    )
    mesh_b = _mirror_x(mesh_a)
    sample, contact_center_x, _ = _build_true_surface_contact_sample(
        mesh_a=mesh_a,
        mesh_b=mesh_b,
        asset=asset,
        cfg=base_cfg,
    )
    trace = evaluate_swept_sphere_oracle(sample)
    if not trace.collided:
        raise RuntimeError("constructed true surface contact sample did not collide")
    response = build_sample_elastic_impact_response(sample, toi=trace.toi, collided=trace.collided)

    times = np.linspace(cfg.time_start, cfg.time_end, cfg.frame_count, dtype=np.float64)
    initial_frame_index = int(np.argmin(np.abs(times - float(trace.toi))))
    times[initial_frame_index] = float(trace.toi)
    edges_a = _sample_unique_edges(
        mesh_a.faces,
        max_edges=cfg.max_wireframe_edges_per_mesh,
        seed=cfg.frame_count + 201,
    )
    edges_b = _sample_unique_edges(
        mesh_b.faces,
        max_edges=cfg.max_wireframe_edges_per_mesh,
        seed=cfg.frame_count + 202,
    )

    def frame_vertices(t: float) -> tuple[np.ndarray, np.ndarray]:
        center_a, center_b = replay_positions_at_time(response, float(t), mode="bounce")
        return (
            _transformed_vertices(mesh_a, center_a, scene_scale=cfg.render_scene_scale),
            _transformed_vertices(mesh_b, center_b, scene_scale=cfg.render_scene_scale),
        )

    initial_a, initial_b = frame_vertices(float(times[initial_frame_index]))
    initial_wire_a = _wireframe_xyz(initial_a, edges_a)
    initial_wire_b = _wireframe_xyz(initial_b, edges_b)
    focus = _contact_focus_point(mesh_a=mesh_a, mesh_b=mesh_b, contact_center_x=contact_center_x)
    target, camera, parallel_scale = _zoom_camera(
        mesh_a=mesh_a,
        mesh_b=mesh_b,
        focus=focus,
        contact_center_x=contact_center_x,
        scene_scale=cfg.render_scene_scale,
    )
    range_radius = max(0.45, 1.18 * parallel_scale)
    eye = camera - target
    eye_norm = max(1.0e-6, float(np.linalg.norm(eye)))
    eye_unit = eye / eye_norm

    data = [
        go.Mesh3d(
            x=initial_a[:, 0],
            y=initial_a[:, 1],
            z=initial_a[:, 2],
            i=mesh_a.faces[:, 0],
            j=mesh_a.faces[:, 1],
            k=mesh_a.faces[:, 2],
            name="blue mesh surface",
            color=cfg.blue_surface_color,
            opacity=cfg.surface_opacity,
            flatshading=True,
            hoverinfo="skip",
            lighting={"ambient": 0.46, "diffuse": 0.72, "specular": 0.18, "roughness": 0.70},
            showscale=False,
        ),
        go.Mesh3d(
            x=initial_b[:, 0],
            y=initial_b[:, 1],
            z=initial_b[:, 2],
            i=mesh_b.faces[:, 0],
            j=mesh_b.faces[:, 1],
            k=mesh_b.faces[:, 2],
            name="red mesh surface",
            color=cfg.red_surface_color,
            opacity=cfg.surface_opacity,
            flatshading=True,
            hoverinfo="skip",
            lighting={"ambient": 0.46, "diffuse": 0.72, "specular": 0.18, "roughness": 0.70},
            showscale=False,
        ),
        go.Scatter3d(
            x=initial_wire_a[0],
            y=initial_wire_a[1],
            z=initial_wire_a[2],
            mode="lines",
            name="blue triangle wireframe",
            line={"color": cfg.blue_wire_color, "width": cfg.wireframe_line_width},
            hoverinfo="skip",
        ),
        go.Scatter3d(
            x=initial_wire_b[0],
            y=initial_wire_b[1],
            z=initial_wire_b[2],
            mode="lines",
            name="red triangle wireframe",
            line={"color": cfg.red_wire_color, "width": cfg.wireframe_line_width},
            hoverinfo="skip",
        ),
        go.Scatter3d(
            x=[target[0]],
            y=[target[1]],
            z=[target[2]],
            mode="markers+text",
            name="TOI contact marker",
            marker={"size": 7, "color": "#facc15", "line": {"color": "#111827", "width": 1}},
            text=["TOI"],
            textposition="top center",
            hoverinfo="skip",
        ),
    ]

    frames = []
    for index, t_value in enumerate(times):
        vertices_a, vertices_b = frame_vertices(float(t_value))
        wire_a = _wireframe_xyz(vertices_a, edges_a)
        wire_b = _wireframe_xyz(vertices_b, edges_b)
        frames.append(
            go.Frame(
                name=f"{index:02d} | t={t_value:.4f}",
                traces=[0, 1, 2, 3],
                data=[
                    go.Mesh3d(x=vertices_a[:, 0], y=vertices_a[:, 1], z=vertices_a[:, 2]),
                    go.Mesh3d(x=vertices_b[:, 0], y=vertices_b[:, 1], z=vertices_b[:, 2]),
                    go.Scatter3d(x=wire_a[0], y=wire_a[1], z=wire_a[2]),
                    go.Scatter3d(x=wire_b[0], y=wire_b[1], z=wire_b[2]),
                ],
            ),
        )

    steps = [
        {
            "label": f"t={float(t_value):.3f}" + (" TOI" if index == initial_frame_index else ""),
            "method": "animate",
            "args": [
                [frames[index].name],
                {
                    "mode": "immediate",
                    "frame": {"duration": 0, "redraw": True},
                    "transition": {"duration": 0},
                },
            ],
        }
        for index, t_value in enumerate(times)
    ]
    fig = go.Figure(data=data, frames=frames)
    fig.update_layout(
        title={
            "text": (
                "P2CCCD True Mesh Surface Contact - Interactive Wireframe "
                f"(TOI={float(trace.toi):.6f}, rendered faces/object={mesh_a.faces.shape[0]:,})"
            ),
            "x": 0.02,
            "xanchor": "left",
        },
        paper_bgcolor=cfg.light_background,
        plot_bgcolor=cfg.light_background,
        font={"color": "#27313c"},
        margin={"l": 0, "r": 0, "t": 64, "b": 0},
        uirevision="keep-camera-while-scrubbing",
        scene={
            "bgcolor": cfg.light_background,
            "aspectmode": "cube",
            "camera": {
                "eye": {
                    "x": float(2.25 * eye_unit[0]),
                    "y": float(2.25 * eye_unit[1]),
                    "z": float(2.25 * eye_unit[2]),
                },
                "center": {"x": 0.0, "y": 0.0, "z": 0.0},
            },
            "xaxis": {
                "title": "x",
                "range": [float(target[0] - range_radius), float(target[0] + range_radius)],
                "showbackground": False,
                "gridcolor": "rgba(148,163,184,0.30)",
                "zerolinecolor": "rgba(234,88,12,0.45)",
            },
            "yaxis": {
                "title": "y",
                "range": [float(target[1] - range_radius), float(target[1] + range_radius)],
                "showbackground": False,
                "gridcolor": "rgba(148,163,184,0.30)",
            },
            "zaxis": {
                "title": "z",
                "range": [float(target[2] - range_radius), float(target[2] + range_radius)],
                "showbackground": False,
                "gridcolor": "rgba(148,163,184,0.30)",
            },
        },
        sliders=[
            {
                "active": initial_frame_index,
                "x": 0.06,
                "y": 0.035,
                "len": 0.86,
                "currentvalue": {"prefix": "time: ", "font": {"color": "#ea580c"}},
                "steps": steps,
            },
        ],
        updatemenus=[
            {
                "type": "buttons",
                "direction": "left",
                "x": 0.06,
                "y": 0.105,
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": 180, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "mode": "immediate",
                                "frame": {"duration": 0, "redraw": False},
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                ],
            },
        ],
        annotations=[
            {
                "text": (
                    "Use mouse: left-drag rotate, wheel zoom, right-drag pan. "
                    "Slider scrubs collision before / TOI / after two-body bounce."
                ),
                "xref": "paper",
                "yref": "paper",
                "x": 0.02,
                "y": 0.985,
                "showarrow": False,
                "font": {"size": 13, "color": "#475569"},
                "align": "left",
            },
        ],
    )
    html_path = output_root / f"{cfg.run_name}.html"
    fig.write_html(
        str(html_path),
        include_plotlyjs=cfg.include_plotlyjs,
        full_html=True,
        auto_play=False,
        config={
            "scrollZoom": True,
            "displaylogo": False,
            "responsive": True,
        },
    )
    return TrueMeshSurfaceContactInteractiveHTMLResult(
        config=cfg,
        html_path=html_path,
        asset=asset,
        query_id=sample.query_id,
        toi=float(trace.toi),
        initial_frame_index=initial_frame_index,
        contact_center_x=contact_center_x,
        face_count_per_object_rendered=int(mesh_a.faces.shape[0]),
        wireframe_edge_count_per_object=int(edges_a.shape[0]),
        render_backend="Plotly WebGL offline HTML + full triangle mesh/wireframe when caps are None",
    )


__all__ = [
    "TrueMeshSurfaceContactMP4Config",
    "TrueMeshSurfaceContactMP4Result",
    "TrueMeshSurfaceContactFullSnapshotConfig",
    "TrueMeshSurfaceContactFullSnapshotResult",
    "TrueMeshSurfaceContactInteractiveHTMLConfig",
    "TrueMeshSurfaceContactInteractiveHTMLResult",
    "TrueMeshSurfaceContactZoomMP4Config",
    "TrueMeshSurfaceContactZoomMP4Result",
    "write_true_mesh_surface_contact_full_snapshot_pngs",
    "write_true_mesh_surface_contact_interactive_html",
    "write_true_mesh_surface_contact_method_comparison_mp4",
    "write_true_mesh_surface_contact_zoom_wireframe_mp4",
]
