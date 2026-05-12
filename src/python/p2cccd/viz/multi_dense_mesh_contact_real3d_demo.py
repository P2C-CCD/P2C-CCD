from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Sequence

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import trimesh

from p2cccd.bench.high_density_mesh_training_benchmark import (
    MeshDensityPair,
    _load_abc_assets,
    _load_fusion360_assets,
    _load_thingi10k_assets,
    _make_pairs,
)
from p2cccd.bench.multi_dense_mesh_contact_pairs import (
    MultiDenseMeshContactPairsConfig,
    _load_large_face_abc_assets,
    _make_cross_pairs,
    _rename_assets,
    _split_pairs,
)


@dataclass(frozen=True, slots=True)
class MultiDenseMeshContactReal3DConfig:
    output_dir: str = "src/MyDemo/paper_multi_dense_mesh_contact_pairs_run_id"
    benchmark_json: str = "src/benchmark/multi_dense_mesh_contact_pairs_run_id.json"
    cases: tuple[str, ...] = (
        "ABC-largeface-intra",
        "ABC-topface-intra",
        "ABCtop-Fusion360-cross",
        "ABCtop-Thingi10K-cross",
        "Fusion360-intra",
        "Fusion360-Thingi10K-cross",
        "Thingi10K-intra",
    )
    width: int = 1280
    height: int = 720
    frame_count: int = 96
    fps: int = 24
    max_faces_per_mesh: int = 36_000
    target_diagonal: float = 2.0
    approach_distance: float = 1.6
    rebound_distance: float = 1.4


@dataclass(frozen=True, slots=True)
class Real3DCaseArtifact:
    case_name: str
    mp4_path: Path
    preview_png_path: Path
    asset_a_path: str
    asset_b_path: str
    asset_a_faces: int
    asset_b_faces: int
    rendered_faces_a: int
    rendered_faces_b: int


@dataclass(frozen=True, slots=True)
class MultiDenseMeshContactReal3DResult:
    output_dir: Path
    artifacts: tuple[Real3DCaseArtifact, ...]
    summary_json_path: Path
    contact_sheet_path: Path


@dataclass(frozen=True, slots=True)
class _RenderMesh:
    vertices: np.ndarray
    faces: np.ndarray
    original_face_count: int
    original_vertex_count: int
    scale: float


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


def _slug(value: str) -> str:
    return value.lower().replace("+", "plus").replace("/", "-").replace("_", "-").replace(" ", "-")


def _load_normalized_mesh(path: str | Path, *, max_faces: int, target_diagonal: float, seed: int) -> _RenderMesh:
    loaded = trimesh.load(str(path), force="mesh", process=False)
    if isinstance(loaded, trimesh.Scene):
        loaded = trimesh.util.concatenate(tuple(loaded.dump()))
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(f"unsupported mesh type for {path}: {type(loaded)!r}")
    mesh = loaded.copy()
    mesh.merge_vertices()
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if faces.shape[0] == 0:
        raise ValueError(f"mesh has no faces: {path}")
    original_face_count = int(faces.shape[0])
    original_vertex_count = int(vertices.shape[0])
    if faces.shape[0] > max_faces:
        rng = np.random.default_rng(seed)
        picked = np.sort(rng.choice(faces.shape[0], size=max_faces, replace=False))
        faces = faces[picked]
        used = np.unique(faces.reshape(-1))
        remap = np.full(vertices.shape[0], -1, dtype=np.int64)
        remap[used] = np.arange(used.shape[0], dtype=np.int64)
        vertices = vertices[used]
        faces = remap[faces]
    center = 0.5 * (vertices.min(axis=0) + vertices.max(axis=0))
    vertices = vertices - center
    diagonal = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
    scale = target_diagonal / max(1.0e-9, diagonal)
    vertices = vertices * scale
    return _RenderMesh(
        vertices=np.ascontiguousarray(vertices, dtype=np.float32),
        faces=np.ascontiguousarray(faces, dtype=np.int32),
        original_face_count=original_face_count,
        original_vertex_count=original_vertex_count,
        scale=scale,
    )


def _polydata(mesh: _RenderMesh):
    import pyvista as pv

    faces = np.empty((mesh.faces.shape[0], 4), dtype=np.int64)
    faces[:, 0] = 3
    faces[:, 1:] = mesh.faces.astype(np.int64, copy=False)
    return pv.PolyData(mesh.vertices.copy(), faces.reshape(-1))


def _case_metrics(data: dict, case_name: str) -> dict:
    for row in data["case_results"]:
        if row["case_name"] == case_name:
            return row
    raise KeyError(case_name)


def _build_eval_pairs() -> dict[str, tuple[MeshDensityPair, ...]]:
    cfg = MultiDenseMeshContactPairsConfig()
    abc_top = _rename_assets(_load_abc_assets(Path(cfg.abc_root), cfg.asset_limit_per_source), "ABC top-face")
    abc_large = _load_large_face_abc_assets(cfg)
    fusion = _rename_assets(_load_fusion360_assets(Path(cfg.fusion360_root), cfg.asset_limit_per_source), "Fusion 360 Gallery")
    thingi = _rename_assets(_load_thingi10k_assets(Path(cfg.thingi10k_root), cfg.asset_limit_per_source), "Thingi10K")
    pairs_by_case = {
        "ABC-largeface-intra": _make_pairs(_rename_assets(abc_large, "ABC-largeface-intra"), cfg.pair_limit_per_case),
        "ABC-topface-intra": _make_pairs(_rename_assets(abc_top, "ABC-topface-intra"), cfg.pair_limit_per_case),
        "Fusion360-intra": _make_pairs(_rename_assets(fusion, "Fusion360-intra"), cfg.pair_limit_per_case),
        "Thingi10K-intra": _make_pairs(_rename_assets(thingi, "Thingi10K-intra"), cfg.pair_limit_per_case),
        "ABCtop-Fusion360-cross": _make_cross_pairs("ABCtop-Fusion360-cross", abc_top, fusion, limit=cfg.pair_limit_per_case),
        "ABCtop-Thingi10K-cross": _make_cross_pairs("ABCtop-Thingi10K-cross", abc_top, thingi, limit=cfg.pair_limit_per_case),
        "Fusion360-Thingi10K-cross": _make_cross_pairs("Fusion360-Thingi10K-cross", fusion, thingi, limit=cfg.pair_limit_per_case),
    }
    eval_pairs: dict[str, tuple[MeshDensityPair, ...]] = {}
    for case_name, pairs in pairs_by_case.items():
        _, eval_split = _split_pairs(pairs, train_fraction=cfg.train_fraction, seed=cfg.seed + len(case_name))
        eval_pairs[case_name] = eval_split
    return eval_pairs


def _centers_for_contact(mesh_a: _RenderMesh, mesh_b: _RenderMesh, t: float, cfg: MultiDenseMeshContactReal3DConfig) -> tuple[np.ndarray, np.ndarray]:
    width_a = float(mesh_a.vertices[:, 0].max() - mesh_a.vertices[:, 0].min())
    width_b = float(mesh_b.vertices[:, 0].max() - mesh_b.vertices[:, 0].min())
    contact_sep = 0.5 * (width_a + width_b)
    if t <= 0.5:
        u = t / 0.5
        sep = contact_sep + cfg.approach_distance * (1.0 - u)
    else:
        u = (t - 0.5) / 0.5
        sep = contact_sep + cfg.rebound_distance * u
    z_bob = 0.035 * np.sin(2.0 * np.pi * t)
    return (
        np.asarray([-0.5 * sep, 0.0, -0.015 * z_bob], dtype=np.float32),
        np.asarray([0.5 * sep, 0.0, z_bob], dtype=np.float32),
    )


def _draw_overlay(frame: Image.Image, *, case: dict, pair: MeshDensityPair, t: float) -> Image.Image:
    draw = ImageDraw.Draw(frame, "RGBA")
    draw.rounded_rectangle((18, 16, frame.width - 18, 112), radius=16, fill=(8, 13, 23, 225), outline=(51, 65, 85), width=1)
    draw.text((42, 32), case["case_name"], font=_font(28, bold=True), fill=(226, 232, 240))
    draw.text(
        (42, 72),
        f"{pair.asset_a.source_name}  x  {pair.asset_b.source_name} | real triangle mesh replay | t={t:.3f}",
        font=_font(17),
        fill=(203, 213, 225),
    )
    draw.text(
        (frame.width - 620, 34),
        f"RTSTPFExact calls {case['trained_stpf']['exact_call_count']:,}/{case['eval_candidate_count']:,}",
        font=_font(17),
        fill=(34, 197, 94),
    )
    draw.text(
        (frame.width - 620, 68),
        f"work reduction {100.0 * case['trained_exact_work_reduction_vs_no_proposal']:.2f}% | FN {case['trained_stpf']['fn_count']}",
        font=_font(17),
        fill=(226, 232, 240),
    )
    draw.rectangle((36, frame.height - 42, frame.width - 36, frame.height - 28), fill=(30, 41, 59, 230))
    draw.rectangle((36, frame.height - 42, int(36 + (frame.width - 72) * t), frame.height - 28), fill=(34, 211, 238, 255))
    draw.text(
        (42, frame.height - 72),
        "visualization-normalized real meshes; trajectory aligns AABB surface contact at t=0.5 and bounces after contact",
        font=_font(14),
        fill=(148, 163, 184),
    )
    return frame


def _render_case(
    cfg: MultiDenseMeshContactReal3DConfig,
    *,
    case: dict,
    pair: MeshDensityPair,
    mp4_path: Path,
    preview_path: Path,
) -> Real3DCaseArtifact:
    import pyvista as pv

    mesh_a = _load_normalized_mesh(pair.asset_a.asset_path, max_faces=cfg.max_faces_per_mesh, target_diagonal=cfg.target_diagonal, seed=11)
    mesh_b = _load_normalized_mesh(pair.asset_b.asset_path, max_faces=cfg.max_faces_per_mesh, target_diagonal=cfg.target_diagonal, seed=29)
    poly_a = _polydata(mesh_a)
    poly_b = _polydata(mesh_b)
    c0a, c0b = _centers_for_contact(mesh_a, mesh_b, 0.0, cfg)
    poly_a.points = mesh_a.vertices + c0a
    poly_b.points = mesh_b.vertices + c0b

    pv.OFF_SCREEN = True
    plotter = pv.Plotter(off_screen=True, window_size=(cfg.width, cfg.height))
    writer = imageio.get_writer(str(mp4_path), fps=cfg.fps, codec="libx264", quality=8, macro_block_size=16)
    try:
        plotter.set_background("#111827")
        plotter.enable_anti_aliasing("ssaa")
        plotter.add_mesh(poly_a, color="#4aa3ff", smooth_shading=True, specular=0.35, roughness=0.55, show_edges=False)
        plotter.add_mesh(poly_b, color="#ff6868", smooth_shading=True, specular=0.35, roughness=0.55, show_edges=False)
        all_pts = np.vstack((mesh_a.vertices, mesh_b.vertices))
        span = float(np.linalg.norm(all_pts.max(axis=0) - all_pts.min(axis=0)) + cfg.approach_distance + cfg.rebound_distance)
        target = np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
        camera = target + np.asarray([0.85 * span, -1.20 * span, 0.72 * span], dtype=np.float32)
        plotter.add_light(pv.Light(position=tuple(camera), focal_point=tuple(target), color="white", intensity=0.85))
        plotter.camera_position = (tuple(camera.tolist()), tuple(target.tolist()), (0.0, 0.0, 1.0))
        plotter.camera.parallel_projection = True
        plotter.camera.parallel_scale = 0.62 * span
        plotter.camera.clipping_range = (0.01, 1000.0)
        for frame_idx in range(cfg.frame_count):
            t = frame_idx / max(1, cfg.frame_count - 1)
            ca, cb = _centers_for_contact(mesh_a, mesh_b, float(t), cfg)
            poly_a.points = mesh_a.vertices + ca
            poly_b.points = mesh_b.vertices + cb
            plotter.render()
            img = Image.fromarray(np.asarray(plotter.screenshot(return_img=True)[:, :, :3], dtype=np.uint8), mode="RGB")
            img = _draw_overlay(img, case=case, pair=pair, t=float(t))
            if frame_idx == cfg.frame_count // 2:
                img.save(preview_path)
            writer.append_data(np.asarray(img))
    finally:
        writer.close()
        plotter.close()
    return Real3DCaseArtifact(
        case_name=case["case_name"],
        mp4_path=mp4_path,
        preview_png_path=preview_path,
        asset_a_path=pair.asset_a.asset_path,
        asset_b_path=pair.asset_b.asset_path,
        asset_a_faces=pair.asset_a.face_count,
        asset_b_faces=pair.asset_b.face_count,
        rendered_faces_a=int(mesh_a.faces.shape[0]),
        rendered_faces_b=int(mesh_b.faces.shape[0]),
    )


def _write_contact_sheet(real3d_dir: Path, artifacts: Sequence[Real3DCaseArtifact]) -> Path:
    tile_w, tile_h = 640, 360
    columns = 2
    rows = int(np.ceil(len(artifacts) / columns))
    margin = 28
    title_h = 84
    sheet = Image.new(
        "RGB",
        (columns * tile_w + (columns + 1) * margin, title_h + rows * tile_h + (rows + 1) * margin),
        (8, 13, 23),
    )
    draw = ImageDraw.Draw(sheet, "RGBA")
    draw.text((margin, 24), "Multi-Dense Mesh Contact Pairs: Real 3D TOI Frames", font=_font(30, bold=True), fill=(226, 232, 240))
    draw.text(
        (margin, 58),
        "ABC / Fusion360 / Thingi10K same-source and cross-source real triangle mesh contact replay",
        font=_font(16),
        fill=(148, 163, 184),
    )
    for idx, artifact in enumerate(artifacts):
        row = idx // columns
        col = idx % columns
        x = margin + col * (tile_w + margin)
        y = title_h + margin + row * (tile_h + margin)
        preview = Image.open(artifact.preview_png_path).convert("RGB").resize((tile_w, tile_h), Image.Resampling.LANCZOS)
        sheet.paste(preview, (x, y))
        draw.rounded_rectangle((x, y, x + tile_w, y + 42), radius=10, fill=(8, 13, 23, 210))
        draw.text((x + 16, y + 10), artifact.case_name, font=_font(18, bold=True), fill=(226, 232, 240))
    contact_sheet_path = real3d_dir / "multi_dense_mesh_contact_pairs_real3d_contact_sheet.png"
    sheet.save(contact_sheet_path)
    return contact_sheet_path


def _update_markdown(output_dir: Path, artifacts: Sequence[Real3DCaseArtifact], contact_sheet_path: Path) -> None:
    readme = output_dir / "README.md"
    report = output_dir / "benchmark_report.md"
    section = [
        "",
        "## Real 3D Mesh Collision Replay",
        "",
        "underdescriptionFileisfrom Multi-Dense Mesh Contact Pairs real mesh pair description 3D descriptioncontact/description replay. mesh descriptionusevisualizationdescription; benchmark Metricsdescriptionfrom dense candidate contact benchmark. ",
        "",
        f"- overview figure: `{contact_sheet_path.relative_to(output_dir)}`",
        f"- Real3D Output directory: `{contact_sheet_path.parent.relative_to(output_dir)}`",
        "",
        "| Case | MP4 | Preview | Rendered faces A/B |",
        "| --- | --- | --- | ---: |",
    ]
    for artifact in artifacts:
        section.append(
            f"| `{artifact.case_name}` | `{artifact.mp4_path.relative_to(output_dir)}` | `{artifact.preview_png_path.relative_to(output_dir)}` | "
            f"`{artifact.rendered_faces_a}/{artifact.rendered_faces_b}` |"
        )
    section.append("")
    section_text = "\n".join(section)
    if readme.exists():
        text = readme.read_text(encoding="utf-8")
        marker = "## Real 3D Mesh Collision Replay"
        if marker in text:
            text = text[: text.index(marker)].rstrip()
        readme.write_text(text.rstrip() + "\n" + section_text, encoding="utf-8")
    if report.exists():
        text = report.read_text(encoding="utf-8")
        marker = "## Real 3D Mesh Collision Replay"
        if marker in text:
            text = text[: text.index(marker)].rstrip()
        report.write_text(text.rstrip() + "\n" + section_text, encoding="utf-8")


def write_multi_dense_mesh_contact_real3d_demo(
    config: MultiDenseMeshContactReal3DConfig | None = None,
) -> MultiDenseMeshContactReal3DResult:
    cfg = config or MultiDenseMeshContactReal3DConfig()
    output_dir = Path(cfg.output_dir)
    real3d_dir = output_dir / "real3d"
    real3d_dir.mkdir(parents=True, exist_ok=True)
    data = json.loads(Path(cfg.benchmark_json).read_text(encoding="utf-8"))
    eval_pairs = _build_eval_pairs()
    artifacts: list[Real3DCaseArtifact] = []
    for case_name in cfg.cases:
        case = _case_metrics(data, case_name)
        pair = eval_pairs[case_name][0]
        slug = _slug(case_name)
        artifacts.append(
            _render_case(
                cfg,
                case=case,
                pair=pair,
                mp4_path=real3d_dir / f"{slug}_real3d_collision.mp4",
                preview_path=real3d_dir / f"{slug}_real3d_toi.png",
            )
        )
    contact_sheet_path = _write_contact_sheet(real3d_dir, artifacts)
    summary_path = real3d_dir / "real3d_metrics.json"
    summary = {
        "config": asdict(cfg),
        "contact_sheet_path": str(contact_sheet_path),
        "artifacts": [
            {
                "case_name": artifact.case_name,
                "mp4_path": str(artifact.mp4_path),
                "preview_png_path": str(artifact.preview_png_path),
                "asset_a_path": artifact.asset_a_path,
                "asset_b_path": artifact.asset_b_path,
                "asset_a_faces": artifact.asset_a_faces,
                "asset_b_faces": artifact.asset_b_faces,
                "rendered_faces_a": artifact.rendered_faces_a,
                "rendered_faces_b": artifact.rendered_faces_b,
            }
            for artifact in artifacts
        ],
        "visualization_note": "Meshes are real triangle meshes from the benchmark sources, normalized for visual replay. AABB surface contact is aligned at t=0.5 for collision visualization.",
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    _update_markdown(output_dir, artifacts, contact_sheet_path)
    return MultiDenseMeshContactReal3DResult(
        output_dir=real3d_dir,
        artifacts=tuple(artifacts),
        summary_json_path=summary_path,
        contact_sheet_path=contact_sheet_path,
    )


def main() -> None:
    result = write_multi_dense_mesh_contact_real3d_demo()
    payload = {
        "output_dir": str(result.output_dir),
        "summary_json_path": str(result.summary_json_path),
        "contact_sheet_path": str(result.contact_sheet_path),
        "artifacts": [
            {"case_name": a.case_name, "mp4_path": str(a.mp4_path), "preview_png_path": str(a.preview_png_path)}
            for a in result.artifacts
        ],
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()


__all__ = [
    "MultiDenseMeshContactReal3DConfig",
    "MultiDenseMeshContactReal3DResult",
    "write_multi_dense_mesh_contact_real3d_demo",
]
