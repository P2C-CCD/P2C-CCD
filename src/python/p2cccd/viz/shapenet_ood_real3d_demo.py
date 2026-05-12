from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path
from typing import Sequence

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import trimesh


@dataclass(frozen=True, slots=True)
class ShapeNetOODReal3DConfig:
    output_dir: str = "src/MyDemo/paper_shapenet_ood_dense_highspeed_thinfeature_run_id"
    benchmark_json: str = "src/benchmark/shapenet_ood_dense_cases_run_id.json"
    selected_assets_csv: str = "src/benchmark/shapenet_ood_dense_cases_run_id_selected_assets.csv"
    case_summary_csv: str = "src/benchmark/shapenet_ood_dense_cases_run_id_case_summary.csv"
    width: int = 1280
    height: int = 720
    frame_count: int = 96
    fps: int = 24
    max_faces_per_mesh: int = 80_000
    target_diagonal: float = 2.8
    contact_window_start: float = 0.35
    contact_window_end: float = 0.65
    use_vtk_decimation: bool = False


@dataclass(frozen=True, slots=True)
class ShapeNetVizCase:
    case_name: str
    case_type: str
    category_a: str
    category_b: str
    title: str
    trajectory: str


@dataclass(frozen=True, slots=True)
class ShapeNetCaseArtifact:
    case_name: str
    case_type: str
    category_a: str
    category_b: str
    output_dir: Path
    mp4_path: Path
    preview_png_path: Path
    global_mp4_path: Path
    global_preview_png_path: Path
    local_zoom_mp4_path: Path
    local_zoom_preview_png_path: Path
    readme_path: Path
    metrics_json_path: Path
    asset_a_path: str
    asset_b_path: str
    asset_a_faces: int
    asset_b_faces: int
    rendered_faces_a: int
    rendered_faces_b: int
    work_reduction: float
    exact_calls: int
    fn_count: int


@dataclass(frozen=True, slots=True)
class ShapeNetOODReal3DResult:
    output_dir: Path
    artifacts: tuple[ShapeNetCaseArtifact, ...]
    contact_sheet_path: Path
    gallery_html_path: Path
    metrics_json_path: Path
    readme_path: Path
    benchmark_report_path: Path
    mydemo_index_path: Path


@dataclass(frozen=True, slots=True)
class _Asset:
    category: str
    model_id: str
    obj_path: Path
    faces: int
    vertices: int
    obj_bytes: int
    solid_binvox_bytes: int
    surface_binvox_bytes: int


@dataclass(frozen=True, slots=True)
class _RenderMesh:
    vertices: np.ndarray
    faces: np.ndarray
    original_face_count: int
    original_vertex_count: int
    scale: float


CASES: tuple[ShapeNetVizCase, ...] = (
    ShapeNetVizCase("S10-car-airplane-ood-cross", "dense_contact", "car", "airplane", "Car-Airplane OOD Dense Contact", "dense"),
    ShapeNetVizCase("S11-car-watercraft-ood-cross", "dense_contact", "car", "watercraft", "Car-Watercraft Dense Contact", "dense"),
    ShapeNetVizCase("S14-chair-sofa-soft-contact", "dense_contact", "chair", "sofa", "Chair-Sofa Wide Contact", "dense"),
    ShapeNetVizCase("S21-car-bus-high-speed", "high_speed", "car", "bus", "Car-Bus High-Speed Head-On", "high_speed"),
    ShapeNetVizCase("S25-airplane-rifle-fast-thin", "high_speed", "airplane", "rifle", "Airplane-Rifle Fast Thin Sweep", "high_speed"),
    ShapeNetVizCase("S30-rifle-rocket-fast-thin", "high_speed", "rifle", "rocket", "Rifle-Rocket Fast Thin Collision", "high_speed"),
    ShapeNetVizCase("S13-chair-table-contact", "thin_feature", "chair", "table", "Chair-Table Thin Feature Contact", "thin"),
    ShapeNetVizCase("S16-lamp-rifle-thin-rotation", "thin_feature", "lamp", "rifle", "Lamp-Rifle Rotating Thin Feature", "thin_rotation"),
    ShapeNetVizCase("S28-table-guitar-thin-grazing", "thin_feature", "table", "guitar", "Table-Guitar Thin Grazing", "thin"),
    ShapeNetVizCase("S19-loudspeaker-chair-cavity-contact", "binvox_proxy", "loudspeaker", "chair", "Loudspeaker-Chair Cavity Contact", "cavity"),
    ShapeNetVizCase("S24-watercraft-loudspeaker-cavity", "binvox_proxy", "watercraft", "loudspeaker", "Watercraft-Loudspeaker Cavity Contact", "cavity"),
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


def _slug(value: str) -> str:
    return value.lower().replace("+", "plus").replace("/", "-").replace("_", "-").replace(" ", "-")


def _load_assets(csv_path: Path) -> dict[str, tuple[_Asset, ...]]:
    by_category: dict[str, list[_Asset]] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            root = Path(row["zip_path"])
            asset = _Asset(
                category=row["category"],
                model_id=row["model_id"],
                obj_path=root / row["obj_entry"],
                faces=int(row["faces"]),
                vertices=int(row["vertices"]),
                obj_bytes=int(row["obj_bytes"]),
                solid_binvox_bytes=int(row["solid_binvox_bytes"]),
                surface_binvox_bytes=int(row["surface_binvox_bytes"]),
            )
            if asset.obj_path.exists():
                by_category.setdefault(asset.category, []).append(asset)
    return {key: tuple(sorted(value, key=lambda item: (-item.faces, -item.obj_bytes, item.model_id))) for key, value in by_category.items()}


def _load_case_summary(csv_path: Path) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("role") == "eval":
                rows[row["case_name"]] = row
    return rows


def _polydata_from_arrays(vertices: np.ndarray, faces_array: np.ndarray):
    import pyvista as pv

    faces = np.empty((faces_array.shape[0], 4), dtype=np.int64)
    faces[:, 0] = 3
    faces[:, 1:] = faces_array.astype(np.int64, copy=False)
    return pv.PolyData(vertices.astype(np.float32, copy=False), faces.reshape(-1))


def _arrays_from_polydata(poly) -> tuple[np.ndarray, np.ndarray]:
    poly = poly.triangulate()
    cells = np.asarray(poly.faces, dtype=np.int64).reshape(-1, 4)
    tri = cells[cells[:, 0] == 3, 1:]
    return (
        np.ascontiguousarray(np.asarray(poly.points, dtype=np.float32)),
        np.ascontiguousarray(tri.astype(np.int32, copy=False)),
    )


def _try_vtk_decimate(vertices: np.ndarray, faces: np.ndarray, *, target_faces: int) -> tuple[np.ndarray, np.ndarray] | None:
    try:
        import vtk

        vtk.vtkObject.GlobalWarningDisplayOff()
        poly = _polydata_from_arrays(vertices, faces)
        for _ in range(4):
            if poly.n_cells <= target_faces * 1.15:
                break
            reduction = max(0.05, min(0.92, 1.0 - float(target_faces) / max(1.0, float(poly.n_cells))))
            decimated = poly.decimate_pro(
                reduction=reduction,
                preserve_topology=False,
                boundary_vertex_deletion=False,
                feature_angle=35.0,
            )
            if decimated.n_cells <= 0 or decimated.n_cells >= poly.n_cells:
                break
            poly = decimated
        return _arrays_from_polydata(poly)
    except Exception:
        return None


def _vertex_cluster_simplify(vertices: np.ndarray, faces: np.ndarray, *, max_faces: int) -> tuple[np.ndarray, np.ndarray] | None:
    # Vertex clustering preserves the whole silhouette better than face subsampling
    # for dirty ShapeNet meshes. It is used only for visualization, not benchmark data.
    bounds_min = vertices.min(axis=0)
    bounds_max = vertices.max(axis=0)
    extent = np.maximum(bounds_max - bounds_min, 1.0e-12)
    best: tuple[np.ndarray, np.ndarray] | None = None
    for resolution in (112, 88, 72, 56):
        q = np.floor((vertices - bounds_min) / extent * float(resolution - 1)).astype(np.int64)
        codes = q[:, 0] + resolution * q[:, 1] + resolution * resolution * q[:, 2]
        _, inverse, counts = np.unique(codes, return_inverse=True, return_counts=True)
        clustered_vertices = np.zeros((counts.shape[0], 3), dtype=np.float64)
        np.add.at(clustered_vertices, inverse, vertices.astype(np.float64, copy=False))
        clustered_vertices /= counts[:, None]
        clustered_faces = inverse[faces]
        valid = (
            (clustered_faces[:, 0] != clustered_faces[:, 1])
            & (clustered_faces[:, 1] != clustered_faces[:, 2])
            & (clustered_faces[:, 2] != clustered_faces[:, 0])
        )
        clustered_faces = clustered_faces[valid]
        if clustered_faces.size == 0:
            continue
        candidate = (
            np.ascontiguousarray(clustered_vertices.astype(np.float32)),
            np.ascontiguousarray(clustered_faces.astype(np.int32, copy=False)),
        )
        best = candidate
        if clustered_faces.shape[0] <= int(max_faces * 1.15):
            return candidate
    return best


def _spatial_face_sample(vertices: np.ndarray, faces: np.ndarray, *, max_faces: int) -> tuple[np.ndarray, np.ndarray]:
    # Last-resort visualization fallback. Stratified centroid sampling keeps broad
    # object coverage after vertex clustering has already reduced the mesh.
    if faces.shape[0] <= max_faces:
        picked = np.arange(faces.shape[0], dtype=np.int64)
    else:
        centroids = vertices[faces].mean(axis=1)
        order = np.lexsort((centroids[:, 2], centroids[:, 1], centroids[:, 0]))
        picked = np.sort(order[np.linspace(0, order.shape[0] - 1, max_faces, dtype=np.int64)])
    sampled_faces = faces[picked]
    used = np.unique(sampled_faces.reshape(-1))
    remap = np.full(vertices.shape[0], -1, dtype=np.int64)
    remap[used] = np.arange(used.shape[0], dtype=np.int64)
    return vertices[used], remap[sampled_faces]


def _load_normalized_mesh(
    asset: _Asset,
    *,
    max_faces: int,
    target_diagonal: float,
    seed: int,
    use_vtk_decimation: bool,
) -> _RenderMesh:
    loaded = trimesh.load(str(asset.obj_path), force="mesh", process=False)
    if isinstance(loaded, trimesh.Scene):
        loaded = trimesh.util.concatenate(tuple(loaded.dump()))
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(f"unsupported mesh type for {asset.obj_path}: {type(loaded)!r}")
    vertices = np.asarray(loaded.vertices, dtype=np.float32)
    faces = np.asarray(loaded.faces, dtype=np.int64)
    if vertices.size == 0 or faces.size == 0:
        raise ValueError(f"empty mesh: {asset.obj_path}")
    original_face_count = int(faces.shape[0])
    original_vertex_count = int(vertices.shape[0])
    if faces.shape[0] > max_faces:
        decimated = _try_vtk_decimate(vertices, faces, target_faces=max_faces) if use_vtk_decimation else None
        if decimated is not None:
            vertices, faces = decimated
        if faces.shape[0] > max_faces:
            clustered = _vertex_cluster_simplify(vertices, faces, max_faces=max_faces)
            if clustered is not None:
                vertices, faces = clustered
        if faces.shape[0] > max_faces:
            vertices, faces = _spatial_face_sample(vertices, faces, max_faces=max_faces)
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


def _select_visual_asset(assets: tuple[_Asset, ...], *, exclude: _Asset | None = None) -> _Asset:
    usable = [asset for asset in assets if exclude is None or asset.model_id != exclude.model_id]
    if not usable:
        return assets[0]
    # Extremely large ShapeNet OBJs are often fragmented scans/part dumps. For
    # paper visualization, choose the densest still-readable mesh from the same
    # category pool and keep benchmark metrics attached to the case family.
    for max_faces in (900_000, 1_200_000, 1_800_000):
        candidates = [asset for asset in usable if 50_000 <= asset.faces <= max_faces]
        if candidates:
            return max(candidates, key=lambda item: (item.faces, item.obj_bytes))
    return max(usable, key=lambda item: (item.faces, item.obj_bytes))


def _rotation_z(angle: float) -> np.ndarray:
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    return np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)


def _rotation_y(angle: float) -> np.ndarray:
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    return np.asarray([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float32)


def _trajectory_params(case: ShapeNetVizCase, t: float) -> tuple[float, float, float, float, float]:
    if case.trajectory == "high_speed":
        gap = 2.55 * abs(2.0 * t - 1.0)
        lateral = 0.035 * np.sin(2.0 * np.pi * t)
        z_offset = 0.03 * np.cos(2.0 * np.pi * t)
        yaw_a = -0.04
        yaw_b = 0.06
    elif case.trajectory == "thin_rotation":
        gap = 1.25 * abs(2.0 * t - 1.0)
        lateral = 0.19 * (1.0 - abs(2.0 * t - 1.0))
        z_offset = 0.04 * np.sin(np.pi * t)
        yaw_a = 0.0
        yaw_b = -0.75 + 1.50 * t
    elif case.trajectory == "thin":
        gap = 1.05 * abs(2.0 * t - 1.0)
        lateral = 0.14 * (1.0 - abs(2.0 * t - 1.0))
        z_offset = 0.025 * np.sin(np.pi * t)
        yaw_a = 0.0
        yaw_b = 0.20
    elif case.trajectory == "cavity":
        gap = 1.35 * abs(2.0 * t - 1.0)
        lateral = -0.16 * (1.0 - abs(2.0 * t - 1.0))
        z_offset = 0.11 * (1.0 - abs(2.0 * t - 1.0))
        yaw_a = -0.20
        yaw_b = 0.35
    else:
        gap = 1.55 * abs(2.0 * t - 1.0)
        lateral = 0.06 * np.sin(np.pi * t)
        z_offset = 0.035 * np.sin(2.0 * np.pi * t)
        yaw_a = -0.08
        yaw_b = 0.08
    return float(gap), float(lateral), float(z_offset), float(yaw_a), float(yaw_b)


def _transform_pair(mesh_a: _RenderMesh, mesh_b: _RenderMesh, case: ShapeNetVizCase, t: float) -> tuple[np.ndarray, np.ndarray]:
    gap, lateral, z_offset, yaw_a, yaw_b = _trajectory_params(case, t)
    rot_a = _rotation_z(yaw_a) @ _rotation_y(0.04 * np.sin(np.pi * t))
    rot_b = _rotation_z(yaw_b) @ _rotation_y(-0.06 * np.sin(np.pi * t))
    va = mesh_a.vertices @ rot_a.T
    vb = mesh_b.vertices @ rot_b.T
    x_shift_b = float(va[:, 0].max() - vb[:, 0].min() + gap)
    pa = va + np.asarray([0.0, -0.5 * lateral, -0.5 * z_offset], dtype=np.float32)
    pb = vb + np.asarray([x_shift_b, 0.5 * lateral, 0.5 * z_offset], dtype=np.float32)
    all_points = np.vstack((pa, pb))
    center = 0.5 * (all_points.min(axis=0) + all_points.max(axis=0))
    return pa - center, pb - center


def _draw_overlay(
    frame: Image.Image,
    *,
    case: ShapeNetVizCase,
    asset_a: _Asset,
    asset_b: _Asset,
    summary: dict[str, str],
    t: float,
    progress: float,
    view_label: str,
    view_note: str,
) -> Image.Image:
    draw = ImageDraw.Draw(frame, "RGBA")
    draw.rounded_rectangle((18, 16, frame.width - 18, 122), radius=16, fill=(8, 13, 23, 225), outline=(51, 65, 85), width=1)
    draw.text((42, 32), case.title, font=_font(27, bold=True), fill=(226, 232, 240))
    draw.text(
        (42, 72),
        f"{asset_a.category}:{asset_a.model_id[:8]}  x  {asset_b.category}:{asset_b.model_id[:8]} | {case.case_type} | {view_label} | t={t:.3f}",
        font=_font(15),
        fill=(203, 213, 225),
    )
    reduction = 100.0 * float(summary.get("exact_work_reduction", 0.0) or 0.0)
    calls = int(float(summary.get("trained_exact_calls", 0) or 0))
    candidates = int(float(summary.get("candidate_count", 0) or 0))
    fn = int(float(summary.get("fn_count", 0) or 0))
    draw.text((frame.width - 560, 34), f"RTSTPFExact calls {calls:,}/{candidates:,}", font=_font(17), fill=(34, 197, 94))
    draw.text((frame.width - 560, 68), f"work reduction {reduction:.2f}% | FN {fn}", font=_font(17), fill=(226, 232, 240))
    draw.text(
        (42, frame.height - 72),
        f"real ShapeNet OBJ triangle meshes; {view_note}",
        font=_font(14),
        fill=(148, 163, 184),
    )
    draw.rectangle((36, frame.height - 42, frame.width - 36, frame.height - 28), fill=(30, 41, 59, 230))
    draw.rectangle((36, frame.height - 42, int(36 + (frame.width - 72) * progress), frame.height - 28), fill=(34, 211, 238, 255))
    return frame


def _write_case_readme(artifact: ShapeNetCaseArtifact, case: ShapeNetVizCase) -> None:
    text = f"""# {case.title}

## Notes

this is ShapeNetCore OOD benchmark real OBJ triangle mesh replay. thisvisualizationused fordescription `{case.case_type}` scenein 3D descriptioncontact/description; CCD descriptionMetrics are from `shapenet_ood_dense_cases_run_id` dense candidate benchmark.

## File

| File | Notes |
| --- | --- |
| `{artifact.global_mp4_path.name}` | description: completedescriptionconnectdescription/collision/separation replay |
| `{artifact.global_preview_png_path.name}` | description TOI/contact preview description |
| `{artifact.local_zoom_mp4_path.name}` | description: TOI descriptioncontactdescription replay |
| `{artifact.local_zoom_preview_png_path.name}` | description TOI/contact preview description |
| `{artifact.metrics_json_path.name}` | case descriptionMetrics |

## Benchmark description

- Case: `{artifact.case_name}`
- Type: `{artifact.case_type}`
- Pair: `{artifact.category_a}` x `{artifact.category_b}`
- Asset faces: `{artifact.asset_a_faces:,}` / `{artifact.asset_b_faces:,}`
- Rendered faces: `{artifact.rendered_faces_a:,}` / `{artifact.rendered_faces_b:,}`
- Exact calls: `{artifact.exact_calls:,}`
- Exact-work reduction: `{100.0 * artifact.work_reduction:.2f}%`
- FN: `{artifact.fn_count}`

## descriptionusedescription

this isdescriptionvisualization replay, is notcompletedescription. description correctness descriptionfrom exact certificate / conservative fallback benchmark.
"""
    artifact.readme_path.write_text(text, encoding="utf-8")


def _render_variant(
    cfg: ShapeNetOODReal3DConfig,
    *,
    case: ShapeNetVizCase,
    asset_a: _Asset,
    asset_b: _Asset,
    summary: dict[str, str],
    mesh_a: _RenderMesh,
    mesh_b: _RenderMesh,
    mp4_path: Path,
    preview_path: Path,
    mode: str,
) -> None:
    import pyvista as pv

    is_global = mode == "global"
    poly_a = _polydata(mesh_a)
    poly_b = _polydata(mesh_b)
    initial_t = 0.0 if is_global else cfg.contact_window_start
    pa0, pb0 = _transform_pair(mesh_a, mesh_b, case, initial_t)
    poly_a.points = pa0
    poly_b.points = pb0

    pv.OFF_SCREEN = True
    plotter = pv.Plotter(off_screen=True, window_size=(cfg.width, cfg.height))
    writer = imageio.get_writer(str(mp4_path), fps=cfg.fps, codec="libx264", quality=8, macro_block_size=16)
    try:
        plotter.set_background("#111827")
        plotter.enable_anti_aliasing("ssaa")
        plotter.add_mesh(
            poly_a,
            color="#38bdf8",
            smooth_shading=True,
            ambient=0.64,
            diffuse=0.82,
            specular=0.22,
            roughness=0.50,
            show_edges=True,
            edge_color="#bae6fd",
            line_width=0.15,
        )
        plotter.add_mesh(
            poly_b,
            color="#fb7185",
            smooth_shading=True,
            ambient=0.64,
            diffuse=0.82,
            specular=0.22,
            roughness=0.50,
            show_edges=True,
            edge_color="#fecaca",
            line_width=0.15,
        )
        if is_global:
            samples = [_transform_pair(mesh_a, mesh_b, case, t) for t in (0.0, 0.5, 1.0)]
            all_mid = np.vstack([points for pair in samples for points in pair])
            span_pad = 1.45
            parallel_scale = 0.40
            view_label = "GLOBAL"
            view_note = "global full-trajectory replay"
        else:
            va_mid, vb_mid = _transform_pair(mesh_a, mesh_b, case, 0.5)
            all_mid = np.vstack((va_mid, vb_mid))
            span_pad = 0.45
            parallel_scale = 0.22
            view_label = "LOCAL ZOOM"
            view_note = "local TOI contact-window zoom replay"
        span = float(np.linalg.norm(all_mid.max(axis=0) - all_mid.min(axis=0)) + span_pad)
        target = np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
        camera = target + np.asarray([0.70 * span, -0.92 * span, 0.56 * span], dtype=np.float32)
        plotter.add_light(pv.Light(position=tuple(camera), focal_point=tuple(target), color="white", intensity=1.25))
        plotter.add_light(pv.Light(position=(-camera[0], camera[1], camera[2]), focal_point=tuple(target), color="#dbeafe", intensity=0.45))
        plotter.camera_position = (tuple(camera.tolist()), tuple(target.tolist()), (0.0, 0.0, 1.0))
        plotter.camera.parallel_projection = True
        plotter.camera.parallel_scale = parallel_scale * span
        plotter.camera.clipping_range = (0.01, 1000.0)
        for frame_idx in range(cfg.frame_count):
            u = frame_idx / max(1, cfg.frame_count - 1)
            t = u if is_global else cfg.contact_window_start + (cfg.contact_window_end - cfg.contact_window_start) * u
            pa, pb = _transform_pair(mesh_a, mesh_b, case, float(t))
            poly_a.points = pa
            poly_b.points = pb
            plotter.render()
            img = Image.fromarray(np.asarray(plotter.screenshot(return_img=True)[:, :, :3], dtype=np.uint8), mode="RGB")
            img = _draw_overlay(
                img,
                case=case,
                asset_a=asset_a,
                asset_b=asset_b,
                summary=summary,
                t=float(t),
                progress=float(u),
                view_label=view_label,
                view_note=view_note,
            )
            if frame_idx == cfg.frame_count // 2:
                img.save(preview_path)
            writer.append_data(np.asarray(img))
    finally:
        writer.close()
        plotter.close()


def _render_case(
    cfg: ShapeNetOODReal3DConfig,
    *,
    case: ShapeNetVizCase,
    asset_a: _Asset,
    asset_b: _Asset,
    summary: dict[str, str],
    output_dir: Path,
    mesh_cache: dict[str, _RenderMesh],
) -> ShapeNetCaseArtifact:
    output_dir.mkdir(parents=True, exist_ok=True)
    slug = _slug(case.case_name)
    global_mp4_path = output_dir / f"{slug}_global_real3d_collision.mp4"
    global_preview_path = output_dir / f"{slug}_global_toi.png"
    local_mp4_path = output_dir / f"{slug}_local_zoom_real3d_collision.mp4"
    local_preview_path = output_dir / f"{slug}_local_zoom_toi.png"
    metrics_path = output_dir / "metrics.json"
    readme_path = output_dir / "README.md"
    key_a = str(asset_a.obj_path)
    key_b = str(asset_b.obj_path)
    mesh_a = mesh_cache.setdefault(
        key_a,
        _load_normalized_mesh(
            asset_a,
            max_faces=cfg.max_faces_per_mesh,
            target_diagonal=cfg.target_diagonal,
            seed=abs(hash(key_a)) % (2**32),
            use_vtk_decimation=cfg.use_vtk_decimation,
        ),
    )
    mesh_b = mesh_cache.setdefault(
        key_b,
        _load_normalized_mesh(
            asset_b,
            max_faces=cfg.max_faces_per_mesh,
            target_diagonal=cfg.target_diagonal,
            seed=abs(hash(key_b)) % (2**32),
            use_vtk_decimation=cfg.use_vtk_decimation,
        ),
    )
    _render_variant(
        cfg,
        case=case,
        asset_a=asset_a,
        asset_b=asset_b,
        summary=summary,
        mesh_a=mesh_a,
        mesh_b=mesh_b,
        mp4_path=global_mp4_path,
        preview_path=global_preview_path,
        mode="global",
    )
    _render_variant(
        cfg,
        case=case,
        asset_a=asset_a,
        asset_b=asset_b,
        summary=summary,
        mesh_a=mesh_a,
        mesh_b=mesh_b,
        mp4_path=local_mp4_path,
        preview_path=local_preview_path,
        mode="local",
    )
    artifact = ShapeNetCaseArtifact(
        case_name=case.case_name,
        case_type=case.case_type,
        category_a=asset_a.category,
        category_b=asset_b.category,
        output_dir=output_dir,
        mp4_path=local_mp4_path,
        preview_png_path=local_preview_path,
        global_mp4_path=global_mp4_path,
        global_preview_png_path=global_preview_path,
        local_zoom_mp4_path=local_mp4_path,
        local_zoom_preview_png_path=local_preview_path,
        readme_path=readme_path,
        metrics_json_path=metrics_path,
        asset_a_path=str(asset_a.obj_path),
        asset_b_path=str(asset_b.obj_path),
        asset_a_faces=asset_a.faces,
        asset_b_faces=asset_b.faces,
        rendered_faces_a=int(mesh_a.faces.shape[0]),
        rendered_faces_b=int(mesh_b.faces.shape[0]),
        work_reduction=float(summary.get("exact_work_reduction", 0.0) or 0.0),
        exact_calls=int(float(summary.get("trained_exact_calls", 0) or 0)),
        fn_count=int(float(summary.get("fn_count", 0) or 0)),
    )
    metrics_path.write_text(json.dumps(asdict(artifact), indent=2, ensure_ascii=False, sort_keys=True, default=str) + "\n", encoding="utf-8")
    _write_case_readme(artifact, case)
    return artifact


def _write_contact_sheet(output_dir: Path, artifacts: Sequence[ShapeNetCaseArtifact], *, mode: str) -> Path:
    tile_w, tile_h = 560, 315
    columns = 3
    rows = int(np.ceil(len(artifacts) / columns))
    margin = 24
    title_h = 90
    sheet = Image.new(
        "RGB",
        (columns * tile_w + (columns + 1) * margin, title_h + rows * tile_h + (rows + 1) * margin),
        (8, 13, 23),
    )
    draw = ImageDraw.Draw(sheet, "RGBA")
    is_global = mode == "global"
    title_suffix = "Global View" if is_global else "Local Zoom View"
    subtitle = "full approach/contact/separation replay" if is_global else "TOI contact-window zoom replay"
    draw.text((margin, 22), f"ShapeNetCore OOD Dense / High-Speed / Thin-Feature Cases - {title_suffix}", font=_font(28, bold=True), fill=(226, 232, 240))
    draw.text((margin, 58), f"Real OBJ triangle mesh {subtitle}, grouped by dense_contact, high_speed, thin_feature, and binvox_proxy", font=_font(15), fill=(148, 163, 184))
    for idx, artifact in enumerate(artifacts):
        row = idx // columns
        col = idx % columns
        x = margin + col * (tile_w + margin)
        y = title_h + margin + row * (tile_h + margin)
        preview_path = artifact.global_preview_png_path if is_global else artifact.local_zoom_preview_png_path
        preview = Image.open(preview_path).convert("RGB").resize((tile_w, tile_h), Image.Resampling.LANCZOS)
        sheet.paste(preview, (x, y))
        draw.rounded_rectangle((x, y, x + tile_w, y + 44), radius=10, fill=(8, 13, 23, 215))
        draw.text((x + 14, y + 9), f"{artifact.case_type}: {artifact.case_name}", font=_font(15, bold=True), fill=(226, 232, 240))
    path = output_dir / ("shapenet_ood_real3d_global_contact_sheet.png" if is_global else "shapenet_ood_real3d_local_zoom_contact_sheet.png")
    sheet.save(path)
    return path


def _write_gallery_html(output_dir: Path, artifacts: Sequence[ShapeNetCaseArtifact]) -> Path:
    cards = []
    for artifact in artifacts:
        rel_global_mp4 = artifact.global_mp4_path.relative_to(output_dir).as_posix()
        rel_global_png = artifact.global_preview_png_path.relative_to(output_dir).as_posix()
        rel_local_mp4 = artifact.local_zoom_mp4_path.relative_to(output_dir).as_posix()
        rel_local_png = artifact.local_zoom_preview_png_path.relative_to(output_dir).as_posix()
        cards.append(
            f"""
<section class="card">
  <h2>{artifact.case_name}</h2>
  <p>{artifact.case_type} | {artifact.category_a} x {artifact.category_b} | reduction {100.0 * artifact.work_reduction:.2f}% | FN {artifact.fn_count}</p>
  <div class="views">
    <div>
      <h3>Global</h3>
      <video controls muted loop preload="metadata" poster="{rel_global_png}" src="{rel_global_mp4}"></video>
    </div>
    <div>
      <h3>Local Zoom</h3>
      <video controls muted loop preload="metadata" poster="{rel_local_png}" src="{rel_local_mp4}"></video>
    </div>
  </div>
</section>
"""
        )
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>ShapeNet OOD Real3D Gallery</title>
<style>
body {{ margin:0; background:#08111f; color:#e5edf7; font-family: Segoe UI, Microsoft YaHei, sans-serif; }}
.wrap {{ max-width: 1400px; margin: 0 auto; padding: 24px; }}
.grid {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(560px, 1fr)); gap:18px; }}
.card {{ background:#0d192c; border:1px solid #1f2f47; border-radius:16px; padding:16px; }}
h1 {{ margin:0 0 6px 0; }}
h2 {{ font-size:18px; margin:0 0 6px 0; }}
h3 {{ font-size:14px; margin:8px 0 6px 0; color:#cbd5e1; }}
p {{ color:#a7b3c7; }}
video {{ width:100%; border-radius:12px; background:#111827; }}
.views {{ display:grid; grid-template-columns: 1fr 1fr; gap:12px; }}
@media (max-width: 900px) {{ .views {{ grid-template-columns: 1fr; }} .grid {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
<div class="wrap">
<h1>ShapeNetCore OOD Dense / High-Speed / Thin-Feature Real3D Gallery</h1>
<p>real ShapeNet OBJ mesh descriptioncontact/description replay. each case descriptionwhendescription global descriptionand local zoom contactdescription; descriptionMetrics are from dense candidate CCD benchmark. </p>
<div class="grid">
{''.join(cards)}
</div>
</div>
</body>
</html>
"""
    path = output_dir / "shapenet_ood_real3d_gallery.html"
    path.write_text(html, encoding="utf-8")
    return path


def _write_root_docs(
    output_dir: Path,
    *,
    benchmark: dict,
    artifacts: Sequence[ShapeNetCaseArtifact],
    global_contact_sheet: Path,
    local_contact_sheet: Path,
    gallery: Path,
) -> tuple[Path, Path, Path]:
    metrics_path = output_dir / "metrics.json"
    readme_path = output_dir / "README.md"
    report_path = output_dir / "benchmark_report.md"
    metrics = {
        "source_benchmark": "src/benchmark/shapenet_ood_dense_cases_run_id.md",
        "benchmark_summary": {
            "selected_asset_count": benchmark.get("selected_asset_count"),
            "case_family_count": benchmark.get("case_family_count"),
            "density": benchmark.get("density"),
            "train_dense_rows": benchmark.get("train_dense_rows"),
            "eval_dense_rows": benchmark.get("eval_dense_rows"),
            "trained_reduction_vs_no_proposal": benchmark.get("trained_reduction_vs_no_proposal"),
            "trained_call_reduction_vs_no_proposal": benchmark.get("trained_call_reduction_vs_no_proposal"),
            "fn_count": benchmark.get("trained_stpf", {}).get("fn_count"),
        },
        "global_contact_sheet": str(global_contact_sheet),
        "local_zoom_contact_sheet": str(local_contact_sheet),
        "gallery_html": str(gallery),
        "artifacts": [asdict(artifact) for artifact in artifacts],
    }
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False, sort_keys=True, default=str) + "\n", encoding="utf-8")
    by_type: dict[str, list[ShapeNetCaseArtifact]] = {}
    for artifact in artifacts:
        by_type.setdefault(artifact.case_type, []).append(artifact)
    rows = [
        "| Case | Type | Pair | Global MP4 | Local Zoom MP4 | Preview | Reduction | FN |",
        "| --- | --- | --- | --- | --- | --- | ---: | ---: |",
    ]
    for artifact in artifacts:
        rows.append(
            f"| `{artifact.case_name}` | `{artifact.case_type}` | `{artifact.category_a} x {artifact.category_b}` | "
            f"`{artifact.global_mp4_path.relative_to(output_dir)}` | `{artifact.local_zoom_mp4_path.relative_to(output_dir)}` | "
            f"`{artifact.local_zoom_preview_png_path.relative_to(output_dir)}` | "
            f"`{100.0 * artifact.work_reduction:.2f}%` | `{artifact.fn_count}` |"
        )
    readme = f"""# Paper Demo: ShapeNetCore OOD Dense / High-Speed / Thin-Feature Cases

## usedescription

thisdescriptionvisualization ShapeNetCore OOD dense / high-speed / thin-feature cases. descriptionusereal ShapeNetCore OBJ description, coverage OOD dense mesh contact, highdescription, description/descriptionand binvox proxy/cavity contact.

## description

- `dense_contact/`: descriptionordescription dense mesh contact.
- `high_speed/`: highdescriptioncollision replay.
- `thin_feature/`: description, description, descriptionanddescription.
- `binvox_proxy/`: descriptionuse binvox occupancy proxy descriptionconstruct cavity/contact scene.

## overview

- descriptionoverview figure: `{global_contact_sheet.relative_to(output_dir)}`
- descriptionoverview figure: `{local_contact_sheet.relative_to(output_dir)}`
- HTML gallery: `{gallery.relative_to(output_dir)}`
- Benchmark: `src/benchmark/shapenet_ood_dense_cases_run_id.md`

## descriptionvisualizationdescription

- `global`: descriptioncompletedescriptionconnectdescription, collision, separationtrajectory, descriptionNotesscenedescriptionanddescription.
- `local_zoom`: description TOI descriptioncontactdescription, descriptionanddescriptioncontactdescriptionNotes.

## description

- Selected assets: `{benchmark.get('selected_asset_count')}`
- Case families: `{benchmark.get('case_family_count')}`
- Density: `{benchmark.get('density')}` candidates/query
- Eval dense rows: `{benchmark.get('eval_dense_rows')}`
- Learned exact-work reduction vs NoProposal: `{100.0 * float(benchmark.get('trained_reduction_vs_no_proposal', 0.0)):.2f}%`
- Learned exact-call reduction vs NoProposal: `{100.0 * float(benchmark.get('trained_call_reduction_vs_no_proposal', 0.0)):.2f}%`
- FN: `{benchmark.get('trained_stpf', {}).get('fn_count')}`

## visualizationFile

{chr(10).join(rows)}

## descriptionusedescription isdescriptionvisualization replay, mesh isreal OBJ, descriptiontrajectoryasdescriptioncontact/description. complete correctness descriptionfrom exact certificate / conservative fallback benchmark; descriptionwritedescriptioncompletephysicsdescription.
"""
    readme_path.write_text(readme, encoding="utf-8")
    report = readme.replace("# Paper Demo:", "# ShapeNetCore OOD Real3D visualizationanddescriptionNotes")
    report_path.write_text(report, encoding="utf-8")
    return metrics_path, readme_path, report_path


def _update_mydemo_index(root: Path, output_dir: Path) -> Path:
    index_path = root / "README.md"
    rows = []
    for directory in sorted([path for path in root.iterdir() if path.is_dir()], key=lambda item: item.name):
        if directory.name.startswith("."):
            continue
        description = {
            "paper_true_mesh_surface_contact_abc_run_id": "ABC real triangle-surface contact, TOI, wireframe zoom, descriptionmethod comparison",
            "paper_large_dense_complex_mesh_cases_run_id": "No. 9 description: ABC megaface + Fusion dense + Thingi10K dirty description dense case",
            "paper_multi_dense_mesh_contact_pairs_run_id": "No. 8 description: ABC/Fusion360/Thingi10K multi-source dense mesh contact",
            "paper_shapenet_ood_dense_highspeed_thinfeature_run_id": "ShapeNetCore OOD dense/high-speed/thin-feature real mesh replay",
            "trained_stpf_high_density_large_run_id": "Synthetic/proxy high-density STPF description, descriptionasrealdescription",
            "_archive_old_results": "descriptionvisualizationdescription",
            "old": "descriptionvisualizationdescription",
        }.get(directory.name, "")
        rows.append(f"| `{directory.name}` | {description} |")
    text = f"""# P2CCCD MyDemo Index

## description

- descriptionvisualizationdescriptionuse `paper_<case>_YYYY-MM-DD`.
- multi-source benchmark demo descriptionuse `<case>_YYYY-MM-DD`, descriptionhas `README.md` and `benchmark_report.md`.
- description `_archive_old_results/`.

## currentdescription

| Folder | Notes |
| --- | --- |
{chr(10).join(rows)}
"""
    index_path.write_text(text, encoding="utf-8")
    return index_path


def write_shapenet_ood_real3d_demo(config: ShapeNetOODReal3DConfig | None = None) -> ShapeNetOODReal3DResult:
    cfg = config or ShapeNetOODReal3DConfig()
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    benchmark = json.loads(Path(cfg.benchmark_json).read_text(encoding="utf-8"))
    assets_by_category = _load_assets(Path(cfg.selected_assets_csv))
    case_summary = _load_case_summary(Path(cfg.case_summary_csv))
    artifacts: list[ShapeNetCaseArtifact] = []
    mesh_cache: dict[str, _RenderMesh] = {}
    for case in CASES:
        if case.category_a not in assets_by_category or case.category_b not in assets_by_category:
            continue
        if case.case_name not in case_summary:
            continue
        asset_a = _select_visual_asset(assets_by_category[case.category_a])
        asset_b = _select_visual_asset(
            assets_by_category[case.category_b],
            exclude=asset_a if case.category_a == case.category_b else None,
        )
        case_dir = output_dir / case.case_type / _slug(case.case_name)
        artifacts.append(
            _render_case(
                cfg,
                case=case,
                asset_a=asset_a,
                asset_b=asset_b,
                summary=case_summary[case.case_name],
                output_dir=case_dir,
                mesh_cache=mesh_cache,
            )
        )
    if not artifacts:
        raise RuntimeError("no ShapeNet OOD real3d artifacts were generated")
    global_contact_sheet = _write_contact_sheet(output_dir, artifacts, mode="global")
    local_contact_sheet = _write_contact_sheet(output_dir, artifacts, mode="local")
    gallery = _write_gallery_html(output_dir, artifacts)
    metrics_path, readme_path, report_path = _write_root_docs(
        output_dir,
        benchmark=benchmark,
        artifacts=artifacts,
        global_contact_sheet=global_contact_sheet,
        local_contact_sheet=local_contact_sheet,
        gallery=gallery,
    )
    mydemo_root = output_dir.parent
    old_dir = mydemo_root / "old"
    archive_dir = mydemo_root / "_archive_old_results"
    if old_dir.exists() and not archive_dir.exists():
        old_dir.rename(archive_dir)
    mydemo_index = _update_mydemo_index(mydemo_root, output_dir)
    return ShapeNetOODReal3DResult(
        output_dir=output_dir,
        artifacts=tuple(artifacts),
        contact_sheet_path=local_contact_sheet,
        gallery_html_path=gallery,
        metrics_json_path=metrics_path,
        readme_path=readme_path,
        benchmark_report_path=report_path,
        mydemo_index_path=mydemo_index,
    )


def main() -> None:
    result = write_shapenet_ood_real3d_demo()
    print(
        json.dumps(
            {
                "output_dir": str(result.output_dir),
                "artifact_count": len(result.artifacts),
                "contact_sheet_path": str(result.contact_sheet_path),
                "gallery_html_path": str(result.gallery_html_path),
                "readme_path": str(result.readme_path),
                "benchmark_report_path": str(result.benchmark_report_path),
                "mydemo_index_path": str(result.mydemo_index_path),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()


__all__ = [
    "ShapeNetOODReal3DConfig",
    "ShapeNetOODReal3DResult",
    "write_shapenet_ood_real3d_demo",
]
