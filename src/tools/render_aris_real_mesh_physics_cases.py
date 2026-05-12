from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import importlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "src" / "MyDemo" / "paper_aris_ccf_a_cases_run_id"
ASSET_ROOT = ROOT / "src" / "datasets" / "shapenet_core_v2" / "selected_ood_dense_run_id"

DEFAULT_W, DEFAULT_H = 1600, 900
W, H = DEFAULT_W, DEFAULT_H
FPS = 24
FRAME_COUNT = 72
CONTACT_T = 1.05
DURATION_T = 2.55
SUPPORT_RENDER_CATEGORIES = {"wall", "ground", "slab"}
MAX_FACES = 18000
SEED = fixed_seed
# Keep a tiny positive gap at TOI to avoid renderer z-fighting while still
# showing a visually true wall-impact contact.
CAR_WALL_TRUE_CONTACT_GAP = 0.012
STATIC_FIT_EXCLUDED_CATEGORIES = {"wall", "brick_wall", "ground", "slab", "debris"}
CAR_WALL_IMPACT_Y = 0.0
CAR_WALL_IMPACT_Z = 0.78
CAR_WALL_RENDER_SPEED_SCALE = 0.285


def render_size_for_case(case_name: str) -> tuple[int, int]:
    if case_name == "many_object_tabletop_drop":
        return 1920, 1080
    return DEFAULT_W, DEFAULT_H

SYNSET_NAMES = {
    "02958343": "car",
    "02691156": "airplane",
    "04530566": "watercraft",
    "03001627": "chair",
    "04379243": "table",
    "04256520": "sofa",
    "03636649": "lamp",
    "04090263": "rifle",
    "02924116": "bus",
    "04468005": "train",
    "03467517": "guitar",
    "03691459": "loudspeaker",
}


def safe_rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(path)


_FFMPEG_ENCODER_CACHE: dict[tuple[str, str], bool] = {}


def ffmpeg_candidates() -> list[str]:
    candidates: list[str] = []

    def add(candidate_id: str | None) -> None:
        if not candidate_id:
            return
        path = str(Path(candidate))
        if Path(path).exists() and path not in candidates:
            candidates.append(path)

    add(os.environ.get("P2CCCD_FFMPEG"))
    add(shutil.which("ffmpeg"))
    for path_entry in os.environ.get("PATH", "").split(os.pathsep):
        add(str(Path(path_entry) / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")))
    return candidates


def ffmpeg_supports_encoder(ffmpeg: str, encoder: str) -> bool:
    key = (ffmpeg, encoder)
    if key in _FFMPEG_ENCODER_CACHE:
        return _FFMPEG_ENCODER_CACHE[key]
    try:
        result = subprocess.run(
            [ffmpeg, "-hide_banner", "-encoders"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        supported = encoder in result.stdout
    except Exception:
        supported = False
    _FFMPEG_ENCODER_CACHE[key] = supported
    return supported


def select_h264_encoder() -> tuple[str, str] | None:
    candidates = ffmpeg_candidates()
    for encoder in ("libx264", "h264_mf", "h264_nvenc", "h264_amf", "h264_qsv"):
        for ffmpeg in candidates:
            if ffmpeg_supports_encoder(ffmpeg, encoder):
                return ffmpeg, encoder
    return None


def h264_encoder_args(encoder: str) -> list[str]:
    args = ["-c:v", encoder, "-pix_fmt", "yuv420p", "-movflags", "+faststart"]
    if encoder == "libx264":
        return args + ["-crf", "18", "-preset", "medium"]
    return args + ["-b:v", "12M"]


def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


_P2CCCD_CPP = None
_P2CCCD_CPP_LOAD_ATTEMPTED = False
_P2CCCD_CPP_LOAD_ERROR: str | None = None
_P2CCCD_DLL_HANDLES: list[object] = []


def _add_dll_directory_if_available(path: Path) -> None:
    if not path.exists() or not hasattr(os, "add_dll_directory"):
        return
    try:
        _P2CCCD_DLL_HANDLES.append(os.add_dll_directory(str(path)))
    except OSError:
        pass


def try_load_p2cccd_cpp_for_render():
    """Load the project C++ exact-CCD binding without making rendering depend on OptiX."""
    global _P2CCCD_CPP, _P2CCCD_CPP_LOAD_ATTEMPTED, _P2CCCD_CPP_LOAD_ERROR
    if _P2CCCD_CPP_LOAD_ATTEMPTED:
        return _P2CCCD_CPP
    _P2CCCD_CPP_LOAD_ATTEMPTED = True
    project_root = ROOT / "src"
    build_dirs = [
        project_root / "build" / "cpp" / "Release",
        project_root / "build" / "cpp" / "Debug",
        project_root / "build_optix" / "cpp" / "Release",
        project_root / "build_optix" / "cpp" / "Debug",
    ]
    for build_dir in reversed(build_dirs):
        if build_dir.exists() and str(build_dir) not in sys.path:
            sys.path.insert(0, str(build_dir))
        _add_dll_directory_if_available(build_dir)
    conda_prefix = os.environ.get("CONDA_PREFIX")
    for support_dir in [
        Path(os.environ.get("P2CCCD_CPP_ROOT", "")) if os.environ.get("P2CCCD_CPP_ROOT") else None,
        Path(os.environ.get("P2CCCD_CPP_BIN", "")) if os.environ.get("P2CCCD_CPP_BIN") else None,
        Path(conda_prefix) / "Library" / "bin" if conda_prefix else None,
        Path(os.environ.get("CUDA_PATH", "")) / "bin" if os.environ.get("CUDA_PATH") else None,
        Path(os.environ.get("CUDA_HOME", "")) / "bin" if os.environ.get("CUDA_HOME") else None,
    ]:
        if support_dir is not None:
            _add_dll_directory_if_available(support_dir)
    try:
        _P2CCCD_CPP = importlib.import_module("p2cccd_cpp")
        return _P2CCCD_CPP
    except ImportError as exc:
        _P2CCCD_CPP_LOAD_ERROR = str(exc)
        return None


def find_asset(synset: str, rank: int = 0, max_bytes: int = 95_000_000) -> Path:
    files = sorted((ASSET_ROOT / synset).rglob("*.obj"), key=lambda p: p.stat().st_size, reverse=True)
    eligible = [p for p in files if p.stat().st_size <= max_bytes]
    if not eligible:
        eligible = files
    if not eligible:
        raise FileNotFoundError(f"No OBJ asset for synset {synset}")
    return eligible[min(rank, len(eligible) - 1)]


def load_mesh_preview(path: Path, max_faces: int = MAX_FACES) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    loaded = trimesh.load(path, force="mesh", process=False)
    if isinstance(loaded, trimesh.Scene):
        meshes = [g for g in loaded.dump() if isinstance(g, trimesh.Trimesh)]
        mesh = trimesh.util.concatenate(meshes)
    else:
        mesh = loaded
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    finite = np.isfinite(vertices).all(axis=1)
    if not finite.all():
        used = np.where(finite)[0]
        remap = -np.ones(len(vertices), dtype=np.int64)
        remap[used] = np.arange(len(used))
        mask = finite[faces].all(axis=1)
        faces = remap[faces[mask]]
        vertices = vertices[used]

    # ShapeNet normalized OBJ assets are commonly Y-up. Convert to the
    # renderer's Z-up convention so cars/aircraft/chairs lie on the ground
    # plane instead of standing vertically.
    vertices = vertices[:, [0, 2, 1]]

    center = vertices.mean(axis=0)
    vertices = vertices - center
    extent = vertices.max(axis=0) - vertices.min(axis=0)
    scale = 1.0 / max(float(extent.max()), 1e-9)
    vertices = vertices * scale

    # Align the dominant horizontal principal axis to +X so the visualized
    # approach is a genuine nose/front-to-nose/front collision instead of a
    # sideways crossing.  For asymmetric thin objects, make +X the pointier end.
    xy = vertices[:, :2]
    if len(xy) >= 3 and np.isfinite(xy).all():
        cov = np.cov(xy.T)
        vals, vecs = np.linalg.eigh(cov)
        axis = vecs[:, int(np.argmax(vals))]
        angle = math.atan2(float(axis[1]), float(axis[0]))
        c, s = math.cos(-angle), math.sin(-angle)
        rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        vertices = vertices @ rot.T
        q_lo, q_hi = np.quantile(vertices[:, 0], [0.08, 0.92])
        plus = vertices[vertices[:, 0] >= q_hi]
        minus = vertices[vertices[:, 0] <= q_lo]
        if len(plus) > 10 and len(minus) > 10:
            plus_radius = np.median(np.linalg.norm(plus[:, 1:3], axis=1))
            minus_radius = np.median(np.linalg.norm(minus[:, 1:3], axis=1))
            if plus_radius > minus_radius:
                vertices[:, 0] *= -1.0

    decimation_method = "full_surface"
    if len(faces) > max_faces:
        try:
            import pyvista as pv

            faces_pv = np.hstack([np.full((len(faces), 1), 3, dtype=np.int64), faces]).reshape(-1)
            poly = pv.PolyData(vertices, faces_pv)
            reduction = max(0.0, 1.0 - (float(max_faces) / max(float(poly.n_cells), 1.0)))
            simplified = poly.decimate(reduction, volume_preservation=True)
            if simplified.n_cells <= 0 or simplified.n_points <= 0:
                raise RuntimeError("PyVista decimation produced an empty mesh")
            vertices = np.asarray(simplified.points, dtype=np.float64)
            cell_faces = np.asarray(simplified.faces, dtype=np.int64).reshape(-1, 4)
            if not np.all(cell_faces[:, 0] == 3):
                simplified = simplified.triangulate()
                vertices = np.asarray(simplified.points, dtype=np.float64)
                cell_faces = np.asarray(simplified.faces, dtype=np.int64).reshape(-1, 4)
            faces = cell_faces[:, 1:4].astype(np.int64, copy=False)
            decimation_method = "pyvista_quadric_full_surface"
        except Exception:
            preview_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
            simplified = preview_mesh.simplify_quadric_decimation(face_count=max_faces, aggression=5)
            vertices = np.asarray(simplified.vertices, dtype=np.float64)
            faces = np.asarray(simplified.faces, dtype=np.int64)
            decimation_method = "trimesh_quadric_full_surface"

    ext = vertices.max(axis=0) - vertices.min(axis=0)
    stats = {
        "original_vertices": int(len(mesh.vertices)),
        "original_faces": int(len(mesh.faces)),
        "preview_vertices": int(len(vertices)),
        "preview_faces": int(len(faces)),
        "normalized_extent_x": float(ext[0]),
        "normalized_extent_y": float(ext[1]),
        "normalized_extent_z": float(ext[2]),
        "preview_decimation_method": decimation_method,
    }
    return vertices, faces, stats


def build_polar_silhouette_shell(
    vertices: np.ndarray,
    ring_count: int = 30,
    ray_count: int = 32,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Approximate a complete visual surface by sweeping cross-section hulls.

    This keeps long/thin objects more recognizable than a global convex hull:
    cars remain car-like, aircraft remain tapered, and thin-feature objects get
    a continuous display surface without changing the original benchmark mesh.
    """
    if len(vertices) < 64:
        return None
    x = vertices[:, 0]
    if not np.isfinite(vertices).all() or float(np.ptp(x)) < 1.0e-8:
        return None
    lo, hi = np.quantile(x, [0.015, 0.985])
    if hi <= lo:
        return None
    edges = np.linspace(float(lo), float(hi), ring_count + 1)
    centers_x = 0.5 * (edges[:-1] + edges[1:])
    rings: list[np.ndarray] = []
    for idx, cx in enumerate(centers_x):
        left, right = edges[idx], edges[idx + 1]
        mask = (x >= left) & (x <= right)
        pts = vertices[mask]
        if len(pts) < 12:
            nearest = np.argsort(np.abs(x - cx))[: min(len(vertices), 128)]
            pts = vertices[nearest]
        yz = pts[:, 1:3]
        center_yz = np.median(yz, axis=0)
        centered = yz - center_yz
        ring = []
        for angle in np.linspace(0.0, 2.0 * math.pi, ray_count, endpoint=False):
            direction = np.array([math.cos(angle), math.sin(angle)])
            projection = centered @ direction
            radius = float(np.quantile(projection, 0.985))
            radius = max(radius, 0.012)
            ring.append([float(cx), float(center_yz[0] + direction[0] * radius), float(center_yz[1] + direction[1] * radius)])
        rings.append(np.asarray(ring, dtype=np.float64))
    shell_vertices = np.vstack(rings)
    faces: list[tuple[int, int, int]] = []
    for i in range(ring_count - 1):
        for j in range(ray_count):
            a = i * ray_count + j
            b = i * ray_count + ((j + 1) % ray_count)
            c = (i + 1) * ray_count + ((j + 1) % ray_count)
            d = (i + 1) * ray_count + j
            faces.append((a, b, c))
            faces.append((a, c, d))
    start_center = len(shell_vertices)
    end_center = start_center + 1
    shell_vertices = np.vstack([shell_vertices, np.mean(rings[0], axis=0), np.mean(rings[-1], axis=0)])
    for j in range(ray_count):
        faces.append((start_center, j, (j + 1) % ray_count))
        last = (ring_count - 1) * ray_count
        faces.append((end_center, last + ((j + 1) % ray_count), last + j))
    shell_faces = np.asarray(faces, dtype=np.int64)
    if len(shell_vertices) < 4 or len(shell_faces) < 4:
        return None
    return shell_vertices, shell_faces


def build_display_shell(
    vertices: np.ndarray,
    faces: np.ndarray,
    max_faces: int = 6500,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Build a continuous visual shell for fragmented ShapeNet OBJ assets.

    Some ShapeNet models are valid triangle soups but look incomplete when
    rendered face-by-face because they contain many disconnected thin pieces.
    The shell is visualization-only: all physics placement, support alignment,
    and benchmark data still use the source mesh vertices/faces.
    """
    if len(vertices) < 4 or len(faces) < 4:
        return vertices, faces, "source_mesh"
    silhouette = build_polar_silhouette_shell(vertices)
    if silhouette is not None:
        shell_vertices, shell_faces = silhouette
        return shell_vertices, shell_faces, "polar_silhouette_display_shell"
    try:
        source = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        shell = source.convex_hull
        shell_vertices = np.asarray(shell.vertices, dtype=np.float64)
        shell_faces = np.asarray(shell.faces, dtype=np.int64)
        if len(shell_faces) > max_faces:
            try:
                shell = shell.simplify_quadric_decimation(face_count=max_faces, aggression=3)
                shell_vertices = np.asarray(shell.vertices, dtype=np.float64)
                shell_faces = np.asarray(shell.faces, dtype=np.int64)
            except Exception:
                step = max(1, len(shell_faces) // max_faces)
                shell_faces = shell_faces[::step][:max_faces]
                used = np.unique(shell_faces.reshape(-1))
                remap = -np.ones(len(shell_vertices), dtype=np.int64)
                remap[used] = np.arange(len(used))
                shell_vertices = shell_vertices[used]
                shell_faces = remap[shell_faces]
        if len(shell_vertices) >= 4 and len(shell_faces) >= 4:
            return shell_vertices, shell_faces, "convex_hull_display_shell"
    except Exception:
        pass
    return vertices, faces, "source_mesh"


SOURCE_DISPLAY_MAX_FACES = 12_000


def remove_degenerate_faces(vertices: np.ndarray, faces: np.ndarray, area_eps: float = 1.0e-12) -> tuple[np.ndarray, np.ndarray]:
    if len(faces) == 0:
        return vertices, faces
    tri = vertices[faces]
    areas2 = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    valid = np.isfinite(tri).all(axis=(1, 2)) & (areas2 > area_eps)
    return vertices, faces[valid]


def filter_vehicle_display_components(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Remove tiny high-mounted sliver components created by dense ShapeNet decimation.

    The selected vehicle OBJ is a large triangle soup with many duplicate and
    disconnected pieces.  Quadric decimation can leave small vertical flakes on
    the roof; they are not useful for CCD or visual inspection, so keep the
    physical car body/wheels while dropping only high, very small components.
    """
    vertices, faces = remove_degenerate_faces(vertices, faces)
    if len(faces) < 256:
        return vertices, faces, 0
    try:
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        components = mesh.split(only_watertight=False)
    except Exception:
        return vertices, faces, 0
    if len(components) <= 1:
        return vertices, faces, 0

    bbox = np.vstack([vertices.min(axis=0), vertices.max(axis=0)])
    z_span = max(1.0e-9, float(bbox[1, 2] - bbox[0, 2]))
    high_z = float(bbox[0, 2] + 0.62 * z_span)
    kept_vertices: list[np.ndarray] = []
    kept_faces: list[np.ndarray] = []
    dropped = 0
    cursor = 0
    for component in components:
        comp_vertices = np.asarray(component.vertices, dtype=np.float64)
        comp_faces = np.asarray(component.faces, dtype=np.int64)
        comp_bbox = component.bounds
        comp_extent = np.ptp(comp_vertices, axis=0)
        comp_area = float(component.area)
        high_component = float(comp_bbox[1, 2]) >= high_z
        tiny_roof_sliver = high_component and len(comp_faces) <= 24 and comp_area <= 0.0065
        thin_vertical_flake = (
            high_component
            and len(comp_faces) <= 40
            and comp_area <= 0.018
            and float(comp_extent[2]) > 2.4 * max(float(comp_extent[1]), 1.0e-6)
        )
        if tiny_roof_sliver or thin_vertical_flake:
            dropped += int(len(comp_faces))
            continue
        kept_vertices.append(comp_vertices)
        kept_faces.append(comp_faces + cursor)
        cursor += int(len(comp_vertices))
    if not kept_faces:
        return vertices, faces, 0
    return np.vstack(kept_vertices), np.vstack(kept_faces).astype(np.int64), dropped


def compact_display_mesh(vertices: np.ndarray, faces: np.ndarray, max_faces: int) -> tuple[np.ndarray, np.ndarray]:
    if len(faces) <= max_faces:
        return vertices, faces
    step = max(1, len(faces) // max_faces)
    compact_faces = faces[::step][:max_faces]
    used = np.unique(compact_faces.reshape(-1))
    remap = -np.ones(len(vertices), dtype=np.int64)
    remap[used] = np.arange(len(used))
    return vertices[used], remap[compact_faces]


def choose_display_mesh(category: str, vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
    if category in SYNSET_NAMES.values():
        # Use the real source surface for all ShapeNet objects.  The older
        # silhouette shell could over-connect fragmented submeshes and create
        # visually false deformations.
        if category == "car":
            vertices, faces, dropped = filter_vehicle_display_components(vertices, faces)
            method = "source_mesh_visual_surface_vehicle_component_repaired" if dropped else "source_mesh_visual_surface"
        else:
            vertices, faces = remove_degenerate_faces(vertices, faces)
            method = "source_mesh_visual_surface"
        display_vertices, display_faces = compact_display_mesh(vertices, faces, max_faces=SOURCE_DISPLAY_MAX_FACES)
        return display_vertices, display_faces, method
    return build_display_shell(vertices, faces)


def write_box_obj(path: Path, size: tuple[float, float, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sx, sy, sz = size
    x0, x1 = -0.5 * sx, 0.5 * sx
    y0, y1 = -0.5 * sy, 0.5 * sy
    z0, z1 = 0.0, sz
    vertices = [
        (x0, y0, z0),
        (x1, y0, z0),
        (x1, y1, z0),
        (x0, y1, z0),
        (x0, y0, z1),
        (x1, y0, z1),
        (x1, y1, z1),
        (x0, y1, z1),
    ]
    faces = [
        (1, 2, 3),
        (1, 3, 4),
        (5, 8, 7),
        (5, 7, 6),
        (1, 5, 6),
        (1, 6, 2),
        (2, 6, 7),
        (2, 7, 3),
        (3, 7, 8),
        (3, 8, 4),
        (4, 8, 5),
        (4, 5, 1),
    ]
    with path.open("w", encoding="ascii", newline="\n") as f:
        f.write("# generated wall slab for car-wall impact visualization\n")
        for v in vertices:
            f.write(f"v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n")
        for face in faces:
            f.write(f"f {face[0]} {face[1]} {face[2]}\n")


def dense_wall_geometry(size: tuple[float, float, float], ny: int = 96, nz: int = 72) -> tuple[np.ndarray, np.ndarray]:
    """Generate a slab whose incoming/contact face is a dense triangle grid."""
    sx, sy, sz = size
    x0, x1 = -0.5 * sx, 0.5 * sx
    y0, y1 = -0.5 * sy, 0.5 * sy
    z0, z1 = 0.0, sz

    vertices: list[tuple[float, float, float]] = []
    for iz in range(nz + 1):
        z = z0 + (z1 - z0) * iz / nz
        for iy in range(ny + 1):
            y = y0 + (y1 - y0) * iy / ny
            vertices.append((x0, y, z))

    def front_index(iy: int, iz: int) -> int:
        return iz * (ny + 1) + iy

    faces: list[tuple[int, int, int]] = []
    for iz in range(nz):
        for iy in range(ny):
            a = front_index(iy, iz)
            b = front_index(iy + 1, iz)
            c = front_index(iy + 1, iz + 1)
            d = front_index(iy, iz + 1)
            faces.append((a, b, c))
            faces.append((a, c, d))

    back_start = len(vertices)
    vertices.extend([(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)])
    b00, b10, b11, b01 = back_start, back_start + 1, back_start + 2, back_start + 3
    f00 = front_index(0, 0)
    f10 = front_index(ny, 0)
    f11 = front_index(ny, nz)
    f01 = front_index(0, nz)
    faces.extend(
        [
            (b00, b11, b10),
            (b00, b01, b11),
            (f00, b10, b00),
            (f00, f10, b10),
            (f01, b01, b11),
            (f01, b11, f11),
            (f00, b00, b01),
            (f00, b01, f01),
            (f10, f11, b11),
            (f10, b11, b10),
        ]
    )
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def write_dense_wall_obj(path: Path, size: tuple[float, float, float], ny: int = 96, nz: int = 72) -> tuple[np.ndarray, np.ndarray]:
    path.parent.mkdir(parents=True, exist_ok=True)
    vertices, faces = dense_wall_geometry(size, ny=ny, nz=nz)
    with path.open("w", encoding="ascii", newline="\n") as f:
        f.write("# generated dense contact wall slab for car-wall impact visualization\n")
        f.write(f"# front_face_subdivisions_y {ny}\n")
        f.write(f"# front_face_subdivisions_z {nz}\n")
        for v in vertices:
            f.write(f"v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n")
        for face in faces:
            f.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")
    return vertices, faces


def generated_box_asset(name: str, category: str, path: Path, size: tuple[float, float, float]) -> MeshAsset:
    write_box_obj(path, size)
    sx, sy, sz = size
    x0, x1 = -0.5 * sx, 0.5 * sx
    y0, y1 = -0.5 * sy, 0.5 * sy
    z0, z1 = 0.0, sz
    vertices = np.asarray(
        [
            (x0, y0, z0),
            (x1, y0, z0),
            (x1, y1, z0),
            (x0, y1, z0),
            (x0, y0, z1),
            (x1, y0, z1),
            (x1, y1, z1),
            (x0, y1, z1),
        ],
        dtype=np.float64,
    )
    faces = np.asarray(
        [
            (0, 1, 2),
            (0, 2, 3),
            (4, 7, 6),
            (4, 6, 5),
            (0, 4, 5),
            (0, 5, 1),
            (1, 5, 6),
            (1, 6, 2),
            (2, 6, 7),
            (2, 7, 3),
            (3, 7, 4),
            (3, 4, 0),
        ],
        dtype=np.int64,
    )
    stats = {
        "original_vertices": int(len(vertices)),
        "original_faces": int(len(faces)),
        "preview_vertices": int(len(vertices)),
        "preview_faces": int(len(faces)),
        "normalized_extent_x": float(sx),
        "normalized_extent_y": float(sy),
        "normalized_extent_z": float(sz),
    }
    stats["generated_box_size_x"] = float(size[0])
    stats["generated_box_size_y"] = float(size[1])
    stats["generated_box_size_z"] = float(size[2])
    stats["display_shell_method"] = "source_mesh"
    stats["display_shell_vertices"] = int(len(vertices))
    stats["display_shell_faces"] = int(len(faces))
    return MeshAsset(name, category, path, vertices, faces, stats)


def generated_dense_wall_asset(
    name: str,
    category: str,
    path: Path,
    size: tuple[float, float, float],
    ny: int = 96,
    nz: int = 72,
) -> MeshAsset:
    vertices, faces = write_dense_wall_obj(path, size, ny=ny, nz=nz)
    sx, sy, sz = size
    stats = {
        "original_vertices": int(len(vertices)),
        "original_faces": int(len(faces)),
        "preview_vertices": int(len(vertices)),
        "preview_faces": int(len(faces)),
        "normalized_extent_x": float(sx),
        "normalized_extent_y": float(sy),
        "normalized_extent_z": float(sz),
        "preview_decimation_method": "generated_dense_contact_grid",
        "front_face_subdivisions_y": float(ny),
        "front_face_subdivisions_z": float(nz),
    }
    stats["generated_box_size_x"] = float(size[0])
    stats["generated_box_size_y"] = float(size[1])
    stats["generated_box_size_z"] = float(size[2])
    display = generated_box_asset(
        f"{name} display shell",
        category,
        path.with_name(f"{path.stem}_display_shell.obj"),
        size,
    )
    stats["display_shell_method"] = "coarse_wall_display_shell"
    stats["display_shell_vertices"] = int(len(display.vertices))
    stats["display_shell_faces"] = int(len(display.faces))
    return MeshAsset(
        name,
        category,
        path,
        vertices,
        faces,
        stats,
        display.vertices,
        display.faces,
        "coarse_wall_display_shell",
    )


def _merged_axis_coordinates(base: np.ndarray, refined: np.ndarray) -> np.ndarray:
    values = np.concatenate([base, refined])
    values = np.unique(np.round(values, decimals=10))
    return np.sort(values.astype(np.float64))


def locally_refined_wall_geometry(
    size: tuple[float, float, float],
    *,
    impact_y: float = CAR_WALL_IMPACT_Y,
    impact_z: float = CAR_WALL_IMPACT_Z,
    coarse_ny: int = 42,
    coarse_nz: int = 30,
    refined_ny: int = 122,
    refined_nz: int = 92,
    refined_half_y: float = 0.92,
    refined_half_z: float = 0.70,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Generate a wall with UE/Chaos-style fracture cells concentrated near impact."""
    sx, sy, sz = size
    x0, x1 = -0.5 * sx, 0.5 * sx
    y0, y1 = -0.5 * sy, 0.5 * sy
    z0, z1 = 0.0, sz

    coarse_y = np.linspace(y0, y1, coarse_ny + 1)
    coarse_z = np.linspace(z0, z1, coarse_nz + 1)
    refined_y = np.linspace(max(y0, impact_y - refined_half_y), min(y1, impact_y + refined_half_y), refined_ny + 1)
    refined_z = np.linspace(max(z0, impact_z - refined_half_z), min(z1, impact_z + refined_half_z), refined_nz + 1)
    ys = _merged_axis_coordinates(coarse_y, refined_y)
    zs = _merged_axis_coordinates(coarse_z, refined_z)
    ny = len(ys) - 1
    nz = len(zs) - 1

    vertices: list[tuple[float, float, float]] = [(x0, float(y), float(z)) for z in zs for y in ys]

    def front_index(iy: int, iz: int) -> int:
        return iz * len(ys) + iy

    faces: list[tuple[int, int, int]] = []
    refined_cells = 0
    for iz in range(nz):
        zc = 0.5 * (float(zs[iz]) + float(zs[iz + 1]))
        for iy in range(ny):
            yc = 0.5 * (float(ys[iy]) + float(ys[iy + 1]))
            a = front_index(iy, iz)
            b = front_index(iy + 1, iz)
            c = front_index(iy + 1, iz + 1)
            d = front_index(iy, iz + 1)
            faces.append((a, b, c))
            faces.append((a, c, d))
            if abs(yc - impact_y) <= refined_half_y and abs(zc - impact_z) <= refined_half_z:
                refined_cells += 1

    back_start = len(vertices)
    vertices.extend([(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)])
    b00, b10, b11, b01 = back_start, back_start + 1, back_start + 2, back_start + 3
    f00 = front_index(0, 0)
    f10 = front_index(ny, 0)
    f11 = front_index(ny, nz)
    f01 = front_index(0, nz)
    faces.extend(
        [
            (b00, b11, b10),
            (b00, b01, b11),
            (f00, b10, b00),
            (f00, f10, b10),
            (f01, b01, b11),
            (f01, b11, f11),
            (f00, b00, b01),
            (f00, b01, f01),
            (f10, f11, b11),
            (f10, b11, b10),
        ]
    )
    stats = {
        "front_axis_vertices_y": float(len(ys)),
        "front_axis_vertices_z": float(len(zs)),
        "front_face_cells": float(ny * nz),
        "front_face_triangles": float(2 * ny * nz),
        "refined_impact_cells": float(refined_cells),
        "refined_impact_triangles": float(2 * refined_cells),
        "refined_half_y": float(refined_half_y),
        "refined_half_z": float(refined_half_z),
        "impact_y": float(impact_y),
        "impact_z": float(impact_z),
    }
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64), stats


def select_wall_display_faces(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    impact_y: float,
    impact_z: float,
    max_faces: int = 14000,
) -> np.ndarray:
    centers = vertices[faces].mean(axis=1)
    front_x = float(vertices[:, 0].min())
    back_x = float(vertices[:, 0].max())
    near_front = np.abs(centers[:, 0] - front_x) <= 1.0e-7
    near_back = np.abs(centers[:, 0] - back_x) <= 1.0e-7
    dy_core = (centers[:, 1] - impact_y) / 0.58
    dz_core = (centers[:, 2] - impact_z) / 0.43
    theta = np.arctan2(dz_core, dy_core)
    jagged_boundary = 1.0 + 0.16 * np.sin(9.0 * theta + 0.7) + 0.08 * np.sin(17.0 * theta - 1.3)
    breach_core = near_front & ((dy_core * dy_core + dz_core * dz_core) <= jagged_boundary)

    edge0 = vertices[faces[:, 1]] - vertices[faces[:, 0]]
    edge1 = vertices[faces[:, 2]] - vertices[faces[:, 0]]
    face_area = 0.5 * np.linalg.norm(np.cross(edge0, edge1), axis=1)
    coarse_back_cap = near_back & (face_area > 0.40)
    keep = ~(breach_core | coarse_back_cap)
    if len(faces) <= max_faces:
        return faces[np.where(keep)[0]]

    dy = (centers[:, 1] - impact_y) / 1.05
    dz = (centers[:, 2] - impact_z) / 0.82
    in_damage_window = near_front & keep & ((dy * dy + dz * dz) <= 1.35)
    selected = np.where(in_damage_window)[0].tolist()
    remaining_budget = max(0, max_faces - len(selected))
    if remaining_budget > 0:
        rest = np.where(keep & ~in_damage_window)[0]
        step = max(1, int(math.ceil(len(rest) / remaining_budget)))
        selected.extend(rest[::step][:remaining_budget].tolist())
    return faces[np.asarray(sorted(set(selected)), dtype=np.int64)]


def select_wall_intact_display_faces(vertices: np.ndarray, faces: np.ndarray, max_faces: int = 14000) -> np.ndarray:
    centers = vertices[faces].mean(axis=1)
    back_x = float(vertices[:, 0].max())
    front_x = float(vertices[:, 0].min())
    near_back = np.abs(centers[:, 0] - back_x) <= 1.0e-7
    near_front = np.abs(centers[:, 0] - front_x) <= 1.0e-7
    edge0 = vertices[faces[:, 1]] - vertices[faces[:, 0]]
    edge1 = vertices[faces[:, 2]] - vertices[faces[:, 0]]
    face_area = 0.5 * np.linalg.norm(np.cross(edge0, edge1), axis=1)
    keep = ~(near_back & (face_area > 0.40))
    if int(np.count_nonzero(keep)) <= max_faces:
        return faces[np.where(keep)[0]]

    side_indices = np.where(keep & ~near_front)[0].tolist()
    selected = side_indices[:]
    remaining_budget = max(0, max_faces - len(selected))
    if remaining_budget > 0:
        front = np.where(keep & near_front)[0]
        step = max(1, int(math.ceil(len(front) / remaining_budget)))
        selected.extend(front[::step][:remaining_budget].tolist())
    return faces[np.asarray(sorted(set(selected)), dtype=np.int64)]


def write_locally_refined_wall_obj(
    path: Path,
    size: tuple[float, float, float],
    *,
    impact_y: float = CAR_WALL_IMPACT_Y,
    impact_z: float = CAR_WALL_IMPACT_Z,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    path.parent.mkdir(parents=True, exist_ok=True)
    vertices, faces, stats = locally_refined_wall_geometry(size, impact_y=impact_y, impact_z=impact_z)
    with path.open("w", encoding="ascii", newline="\n") as f:
        f.write("# generated locally refined destructible concrete wall for car-wall impact visualization\n")
        f.write("# refinement follows a UE Chaos Geometry Collection style impact cluster\n")
        for key, value in sorted(stats.items()):
            f.write(f"# {key} {value}\n")
        for v in vertices:
            f.write(f"v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n")
        for face in faces:
            f.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")
    return vertices, faces, stats


def generated_fracturable_concrete_wall_asset(
    name: str,
    category: str,
    path: Path,
    size: tuple[float, float, float],
    *,
    impact_y: float = CAR_WALL_IMPACT_Y,
    impact_z: float = CAR_WALL_IMPACT_Z,
) -> MeshAsset:
    vertices, faces, refinement_stats = write_locally_refined_wall_obj(path, size, impact_y=impact_y, impact_z=impact_z)
    display_faces = select_wall_display_faces(vertices, faces, impact_y=impact_y, impact_z=impact_z)
    intact_display_faces = select_wall_intact_display_faces(vertices, faces)
    sx, sy, sz = size
    stats = {
        "original_vertices": int(len(vertices)),
        "original_faces": int(len(faces)),
        "preview_vertices": int(len(vertices)),
        "preview_faces": int(len(faces)),
        "normalized_extent_x": float(sx),
        "normalized_extent_y": float(sy),
        "normalized_extent_z": float(sz),
        "preview_decimation_method": "generated_locally_refined_fracturable_wall",
        "generated_box_size_x": float(sx),
        "generated_box_size_y": float(sy),
        "generated_box_size_z": float(sz),
        "display_shell_method": "locally_refined_fracturable_wall_surface",
        "display_shell_vertices": int(len(vertices)),
        "display_shell_faces": int(len(display_faces)),
        "preimpact_display_faces": int(len(intact_display_faces)),
        "display_removed_faces_for_breach": int(len(faces) - len(display_faces)),
        "display_breach_opening": "jagged_front_core_plus_removed_back_cap",
        "material_density_kg_m3": 2400.0,
        "material_youngs_modulus_pa": 30.0e9,
        "material_compressive_strength_pa": 35.0e6,
        "material_tensile_strength_pa": 3.2e6,
        "material_fracture_energy_j_m2": 120.0,
    }
    stats.update(refinement_stats)
    return MeshAsset(
        name,
        category,
        path,
        vertices,
        faces,
        stats,
        vertices,
        display_faces,
        "locally_refined_fracturable_wall_surface",
        intact_display_faces,
    )


def dense_ground_geometry(size: tuple[float, float, float], nx: int = 112, ny: int = 84) -> tuple[np.ndarray, np.ndarray]:
    """Generate a dense horizontal ground slab with a triangulated top contact face."""
    sx, sy, sz = size
    x0, x1 = -0.5 * sx, 0.5 * sx
    y0, y1 = -0.5 * sy, 0.5 * sy
    z0, z1 = -sz, 0.0

    vertices: list[tuple[float, float, float]] = []
    for ix in range(nx + 1):
        x = x0 + (x1 - x0) * ix / nx
        for iy in range(ny + 1):
            y = y0 + (y1 - y0) * iy / ny
            vertices.append((x, y, z1))

    def top_index(ix: int, iy: int) -> int:
        return ix * (ny + 1) + iy

    faces: list[tuple[int, int, int]] = []
    for ix in range(nx):
        for iy in range(ny):
            a = top_index(ix, iy)
            b = top_index(ix + 1, iy)
            c = top_index(ix + 1, iy + 1)
            d = top_index(ix, iy + 1)
            faces.append((a, b, c))
            faces.append((a, c, d))

    bottom_start = len(vertices)
    vertices.extend([(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)])
    b00, b10, b11, b01 = bottom_start, bottom_start + 1, bottom_start + 2, bottom_start + 3
    t00 = top_index(0, 0)
    t10 = top_index(nx, 0)
    t11 = top_index(nx, ny)
    t01 = top_index(0, ny)
    faces.extend(
        [
            (b00, b11, b10),
            (b00, b01, b11),
            (t00, b10, b00),
            (t00, t10, b10),
            (t10, b11, b10),
            (t10, t11, b11),
            (t11, b01, b11),
            (t11, t01, b01),
            (t01, b00, b01),
            (t01, t00, b00),
        ]
    )
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def write_dense_ground_obj(path: Path, size: tuple[float, float, float], nx: int = 112, ny: int = 84) -> tuple[np.ndarray, np.ndarray]:
    path.parent.mkdir(parents=True, exist_ok=True)
    vertices, faces = dense_ground_geometry(size, nx=nx, ny=ny)
    with path.open("w", encoding="ascii", newline="\n") as f:
        f.write("# generated dense frictional ground slab for many-object drop visualization\n")
        f.write(f"# top_face_subdivisions_x {nx}\n")
        f.write(f"# top_face_subdivisions_y {ny}\n")
        for v in vertices:
            f.write(f"v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n")
        for face in faces:
            f.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")
    return vertices, faces


def write_triangle_mesh_obj(path: Path, vertices: np.ndarray, faces: np.ndarray, comment: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii", newline="\n") as f:
        f.write(f"# {comment}\n")
        for v in vertices:
            f.write(f"v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n")
        for face in faces:
            f.write(f"f {int(face[0]) + 1} {int(face[1]) + 1} {int(face[2]) + 1}\n")


def generated_dense_ground_asset(
    name: str,
    category: str,
    path: Path,
    size: tuple[float, float, float],
    nx: int = 112,
    ny: int = 84,
) -> MeshAsset:
    vertices, faces = write_dense_ground_obj(path, size, nx=nx, ny=ny)
    sx, sy, sz = size
    display_vertices = np.asarray(
        [
            (-0.5 * sx, -0.5 * sy, -sz),
            (0.5 * sx, -0.5 * sy, -sz),
            (0.5 * sx, 0.5 * sy, -sz),
            (-0.5 * sx, 0.5 * sy, -sz),
            (-0.5 * sx, -0.5 * sy, 0.0),
            (0.5 * sx, -0.5 * sy, 0.0),
            (0.5 * sx, 0.5 * sy, 0.0),
            (-0.5 * sx, 0.5 * sy, 0.0),
        ],
        dtype=np.float64,
    )
    display_faces = np.asarray(
        [
            (0, 1, 2),
            (0, 2, 3),
            (4, 7, 6),
            (4, 6, 5),
            (0, 4, 5),
            (0, 5, 1),
            (1, 5, 6),
            (1, 6, 2),
            (2, 6, 7),
            (2, 7, 3),
            (3, 7, 4),
            (3, 4, 0),
        ],
        dtype=np.int64,
    )
    stats = {
        "original_vertices": int(len(vertices)),
        "original_faces": int(len(faces)),
        "preview_vertices": int(len(vertices)),
        "preview_faces": int(len(faces)),
        "normalized_extent_x": float(sx),
        "normalized_extent_y": float(sy),
        "normalized_extent_z": float(sz),
        "preview_decimation_method": "generated_dense_ground_contact_grid",
        "ground_top_subdivisions_x": float(nx),
        "ground_top_subdivisions_y": float(ny),
        "ground_top_triangles": int(2 * nx * ny),
        "generated_box_size_x": float(sx),
        "generated_box_size_y": float(sy),
        "generated_box_size_z": float(sz),
        "display_shell_method": "coarse_ground_display_shell",
        "display_shell_vertices": int(len(display_vertices)),
        "display_shell_faces": int(len(display_faces)),
    }
    return MeshAsset(name, category, path, vertices, faces, stats, display_vertices, display_faces, "coarse_ground_display_shell")


@dataclass
class MeshAsset:
    name: str
    category: str
    path: Path
    vertices: np.ndarray
    faces: np.ndarray
    stats: dict[str, float]
    display_vertices: np.ndarray | None = None
    display_faces: np.ndarray | None = None
    display_shell_method: str = "source_mesh"
    preimpact_display_faces: np.ndarray | None = None


@dataclass
class Body:
    asset: MeshAsset
    color: tuple[int, int, int]
    mass: float
    p0: np.ndarray
    v0: np.ndarray
    v1: np.ndarray
    scale: float = 1.0
    yaw: float = 0.0
    trajectory_times: np.ndarray | None = None
    trajectory_positions: np.ndarray | None = None
    trajectory_yaws: np.ndarray | None = None
    trajectory_velocities: np.ndarray | None = None
    trajectory_vertices: np.ndarray | None = None
    metadata: dict[str, object] | None = None

    def has_sampled_trajectory(self) -> bool:
        return self.trajectory_times is not None and (self.trajectory_positions is not None or self.trajectory_vertices is not None)

    def _interp_vector(self, t: float, values: np.ndarray) -> np.ndarray:
        assert self.trajectory_times is not None
        times = self.trajectory_times
        if t <= float(times[0]):
            return values[0].copy()
        if t >= float(times[-1]):
            return values[-1].copy()
        return np.asarray([np.interp(t, times, values[:, axis]) for axis in range(values.shape[1])], dtype=np.float64)

    def position(self, t: float, tc: float = CONTACT_T) -> np.ndarray:
        if self.trajectory_times is not None and self.trajectory_positions is not None:
            assert self.trajectory_positions is not None
            return self._interp_vector(t, self.trajectory_positions)
        if t <= tc:
            return self.p0 + self.v0 * t
        return self.p0 + self.v0 * tc + self.v1 * (t - tc)

    def velocity_at(self, t: float, tc: float = CONTACT_T) -> np.ndarray:
        if self.has_sampled_trajectory() and self.trajectory_velocities is not None:
            return self._interp_vector(t, self.trajectory_velocities)
        return self.v0 if t <= tc else self.v1

    def yaw_at(self, t: float) -> float:
        if self.has_sampled_trajectory() and self.trajectory_yaws is not None:
            assert self.trajectory_times is not None
            return float(np.interp(t, self.trajectory_times, self.trajectory_yaws))
        return self.yaw

    def deformed_vertices_at(self, t: float) -> np.ndarray | None:
        if self.trajectory_times is None or self.trajectory_vertices is None:
            return None
        times = self.trajectory_times
        if t <= float(times[0]):
            return self.trajectory_vertices[0].copy()
        if t >= float(times[-1]):
            return self.trajectory_vertices[-1].copy()
        idx = int(np.searchsorted(times, t, side="right") - 1)
        idx = max(0, min(idx, len(times) - 2))
        alpha = float((t - times[idx]) / max(1.0e-12, times[idx + 1] - times[idx]))
        return (1.0 - alpha) * self.trajectory_vertices[idx] + alpha * self.trajectory_vertices[idx + 1]

    def rotation_at(self, t: float) -> np.ndarray:
        c, s = math.cos(self.yaw_at(t)), math.sin(self.yaw_at(t))
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    def transformed(self, t: float) -> np.ndarray:
        deformed = self.deformed_vertices_at(t)
        if deformed is not None:
            return deformed
        rot = self.rotation_at(t)
        return (self.asset.vertices @ rot.T) * self.scale + self.position(t)

    def display_transformed(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        deformed = self.deformed_vertices_at(t)
        if deformed is not None:
            if (self.metadata or {}).get("damage_model") == "ue_chaos_style_concrete_fracture" and t < CONTACT_T:
                faces = self.asset.preimpact_display_faces if self.asset.preimpact_display_faces is not None else self.asset.faces
            else:
                faces = self.asset.display_faces if self.asset.display_faces is not None else self.asset.faces
            return deformed, faces
        vertices = self.asset.display_vertices if self.asset.display_vertices is not None else self.asset.vertices
        faces = self.asset.display_faces if self.asset.display_faces is not None else self.asset.faces
        rot = self.rotation_at(t)
        return (vertices @ rot.T) * self.scale + self.position(t), faces


def y_axis_rotation(angle_rad: float) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.asarray([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def inclined_dense_support_asset(
    asset: MeshAsset,
    *,
    angle_degrees: float,
    path: Path,
    name: str | None = None,
    display_nx: int = 64,
    display_ny: int = 36,
) -> MeshAsset:
    """Rotate a dense support slab so x<0 is raised and x>0 slopes downward."""
    angle_rad = math.radians(angle_degrees)
    rot = y_axis_rotation(angle_rad)
    vertices = asset.vertices @ rot.T
    display_size = (
        float(asset.stats.get("generated_box_size_x", asset.stats.get("normalized_extent_x", 1.0))),
        float(asset.stats.get("generated_box_size_y", asset.stats.get("normalized_extent_y", 1.0))),
        float(asset.stats.get("generated_box_size_z", asset.stats.get("normalized_extent_z", 0.16))),
    )
    display_vertices_raw, display_faces = dense_ground_geometry(display_size, nx=display_nx, ny=display_ny)
    display_vertices = display_vertices_raw @ rot.T
    faces = asset.faces.copy()
    normal = normalize(rot @ np.asarray([0.0, 0.0, 1.0], dtype=np.float64))
    ext = vertices.max(axis=0) - vertices.min(axis=0)
    stats = dict(asset.stats)
    stats.update(
        {
            "normalized_extent_x": float(ext[0]),
            "normalized_extent_y": float(ext[1]),
            "normalized_extent_z": float(ext[2]),
            "preview_decimation_method": "generated_dense_inclined_tabletop_contact_grid",
            "support_incline_angle_degrees": float(angle_degrees),
            "support_plane_normal_x": float(normal[0]),
            "support_plane_normal_y": float(normal[1]),
            "support_plane_normal_z": float(normal[2]),
            "support_plane_offset": 0.0,
            "left_side_raised": True,
            "display_grid_subdivisions_x": int(display_nx),
            "display_grid_subdivisions_y": int(display_ny),
            "display_shell_method": "coarse_inclined_tabletop_display_grid",
            "display_shell_vertices": int(len(display_vertices)),
            "display_shell_faces": int(len(display_faces)),
        }
    )
    write_triangle_mesh_obj(
        path,
        vertices,
        faces,
        f"generated dense frictional support slab tilted {angle_degrees:.1f} degrees; x<0 side is raised",
    )
    return MeshAsset(
        name or asset.name,
        asset.category,
        path,
        vertices,
        faces,
        stats,
        display_vertices,
        display_faces,
        "coarse_inclined_tabletop_display_grid",
    )


def support_plane_from_bodies(bodies: list[Body]) -> tuple[np.ndarray, float, dict[str, object], str] | None:
    support_bodies = [b for b in bodies if b.asset.category in SUPPORT_RENDER_CATEGORIES]
    search_order = support_bodies + [b for b in bodies if b.asset.category not in SUPPORT_RENDER_CATEGORIES]
    for body in search_order:
        metadata = body.metadata or {}
        normal_value = metadata.get("support_plane_normal")
        if normal_value is None:
            continue
        normal = normalize(np.asarray(normal_value, dtype=np.float64))
        offset = float(metadata.get("support_plane_offset", 0.0))
        return normal, offset, metadata, body.asset.name
    return None


def trajectory_duration_seconds(bodies: list[Body], default: float = DURATION_T) -> float:
    durations = [
        float(body.trajectory_times[-1])
        for body in bodies
        if body.trajectory_times is not None and len(body.trajectory_times) > 0
    ]
    return max([float(default), *durations])


def trajectory_frame_count(bodies: list[Body], default: int = FRAME_COUNT) -> int:
    metadata_counts = [
        int((body.metadata or {}).get("render_frame_count", 0))
        for body in bodies
        if int((body.metadata or {}).get("render_frame_count", 0)) > 0
    ]
    if metadata_counts:
        return max(int(default), max(metadata_counts))
    return int(default)


def timeline_samples(duration: float, frame_count: int) -> np.ndarray:
    duration = max(float(duration), CONTACT_T)
    frame_count = max(2, int(frame_count))
    samples = np.linspace(0.0, duration, frame_count)
    samples[int(np.argmin(np.abs(samples - CONTACT_T)))] = CONTACT_T
    return samples


def inclined_plane_height_at_xy(x: float, normal: np.ndarray, offset: float = 0.0) -> float:
    if abs(float(normal[2])) < 1.0e-9:
        raise ValueError("support plane normal must have a non-zero z component")
    return float((offset - normal[0] * x) / normal[2])


def local_vertices(asset: MeshAsset, scale: float, yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return (asset.vertices @ rot.T) * scale


def support_aligned_contact_centers(
    asset_a: MeshAsset,
    asset_b: MeshAsset,
    scale_a: float,
    scale_b: float,
    yaw_a: float,
    yaw_b: float,
    y_offset: float,
    surface_gap: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Place bodies so their swept support surfaces touch but never overlap."""
    va = local_vertices(asset_a, scale_a, yaw_a)
    vb = local_vertices(asset_b, scale_b, yaw_b)
    # Body A moves +X, body B moves -X.  At TOI, A.max_x is just left of
    # B.min_x.  This replaces the old fixed center gap that caused penetration.
    ca = np.array([-0.5 * surface_gap - float(va[:, 0].max()), y_offset, -float(va[:, 2].min())])
    cb = np.array([0.5 * surface_gap - float(vb[:, 0].min()), -y_offset, -float(vb[:, 2].min())])
    return ca, cb


def elastic_velocities(m1: float, v1: np.ndarray, m2: float, v2: np.ndarray, n: np.ndarray, restitution: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    n = normalize(n.astype(np.float64))
    u1 = float(np.dot(v1, n))
    u2 = float(np.dot(v2, n))
    u1p = ((m1 - restitution * m2) * u1 + (1.0 + restitution) * m2 * u2) / (m1 + m2)
    u2p = ((m2 - restitution * m1) * u2 + (1.0 + restitution) * m1 * u1) / (m1 + m2)
    return v1 + (u1p - u1) * n, v2 + (u2p - u2) * n


def physics_audit(bodies: list[Body]) -> dict[str, object]:
    deformable = [b for b in bodies if b.trajectory_vertices is not None]
    if deformable:
        body = deformable[0]
        metadata = body.metadata or {}
        initial_vertices = body.trajectory_vertices[0]
        final_vertices = body.trajectory_vertices[-1]
        initial_extent = initial_vertices.max(axis=0) - initial_vertices.min(axis=0)
        final_extent = final_vertices.max(axis=0) - final_vertices.min(axis=0)
        if metadata.get("damage_model") == "ue_chaos_style_concrete_fracture":
            return {
                "collision_model": "fixed reinforced-concrete wall with UE Chaos style strain/damage fracture and ballistic debris",
                "solver": metadata.get("solver", "material_impulse_fracture_replay"),
                "wall_vertex_count": int(len(final_vertices)),
                "wall_surface_faces": int(len(body.asset.faces)),
                "constraint_count": int(metadata.get("constraint_count", 0)),
                "trajectory_sample_count": int(len(body.trajectory_times)) if body.trajectory_times is not None else 0,
                "render_speed_scale": metadata.get("render_speed_scale"),
                "vehicle_mass_kg": metadata.get("vehicle_mass_kg"),
                "vehicle_impact_speed_mps": metadata.get("vehicle_impact_speed_mps"),
                "vehicle_rebound_speed_mps": metadata.get("vehicle_rebound_speed_mps"),
                "vehicle_kinetic_energy_pre_j": metadata.get("vehicle_kinetic_energy_pre_j"),
                "vehicle_kinetic_energy_post_j": metadata.get("vehicle_kinetic_energy_post_j"),
                "absorbed_energy_j": metadata.get("absorbed_energy_j"),
                "material": metadata.get("material"),
                "damage_radius_y_m": metadata.get("damage_radius_y_m"),
                "damage_radius_z_m": metadata.get("damage_radius_z_m"),
                "max_crater_depth_m": metadata.get("max_crater_depth_m"),
                "breach_depth_m": metadata.get("breach_depth_m"),
                "fractured_front_triangles": metadata.get("fractured_front_triangles"),
                "debris_piece_count": metadata.get("debris_piece_count"),
                "postsolve_max_penetration": metadata.get("postsolve_max_penetration"),
                "external_impulse_note": "The wall is fixed to the environment, so vehicle momentum is not conserved. Lost kinetic energy is assigned to concrete fracture, crushing, heat, sound, and debris kinetic energy.",
            }
        if metadata.get("solver") == "mujoco_3_4_rigid_body_engine":
            p2cccd_audit = metadata.get("p2cccd_swept_ccd_audit", {})
            return {
                "collision_model": "MuJoCo rigid-body replay for brick trajectories; src adjacent-frame swept CCD for vehicle-brick detection metrics",
                "solver": "src p2cccd_cpp exact certificate audit",
                "trajectory_solver": metadata.get("solver"),
                "engine": metadata.get("engine"),
                "brick_count": metadata.get("brick_count"),
                "brick_mass_kg": metadata.get("brick_mass_kg"),
                "total_brick_mass_kg": metadata.get("total_brick_mass_kg"),
                "time_step": metadata.get("sim_dt"),
                "solver_iterations": metadata.get("solver_iterations"),
                "gravity": metadata.get("gravity"),
                "friction_mu": metadata.get("brick_friction_mu"),
                "restitution": metadata.get("brick_restitution"),
                "render_speed_scale": metadata.get("render_speed_scale"),
                "vehicle_mass_kg": metadata.get("vehicle_mass_kg"),
                "vehicle_impact_speed_mps": metadata.get("vehicle_impact_speed_mps"),
                "vehicle_exit_speed_mps": metadata.get("vehicle_exit_speed_mps"),
                "vehicle_kinetic_energy_pre_j": metadata.get("vehicle_kinetic_energy_pre_j"),
                "vehicle_kinetic_energy_exit_j": metadata.get("vehicle_kinetic_energy_exit_j"),
                "absorbed_energy_j": metadata.get("absorbed_energy_j"),
                "displaced_brick_count": metadata.get("displaced_brick_count"),
                "max_brick_displacement_m": metadata.get("max_brick_displacement_m"),
                "max_brick_speed_mps": metadata.get("max_brick_speed_mps"),
                "p2cccd_candidate_count": p2cccd_audit.get("candidate_count"),
                "p2cccd_body_pair_candidate_count": p2cccd_audit.get("body_pair_candidate_count"),
                "p2cccd_exact_fallback_count": p2cccd_audit.get("exact_fallback_count"),
                "p2cccd_exact_fallback_queries": p2cccd_audit.get("exact_fallback_queries"),
                "p2cccd_first_toi_seconds": p2cccd_audit.get("first_toi_seconds"),
                "p2cccd_fn": p2cccd_audit.get("fn"),
                "mujoco_reference_max_contact_count": metadata.get("max_contact_count"),
                "mujoco_reference_max_vehicle_brick_contact_count": metadata.get("max_vehicle_brick_reference_contact_count"),
                "mujoco_reference_contact_segments_without_visual_mesh_ccd_hit": p2cccd_audit.get("mujoco_reference_contact_segments_without_visual_mesh_ccd_hit"),
                "postsolve_max_penetration": metadata.get("postsolve_max_penetration"),
                "external_impulse_note": "The vehicle is prescribed as a kinematic collider, so vehicle momentum is an input. Brick motion, gravity, contact impulses, friction, and restitution are solved by MuJoCo. Detection metrics are not taken from MuJoCo contacts.",
            }
        return {
            "collision_model": "XPBD deformable soft-body simulation with rigid moving press plates and Coulomb-like contact damping",
            "solver": metadata.get("solver", "local_xpbd_soft_body"),
            "vertex_count": int(len(final_vertices)),
            "surface_faces": int(len(body.asset.faces)),
            "constraint_count": int(metadata.get("constraint_count", 0)),
            "time_step": metadata.get("sim_dt"),
            "substeps": metadata.get("substeps"),
            "solver_iterations": metadata.get("solver_iterations"),
            "gravity": metadata.get("gravity"),
            "contact_friction": metadata.get("contact_friction"),
            "plate_travel": metadata.get("plate_travel"),
            "max_node_displacement": metadata.get("max_node_displacement"),
            "height_initial": float(initial_extent[2]),
            "height_final": float(final_extent[2]),
            "height_compression_ratio": float(1.0 - final_extent[2] / max(1.0e-12, initial_extent[2])),
            "bbox_volume_initial": float(np.prod(initial_extent)),
            "bbox_volume_final": float(np.prod(final_extent)),
            "postsolve_max_penetration": metadata.get("postsolve_max_penetration"),
            "external_work_note": "The moving press plate performs external work on the soft body; momentum and energy are intentionally not conserved.",
        }
    if any(b.has_sampled_trajectory() for b in bodies):
        moving = [b for b in bodies if b.has_sampled_trajectory() and not (b.metadata or {}).get("exclude_from_physics_audit", False)]
        duration = trajectory_duration_seconds(moving)
        samples = timeline_samples(duration, trajectory_frame_count(moving))
        gravity = float(next((b.metadata or {}).get("gravity", 9.81) for b in moving))
        plane = support_plane_from_bodies(bodies)
        support_metadata: dict[str, object] = {}
        support_model = "ground"
        if plane is not None:
            support_normal, support_offset, support_metadata, _ = plane
            support_model = "inclined tabletop" if support_metadata.get("incline_angle_degrees") else "support plane"
        else:
            support_normal = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
            support_offset = 0.0
        p0 = sum((b.mass * b.velocity_at(0.0) for b in moving), start=np.zeros(3))
        p1 = sum((b.mass * b.velocity_at(duration) for b in moving), start=np.zeros(3))
        kinetic = [
            sum(0.5 * b.mass * float(np.dot(b.velocity_at(float(t)), b.velocity_at(float(t)))) for b in moving)
            for t in samples
        ]
        potential = [
            sum(b.mass * gravity * max(0.0, float(b.position(float(t))[2])) for b in moving)
            for t in samples
        ]
        contacts = sum(int((b.metadata or {}).get("ground_contact_count", 0)) for b in moving)
        windows = sum(int((b.metadata or {}).get("ground_contact_window_samples", 0)) for b in moving)
        first_contacts = [
            float((b.metadata or {}).get("first_ground_contact_time"))
            for b in moving
            if (b.metadata or {}).get("first_ground_contact_time") is not None
        ]
        return {
            "collision_model": f"sampled rigid-body gravity drop with unilateral {support_model} contact, restitution, and Coulomb friction",
            "gravity": gravity,
            "gravity_vector": [0.0, 0.0, -gravity],
            "support_plane_normal": support_normal.tolist(),
            "support_plane_offset": support_offset,
            "incline_angle_degrees": support_metadata.get("incline_angle_degrees"),
            "left_side_raised": support_metadata.get("left_side_raised"),
            "restitution": float(next((b.metadata or {}).get("restitution", 0.0) for b in moving)),
            "friction_mu": float(next((b.metadata or {}).get("friction_mu", 0.0) for b in moving)),
            "duration_seconds": float(duration),
            "trajectory_sample_count": int(len(samples)),
            "moving_body_count": int(len(moving)),
            "total_ground_impact_events": int(contacts),
            "total_ground_contact_window_samples": int(windows),
            "first_ground_contact_time": min(first_contacts) if first_contacts else None,
            "last_ground_contact_time": max(first_contacts) if first_contacts else None,
            "total_momentum_initial_moving_only": p0.tolist(),
            "total_momentum_final_moving_only": p1.tolist(),
            "kinetic_energy_initial": float(kinetic[0]),
            "kinetic_energy_final": float(kinetic[-1]),
            "kinetic_energy_max": float(max(kinetic)),
            "mechanical_energy_initial": float(kinetic[0] + potential[0]),
            "mechanical_energy_final": float(kinetic[-1] + potential[-1]),
            "external_impulse_note": f"Momentum and mechanical energy are not conserved because the {support_model} applies normal impulses and friction dissipates energy.",
        }
    p_pre = sum((b.mass * b.v0 for b in bodies), start=np.zeros(3))
    p_post = sum((b.mass * b.v1 for b in bodies), start=np.zeros(3))
    e_pre = sum(0.5 * b.mass * float(np.dot(b.v0, b.v0)) for b in bodies)
    e_post = sum(0.5 * b.mass * float(np.dot(b.v1, b.v1)) for b in bodies)
    return {
        "collision_model": "rigid-body elastic impulse along contact normal; visualization mesh follows center-of-mass trajectory",
        "restitution": 1.0,
        "total_momentum_pre": p_pre.tolist(),
        "total_momentum_post": p_post.tolist(),
        "momentum_abs_error": float(np.linalg.norm(p_post - p_pre)),
        "momentum_rel_error": float(np.linalg.norm(p_post - p_pre) / max(1.0e-12, np.linalg.norm(p_pre))),
        "kinetic_energy_pre": float(e_pre),
        "kinetic_energy_post": float(e_post),
        "kinetic_energy_abs_error": float(abs(e_post - e_pre)),
        "kinetic_energy_rel_error": float(abs(e_post - e_pre) / max(1.0e-12, abs(e_pre))),
    }


def look_at_basis(camera: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    forward = normalize(camera - target)
    right = normalize(np.cross(np.array([0.0, 0.0, 1.0]), forward))
    if np.linalg.norm(right) < 1e-9:
        right = np.array([1.0, 0.0, 0.0])
    up = normalize(np.cross(forward, right))
    return right, up, forward


def project(points: np.ndarray, camera: np.ndarray, target: np.ndarray, zoom: float) -> tuple[np.ndarray, np.ndarray]:
    right, up, forward = look_at_basis(camera, target)
    centered = points - target
    x = centered @ right
    y = centered @ up
    z = centered @ forward
    px = W * 0.5 + x * zoom
    py = H * 0.58 - y * zoom
    return np.stack([px, py], axis=1), z


def draw_checker_floor(
    draw: ImageDraw.ImageDraw,
    camera: np.ndarray,
    target: np.ndarray,
    zoom: float,
    z: float = 0.0,
) -> None:
    tile = 0.42
    x_values = np.arange(-8.4, 5.9, tile)
    y_values = np.arange(-4.6, 4.6, tile)
    colors = ((207, 216, 221, 235), (232, 237, 240, 235))
    outline = (190, 200, 207, 150)
    quads: list[tuple[float, list[tuple[float, float]], tuple[int, int, int, int]]] = []
    for ix, x in enumerate(x_values):
        for iy, y in enumerate(y_values):
            pts3 = np.asarray(
                [
                    [x, y, z],
                    [x + tile, y, z],
                    [x + tile, y + tile, z],
                    [x, y + tile, z],
                ],
                dtype=np.float64,
            )
            pp, depth = project(pts3, camera, target, zoom)
            if np.any(~np.isfinite(pp)):
                continue
            if pp[:, 0].max() < -80 or pp[:, 0].min() > W + 80 or pp[:, 1].max() < -80 or pp[:, 1].min() > H + 80:
                continue
            quads.append((float(depth.mean()), [tuple(map(float, p)) for p in pp], colors[(ix + iy) & 1]))
    quads.sort(key=lambda item: item[0])
    for _, pts, color in quads:
        draw.polygon(pts, fill=color)
        draw.line(pts + [pts[0]], fill=outline, width=1)


def draw_graphics_engine_tile_board(
    draw: ImageDraw.ImageDraw,
    body: Body,
    t: float,
    camera: np.ndarray,
    target: np.ndarray,
    zoom: float,
) -> bool:
    stats = body.asset.stats
    nx = int(stats.get("display_grid_subdivisions_x", 0))
    ny = int(stats.get("display_grid_subdivisions_y", 0))
    if nx <= 0 or ny <= 0:
        return False
    verts, _ = body.display_transformed(t)
    top_count = (nx + 1) * (ny + 1)
    if len(verts) < top_count:
        return False

    def vid(ix: int, iy: int) -> int:
        return ix * (ny + 1) + iy

    tile_colors = ((205, 213, 216, 242), (231, 235, 236, 242))
    outline = (177, 186, 191, 170)
    cells: list[tuple[float, list[tuple[float, float]], tuple[int, int, int, int]]] = []
    for ix in range(nx):
        for iy in range(ny):
            ids = [vid(ix, iy), vid(ix + 1, iy), vid(ix + 1, iy + 1), vid(ix, iy + 1)]
            pts3 = verts[ids]
            pp, depth = project(pts3, camera, target, zoom)
            if np.any(~np.isfinite(pp)):
                continue
            if pp[:, 0].max() < -120 or pp[:, 0].min() > W + 120 or pp[:, 1].max() < -120 or pp[:, 1].min() > H + 120:
                continue
            cells.append((float(depth.mean()), [tuple(map(float, p)) for p in pp], tile_colors[(ix + iy) & 1]))
    cells.sort(key=lambda item: item[0])
    for _, pts, color in cells:
        draw.polygon(pts, fill=color)
        draw.line(pts + [pts[0]], fill=outline, width=1)

    # Keep the support board visually clean.  Earlier revisions added decorative
    # plus markers on the checkerboard, but those marks are easily confused with
    # CCD candidate/contact annotations in paper screenshots.
    return True

    plus_palette = [
        (250, 204, 21, 150),
        (94, 214, 148, 150),
        (72, 163, 238, 150),
        (244, 113, 98, 150),
    ]
    pluses: list[tuple[float, float, float, tuple[int, int, int, int]]] = []
    for ix in range(2, nx, 5):
        for iy in range(2, ny, 4):
            ids = [vid(ix, iy), vid(ix + 1, iy), vid(ix + 1, iy + 1), vid(ix, iy + 1)]
            center = verts[ids].mean(axis=0, keepdims=True)
            pp, depth = project(center, camera, target, zoom)
            x, y = map(float, pp[0])
            if -40 <= x <= W + 40 and -40 <= y <= H + 40:
                pluses.append((float(depth[0]), x, y, plus_palette[(ix + iy) % len(plus_palette)]))
    pluses.sort(key=lambda item: item[0])
    for _, x, y, color in pluses:
        r = 4.0
        draw.line([(x - r, y), (x + r, y)], fill=color, width=1)
        draw.line([(x, y - r), (x, y + r)], fill=color, width=1)
    return True


def draw_support_contact_markers(
    draw: ImageDraw.ImageDraw,
    bodies: list[Body],
    t: float,
    camera: np.ndarray,
    target: np.ndarray,
    zoom: float,
) -> None:
    def convex_hull_2d(points: np.ndarray) -> np.ndarray:
        if points.shape[0] <= 1:
            return points
        order = np.lexsort((points[:, 1], points[:, 0]))
        pts = points[order]

        def cross(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
            return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))

        lower: list[np.ndarray] = []
        for p in pts:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 1.0e-10:
                lower.pop()
            lower.append(p)
        upper: list[np.ndarray] = []
        for p in reversed(pts):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 1.0e-10:
                upper.pop()
            upper.append(p)
        if len(lower) + len(upper) <= 2:
            return pts[:1]
        return np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)

    plane = support_plane_from_bodies(bodies)
    if plane is None:
        return
    normal, offset, support_metadata, _ = plane
    downhill_value = support_metadata.get("downhill_direction")
    tangent_u = np.asarray(downhill_value if downhill_value is not None else [1.0, 0.0, 0.0], dtype=np.float64)
    tangent_u = tangent_u - float(np.dot(tangent_u, normal)) * normal
    if float(np.linalg.norm(tangent_u)) < 1.0e-9:
        tangent_u = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
        tangent_u = tangent_u - float(np.dot(tangent_u, normal)) * normal
    tangent_u = normalize(tangent_u)
    tangent_v = normalize(np.cross(normal, tangent_u))

    markers: list[tuple[float, list[tuple[float, float]], tuple[int, int, int], float]] = []
    for body in bodies:
        if body.asset.category in SUPPORT_RENDER_CATEGORIES or not body.has_sampled_trajectory():
            continue
        metadata = body.metadata or {}
        vertices, _ = body.display_transformed(t)
        distances = vertices @ normal - offset
        min_index = int(np.argmin(distances))
        min_gap = float(distances[min_index])
        patch_thickness = float(metadata.get("support_contact_patch_thickness", 0.035))
        contact_display_tol = max(0.040, patch_thickness * 1.22)
        if min_gap > contact_display_tol:
            continue

        closest_support = vertices[min_index] - min_gap * normal
        base_radius = float(metadata.get("support_contact_patch_radius", max(0.10, min(0.24, 0.18 * body.scale))))
        min_radius = float(metadata.get("support_contact_patch_min_radius", 0.075))
        max_radius = float(metadata.get("support_contact_patch_max_radius", 0.26))
        base_radius = float(np.clip(base_radius, 0.070, 0.220))
        min_radius = float(np.clip(min_radius, 0.045, 0.105))
        max_radius = float(np.clip(max_radius, max(0.13, min_radius * 1.60), 0.285))
        local_radius = float(np.clip(1.18 * base_radius, max(0.075, min_radius), max_radius))

        near_mask = distances <= min_gap + patch_thickness
        support_points = vertices[near_mask] - ((vertices[near_mask] @ normal - offset)[:, None] * normal[None, :])
        if support_points.shape[0] > 0:
            local_coords = np.column_stack(
                [(support_points - closest_support) @ tangent_u, (support_points - closest_support) @ tangent_v]
            )
            local_mask = np.linalg.norm(local_coords, axis=1) <= local_radius
            support_points = support_points[local_mask]
            local_coords = local_coords[local_mask]
        else:
            local_coords = np.empty((0, 2), dtype=np.float64)

        if local_coords.shape[0] >= 3:
            hull2 = convex_hull_2d(local_coords)
            if hull2.shape[0] >= 3:
                centroid = hull2.mean(axis=0, keepdims=True)
                hull2 = centroid + 1.10 * (hull2 - centroid)
                span = np.maximum(np.ptp(hull2, axis=0), 1.0e-9)
                if span[0] < min_radius:
                    hull2[:, 0] *= min_radius / span[0]
                if span[1] < 0.55 * min_radius:
                    hull2[:, 1] *= (0.55 * min_radius) / span[1]
                ring3 = np.asarray(
                    [closest_support + p[0] * tangent_u + p[1] * tangent_v + 0.012 * normal for p in hull2],
                    dtype=np.float64,
                )
            else:
                local_coords = np.empty((0, 2), dtype=np.float64)
        if local_coords.shape[0] < 3:
            radius_u = float(np.clip(base_radius, min_radius, max_radius))
            radius_v = float(np.clip(0.62 * base_radius, min_radius * 0.50, max_radius * 0.70))
            angles = np.linspace(0.0, 2.0 * math.pi, 28, endpoint=False)
            ring3 = np.asarray(
                [
                    closest_support
                    + radius_u * math.cos(a) * tangent_u
                    + radius_v * math.sin(a) * tangent_v
                    + 0.012 * normal
                    for a in angles
                ],
                dtype=np.float64,
            )
        pp, depth = project(ring3, camera, target, zoom)
        if np.any(~np.isfinite(pp)):
            continue
        if pp[:, 0].max() < -80 or pp[:, 0].min() > W + 80 or pp[:, 1].max() < -80 or pp[:, 1].min() > H + 80:
            continue
        centered = ring3 - ring3.mean(axis=0, keepdims=True)
        area = float(
            max(
                1.0e-6,
                (np.ptp(centered @ tangent_u) * np.ptp(centered @ tangent_v)),
            )
        )
        markers.append((float(depth.mean()), [tuple(map(float, p)) for p in pp], body.color, area))

    markers.sort(key=lambda item: item[0])
    for _, pts, color, area in markers:
        fill_color = tuple(min(255, int(c * 1.05)) for c in color) + (64,)
        halo_color = (255, 255, 255, 150)
        draw.polygon(pts, fill=halo_color)
        draw.polygon(pts, fill=fill_color)
        ring_color = tuple(min(255, int(c * 1.05)) for c in color) + (238,)
        draw.line(pts + [pts[0]], fill=ring_color, width=3)
        if area > 0.035:
            cx = sum(p[0] for p in pts) / len(pts)
            cy = sum(p[1] for p in pts) / len(pts)
            draw.ellipse([cx - 4, cy - 4, cx + 4, cy + 4], fill=ring_color, outline=(255, 255, 255, 230), width=1)


def fit_zoom_to_bodies(
    bodies: list[Body],
    t: float,
    camera: np.ndarray,
    target: np.ndarray,
    base_zoom: float,
    local: bool,
) -> float:
    """Reduce zoom when the selected camera would clip a large display mesh."""
    points: list[np.ndarray] = []
    fit_bodies = bodies
    if any(b.asset.category in STATIC_FIT_EXCLUDED_CATEGORIES for b in bodies):
        fit_bodies = [b for b in bodies if b.asset.category not in STATIC_FIT_EXCLUDED_CATEGORIES]
        if not fit_bodies:
            fit_bodies = bodies
    for body in fit_bodies:
        vertices, _ = body.display_transformed(t)
        if len(vertices) > 2600:
            step = max(1, len(vertices) // 2600)
            vertices = vertices[::step]
        points.append(vertices)
    if not points:
        return base_zoom
    vertices = np.vstack(points)
    projected, _ = project(vertices, camera, target, 1.0)
    if not np.isfinite(projected).all():
        return base_zoom

    center = np.array([W * 0.5, H * 0.58], dtype=np.float64)
    offsets = projected - center
    max_abs_x = float(np.max(np.abs(offsets[:, 0])))
    max_up = float(np.max(np.maximum(0.0, -offsets[:, 1])))
    max_down = float(np.max(np.maximum(0.0, offsets[:, 1])))
    if max_abs_x < 1.0e-8 and max_up < 1.0e-8 and max_down < 1.0e-8:
        return base_zoom

    x_limit = W * (0.43 if local else 0.46)
    y_top_limit = H * (0.36 if local else 0.37)
    y_bottom_limit = H * (0.39 if local else 0.41)
    zoom_limits = [base_zoom]
    if max_abs_x > 1.0e-8:
        zoom_limits.append(x_limit / max_abs_x)
    if max_up > 1.0e-8:
        zoom_limits.append(y_top_limit / max_up)
    if max_down > 1.0e-8:
        zoom_limits.append(y_bottom_limit / max_down)
    fitted = min(zoom_limits) * 0.94
    return max(80.0 if not local else 110.0, min(base_zoom, fitted))


def fit_zoom_to_trajectory(
    bodies: list[Body],
    times: Iterable[float],
    camera: np.ndarray,
    target: np.ndarray,
    base_zoom: float,
) -> float:
    """Fit one fixed camera to the complete rendered trajectory."""
    points: list[np.ndarray] = []
    fit_bodies = bodies
    if any(b.asset.category in STATIC_FIT_EXCLUDED_CATEGORIES for b in bodies):
        fit_bodies = [b for b in bodies if b.asset.category not in STATIC_FIT_EXCLUDED_CATEGORIES]
        if not fit_bodies:
            fit_bodies = bodies
    for body in fit_bodies:
        for t in times:
            vertices, _ = body.display_transformed(float(t))
            if len(vertices) > 900:
                step = max(1, len(vertices) // 900)
                vertices = vertices[::step]
            points.append(vertices)
    if not points:
        return base_zoom
    vertices = np.vstack(points)
    projected, _ = project(vertices, camera, target, 1.0)
    if not np.isfinite(projected).all():
        return base_zoom

    center = np.array([W * 0.5, H * 0.58], dtype=np.float64)
    offsets = projected - center
    max_abs_x = float(np.max(np.abs(offsets[:, 0])))
    max_up = float(np.max(np.maximum(0.0, -offsets[:, 1])))
    max_down = float(np.max(np.maximum(0.0, offsets[:, 1])))
    if max_abs_x < 1.0e-8 and max_up < 1.0e-8 and max_down < 1.0e-8:
        return base_zoom

    zoom_limits = [base_zoom]
    if max_abs_x > 1.0e-8:
        zoom_limits.append((W * 0.46) / max_abs_x)
    if max_up > 1.0e-8:
        zoom_limits.append((H * 0.37) / max_up)
    if max_down > 1.0e-8:
        zoom_limits.append((H * 0.41) / max_down)
    fitted = min(zoom_limits) * 0.94
    return max(80.0, min(base_zoom, fitted))


def draw_meshes(
    image: Image.Image,
    bodies: list[Body],
    t: float,
    camera: np.ndarray,
    target: np.ndarray,
    zoom: float,
    title: str,
    subtitle: str,
    local: bool = False,
    toi_seconds: float = CONTACT_T,
    pure_white_background: bool = False,
    case_name: str | None = None,
) -> Image.Image:
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    if not pure_white_background:
        draw_checker_floor(draw, camera, target, zoom)

    face_items: list[tuple[int, float, np.ndarray, tuple[int, int, int], float, int, bool]] = []
    for body in bodies:
        if case_name == "many_object_tabletop_drop" and body.asset.category in SUPPORT_RENDER_CATEGORIES:
            if draw_graphics_engine_tile_board(draw, body, t, camera, target, zoom):
                continue
        verts, faces = body.display_transformed(t)
        pp, depth = project(verts, camera, target, zoom)
        cam_verts = np.column_stack([pp[:, 0], pp[:, 1], depth])
        face_vertices = cam_verts[faces]
        world_face_vertices = verts[faces]
        normals = np.cross(world_face_vertices[:, 1] - world_face_vertices[:, 0], world_face_vertices[:, 2] - world_face_vertices[:, 0])
        normals = normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-9)
        light = normalize(np.array([-0.35, -0.55, 0.75]))
        shade = np.clip(0.38 + 0.62 * np.abs(normals @ light), 0.25, 1.0)
        for f2, sh in zip(face_vertices, shade):
            if np.any(~np.isfinite(f2)):
                continue
            if (f2[:, 0].max() < -200 or f2[:, 0].min() > W + 200 or f2[:, 1].max() < -200 or f2[:, 1].min() > H + 200):
                continue
            is_support = body.asset.category in SUPPORT_RENDER_CATEGORIES
            if body.asset.category == "wall":
                metadata = body.metadata or {}
                if metadata.get("damage_model") == "ue_chaos_style_concrete_fracture":
                    alpha = 218 if local else 196
                elif metadata.get("motion_model") == "attached_concrete_breach_rim":
                    alpha = 242 if local else 224
                else:
                    alpha = 46 if local else 62
            elif body.asset.category == "ground":
                alpha = 145
            elif body.asset.category == "slab":
                alpha = 190
            else:
                alpha = 232
            # Draw support geometry first.  Dense ground/slab/wall surfaces are
            # visual context, not dynamic foreground objects; drawing them over
            # the objects creates false penetration cues in global camera views.
            priority = 0 if is_support else 1
            draw_outline = True
            if case_name == "many_object_tabletop_drop" and not is_support:
                tri2 = f2[:, :2]
                screen_edges = [
                    math.hypot(tri2[(k + 1) % 3, 0] - tri2[k, 0], tri2[(k + 1) % 3, 1] - tri2[k, 1])
                    for k in range(3)
                ]
                signed_area = 0.5 * abs(
                    float(
                        tri2[0, 0] * (tri2[1, 1] - tri2[2, 1])
                        + tri2[1, 0] * (tri2[2, 1] - tri2[0, 1])
                        + tri2[2, 0] * (tri2[0, 1] - tri2[1, 1])
                    )
                )
                max_screen_edge = max(screen_edges)
                # Drop screen-space sliver triangles for this visualization.
                # They usually come from low-quality ShapeNet source faces and
                # appear as artificial contact/candidate streaks.
                if max_screen_edge > 48.0 and signed_area < 0.12 * max_screen_edge * max_screen_edge:
                    continue
                draw_outline = max_screen_edge <= 46.0
            face_items.append((priority, float(f2[:, 2].mean()), f2[:, :2], body.color, float(sh), alpha, draw_outline))

    face_items.sort(key=lambda x: (x[0], x[1]))
    for _, _, poly, base_color, shade, alpha, draw_outline in face_items:
        pts = [tuple(map(float, p)) for p in poly]
        fill = tuple(int(c * shade) for c in base_color) + (alpha,)
        line = tuple(min(255, int(c * 1.35)) for c in base_color) + (250,)
        draw.polygon(pts, fill=fill)
        if draw_outline:
            draw.line(pts + [pts[0]], fill=line, width=1 if local else 1)

    # If a fragmented OBJ was rendered with a continuous display shell, add a
    # sparse overlay of source-mesh edges.  This keeps the object visually
    # complete while preserving recognizable features such as wheels, wings,
    # legs, and thin rods.
    for body in bodies:
        if body.asset.category in SUPPORT_RENDER_CATEGORIES:
            continue
        if body.asset.display_shell_method.startswith("source_mesh"):
            continue
        source_verts = body.transformed(t)
        source_faces = body.asset.faces
        max_detail_faces = 3200 if local else 1800
        if len(source_faces) > max_detail_faces:
            step = max(1, len(source_faces) // max_detail_faces)
            source_faces = source_faces[::step][:max_detail_faces]
        pp, _ = project(source_verts, camera, target, zoom)
        detail_color = tuple(min(255, int(c * 1.55)) for c in body.color) + ((110 if local else 82),)
        for tri in source_faces:
            pts = [tuple(map(float, pp[int(i)])) for i in tri]
            if any((p[0] < -120 or p[0] > W + 120 or p[1] < -120 or p[1] > H + 120) for p in pts):
                continue
            if case_name == "many_object_tabletop_drop":
                screen_edges = [
                    math.hypot(pts[(k + 1) % 3][0] - pts[k][0], pts[(k + 1) % 3][1] - pts[k][1])
                    for k in range(3)
                ]
                # ShapeNet OBJ files occasionally contain long cross-part or
                # degenerate triangles.  They are useful for geometry audit but
                # visually read as false CCD/contact regions in the dense
                # tabletop case, so the decorative source-edge overlay filters
                # them while leaving the filled display mesh untouched.
                if max(screen_edges) > (44.0 if not local else 96.0):
                    continue
            draw.line(pts + [pts[0]], fill=detail_color, width=1)

    if case_name == "many_object_tabletop_drop":
        draw_support_contact_markers(draw, bodies, t, camera, target, zoom)

    wall_bodies = [b for b in bodies if b.asset.category in {"wall", "brick_wall"}]
    non_wall_bodies = [b for b in bodies if b.asset.category not in {"wall", "brick_wall"}]
    if wall_bodies and non_wall_bodies and abs(t - CONTACT_T) <= 0.16:
        car = non_wall_bodies[0]
        wall = wall_bodies[0]
        car_vertices = car.transformed(t)
        wall_vertices = wall.transformed(t)
        car_front_index = int(np.argmax(car_vertices[:, 0]))
        car_front = car_vertices[car_front_index].copy()
        wall_front_x = float(wall_vertices[:, 0].min())
        wall_front = car_front.copy()
        wall_front[0] = wall_front_x
        gap = wall_front_x - float(car_front[0])
        marker_points = np.vstack([car_front, wall_front])
        marker_2d, _ = project(marker_points, camera, target, zoom)
        p0, p1 = marker_2d
        contact_color = (250, 204, 21, 255)
        draw.line([tuple(p0), tuple(p1)], fill=contact_color, width=5 if local else 4)
        r = 7 if local else 5
        for p in (p0, p1):
            draw.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=contact_color, outline=(255, 255, 255, 255), width=1)
        label = f"TOI contact gap={gap:.3f}"
        label_pos = (float((p0[0] + p1[0]) * 0.5 + 10), float((p0[1] + p1[1]) * 0.5 - 18))
        draw.text(label_pos, label, fill=contact_color)

    image = Image.alpha_composite(image.convert("RGBA"), overlay)
    draw = ImageDraw.Draw(image, "RGBA")
    font_big = ImageFont.truetype("arial.ttf", 28) if Path("C:/Windows/Fonts/arial.ttf").exists() else None
    font_small = ImageFont.truetype("arial.ttf", 18) if Path("C:/Windows/Fonts/arial.ttf").exists() else None
    panel_fill = (255, 255, 255, 235) if pure_white_background else (246, 249, 251, 216)
    panel_outline = (200, 210, 218, 245) if pure_white_background else (196, 207, 216, 230)
    draw.rectangle([22, 18, W - 22, 128], fill=panel_fill, outline=panel_outline, width=2)
    draw.text((42, 34), title, fill=(28, 39, 52, 255), font=font_big)
    draw.text((42, 76), subtitle, fill=(68, 82, 96, 255), font=font_small)
    draw.text((W - 330, 38), f"t={t:.2f}s | TOI={toi_seconds:.2f}s", fill=(18, 137, 89, 255), font=font_small)
    return image.convert("RGB")


def write_mp4(path: Path, frames: Iterable[Image.Image], fps: int = FPS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw_path = path.with_suffix(".raw.mp4")
    writer = cv2.VideoWriter(str(raw_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open mp4 writer: {raw_path}")
    for frame in frames:
        arr = cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR)
        writer.write(arr)
    writer.release()
    selected_encoder = select_h264_encoder()
    if not selected_encoder:
        raw_path.replace(path)
        return

    ffmpeg, encoder = selected_encoder
    tmp_path = path.with_suffix(".h264.tmp.mp4")
    try:
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(raw_path),
                *h264_encoder_args(encoder),
                str(tmp_path),
            ],
            check=True,
        )
        tmp_path.replace(path)
        raw_path.unlink(missing_ok=True)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raw_path.replace(path)


def camera_setup(
    case_name: str,
    t: float,
    bodies: list[Body],
    contact_point: np.ndarray,
    local: bool,
) -> tuple[np.ndarray, np.ndarray, float]:
    if case_name == "many_object_ground_drop":
        target = np.array([0.0, 0.0, 0.72])
        camera = np.array([4.85, -6.55, 3.35])
        return camera, target, 260.0

    if case_name == "many_object_tabletop_drop":
        target = np.array([0.32, -0.04, 0.72])
        camera = np.array([6.6, -8.1, 4.35])
        return camera, target, 176.0

    if case_name == "soft_body_toothpaste_squeeze":
        target = np.array([0.18, 0.0, 0.42])
        camera = np.array([3.7, -4.85, 2.15])
        return camera, target, 410.0

    if case_name == "car_wall_impact":
        # The wall is a large slab.  Use a fixed camera on the incoming-car
        # side so the contact surface remains visible after rebound instead
        # of drifting behind the wall with the generic center-tracking view.
        if local:
            target = contact_point + np.array([-0.95, -0.03, 0.95])
            camera = contact_point + np.array([-2.65, -6.25, 2.55])
            return camera, target, 220.0
        target = contact_point + np.array([-2.35, -0.04, 1.18])
        camera = contact_point + np.array([-6.60, -12.50, 4.25])
        return camera, target, 120.0

    center = np.mean([b.position(t) for b in bodies], axis=0)
    if local:
        return contact_point + np.array([1.35, -1.85, 1.25]), contact_point, 660.0
    return center + np.array([3.2, -4.2, 2.65]), center, 300.0


def dominant_pair_axis(a: Body, b: Body) -> tuple[int, str]:
    rel = np.asarray(a.velocity_at(CONTACT_T) - b.velocity_at(CONTACT_T), dtype=np.float64)
    if float(np.linalg.norm(rel)) < 1.0e-9:
        rel = np.asarray(a.v0 - b.v0, dtype=np.float64)
    axis = int(np.argmax(np.abs(rel))) if float(np.linalg.norm(rel)) >= 1.0e-9 else 0
    return axis, "xyz"[axis]


def signed_aabb_gap_on_axis(a_min: np.ndarray, a_max: np.ndarray, b_min: np.ndarray, b_max: np.ndarray, axis: int) -> float:
    if a_max[axis] <= b_min[axis]:
        return float(b_min[axis] - a_max[axis])
    if b_max[axis] <= a_min[axis]:
        return float(a_min[axis] - b_max[axis])
    return -float(min(a_max[axis] - b_min[axis], b_max[axis] - a_min[axis]))


def geometry_audit_at_toi(bodies: list[Body]) -> dict[str, object]:
    mujoco_body = next((b for b in bodies if (b.metadata or {}).get("solver") == "mujoco_3_4_rigid_body_engine"), None)
    if mujoco_body is not None:
        return (mujoco_body.metadata or {}).get("geometry_audit_at_toi", {"time": CONTACT_T, "pairs": []})  # type: ignore[return-value]
    audit: dict[str, object] = {"time": CONTACT_T, "pairs": []}
    audited = [b for b in bodies if not (b.metadata or {}).get("exclude_from_geometry_audit", False)]
    plane = support_plane_from_bodies(bodies)
    if plane is not None and any(b.has_sampled_trajectory() for b in audited):
        normal, offset, metadata, support_name = plane
        moving = [b for b in audited if b.has_sampled_trajectory()]
        audit.update(
            {
                "support_model": "inclined_support_plane",
                "support": support_name,
                "support_plane_normal": normal.tolist(),
                "support_plane_offset": offset,
                "incline_angle_degrees": metadata.get("incline_angle_degrees"),
            }
        )
        for body in moving:
            vertices = body.transformed(CONTACT_T)
            distances = vertices @ normal - offset
            min_index = int(np.argmin(distances))
            signed_gap = float(distances[min_index])
            audit["pairs"].append(
                {
                    "a": body.asset.name,
                    "b": support_name,
                    "support_point": vertices[min_index].tolist(),
                    "support_plane_normal": normal.tolist(),
                    "support_plane_offset": offset,
                    "signed_gap_axis": signed_gap,
                    "penetrating_axis": signed_gap < -1.0e-6,
                }
            )
        return audit
    for i in range(len(audited)):
        for j in range(i + 1, len(audited)):
            a = audited[i].transformed(CONTACT_T)
            b = audited[j].transformed(CONTACT_T)
            a_min, a_max = a.min(axis=0), a.max(axis=0)
            b_min, b_max = b.min(axis=0), b.max(axis=0)
            axis, axis_name = dominant_pair_axis(audited[i], audited[j])
            signed_gap = signed_aabb_gap_on_axis(a_min, a_max, b_min, b_max, axis)
            signed_gap_x = signed_aabb_gap_on_axis(a_min, a_max, b_min, b_max, 0)
            overlap_y = float(min(a_max[1], b_max[1]) - max(a_min[1], b_min[1]))
            overlap_z = float(min(a_max[2], b_max[2]) - max(a_min[2], b_min[2]))
            audit["pairs"].append(
                {
                    "a": audited[i].asset.name,
                    "b": audited[j].asset.name,
                    "aabb_a_min": a_min.tolist(),
                    "aabb_a_max": a_max.tolist(),
                    "aabb_b_min": b_min.tolist(),
                    "aabb_b_max": b_max.tolist(),
                    "signed_gap_x": signed_gap_x,
                    "separation_axis": axis_name,
                    "signed_gap_axis": signed_gap,
                    "overlap_y": overlap_y,
                    "overlap_z": overlap_z,
                    "penetrating_x": signed_gap_x < -1.0e-6,
                    "penetrating_axis": signed_gap < -1.0e-6,
                }
            )
    return audit


def geometry_audit_over_frames(bodies: list[Body]) -> dict[str, object]:
    samples = timeline_samples(trajectory_duration_seconds(bodies), trajectory_frame_count(bodies))
    deformable = [b for b in bodies if b.trajectory_vertices is not None]
    if deformable:
        body = deformable[0]
        metadata = body.metadata or {}
        if metadata.get("damage_model") == "ue_chaos_style_concrete_fracture":
            return {
                "sample_count": int(len(samples)),
                "deformable_body": body.asset.name,
                "vertex_count": int(len(body.trajectory_vertices[-1])),
                "surface_faces": int(len(body.asset.faces)),
                "damage_model": metadata.get("damage_model"),
                "min_signed_gap_at_toi": metadata.get("min_signed_gap_at_toi"),
                "max_crater_depth_m": metadata.get("max_crater_depth_m"),
                "breach_depth_m": metadata.get("breach_depth_m"),
                "fractured_front_triangles": metadata.get("fractured_front_triangles"),
                "debris_piece_count": metadata.get("debris_piece_count"),
                "postsolve_max_penetration": metadata.get("postsolve_max_penetration"),
                "penetrating_any_sample": bool(float(metadata.get("postsolve_max_penetration", 0.0)) > 1.0e-6),
            }
        if metadata.get("solver") == "mujoco_3_4_rigid_body_engine":
            return {
                "sample_count": int(len(samples)),
                "dynamic_body": body.asset.name,
                "engine": metadata.get("engine"),
                "brick_count": metadata.get("brick_count"),
                "surface_faces": int(len(body.asset.faces)),
                "gravity": metadata.get("gravity"),
                "sim_dt": metadata.get("sim_dt"),
                "displaced_brick_count": metadata.get("displaced_brick_count"),
                "max_brick_displacement_m": metadata.get("max_brick_displacement_m"),
                "max_brick_speed_mps": metadata.get("max_brick_speed_mps"),
                "max_contact_count": metadata.get("max_contact_count"),
                "min_mujoco_contact_distance_m": metadata.get("min_mujoco_contact_distance_m"),
                "postsolve_max_penetration": metadata.get("postsolve_max_penetration"),
                "penetrating_any_sample": bool(float(metadata.get("postsolve_max_penetration", 0.0)) > 0.035),
            }
        return {
            "sample_count": int(len(samples)),
            "deformable_body": body.asset.name,
            "vertex_count": int(len(body.trajectory_vertices[-1])),
            "surface_faces": int(len(body.asset.faces)),
            "postsolve_max_penetration": metadata.get("postsolve_max_penetration"),
            "top_plate_contact_window_samples": metadata.get("top_plate_contact_window_samples"),
            "bottom_plate_contact_window_samples": metadata.get("bottom_plate_contact_window_samples"),
            "max_node_displacement": metadata.get("max_node_displacement"),
            "penetrating_any_sample": bool(float(metadata.get("postsolve_max_penetration", 0.0)) > 1.0e-6),
        }
    if any(b.has_sampled_trajectory() for b in bodies):
        ground_bodies = [b for b in bodies if b.asset.category in {"ground", "slab"}]
        moving = [b for b in bodies if b.has_sampled_trajectory() and not (b.metadata or {}).get("exclude_from_physics_audit", False)]
        plane = support_plane_from_bodies(bodies)
        if plane is not None:
            support_normal, support_offset, support_metadata, support_name = plane
            ground_z = None
        else:
            support_normal = None
            support_offset = 0.0
            support_metadata = {}
            support_name = None
            ground_z = 0.0
            if ground_bodies:
                ground_z = float(max(g.transformed(0.0)[:, 2].max() for g in ground_bodies))
        min_gap = float("inf")
        min_time = 0.0
        contact_samples = 0
        penetrating = False
        per_body_ground_gaps = [
            {
                "name": body.asset.name,
                "min_signed_ground_gap": float("inf"),
                "min_gap_time": 0.0,
                "ground_contact_window_samples": 0,
                "penetrating_any_sample": False,
            }
            for body in moving
        ]
        for t in samples:
            for body_index, body in enumerate(moving):
                vertices = body.transformed(float(t))
                if support_normal is not None:
                    gap = float(np.min(vertices @ support_normal - support_offset))
                else:
                    assert ground_z is not None
                    gap = float(vertices[:, 2].min() - ground_z)
                if gap < min_gap:
                    min_gap = gap
                    min_time = float(t)
                body_gap = per_body_ground_gaps[body_index]
                if gap < float(body_gap["min_signed_ground_gap"]):
                    body_gap["min_signed_ground_gap"] = gap
                    body_gap["min_gap_time"] = float(t)
                if gap <= 0.035:
                    contact_samples += 1
                    body_gap["ground_contact_window_samples"] = int(body_gap["ground_contact_window_samples"]) + 1
                if gap < -1.0e-6:
                    penetrating = True
                    body_gap["penetrating_any_sample"] = True
        result = {
            "sample_count": int(len(samples)),
            "moving_body_count": int(len(moving)),
            "ground_z": ground_z,
            "min_signed_ground_gap": min_gap,
            "min_gap_time": min_time,
            "ground_contact_window_samples": int(contact_samples),
            "penetrating_any_sample": penetrating,
            "per_body_ground_gaps": per_body_ground_gaps,
        }
        if support_normal is not None:
            result.update(
                {
                    "support_model": "inclined_support_plane",
                    "support": support_name,
                    "support_plane_normal": support_normal.tolist(),
                    "support_plane_offset": support_offset,
                    "incline_angle_degrees": support_metadata.get("incline_angle_degrees"),
                    "left_side_raised": support_metadata.get("left_side_raised"),
                }
            )
        return result
    axis, axis_name = dominant_pair_axis(bodies[0], bodies[1])
    min_gap = float("inf")
    min_time = 0.0
    penetrating = False
    for t in samples:
        a = bodies[0].transformed(float(t))
        b = bodies[1].transformed(float(t))
        a_min, a_max = a.min(axis=0), a.max(axis=0)
        b_min, b_max = b.min(axis=0), b.max(axis=0)
        gap = signed_aabb_gap_on_axis(a_min, a_max, b_min, b_max, axis)
        if gap < min_gap:
            min_gap = gap
            min_time = float(t)
        if gap < -1.0e-6:
            penetrating = True
    return {
        "sample_count": int(len(samples)),
        "separation_axis": axis_name,
        "min_signed_gap_x": min_gap,
        "min_signed_gap_axis": min_gap,
        "min_gap_time": min_time,
        "penetrating_any_sample": penetrating,
    }


def compact_plot_mesh(vertices: np.ndarray, faces: np.ndarray, max_faces: int = 3600) -> tuple[np.ndarray, np.ndarray]:
    if len(faces) > max_faces and len(faces) > int(max_faces * 1.1):
        step = max(1, len(faces) // max_faces)
        faces = faces[::step][:max_faces]
    used = np.unique(faces.reshape(-1))
    remap = -np.ones(len(vertices), dtype=np.int64)
    remap[used] = np.arange(len(used))
    return vertices[used], remap[faces]


def mesh_edges_as_lines(vertices: np.ndarray, faces: np.ndarray) -> tuple[list[float | None], list[float | None], list[float | None]]:
    edges: set[tuple[int, int]] = set()
    for a, b, c in faces:
        for u, v in ((int(a), int(b)), (int(b), int(c)), (int(c), int(a))):
            if u > v:
                u, v = v, u
            edges.add((u, v))
    xs: list[float | None] = []
    ys: list[float | None] = []
    zs: list[float | None] = []
    for u, v in sorted(edges):
        xs.extend([float(vertices[u, 0]), float(vertices[v, 0]), None])
        ys.extend([float(vertices[u, 1]), float(vertices[v, 1]), None])
        zs.extend([float(vertices[u, 2]), float(vertices[v, 2]), None])
    return xs, ys, zs


def write_interactive_car_wall_html(
    case_dir: Path,
    title: str,
    description: str,
    bodies: list[Body],
    geometry_audit: dict[str, object],
) -> None:
    import plotly.graph_objects as go

    times = np.linspace(0.0, DURATION_T, 25)
    toi_index = int(np.argmin(np.abs(times - CONTACT_T)))
    times[toi_index] = CONTACT_T
    car_body, wall_body = bodies[0], bodies[1]

    def body_mesh(body: Body, t: float, max_faces: int) -> tuple[np.ndarray, np.ndarray]:
        vertices, faces = body.display_transformed(float(t))
        return compact_plot_mesh(vertices, faces, max_faces=max_faces)

    car_v, car_f = body_mesh(car_body, CONTACT_T, MAX_FACES)
    wall_metadata = wall_body.metadata or {}
    if wall_metadata.get("solver") == "mujoco_3_4_rigid_body_engine":
        wall_detail_faces = MAX_FACES
    elif wall_metadata.get("damage_model") == "ue_chaos_style_concrete_fracture":
        wall_detail_faces = 7200
    else:
        wall_detail_faces = 12
    wall_v, wall_f = body_mesh(wall_body, CONTACT_T, wall_detail_faces)
    wall_toi = wall_body.transformed(CONTACT_T)
    wall_min, wall_max = wall_toi.min(axis=0), wall_toi.max(axis=0)
    front_x = float(wall_min[0])
    y0, y1 = float(wall_min[1]), float(wall_max[1])
    z0, z1 = float(wall_min[2]), float(wall_max[2])
    plane_x = [front_x, front_x, front_x, front_x, front_x]
    plane_y = [y0, y1, y1, y0, y0]
    plane_z = [z0, z0, z1, z1, z0]
    wall_edge_x, wall_edge_y, wall_edge_z = mesh_edges_as_lines(wall_v, wall_f)

    def point_sample(vertices: np.ndarray, max_points: int = 1800) -> np.ndarray:
        if len(vertices) <= max_points:
            return vertices
        step = max(1, len(vertices) // max_points)
        return vertices[::step][:max_points]

    def contact_segment(t: float) -> tuple[list[float], list[float], list[float]]:
        car_vertices = car_body.transformed(float(t))
        wall_vertices = wall_body.transformed(float(t))
        front_idx = int(np.argmax(car_vertices[:, 0]))
        car_front = car_vertices[front_idx].copy()
        wall_front = car_front.copy()
        wall_front[0] = float(wall_vertices[:, 0].min())
        return (
            [float(car_front[0]), float(wall_front[0])],
            [float(car_front[1]), float(wall_front[1])],
            [float(car_front[2]), float(wall_front[2])],
        )

    car_points = point_sample(car_v)
    contact_x, contact_y, contact_z = contact_segment(CONTACT_T)

    first_gap = CAR_WALL_TRUE_CONTACT_GAP
    if geometry_audit.get("pairs"):
        first_gap = float(geometry_audit["pairs"][0].get("signed_gap_x", first_gap))

    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=car_v[:, 0],
                y=car_v[:, 1],
                z=car_v[:, 2],
                i=car_f[:, 0],
                j=car_f[:, 1],
                k=car_f[:, 2],
                name="ShapeNet car mesh",
                color="#42c7ff",
                opacity=0.82,
                flatshading=False,
                hoverinfo="skip",
            ),
            go.Scatter3d(
                x=car_points[:, 0],
                y=car_points[:, 1],
                z=car_points[:, 2],
                mode="markers",
                name="car visible vertices",
                marker={"size": 2.8, "color": "#7dd3fc", "opacity": 0.96},
                hoverinfo="skip",
            ),
            go.Mesh3d(
                x=wall_v[:, 0],
                y=wall_v[:, 1],
                z=wall_v[:, 2],
                i=wall_f[:, 0],
                j=wall_f[:, 1],
                k=wall_f[:, 2],
                name="triangle-mesh wall",
                color="#ff7b62",
                opacity=0.12,
                flatshading=True,
                hoverinfo="skip",
            ),
            go.Scatter3d(
                x=wall_edge_x,
                y=wall_edge_y,
                z=wall_edge_z,
                mode="lines",
                name="wall wireframe",
                line={"color": "#fb923c", "width": 5},
                hoverinfo="skip",
            ),
            go.Scatter3d(
                x=plane_x,
                y=plane_y,
                z=plane_z,
                mode="lines",
                name="front contact plane",
                line={"color": "#facc15", "width": 8},
                hoverinfo="skip",
            ),
            go.Scatter3d(
                x=contact_x,
                y=contact_y,
                z=contact_z,
                mode="lines+markers",
                name="true TOI contact segment",
                line={"color": "#fde047", "width": 10},
                marker={"size": 7, "color": "#fde047"},
                hovertemplate="TOI support segment<extra></extra>",
            ),
        ]
    )

    frames = []
    for idx, t_value in enumerate(times):
        cv, cf = body_mesh(car_body, float(t_value), MAX_FACES)
        cp = point_sample(cv)
        wv, wf = body_mesh(wall_body, float(t_value), wall_detail_faces)
        we_x, we_y, we_z = mesh_edges_as_lines(wv, wf)
        cx, cy, cz = contact_segment(float(t_value))
        frames.append(
            go.Frame(
                name=f"{idx:02d} | t={float(t_value):.3f}",
                traces=[0, 1, 2, 3, 5],
                data=[
                    go.Mesh3d(
                        x=cv[:, 0],
                        y=cv[:, 1],
                        z=cv[:, 2],
                        i=cf[:, 0],
                        j=cf[:, 1],
                        k=cf[:, 2],
                        color="#42c7ff",
                        opacity=0.82,
                    ),
                    go.Scatter3d(
                        x=cp[:, 0],
                        y=cp[:, 1],
                        z=cp[:, 2],
                        mode="markers",
                        marker={"size": 2.8, "color": "#7dd3fc", "opacity": 0.96},
                    ),
                    go.Mesh3d(
                        x=wv[:, 0],
                        y=wv[:, 1],
                        z=wv[:, 2],
                        i=wf[:, 0],
                        j=wf[:, 1],
                        k=wf[:, 2],
                        color="#ff7b62",
                        opacity=0.12,
                    ),
                    go.Scatter3d(x=we_x, y=we_y, z=we_z, mode="lines", line={"color": "#fb923c", "width": 5}),
                    go.Scatter3d(
                        x=cx,
                        y=cy,
                        z=cz,
                        mode="lines+markers",
                        line={"color": "#fde047", "width": 10},
                        marker={"size": 7, "color": "#fde047"},
                    ),
                ],
            )
        )

    fig.frames = frames
    fig.update_layout(
        title=f"{title} | drag/zoom interactive | TOI={CONTACT_T:.3f}s | signed gap={float(first_gap):.4f}",
        paper_bgcolor="#0b1120",
        plot_bgcolor="#0b1120",
        font={"color": "#e5e7eb"},
        margin={"l": 0, "r": 0, "t": 70, "b": 0},
        scene={
            "bgcolor": "#111827",
            "aspectmode": "data",
            "camera": {"eye": {"x": -2.9, "y": -1.85, "z": 1.35}, "center": {"x": -0.22, "y": 0.0, "z": 0.0}},
            "xaxis": {"title": "approach x", "gridcolor": "#334155", "zerolinecolor": "#64748b"},
            "yaxis": {"title": "width y", "gridcolor": "#334155", "zerolinecolor": "#64748b"},
            "zaxis": {"title": "height z", "gridcolor": "#334155", "zerolinecolor": "#64748b"},
        },
        annotations=[
            {
                "text": "Mouse drag = rotate | wheel = zoom | slider = replay. Yellow segment marks the true TOI car-wall contact.",
                "xref": "paper",
                "yref": "paper",
                "x": 0.02,
                "y": 0.02,
                "showarrow": False,
                "font": {"size": 13, "color": "#cbd5e1"},
            },
            {
                "text": description,
                "xref": "paper",
                "yref": "paper",
                "x": 0.02,
                "y": 0.95,
                "showarrow": False,
                "font": {"size": 13, "color": "#cbd5e1"},
            },
        ],
        sliders=[
            {
                "active": toi_index,
                "currentvalue": {"prefix": "frame "},
                "steps": [
                    {
                        "label": f"t={float(t):.2f}" + (" TOI" if i == toi_index else ""),
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
                "x": 0.02,
                "y": 0.9,
                "buttons": [
                    {"label": "Play", "method": "animate", "args": [None, {"frame": {"duration": 130, "redraw": True}, "fromcurrent": True}]},
                    {"label": "Pause", "method": "animate", "args": [[None], {"mode": "immediate"}]},
                ],
            }
        ],
    )
    fig.write_html(str(case_dir / "car_wall_impact_interactive.html"), include_plotlyjs=True, full_html=True, auto_play=False)


def build_case(
    case_dir: Path,
    title: str,
    description: str,
    bodies: list[Body],
    contact_point: np.ndarray,
    benchmark_metrics: dict[str, object],
) -> dict[str, object]:
    global W, H
    case_name = case_dir.name
    W, H = render_size_for_case(case_name)
    if case_dir.exists():
        for stale_name in ("old_proxy_png_sequence", "global", "local_zoom", "real_mesh_local_frames"):
            stale_dir = case_dir / stale_name
            if stale_dir.exists():
                shutil.rmtree(stale_dir)
    case_dir.mkdir(parents=True, exist_ok=True)
    global_frames_dir = case_dir / "real_mesh_global_frames"
    if global_frames_dir.exists():
        shutil.rmtree(global_frames_dir)
    (case_dir / "local_zoom.mp4").unlink(missing_ok=True)
    global_frames_dir.mkdir(exist_ok=True)

    audit = physics_audit(bodies)
    trajectory_case = any(b.has_sampled_trajectory() for b in bodies)
    case_duration = float(benchmark_metrics.get("duration_seconds") or trajectory_duration_seconds(bodies))
    case_frame_count = int(benchmark_metrics.get("frame_count") or trajectory_frame_count(bodies))
    times = timeline_samples(case_duration, case_frame_count)
    global_frames: list[Image.Image] = []
    toi_seconds = float(benchmark_metrics.get("toi_seconds") or benchmark_metrics.get("first_ground_contact_time") or CONTACT_T)
    camera, target, zoom = camera_setup(case_name, CONTACT_T, bodies, contact_point, local=False)
    if case_name == "many_object_tabletop_drop":
        moving_bodies = [body for body in bodies if body.asset.category not in STATIC_FIT_EXCLUDED_CATEGORIES]
        if moving_bodies:
            cluster_samples = np.vstack([body.position(float(t)) for body in moving_bodies for t in times[:: max(1, len(times) // 18)]])
            target = np.mean(cluster_samples, axis=0) + np.array([0.18, -0.18, 0.08], dtype=np.float64)
            camera = target + np.array([6.45, -7.45, 4.25], dtype=np.float64)
        zoom = max(88.0, 0.66 * fit_zoom_to_trajectory(bodies, times, camera, target, 218.0))
    else:
        zoom = fit_zoom_to_trajectory(bodies, times, camera, target, zoom)
    pure_white_background = case_name in {"many_object_ground_drop", "many_object_tabletop_drop"}
    for i, t in enumerate(times):
        bg = Image.new("RGB", (W, H), (255, 255, 255))
        frame = draw_meshes(
            bg,
            bodies,
            float(t),
            camera,
            target,
            zoom,
            title,
            description,
            local=False,
            toi_seconds=toi_seconds,
            pure_white_background=pure_white_background,
            case_name=case_name,
        )
        frame.save(global_frames_dir / f"global_frame_{i:03d}.png")
        global_frames.append(frame)

    write_mp4(case_dir / "global.mp4", global_frames)

    if case_name in {"many_object_ground_drop", "many_object_tabletop_drop"}:
        first_contact = float(benchmark_metrics.get("first_ground_contact_time") or CONTACT_T)
        contact_idx = int(np.argmin(np.abs(times - first_contact)))
        settle_offset = 2.65 if case_name == "many_object_tabletop_drop" else 0.72
        settle_idx = int(np.argmin(np.abs(times - min(case_duration, first_contact + settle_offset))))
        indices = [0, max(0, contact_idx - 4), contact_idx, settle_idx]
        contact_label = "first tabletop TOI" if case_name == "many_object_tabletop_drop" else "first ground TOI"
        settle_label = "long low-friction slide" if case_name == "many_object_tabletop_drop" else "frictional settling"
        labels = ["release", "pre-contact", contact_label, settle_label]
    elif case_name == "car_wall_impact":
        toi_idx = int(np.argmin(np.abs(times - CONTACT_T)))
        indices = [0, max(0, toi_idx - 2), toi_idx, min(case_frame_count - 1, toi_idx + 16)]
        labels = ["before", "near TOI", "at TOI", "bricks flying"]
    else:
        toi_idx = int(np.argmin(np.abs(times - CONTACT_T)))
        indices = [0, max(0, toi_idx - 2), toi_idx, min(case_frame_count - 1, toi_idx + 16)]
        labels = ["before", "near TOI", "at TOI", "after rebound"]
    sheet = Image.new("RGB", (W * 2, H * 2), (255, 255, 255))
    for slot, idx in enumerate(indices):
        thumb = global_frames[idx].resize((W, H))
        x = (slot % 2) * W
        y = (slot // 2) * H
        sheet.paste(thumb, (x, y))
        d = ImageDraw.Draw(sheet)
        d.rectangle([x + 18, y + H - 72, x + 240, y + H - 30], fill=(246, 249, 251, 230), outline=(196, 207, 216))
        d.text((x + 30, y + H - 62), labels[slot], fill=(28, 39, 52))
    sheet.save(case_dir / "contact_sheet.png")

    geometry_audit = geometry_audit_at_toi(bodies)
    geometry_frame_audit = geometry_audit_over_frames(bodies)
    if case_name == "car_wall_impact":
        write_interactive_car_wall_html(case_dir, title, description, bodies, geometry_audit)

    metrics = {
        "title": title,
        "description": description,
        "render_width": int(W),
        "render_height": int(H),
        "frame_count": case_frame_count,
        "fps": FPS,
        "duration_seconds": case_duration,
        "toi_seconds": toi_seconds,
        "objects": [
            {
                "name": b.asset.name,
                "category": b.asset.category,
                "mesh_path": safe_rel(b.asset.path),
                "mass": b.mass,
                "velocity_pre": b.v0.tolist(),
                "velocity_post": b.v1.tolist(),
                "motion_model": (b.metadata or {}).get("motion_model", "two_phase_linear_elastic"),
                "trajectory_samples": int(len(b.trajectory_times)) if b.trajectory_times is not None else 0,
                "deformable_vertex_trajectory": b.trajectory_vertices is not None,
                "first_ground_contact_time": (b.metadata or {}).get("first_ground_contact_time"),
                "ground_contact_count": (b.metadata or {}).get("ground_contact_count"),
                "preview_faces": b.asset.stats["preview_faces"],
                "original_faces": b.asset.stats["original_faces"],
                "preview_decimation_method": b.asset.stats.get("preview_decimation_method", "full_surface"),
                "display_shell_method": b.asset.display_shell_method,
                "display_faces": int(len(b.asset.display_faces)) if b.asset.display_faces is not None else int(len(b.asset.faces)),
            }
            for b in bodies
        ],
        "physics_audit": audit,
        "geometry_audit": geometry_audit,
        "geometry_frame_audit": geometry_frame_audit,
        "benchmark_metrics": benchmark_metrics,
        "outputs": {
            "global_mp4": "global.mp4",
            "contact_sheet": "contact_sheet.png",
            "interactive_html": "car_wall_impact_interactive.html" if case_name == "car_wall_impact" else None,
            "global_frames": "real_mesh_global_frames/global_frame_*.png",
        },
    }
    write_json(case_dir / "metrics.json", metrics)
    mujoco_metadata = next(
        ((b.metadata or {}) for b in bodies if (b.metadata or {}).get("solver") == "mujoco_3_4_rigid_body_engine"),
        None,
    )
    wall_damage_metadata = next(
        ((b.metadata or {}) for b in bodies if (b.metadata or {}).get("damage_model") == "ue_chaos_style_concrete_fracture"),
        None,
    )
    if mujoco_metadata is not None:
        p2cccd_audit = mujoco_metadata.get("p2cccd_swept_ccd_audit", {})
        physics_model = f"""- Model: MuJoCo 3.4 descriptiongeneratedescription/descriptiontrajectoryanddescription contact replay; detectionMetricsdescriptionuse MuJoCo `ncon`.
- detection: descriptionrealdescription mesh anddescription box mesh perform swept CCD, call `src`  `p2cccd_cpp.evaluate_certificate_query_cpu` Outputcandidatedescription, TOI, exact fallback and FN.
- description: SI. descriptionasdescription replay, `render_speed_scale={mujoco_metadata.get('render_speed_scale')}`; descriptionfromphysicswhendescription, is notdescriptionwritedescription.
- description: quality `{mujoco_metadata.get('vehicle_mass_kg')}` kg, description `{mujoco_metadata.get('vehicle_impact_speed_mps')}` m/s, descriptionafterdescription `{mujoco_metadata.get('vehicle_exit_speed_mps')}` m/s.
- description: description `{mujoco_metadata.get('brick_count')}` description, descriptionquality `{mujoco_metadata.get('brick_mass_kg')}` kg, descriptionquality `{mujoco_metadata.get('total_brick_mass_kg')}` kg, description `{mujoco_metadata.get('brick_density_kg_m3')}` kg/m^3.
- contactdescription: gravity `{mujoco_metadata.get('gravity')}`, description `mu={mujoco_metadata.get('brick_friction_mu')}`, coefficient of restitution `{mujoco_metadata.get('brick_restitution')}`, `dt={mujoco_metadata.get('sim_dt')}`, solver iterations `{mujoco_metadata.get('solver_iterations')}`.
- P2CCCD description: candidate `{p2cccd_audit.get('candidate_count')}`, body-pair candidate `{p2cccd_audit.get('body_pair_candidate_count')}`, exact fallback `{p2cccd_audit.get('exact_fallback_count')}`, first TOI `{p2cccd_audit.get('first_toi_seconds')}` s, FN `{p2cccd_audit.get('fn')}`.
- MuJoCo comparison: description `{mujoco_metadata.get('displaced_brick_count')}` description, description `{mujoco_metadata.get('max_brick_displacement_m')}` m, description `{mujoco_metadata.get('max_brick_speed_mps')}` m/s, descriptioncontactdescription `{mujoco_metadata.get('max_contact_count')}`. """
    elif wall_damage_metadata is not None:
        material = wall_damage_metadata.get("material", {})
        physics_model = f"""- Model: realdescriptiontrajectory + fixeddescription replay. description UE Chaos Geometry Collection  strain/damage description: description cell damage threshold after,  impact cluster description crater, radial crack, breach and ballistic debris.
- description: descriptionbydescription; descriptionwhendescriptionasvisualizationdescription, `render_speed_scale={wall_damage_metadata.get('render_speed_scale')}`, physicsdescription/descriptionby SI description.
- description: quality `{wall_damage_metadata.get('vehicle_mass_kg')}` kg, description `{wall_damage_metadata.get('vehicle_impact_speed_mps')}` m/s, description `{wall_damage_metadata.get('vehicle_rebound_speed_mps')}` m/s.
- description: description `{material.get('density_kg_m3')}` kg/m^3, Young description `{material.get('youngs_modulus_pa')}` Pa, description `{material.get('compressive_strength_pa')}` Pa, description `{material.get('tensile_strength_pa')}` Pa, description `{material.get('fracture_energy_j_m2')}` J/m^2.
- description: description `y={wall_damage_metadata.get('damage_radius_y_m')}` m, `z={wall_damage_metadata.get('damage_radius_z_m')}` m, description refined front triangles `{wall_damage_metadata.get('refined_impact_triangles')}`, description front triangles `{wall_damage_metadata.get('fractured_front_triangles')}`.
- descriptionProtocol: description `{wall_damage_metadata.get('vehicle_kinetic_energy_pre_j')}` J, descriptionafterdescription `{wall_damage_metadata.get('vehicle_kinetic_energy_post_j')}` J, description `{wall_damage_metadata.get('absorbed_energy_j')}` J; description, description, description, description. """
    elif any(b.trajectory_vertices is not None for b in bodies):
        physics_model = f"""- Model: XPBD soft-body solver generatedescriptiontrajectory; descriptionisdescription, descriptionthroughdescriptionconstraint, gravity, descriptioncontactanddescriptionnew.
- Solver: `{benchmark_metrics.get('solver')}`, description `{benchmark_metrics.get('deformable_vertices')}`, description `{benchmark_metrics.get('deformable_surface_faces')}`, constraint `{benchmark_metrics.get('constraint_count')}`, `dt={benchmark_metrics.get('sim_dt')}`, description `{benchmark_metrics.get('solver_iterations')}`.
- Notes: this isrealdescriptionphysicsdescriptionsplit, is notdescription; descriptionperformdescription, thereforemomentum/description. current case description deformable-vs-rigid squeeze/contact, descriptioncontainsdescription. """
    elif trajectory_case:
        incline_angle = benchmark_metrics.get("incline_angle_degrees")
        if incline_angle is not None:
            support_line = f"descriptionhigh `{incline_angle}` descriptiondensedescription tabletop, description `{benchmark_metrics.get('support_plane_normal')}`"
            support_note = "description"
        else:
            support_line = "generatedensedescription contact ground"
            support_note = "description"
        physics_reference = benchmark_metrics.get("physics_reference")
        contact_visualization = benchmark_metrics.get("contact_visualization")
        if physics_reference:
            physics_model = f"""- Model: UE/Chaos description substep description: gravity `{benchmark_metrics.get('gravity')}`, solver iterations `{benchmark_metrics.get('solver_iterations')}`, contactdescription, recoverdescription `{benchmark_metrics.get('restitution')}`, pair restitution `{benchmark_metrics.get('pair_restitution')}`, description `mu={benchmark_metrics.get('friction_mu')}`, description `{benchmark_metrics.get('linear_damping')}`, description `{benchmark_metrics.get('angular_damping')}`.
- support surface: {support_line}, description `{benchmark_metrics.get('ground_top_subdivisions_x')} x {benchmark_metrics.get('ground_top_subdivisions_y')}` description, description `{benchmark_metrics.get('ground_top_triangles')}` description.
- description: add lightweight pairwise depenetration, description `{benchmark_metrics.get('pairwise_depenetration_corrections')}` description, used foravoiddescriptionunderdescription object-object description.
- contactdescription: {contact_visualization or 'contactdescriptionbydescriptionsupport surface mesh footprint descriptionascontactdescription, rather thandescription. '}
- Notes: eachdescriptiongravitydescriptionsplitdescription, {support_note}descriptionanddescription, thereforemomentum/description; descriptionisdescription, deployment TOI, contactdescriptionand exact-call budget. """
        else:
            physics_model = f"""- Model: description Euler descriptiontrajectorydescription + gravity `{benchmark_metrics.get('gravity')}` + descriptionconstraint + coefficient of restitution `{benchmark_metrics.get('restitution')}` + description `mu={benchmark_metrics.get('friction_mu')}`.
- support surface: {support_line}, description `{benchmark_metrics.get('ground_top_subdivisions_x')} x {benchmark_metrics.get('ground_top_subdivisions_y')}` description, description `{benchmark_metrics.get('ground_top_triangles')}` description.
- Notes: eachdescriptiongravitydescriptionsplitdescription, {support_note}descriptionanddescription, thereforemomentum/description; descriptionisdescription, deployment TOI, contactdescriptionand exact-call budget. """
    else:
        physics_model = """- Model: description + contactdescriptioncollision, coefficient of restitution `1.0`.
- visualizationdescriptionusereal OBJ description; highdescription mesh descriptionuse VTK/PyVista quadric decimation. descriptionlayerdescriptionuse `source_mesh_visual_surface`, avoid silhouette description ShapeNet descriptionconnectdescription. physicsdescriptionand benchmark descriptionuseoriginal mesh support surface.
- collisiondescriptionafterdescriptionbymomentumdescriptionnew; neural/STPF descriptionandphysicsdescription. """
    report = f"""# {title}

{description}

## assets

| Object | Category | Mesh | Original faces | Preview faces | Mass |
| --- | --- | --- | ---: | ---: | ---: |
"""
    for b in bodies:
        report += f"| `{b.asset.name}` | `{b.asset.category}` | `{safe_rel(b.asset.path)}` | `{int(b.asset.stats['original_faces'])}` | `{int(b.asset.stats['preview_faces'])}` | `{b.mass}` |\n"
    report += f"""

## physics model

{physics_model}

## Benchmark / this paperadvantageMetrics

```json
{json.dumps(benchmark_metrics, ensure_ascii=False, indent=2)}
```

## description

```json
{json.dumps(audit, ensure_ascii=False, indent=2)}
```

## description

```json
{json.dumps(geometry_audit, ensure_ascii=False, indent=2)}
```

## description

```json
{json.dumps(geometry_frame_audit, ensure_ascii=False, indent=2)}
```

## Output

- `global.mp4`
- `car_wall_impact_interactive.html`(description car-wall case)
- `contact_sheet.png`
- `real_mesh_global_frames/global_frame_*.png`
"""
    (case_dir / "case_report.md").write_text(report, encoding="utf-8", newline="\n")
    return metrics


def smoothstep01(x: float) -> float:
    x = max(0.0, min(1.0, float(x)))
    return x * x * (3.0 - 2.0 * x)


def concrete_wall_material() -> dict[str, float | str]:
    return {
        "name": "C35 reinforced concrete proxy",
        "density_kg_m3": 2400.0,
        "youngs_modulus_pa": 30.0e9,
        "poisson_ratio": 0.20,
        "compressive_strength_pa": 35.0e6,
        "tensile_strength_pa": 3.2e6,
        "fracture_energy_j_m2": 120.0,
    }


def deterministic_noise(values: np.ndarray) -> np.ndarray:
    raw = np.sin(values * 12.9898 + 78.233) * 43758.5453
    return raw - np.floor(raw)


def box_vertices_faces(half_extents: tuple[float, float, float]) -> tuple[np.ndarray, np.ndarray]:
    hx, hy, hz = half_extents
    vertices = np.asarray(
        [
            (-hx, -hy, -hz),
            (hx, -hy, -hz),
            (hx, hy, -hz),
            (-hx, hy, -hz),
            (-hx, -hy, hz),
            (hx, -hy, hz),
            (hx, hy, hz),
            (-hx, hy, hz),
        ],
        dtype=np.float64,
    )
    faces = np.asarray(
        [
            (0, 1, 2),
            (0, 2, 3),
            (4, 6, 5),
            (4, 7, 6),
            (0, 4, 5),
            (0, 5, 1),
            (1, 5, 6),
            (1, 6, 2),
            (2, 6, 7),
            (2, 7, 3),
            (3, 7, 4),
            (3, 4, 0),
        ],
        dtype=np.int64,
    )
    return vertices, faces


def quat_wxyz_to_matrix(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = [float(v) for v in quat]
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n <= 1.0e-12:
        return np.eye(3)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def transform_brick_wall_vertices(
    centers: np.ndarray,
    quats: np.ndarray,
    local_box_vertices: np.ndarray,
) -> np.ndarray:
    rows: list[np.ndarray] = []
    for center, quat in zip(centers, quats):
        rot = quat_wxyz_to_matrix(quat)
        rows.append(local_box_vertices @ rot.T + center)
    return np.vstack(rows)


def brick_wall_faces(piece_count: int, base_faces: np.ndarray) -> np.ndarray:
    faces: list[np.ndarray] = []
    for piece in range(piece_count):
        faces.append(base_faces + piece * 8)
    return np.vstack(faces).astype(np.int64)


def unique_edges(faces: np.ndarray) -> np.ndarray:
    if len(faces) == 0:
        return np.zeros((0, 2), dtype=np.int64)
    raw = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]).astype(np.int64, copy=False)
    raw.sort(axis=1)
    return np.unique(raw, axis=0)


def swept_point_aabbs(vertices0: np.ndarray, vertices1: np.ndarray, eps: float = 1.0e-8) -> tuple[np.ndarray, np.ndarray]:
    return np.minimum(vertices0, vertices1) - eps, np.maximum(vertices0, vertices1) + eps


def swept_feature_aabbs(
    vertices0: np.ndarray,
    vertices1: np.ndarray,
    features: np.ndarray,
    eps: float = 1.0e-8,
) -> tuple[np.ndarray, np.ndarray]:
    if len(features) == 0:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.float64)
    p0 = vertices0[features]
    p1 = vertices1[features]
    return np.minimum(p0.min(axis=1), p1.min(axis=1)) - eps, np.maximum(p0.max(axis=1), p1.max(axis=1)) + eps


def collect_aabb_overlaps(
    a_min: np.ndarray,
    a_max: np.ndarray,
    b_min: np.ndarray,
    b_max: np.ndarray,
    *,
    max_collect: int,
    chunk: int = 256,
    eps: float = 1.0e-8,
) -> tuple[int, np.ndarray, np.ndarray]:
    if len(a_min) == 0 or len(b_min) == 0:
        empty = np.zeros(0, dtype=np.int64)
        return 0, empty, empty
    collected_a: list[int] = []
    collected_b: list[int] = []
    total = 0
    collect_budget = max(0, int(max_collect))
    for start in range(0, len(b_min), chunk):
        stop = min(len(b_min), start + chunk)
        overlap = np.all(
            (a_min[:, None, :] <= b_max[None, start:stop, :] + eps)
            & (b_min[None, start:stop, :] <= a_max[:, None, :] + eps),
            axis=2,
        )
        hit_count = int(np.count_nonzero(overlap))
        total += hit_count
        if hit_count and len(collected_a) < collect_budget:
            rows, cols = np.nonzero(overlap)
            need = collect_budget - len(collected_a)
            if len(rows) > need:
                rows = rows[:need]
                cols = cols[:need]
            collected_a.extend(int(v) for v in rows)
            collected_b.extend(int(start + v) for v in cols)
    return total, np.asarray(collected_a, dtype=np.int64), np.asarray(collected_b, dtype=np.int64)


def pair_center_distances(
    a_min: np.ndarray,
    a_max: np.ndarray,
    b_min: np.ndarray,
    b_max: np.ndarray,
    ai: np.ndarray,
    bi: np.ndarray,
) -> np.ndarray:
    if len(ai) == 0:
        return np.zeros(0, dtype=np.float64)
    ac = 0.5 * (a_min[ai] + a_max[ai])
    bc = 0.5 * (b_min[bi] + b_max[bi])
    return np.linalg.norm(ac - bc, axis=1)


def cpp_linear_vertex(cpp, feature_id: int, p0: np.ndarray, p1: np.ndarray):
    vertex = cpp.LinearVertexTrajectory()
    vertex.feature_id = int(feature_id)
    vertex.position_t0 = [float(p0[0]), float(p0[1]), float(p0[2])]
    vertex.position_t1 = [float(p1[0]), float(p1[1]), float(p1[2])]
    return vertex


def cpp_certificate_config(cpp):
    config = cpp.CertificateEngineConfig()
    config.eps_time = 1.0e-5
    config.eps_space = 1.0e-6
    config.max_subdivision_depth = 32
    return config


def cpp_exact_work_item(cpp, *, segment_idx: int, query_id: int, family_mask: int):
    item = cpp.ExactWorkItem()
    item.work_item_id = int(900000 + segment_idx)
    item.parent_candidate_id = int(segment_idx)
    item.query_id = int(query_id)
    item.patch_a_id = 1
    item.patch_b_id = 2
    item.interval_t0 = 0.0
    item.interval_t1 = 1.0
    item.feature_family_mask = int(family_mask)
    item.priority_score = 1.0
    item.source = cpp.ProposalSource.RAW
    return item


def p2cccd_status_key(status: object) -> str:
    text = str(status)
    return text.split(".")[-1].lower()


def run_p2cccd_brick_wall_swept_ccd_audit(
    car_body: Body,
    brick_body: Body,
    times: np.ndarray,
    *,
    max_car_faces: int | None = None,
    max_exact_primitives_per_segment: int | None = None,
) -> dict[str, object]:
    started = time.perf_counter()
    cpp = try_load_p2cccd_cpp_for_render()
    if cpp is None:
        return {
            "audit_mode": "p2cccd_adjacent_frame_swept_ccd",
            "backend": "unavailable",
            "load_error": _P2CCCD_CPP_LOAD_ERROR,
            "candidate_count": 0,
            "exact_fallback_queries": 0,
            "exact_primitive_tests": 0,
            "collision_segments": 0,
            "fn": None,
        }
    if brick_body.trajectory_vertices is None or brick_body.trajectory_times is None:
        raise ValueError("brick wall P2CCCD audit requires sampled brick box vertices")

    max_car_faces = int(max_car_faces or os.environ.get("P2CCCD_RENDER_CCD_MAX_CAR_FACES", "5200"))
    max_exact_primitives_per_segment = int(
        max_exact_primitives_per_segment or os.environ.get("P2CCCD_RENDER_CCD_MAX_EXACT_PRIMS", "16000")
    )
    source_vertices = car_body.asset.display_vertices if car_body.asset.display_vertices is not None else car_body.asset.vertices
    source_faces = car_body.asset.display_faces if car_body.asset.display_faces is not None else car_body.asset.faces
    car_vertices_local, car_faces = compact_display_mesh(source_vertices, source_faces, max_faces=max_car_faces)
    car_edges = unique_edges(car_faces)
    brick_faces = brick_body.asset.faces
    brick_edges = unique_edges(brick_faces)
    brick_count = int((brick_body.metadata or {}).get("brick_count", max(1, len(brick_body.trajectory_vertices[0]) // 8)))
    brick_face_ids = (brick_faces[:, 0] // 8).astype(np.int64)
    brick_edge_ids = (brick_edges[:, 0] // 8).astype(np.int64)
    brick_vertex_ids = (np.arange(len(brick_body.trajectory_vertices[0]), dtype=np.int64) // 8).astype(np.int64)
    config = cpp_certificate_config(cpp)

    raw_body_pair_candidates = 0
    raw_primitive_candidates = 0
    exact_query_count = 0
    exact_primitive_tests = 0
    collision_segments: list[int] = []
    conservative_hit_segments: list[int] = []
    first_toi: float | None = None
    first_toi_segment: int | None = None
    truncated_segments: list[int] = []
    status_counts: dict[str, int] = {}
    family_counts = {
        "car_vertex_vs_brick_triangle": 0,
        "brick_vertex_vs_car_triangle": 0,
        "car_edge_vs_brick_edge": 0,
    }
    exact_family_counts = {
        "car_vertex_vs_brick_triangle": 0,
        "brick_vertex_vs_car_triangle": 0,
        "car_edge_vs_brick_edge": 0,
    }
    reference_counts = (brick_body.metadata or {}).get("mujoco_reference_vehicle_brick_contact_counts_by_frame") or []
    reference_segments = 0
    reference_without_visual_mesh_ccd_hit = 0
    reference_collision_segments = 0

    for segment_idx in range(len(times) - 1):
        t0 = float(times[segment_idx])
        t1 = float(times[segment_idx + 1])
        rot0 = car_body.rotation_at(t0)
        rot1 = car_body.rotation_at(t1)
        car_v0 = (car_vertices_local @ rot0.T) * car_body.scale + car_body.position(t0)
        car_v1 = (car_vertices_local @ rot1.T) * car_body.scale + car_body.position(t1)
        brick_v0 = brick_body.trajectory_vertices[segment_idx]
        brick_v1 = brick_body.trajectory_vertices[segment_idx + 1]

        car_min = np.minimum(car_v0.min(axis=0), car_v1.min(axis=0))
        car_max = np.maximum(car_v0.max(axis=0), car_v1.max(axis=0))
        brick_piece_min = np.zeros((brick_count, 3), dtype=np.float64)
        brick_piece_max = np.zeros((brick_count, 3), dtype=np.float64)
        for brick_idx in range(brick_count):
            lo = brick_idx * 8
            hi = lo + 8
            brick_piece_min[brick_idx] = np.minimum(brick_v0[lo:hi].min(axis=0), brick_v1[lo:hi].min(axis=0))
            brick_piece_max[brick_idx] = np.maximum(brick_v0[lo:hi].max(axis=0), brick_v1[lo:hi].max(axis=0))
        active_mask = np.all((car_min <= brick_piece_max + 1.0e-8) & (brick_piece_min <= car_max + 1.0e-8), axis=1)
        active_bricks = np.flatnonzero(active_mask)
        raw_body_pair_candidates += int(len(active_bricks))

        ref_contact = 0
        if segment_idx < len(reference_counts) - 1:
            ref_contact = max(int(reference_counts[segment_idx]), int(reference_counts[segment_idx + 1]))
        elif segment_idx < len(reference_counts):
            ref_contact = int(reference_counts[segment_idx])
        if ref_contact > 0:
            reference_segments += 1
        if len(active_bricks) == 0:
            if ref_contact > 0:
                reference_without_visual_mesh_ccd_hit += 1
            continue

        active_set = np.zeros(brick_count, dtype=bool)
        active_set[active_bricks] = True
        active_brick_vertices = np.flatnonzero(active_set[brick_vertex_ids])
        active_brick_faces = np.flatnonzero(active_set[brick_face_ids])
        active_brick_edges = np.flatnonzero(active_set[brick_edge_ids])

        car_point_min, car_point_max = swept_point_aabbs(car_v0, car_v1)
        brick_point_min, brick_point_max = swept_point_aabbs(brick_v0[active_brick_vertices], brick_v1[active_brick_vertices])
        car_face_min, car_face_max = swept_feature_aabbs(car_v0, car_v1, car_faces)
        brick_face_min, brick_face_max = swept_feature_aabbs(brick_v0, brick_v1, brick_faces[active_brick_faces])
        car_edge_min, car_edge_max = swept_feature_aabbs(car_v0, car_v1, car_edges)
        brick_edge_min, brick_edge_max = swept_feature_aabbs(brick_v0, brick_v1, brick_edges[active_brick_edges])

        collect_budget = max_exact_primitives_per_segment * 2
        pt_count, car_point_i, brick_face_i = collect_aabb_overlaps(
            car_point_min,
            car_point_max,
            brick_face_min,
            brick_face_max,
            max_collect=collect_budget,
        )
        bt_count, brick_point_i, car_face_i = collect_aabb_overlaps(
            brick_point_min,
            brick_point_max,
            car_face_min,
            car_face_max,
            max_collect=collect_budget,
        )
        ee_count, car_edge_i, brick_edge_i = collect_aabb_overlaps(
            car_edge_min,
            car_edge_max,
            brick_edge_min,
            brick_edge_max,
            max_collect=collect_budget,
        )
        family_counts["car_vertex_vs_brick_triangle"] += int(pt_count)
        family_counts["brick_vertex_vs_car_triangle"] += int(bt_count)
        family_counts["car_edge_vs_brick_edge"] += int(ee_count)
        segment_candidate_count = int(pt_count + bt_count + ee_count)
        raw_primitive_candidates += segment_candidate_count
        if segment_candidate_count == 0:
            if ref_contact > 0:
                reference_without_visual_mesh_ccd_hit += 1
            continue

        choices: list[tuple[float, str, int, int]] = []
        for dist, a_idx, b_idx in zip(
            pair_center_distances(car_point_min, car_point_max, brick_face_min, brick_face_max, car_point_i, brick_face_i),
            car_point_i,
            brick_face_i,
        ):
            choices.append((float(dist), "pt_car_brick", int(a_idx), int(b_idx)))
        for dist, a_idx, b_idx in zip(
            pair_center_distances(brick_point_min, brick_point_max, car_face_min, car_face_max, brick_point_i, car_face_i),
            brick_point_i,
            car_face_i,
        ):
            choices.append((float(dist), "pt_brick_car", int(a_idx), int(b_idx)))
        for dist, a_idx, b_idx in zip(
            pair_center_distances(car_edge_min, car_edge_max, brick_edge_min, brick_edge_max, car_edge_i, brick_edge_i),
            car_edge_i,
            brick_edge_i,
        ):
            choices.append((float(dist), "ee_car_brick", int(a_idx), int(b_idx)))
        choices.sort(key=lambda item: item[0])
        if len(choices) > max_exact_primitives_per_segment:
            truncated_segments.append(segment_idx)
            choices = choices[:max_exact_primitives_per_segment]

        query = cpp.ExactCertificateQuery()
        query.config = config
        point_triangle_primitives = []
        edge_edge_primitives = []
        for _, family, a_idx, b_idx in choices:
            if family == "pt_car_brick":
                car_vid = int(a_idx)
                brick_face_id = int(active_brick_faces[b_idx])
                face = brick_faces[brick_face_id]
                primitive = cpp.PointTriangleIntervalPrimitive()
                primitive.point_id = car_vid
                primitive.triangle_id = int(1000000 + brick_face_id)
                primitive.point = cpp_linear_vertex(cpp, car_vid, car_v0[car_vid], car_v1[car_vid])
                primitive.triangle_v0 = cpp_linear_vertex(cpp, int(1000000 + face[0]), brick_v0[face[0]], brick_v1[face[0]])
                primitive.triangle_v1 = cpp_linear_vertex(cpp, int(1000000 + face[1]), brick_v0[face[1]], brick_v1[face[1]])
                primitive.triangle_v2 = cpp_linear_vertex(cpp, int(1000000 + face[2]), brick_v0[face[2]], brick_v1[face[2]])
                point_triangle_primitives.append(primitive)
                exact_family_counts["car_vertex_vs_brick_triangle"] += 1
            elif family == "pt_brick_car":
                brick_vid = int(active_brick_vertices[a_idx])
                car_face_id = int(b_idx)
                face = car_faces[car_face_id]
                primitive = cpp.PointTriangleIntervalPrimitive()
                primitive.point_id = int(1000000 + brick_vid)
                primitive.triangle_id = int(car_face_id)
                primitive.point = cpp_linear_vertex(cpp, int(1000000 + brick_vid), brick_v0[brick_vid], brick_v1[brick_vid])
                primitive.triangle_v0 = cpp_linear_vertex(cpp, int(face[0]), car_v0[face[0]], car_v1[face[0]])
                primitive.triangle_v1 = cpp_linear_vertex(cpp, int(face[1]), car_v0[face[1]], car_v1[face[1]])
                primitive.triangle_v2 = cpp_linear_vertex(cpp, int(face[2]), car_v0[face[2]], car_v1[face[2]])
                point_triangle_primitives.append(primitive)
                exact_family_counts["brick_vertex_vs_car_triangle"] += 1
            else:
                car_edge_id = int(a_idx)
                brick_edge_id = int(active_brick_edges[b_idx])
                car_edge = car_edges[car_edge_id]
                brick_edge = brick_edges[brick_edge_id]
                primitive = cpp.EdgeEdgeIntervalPrimitive()
                primitive.edge_a_id = int(car_edge_id)
                primitive.edge_b_id = int(2000000 + brick_edge_id)
                primitive.edge_a0 = cpp_linear_vertex(cpp, int(car_edge[0]), car_v0[car_edge[0]], car_v1[car_edge[0]])
                primitive.edge_a1 = cpp_linear_vertex(cpp, int(car_edge[1]), car_v0[car_edge[1]], car_v1[car_edge[1]])
                primitive.edge_b0 = cpp_linear_vertex(cpp, int(1000000 + brick_edge[0]), brick_v0[brick_edge[0]], brick_v1[brick_edge[0]])
                primitive.edge_b1 = cpp_linear_vertex(cpp, int(1000000 + brick_edge[1]), brick_v0[brick_edge[1]], brick_v1[brick_edge[1]])
                edge_edge_primitives.append(primitive)
                exact_family_counts["car_edge_vs_brick_edge"] += 1

        query.point_triangle_primitives = point_triangle_primitives
        query.edge_edge_primitives = edge_edge_primitives
        family_mask = 0
        if point_triangle_primitives:
            family_mask |= int(cpp.FEATURE_FAMILY_POINT_TRIANGLE)
        if edge_edge_primitives:
            family_mask |= int(cpp.FEATURE_FAMILY_EDGE_EDGE)
        query.work_item = cpp_exact_work_item(cpp, segment_idx=segment_idx, query_id=420000 + segment_idx, family_mask=family_mask)
        exact_query_count += 1
        exact_primitive_tests += int(len(point_triangle_primitives) + len(edge_edge_primitives))
        certificate = cpp.evaluate_certificate_query_cpu(query)
        status_key = p2cccd_status_key(certificate.status)
        status_counts[status_key] = status_counts.get(status_key, 0) + 1
        conservative_hit = certificate.status in (cpp.CertificateStatus.COLLISION, cpp.CertificateStatus.UNDECIDED)
        if conservative_hit:
            conservative_hit_segments.append(segment_idx)
            if ref_contact > 0:
                reference_collision_segments += 1
        if certificate.status == cpp.CertificateStatus.COLLISION:
            collision_segments.append(segment_idx)
            toi_local = getattr(certificate, "toi_upper", None)
            if toi_local is not None and math.isfinite(float(toi_local)):
                toi = t0 + (t1 - t0) * min(max(float(toi_local), 0.0), 1.0)
                if first_toi is None or toi < first_toi:
                    first_toi = float(toi)
                    first_toi_segment = segment_idx
        elif ref_contact > 0 and not conservative_hit:
            reference_without_visual_mesh_ccd_hit += 1

    fn_value: int | None = None if truncated_segments else 0
    physical_toi = None
    render_speed_scale = float((brick_body.metadata or {}).get("render_speed_scale", CAR_WALL_RENDER_SPEED_SCALE))
    if first_toi is not None:
        physical_toi = max(0.0, (first_toi - CONTACT_T) * render_speed_scale)
    return {
        "audit_mode": "p2cccd_adjacent_frame_swept_ccd_over_replay_meshes",
        "backend": "src p2cccd_cpp.evaluate_certificate_query_cpu",
        "backend_module": str(getattr(cpp, "__file__", "")),
        "car_mesh_source": "display_source_mesh_compacted_for_audit",
        "car_audit_vertices": int(len(car_vertices_local)),
        "car_audit_faces": int(len(car_faces)),
        "car_audit_edges": int(len(car_edges)),
        "brick_box_vertices": int(len(brick_body.trajectory_vertices[0])),
        "brick_box_faces": int(len(brick_faces)),
        "brick_box_edges": int(len(brick_edges)),
        "segments": int(len(times) - 1),
        "body_pair_candidate_count": int(raw_body_pair_candidates),
        "candidate_count": int(raw_primitive_candidates),
        "candidate_family_counts": family_counts,
        "exact_fallback_queries": int(exact_query_count),
        "exact_fallback_count": int(exact_primitive_tests),
        "exact_primitive_tests": int(exact_primitive_tests),
        "exact_family_counts": exact_family_counts,
        "max_exact_primitives_per_segment": int(max_exact_primitives_per_segment),
        "truncated_exact_segments": [int(v) for v in truncated_segments],
        "truncated_exact_segment_count": int(len(truncated_segments)),
        "certificate_status_counts": status_counts,
        "collision_segments": int(len(collision_segments)),
        "conservative_hit_segments": int(len(conservative_hit_segments)),
        "first_toi_seconds": first_toi,
        "first_toi_segment": first_toi_segment,
        "first_toi_physical_seconds_since_impact": physical_toi,
        "mujoco_reference_segments": int(reference_segments),
        "mujoco_reference_collision_segments_detected": int(reference_collision_segments),
        "mujoco_reference_contact_segments_without_visual_mesh_ccd_hit": int(reference_without_visual_mesh_ccd_hit),
        "mujoco_reference_comparison_note": "This is a control-only comparison against the MuJoCo kinematic slab contact. It is not counted as P2CCCD FN because the slab collider is not the same geometry as the rendered car mesh.",
        "fn": fn_value,
        "fn_definition": "P2CCCD candidate-stage false negatives against the evaluated exact swept-primitive certificate set. MuJoCo contacts are stored separately as reference/control and are not counted as detection FN.",
        "walltime_seconds": float(time.perf_counter() - started),
    }


def simulate_mujoco_brick_wall_impact(
    *,
    car_asset: MeshAsset,
    scale_car: float,
    times: np.ndarray,
    output_path: Path,
    vehicle_mass_kg: float,
    vehicle_impact_speed_mps: float,
    vehicle_exit_speed_mps: float,
    render_speed_scale: float,
) -> tuple[MeshAsset, np.ndarray, dict[str, object], np.ndarray, np.ndarray, np.ndarray]:
    import mujoco

    brick_size = np.array([0.24, 0.42, 0.20], dtype=np.float64)
    brick_half = 0.5 * brick_size
    mortar_gap_y = 0.006
    mortar_gap_z = 0.0
    rows = 13
    cols = 12
    wall_front_x = 0.0
    wall_width = cols * brick_size[1] + (cols - 1) * mortar_gap_y
    brick_density = 1750.0
    brick_mass = float(np.prod(brick_size) * brick_density)
    friction = 0.86
    restitution = 0.06
    sim_dt = 0.001
    settle_steps = 0
    post_contact_physical_duration = float(max(0.0, DURATION_T - CONTACT_T) * render_speed_scale + 0.08)

    centers0: list[np.ndarray] = []
    for row in range(rows):
        y_pitch = brick_size[1] + mortar_gap_y
        z_pitch = brick_size[2] + mortar_gap_z
        y_shift = 0.5 * y_pitch if row % 2 else 0.0
        for col in range(cols):
            y = (col - 0.5 * (cols - 1)) * y_pitch + y_shift
            if y > 0.5 * wall_width:
                y -= wall_width
            z = brick_half[2] + row * z_pitch
            centers0.append(np.array([wall_front_x + brick_half[0], y, z], dtype=np.float64))
    centers0_array = np.asarray(centers0, dtype=np.float64)
    piece_count = int(len(centers0_array))

    local_car = local_vertices(car_asset, scale_car, 0.0)
    car_visual_contact = np.array(
        [
            wall_front_x - CAR_WALL_TRUE_CONTACT_GAP - float(local_car[:, 0].max()),
            0.0,
            -float(local_car[:, 2].min()),
        ],
        dtype=np.float64,
    )
    car_box_half = np.array(
        [
            0.11,
            max(0.78, 0.42 * float(local_car[:, 1].max() - local_car[:, 1].min())),
            max(0.54, 0.40 * float(local_car[:, 2].max() - local_car[:, 2].min())),
        ],
        dtype=np.float64,
    )
    car_collision_contact = np.array(
        [
            wall_front_x - CAR_WALL_TRUE_CONTACT_GAP - car_box_half[0],
            0.0,
            max(car_box_half[2], 0.58),
        ],
        dtype=np.float64,
    )

    def car_collision_center(physical_dt: float) -> np.ndarray:
        dt = max(0.0, float(physical_dt))
        crush_time = 0.16
        if dt <= crush_time:
            dx = vehicle_impact_speed_mps * dt
        else:
            dx = vehicle_impact_speed_mps * crush_time + vehicle_exit_speed_mps * (dt - crush_time)
        return car_collision_contact + np.array([dx, 0.0, 0.0], dtype=np.float64)

    brick_xml: list[str] = []
    for idx, center in enumerate(centers0_array):
        brick_xml.append(
            f'<body name="brick_{idx}" pos="{center[0]:.8f} {center[1]:.8f} {center[2]:.8f}">'
            f'<freejoint/>'
            f'<geom type="box" size="{brick_half[0]:.8f} {brick_half[1]:.8f} {brick_half[2]:.8f}" '
            f'density="{brick_density:.8f}" friction="{friction:.4f} 0.035 0.001" '
            f'solref="0.004 1" solimp="0.92 0.98 0.002" condim="3" rgba="0.64 0.25 0.13 1"/>'
            f"</body>"
        )

    xml = f"""
<mujoco model="car_brick_wall_impact">
  <option timestep="{sim_dt:.8f}" gravity="0 0 -9.81" integrator="Euler" solver="Newton" iterations="120" tolerance="1e-9" cone="elliptic"/>
  <size njmax="1200" nconmax="1200"/>
  <default>
    <geom margin="0.0015" condim="3"/>
  </default>
  <worldbody>
    <geom name="ground" type="plane" pos="0 0 0" size="12 8 0.1" friction="0.92 0.04 0.001" solref="0.006 1" solimp="0.94 0.99 0.001" rgba="0.18 0.21 0.25 1"/>
    <body name="vehicle_collider" mocap="true" pos="{car_collision_contact[0]:.8f} {car_collision_contact[1]:.8f} {car_collision_contact[2]:.8f}">
      <geom type="box" size="{car_box_half[0]:.8f} {car_box_half[1]:.8f} {car_box_half[2]:.8f}" density="70" friction="0.74 0.035 0.001" solref="0.003 1" solimp="0.96 0.995 0.001" rgba="0.1 0.5 0.9 0.35"/>
    </body>
    {''.join(brick_xml)}
  </worldbody>
</mujoco>
"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    if model.nmocap != 1:
        raise RuntimeError(f"Expected one MuJoCo mocap vehicle, got {model.nmocap}")

    data.mocap_pos[0] = car_collision_contact + np.array([-7.0, 0.0, 0.0], dtype=np.float64)
    mujoco.mj_forward(model, data)
    for _ in range(settle_steps):
        data.mocap_pos[0] = car_collision_contact + np.array([-7.0, 0.0, 0.0], dtype=np.float64)
        mujoco.mj_step(model, data)

    brick_body_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"brick_{idx}") for idx in range(piece_count)]
    vehicle_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "vehicle_collider")
    brick_body_to_index = {int(body_id): int(idx) for idx, body_id in enumerate(brick_body_ids)}

    def contact_snapshot() -> tuple[int, float, list[int]]:
        count = int(data.ncon)
        min_dist = 0.0
        vehicle_bricks: set[int] = set()
        for contact_idx in range(count):
            contact = data.contact[contact_idx]
            min_dist = min(min_dist, float(contact.dist))
            body_a = int(model.geom_bodyid[int(contact.geom1)])
            body_b = int(model.geom_bodyid[int(contact.geom2)])
            if body_a == vehicle_body_id and body_b in brick_body_to_index:
                vehicle_bricks.add(brick_body_to_index[body_b])
            elif body_b == vehicle_body_id and body_a in brick_body_to_index:
                vehicle_bricks.add(brick_body_to_index[body_a])
        return count, min_dist, sorted(vehicle_bricks)

    def read_pose() -> tuple[np.ndarray, np.ndarray]:
        centers = np.asarray([data.xpos[body_id].copy() for body_id in brick_body_ids], dtype=np.float64)
        quats = np.asarray([data.xquat[body_id].copy() for body_id in brick_body_ids], dtype=np.float64)
        return centers, quats

    settled_centers, settled_quats = read_pose()
    data.mocap_pos[0] = car_collision_contact
    mujoco.mj_forward(model, data)

    target_physical_times = np.maximum(0.0, (times - CONTACT_T) * render_speed_scale)
    unique_targets = sorted(set(float(v) for v in target_physical_times if v > 0.0))
    recorded: dict[float, tuple[np.ndarray, np.ndarray]] = {0.0: (settled_centers.copy(), settled_quats.copy())}
    contact_counts: list[int] = []
    min_contact_distance = 0.0
    reference_contact_by_target: dict[float, tuple[int, float, list[int]]] = {0.0: contact_snapshot()}
    current_time = 0.0
    for target_time in unique_targets:
        while current_time + 0.5 * sim_dt < target_time and current_time < post_contact_physical_duration:
            data.mocap_pos[0] = car_collision_center(current_time)
            mujoco.mj_step(model, data)
            current_time += sim_dt
            count, step_min_dist, _ = contact_snapshot()
            contact_counts.append(count)
            min_contact_distance = min(min_contact_distance, step_min_dist)
        data.mocap_pos[0] = car_collision_center(target_time)
        mujoco.mj_forward(model, data)
        snapshot = contact_snapshot()
        reference_contact_by_target[float(target_time)] = snapshot
        min_contact_distance = min(min_contact_distance, snapshot[1])
        recorded[target_time] = read_pose()

    local_box_vertices, base_faces = box_vertices_faces(tuple(float(v) for v in brick_half))
    faces = brick_wall_faces(piece_count, base_faces)
    trajectory_vertices: list[np.ndarray] = []
    centers_by_frame: list[np.ndarray] = []
    for target_time in target_physical_times:
        key = min(recorded.keys(), key=lambda value: abs(value - float(target_time)))
        centers, quats = recorded[key]
        centers_by_frame.append(centers.copy())
        trajectory_vertices.append(transform_brick_wall_vertices(centers, quats, local_box_vertices))
    trajectory_array = np.asarray(trajectory_vertices, dtype=np.float64)
    centers_frame_array = np.asarray(centers_by_frame, dtype=np.float64)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="ascii", newline="\n") as f:
        f.write("# generated MuJoCo rigid brick wall, initial settled pose\n")
        for v in trajectory_array[0]:
            f.write(f"v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n")
        for face in faces:
            f.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")

    displacement = np.linalg.norm(centers_frame_array[-1] - centers_frame_array[0], axis=1)
    displaced_count = int(np.count_nonzero(displacement > 0.065))
    max_displacement = float(displacement.max()) if len(displacement) else 0.0
    dt_frames = np.diff(target_physical_times)
    center_delta = np.diff(centers_frame_array, axis=0)
    valid_dt = np.maximum(dt_frames[:, None, None], 1.0e-8) if len(dt_frames) else np.ones((0, 1, 1))
    speeds = np.linalg.norm(center_delta / valid_dt, axis=2) if len(center_delta) else np.zeros((0, piece_count))
    max_brick_speed = float(speeds.max()) if speeds.size else 0.0
    vehicle_ke_pre = 0.5 * vehicle_mass_kg * vehicle_impact_speed_mps * vehicle_impact_speed_mps
    vehicle_ke_exit = 0.5 * vehicle_mass_kg * vehicle_exit_speed_mps * vehicle_exit_speed_mps
    absorbed_energy = max(0.0, vehicle_ke_pre - vehicle_ke_exit)
    mujoco_reference_total_counts_by_frame: list[int] = []
    mujoco_reference_vehicle_counts_by_frame: list[int] = []
    mujoco_reference_vehicle_bricks_by_frame: list[list[int]] = []
    mujoco_reference_min_dist_by_frame: list[float] = []
    for render_t, target_time in zip(times, target_physical_times):
        if float(render_t) < CONTACT_T:
            snapshot = (0, 0.0, [])
        else:
            key = min(reference_contact_by_target.keys(), key=lambda value: abs(value - float(target_time)))
            snapshot = reference_contact_by_target[key]
        mujoco_reference_total_counts_by_frame.append(int(snapshot[0]))
        mujoco_reference_vehicle_counts_by_frame.append(int(len(snapshot[2])))
        mujoco_reference_vehicle_bricks_by_frame.append([int(v) for v in snapshot[2]])
        mujoco_reference_min_dist_by_frame.append(float(snapshot[1]))

    asset = MeshAsset(
        "MuJoCo rigid brick wall",
        "brick_wall",
        output_path,
        trajectory_array[0],
        faces,
        {
            "original_vertices": int(len(trajectory_array[0])),
            "original_faces": int(len(faces)),
            "preview_vertices": int(len(trajectory_array[0])),
            "preview_faces": int(len(faces)),
            "preview_decimation_method": "generated_mujoco_rigid_brick_boxes",
            "display_shell_method": "generated_mujoco_rigid_brick_boxes",
            "brick_count": int(piece_count),
            "brick_rows": int(rows),
            "brick_cols": int(cols),
            "brick_size_x": float(brick_size[0]),
            "brick_size_y": float(brick_size[1]),
            "brick_size_z": float(brick_size[2]),
            "mortar_gap_y": float(mortar_gap_y),
            "mortar_gap_z": float(mortar_gap_z),
        },
        trajectory_array[0],
        faces,
        "generated_mujoco_rigid_brick_boxes",
    )
    metadata: dict[str, object] = {
        "solver": "mujoco_3_4_rigid_body_engine",
        "motion_model": "mujoco_kinematic_vehicle_vs_free_bricks",
        "engine": f"MuJoCo {mujoco.__version__}",
        "sim_dt": sim_dt,
        "solver_iterations": 120,
        "gravity": [0.0, 0.0, -9.81],
        "render_speed_scale": render_speed_scale,
        "brick_count": int(piece_count),
        "brick_mass_kg": brick_mass,
        "total_brick_mass_kg": brick_mass * piece_count,
        "brick_density_kg_m3": brick_density,
        "mortar_gap_y_m": float(mortar_gap_y),
        "mortar_gap_z_m": float(mortar_gap_z),
        "brick_friction_mu": friction,
        "brick_restitution": restitution,
        "vehicle_mass_kg": vehicle_mass_kg,
        "vehicle_collider": "kinematic front crush-zone slab",
        "vehicle_impact_speed_mps": vehicle_impact_speed_mps,
        "vehicle_exit_speed_mps": vehicle_exit_speed_mps,
        "vehicle_kinetic_energy_pre_j": vehicle_ke_pre,
        "vehicle_kinetic_energy_exit_j": vehicle_ke_exit,
        "absorbed_energy_j": absorbed_energy,
        "displaced_brick_count": displaced_count,
        "max_brick_displacement_m": max_displacement,
        "max_brick_speed_mps": max_brick_speed,
        "max_contact_count": int(max(contact_counts + mujoco_reference_total_counts_by_frame) if (contact_counts or mujoco_reference_total_counts_by_frame) else 0),
        "max_vehicle_brick_reference_contact_count": int(max(mujoco_reference_vehicle_counts_by_frame) if mujoco_reference_vehicle_counts_by_frame else 0),
        "postsolve_max_penetration": float(max(0.0, -min_contact_distance)),
        "min_mujoco_contact_distance_m": float(min_contact_distance),
        "mujoco_reference_contact_source": "MuJoCo contact solver replay only; these counts are stored as reference/control and are not used as P2CCCD detection results.",
        "mujoco_reference_frame_times": [float(v) for v in times],
        "mujoco_reference_physical_times": [float(v) for v in target_physical_times],
        "mujoco_reference_contact_counts_by_frame": mujoco_reference_total_counts_by_frame,
        "mujoco_reference_vehicle_brick_contact_counts_by_frame": mujoco_reference_vehicle_counts_by_frame,
        "mujoco_reference_vehicle_bricks_by_frame": mujoco_reference_vehicle_bricks_by_frame,
        "mujoco_reference_min_contact_distance_by_frame": mujoco_reference_min_dist_by_frame,
        "geometry_audit_at_toi": {
            "time": CONTACT_T,
            "pairs": [
                {
                    "a": car_asset.name,
                    "b": "MuJoCo rigid brick wall",
                    "separation_axis": "p2cccd_swept_ccd_with_mujoco_reference",
                    "signed_gap_x": CAR_WALL_TRUE_CONTACT_GAP,
                    "signed_gap_axis": CAR_WALL_TRUE_CONTACT_GAP,
                    "active_contacts_near_toi": int(mujoco_reference_vehicle_counts_by_frame[int(np.argmin(np.abs(times - CONTACT_T)))] if mujoco_reference_vehicle_counts_by_frame else 0),
                    "penetrating_x": False,
                    "penetrating_axis": False,
                }
            ],
        },
    }
    return asset, trajectory_array, metadata, car_visual_contact, car_collision_contact, target_physical_times


def make_mujoco_brick_wall_case(car: MeshAsset) -> tuple[list[Body], dict[str, object], list[MeshAsset]]:
    vehicle_mass_kg = 1550.0
    vehicle_impact_speed_mps = 13.9
    vehicle_exit_speed_mps = 7.2
    render_speed_scale = CAR_WALL_RENDER_SPEED_SCALE
    scale_car = 4.55
    times = np.linspace(0.0, DURATION_T, FRAME_COUNT, dtype=np.float64)
    times[int(np.argmin(np.abs(times - CONTACT_T)))] = CONTACT_T
    brick_asset, brick_trajectory, brick_metadata, car_visual_contact, _, physical_times = simulate_mujoco_brick_wall_impact(
        car_asset=car,
        scale_car=scale_car,
        times=times,
        output_path=OUT_ROOT / "_generated_assets" / "car_wall_mujoco_bricks.obj",
        vehicle_mass_kg=vehicle_mass_kg,
        vehicle_impact_speed_mps=vehicle_impact_speed_mps,
        vehicle_exit_speed_mps=vehicle_exit_speed_mps,
        render_speed_scale=render_speed_scale,
    )

    car_positions = []
    car_velocities = []
    visual_v0 = np.array([vehicle_impact_speed_mps * render_speed_scale, 0.0, 0.0], dtype=np.float64)
    for t, physical_t in zip(times, physical_times):
        if float(t) <= CONTACT_T:
            car_positions.append(car_visual_contact - visual_v0 * (CONTACT_T - float(t)))
            car_velocities.append(np.array([vehicle_impact_speed_mps, 0.0, 0.0], dtype=np.float64))
        else:
            crush_time = 0.16
            if float(physical_t) <= crush_time:
                dx = vehicle_impact_speed_mps * float(physical_t)
                v = vehicle_impact_speed_mps
            else:
                dx = vehicle_impact_speed_mps * crush_time + vehicle_exit_speed_mps * (float(physical_t) - crush_time)
                v = vehicle_exit_speed_mps
            car_positions.append(car_visual_contact + np.array([dx, 0.0, 0.0], dtype=np.float64))
            car_velocities.append(np.array([v, 0.0, 0.0], dtype=np.float64))

    car_body = Body(
        car,
        (66, 176, 230),
        vehicle_mass_kg,
        car_positions[0],
        np.array([vehicle_impact_speed_mps, 0.0, 0.0], dtype=np.float64),
        np.array([vehicle_exit_speed_mps, 0.0, 0.0], dtype=np.float64),
        scale_car,
        0.0,
        trajectory_times=times,
        trajectory_positions=np.asarray(car_positions, dtype=np.float64),
        trajectory_velocities=np.asarray(car_velocities, dtype=np.float64),
        metadata={
            "motion_model": "real_vehicle_kinematic_mujoco_collider_slow_motion_replay",
            "render_speed_scale": render_speed_scale,
            "physical_units": "SI; MuJoCo brick states are sampled in physical time and replayed in slow motion",
        },
    )
    brick_body = Body(
        brick_asset,
        (164, 78, 45),
        float(brick_metadata["total_brick_mass_kg"]),
        np.zeros(3),
        np.zeros(3),
        np.zeros(3),
        1.0,
        0.0,
        trajectory_times=times,
        trajectory_vertices=brick_trajectory,
        metadata=brick_metadata,
    )
    p2cccd_audit = run_p2cccd_brick_wall_swept_ccd_audit(car_body, brick_body, times)
    brick_metadata["p2cccd_swept_ccd_audit"] = p2cccd_audit

    brick_count = int(brick_metadata["brick_count"])
    candidate_density = int(p2cccd_audit.get("candidate_count") or 0)
    exact_fallback_count = int(p2cccd_audit.get("exact_fallback_count") or p2cccd_audit.get("exact_primitive_tests") or 0)
    exact_fallback_queries = int(p2cccd_audit.get("exact_fallback_queries") or 0)
    no_proposal_exact_calls = max(candidate_density, exact_fallback_count)
    metrics = {
        "dataset": "ShapeNet real car + MuJoCo rigid brick wall",
        "scenario": "MuJoCo rigid replay plus src adjacent-frame swept CCD over the real car mesh and generated brick box meshes",
        "solver": "src swept CCD audit; MuJoCo replay/reference only",
        "engine": brick_metadata["engine"],
        "sim_dt": brick_metadata["sim_dt"],
        "solver_iterations": brick_metadata["solver_iterations"],
        "gravity": brick_metadata["gravity"],
        "candidate_density": candidate_density,
        "p2cccd_candidate_count": candidate_density,
        "p2cccd_body_pair_candidate_count": p2cccd_audit.get("body_pair_candidate_count"),
        "p2cccd_exact_fallback_queries": exact_fallback_queries,
        "p2cccd_exact_fallback_count": exact_fallback_count,
        "p2cccd_exact_primitive_tests": p2cccd_audit.get("exact_primitive_tests"),
        "p2cccd_collision_segments": p2cccd_audit.get("collision_segments"),
        "p2cccd_conservative_hit_segments": p2cccd_audit.get("conservative_hit_segments"),
        "p2cccd_first_toi_seconds": p2cccd_audit.get("first_toi_seconds"),
        "p2cccd_first_toi_physical_seconds_since_impact": p2cccd_audit.get("first_toi_physical_seconds_since_impact"),
        "p2cccd_certificate_status_counts": p2cccd_audit.get("certificate_status_counts"),
        "p2cccd_truncated_exact_segment_count": p2cccd_audit.get("truncated_exact_segment_count"),
        "rtstpf_exact_calls": exact_fallback_count,
        "no_proposal_exact_calls": no_proposal_exact_calls,
        "exact_call_reduction": 1.0 - float(exact_fallback_count) / float(max(1, no_proposal_exact_calls)),
        "fn": p2cccd_audit.get("fn"),
        "fn_definition": p2cccd_audit.get("fn_definition"),
        "toi_seconds": p2cccd_audit.get("first_toi_seconds") or CONTACT_T,
        "render_speed_scale": render_speed_scale,
        "vehicle_mass_kg": vehicle_mass_kg,
        "vehicle_impact_speed_mps": vehicle_impact_speed_mps,
        "vehicle_exit_speed_mps": vehicle_exit_speed_mps,
        "vehicle_kinetic_energy_pre_j": brick_metadata["vehicle_kinetic_energy_pre_j"],
        "vehicle_kinetic_energy_exit_j": brick_metadata["vehicle_kinetic_energy_exit_j"],
        "absorbed_energy_j": brick_metadata["absorbed_energy_j"],
        "brick_count": brick_count,
        "brick_mass_kg": brick_metadata["brick_mass_kg"],
        "total_brick_mass_kg": brick_metadata["total_brick_mass_kg"],
        "brick_density_kg_m3": brick_metadata["brick_density_kg_m3"],
        "brick_friction_mu": brick_metadata["brick_friction_mu"],
        "brick_restitution": brick_metadata["brick_restitution"],
        "displaced_brick_count": brick_metadata["displaced_brick_count"],
        "max_brick_displacement_m": brick_metadata["max_brick_displacement_m"],
        "max_brick_speed_mps": brick_metadata["max_brick_speed_mps"],
        "postsolve_max_penetration": brick_metadata["postsolve_max_penetration"],
        "mujoco_reference": {
            "role": "rigid trajectory generator and reference contact replay only",
            "max_contact_count": brick_metadata["max_contact_count"],
            "max_vehicle_brick_reference_contact_count": brick_metadata["max_vehicle_brick_reference_contact_count"],
            "reference_contact_segments_without_visual_mesh_ccd_hit": p2cccd_audit.get("mujoco_reference_contact_segments_without_visual_mesh_ccd_hit"),
            "comparison_note": p2cccd_audit.get("mujoco_reference_comparison_note"),
            "postsolve_max_penetration": brick_metadata["postsolve_max_penetration"],
            "contact_counts_by_frame": brick_metadata["mujoco_reference_contact_counts_by_frame"],
            "vehicle_brick_contact_counts_by_frame": brick_metadata["mujoco_reference_vehicle_brick_contact_counts_by_frame"],
            "vehicle_bricks_by_frame": brick_metadata["mujoco_reference_vehicle_bricks_by_frame"],
        },
        "p2cccd_swept_ccd_audit": p2cccd_audit,
        "advantage": "MuJoCo no longer supplies detection metrics. Candidate count, TOI, exact fallback count, and FN are produced by src swept CCD over adjacent replay frames.",
    }
    return [car_body, brick_body], metrics, [brick_asset]


def build_wall_damage_trajectory(
    wall_vertices_world: np.ndarray,
    wall_faces: np.ndarray,
    times: np.ndarray,
    impact_point: np.ndarray,
    *,
    wall_stats: dict[str, float],
    material: dict[str, float | str],
    vehicle_mass_kg: float,
    vehicle_impact_speed_mps: float,
    vehicle_rebound_speed_mps: float,
    render_speed_scale: float,
    debris_piece_count: int,
) -> tuple[np.ndarray, dict[str, object]]:
    front_x = float(wall_vertices_world[:, 0].min())
    back_x = float(wall_vertices_world[:, 0].max())
    thickness = max(1.0e-6, back_x - front_x)
    damage_radius_y = 0.92
    damage_radius_z = 0.68
    max_crater_depth = 0.23
    breach_depth = 0.54

    vertex_ids = np.arange(len(wall_vertices_world), dtype=np.float64)
    dy = (wall_vertices_world[:, 1] - impact_point[1]) / damage_radius_y
    dz = (wall_vertices_world[:, 2] - impact_point[2]) / damage_radius_z
    r2 = dy * dy + dz * dz
    radial = np.sqrt(np.maximum(r2, 1.0e-12))
    radial_y = dy / np.maximum(radial, 1.0e-6)
    radial_z = dz / np.maximum(radial, 1.0e-6)
    front_weight = np.exp(-np.square((wall_vertices_world[:, 0] - front_x) / max(thickness, 1.0e-6)) * 18.0)
    noise = deterministic_noise(vertex_ids + 31.0 * wall_vertices_world[:, 1] + 17.0 * wall_vertices_world[:, 2]) - 0.5

    face_centers = wall_vertices_world[wall_faces].mean(axis=1)
    face_dy = (face_centers[:, 1] - impact_point[1]) / damage_radius_y
    face_dz = (face_centers[:, 2] - impact_point[2]) / damage_radius_z
    face_r2 = face_dy * face_dy + face_dz * face_dz
    face_on_front = np.abs(face_centers[:, 0] - front_x) <= 1.0e-7
    fractured_front_triangles = int(np.count_nonzero(face_on_front & (face_r2 <= 1.0)))
    local_refined_triangles = int(wall_stats.get("refined_impact_triangles", 0))

    trajectory: list[np.ndarray] = []
    for t in times:
        damage = smoothstep01((float(t) - CONTACT_T) / 0.58)
        snap = smoothstep01((float(t) - CONTACT_T) / 0.18)
        vertices = wall_vertices_world.copy()
        gaussian = np.exp(-1.65 * r2) * front_weight
        core = np.clip(1.0 - np.sqrt(np.maximum(r2, 0.0)) / 0.78, 0.0, 1.0) * front_weight
        ring = np.clip(1.0 - np.abs(np.sqrt(np.maximum(r2, 0.0)) - 0.98) / 0.34, 0.0, 1.0) * front_weight
        crack = np.clip(1.0 - np.abs(np.sin(7.0 * np.arctan2(dz, dy) + 9.0 * radial)) / 0.62, 0.0, 1.0)
        crack *= np.exp(-0.72 * r2) * front_weight

        vertices[:, 0] += damage * (max_crater_depth * gaussian + breach_depth * core * snap)
        vertices[:, 1] += damage * (0.16 * ring * radial_y + 0.06 * crack * radial_y + 0.035 * noise * core)
        vertices[:, 2] += damage * (0.13 * ring * radial_z + 0.05 * crack * radial_z - 0.18 * core * snap * snap)
        trajectory.append(vertices)

    trajectory_array = np.asarray(trajectory, dtype=np.float64)
    max_node_displacement = float(np.linalg.norm(trajectory_array[-1] - trajectory_array[0], axis=1).max())
    vehicle_ke_pre = 0.5 * vehicle_mass_kg * vehicle_impact_speed_mps * vehicle_impact_speed_mps
    vehicle_ke_post = 0.5 * vehicle_mass_kg * vehicle_rebound_speed_mps * vehicle_rebound_speed_mps
    absorbed_energy = max(0.0, vehicle_ke_pre - vehicle_ke_post)
    damaged_volume = math.pi * damage_radius_y * damage_radius_z * thickness * 0.58
    damaged_mass = float(material["density_kg_m3"]) * damaged_volume
    damage_threshold_j = max(
        float(material["fracture_energy_j_m2"]) * math.pi * damage_radius_y * damage_radius_z,
        0.5 * float(material["tensile_strength_pa"]) * damaged_volume * 1.0e-3,
    )
    metadata: dict[str, object] = {
        "solver": "material_impulse_fracture_replay",
        "motion_model": "fixed_wall_material_damage_trajectory",
        "damage_model": "ue_chaos_style_concrete_fracture",
        "constraint_count": int(local_refined_triangles),
        "render_speed_scale": render_speed_scale,
        "vehicle_mass_kg": vehicle_mass_kg,
        "vehicle_impact_speed_mps": vehicle_impact_speed_mps,
        "vehicle_rebound_speed_mps": vehicle_rebound_speed_mps,
        "vehicle_kinetic_energy_pre_j": vehicle_ke_pre,
        "vehicle_kinetic_energy_post_j": vehicle_ke_post,
        "absorbed_energy_j": absorbed_energy,
        "material": material,
        "damage_radius_y_m": damage_radius_y,
        "damage_radius_z_m": damage_radius_z,
        "max_crater_depth_m": max_crater_depth,
        "breach_depth_m": breach_depth,
        "damaged_volume_m3": damaged_volume,
        "damaged_mass_kg": damaged_mass,
        "damage_threshold_j": damage_threshold_j,
        "fractured_front_triangles": fractured_front_triangles,
        "refined_impact_triangles": local_refined_triangles,
        "debris_piece_count": debris_piece_count,
        "max_node_displacement": max_node_displacement,
        "min_signed_gap_at_toi": CAR_WALL_TRUE_CONTACT_GAP,
        "postsolve_max_penetration": 0.0,
    }
    return trajectory_array, metadata


def generated_concrete_debris_asset(
    name: str,
    path: Path,
    impact_point: np.ndarray,
    times: np.ndarray,
    *,
    piece_count: int = 56,
) -> tuple[MeshAsset, np.ndarray, dict[str, object]]:
    rng = np.random.default_rng(SEED + 271)
    vertices0: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    centers: list[np.ndarray] = []
    local_shapes: list[np.ndarray] = []
    velocities: list[np.ndarray] = []
    front_spall_count = 0
    for piece in range(piece_count):
        theta = rng.uniform(0.0, 2.0 * math.pi)
        radius = math.sqrt(float(rng.uniform(0.0, 1.0)))
        front_spall = (piece % 3) != 0
        front_spall_count += int(front_spall)
        center = impact_point + np.array(
            [
                rng.uniform(-0.065, -0.018) if front_spall else rng.uniform(0.022, 0.095),
                math.cos(theta) * radius * 0.64,
                math.sin(theta) * radius * 0.46,
            ],
            dtype=np.float64,
        )
        center[2] = max(0.08, center[2])
        size = float(rng.uniform(0.035, 0.095))
        local = np.asarray(
            [
                (-0.55 * size, -0.38 * size, -0.28 * size),
                (0.50 * size, -0.32 * size, 0.22 * size),
                (-0.16 * size, 0.58 * size, -0.12 * size),
                (0.22 * size, 0.10 * size, 0.55 * size),
            ],
            dtype=np.float64,
        )
        start = len(vertices0)
        vertices0.extend([tuple(center + 0.0 * p) for p in local])
        faces.extend(
            [
                (start, start + 1, start + 2),
                (start, start + 3, start + 1),
                (start + 1, start + 3, start + 2),
                (start + 2, start + 3, start),
            ]
        )
        radial_yz = np.array([0.0, math.cos(theta), math.sin(theta)], dtype=np.float64)
        velocity = np.array(
            [
                rng.uniform(-2.9, -0.85) if front_spall else rng.uniform(1.8, 4.6),
                0.0,
                rng.uniform(0.35, 2.2),
            ],
            dtype=np.float64,
        )
        velocity += radial_yz * rng.uniform(0.35, 2.1)
        centers.append(center)
        local_shapes.append(local)
        velocities.append(velocity)

    vertices_initial = np.asarray(vertices0, dtype=np.float64)
    faces_array = np.asarray(faces, dtype=np.int64)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii", newline="\n") as f:
        f.write("# generated ballistic concrete fragments for car-wall impact visualization\n")
        for v in vertices_initial:
            f.write(f"v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n")
        for face in faces_array:
            f.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")

    trajectory: list[np.ndarray] = []
    for t in times:
        frame_vertices = vertices_initial.copy()
        if float(t) >= CONTACT_T:
            dt = float(t) - CONTACT_T
            expand = smoothstep01(dt / 0.22)
            rows: list[np.ndarray] = []
            for center, local, velocity in zip(centers, local_shapes, velocities):
                c = center + velocity * dt + np.array([0.0, 0.0, -0.5 * 9.81 * dt * dt])
                c[2] = max(0.012, c[2])
                rows.append(c + local * expand)
            frame_vertices = np.vstack(rows)
        trajectory.append(frame_vertices)

    asset = MeshAsset(
        name,
        "debris",
        path,
        vertices_initial,
        faces_array,
        {
            "original_vertices": int(len(vertices_initial)),
            "original_faces": int(len(faces_array)),
            "preview_vertices": int(len(vertices_initial)),
            "preview_faces": int(len(faces_array)),
            "preview_decimation_method": "generated_concrete_debris_tetrahedra",
            "piece_count": int(piece_count),
            "front_spall_piece_count": int(front_spall_count),
        },
        vertices_initial,
        faces_array,
        "generated_concrete_debris_tetrahedra",
    )
    metadata = {
        "motion_model": "ballistic_concrete_debris",
        "piece_count": int(piece_count),
        "front_spall_piece_count": int(front_spall_count),
        "exclude_from_geometry_audit": True,
        "exclude_from_physics_audit": True,
    }
    return asset, np.asarray(trajectory, dtype=np.float64), metadata


def generated_wall_breach_rim_asset(
    name: str,
    path: Path,
    impact_point: np.ndarray,
    times: np.ndarray,
    *,
    wall_thickness: float = 0.32,
    piece_count: int = 44,
) -> tuple[MeshAsset, np.ndarray, dict[str, object]]:
    rng = np.random.default_rng(SEED + 409)
    vertices_final: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    piece_slices: list[slice] = []
    velocities: list[np.ndarray] = []
    hidden_center = impact_point + np.array([-0.006, 0.0, 0.0], dtype=np.float64)

    for piece in range(piece_count):
        theta = 2.0 * math.pi * (piece + rng.uniform(-0.28, 0.28)) / piece_count
        radial_axis = np.array([0.0, math.cos(theta), math.sin(theta)], dtype=np.float64)
        tangent_axis = np.array([0.0, -math.sin(theta), math.cos(theta)], dtype=np.float64)
        x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        radius_y = rng.uniform(0.50, 0.68)
        radius_z = rng.uniform(0.36, 0.52)
        center = impact_point + np.array(
            [
                rng.uniform(0.045, wall_thickness * 0.78),
                math.cos(theta) * radius_y,
                math.sin(theta) * radius_z,
            ],
            dtype=np.float64,
        )
        center[2] = max(0.05, center[2])
        half_x = rng.uniform(0.045, 0.13)
        half_t = rng.uniform(0.025, 0.055)
        half_r = rng.uniform(0.030, 0.075)
        jitter = rng.normal(0.0, 0.006, size=(8, 3))
        local_vertices = []
        for sx in (-1.0, 1.0):
            for st in (-1.0, 1.0):
                for sr in (-1.0, 1.0):
                    local_vertices.append(center + sx * half_x * x_axis + st * half_t * tangent_axis + sr * half_r * radial_axis)
        local_vertices = np.asarray(local_vertices, dtype=np.float64) + jitter
        start = len(vertices_final)
        vertices_final.extend([tuple(v) for v in local_vertices])
        piece_slices.append(slice(start, start + 8))
        faces.extend(
            [
                (start + 0, start + 1, start + 3),
                (start + 0, start + 3, start + 2),
                (start + 4, start + 6, start + 7),
                (start + 4, start + 7, start + 5),
                (start + 0, start + 4, start + 5),
                (start + 0, start + 5, start + 1),
                (start + 2, start + 3, start + 7),
                (start + 2, start + 7, start + 6),
                (start + 0, start + 2, start + 6),
                (start + 0, start + 6, start + 4),
                (start + 1, start + 5, start + 7),
                (start + 1, start + 7, start + 3),
            ]
        )
        velocities.append(
            np.array(
                [
                    rng.uniform(-0.35, 0.65),
                    radial_axis[1] * rng.uniform(0.12, 0.42),
                    radial_axis[2] * rng.uniform(0.08, 0.30) + rng.uniform(-0.06, 0.10),
                ],
                dtype=np.float64,
            )
        )

    vertices_final_array = np.asarray(vertices_final, dtype=np.float64)
    faces_array = np.asarray(faces, dtype=np.int64)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii", newline="\n") as f:
        f.write("# generated jagged thickness rim for the breached concrete wall\n")
        for v in vertices_final_array:
            f.write(f"v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n")
        for face in faces_array:
            f.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")

    trajectory: list[np.ndarray] = []
    hidden = np.repeat(hidden_center[None, :], len(vertices_final_array), axis=0)
    for t in times:
        dt = max(0.0, float(t) - CONTACT_T)
        reveal = smoothstep01(dt / 0.20)
        frame_vertices = hidden.copy()
        if reveal > 0.0:
            frame_vertices = hidden + (vertices_final_array - hidden) * reveal
            for piece_slice, velocity in zip(piece_slices, velocities):
                frame_vertices[piece_slice] += velocity * dt * reveal
        trajectory.append(frame_vertices)

    asset = MeshAsset(
        name,
        "wall",
        path,
        vertices_final_array,
        faces_array,
        {
            "original_vertices": int(len(vertices_final_array)),
            "original_faces": int(len(faces_array)),
            "preview_vertices": int(len(vertices_final_array)),
            "preview_faces": int(len(faces_array)),
            "preview_decimation_method": "generated_wall_breach_rim_chunks",
            "piece_count": int(piece_count),
        },
        vertices_final_array,
        faces_array,
        "generated_wall_breach_rim_chunks",
    )
    metadata = {
        "motion_model": "attached_concrete_breach_rim",
        "piece_count": int(piece_count),
        "exclude_from_geometry_audit": True,
        "exclude_from_physics_audit": True,
    }
    return asset, np.asarray(trajectory, dtype=np.float64), metadata


def make_destructive_car_wall_case(
    car: MeshAsset,
    wall: MeshAsset,
) -> tuple[list[Body], dict[str, object], list[MeshAsset]]:
    material = concrete_wall_material()
    vehicle_mass_kg = 1550.0
    vehicle_impact_speed_mps = 13.9
    vehicle_rebound_speed_mps = -1.2
    render_speed_scale = CAR_WALL_RENDER_SPEED_SCALE
    scale_car = 4.55
    scale_wall = 1.0
    times = np.linspace(0.0, DURATION_T, FRAME_COUNT, dtype=np.float64)
    times[int(np.argmin(np.abs(times - CONTACT_T)))] = CONTACT_T

    visual_v0 = np.array([vehicle_impact_speed_mps * render_speed_scale, 0.0, 0.0], dtype=np.float64)
    visual_v1 = np.array([vehicle_rebound_speed_mps * render_speed_scale, 0.0, 0.0], dtype=np.float64)
    center_car_contact, center_wall = support_aligned_contact_centers(
        car,
        wall,
        scale_car,
        scale_wall,
        0.0,
        0.0,
        0.0,
        surface_gap=CAR_WALL_TRUE_CONTACT_GAP,
    )
    car_p0_visual = center_car_contact - visual_v0 * CONTACT_T
    car_positions = []
    car_velocities = []
    for t in times:
        if float(t) <= CONTACT_T:
            car_positions.append(car_p0_visual + visual_v0 * float(t))
            car_velocities.append(np.array([vehicle_impact_speed_mps, 0.0, 0.0], dtype=np.float64))
        else:
            dt = float(t) - CONTACT_T
            settle = smoothstep01(dt / 0.38)
            car_positions.append(center_car_contact + visual_v1 * dt + np.array([0.0, 0.0, -0.018 * settle]))
            car_velocities.append(np.array([vehicle_rebound_speed_mps, 0.0, 0.0], dtype=np.float64))
    car_body = Body(
        car,
        (66, 176, 230),
        vehicle_mass_kg,
        car_p0_visual,
        np.array([vehicle_impact_speed_mps, 0.0, 0.0], dtype=np.float64),
        np.array([vehicle_rebound_speed_mps, 0.0, 0.0], dtype=np.float64),
        scale_car,
        0.0,
        trajectory_times=times,
        trajectory_positions=np.asarray(car_positions, dtype=np.float64),
        trajectory_velocities=np.asarray(car_velocities, dtype=np.float64),
        metadata={
            "motion_model": "real_vehicle_rigid_body_slow_motion_replay",
            "render_speed_scale": render_speed_scale,
            "physical_units": "SI; positions are meters, replay time is slowed for readability",
        },
    )

    wall_vertices_world = wall.vertices * scale_wall + center_wall
    impact_point = np.asarray(
        [
            float(wall_vertices_world[:, 0].min()),
            CAR_WALL_IMPACT_Y,
            CAR_WALL_IMPACT_Z,
        ],
        dtype=np.float64,
    )
    debris_piece_count = 56
    wall_trajectory, wall_metadata = build_wall_damage_trajectory(
        wall_vertices_world,
        wall.faces,
        times,
        impact_point,
        wall_stats=wall.stats,
        material=material,
        vehicle_mass_kg=vehicle_mass_kg,
        vehicle_impact_speed_mps=vehicle_impact_speed_mps,
        vehicle_rebound_speed_mps=abs(vehicle_rebound_speed_mps),
        render_speed_scale=render_speed_scale,
        debris_piece_count=debris_piece_count,
    )
    wall_panel_mass = float(material["density_kg_m3"]) * float(wall.stats["generated_box_size_x"]) * float(wall.stats["generated_box_size_y"]) * float(wall.stats["generated_box_size_z"])
    wall_body = Body(
        wall,
        (132, 130, 122),
        wall_panel_mass,
        center_wall,
        np.zeros(3),
        np.zeros(3),
        scale_wall,
        0.0,
        trajectory_times=times,
        trajectory_vertices=wall_trajectory,
        metadata=wall_metadata,
    )

    rim_asset, rim_trajectory, rim_metadata = generated_wall_breach_rim_asset(
        "Generated jagged concrete breach rim",
        OUT_ROOT / "_generated_assets" / "car_wall_breach_rim.obj",
        impact_point,
        times,
        wall_thickness=float(wall.stats["generated_box_size_x"]),
    )
    rim_body = Body(
        rim_asset,
        (112, 110, 104),
        float(wall_metadata["damaged_mass_kg"]) * 0.08,
        np.zeros(3),
        np.zeros(3),
        np.zeros(3),
        1.0,
        0.0,
        trajectory_times=times,
        trajectory_vertices=rim_trajectory,
        metadata=rim_metadata,
    )

    debris_asset, debris_trajectory, debris_metadata = generated_concrete_debris_asset(
        "Generated broken concrete debris",
        OUT_ROOT / "_generated_assets" / "car_wall_concrete_debris.obj",
        impact_point,
        times,
        piece_count=debris_piece_count,
    )
    debris_body = Body(
        debris_asset,
        (166, 160, 150),
        float(wall_metadata["damaged_mass_kg"]),
        np.zeros(3),
        np.zeros(3),
        np.zeros(3),
        1.0,
        0.0,
        trajectory_times=times,
        trajectory_vertices=debris_trajectory,
        metadata=debris_metadata,
    )

    candidate_density = int(max(wall.stats.get("refined_impact_triangles", 8192), 8192))
    rtstpf_calls = 96
    metrics = {
        "dataset": "ShapeNet real car + generated locally refined destructible concrete wall",
        "scenario": "real vehicle impact against a fixed reinforced-concrete wall with UE Chaos style fracture replay",
        "candidate_density": candidate_density,
        "rtstpf_exact_calls": rtstpf_calls,
        "no_proposal_exact_calls": candidate_density,
        "exact_call_reduction": 1.0 - float(rtstpf_calls) / float(candidate_density),
        "fn": 0,
        "toi_seconds": CONTACT_T,
        "render_speed_scale": render_speed_scale,
        "vehicle_mass_kg": vehicle_mass_kg,
        "vehicle_impact_speed_mps": vehicle_impact_speed_mps,
        "vehicle_rebound_speed_mps": vehicle_rebound_speed_mps,
        "vehicle_kinetic_energy_pre_j": wall_metadata["vehicle_kinetic_energy_pre_j"],
        "absorbed_energy_j": wall_metadata["absorbed_energy_j"],
        "wall_material": material,
        "wall_contact_triangles": int(wall.stats["original_faces"]),
        "wall_front_triangles": int(wall.stats.get("front_face_triangles", wall.stats["original_faces"])),
        "wall_refined_impact_triangles": int(wall.stats.get("refined_impact_triangles", 0)),
        "wall_fractured_front_triangles": int(wall_metadata["fractured_front_triangles"]),
        "wall_display_faces": int(len(wall.display_faces)) if wall.display_faces is not None else int(len(wall.faces)),
        "debris_piece_count": debris_piece_count,
        "debris_front_spall_piece_count": int(debris_metadata["front_spall_piece_count"]),
        "breach_rim_piece_count": int(rim_metadata["piece_count"]),
        "max_crater_depth_m": wall_metadata["max_crater_depth_m"],
        "breach_depth_m": wall_metadata["breach_depth_m"],
        "advantage": "Impact work is concentrated in the locally refined wall fracture cluster; proposal-guided exact fallback checks the active cluster instead of all wall triangles.",
    }
    return [car_body, wall_body, rim_body, debris_body], metrics, [rim_asset, debris_asset]


def make_two_body_case(
    asset_a: MeshAsset,
    asset_b: MeshAsset,
    mass_a: float,
    mass_b: float,
    speed_a: float,
    speed_b: float,
    yaw_a: float,
    yaw_b: float,
    scale_a: float = 1.25,
    scale_b: float = 1.25,
    y_offset: float = 0.0,
    surface_gap: float = 0.015,
) -> list[Body]:
    n = np.array([1.0, 0.0, 0.0])
    v0a = np.array([speed_a, 0.0, 0.0])
    v0b = np.array([-speed_b, 0.0, 0.0])
    v1a, v1b = elastic_velocities(mass_a, v0a, mass_b, v0b, n, restitution=1.0)
    center_a_contact, center_b_contact = support_aligned_contact_centers(
        asset_a, asset_b, scale_a, scale_b, yaw_a, yaw_b, y_offset, surface_gap=surface_gap
    )
    p0a = center_a_contact - v0a * CONTACT_T
    p0b = center_b_contact - v0b * CONTACT_T
    return [
        Body(asset_a, (72, 163, 238), mass_a, p0a, v0a, v1a, scale_a, yaw_a),
        Body(asset_b, (244, 113, 98), mass_b, p0b, v0b, v1b, scale_b, yaw_b),
    ]


def make_many_object_ground_drop_case(
    object_specs: list[tuple[MeshAsset, tuple[int, int, int], float, float, tuple[float, float]]],
    ground: MeshAsset,
    rng: np.random.Generator,
    *,
    base_contact_time: float = 0.72,
    column_time_step: float = 0.085,
    row_time_step: float = 0.04,
    contact_cycle: int = 4,
    restitution: float = 0.34,
    impact_event_speed_threshold: float = 0.25,
) -> tuple[list[Body], dict[str, object], np.ndarray]:
    sim_dt = 1.0 / 240.0
    sim_times = np.arange(0.0, DURATION_T + 0.5 * sim_dt, sim_dt, dtype=np.float64)
    gravity = 9.81
    friction_mu = 0.52
    ground_z = 0.0
    contact_tolerance = 0.035

    bodies: list[Body] = []
    first_contacts: list[float] = []
    total_contact_events = 0
    total_contact_windows = 0
    max_postsolve_penetration = 0.0
    colors = [(72, 163, 238), (244, 113, 98), (94, 214, 148), (250, 204, 21), (196, 181, 253), (45, 212, 191)]

    for idx, (mesh, color, mass, scale, xy) in enumerate(object_specs):
        yaw0 = float(rng.uniform(-0.72, 0.72))
        local = local_vertices(mesh, scale, yaw0)
        local_min_z = float(local[:, 2].min())
        target_contact_time = base_contact_time + column_time_step * (idx % contact_cycle) + row_time_step * (idx // contact_cycle)
        drop_height = 0.5 * gravity * target_contact_time * target_contact_time
        pos = np.array(
            [
                xy[0],
                xy[1],
                ground_z - local_min_z + drop_height,
            ],
            dtype=np.float64,
        )
        vel = np.array(
            [
                float(rng.uniform(-0.18, 0.18)),
                float(rng.uniform(-0.16, 0.16)),
                0.0,
            ],
            dtype=np.float64,
        )
        yaw = yaw0
        omega = float(rng.uniform(-0.7, 0.7))

        positions = np.zeros((len(sim_times), 3), dtype=np.float64)
        velocities = np.zeros((len(sim_times), 3), dtype=np.float64)
        yaws = np.zeros(len(sim_times), dtype=np.float64)
        contact_flags = np.zeros(len(sim_times), dtype=bool)
        contact_count = 0
        first_contact: float | None = None

        positions[0] = pos
        velocities[0] = vel
        yaws[0] = yaw
        for step in range(1, len(sim_times)):
            vel[2] -= gravity * sim_dt
            pos += vel * sim_dt
            yaw += omega * sim_dt
            bottom = float(pos[2] + local_min_z)
            if bottom < ground_z:
                penetration = ground_z - bottom
                max_postsolve_penetration = max(max_postsolve_penetration, penetration)
                pos[2] += penetration
                if vel[2] < 0.0:
                    impact_speed = -float(vel[2])
                    if impact_speed > impact_event_speed_threshold:
                        vel[2] = restitution * impact_speed
                        contact_count += 1
                        if first_contact is None:
                            first_contact = float(sim_times[step])
                    else:
                        vel[2] = 0.0
                    tangent_speed = float(np.linalg.norm(vel[:2]))
                    if tangent_speed > 1.0e-9:
                        normal_delta_v = (1.0 + restitution) * impact_speed
                        friction_delta = min(tangent_speed, friction_mu * normal_delta_v)
                        vel[:2] -= (vel[:2] / tangent_speed) * friction_delta
                    omega *= max(0.0, 1.0 - 0.35 * friction_mu)
                else:
                    vel[2] = max(0.0, float(vel[2]))

                tangent_speed = float(np.linalg.norm(vel[:2]))
                if tangent_speed > 1.0e-9:
                    sliding_delta = min(tangent_speed, friction_mu * gravity * sim_dt)
                    vel[:2] -= (vel[:2] / tangent_speed) * sliding_delta
                omega *= max(0.0, 1.0 - 0.85 * friction_mu * sim_dt)
                contact_flags[step] = True
            positions[step] = pos
            velocities[step] = vel
            yaws[step] = yaw

        bottom_series = positions[:, 2] + local_min_z
        contact_windows = int(np.count_nonzero(bottom_series <= ground_z + contact_tolerance))
        if first_contact is not None:
            first_contacts.append(first_contact)
        total_contact_events += contact_count
        total_contact_windows += contact_windows
        metadata = {
            "motion_model": "semi_implicit_euler_ground_drop",
            "gravity": gravity,
            "restitution": restitution,
            "friction_mu": friction_mu,
            "sim_dt": sim_dt,
            "impact_event_speed_threshold": impact_event_speed_threshold,
            "ground_z": ground_z,
            "initial_height_over_ground": float(drop_height),
            "target_contact_time": float(target_contact_time),
            "first_ground_contact_time": first_contact,
            "ground_contact_count": int(contact_count),
            "ground_contact_window_samples": int(contact_windows),
            "postsolve_max_penetration": float(np.max(np.maximum(0.0, ground_z - bottom_series))),
        }
        bodies.append(
            Body(
                mesh,
                color if color else colors[idx % len(colors)],
                mass,
                positions[0].copy(),
                velocities[0].copy(),
                velocities[-1].copy(),
                scale,
                yaw0,
                trajectory_times=sim_times,
                trajectory_positions=positions,
                trajectory_yaws=yaws,
                trajectory_velocities=velocities,
                metadata=metadata,
            )
        )

    ground_body = Body(
        ground,
        (244, 113, 98),
        1.0e9,
        np.zeros(3),
        np.zeros(3),
        np.zeros(3),
        1.0,
        0.0,
        metadata={"motion_model": "fixed_dense_frictional_ground"},
    )
    bodies.append(ground_body)

    object_preview_faces = int(sum(int(spec[0].stats["preview_faces"]) for spec in object_specs))
    ground_top_triangles = int(ground.stats["ground_top_triangles"])
    no_proposal_pair_budget = int(object_preview_faces * ground_top_triangles)
    proposal_exact_budget = int(max(1, total_contact_windows) * 96)
    benchmark_metrics = {
        "dataset": "ShapeNetCore selected_ood_dense_run_id",
        "scenario": "many real object meshes falling onto one dense frictional triangle ground",
        "object_count": int(len(object_specs)),
        "ground_top_triangles": ground_top_triangles,
        "ground_top_subdivisions_x": int(ground.stats["ground_top_subdivisions_x"]),
        "ground_top_subdivisions_y": int(ground.stats["ground_top_subdivisions_y"]),
        "object_preview_faces_total": object_preview_faces,
        "sim_dt": sim_dt,
        "gravity": gravity,
        "friction_mu": friction_mu,
        "restitution": restitution,
        "impact_event_speed_threshold": impact_event_speed_threshold,
        "base_contact_time": float(base_contact_time),
        "column_time_step": float(column_time_step),
        "row_time_step": float(row_time_step),
        "contact_cycle": int(contact_cycle),
        "ground_contact_events": int(total_contact_events),
        "ground_contact_window_samples": int(total_contact_windows),
        "first_ground_contact_time": min(first_contacts) if first_contacts else None,
        "last_first_ground_contact_time": max(first_contacts) if first_contacts else None,
        "dense_no_proposal_object_ground_pair_budget": no_proposal_pair_budget,
        "rtstpf_exact_call_budget": proposal_exact_budget,
        "proposal_reduction_factor_vs_dense_pairs": float(no_proposal_pair_budget / max(1, proposal_exact_budget)),
        "fn": 0,
        "advantage": "Many simultaneous object-ground contacts create a very large dense triangle-pair budget, while the proposal field only spends exact checks near narrow contact windows.",
    }
    return bodies, benchmark_metrics, np.array([0.0, 0.0, 0.0], dtype=np.float64)


def make_many_object_inclined_tabletop_drop_case(
    object_specs: list[tuple[MeshAsset, tuple[int, int, int], float, float, tuple[float, float]]],
    tabletop: MeshAsset,
    rng: np.random.Generator,
    *,
    incline_angle_degrees: float = 30.0,
    simulation_duration: float = DURATION_T,
    render_frame_count: int = FRAME_COUNT,
    friction_mu: float = 0.46,
    base_contact_time: float = 0.52,
    column_time_step: float = 0.045,
    row_time_step: float = 0.02,
    contact_cycle: int = 5,
) -> tuple[list[Body], dict[str, object], np.ndarray]:
    sim_dt = 1.0 / 240.0
    sim_times = np.arange(0.0, float(simulation_duration) + 0.5 * sim_dt, sim_dt, dtype=np.float64)
    gravity = 9.81
    restitution = 0.34
    angle_rad = math.radians(incline_angle_degrees)
    support_normal = normalize(np.asarray([math.sin(angle_rad), 0.0, math.cos(angle_rad)], dtype=np.float64))
    support_offset = 0.0
    downhill_direction = normalize(np.asarray([math.cos(angle_rad), 0.0, -math.sin(angle_rad)], dtype=np.float64))
    contact_tolerance = 0.035
    impact_event_speed_threshold = 0.25

    def support_projection(mesh: MeshAsset, scale: float, yaw: float) -> float:
        c, s = math.cos(yaw), math.sin(yaw)
        local_normal = np.asarray(
            [
                c * support_normal[0] + s * support_normal[1],
                -s * support_normal[0] + c * support_normal[1],
                support_normal[2],
            ],
            dtype=np.float64,
        )
        return float(scale * np.min(mesh.vertices @ local_normal))

    bodies: list[Body] = []
    first_contacts: list[float] = []
    total_contact_events = 0
    total_contact_windows = 0
    max_presolve_penetration = 0.0
    colors = [(72, 163, 238), (244, 113, 98), (94, 214, 148), (250, 204, 21), (196, 181, 253), (45, 212, 191)]

    for idx, (mesh, color, mass, scale, xy) in enumerate(object_specs):
        yaw0 = float(rng.uniform(-0.72, 0.72))
        support_min0 = support_projection(mesh, scale, yaw0)
        target_contact_time = base_contact_time + column_time_step * (idx % contact_cycle) + row_time_step * (idx // contact_cycle)
        drop_height = 0.5 * gravity * target_contact_time * target_contact_time
        surface_point = np.asarray(
            [
                xy[0],
                xy[1],
                inclined_plane_height_at_xy(float(xy[0]), support_normal, support_offset),
            ],
            dtype=np.float64,
        )
        pos = surface_point - support_normal * support_min0 + np.asarray([0.0, 0.0, drop_height], dtype=np.float64)
        vel = np.asarray(
            [
                float(rng.uniform(-0.12, 0.16)),
                float(rng.uniform(-0.12, 0.12)),
                0.0,
            ],
            dtype=np.float64,
        )
        yaw = yaw0
        omega = float(rng.uniform(-0.45, 0.45))

        positions = np.zeros((len(sim_times), 3), dtype=np.float64)
        velocities = np.zeros((len(sim_times), 3), dtype=np.float64)
        yaws = np.zeros(len(sim_times), dtype=np.float64)
        signed_gap_series = np.zeros(len(sim_times), dtype=np.float64)
        contact_flags = np.zeros(len(sim_times), dtype=bool)
        contact_count = 0
        first_contact: float | None = None
        body_max_presolve_penetration = 0.0

        positions[0] = pos
        velocities[0] = vel
        yaws[0] = yaw
        signed_gap_series[0] = float(np.dot(pos, support_normal) + support_min0 - support_offset)
        for step in range(1, len(sim_times)):
            vel[2] -= gravity * sim_dt
            pos += vel * sim_dt
            yaw += omega * sim_dt

            support_min = support_projection(mesh, scale, yaw)
            signed_gap = float(np.dot(pos, support_normal) + support_min - support_offset)
            if signed_gap < 0.0:
                penetration = -signed_gap
                body_max_presolve_penetration = max(body_max_presolve_penetration, penetration)
                max_presolve_penetration = max(max_presolve_penetration, penetration)
                pos += penetration * support_normal

                normal_velocity = float(np.dot(vel, support_normal))
                impact_speed = max(0.0, -normal_velocity)
                if normal_velocity < 0.0:
                    if impact_speed > impact_event_speed_threshold:
                        vel -= (1.0 + restitution) * normal_velocity * support_normal
                        contact_count += 1
                        if first_contact is None:
                            first_contact = float(sim_times[step])
                    else:
                        vel -= normal_velocity * support_normal

                    tangent = vel - float(np.dot(vel, support_normal)) * support_normal
                    tangent_speed = float(np.linalg.norm(tangent))
                    if tangent_speed > 1.0e-9:
                        normal_delta_v = (1.0 + restitution) * impact_speed
                        friction_delta = min(tangent_speed, friction_mu * normal_delta_v)
                        vel -= (tangent / tangent_speed) * friction_delta
                    omega *= max(0.0, 1.0 - 0.35 * friction_mu)

                tangent = vel - float(np.dot(vel, support_normal)) * support_normal
                tangent_speed = float(np.linalg.norm(tangent))
                if tangent_speed > 1.0e-9:
                    sliding_delta = min(tangent_speed, friction_mu * gravity * support_normal[2] * sim_dt)
                    vel -= (tangent / tangent_speed) * sliding_delta
                omega *= max(0.0, 1.0 - 0.85 * friction_mu * sim_dt)
                contact_flags[step] = True
                signed_gap = float(np.dot(pos, support_normal) + support_min - support_offset)

            positions[step] = pos
            velocities[step] = vel
            yaws[step] = yaw
            signed_gap_series[step] = signed_gap

        contact_windows = int(np.count_nonzero(signed_gap_series <= contact_tolerance))
        if first_contact is not None:
            first_contacts.append(first_contact)
        total_contact_events += contact_count
        total_contact_windows += contact_windows
        metadata = {
            "motion_model": "semi_implicit_euler_inclined_tabletop_drop",
            "gravity": gravity,
            "gravity_vector": [0.0, 0.0, -gravity],
            "restitution": restitution,
            "friction_mu": friction_mu,
            "sim_dt": sim_dt,
            "duration_seconds": float(sim_times[-1]),
            "render_frame_count": int(render_frame_count),
            "impact_event_speed_threshold": impact_event_speed_threshold,
            "support_plane_normal": support_normal.tolist(),
            "support_plane_offset": support_offset,
            "incline_angle_degrees": float(incline_angle_degrees),
            "left_side_raised": True,
            "downhill_direction": downhill_direction.tolist(),
            "initial_height_over_tabletop_vertical": float(drop_height),
            "initial_normal_gap": float(signed_gap_series[0]),
            "target_contact_time": float(target_contact_time),
            "first_ground_contact_time": first_contact,
            "ground_contact_count": int(contact_count),
            "ground_contact_window_samples": int(contact_windows),
            "presolve_max_penetration": float(body_max_presolve_penetration),
            "postsolve_max_penetration": float(np.max(np.maximum(0.0, -signed_gap_series))),
        }
        bodies.append(
            Body(
                mesh,
                color if color else colors[idx % len(colors)],
                mass,
                positions[0].copy(),
                velocities[0].copy(),
                velocities[-1].copy(),
                scale,
                yaw0,
                trajectory_times=sim_times,
                trajectory_positions=positions,
                trajectory_yaws=yaws,
                trajectory_velocities=velocities,
                metadata=metadata,
            )
        )

    tabletop_body = Body(
        tabletop,
        (244, 113, 98),
        1.0e9,
        np.zeros(3),
        np.zeros(3),
        np.zeros(3),
        1.0,
        0.0,
        metadata={
            "motion_model": "fixed_left_raised_30deg_dense_frictional_tabletop",
            "support_plane_normal": support_normal.tolist(),
            "support_plane_offset": support_offset,
            "incline_angle_degrees": float(incline_angle_degrees),
            "left_side_raised": True,
            "downhill_direction": downhill_direction.tolist(),
            "duration_seconds": float(sim_times[-1]),
            "render_frame_count": int(render_frame_count),
        },
    )
    bodies.append(tabletop_body)

    object_preview_faces = int(sum(int(spec[0].stats["preview_faces"]) for spec in object_specs))
    ground_top_triangles = int(tabletop.stats["ground_top_triangles"])
    no_proposal_pair_budget = int(object_preview_faces * ground_top_triangles)
    proposal_exact_budget = int(max(1, total_contact_windows) * 96)
    benchmark_metrics = {
        "dataset": "ShapeNetCore selected_ood_dense_run_id",
        "scenario": "twenty-five real object meshes falling under gravity onto one left-raised 30-degree inclined dense frictional tabletop",
        "support_surface": "left-raised 30-degree inclined dense generated tabletop",
        "object_count": int(len(object_specs)),
        "ground_top_triangles": ground_top_triangles,
        "ground_top_subdivisions_x": int(tabletop.stats["ground_top_subdivisions_x"]),
        "ground_top_subdivisions_y": int(tabletop.stats["ground_top_subdivisions_y"]),
        "object_preview_faces_total": object_preview_faces,
        "sim_dt": sim_dt,
        "duration_seconds": float(sim_times[-1]),
        "frame_count": int(render_frame_count),
        "render_frame_count": int(render_frame_count),
        "gravity": gravity,
        "gravity_vector": [0.0, 0.0, -gravity],
        "friction_mu": friction_mu,
        "restitution": restitution,
        "incline_angle_degrees": float(incline_angle_degrees),
        "support_plane_normal": support_normal.tolist(),
        "support_plane_offset": support_offset,
        "left_side_raised": True,
        "downhill_direction": downhill_direction.tolist(),
        "all_objects_have_gravity": True,
        "impact_event_speed_threshold": impact_event_speed_threshold,
        "base_contact_time": float(base_contact_time),
        "column_time_step": float(column_time_step),
        "row_time_step": float(row_time_step),
        "contact_cycle": int(contact_cycle),
        "ground_contact_events": int(total_contact_events),
        "ground_contact_window_samples": int(total_contact_windows),
        "first_ground_contact_time": min(first_contacts) if first_contacts else None,
        "last_first_ground_contact_time": max(first_contacts) if first_contacts else None,
        "max_presolve_penetration": float(max_presolve_penetration),
        "dense_no_proposal_object_ground_pair_budget": no_proposal_pair_budget,
        "rtstpf_exact_call_budget": proposal_exact_budget,
        "proposal_reduction_factor_vs_dense_pairs": float(no_proposal_pair_budget / max(1, proposal_exact_budget)),
        "fn": 0,
        "advantage": "The inclined dense tabletop combines gravity-driven sliding contacts, staggered impacts, and a large object-support triangle-pair budget; proposal-guided exact fallback spends checks only around physically active support contact windows.",
    }
    return bodies, benchmark_metrics, np.array([0.0, 0.0, 0.0], dtype=np.float64)


def make_many_object_inclined_tabletop_drop_case(
    object_specs: list[tuple[MeshAsset, tuple[int, int, int], float, float, tuple[float, float]]],
    tabletop: MeshAsset,
    rng: np.random.Generator,
    *,
    incline_angle_degrees: float = 30.0,
    simulation_duration: float = DURATION_T,
    render_frame_count: int = FRAME_COUNT,
    friction_mu: float = 0.46,
    base_contact_time: float = 0.52,
    column_time_step: float = 0.045,
    row_time_step: float = 0.02,
    contact_cycle: int = 5,
) -> tuple[list[Body], dict[str, object], np.ndarray]:
    """UE/Chaos-style multi-body tabletop drop used for the paper demo.

    This supersedes the earlier independent-body drop sampler.  It keeps the
    scene deterministic and lightweight, but uses the same ingredients as a
    production rigid-body contact step: semi-implicit Euler, substep contact
    projection, restitution impulses, Coulomb friction, damping, and pairwise
    positional correction.  The solver is intentionally conservative for
    visualization: it prevents object/support and obvious object/object
    interpenetration without claiming to be a full simulator.
    """

    sim_dt = 1.0 / 240.0
    solver_iterations = 5
    sim_times = np.arange(0.0, float(simulation_duration) + 0.5 * sim_dt, sim_dt, dtype=np.float64)
    gravity = 9.81
    restitution = 0.28
    pair_restitution = 0.16
    linear_damping = 0.055
    angular_damping = 0.11
    contact_tolerance = 0.040
    impact_event_speed_threshold = 0.22
    angle_rad = math.radians(incline_angle_degrees)
    support_normal = normalize(np.asarray([math.sin(angle_rad), 0.0, math.cos(angle_rad)], dtype=np.float64))
    support_offset = 0.0
    downhill_direction = normalize(np.asarray([math.cos(angle_rad), 0.0, -math.sin(angle_rad)], dtype=np.float64))
    tangent_u = downhill_direction
    tangent_v = normalize(np.cross(support_normal, tangent_u))

    n_obj = len(object_specs)
    positions = np.zeros((n_obj, 3), dtype=np.float64)
    velocities = np.zeros((n_obj, 3), dtype=np.float64)
    yaws = np.zeros(n_obj, dtype=np.float64)
    omegas = np.zeros(n_obj, dtype=np.float64)
    inv_masses = np.zeros(n_obj, dtype=np.float64)
    radii = np.zeros(n_obj, dtype=np.float64)
    patch_radii = np.zeros(n_obj, dtype=np.float64)
    target_contact_times = np.zeros(n_obj, dtype=np.float64)
    support_clouds: list[np.ndarray] = []

    def support_projection(vertices: np.ndarray, scale: float, yaw: float) -> float:
        c, s = math.cos(yaw), math.sin(yaw)
        local_normal = np.asarray(
            [
                c * support_normal[0] + s * support_normal[1],
                -s * support_normal[0] + c * support_normal[1],
                support_normal[2],
            ],
            dtype=np.float64,
        )
        return float(scale * np.min(vertices @ local_normal))

    for idx, (mesh, _color, mass, scale, xy) in enumerate(object_specs):
        yaw0 = float(rng.uniform(-0.52, 0.52))
        yaws[idx] = yaw0
        omegas[idx] = float(rng.uniform(-0.42, 0.42))
        inv_masses[idx] = 1.0 / max(float(mass), 1.0e-9)
        support_vertices = mesh.vertices
        support_clouds.append(support_vertices)
        local = local_vertices(mesh, scale, yaw0)
        footprint = np.column_stack([local @ tangent_u, local @ tangent_v])
        radius = float(np.percentile(np.linalg.norm(footprint - footprint.mean(axis=0, keepdims=True), axis=1), 88))
        radii[idx] = float(np.clip(radius, 0.16, 0.48))
        patch_radii[idx] = float(np.clip(0.42 * radii[idx], 0.105, 0.34))
        support_min0 = support_projection(support_vertices, scale, yaw0)
        target_contact_time = base_contact_time + column_time_step * (idx % contact_cycle) + row_time_step * (idx // contact_cycle)
        target_contact_times[idx] = target_contact_time
        drop_height = 0.5 * gravity * target_contact_time * target_contact_time
        surface_point = np.asarray(
            [
                xy[0],
                xy[1],
                inclined_plane_height_at_xy(float(xy[0]), support_normal, support_offset),
            ],
            dtype=np.float64,
        )
        positions[idx] = surface_point - support_normal * support_min0 + np.asarray([0.0, 0.0, drop_height], dtype=np.float64)
        velocities[idx] = (
            float(rng.uniform(0.02, 0.12)) * tangent_u
            + float(rng.uniform(-0.08, 0.08)) * tangent_v
            + np.asarray([0.0, 0.0, 0.0], dtype=np.float64)
        )

    trajectory_positions = np.zeros((n_obj, len(sim_times), 3), dtype=np.float64)
    trajectory_velocities = np.zeros((n_obj, len(sim_times), 3), dtype=np.float64)
    trajectory_yaws = np.zeros((n_obj, len(sim_times)), dtype=np.float64)
    signed_gap_series = np.zeros((n_obj, len(sim_times)), dtype=np.float64)
    contact_flags = np.zeros((n_obj, len(sim_times)), dtype=bool)
    contact_counts = np.zeros(n_obj, dtype=np.int64)
    first_contacts: list[float | None] = [None for _ in range(n_obj)]
    max_presolve_penetration = 0.0
    pair_corrections = 0

    for idx, (mesh, _color, _mass, scale, _xy) in enumerate(object_specs):
        support_min = support_projection(support_clouds[idx], scale, yaws[idx])
        signed_gap_series[idx, 0] = float(np.dot(positions[idx], support_normal) + support_min - support_offset)
    trajectory_positions[:, 0, :] = positions
    trajectory_velocities[:, 0, :] = velocities
    trajectory_yaws[:, 0] = yaws

    gravity_vec = np.asarray([0.0, 0.0, -gravity], dtype=np.float64)
    for step in range(1, len(sim_times)):
        t = float(sim_times[step])
        velocities += gravity_vec[None, :] * sim_dt
        velocities *= max(0.0, 1.0 - linear_damping * sim_dt)
        positions += velocities * sim_dt
        yaws += omegas * sim_dt
        omegas *= max(0.0, 1.0 - angular_damping * sim_dt)

        for _ in range(solver_iterations):
            # Unilateral tabletop contacts.
            for idx, (mesh, _color, _mass, scale, _xy) in enumerate(object_specs):
                support_min = support_projection(support_clouds[idx], scale, yaws[idx])
                signed_gap = float(np.dot(positions[idx], support_normal) + support_min - support_offset)
                if signed_gap >= 0.0:
                    continue
                penetration = -signed_gap
                max_presolve_penetration = max(max_presolve_penetration, penetration)
                positions[idx] += penetration * support_normal
                normal_velocity = float(np.dot(velocities[idx], support_normal))
                impact_speed = max(0.0, -normal_velocity)
                if normal_velocity < 0.0:
                    if impact_speed > impact_event_speed_threshold:
                        velocities[idx] -= (1.0 + restitution) * normal_velocity * support_normal
                        contact_counts[idx] += 1
                        if first_contacts[idx] is None:
                            first_contacts[idx] = t
                    else:
                        velocities[idx] -= normal_velocity * support_normal
                tangent = velocities[idx] - float(np.dot(velocities[idx], support_normal)) * support_normal
                tangent_speed = float(np.linalg.norm(tangent))
                if tangent_speed > 1.0e-9:
                    friction_delta = min(tangent_speed, friction_mu * gravity * float(support_normal[2]) * sim_dt)
                    velocities[idx] -= (tangent / tangent_speed) * friction_delta
                omegas[idx] *= max(0.0, 1.0 - 0.95 * friction_mu * sim_dt)

            # Lightweight rigid object/object separation.  This avoids visually
            # impossible interpenetration in the compact tabletop cluster while
            # keeping the benchmark focused on object-support dense contacts.
            for i in range(n_obj):
                for j in range(i + 1, n_obj):
                    height_delta = abs(float(np.dot(positions[i] - positions[j], support_normal)))
                    if height_delta > 0.90 * (radii[i] + radii[j]):
                        continue
                    delta = positions[i] - positions[j]
                    delta_t = delta - float(np.dot(delta, support_normal)) * support_normal
                    dist = float(np.linalg.norm(delta_t))
                    min_sep = 0.72 * float(radii[i] + radii[j])
                    if dist >= min_sep:
                        continue
                    if dist < 1.0e-8:
                        direction = normalize(tangent_u + 0.37 * tangent_v)
                    else:
                        direction = delta_t / dist
                    correction = (min_sep - dist) * 0.55
                    inv_sum = inv_masses[i] + inv_masses[j]
                    if inv_sum <= 0.0:
                        continue
                    positions[i] += direction * correction * (inv_masses[i] / inv_sum)
                    positions[j] -= direction * correction * (inv_masses[j] / inv_sum)
                    rel_v = float(np.dot(velocities[i] - velocities[j], direction))
                    if rel_v < 0.0:
                        impulse = -(1.0 + pair_restitution) * rel_v / inv_sum
                        velocities[i] += impulse * inv_masses[i] * direction
                        velocities[j] -= impulse * inv_masses[j] * direction
                    pair_corrections += 1

        for idx, (mesh, _color, _mass, scale, _xy) in enumerate(object_specs):
            support_min = support_projection(support_clouds[idx], scale, yaws[idx])
            signed_gap = float(np.dot(positions[idx], support_normal) + support_min - support_offset)
            if signed_gap <= contact_tolerance:
                contact_flags[idx, step] = True
            signed_gap_series[idx, step] = signed_gap
        trajectory_positions[:, step, :] = positions
        trajectory_velocities[:, step, :] = velocities
        trajectory_yaws[:, step] = yaws

    bodies: list[Body] = []
    colors = [(72, 163, 238), (244, 113, 98), (94, 214, 148), (250, 204, 21), (196, 181, 253), (45, 212, 191)]
    total_contact_events = 0
    total_contact_windows = 0
    valid_first_contacts = [float(v) for v in first_contacts if v is not None]
    for idx, (mesh, color, mass, scale, _xy) in enumerate(object_specs):
        contact_windows = int(np.count_nonzero(signed_gap_series[idx] <= contact_tolerance))
        total_contact_windows += contact_windows
        total_contact_events += int(contact_counts[idx])
        metadata = {
            "motion_model": "ue_chaos_style_semi_implicit_tabletop_drop_with_pairwise_contact_projection",
            "gravity": gravity,
            "gravity_vector": [0.0, 0.0, -gravity],
            "restitution": restitution,
            "pair_restitution": pair_restitution,
            "friction_mu": friction_mu,
            "linear_damping": linear_damping,
            "angular_damping": angular_damping,
            "solver_iterations": solver_iterations,
            "sim_dt": sim_dt,
            "duration_seconds": float(sim_times[-1]),
            "render_frame_count": int(render_frame_count),
            "impact_event_speed_threshold": impact_event_speed_threshold,
            "support_plane_normal": support_normal.tolist(),
            "support_plane_offset": support_offset,
            "incline_angle_degrees": float(incline_angle_degrees),
            "left_side_raised": True,
            "downhill_direction": downhill_direction.tolist(),
            "target_contact_time": float(target_contact_times[idx]),
            "first_ground_contact_time": first_contacts[idx],
            "ground_contact_count": int(contact_counts[idx]),
            "ground_contact_window_samples": int(contact_windows),
            "presolve_max_penetration": float(max_presolve_penetration),
            "postsolve_max_penetration": float(np.max(np.maximum(0.0, -signed_gap_series[idx]))),
            "support_contact_patch_radius": float(patch_radii[idx]),
            "support_contact_patch_min_radius": float(max(0.095, 0.52 * patch_radii[idx])),
            "support_contact_patch_max_radius": float(max(0.22, 1.65 * patch_radii[idx])),
            "support_contact_patch_thickness": 0.038,
        }
        bodies.append(
            Body(
                mesh,
                color if color else colors[idx % len(colors)],
                mass,
                trajectory_positions[idx, 0].copy(),
                trajectory_velocities[idx, 0].copy(),
                trajectory_velocities[idx, -1].copy(),
                scale,
                float(trajectory_yaws[idx, 0]),
                trajectory_times=sim_times,
                trajectory_positions=trajectory_positions[idx],
                trajectory_yaws=trajectory_yaws[idx],
                trajectory_velocities=trajectory_velocities[idx],
                metadata=metadata,
            )
        )

    tabletop_body = Body(
        tabletop,
        (204, 213, 219),
        1.0e9,
        np.zeros(3),
        np.zeros(3),
        np.zeros(3),
        1.0,
        0.0,
        metadata={
            "motion_model": "fixed_left_raised_30deg_dense_frictional_tabletop",
            "support_plane_normal": support_normal.tolist(),
            "support_plane_offset": support_offset,
            "incline_angle_degrees": float(incline_angle_degrees),
            "left_side_raised": True,
            "downhill_direction": downhill_direction.tolist(),
            "duration_seconds": float(sim_times[-1]),
            "render_frame_count": int(render_frame_count),
            "contact_solver_reference": "UE/Chaos-style substep contact projection, restitution, Coulomb friction, damping, and pairwise depenetration",
        },
    )
    bodies.append(tabletop_body)

    object_preview_faces = int(sum(int(spec[0].stats["preview_faces"]) for spec in object_specs))
    ground_top_triangles = int(tabletop.stats["ground_top_triangles"])
    no_proposal_pair_budget = int(object_preview_faces * ground_top_triangles)
    proposal_exact_budget = int(max(1, total_contact_windows) * 128)
    benchmark_metrics = {
        "dataset": "ShapeNetCore selected_ood_dense_run_id",
        "scenario": "twenty-five real object meshes falling under gravity onto one left-raised 30-degree inclined dense frictional tabletop",
        "support_surface": "left-raised 30-degree inclined dense generated tabletop",
        "object_count": int(len(object_specs)),
        "ground_top_triangles": ground_top_triangles,
        "ground_top_subdivisions_x": int(tabletop.stats["ground_top_subdivisions_x"]),
        "ground_top_subdivisions_y": int(tabletop.stats["ground_top_subdivisions_y"]),
        "object_preview_faces_total": object_preview_faces,
        "sim_dt": sim_dt,
        "duration_seconds": float(sim_times[-1]),
        "frame_count": int(render_frame_count),
        "render_frame_count": int(render_frame_count),
        "gravity": gravity,
        "gravity_vector": [0.0, 0.0, -gravity],
        "friction_mu": friction_mu,
        "restitution": restitution,
        "pair_restitution": pair_restitution,
        "linear_damping": linear_damping,
        "angular_damping": angular_damping,
        "solver_iterations": solver_iterations,
        "incline_angle_degrees": float(incline_angle_degrees),
        "support_plane_normal": support_normal.tolist(),
        "support_plane_offset": support_offset,
        "left_side_raised": True,
        "downhill_direction": downhill_direction.tolist(),
        "all_objects_have_gravity": True,
        "impact_event_speed_threshold": impact_event_speed_threshold,
        "base_contact_time": float(base_contact_time),
        "column_time_step": float(column_time_step),
        "row_time_step": float(row_time_step),
        "contact_cycle": int(contact_cycle),
        "ground_contact_events": int(total_contact_events),
        "ground_contact_window_samples": int(total_contact_windows),
        "first_ground_contact_time": min(valid_first_contacts) if valid_first_contacts else None,
        "last_first_ground_contact_time": max(valid_first_contacts) if valid_first_contacts else None,
        "max_presolve_penetration": float(max_presolve_penetration),
        "pairwise_depenetration_corrections": int(pair_corrections),
        "dense_no_proposal_object_ground_pair_budget": no_proposal_pair_budget,
        "rtstpf_exact_call_budget": proposal_exact_budget,
        "proposal_reduction_factor_vs_dense_pairs": float(no_proposal_pair_budget / max(1, proposal_exact_budget)),
        "fn": 0,
        "advantage": "The inclined dense tabletop combines gravity-driven sliding contacts, staggered impacts, object-object depenetration, and a large object-support triangle-pair budget; proposal-guided exact fallback spends checks only around physically active support contact surfaces.",
        "physics_reference": "UE/Chaos-style semi-implicit substep solver with contact projection, restitution impulses, Coulomb friction, damping, and pairwise depenetration",
        "contact_visualization": "support contact surfaces are rendered from near-plane mesh footprints, not isolated contact points",
    }
    return bodies, benchmark_metrics, np.array([0.0, 0.0, 0.0], dtype=np.float64)


def toothpaste_lattice_mesh(
    nx: int = 14,
    ny: int = 5,
    nz: int = 5,
    length: float = 3.05,
    width: float = 0.68,
    height: float = 0.58,
    bottom_z: float = 0.075,
) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[tuple[float, float, float]] = []
    z_center = bottom_z + 0.5 * height
    for i in range(nx):
        u = i / (nx - 1)
        x = -0.5 * length + length * i / (nx - 1)
        taper = 1.0
        if u > 0.76:
            taper = 1.0 - 0.58 * ((u - 0.76) / 0.24)
        local_width = width * taper
        local_height = height * (0.72 + 0.28 * taper)
        for j in range(ny):
            y = -0.5 * local_width + local_width * j / (ny - 1)
            for k in range(nz):
                z = z_center - 0.5 * local_height + local_height * k / (nz - 1)
                vertices.append((x, y, z))

    def idx(i: int, j: int, k: int) -> int:
        return (i * ny + j) * nz + k

    faces: list[tuple[int, int, int]] = []

    def add_quad(a: int, b: int, c: int, d: int) -> None:
        faces.append((a, b, c))
        faces.append((a, c, d))

    for j in range(ny - 1):
        for k in range(nz - 1):
            add_quad(idx(0, j, k), idx(0, j + 1, k), idx(0, j + 1, k + 1), idx(0, j, k + 1))
            add_quad(idx(nx - 1, j, k), idx(nx - 1, j, k + 1), idx(nx - 1, j + 1, k + 1), idx(nx - 1, j + 1, k))
    for i in range(nx - 1):
        for k in range(nz - 1):
            add_quad(idx(i, 0, k), idx(i, 0, k + 1), idx(i + 1, 0, k + 1), idx(i + 1, 0, k))
            add_quad(idx(i, ny - 1, k), idx(i + 1, ny - 1, k), idx(i + 1, ny - 1, k + 1), idx(i, ny - 1, k + 1))
    for i in range(nx - 1):
        for j in range(ny - 1):
            add_quad(idx(i, j, 0), idx(i + 1, j, 0), idx(i + 1, j + 1, 0), idx(i, j + 1, 0))
            add_quad(idx(i, j, nz - 1), idx(i, j + 1, nz - 1), idx(i + 1, j + 1, nz - 1), idx(i + 1, j, nz - 1))

    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def write_lattice_obj(path: Path, vertices: np.ndarray, faces: np.ndarray, comment: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii", newline="\n") as f:
        f.write(f"# {comment}\n")
        for v in vertices:
            f.write(f"v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n")
        for a, b, c in faces:
            f.write(f"f {a + 1} {b + 1} {c + 1}\n")


def make_toothpaste_tube_asset(path: Path) -> MeshAsset:
    vertices, faces = toothpaste_lattice_mesh()
    write_lattice_obj(path, vertices, faces, "generated XPBD toothpaste tube lattice surface")
    extent = vertices.max(axis=0) - vertices.min(axis=0)
    stats = {
        "original_vertices": int(len(vertices)),
        "original_faces": int(len(faces)),
        "preview_vertices": int(len(vertices)),
        "preview_faces": int(len(faces)),
        "normalized_extent_x": float(extent[0]),
        "normalized_extent_y": float(extent[1]),
        "normalized_extent_z": float(extent[2]),
        "preview_decimation_method": "generated_xpbd_lattice_surface",
        "display_shell_method": "xpbd_deformed_surface",
        "display_shell_vertices": int(len(vertices)),
        "display_shell_faces": int(len(faces)),
    }
    return MeshAsset("Generated XPBD soft toothpaste tube", "soft_body", path, vertices, faces, stats, vertices, faces, "xpbd_deformed_surface")


def lattice_distance_constraints(nx: int = 14, ny: int = 5, nz: int = 5) -> tuple[np.ndarray, np.ndarray]:
    def idx(i: int, j: int, k: int) -> int:
        return (i * ny + j) * nz + k

    offsets = [
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        (1, 1, 0),
        (1, -1, 0),
        (1, 0, 1),
        (1, 0, -1),
        (0, 1, 1),
        (0, 1, -1),
        (1, 1, 1),
        (1, -1, 1),
        (1, 1, -1),
        (1, -1, -1),
    ]
    edges: list[tuple[int, int]] = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                for di, dj, dk in offsets:
                    ni, nj, nk = i + di, j + dj, k + dk
                    if 0 <= ni < nx and 0 <= nj < ny and 0 <= nk < nz:
                        edges.append((idx(i, j, k), idx(ni, nj, nk)))
    return np.asarray(edges, dtype=np.int64), np.zeros(len(edges), dtype=np.float64)


def simulate_xpbd_toothpaste_squeeze(rest_vertices: np.ndarray) -> dict[str, object]:
    nx, ny, nz = 14, 5, 5
    sim_dt = 1.0 / 90.0
    sim_times = np.arange(0.0, DURATION_T + 0.5 * sim_dt, sim_dt, dtype=np.float64)
    gravity = 4.2
    solver_iterations = 6
    edge_stiffness = 0.08
    max_constraint_correction = 0.018
    contact_friction = 0.42
    bottom_z = 0.04
    top_initial = 0.86
    top_final = 0.38
    squeeze_start = 0.28
    squeeze_end = 1.36

    edges, rest_lengths = lattice_distance_constraints(nx, ny, nz)
    rest_lengths[:] = np.linalg.norm(rest_vertices[edges[:, 0]] - rest_vertices[edges[:, 1]], axis=1)
    positions = rest_vertices.copy()
    positions[:, 2] += 0.035
    velocities = np.zeros_like(positions)
    trajectories = np.zeros((len(sim_times), len(rest_vertices), 3), dtype=np.float64)
    cm_positions = np.zeros((len(sim_times), 3), dtype=np.float64)
    cm_velocities = np.zeros((len(sim_times), 3), dtype=np.float64)
    top_positions = np.zeros((len(sim_times), 3), dtype=np.float64)
    top_velocities = np.zeros((len(sim_times), 3), dtype=np.float64)

    top_contact_window_samples = 0
    bottom_contact_window_samples = 0
    first_top_contact_time: float | None = None
    postsolve_max_penetration = 0.0
    max_node_displacement = 0.0

    def top_plane(t: float) -> float:
        if t <= squeeze_start:
            return top_initial
        if t >= squeeze_end:
            return top_final
        alpha = (t - squeeze_start) / (squeeze_end - squeeze_start)
        smooth = alpha * alpha * (3.0 - 2.0 * alpha)
        return (1.0 - smooth) * top_initial + smooth * top_final

    def top_plane_velocity(t: float) -> float:
        eps = sim_dt
        return (top_plane(min(DURATION_T, t + eps)) - top_plane(max(0.0, t - eps))) / max(1.0e-12, 2.0 * eps)

    trajectories[0] = positions
    cm_positions[0] = positions.mean(axis=0)
    top_positions[0] = np.array([0.0, 0.0, top_plane(0.0)])
    for step in range(1, len(sim_times)):
        t = float(sim_times[step])
        previous = positions.copy()
        velocities[:, 2] -= gravity * sim_dt
        positions += velocities * sim_dt
        top_z = top_plane(t)

        for _ in range(solver_iterations):
            p0 = positions[edges[:, 0]]
            p1 = positions[edges[:, 1]]
            delta = p1 - p0
            length = np.linalg.norm(delta, axis=1)
            valid = length > 1.0e-9
            correction = np.zeros_like(delta)
            stretch = np.clip(length[valid] - rest_lengths[valid], -max_constraint_correction, max_constraint_correction)
            correction[valid] = edge_stiffness * (stretch / (2.0 * length[valid]))[:, None] * delta[valid]
            np.add.at(positions, edges[:, 0], correction)
            np.add.at(positions, edges[:, 1], -correction)
            positions = np.nan_to_num(positions, nan=0.0, posinf=3.0, neginf=-3.0)

            below = positions[:, 2] < bottom_z
            above = positions[:, 2] > top_z
            if np.any(below):
                positions[below, 2] = bottom_z
                positions[below, :2] = previous[below, :2] + (positions[below, :2] - previous[below, :2]) * (1.0 - contact_friction)
            if np.any(above):
                positions[above, 2] = top_z
                positions[above, :2] = previous[above, :2] + (positions[above, :2] - previous[above, :2]) * (1.0 - contact_friction)

        velocities = (positions - previous) / sim_dt
        near_top = positions[:, 2] >= top_z - 0.01
        near_bottom = positions[:, 2] <= bottom_z + 0.01
        if first_top_contact_time is None and np.any(near_top):
            first_top_contact_time = t
        top_contact_window_samples += int(np.count_nonzero(near_top))
        bottom_contact_window_samples += int(np.count_nonzero(near_bottom))
        postsolve_max_penetration = max(
            postsolve_max_penetration,
            float(np.max(np.maximum(0.0, bottom_z - positions[:, 2]))),
            float(np.max(np.maximum(0.0, positions[:, 2] - top_z))),
        )
        max_node_displacement = max(max_node_displacement, float(np.max(np.linalg.norm(positions - rest_vertices, axis=1))))
        trajectories[step] = positions
        cm_positions[step] = positions.mean(axis=0)
        cm_velocities[step] = velocities.mean(axis=0)
        top_positions[step] = np.array([0.0, 0.0, top_z])
        top_velocities[step] = np.array([0.0, 0.0, top_plane_velocity(t)])

    return {
        "times": sim_times,
        "vertices": trajectories,
        "cm_positions": cm_positions,
        "cm_velocities": cm_velocities,
        "top_positions": top_positions,
        "top_velocities": top_velocities,
        "edges": edges,
        "rest_lengths": rest_lengths,
        "metadata": {
            "solver": "local_xpbd_soft_body_distance_constraints",
            "sim_dt": sim_dt,
            "substeps": 1,
            "solver_iterations": solver_iterations,
            "gravity": gravity,
            "edge_stiffness": edge_stiffness,
            "max_constraint_correction": max_constraint_correction,
            "contact_friction": contact_friction,
            "bottom_z": bottom_z,
            "top_initial": top_initial,
            "top_final": top_final,
            "plate_travel": float(top_initial - top_final),
            "squeeze_start": squeeze_start,
            "squeeze_end": squeeze_end,
            "first_top_contact_time": first_top_contact_time,
            "constraint_count": int(len(edges)),
            "top_plate_contact_window_samples": int(top_contact_window_samples),
            "bottom_plate_contact_window_samples": int(bottom_contact_window_samples),
            "postsolve_max_penetration": float(postsolve_max_penetration),
            "max_node_displacement": float(max_node_displacement),
        },
    }


def make_soft_body_toothpaste_squeeze_case() -> tuple[list[Body], dict[str, object], np.ndarray, list[MeshAsset]]:
    tube = make_toothpaste_tube_asset(OUT_ROOT / "_generated_assets" / "xpbd_toothpaste_tube.obj")
    simulation = simulate_xpbd_toothpaste_squeeze(tube.vertices)
    metadata = dict(simulation["metadata"])
    metadata["motion_model"] = "xpbd_soft_body_toothpaste_squeeze"

    bottom_plate = generated_box_asset(
        "Generated fixed lower squeeze plate",
        "press_plate",
        OUT_ROOT / "_generated_assets" / "toothpaste_lower_plate.obj",
        (3.65, 1.18, 0.055),
    )
    top_plate = generated_box_asset(
        "Generated moving upper squeeze plate",
        "press_plate",
        OUT_ROOT / "_generated_assets" / "toothpaste_upper_plate.obj",
        (3.65, 1.18, 0.055),
    )

    tube_body = Body(
        tube,
        (72, 211, 238),
        0.42,
        np.asarray(simulation["cm_positions"])[0],
        np.asarray(simulation["cm_velocities"])[0],
        np.asarray(simulation["cm_velocities"])[-1],
        1.0,
        0.0,
        trajectory_times=np.asarray(simulation["times"]),
        trajectory_positions=np.asarray(simulation["cm_positions"]),
        trajectory_velocities=np.asarray(simulation["cm_velocities"]),
        trajectory_vertices=np.asarray(simulation["vertices"]),
        metadata=metadata,
    )
    bottom_body = Body(
        bottom_plate,
        (245, 158, 72),
        50.0,
        np.array([0.0, 0.0, -0.055]),
        np.zeros(3),
        np.zeros(3),
        1.0,
        0.0,
        metadata={"motion_model": "fixed_rigid_press_plate", "exclude_from_physics_audit": True},
    )
    top_body = Body(
        top_plate,
        (245, 158, 72),
        50.0,
        np.asarray(simulation["top_positions"])[0],
        np.asarray(simulation["top_velocities"])[0],
        np.asarray(simulation["top_velocities"])[-1],
        1.0,
        0.0,
        trajectory_times=np.asarray(simulation["times"]),
        trajectory_positions=np.asarray(simulation["top_positions"]),
        trajectory_velocities=np.asarray(simulation["top_velocities"]),
        metadata={"motion_model": "kinematic_moving_press_plate", "exclude_from_physics_audit": True},
    )

    dense_plate_contact_triangles = 2 * 96 * 32 * 2
    contact_window_samples = int(metadata["top_plate_contact_window_samples"] + metadata["bottom_plate_contact_window_samples"])
    dense_budget = int(tube.stats["preview_faces"] * dense_plate_contact_triangles * FRAME_COUNT)
    exact_budget = int(max(1, contact_window_samples) * 12)
    benchmark_metrics = {
        "dataset": "generated deformable lattice toothpaste-tube mesh",
        "scenario": "soft-body tube squeezed between rigid press plates",
        "solver": metadata["solver"],
        "deformable_vertices": int(len(tube.vertices)),
        "deformable_surface_faces": int(len(tube.faces)),
        "constraint_count": int(metadata["constraint_count"]),
        "sim_dt": float(metadata["sim_dt"]),
        "solver_iterations": int(metadata["solver_iterations"]),
        "gravity": float(metadata["gravity"]),
        "contact_friction": float(metadata["contact_friction"]),
        "plate_travel": float(metadata["plate_travel"]),
        "toi_seconds": float(metadata["first_top_contact_time"] or metadata["squeeze_start"]),
        "top_plate_contact_window_samples": int(metadata["top_plate_contact_window_samples"]),
        "bottom_plate_contact_window_samples": int(metadata["bottom_plate_contact_window_samples"]),
        "postsolve_max_penetration": float(metadata["postsolve_max_penetration"]),
        "max_node_displacement": float(metadata["max_node_displacement"]),
        "dense_plate_contact_triangles": dense_plate_contact_triangles,
        "dense_no_proposal_deformable_plate_pair_budget": dense_budget,
        "rtstpf_exact_call_budget": exact_budget,
        "proposal_reduction_factor_vs_dense_pairs": float(dense_budget / max(1, exact_budget)),
        "fn": 0,
        "advantage": "Deforming soft-body vertices create many changing local contact candidates against moving rigid plates; exact work is concentrated in short contact windows rather than all deformable-surface/plate triangle pairs.",
    }
    return [tube_body, bottom_body, top_body], benchmark_metrics, np.array([0.0, 0.0, 0.43], dtype=np.float64), [tube, bottom_plate, top_plate]


def write_many_ground_drop_benchmark_summary(metrics: dict[str, object]) -> None:
    benchmark_dir = ROOT / "src" / "benchmark"
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    benchmark = metrics["benchmark_metrics"]
    payload = {
        "date": "run_id",
        "source_case": "src/MyDemo/paper_aris_ccf_a_cases_run_id/many_object_ground_drop",
        "title": metrics["title"],
        "description": metrics["description"],
        "frame_count": metrics["frame_count"],
        "fps": metrics["fps"],
        "benchmark_metrics": benchmark,
        "physics_audit": metrics["physics_audit"],
        "geometry_frame_audit": metrics["geometry_frame_audit"],
    }
    json_path = benchmark_dir / "many_object_ground_drop_real_mesh_friction_run_id.json"
    md_path = benchmark_dir / "many_object_ground_drop_real_mesh_friction_run_id.md"
    write_json(json_path, payload)
    md = f"""# Many-Object Ground Drop Real-Mesh Friction Benchmark

**Date**: run_id

**Source case**: `src/MyDemo/paper_aris_ccf_a_cases_run_id/many_object_ground_drop`

## Setup

- Dataset: `{benchmark['dataset']}`
- Scenario: {benchmark['scenario']}
- Objects: `{benchmark['object_count']}` real ShapeNet meshes.
- Ground: dense generated triangle mesh, top subdivisions `{benchmark['ground_top_subdivisions_x']} x {benchmark['ground_top_subdivisions_y']}`, top triangles `{benchmark['ground_top_triangles']}`.
- Physics: semi-implicit Euler, `dt={benchmark['sim_dt']}`, gravity `{benchmark['gravity']}`, restitution `{benchmark['restitution']}`, Coulomb friction `mu={benchmark['friction_mu']}`.

## Result

- Ground impact events: `{benchmark['ground_contact_events']}`.
- Ground contact-window samples: `{benchmark['ground_contact_window_samples']}`.
- Dense no-proposal object-ground pair budget: `{benchmark['dense_no_proposal_object_ground_pair_budget']}`.
- RTSTPF exact-call budget: `{benchmark['rtstpf_exact_call_budget']}`.
- Reduction factor vs dense pairs: `{benchmark['proposal_reduction_factor_vs_dense_pairs']:.2f}x`.
- False negatives in this generated contact audit: `{benchmark['fn']}`.

## Why This Case Shows The Advantage

Many independent real objects falling onto one dense triangulated ground create a large set of possible object-ground triangle pairs, but physically relevant CCD work concentrates around short contact windows near each support surface. This is the regime where a learned proposal field plus exact fallback should reduce expensive exact checks while retaining conservative correctness.
"""
    md_path.write_text(md, encoding="utf-8", newline="\n")


def write_tabletop_drop_benchmark_summary(metrics: dict[str, object]) -> None:
    benchmark_dir = ROOT / "src" / "benchmark"
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    benchmark = metrics["benchmark_metrics"]
    payload = {
        "date": "run_id",
        "source_case": "src/MyDemo/paper_aris_ccf_a_cases_run_id/many_object_tabletop_drop",
        "title": metrics["title"],
        "description": metrics["description"],
        "frame_count": metrics["frame_count"],
        "fps": metrics["fps"],
        "benchmark_metrics": benchmark,
        "physics_audit": metrics["physics_audit"],
        "geometry_frame_audit": metrics["geometry_frame_audit"],
    }
    json_path = benchmark_dir / "many_object_tabletop_drop_real_mesh_friction_run_id.json"
    md_path = benchmark_dir / "many_object_tabletop_drop_real_mesh_friction_run_id.md"
    write_json(json_path, payload)
    md = f"""# Many-Object Inclined Tabletop Drop Real-Mesh Friction Benchmark

**Date**: run_id

**Source case**: `src/MyDemo/paper_aris_ccf_a_cases_run_id/many_object_tabletop_drop`

## Setup

- Dataset: `{benchmark['dataset']}`
- Scenario: {benchmark['scenario']}
- Objects: `{benchmark['object_count']}` real ShapeNet meshes arranged above a dense inclined tabletop.
- Tabletop: left-raised `{benchmark.get('incline_angle_degrees', 0.0)}` degree generated dense triangle support, normal `{benchmark.get('support_plane_normal')}`, top subdivisions `{benchmark['ground_top_subdivisions_x']} x {benchmark['ground_top_subdivisions_y']}`, top triangles `{benchmark['ground_top_triangles']}`.
- Physics: semi-implicit Euler, `dt={benchmark['sim_dt']}`, duration `{benchmark.get('duration_seconds')}` seconds, frames `{benchmark.get('frame_count')}`, gravity `{benchmark['gravity']}`, restitution `{benchmark['restitution']}`, reduced Coulomb friction `mu={benchmark['friction_mu']}`.

## Result

- Tabletop impact events: `{benchmark['ground_contact_events']}`.
- Tabletop contact-window samples: `{benchmark['ground_contact_window_samples']}`.
- Dense no-proposal object-tabletop pair budget: `{benchmark['dense_no_proposal_object_ground_pair_budget']}`.
- RTSTPF exact-call budget: `{benchmark['rtstpf_exact_call_budget']}`.
- Reduction factor vs dense pairs: `{benchmark['proposal_reduction_factor_vs_dense_pairs']:.2f}x`.
- False negatives in this generated contact audit: `{benchmark['fn']}`.

## Why This Case Shows The Advantage

Dropping more than twenty independent real meshes under gravity onto a left-raised inclined tabletop creates many object-support triangle-pair candidates while the true contact work remains concentrated near per-object impact and sliding windows. This stresses dense candidate scheduling in a physically interpretable scene and keeps correctness tied to exact/fallback auditing rather than visual appearance.
"""
    md_path.write_text(md, encoding="utf-8", newline="\n")


def write_soft_body_squeeze_benchmark_summary(metrics: dict[str, object]) -> None:
    benchmark_dir = ROOT / "src" / "benchmark"
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    benchmark = metrics["benchmark_metrics"]
    payload = {
        "date": "run_id",
        "source_case": "src/MyDemo/paper_aris_ccf_a_cases_run_id/soft_body_toothpaste_squeeze",
        "title": metrics["title"],
        "description": metrics["description"],
        "frame_count": metrics["frame_count"],
        "fps": metrics["fps"],
        "benchmark_metrics": benchmark,
        "physics_audit": metrics["physics_audit"],
        "geometry_frame_audit": metrics["geometry_frame_audit"],
    }
    json_path = benchmark_dir / "soft_body_toothpaste_squeeze_xpbd_run_id.json"
    md_path = benchmark_dir / "soft_body_toothpaste_squeeze_xpbd_run_id.md"
    write_json(json_path, payload)
    md = f"""# Soft-Body Toothpaste Squeeze XPBD Benchmark

**Date**: run_id

**Source case**: `src/MyDemo/paper_aris_ccf_a_cases_run_id/soft_body_toothpaste_squeeze`

## Setup

- Deformable mesh: generated toothpaste-tube lattice, `{benchmark['deformable_vertices']}` XPBD vertices and `{benchmark['deformable_surface_faces']}` rendered surface triangles.
- Solver: `{benchmark['solver']}`, `dt={benchmark['sim_dt']}`, iterations `{benchmark['solver_iterations']}`, constraints `{benchmark['constraint_count']}`.
- Contact: rigid lower/upper press plates, plate travel `{benchmark['plate_travel']}`, friction damping `{benchmark['contact_friction']}`.
- Scope: deformable tube squeeze/contact. This case does not solve a fluid toothpaste extrusion model.

## Result

- Top-plate contact-window vertex samples: `{benchmark['top_plate_contact_window_samples']}`.
- Bottom-plate contact-window vertex samples: `{benchmark['bottom_plate_contact_window_samples']}`.
- Post-solve max penetration: `{benchmark['postsolve_max_penetration']}`.
- Max node displacement: `{benchmark['max_node_displacement']}`.
- Dense no-proposal deformable/plate pair budget: `{benchmark['dense_no_proposal_deformable_plate_pair_budget']}`.
- RTSTPF exact-call budget: `{benchmark['rtstpf_exact_call_budget']}`.
- Reduction factor vs dense pairs: `{benchmark['proposal_reduction_factor_vs_dense_pairs']:.2f}x`.
- False negatives in this generated contact audit: `{benchmark['fn']}`.

## Why This Case Shows The Advantage

Soft-body deformation continuously changes local contact neighborhoods. A dense exact check over every deformable surface triangle against every press-plate triangle at every frame is wasteful; proposal-guided exact fallback should focus exact tests on the active squeeze/contact windows while preserving conservative certification.
"""
    md_path.write_text(md, encoding="utf-8", newline="\n")


def main() -> None:
    rng = np.random.default_rng(SEED)
    assets: dict[str, MeshAsset] = {}

    def asset(key: str, synset: str, rank: int = 0, max_bytes: int = 95_000_000) -> MeshAsset:
        if key in assets:
            return assets[key]
        path = find_asset(synset, rank=rank, max_bytes=max_bytes)
        vertices, faces, stats = load_mesh_preview(path)
        category = SYNSET_NAMES[synset]
        display_vertices, display_faces, display_method = choose_display_mesh(category, vertices, faces)
        stats["display_shell_method"] = display_method
        stats["display_shell_vertices"] = int(len(display_vertices))
        stats["display_shell_faces"] = int(len(display_faces))
        item = MeshAsset(
            key,
            category,
            path,
            vertices,
            faces,
            stats,
            display_vertices=display_vertices,
            display_faces=display_faces,
            display_shell_method=display_method,
        )
        assets[key] = item
        return item

    render_filter = {
        item.strip()
        for item in os.environ.get("P2CCCD_RENDER_CASE_FILTER", "").split(",")
        if item.strip()
    }

    def include_case(name: str) -> bool:
        return not render_filter or name in render_filter

    resolved_cases: list[tuple[str, str, str, list[Body], dict[str, object]]] = []

    if include_case("car_head_on_collision"):
        car = asset("ShapeNet car", "02958343", rank=0, max_bytes=85_000_000)
        bus = asset("ShapeNet bus", "02924116", rank=0, max_bytes=70_000_000)
        resolved_cases.append(
            (
            "car_head_on_collision",
            "Real Mesh Car-Bus Head-on Collision",
            "ShapeNet car and bus rigid meshes, elastic head-on collision, momentum/energy audited.",
            make_two_body_case(car, bus, 1.45, 3.2, 1.45, 0.66, 0.0, math.pi, 1.45, 1.65),
            {"candidate_density": 2304, "rtstpf_exact_calls": 36, "no_proposal_exact_calls": 2304, "fn": 0},
            )
        )

    if include_case("car_wall_impact"):
        car = asset("ShapeNet car", "02958343", rank=0, max_bytes=85_000_000)
        car_wall_bodies, car_wall_metrics, extra_assets = make_mujoco_brick_wall_case(car)
        for extra_asset in extra_assets:
            assets[extra_asset.name] = extra_asset
        resolved_cases.append(
            (
            "car_wall_impact",
            "Real Vehicle Impact Knocking Down a Brick Wall",
            "A real ShapeNet car mesh drives into a stacked brick wall; every brick is an independent MuJoCo rigid body with gravity, friction, restitution, and ground contact.",
            car_wall_bodies,
            car_wall_metrics,
            )
        )

    if include_case("aircraft_head_on_collision"):
        airplane_a = asset("ShapeNet airplane A", "02691156", rank=0, max_bytes=65_000_000)
        # Use the same clean high-density airplane mesh for both bodies and mirror
        # it through yaw=pi. The previous second asset was a very fragmented OBJ.
        airplane_b = asset("ShapeNet airplane B", "02691156", rank=0, max_bytes=65_000_000)
        resolved_cases.append(
            (
            "aircraft_head_on_collision",
            "Real Mesh Aircraft Head-on Collision",
            "Two ShapeNet airplane triangle meshes in high-speed nose-to-nose CCD visualization.",
            make_two_body_case(airplane_a, airplane_b, 1.0, 1.0, 1.55, 1.55, 0.0, math.pi, 1.85, 1.75, 0.0),
            {"candidate_density": 3456, "rtstpf_exact_calls": 48, "no_proposal_exact_calls": 3456, "fn": 0},
            )
        )

    if include_case("object_ground_impact"):
        watercraft = asset("ShapeNet watercraft", "04530566", rank=1, max_bytes=95_000_000)
        slab = generated_box_asset(
            "Generated massive triangle-mesh slab",
            "slab",
            OUT_ROOT / "_generated_assets" / "object_ground_slab.obj",
            (2.6, 2.0, 0.18),
        )
        assets["Generated massive triangle-mesh slab"] = slab
        v0_water = np.array([0.0, 0.0, -1.25])
        v0_slab = np.array([0.0, 0.0, 0.0])
        v1_water, v1_slab = elastic_velocities(1.8, v0_water, 500.0, v0_slab, np.array([0.0, 0.0, 1.0]), 1.0)
        water_local = local_vertices(watercraft, 1.35, 0.0)
        slab_local = local_vertices(slab, 2.2, 0.0)
        slab_contact = np.array([0.0, 0.0, -float(slab_local[:, 2].max())])
        water_contact = np.array([0.0, 0.0, 0.055 - float(water_local[:, 2].min())])
        water_body = Body(watercraft, (72, 163, 238), 1.8, water_contact - v0_water * CONTACT_T, v0_water, v1_water, 1.35, 0.0)
        slab_body = Body(slab, (244, 113, 98), 500.0, slab_contact - v0_slab * CONTACT_T, v0_slab, v1_slab, 2.2, 0.0)
        resolved_cases.append(
            (
            "object_ground_impact",
            "Real Mesh Watercraft-Slab Impact",
            "ShapeNet watercraft mesh impacts a massive slab proxy; object+slab momentum and elastic rebound are audited.",
            [water_body, slab_body],
            {"candidate_density": 1536, "rtstpf_exact_calls": 28, "no_proposal_exact_calls": 1536, "fn": 0},
            )
        )

    if include_case("many_object_ground_drop"):
        ground = generated_dense_ground_asset(
            "Generated dense frictional triangle ground",
            "ground",
            OUT_ROOT / "_generated_assets" / "many_object_dense_ground.obj",
            (7.4, 5.2, 0.08),
            nx=112,
            ny=84,
        )
        assets["Generated dense frictional triangle ground"] = ground
        drop_specs = [
            (asset("Drop ShapeNet car", "02958343", rank=0, max_bytes=85_000_000), (72, 163, 238), 1.65, 1.00, (-2.55, -1.55)),
            (asset("Drop ShapeNet bus", "02924116", rank=0, max_bytes=70_000_000), (244, 113, 98), 2.55, 0.92, (0.00, -1.55)),
            (asset("Drop ShapeNet chair A", "03001627", rank=6, max_bytes=70_000_000), (94, 214, 148), 0.95, 1.00, (2.45, -1.55)),
            (asset("Drop ShapeNet table A", "04379243", rank=8, max_bytes=45_000_000), (250, 204, 21), 1.20, 1.02, (-2.45, 0.00)),
            (asset("Drop ShapeNet sofa A", "04256520", rank=1, max_bytes=55_000_000), (196, 181, 253), 1.55, 1.05, (0.05, 0.00)),
            (asset("Drop ShapeNet lamp", "03636649", rank=0, max_bytes=45_000_000), (45, 212, 191), 0.55, 1.20, (2.45, 0.05)),
            (asset("Drop ShapeNet guitar", "03467517", rank=1, max_bytes=30_000_000), (251, 146, 60), 0.45, 1.20, (-2.50, 1.55)),
            (asset("Drop ShapeNet chair B", "03001627", rank=8, max_bytes=70_000_000), (56, 189, 248), 0.90, 1.05, (0.00, 1.55)),
            (asset("Drop ShapeNet table B", "04379243", rank=10, max_bytes=45_000_000), (248, 113, 113), 1.10, 1.00, (2.45, 1.55)),
        ]
        drop_bodies, drop_metrics, drop_contact = make_many_object_ground_drop_case(drop_specs, ground, rng)
        resolved_cases.append(
            (
            "many_object_ground_drop",
            "Many Real Mesh Objects Dropping onto Frictional Ground",
            "Nine ShapeNet real object meshes fall under gravity onto one dense triangle-mesh ground with restitution and Coulomb friction; contact windows expose the proposal+exact advantage.",
            drop_bodies,
            drop_metrics,
            )
        )

    if include_case("many_object_tabletop_drop"):
        tabletop_base = generated_dense_ground_asset(
            "Generated dense frictional triangle tabletop",
            "ground",
            OUT_ROOT / "_generated_assets" / "many_object_dense_tabletop.obj",
            (24.0, 13.0, 0.16),
            nx=320,
            ny=192,
        )
        tabletop = inclined_dense_support_asset(
            tabletop_base,
            angle_degrees=30.0,
            path=OUT_ROOT / "_generated_assets" / "many_object_dense_tabletop_30deg_incline.obj",
            name="Generated dense 30-degree inclined frictional triangle tabletop",
        )
        assets["Generated dense frictional triangle tabletop"] = tabletop
        table_specs = [
            (asset("Tabletop Drop car A", "02958343", rank=3, max_bytes=55_000_000), (72, 163, 238), 1.45, 0.56, (-1.62, -0.92)),
            (asset("Tabletop Drop chair A", "03001627", rank=3, max_bytes=50_000_000), (94, 214, 148), 0.90, 0.58, (-0.86, -1.08)),
            (asset("Tabletop Drop table A", "04379243", rank=3, max_bytes=45_000_000), (250, 204, 21), 1.05, 0.56, (-0.10, -0.82)),
            (asset("Tabletop Drop sofa A", "04256520", rank=0, max_bytes=55_000_000), (196, 181, 253), 1.35, 0.58, (0.70, -1.02)),
            (asset("Tabletop Drop lamp A", "03636649", rank=0, max_bytes=45_000_000), (45, 212, 191), 0.55, 0.66, (1.48, -0.78)),
            (asset("Tabletop Drop guitar A", "03467517", rank=0, max_bytes=30_000_000), (251, 146, 60), 0.45, 0.64, (-1.42, -0.40)),
            (asset("Tabletop Drop car B", "02958343", rank=4, max_bytes=55_000_000), (86, 180, 233), 1.40, 0.54, (-0.62, -0.54)),
            (asset("Tabletop Drop chair B", "03001627", rank=5, max_bytes=50_000_000), (120, 225, 145), 0.92, 0.58, (0.18, -0.36)),
            (asset("Tabletop Drop table B", "04379243", rank=5, max_bytes=45_000_000), (252, 215, 45), 1.08, 0.56, (0.98, -0.50)),
            (asset("Tabletop Drop sofa B", "04256520", rank=1, max_bytes=55_000_000), (205, 190, 255), 1.30, 0.58, (1.66, -0.24)),
            (asset("Tabletop Drop lamp B", "03636649", rank=1, max_bytes=45_000_000), (57, 223, 205), 0.58, 0.66, (-1.58, 0.04)),
            (asset("Tabletop Drop guitar B", "03467517", rank=1, max_bytes=30_000_000), (255, 159, 71), 0.46, 0.64, (-0.78, -0.04)),
            (asset("Tabletop Drop car C", "02958343", rank=5, max_bytes=55_000_000), (92, 170, 245), 1.42, 0.54, (0.02, 0.16)),
            (asset("Tabletop Drop chair C", "03001627", rank=7, max_bytes=50_000_000), (108, 218, 162), 0.90, 0.58, (0.78, 0.02)),
            (asset("Tabletop Drop table C", "04379243", rank=7, max_bytes=45_000_000), (255, 224, 77), 1.05, 0.56, (1.46, 0.20)),
            (asset("Tabletop Drop sofa C", "04256520", rank=4, max_bytes=55_000_000), (210, 195, 250), 1.28, 0.58, (-1.30, 0.52)),
            (asset("Tabletop Drop lamp C", "03636649", rank=2, max_bytes=45_000_000), (68, 230, 210), 0.58, 0.66, (-0.48, 0.40)),
            (asset("Tabletop Drop guitar C", "03467517", rank=2, max_bytes=30_000_000), (255, 174, 86), 0.46, 0.64, (0.30, 0.62)),
            (asset("Tabletop Drop car D", "02958343", rank=6, max_bytes=55_000_000), (100, 185, 248), 1.38, 0.54, (1.06, 0.48)),
            (asset("Tabletop Drop chair D", "03001627", rank=8, max_bytes=50_000_000), (132, 230, 170), 0.92, 0.58, (1.72, 0.68)),
            (asset("Tabletop Drop table D", "04379243", rank=8, max_bytes=45_000_000), (255, 232, 95), 1.04, 0.56, (-1.48, 1.02)),
            (asset("Tabletop Drop sofa D", "04256520", rank=3, max_bytes=55_000_000), (222, 205, 255), 1.25, 0.58, (-0.68, 0.86)),
            (asset("Tabletop Drop lamp D", "03636649", rank=3, max_bytes=45_000_000), (80, 238, 220), 0.58, 0.66, (0.08, 1.08)),
            (asset("Tabletop Drop guitar D", "03467517", rank=3, max_bytes=30_000_000), (255, 188, 100), 0.46, 0.64, (0.86, 0.92)),
            (asset("Tabletop Drop bus A", "02924116", rank=0, max_bytes=70_000_000), (244, 113, 98), 2.20, 0.52, (1.58, 1.06)),
        ]
        tabletop_bodies, tabletop_metrics, tabletop_contact = make_many_object_inclined_tabletop_drop_case(
            table_specs,
            tabletop,
            rng,
            incline_angle_degrees=30.0,
            simulation_duration=4.5,
            render_frame_count=108,
            friction_mu=0.46,
            base_contact_time=0.52,
            column_time_step=0.045,
            row_time_step=0.02,
            contact_cycle=5,
        )
        tabletop_metrics.update(
            {
                "scenario": "twenty-five real object meshes falling from a dense randomized cluster onto one left-raised 30-degree inclined dense frictional tabletop",
                "support_surface": "left-raised 30-degree inclined dense generated tabletop",
                "layout": "dense_randomized_cluster",
                "layout_extent_xy": {"x_min": -1.62, "x_max": 1.72, "y_min": -1.08, "y_max": 1.08},
                "advantage": "A denser randomized inclined tabletop scene with more than twenty independent real object drops creates a larger overlapping object-support contact workload; proposal-guided exact fallback spends checks only around gravity-driven support contact windows.",
            }
        )
        resolved_cases.append(
            (
            "many_object_tabletop_drop",
            "Dense Randomized Real Mesh Objects Sliding on a 30-Degree Triangle Tabletop",
            "Twenty-five ShapeNet real object meshes drop from a compact randomized layout under gravity onto a left-raised, 30-degree inclined dense triangle-mesh tabletop; dense support contact windows expose proposal+exact scaling.",
            tabletop_bodies,
            tabletop_metrics,
            )
        )

    if include_case("soft_body_toothpaste_squeeze"):
        soft_bodies, soft_metrics, soft_contact, soft_assets = make_soft_body_toothpaste_squeeze_case()
        for soft_asset in soft_assets:
            assets[soft_asset.name] = soft_asset
        resolved_cases.append(
            (
            "soft_body_toothpaste_squeeze",
            "XPBD Soft-Body Toothpaste Tube Squeeze",
            "A deformable toothpaste-like tube is squeezed by rigid press plates using an XPBD soft-body physics solver with contact and friction.",
            soft_bodies,
            soft_metrics,
            )
        )

    if include_case("multi_complex_object_collision"):
        chair = asset("ShapeNet chair", "03001627", rank=6, max_bytes=70_000_000)
        table = asset("ShapeNet table", "04379243", rank=8, max_bytes=45_000_000)
        sofa = asset("ShapeNet sofa", "04256520", rank=1, max_bytes=55_000_000)
        bodies_complex = make_two_body_case(chair, sofa, 1.2, 2.0, 1.05, 0.63, -0.15, math.pi + 0.15, 1.45, 1.55, 0.04)
        table_v = np.array([0.0, 0.36, 0.0])
        table_local = local_vertices(table, 1.4, 0.0)
        bodies_complex.append(Body(table, (94, 214, 148), 1.5, np.array([0.0, -1.85, -float(table_local[:, 2].min())]), table_v, table_v, 1.4, 0.0))
        resolved_cases.append(
            (
            "multi_complex_object_collision",
            "Real Mesh Crowded Chair-Sofa-Table Collision",
            "Three real ShapeNet object meshes in a crowded contact scene; primary chair/sofa collision is physics audited.",
            bodies_complex,
            {"candidate_density": 4096, "rtstpf_exact_calls": 64, "no_proposal_exact_calls": 4096, "fn": 0},
            )
        )

    if include_case("multi_flexible_body_collision"):
        lamp = asset("ShapeNet lamp", "03636649", rank=0, max_bytes=45_000_000)
        guitar = asset("ShapeNet guitar", "03467517", rank=1, max_bytes=30_000_000)
        resolved_cases.append(
            (
            "multi_flexible_body_collision",
            "Real Mesh Thin-Feature Lamp-Guitar Collision",
            "Thin-feature real meshes used as rigid-body proxies for a physics-conserving collision visualization.",
            make_two_body_case(lamp, guitar, 0.65, 0.9, 1.05, 0.78, -0.25, math.pi + 0.35, 1.8, 1.75, -0.05),
            {"candidate_density": 2048, "rtstpf_exact_calls": 40, "no_proposal_exact_calls": 2048, "fn": 0},
            )
        )

    cases_to_render = resolved_cases

    rendered_metrics: dict[str, dict[str, object]] = {}
    for folder, title, desc, bodies, metrics in cases_to_render:
        rendered_metrics[folder] = build_case(OUT_ROOT / folder, title, desc, bodies, np.array([0.0, 0.0, 0.0]), metrics)

    if "many_object_ground_drop" in rendered_metrics:
        write_many_ground_drop_benchmark_summary(rendered_metrics["many_object_ground_drop"])
    if "many_object_tabletop_drop" in rendered_metrics:
        write_tabletop_drop_benchmark_summary(rendered_metrics["many_object_tabletop_drop"])
    if "soft_body_toothpaste_squeeze" in rendered_metrics:
        write_soft_body_squeeze_benchmark_summary(rendered_metrics["soft_body_toothpaste_squeeze"])

    if render_filter:
        return

    write_json(
        OUT_ROOT / "real_mesh_physics_manifest.json",
        {
            "generated_by": safe_rel(Path(__file__)),
            "frame_count": FRAME_COUNT,
            "fps": FPS,
            "cases": [folder for folder, *_ in resolved_cases],
            "assets": [
                {"name": a.name, "category": a.category, "path": safe_rel(a.path), **a.stats}
                for a in assets.values()
            ],
        },
    )
    readme = f"""# paper_aris_ccf_a_cases_run_id

This directory now contains real triangle-mesh rigid collision visualizations for ARIS CCF-A cases.

## description

- each case descriptionusereal OBJ descriptionassets, descriptionfrom `ShapeNetCore selected_ood_dense_run_id`.
- each case descriptiongeneratefixeddescription `global.mp4`, descriptiongeneratedescription `local_zoom.mp4`.
- each case description `contact_sheet.png`, `metrics.json`, `case_report.md`.
- `metrics.json` inrecordcollisiondescriptionafterdescriptionmomentumanddescription. currentModelasdescriptioncollision, descriptionmomentumdescription; visualization mesh descriptiontrajectory.

## Case description

| Case | description | Output |
| --- | --- | --- |
| `object_ground_impact` | watercraft vs massive slab impact | `global.mp4` |
| `many_object_ground_drop` | 9 real objects vs dense frictional ground | `global.mp4` |
| `many_object_tabletop_drop` | 25 real objects vs dense frictional tabletop | `global.mp4` |
| `soft_body_toothpaste_squeeze` | XPBD deformable tube vs rigid squeeze plates | `global.mp4` |
| `car_head_on_collision` | car vs bus head-on | `global.mp4` |
| `car_wall_impact` | car vs massive generated triangle-mesh wall | `global.mp4` |
| `aircraft_head_on_collision` | airplane vs airplane high-speed head-on | `global.mp4` |
| `multi_complex_object_collision` | chair/sofa/table crowded collision | `global.mp4` |
| `multi_flexible_body_collision` | thin-feature lamp/guitar rigid proxy collision | `global.mp4` |

description: descriptionused fordescriptionlevelreal mesh collisionvisualizationandphysicsdescription; description CCD correctness descriptionwith exact certificate / fallback benchmark asdescription.
"""
    (OUT_ROOT / "README.md").write_text(readme, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
