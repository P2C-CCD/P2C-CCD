from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


os.environ.setdefault("PYTHONIOENCODING", "utf-8")
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[2]
P2CCCD = ROOT / "src"
RUN_TAG = "repeated_rifle_assembly_run_id"
MYDEMO_DIR = P2CCCD / "MyDemo" / RUN_TAG
BENCHMARK_DIR = P2CCCD / "benchmark"
TRAIN_DIR = P2CCCD / "datasets" / "training" / RUN_TAG
OUTPUT_DIR = P2CCCD / "outputs" / "stpf_training" / RUN_TAG
RIFLE_ROOT = P2CCCD / "datasets" / "shapenet_core_v2" / "selected_ood_dense_run_id" / "04090263"


@dataclass
class RifleAssemblyConfig:
    seed: int = fixed_seed
    render_fps: int = 24
    render_frames: int = 288
    cycles: int = 8
    feature_dim: int = 32
    train_epochs: int = 6
    train_batch_size: int = 32768
    max_mesh_edges_per_part: int = 3600
    max_mesh_faces_per_part: int = 7200
    max_mesh_points_per_part: int = 18000
    candidate_rows_per_cycle: int = 9000
    dense_negative_fraction: float = 0.28
    clearance_threshold: float = 0.035
    contact_window: float = 0.18
    insertion_travel: float = 0.58
    bolt_travel: float = 0.28
    magazine_travel: float = 0.46
    exact_cost_scale: float = 1350.0


def ensure_dirs() -> None:
    for path in [MYDEMO_DIR, BENCHMARK_DIR, TRAIN_DIR, OUTPUT_DIR, MYDEMO_DIR / "frames", MYDEMO_DIR / "assets"]:
        path.mkdir(parents=True, exist_ok=True)


def safe_rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def font(size: int) -> ImageFont.ImageFont:
    for candidate in [Path("C:/Windows/Fonts/segoeui.ttf"), Path("C:/Windows/Fonts/arial.ttf")]:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def find_rifle_obj() -> Path:
    objs = sorted(RIFLE_ROOT.rglob("*.obj"), key=lambda p: p.stat().st_size, reverse=True)
    if not objs:
        raise FileNotFoundError(f"No ShapeNet rifle OBJ found under {RIFLE_ROOT}")
    return objs[0]


def load_obj_mesh(path: Path, face_limit: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                if len(parts) >= 4:
                    vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif line.startswith("f "):
                idx: list[int] = []
                for token in line.split()[1:]:
                    base = token.split("/")[0]
                    if base:
                        value = int(base)
                        idx.append(value - 1 if value > 0 else len(vertices) + value)
                if len(idx) >= 3:
                    root = idx[0]
                    for i in range(1, len(idx) - 1):
                        faces.append((root, idx[i], idx[i + 1]))
                        if face_limit is not None and len(faces) >= face_limit:
                            break
            if face_limit is not None and len(faces) >= face_limit and len(vertices) > 20000:
                # Keep loading vertices is unnecessary for visualization once enough faces
                # have been collected from the high-density mesh.
                pass
    if not vertices or not faces:
        raise ValueError(f"OBJ contains insufficient mesh data: {path}")
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int32)
    return v, f


def normalize_and_align(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = vertices.mean(axis=0)
    centered = vertices - center
    sample = centered
    if len(sample) > 60000:
        rng = np.random.default_rng(123)
        sample = sample[rng.choice(len(sample), 60000, replace=False)]
    cov = np.cov(sample.T)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    axes = evecs[:, order]
    aligned = centered @ axes
    if np.percentile(aligned[:, 0], 95) < -np.percentile(aligned[:, 0], 5):
        aligned[:, 0] *= -1
        axes[:, 0] *= -1
    scale = max(float(np.ptp(aligned[:, 0])), 1.0e-9)
    aligned = aligned / scale * 4.8
    aligned[:, 2] -= np.percentile(aligned[:, 2], 2)
    aligned[:, 1] *= 0.72
    aligned[:, 2] *= 0.72
    return aligned, center, axes


def sample_edges(vertices: np.ndarray, faces: np.ndarray, mask: np.ndarray, max_edges: int, seed: int) -> np.ndarray:
    face_ids = np.flatnonzero(mask)
    if face_ids.size == 0:
        return np.zeros((0, 2, 3), dtype=np.float32)
    rng = np.random.default_rng(seed)
    if face_ids.size > max_edges:
        face_ids = rng.choice(face_ids, size=max_edges, replace=False)
    tri = vertices[faces[face_ids]]
    edges = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]], axis=0)
    if len(edges) > max_edges:
        edges = edges[rng.choice(len(edges), size=max_edges, replace=False)]
    return edges.astype(np.float32)


def sample_triangles(vertices: np.ndarray, faces: np.ndarray, mask: np.ndarray, max_faces: int, seed: int) -> np.ndarray:
    face_ids = np.flatnonzero(mask)
    if face_ids.size == 0:
        return np.zeros((0, 3, 3), dtype=np.float32)
    rng = np.random.default_rng(seed)
    if face_ids.size > max_faces:
        face_ids = rng.choice(face_ids, size=max_faces, replace=False)
    return vertices[faces[face_ids]].astype(np.float32)


def sample_surface_points(vertices: np.ndarray, faces: np.ndarray, mask: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    face_ids = np.flatnonzero(mask)
    if face_ids.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    rng = np.random.default_rng(seed)
    tri = vertices[faces[face_ids]]
    areas = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    prob = areas / max(float(areas.sum()), 1.0e-9)
    chosen = rng.choice(face_ids, size=max_points, replace=True, p=prob)
    chosen_tri = vertices[faces[chosen]]
    u = rng.random(max_points)
    v = rng.random(max_points)
    swap = u + v > 1.0
    u[swap] = 1.0 - u[swap]
    v[swap] = 1.0 - v[swap]
    pts = chosen_tri[:, 0] + u[:, None] * (chosen_tri[:, 1] - chosen_tri[:, 0]) + v[:, None] * (chosen_tri[:, 2] - chosen_tri[:, 0])
    return pts.astype(np.float32)


def cuboid_edges(center: np.ndarray, size: np.ndarray) -> np.ndarray:
    sx, sy, sz = size / 2.0
    corners = np.array(
        [
            [-sx, -sy, -sz],
            [sx, -sy, -sz],
            [sx, sy, -sz],
            [-sx, sy, -sz],
            [-sx, -sy, sz],
            [sx, -sy, sz],
            [sx, sy, sz],
            [-sx, sy, sz],
        ],
        dtype=np.float32,
    ) + center.astype(np.float32)
    pairs = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    return np.asarray([[corners[a], corners[b]] for a, b in pairs], dtype=np.float32)


def cuboid_triangles(center: np.ndarray, size: np.ndarray) -> np.ndarray:
    sx, sy, sz = size / 2.0
    corners = np.array(
        [
            [-sx, -sy, -sz],
            [sx, -sy, -sz],
            [sx, sy, -sz],
            [-sx, sy, -sz],
            [-sx, -sy, sz],
            [sx, -sy, sz],
            [sx, sy, sz],
            [-sx, sy, sz],
        ],
        dtype=np.float32,
    ) + center.astype(np.float32)
    quads = [(0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1), (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0)]
    tris: list[np.ndarray] = []
    for a, b, c, d in quads:
        tris.append(corners[[a, b, c]])
        tris.append(corners[[a, c, d]])
    return np.asarray(tris, dtype=np.float32)


def smoothstep(x: np.ndarray | float) -> np.ndarray | float:
    y = np.clip(x, 0.0, 1.0)
    return y * y * (3.0 - 2.0 * y)


def assembly_phase(frame: int, cfg: RifleAssemblyConfig) -> dict[str, float]:
    frames_per_cycle = cfg.render_frames / cfg.cycles
    cycle = min(cfg.cycles - 1, int(frame / frames_per_cycle))
    local = (frame - cycle * frames_per_cycle) / frames_per_cycle
    insert = float(smoothstep(np.clip(local / 0.48, 0.0, 1.0)))
    lock = float(smoothstep(np.clip((local - 0.40) / 0.32, 0.0, 1.0)))
    retract = float(smoothstep(np.clip((local - 0.78) / 0.18, 0.0, 1.0)))
    reset = float(max(0.0, retract))
    repeat_weight = 1.0 - reset
    return {
        "cycle": float(cycle),
        "local": float(local),
        "insert": insert * repeat_weight,
        "lock": lock * repeat_weight,
        "reset": reset,
    }


def build_scene(cfg: RifleAssemblyConfig) -> dict[str, Any]:
    obj_path = find_rifle_obj()
    vertices_raw, faces = load_obj_mesh(obj_path)
    vertices, _, _ = normalize_and_align(vertices_raw)
    centers = vertices[faces].mean(axis=1)
    x = centers[:, 0]
    q = np.quantile(x, [0.16, 0.48, 0.62, 0.80])
    stock_mask = x < q[1]
    receiver_mask = (x >= q[0]) & (x < q[2])
    barrel_mask = x >= q[2]
    static_mask = stock_mask | receiver_mask
    tris_static = sample_triangles(vertices, faces, static_mask, cfg.max_mesh_faces_per_part, cfg.seed + 10)
    tris_barrel = sample_triangles(vertices, faces, barrel_mask, cfg.max_mesh_faces_per_part, cfg.seed + 11)
    points_static = sample_surface_points(vertices, faces, static_mask, cfg.max_mesh_points_per_part, cfg.seed + 20)
    points_barrel = sample_surface_points(vertices, faces, barrel_mask, cfg.max_mesh_points_per_part, cfg.seed + 21)
    edges_static = sample_edges(vertices, faces, static_mask, cfg.max_mesh_edges_per_part, cfg.seed)
    edges_barrel = sample_edges(vertices, faces, barrel_mask, cfg.max_mesh_edges_per_part, cfg.seed + 1)
    bbox_min, bbox_max = vertices.min(axis=0), vertices.max(axis=0)
    connection_x = float(q[2])
    receiver_center = np.array([float(q[1]), 0.0, float(np.percentile(vertices[:, 2], 44))], dtype=np.float32)
    mag_center_final = np.array([float(q[1] - 0.15), -0.03, float(np.percentile(vertices[:, 2], 16))], dtype=np.float32)
    bolt_center_final = np.array([float(q[1] + 0.35), 0.0, float(np.percentile(vertices[:, 2], 72))], dtype=np.float32)
    mesh_stats = {
        "source_obj": safe_rel(obj_path),
        "obj_bytes": int(obj_path.stat().st_size),
        "vertices": int(len(vertices_raw)),
        "faces": int(len(faces)),
        "static_triangles": int(len(tris_static)),
        "barrel_triangles": int(len(tris_barrel)),
        "static_surface_points": int(len(points_static)),
        "barrel_surface_points": int(len(points_barrel)),
        "static_edges": int(len(edges_static)),
        "barrel_edges": int(len(edges_barrel)),
    }
    return {
        "vertices": vertices.astype(np.float32),
        "faces": faces,
        "triangles_static": tris_static,
        "triangles_barrel": tris_barrel,
        "points_static": points_static,
        "points_barrel": points_barrel,
        "edges_static": edges_static,
        "edges_barrel": edges_barrel,
        "bbox_min": bbox_min.astype(np.float32),
        "bbox_max": bbox_max.astype(np.float32),
        "connection_x": connection_x,
        "receiver_center": receiver_center,
        "mag_center_final": mag_center_final,
        "bolt_center_final": bolt_center_final,
        "mesh_stats": mesh_stats,
    }


def part_transforms(frame: int, cfg: RifleAssemblyConfig, scene: dict[str, Any]) -> dict[str, np.ndarray]:
    ph = assembly_phase(frame, cfg)
    insert = ph["insert"]
    lock = ph["lock"]
    barrel_offset = np.array([cfg.insertion_travel * (1.0 - insert), 0.025 * math.sin(2.0 * math.pi * ph["local"]), 0.0], dtype=np.float32)
    mag_offset = np.array([0.0, 0.0, -cfg.magazine_travel * (1.0 - insert)], dtype=np.float32)
    bolt_offset = np.array([cfg.bolt_travel * (1.0 - lock), 0.0, 0.035 * math.sin(4.0 * math.pi * ph["local"])], dtype=np.float32)
    return {
        "barrel": barrel_offset,
        "magazine": scene["mag_center_final"] + mag_offset,
        "bolt": scene["bolt_center_final"] + bolt_offset,
    }


def camera_project(points: np.ndarray, width: int = 1920, height: int = 1080, scale: float = 330.0) -> tuple[np.ndarray, np.ndarray]:
    eye = np.array([3.9, -5.4, 2.75], dtype=np.float64)
    target = np.array([0.25, 0.0, 0.35], dtype=np.float64)
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    rel = points.astype(np.float64) - target
    x = rel @ right
    y = rel @ up
    z = rel @ forward
    xy = np.column_stack([width * 0.50 + scale * x, height * 0.58 - scale * y])
    return xy, z


def draw_grid(draw: ImageDraw.ImageDraw, width: int = 1920, height: int = 1080) -> None:
    grid_lines = []
    for v in np.linspace(-3.2, 3.2, 17):
        grid_lines.append(np.array([[-3.4, v, -0.08], [3.4, v, -0.08]], dtype=np.float32))
        grid_lines.append(np.array([[v, -2.6, -0.08], [v, 2.6, -0.08]], dtype=np.float32))
    for seg in grid_lines:
        xy, _ = camera_project(seg, width, height)
        draw.line([tuple(xy[0]), tuple(xy[1])], fill=(214, 221, 229, 135), width=1)


def draw_edges(draw: ImageDraw.ImageDraw, edges: np.ndarray, color: tuple[int, int, int, int], width_px: int = 1) -> None:
    if edges.size == 0:
        return
    pts = edges.reshape(-1, 3)
    xy, depth = camera_project(pts)
    xy = xy.reshape(-1, 2, 2)
    depth = depth.reshape(-1, 2).mean(axis=1)
    order = np.argsort(depth)
    for i in order:
        a, b = xy[i]
        if -100 <= a[0] <= 2020 and -100 <= b[0] <= 2020 and -100 <= a[1] <= 1180 and -100 <= b[1] <= 1180:
            draw.line([tuple(a), tuple(b)], fill=color, width=width_px)


def draw_triangles(draw: ImageDraw.ImageDraw, triangles: np.ndarray, color: tuple[int, int, int, int]) -> None:
    if triangles.size == 0:
        return
    pts = triangles.reshape(-1, 3)
    xy, depth = camera_project(pts)
    xy = xy.reshape(-1, 3, 2)
    depth = depth.reshape(-1, 3).mean(axis=1)
    v0 = triangles[:, 1] - triangles[:, 0]
    v1 = triangles[:, 2] - triangles[:, 0]
    normals = np.cross(v0, v1)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1.0e-9)
    light = np.array([0.30, -0.42, 0.86], dtype=np.float32)
    light /= np.linalg.norm(light)
    shade = 0.58 + 0.42 * np.clip(np.abs(normals @ light), 0.0, 1.0)
    order = np.argsort(depth)
    r, g, b, a = color
    for i in order:
        poly = xy[i]
        if np.any(poly[:, 0] < -120) or np.any(poly[:, 0] > 2040) or np.any(poly[:, 1] < -120) or np.any(poly[:, 1] > 1200):
            continue
        fill = (int(r * shade[i]), int(g * shade[i]), int(b * shade[i]), a)
        draw.polygon([tuple(p) for p in poly], fill=fill)


def draw_surface_points(draw: ImageDraw.ImageDraw, points: np.ndarray, color: tuple[int, int, int, int], radius_px: int = 1) -> None:
    if points.size == 0:
        return
    xy, depth = camera_project(points)
    order = np.argsort(depth)
    r, g, b, a = color
    for p in xy[order]:
        x, y = float(p[0]), float(p[1])
        if -80 <= x <= 2000 and -80 <= y <= 1160:
            draw.ellipse([x - radius_px, y - radius_px, x + radius_px, y + radius_px], fill=(r, g, b, a))


def draw_contact_blob(draw: ImageDraw.ImageDraw, center: np.ndarray, radius_world: float, color: tuple[int, int, int, int]) -> None:
    xy, _ = camera_project(center.reshape(1, 3))
    x, y = xy[0]
    r = radius_world * 330.0
    draw.ellipse([x - r, y - 0.45 * r, x + r, y + 0.45 * r], fill=color, outline=(255, 120, 50, 150), width=2)
    draw.line([x - 9, y, x + 9, y], fill=(255, 88, 60, 210), width=2)
    draw.line([x, y - 9, x, y + 9], fill=(255, 88, 60, 210), width=2)


def render_frame(scene: dict[str, Any], frame: int, cfg: RifleAssemblyConfig) -> Image.Image:
    width, height = 1920, 1080
    image = Image.new("RGB", (width, height), (248, 250, 252))
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    draw_grid(draw, width, height)
    transforms = part_transforms(frame, cfg, scene)
    ph = assembly_phase(frame, cfg)
    barrel_tris = scene["triangles_barrel"] + transforms["barrel"].reshape(1, 1, 3)
    barrel_points = scene["points_barrel"] + transforms["barrel"].reshape(1, 3)
    barrel_edges = scene["edges_barrel"] + transforms["barrel"].reshape(1, 1, 3)
    mag_tris = cuboid_triangles(transforms["magazine"], np.array([0.38, 0.16, 0.55], dtype=np.float32))
    mag_edges = cuboid_edges(transforms["magazine"], np.array([0.38, 0.16, 0.55], dtype=np.float32))
    bolt_tris = cuboid_triangles(transforms["bolt"], np.array([0.62, 0.13, 0.16], dtype=np.float32))
    bolt_edges = cuboid_edges(transforms["bolt"], np.array([0.62, 0.13, 0.16], dtype=np.float32))
    draw_triangles(draw, scene["triangles_static"], (78, 91, 105, 220))
    draw_triangles(draw, barrel_tris, (58, 154, 231, 220))
    draw_triangles(draw, mag_tris, (242, 174, 70, 196))
    draw_triangles(draw, bolt_tris, (136, 104, 232, 180))
    draw_surface_points(draw, scene["points_static"], (36, 48, 60, 118), radius_px=1)
    draw_surface_points(draw, barrel_points, (10, 106, 205, 145), radius_px=1)
    draw_edges(draw, scene["edges_static"], (43, 54, 66, 190), 1)
    draw_edges(draw, barrel_edges, (24, 128, 215, 235), 2)
    draw_edges(draw, mag_edges, (238, 166, 62, 230), 3)
    draw_edges(draw, bolt_edges, (126, 92, 222, 210), 2)

    insert_contact = ph["insert"] > 0.62 and ph["reset"] < 0.4
    if insert_contact:
        c0 = np.array([scene["connection_x"], 0.0, scene["receiver_center"][2]], dtype=np.float32)
        c1 = transforms["magazine"] + np.array([0.0, 0.0, 0.24], dtype=np.float32)
        c2 = transforms["bolt"] + np.array([-0.20, 0.0, -0.02], dtype=np.float32)
        draw_contact_blob(draw, c0, 0.23, (255, 126, 52, 82))
        draw_contact_blob(draw, c1, 0.19, (255, 206, 69, 86))
        draw_contact_blob(draw, c2, 0.15, (156, 94, 255, 80))

    title_font = font(36)
    small_font = font(20)
    draw.rounded_rectangle([34, 32, width - 34, 142], radius=0, fill=(255, 255, 255, 218), outline=(122, 163, 205, 190), width=2)
    draw.text((58, 55), "Repeated Rifle-Shaped Assembly: dense sliding CCD windows", fill=(28, 36, 46, 255), font=title_font)
    draw.text(
        (58, 96),
        "ShapeNet rifle mesh is split into receiver/stock and moving barrel; procedural magazine and bolt repeat insertion with conservative fallback.",
        fill=(55, 64, 76, 255),
        font=small_font,
    )
    draw.text((width - 360, 60), f"cycle {int(ph['cycle']) + 1}/{cfg.cycles} | t={frame / cfg.render_fps:.2f}s", fill=(0, 128, 96, 255), font=small_font)
    legend_y = height - 96
    draw.rounded_rectangle([36, legend_y, 1010, height - 30], radius=10, fill=(255, 255, 255, 230), outline=(188, 198, 210, 220), width=1)
    legend = [
        ((71, 84, 99, 210), "receiver/stock mesh"),
        ((41, 142, 220, 230), "moving barrel mesh"),
        ((238, 166, 62, 230), "magazine insert"),
        ((126, 92, 222, 220), "bolt slide"),
        ((255, 126, 52, 100), "contact regions/witnesses"),
    ]
    x0 = 60
    for color, text in legend:
        draw.rectangle([x0, legend_y + 22, x0 + 24, legend_y + 44], fill=color)
        draw.text((x0 + 34, legend_y + 18), text, fill=(50, 58, 68, 255), font=small_font)
        x0 += 190
    image = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
    return image


def render_outputs(scene: dict[str, Any], cfg: RifleAssemblyConfig) -> dict[str, str]:
    frames: list[Image.Image] = []
    for i in range(cfg.render_frames):
        frame = render_frame(scene, i, cfg)
        frames.append(frame)
        if i % 4 == 0 or i in {0, cfg.render_frames - 1}:
            frame.save(MYDEMO_DIR / "frames" / f"frame_{i:04d}.png")
    mp4_path = MYDEMO_DIR / "repeated_rifle_assembly.mp4"
    with imageio.get_writer(mp4_path, fps=cfg.render_fps, codec="libx264", quality=8, macro_block_size=1) as writer:
        for frame in frames:
            writer.append_data(np.asarray(frame))
    frames_per_cycle = cfg.render_frames / cfg.cycles
    sheet_ids = [
        0,
        int(0.66 * frames_per_cycle),
        int(1.0 * frames_per_cycle + 0.70 * frames_per_cycle),
        int(3.0 * frames_per_cycle + 0.74 * frames_per_cycle),
        cfg.render_frames - 1,
    ]
    thumbs = [frames[i].resize((640, 360), Image.Resampling.LANCZOS) for i in sheet_ids]
    sheet = Image.new("RGB", (1280, 1080), (248, 250, 252))
    positions = [(0, 0), (640, 0), (0, 360), (640, 360), (320, 720)]
    for thumb, pos in zip(thumbs, positions):
        sheet.paste(thumb, pos)
    contact_sheet = MYDEMO_DIR / "contact_sheet.png"
    sheet.save(contact_sheet)
    return {"mp4": safe_rel(mp4_path), "contact_sheet": safe_rel(contact_sheet)}


def build_candidate_dataset(scene: dict[str, Any], cfg: RifleAssemblyConfig) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    rng = np.random.default_rng(cfg.seed + 77)
    rows = cfg.cycles * cfg.candidate_rows_per_cycle
    features = np.zeros((rows, cfg.feature_dim), dtype=np.float32)
    labels = np.zeros(rows, dtype=np.int64)
    costs = np.zeros(rows, dtype=np.float32)
    qids = np.zeros(rows, dtype=np.int64)
    kinds = np.zeros(rows, dtype=np.int64)
    idx = 0
    groups_per_cycle = 72
    for cycle in range(cfg.cycles):
        for local_group in range(groups_per_cycle):
            phase = (local_group + 0.5) / groups_per_cycle
            group_id = cycle * groups_per_cycle + local_group
            stage = 0 if phase < 0.50 else (1 if phase < 0.75 else 2)
            insert = float(smoothstep(min(1.0, phase / 0.48)))
            lock = float(smoothstep(np.clip((phase - 0.40) / 0.32, 0.0, 1.0)))
            reset = float(smoothstep(np.clip((phase - 0.78) / 0.18, 0.0, 1.0)))
            barrel_gap = abs(cfg.insertion_travel * (1.0 - insert)) + rng.normal(0.0, 0.006)
            mag_gap = abs(cfg.magazine_travel * (1.0 - insert)) + rng.normal(0.0, 0.005)
            bolt_gap = abs(cfg.bolt_travel * (1.0 - lock)) + rng.normal(0.0, 0.004)
            base_gaps = np.array([barrel_gap, mag_gap, bolt_gap], dtype=np.float64)
            group_n = max(36, int(cfg.candidate_rows_per_cycle / groups_per_cycle + rng.normal(0, 8)))
            for _ in range(group_n):
                if idx >= rows:
                    break
                kind = int(rng.choice([0, 1, 2], p=[0.46, 0.34, 0.20]))
                dense_negative = rng.random() < cfg.dense_negative_fraction
                local_gap = float(base_gaps[kind] + rng.normal(0.0, 0.022 if dense_negative else 0.011))
                positive = (local_gap < cfg.clearance_threshold) and (reset < 0.55)
                near = math.exp(-max(local_gap, 0.0) / max(cfg.contact_window, 1.0e-6))
                speed = [1.35, 1.05, 0.78][kind] * (1.0 - 0.45 * lock) + 0.08 * rng.random()
                local_density = [4200.0, 1800.0, 950.0][kind] * (1.0 + 0.25 * rng.random())
                cost = cfg.exact_cost_scale * (0.45 + 1.25 * near) * (1.0 + local_density / 4500.0)
                if dense_negative and not positive:
                    cost *= 1.25
                features[idx, 0] = kind == 0
                features[idx, 1] = kind == 1
                features[idx, 2] = kind == 2
                features[idx, 3] = phase
                features[idx, 4] = insert
                features[idx, 5] = lock
                features[idx, 6] = reset
                features[idx, 7] = local_gap
                features[idx, 8] = near
                features[idx, 9] = speed
                features[idx, 10] = local_density / 5000.0
                features[idx, 11] = float(cycle) / max(1, cfg.cycles - 1)
                features[idx, 12] = math.sin(2.0 * math.pi * phase)
                features[idx, 13] = math.cos(2.0 * math.pi * phase)
                features[idx, 14] = float(scene["mesh_stats"]["faces"]) / 1.0e6
                features[idx, 15] = float(scene["mesh_stats"]["vertices"]) / 1.0e6
                noise = rng.normal(0.0, 0.15, size=cfg.feature_dim - 16)
                features[idx, 16:] = noise
                labels[idx] = 1 if positive else 0
                costs[idx] = float(cost)
                qids[idx] = group_id
                kinds[idx] = kind
                idx += 1
    features = features[:idx]
    labels = labels[:idx]
    costs = costs[:idx]
    qids = qids[:idx]
    kinds = kinds[:idx]
    order = np.arange(idx)
    rng.shuffle(order)
    n_train = int(0.70 * idx)
    n_val = int(0.15 * idx)
    split_ids = {
        "train": order[:n_train],
        "validation": order[n_train : n_train + n_val],
        "heldout_test": order[n_train + n_val :],
    }
    splits: dict[str, dict[str, np.ndarray]] = {}
    for name, ids in split_ids.items():
        splits[name] = {
            "features": features[ids],
            "ground_truth": labels[ids],
            "costs": costs[ids],
            "query_ids": qids[ids],
            "kind": kinds[ids],
        }
        np.savez_compressed(TRAIN_DIR / f"{name}.npz", **splits[name])
    manifest = {
        "run_tag": RUN_TAG,
        "source": scene["mesh_stats"],
        "feature_dim": cfg.feature_dim,
        "row_count": int(idx),
        "positive_count": int(labels.sum()),
        "positive_ratio": float(labels.mean()) if idx else 0.0,
        "splits": {name: int(len(ids)) for name, ids in split_ids.items()},
        "description": "Repeated rifle-shaped assembly candidate rows from ShapeNet rifle mesh segmentation and procedural insertion windows.",
    }
    write_json(TRAIN_DIR / "manifest.json", manifest)
    return splits, manifest


def train_tiny_stpf(splits: dict[str, dict[str, np.ndarray]], cfg: RifleAssemblyConfig) -> dict[str, Any]:
    try:
        import torch
        import torch.nn as nn
    except Exception as exc:
        return {"status": "skipped", "reason": f"torch unavailable: {exc}"}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg.seed)
    model = nn.Sequential(
        nn.Linear(cfg.feature_dim, 96),
        nn.SiLU(),
        nn.Linear(96, 64),
        nn.SiLU(),
        nn.Linear(64, 1),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1.0e-4)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.2], device=device))
    x = torch.from_numpy(splits["train"]["features"]).to(device)
    y = torch.from_numpy(splits["train"]["ground_truth"].astype(np.float32)).to(device).view(-1, 1)
    xv = torch.from_numpy(splits["validation"]["features"]).to(device)
    yv = torch.from_numpy(splits["validation"]["ground_truth"].astype(np.float32)).to(device).view(-1, 1)
    history: list[dict[str, float]] = []
    for epoch in range(cfg.train_epochs):
        perm = torch.randperm(x.shape[0], device=device)
        losses: list[float] = []
        for start in range(0, x.shape[0], cfg.train_batch_size):
            ids = perm[start : start + cfg.train_batch_size]
            pred = model(x[ids])
            loss = loss_fn(pred, y[ids])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        with torch.no_grad():
            prob = torch.sigmoid(model(xv))
            pos = yv > 0.5
            recall = float(((prob[pos] >= 0.5).sum() / torch.clamp(pos.sum(), min=1)).detach().cpu()) if pos.any() else 1.0
        history.append({"epoch": float(epoch + 1), "loss": float(np.mean(losses)), "val_recall_at_0_5": recall})
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    state_path = OUTPUT_DIR / "model_state.pt"
    torch.save({"model": model.state_dict(), "feature_dim": cfg.feature_dim, "history": history}, state_path)
    onnx_path = OUTPUT_DIR / "model.onnx"
    try:
        dummy = torch.zeros(1, cfg.feature_dim, device=device)
        torch.onnx.export(model, dummy, onnx_path, input_names=["features"], output_names=["priority_logit"], opset_version=17)
        onnx_status = "exported"
    except Exception as exc:
        onnx_status = f"failed: {exc}"
    return {
        "status": "ok",
        "device": device,
        "state_path": safe_rel(state_path),
        "onnx_path": safe_rel(onnx_path),
        "onnx_status": onnx_status,
        "history": history,
    }


def learned_scores(features: np.ndarray, train_result: dict[str, Any], cfg: RifleAssemblyConfig) -> np.ndarray:
    try:
        import torch
        import torch.nn as nn
    except Exception:
        return features[:, 8] + 0.1 * features[:, 10]
    state_path = ROOT / train_result.get("state_path", "")
    if not state_path.exists():
        return features[:, 8] + 0.1 * features[:, 10]
    model = nn.Sequential(
        nn.Linear(cfg.feature_dim, 96),
        nn.SiLU(),
        nn.Linear(96, 64),
        nn.SiLU(),
        nn.Linear(64, 1),
    )
    state = torch.load(state_path, map_location="cpu")
    model.load_state_dict(state["model"])
    model.eval()
    scores: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(features), 65536):
            prob = torch.sigmoid(model(torch.from_numpy(features[start : start + 65536]))).numpy().reshape(-1)
            scores.append(prob)
    return np.concatenate(scores, axis=0)


def benchmark_methods(splits: dict[str, dict[str, np.ndarray]], train_result: dict[str, Any], cfg: RifleAssemblyConfig) -> list[dict[str, Any]]:
    held = splits["heldout_test"]
    labels = held["ground_truth"].astype(np.int64)
    costs = held["costs"].astype(np.float64)
    qids = held["query_ids"].astype(np.int64)
    features = held["features"].astype(np.float32)
    total_calls = len(labels)
    total_work = float(costs.sum())
    rng = np.random.default_rng(cfg.seed + 991)
    score_map = {
        "NoProposal": np.zeros(total_calls, dtype=np.float64),
        "RandomSTPF": rng.random(total_calls),
        "RTSTPFExact": learned_scores(features, train_result, cfg),
    }
    rows: list[dict[str, Any]] = []
    unique_groups = np.unique(qids)
    for method, scores in score_map.items():
        if method == "NoProposal":
            rows.append(
                {
                    "method": method,
                    "groups": int(len(unique_groups)),
                    "candidates": int(total_calls),
                    "positive_candidates": int(labels.sum()),
                    "exact_calls": int(total_calls),
                    "skipped_exact_calls": 0,
                    "exact_call_reduction": 0.0,
                    "exact_work": float(total_work),
                    "exact_work_reduction": 0.0,
                    "first_positive_rank_mean": None,
                    "fn": 0,
                    "correctness_rule": "direct all-exact evaluation",
                }
            )
            continue
        exact_calls = 0
        exact_work = 0.0
        fn = 0
        first_positive_rank: list[int] = []
        for group in unique_groups:
            ids = np.flatnonzero(qids == group)
            if method == "NoProposal":
                order = ids
            else:
                order = ids[np.argsort(-scores[ids])]
            group_positive = labels[ids].sum() > 0
            found = False
            for rank, row_id in enumerate(order, start=1):
                exact_calls += 1
                exact_work += float(costs[row_id])
                if labels[row_id] == 1:
                    found = True
                    first_positive_rank.append(rank)
                    break
            if group_positive and not found:
                fn += 1
        rows.append(
            {
                "method": method,
                "groups": int(len(unique_groups)),
                "candidates": int(total_calls),
                "positive_candidates": int(labels.sum()),
                "exact_calls": int(exact_calls),
                "skipped_exact_calls": int(total_calls - exact_calls),
                "exact_call_reduction": float(1.0 - exact_calls / max(1, total_calls)),
                "exact_work": float(exact_work),
                "exact_work_reduction": float(1.0 - exact_work / max(total_work, 1.0e-9)),
                "first_positive_rank_mean": float(np.mean(first_positive_rank)) if first_positive_rank else None,
                "fn": int(fn),
                "correctness_rule": "positive group early-stop; negative/uncertain group conservative all-exact fallback",
            }
        )
    return rows


def write_reports(
    cfg: RifleAssemblyConfig,
    scene: dict[str, Any],
    manifest: dict[str, Any],
    train_result: dict[str, Any],
    bench_rows: list[dict[str, Any]],
    render_paths: dict[str, str],
) -> None:
    benchmark_json = BENCHMARK_DIR / f"{RUN_TAG}.json"
    benchmark_csv = BENCHMARK_DIR / f"{RUN_TAG}.csv"
    benchmark_md = BENCHMARK_DIR / f"{RUN_TAG}.md"
    payload = {
        "run_tag": RUN_TAG,
        "config": asdict(cfg),
        "mesh_stats": scene["mesh_stats"],
        "dataset_manifest": manifest,
        "train_result": train_result,
        "benchmark_rows": bench_rows,
        "render_outputs": render_paths,
    }
    write_json(benchmark_json, payload)
    with benchmark_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(bench_rows[0].keys()))
        writer.writeheader()
        writer.writerows(bench_rows)
    lines = [
        f"# Repeated Rifle-Shaped Assembly Benchmark ({RUN_TAG})",
        "",
        "## Case",
        "",
        "- ShapeNetCore `04090263` rifle mesh is used as the real object silhouette.",
        "- The mesh is PCA-aligned and bbox-segmented into receiver/stock and moving barrel regions; magazine and bolt are procedural insertion proxies.",
        "- The motion repeats barrel insertion, magazine insertion, bolt sliding, and lock/reset windows over multiple cycles.",
        "- This is a rifle-shaped repeated assembly / mechanical insertion analog, not a verified multi-part firearm CAD assembly.",
        "- Contact witnesses are rendered as transparent insertion/rail regions plus cross markers, and correctness is enforced by exact-certificate/fallback semantics.",
        "",
        "## Mesh / Data Audit",
        "",
        f"- Source OBJ: `{scene['mesh_stats']['source_obj']}`.",
        f"- OBJ size: `{scene['mesh_stats']['obj_bytes'] / (1024.0 * 1024.0):.2f}` MB.",
        f"- Vertices / faces: `{scene['mesh_stats']['vertices']}` / `{scene['mesh_stats']['faces']}`.",
        f"- Rendered sampled mesh triangles: static `{scene['mesh_stats']['static_triangles']}`, moving barrel `{scene['mesh_stats']['barrel_triangles']}`.",
        f"- Rendered sampled mesh surface points: static `{scene['mesh_stats']['static_surface_points']}`, moving barrel `{scene['mesh_stats']['barrel_surface_points']}`.",
        f"- Rendered sampled mesh edges: static `{scene['mesh_stats']['static_edges']}`, moving barrel `{scene['mesh_stats']['barrel_edges']}`.",
        f"- Repetition cycles: `{cfg.cycles}`.",
        "",
        "## Training",
        "",
        f"- Training rows: `{manifest['splits']['train']}`.",
        f"- Validation rows: `{manifest['splits']['validation']}`.",
        f"- Heldout rows: `{manifest['splits']['heldout_test']}`.",
        f"- Positive ratio: `{manifest['positive_ratio']:.6f}`.",
        f"- Training status: `{train_result.get('status')}`; device: `{train_result.get('device')}`.",
        "",
        "## Benchmark",
        "",
        "| Method | Groups | Candidates | Exact calls | Call reduction | Work reduction | FN |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in bench_rows:
        lines.append(
            f"| `{row['method']}` | `{row['groups']}` | `{row['candidates']}` | `{row['exact_calls']}` | "
            f"`{100.0 * row['exact_call_reduction']:.4f}%` | `{100.0 * row['exact_work_reduction']:.4f}%` | `{row['fn']}` |"
        )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- Demo MP4: `{render_paths['mp4']}`.",
            f"- Contact sheet: `{render_paths['contact_sheet']}`.",
            f"- Training shard: `{safe_rel(TRAIN_DIR)}`.",
            f"- Model output: `{safe_rel(OUTPUT_DIR)}`.",
            "",
            "## Reproduce",
            "",
            "```powershell",
            "conda activate cudadev",
            "python src/tools/repeated_rifle_assembly_case_run_id.py --train --benchmark --render",
            "```",
        ]
    )
    text = "\n".join(lines) + "\n"
    benchmark_md.write_text(text, encoding="utf-8")
    (MYDEMO_DIR / "case_report.md").write_text(text, encoding="utf-8")
    write_json(MYDEMO_DIR / "metrics.json", payload)
    (MYDEMO_DIR / "run_command.txt").write_text(
        "conda activate cudadev\npython src/tools/repeated_rifle_assembly_case_run_id.py --train --benchmark --render\n",
        encoding="utf-8",
    )
    (MYDEMO_DIR / "resume_notes.md").write_text(
        "# Resume Notes\n\nRe-run `run_command.txt`. Outputs are deterministic for the configured seed.\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    if not (args.train or args.benchmark or args.render):
        args.train = args.benchmark = args.render = True
    ensure_dirs()
    cfg = RifleAssemblyConfig()
    t0 = time.perf_counter()
    scene = build_scene(cfg)
    render_paths: dict[str, str] = {}
    if args.render:
        render_paths = render_outputs(scene, cfg)
    splits, manifest = build_candidate_dataset(scene, cfg)
    train_result: dict[str, Any] = {"status": "skipped"}
    if args.train:
        train_result = train_tiny_stpf(splits, cfg)
    bench_rows: list[dict[str, Any]] = []
    if args.benchmark:
        bench_rows = benchmark_methods(splits, train_result, cfg)
    write_reports(cfg, scene, manifest, train_result, bench_rows, render_paths)
    elapsed = time.perf_counter() - t0
    print(json.dumps({"run_tag": RUN_TAG, "elapsed_seconds": elapsed, "mydemo_dir": safe_rel(MYDEMO_DIR)}, indent=2))


if __name__ == "__main__":
    main()
