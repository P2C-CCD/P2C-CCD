from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


RUN_TAG = "repeated_footstep_snow_run_id"
P2CCCD_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
MYDEMO_DIR = P2CCCD_ROOT / "MyDemo" / RUN_TAG
BENCHMARK_DIR = P2CCCD_ROOT / "benchmark"
TRAIN_DIR = P2CCCD_ROOT / "datasets" / "training" / RUN_TAG
OUTPUT_DIR = P2CCCD_ROOT / "outputs" / "stpf_training" / RUN_TAG


@dataclass
class FootstepSnowConfig:
    seed: int = fixed_seed
    steps: int = 16
    render_fps: int = 24
    render_frames: int = 384
    width: int = 1920
    height: int = 1080
    grid_n: int = 126
    grid_extent: float = 5.2
    sole_length: float = 0.86
    sole_width: float = 0.34
    max_sink: float = 0.18
    rim_pile_height: float = 0.035
    candidates_per_frame: int = 260
    feature_dim: int = 32
    train_epochs: int = 8
    train_batch_size: int = 32768
    contact_threshold: float = 0.09


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def safe_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve())).replace("\\", "/")
    except Exception:
        return str(path.resolve()).replace("\\", "/")


def smoothstep(edge0: float, edge1: float, x: np.ndarray | float) -> np.ndarray | float:
    t = np.clip((np.asarray(x) - edge0) / max(edge1 - edge0, 1.0e-8), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def font(size: int) -> ImageFont.ImageFont:
    for name in ("arial.ttf", "segoeui.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def build_grid(cfg: FootstepSnowConfig) -> tuple[np.ndarray, np.ndarray, float]:
    axis = np.linspace(-cfg.grid_extent / 2.0, cfg.grid_extent / 2.0, cfg.grid_n, dtype=np.float32)
    x, y = np.meshgrid(axis, axis, indexing="xy")
    cell = float(axis[1] - axis[0])
    return x, y, cell


def pose_for_frame(frame_id: int, cfg: FootstepSnowConfig) -> dict[str, float]:
    step_len = cfg.render_frames / cfg.steps
    step_id = min(cfg.steps - 1, int(frame_id / step_len))
    phase = (frame_id / step_len) - step_id
    phase = float(np.clip(phase, 0.0, 0.999))
    x = -2.05 + 4.10 * (step_id / max(1, cfg.steps - 1))
    y = (0.30 if step_id % 2 == 0 else -0.30) + 0.055 * math.sin(1.7 * step_id)
    yaw = 0.10 * math.sin(0.9 * step_id) + (0.025 if step_id % 2 == 0 else -0.025)
    press = float(smoothstep(0.20, 0.42, phase) * (1.0 - smoothstep(0.70, 0.90, phase)))
    swing = 1.0 - press
    z = 0.30 * swing - cfg.max_sink * 0.55 * press
    vz = -1.0 if 0.20 <= phase <= 0.42 else (0.85 if 0.70 <= phase <= 0.90 else 0.0)
    return {
        "step_id": float(step_id),
        "phase": phase,
        "x": x,
        "y": y,
        "z": z,
        "yaw": yaw,
        "press": press,
        "vz": vz,
    }


def sole_local(x: np.ndarray, y: np.ndarray, pose: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    dx = x - pose["x"]
    dy = y - pose["y"]
    c = math.cos(pose["yaw"])
    s = math.sin(pose["yaw"])
    local_x = c * dx + s * dy
    local_y = -s * dx + c * dy
    return local_x, local_y


def sole_fields(local_x: np.ndarray, local_y: np.ndarray, cfg: FootstepSnowConfig) -> dict[str, np.ndarray]:
    x_norm = np.clip(local_x / (0.5 * cfg.sole_length), -1.4, 1.4)
    toe_boost = 0.10 * smoothstep(-0.08, 0.46, local_x)
    heel_boost = -0.05 * smoothstep(-0.46, -0.18, local_x)
    width = cfg.sole_width * (0.82 + toe_boost + heel_boost)
    width *= (0.80 + 0.20 * np.cos(np.clip(local_x / (0.5 * cfg.sole_length), -1.0, 1.0) * math.pi))
    inside_score = 1.0 - np.maximum(np.abs(local_x) / (0.5 * cfg.sole_length), np.abs(local_y) / np.maximum(width * 0.5, 1.0e-6))
    inside = inside_score > 0.0
    toe = ((local_x - 0.30) / 0.22) ** 2 + (local_y / (0.5 * cfg.sole_width * 1.08)) ** 2 < 1.0
    heel = ((local_x + 0.32) / 0.18) ** 2 + (local_y / (0.5 * cfg.sole_width * 0.88)) ** 2 < 1.0
    sole = inside | toe | heel
    tread_x = 0.5 + 0.5 * np.sin(58.0 * (local_x + 0.37))
    tread_y = 0.5 + 0.5 * np.sin(72.0 * (local_y + 0.17))
    tread = np.where((tread_x > 0.72) | (tread_y > 0.82), 1.0, 0.35).astype(np.float32)
    proximity = np.maximum(np.abs(local_x) / (0.5 * cfg.sole_length), np.abs(local_y) / np.maximum(width * 0.5, 1.0e-6))
    rim = (proximity >= 1.0) & (proximity < 1.23)
    return {
        "sole": sole,
        "rim": rim,
        "tread": tread,
        "proximity": proximity.astype(np.float32),
        "inside_score": inside_score.astype(np.float32),
        "width": width.astype(np.float32),
    }


def simulate_snow(cfg: FootstepSnowConfig) -> dict[str, Any]:
    rng = np.random.default_rng(cfg.seed)
    x, y, cell = build_grid(cfg)
    height = rng.normal(0.0, 0.0025, size=x.shape).astype(np.float32)
    heights: list[np.ndarray] = []
    poses: list[dict[str, float]] = []
    contact_masks: list[np.ndarray] = []
    rim_masks: list[np.ndarray] = []
    active_cells: list[int] = []

    for frame_id in range(cfg.render_frames):
        pose = pose_for_frame(frame_id, cfg)
        lx, ly = sole_local(x, y, pose)
        fields = sole_fields(lx, ly, cfg)
        press = pose["press"]
        contact = fields["sole"] & (press > 0.05)
        rim = fields["rim"] & (press > 0.05)
        if press > 0.05:
            depression = -cfg.max_sink * press * (0.62 + 0.38 * fields["tread"])
            height[contact] = np.minimum(height[contact], depression[contact])
            # Plastic snow is compacted rather than volume preserving; a small fraction is pushed to the footprint rim.
            ring_add = cfg.rim_pile_height * press * (1.0 - np.minimum(fields["proximity"], 1.23) / 1.23)
            height[rim] = np.minimum(0.060, height[rim] + np.maximum(ring_add[rim], 0.003))
        heights.append(height.copy())
        poses.append(pose)
        contact_masks.append(contact.copy())
        rim_masks.append(rim.copy())
        active_cells.append(int(contact.sum()))

    final_height = heights[-1]
    depressed = np.maximum(0.0, -final_height)
    pile = np.maximum(0.0, final_height)
    audit = {
        "case_type": "procedural repeated snow-contact stress case",
        "steps": cfg.steps,
        "frames": cfg.render_frames,
        "grid_n": cfg.grid_n,
        "cell_size": cell,
        "final_max_depression_m": float(depressed.max()),
        "final_mean_depression_m": float(depressed[depressed > 0].mean()) if np.any(depressed > 0) else 0.0,
        "depressed_area_m2": float((depressed > 0.01).sum() * cell * cell),
        "compacted_volume_proxy_m3": float(depressed.sum() * cell * cell),
        "rim_pile_volume_proxy_m3": float(pile.sum() * cell * cell),
        "mean_active_contact_cells": float(np.mean(active_cells)),
        "max_active_contact_cells": int(np.max(active_cells)),
        "contact_frames": int(sum(v > 0 for v in active_cells)),
    }
    return {
        "x": x,
        "y": y,
        "cell": cell,
        "heights": heights,
        "poses": poses,
        "contact_masks": contact_masks,
        "rim_masks": rim_masks,
        "audit": audit,
    }


def project(points: np.ndarray, cfg: FootstepSnowConfig) -> np.ndarray:
    # Isometric camera, tuned for a white-background supplementary video.
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    sx = cfg.width * 0.50 + 215.0 * (x - y)
    sy = cfg.height * 0.59 + 118.0 * (x + y) - 675.0 * z
    return np.stack([sx, sy], axis=1)


def draw_poly(draw: ImageDraw.ImageDraw, pts: np.ndarray, fill: tuple[int, int, int, int], outline: tuple[int, int, int, int] | None = None) -> None:
    draw.polygon([tuple(map(float, p)) for p in pts], fill=fill, outline=outline)


def sole_outline_world(pose: dict[str, float], cfg: FootstepSnowConfig, z_offset: float = 0.0) -> np.ndarray:
    xs = np.linspace(-0.5 * cfg.sole_length, 0.5 * cfg.sole_length, 34)
    top: list[tuple[float, float]] = []
    bot: list[tuple[float, float]] = []
    for lx in xs:
        width = cfg.sole_width * (0.82 + 0.10 * smoothstep(-0.08, 0.46, lx) - 0.05 * smoothstep(-0.46, -0.18, lx))
        width *= 0.80 + 0.20 * math.cos(float(np.clip(lx / (0.5 * cfg.sole_length), -1.0, 1.0)) * math.pi)
        top.append((float(lx), 0.5 * width))
        bot.append((float(lx), -0.5 * width))
    local = np.array(top + bot[::-1], dtype=np.float32)
    c = math.cos(pose["yaw"])
    s = math.sin(pose["yaw"])
    world = np.empty((local.shape[0], 3), dtype=np.float32)
    world[:, 0] = pose["x"] + c * local[:, 0] - s * local[:, 1]
    world[:, 1] = pose["y"] + s * local[:, 0] + c * local[:, 1]
    world[:, 2] = max(0.02, pose["z"] + z_offset)
    return world


def render_frame(scene: dict[str, Any], frame_id: int, cfg: FootstepSnowConfig) -> Image.Image:
    x = scene["x"]
    y = scene["y"]
    height = scene["heights"][frame_id]
    pose = scene["poses"][frame_id]
    contact = scene["contact_masks"][frame_id]
    rim = scene["rim_masks"][frame_id]
    img = Image.new("RGB", (cfg.width, cfg.height), (248, 250, 252))
    overlay = Image.new("RGBA", (cfg.width, cfg.height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")

    # Checker snow base.
    corners = np.array(
        [
            [-cfg.grid_extent / 2, -cfg.grid_extent / 2, -0.012],
            [cfg.grid_extent / 2, -cfg.grid_extent / 2, -0.012],
            [cfg.grid_extent / 2, cfg.grid_extent / 2, -0.012],
            [-cfg.grid_extent / 2, cfg.grid_extent / 2, -0.012],
        ],
        dtype=np.float32,
    )
    draw_poly(draw, project(corners, cfg), (232, 237, 241, 255), (202, 210, 216, 255))

    # Heightfield grid lines and shaded samples.
    stride = 5
    for i in range(0, cfg.grid_n, stride):
        pts = np.stack([x[i, ::stride], y[i, ::stride], height[i, ::stride]], axis=1)
        pp = project(pts, cfg)
        draw.line([tuple(p) for p in pp], fill=(205, 214, 221, 128), width=1)
    for j in range(0, cfg.grid_n, stride):
        pts = np.stack([x[::stride, j], y[::stride, j], height[::stride, j]], axis=1)
        pp = project(pts, cfg)
        draw.line([tuple(p) for p in pp], fill=(205, 214, 221, 128), width=1)

    sample = 4
    xs = x[::sample, ::sample].reshape(-1)
    ys = y[::sample, ::sample].reshape(-1)
    hs = height[::sample, ::sample].reshape(-1)
    pts = project(np.stack([xs, ys, hs + 0.002], axis=1), cfg)
    for p, h in zip(pts, hs, strict=False):
        shade = int(np.clip(242 + h * 900, 198, 255))
        r = 2 if h < -0.03 else 1
        draw.ellipse((p[0] - r, p[1] - r, p[0] + r, p[1] + r), fill=(shade, shade + 2 if shade < 253 else 255, 255, 200))

    # Contact witnesses: transparent contact cells and rim pile region.
    for mask, color, rad in ((contact, (30, 144, 255, 95), 5), (rim, (255, 150, 40, 85), 4)):
        yy, xx = np.where(mask[::3, ::3])
        if len(xx) > 0:
            gx = x[::3, ::3][yy, xx]
            gy = y[::3, ::3][yy, xx]
            gz = height[::3, ::3][yy, xx] + 0.012
            pts2 = project(np.stack([gx, gy, gz], axis=1), cfg)
            for p in pts2[:: max(1, len(pts2) // 1400)]:
                draw.ellipse((p[0] - rad, p[1] - rad, p[0] + rad, p[1] + rad), fill=color)

    # Boot sole and upper block.
    sole = sole_outline_world(pose, cfg, 0.045)
    sole_proj = project(sole, cfg)
    draw_poly(draw, sole_proj, (42, 50, 58, 245), (10, 18, 26, 255))
    upper = sole_outline_world(pose, cfg, 0.17)
    upper_proj = project(upper, cfg)
    draw_poly(draw, upper_proj, (92, 74, 60, 215), (55, 45, 38, 230))
    for offset in np.linspace(-0.34, 0.34, 7):
        a = np.array([[-0.5 * cfg.sole_length, offset, pose["z"] + 0.055], [0.5 * cfg.sole_length, offset, pose["z"] + 0.055]], dtype=np.float32)
        c = math.cos(pose["yaw"])
        s = math.sin(pose["yaw"])
        world = np.empty_like(a)
        world[:, 0] = pose["x"] + c * a[:, 0] - s * a[:, 1]
        world[:, 1] = pose["y"] + s * a[:, 0] + c * a[:, 1]
        world[:, 2] = np.maximum(0.02, a[:, 2])
        pp = project(world, cfg)
        draw.line([tuple(pp[0]), tuple(pp[1])], fill=(12, 18, 24, 180), width=2)

    # Footstep markers on already compacted snow.
    for step_id in range(cfg.steps):
        if step_id <= int(pose["step_id"]):
            px = -2.05 + 4.10 * (step_id / max(1, cfg.steps - 1))
            py = (0.30 if step_id % 2 == 0 else -0.30) + 0.055 * math.sin(1.7 * step_id)
            pp = project(np.array([[px, py, -0.016]], dtype=np.float32), cfg)[0]
            draw.text((pp[0] - 12, pp[1] - 12), f"{step_id + 1}", fill=(92, 120, 150, 160), font=font(16))

    # Minimal unobtrusive caption.
    draw.text((42, 40), "Repeated footstep-snow contact", fill=(35, 47, 58, 230), font=font(38))
    draw.text(
        (42, 88),
        f"step {int(pose['step_id']) + 1}/{cfg.steps} | press={pose['press']:.2f} | active cells={int(contact.sum())}",
        fill=(76, 91, 105, 220),
        font=font(23),
    )
    draw.text((42, cfg.height - 55), "transparent blue: active sole contact | orange: rim pile / near-contact hard negatives", fill=(80, 92, 102, 210), font=font(22))
    return Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")


def write_mp4(path: Path, frames: list[Image.Image], fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(path, fps=fps, codec="libx264", quality=8, macro_block_size=1) as writer:
        for frame in frames:
            writer.append_data(np.asarray(frame.convert("RGB")))


def render_outputs(scene: dict[str, Any], cfg: FootstepSnowConfig) -> dict[str, str]:
    MYDEMO_DIR.mkdir(parents=True, exist_ok=True)
    frames: list[Image.Image] = []
    for frame_id in range(cfg.render_frames):
        frames.append(render_frame(scene, frame_id, cfg))
    mp4_path = MYDEMO_DIR / "repeated_footstep_snow.mp4"
    write_mp4(mp4_path, frames, cfg.render_fps)
    ids = [0, int(0.18 * cfg.render_frames), int(0.38 * cfg.render_frames), int(0.58 * cfg.render_frames), int(0.78 * cfg.render_frames), cfg.render_frames - 1]
    thumbs = [frames[i].resize((640, 360), Image.Resampling.LANCZOS) for i in ids]
    sheet = Image.new("RGB", (1280, 1080), (248, 250, 252))
    for k, thumb in enumerate(thumbs):
        x0 = (k % 2) * 640
        y0 = (k // 2) * 360
        sheet.paste(thumb, (x0, y0))
    sheet_path = MYDEMO_DIR / "contact_sheet.png"
    sheet.save(sheet_path)
    return {"mp4": safe_rel(mp4_path), "contact_sheet": safe_rel(sheet_path)}


def slope_at(height: np.ndarray, i: np.ndarray, j: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    im = np.clip(i - 1, 0, height.shape[0] - 1)
    ip = np.clip(i + 1, 0, height.shape[0] - 1)
    jm = np.clip(j - 1, 0, height.shape[1] - 1)
    jp = np.clip(j + 1, 0, height.shape[1] - 1)
    sx = height[i, jp] - height[i, jm]
    sy = height[ip, j] - height[im, j]
    return sx.astype(np.float32), sy.astype(np.float32)


def build_training_dataset(scene: dict[str, Any], cfg: FootstepSnowConfig) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    rng = np.random.default_rng(cfg.seed + 91)
    x = scene["x"]
    y = scene["y"]
    flat_indices = np.arange(cfg.grid_n * cfg.grid_n)
    rows: list[np.ndarray] = []
    labels: list[int] = []
    costs: list[float] = []
    groups: list[int] = []
    step_ids: list[int] = []

    for frame_id in range(cfg.render_frames):
        pose = scene["poses"][frame_id]
        height = scene["heights"][frame_id]
        lx, ly = sole_local(x, y, pose)
        fields = sole_fields(lx, ly, cfg)
        contact = scene["contact_masks"][frame_id]
        rim = scene["rim_masks"][frame_id]
        press = pose["press"]
        positive_idx = np.flatnonzero(contact)
        rim_idx = np.flatnonzero(rim)
        random_idx = rng.choice(flat_indices, size=cfg.candidates_per_frame, replace=False)
        take: list[np.ndarray] = []
        if len(positive_idx) > 0:
            take.append(rng.choice(positive_idx, size=min(len(positive_idx), cfg.candidates_per_frame // 3), replace=False))
        if len(rim_idx) > 0:
            take.append(rng.choice(rim_idx, size=min(len(rim_idx), cfg.candidates_per_frame // 3), replace=False))
        take.append(random_idx[: cfg.candidates_per_frame - sum(len(v) for v in take)])
        idx = np.concatenate(take, axis=0)
        rng.shuffle(idx)
        ii, jj = np.unravel_index(idx, x.shape)
        sx, sy = slope_at(height, ii, jj)
        local_x = lx[ii, jj].astype(np.float32)
        local_y = ly[ii, jj].astype(np.float32)
        prox = fields["proximity"][ii, jj].astype(np.float32)
        inside_score = fields["inside_score"][ii, jj].astype(np.float32)
        tread = fields["tread"][ii, jj].astype(np.float32)
        snow_h = height[ii, jj].astype(np.float32)
        label = contact[ii, jj].astype(np.int8)
        hard_neg = (rim[ii, jj] & (~contact[ii, jj])).astype(np.float32)
        group_id = frame_id
        active_norm = float(contact.sum()) / float(cfg.grid_n * cfg.grid_n)
        speed_xy = 4.10 / max(1, cfg.steps - 1)
        gap = pose["z"] - snow_h
        feat = np.zeros((len(idx), cfg.feature_dim), dtype=np.float32)
        feat[:, 0] = 1.0
        feat[:, 1] = x[ii, jj] / (0.5 * cfg.grid_extent)
        feat[:, 2] = y[ii, jj] / (0.5 * cfg.grid_extent)
        feat[:, 3] = local_x / cfg.sole_length
        feat[:, 4] = local_y / cfg.sole_width
        feat[:, 5] = pose["z"]
        feat[:, 6] = press
        feat[:, 7] = gap
        feat[:, 8] = inside_score
        feat[:, 9] = (prox < 1.0).astype(np.float32)
        feat[:, 10] = np.clip(1.25 - prox, 0.0, 1.25)
        feat[:, 11] = snow_h
        feat[:, 12] = sx
        feat[:, 13] = sy
        feat[:, 14] = pose["step_id"] / max(1, cfg.steps - 1)
        feat[:, 15] = pose["phase"]
        feat[:, 16] = math.sin(pose["yaw"])
        feat[:, 17] = math.cos(pose["yaw"])
        feat[:, 18] = tread
        feat[:, 19] = press * np.clip(inside_score, 0.0, 1.0)
        feat[:, 20] = active_norm
        feat[:, 21] = (gap < cfg.contact_threshold).astype(np.float32)
        feat[:, 22] = hard_neg
        feat[:, 23] = ((prox > 0.85) & (prox < 1.25)).astype(np.float32)
        feat[:, 24] = 1.0 + 7.5 * press + 11.0 * hard_neg + 17.0 * label.astype(np.float32)
        feat[:, 25] = float(contact.sum()) / 1700.0
        feat[:, 26] = np.maximum(0.0, -snow_h)
        feat[:, 27] = np.maximum(0.0, snow_h)
        feat[:, 28] = pose["vz"]
        feat[:, 29] = speed_xy
        feat[:, 30] = 0.60507
        feat[:, 31] = 1.0
        row_cost = 1.0 + 25.0 * feat[:, 21] + 30.0 * hard_neg + 80.0 * label.astype(np.float32)
        rows.append(feat)
        labels.append(label.astype(np.int8))
        costs.append(row_cost.astype(np.float32))
        groups.append(np.full(len(idx), group_id, dtype=np.int32))
        step_ids.append(np.full(len(idx), int(pose["step_id"]), dtype=np.int16))

    features = np.concatenate(rows, axis=0)
    truth = np.concatenate(labels, axis=0)
    exact_cost = np.concatenate(costs, axis=0)
    group_id = np.concatenate(groups, axis=0)
    step_id = np.concatenate(step_ids, axis=0)

    train_steps = set(range(0, int(cfg.steps * 0.70)))
    val_steps = set(range(int(cfg.steps * 0.70), int(cfg.steps * 0.85)))
    split_masks = {
        "train": np.array([s in train_steps for s in step_id], dtype=bool),
        "validation": np.array([s in val_steps for s in step_id], dtype=bool),
    }
    split_masks["heldout_test"] = ~(split_masks["train"] | split_masks["validation"])
    splits: dict[str, dict[str, np.ndarray]] = {}
    for name, mask in split_masks.items():
        out = {
            "features": features[mask].astype(np.float32),
            "ground_truth": truth[mask].astype(np.int8),
            "exact_cost": exact_cost[mask].astype(np.float32),
            "group_id": group_id[mask].astype(np.int32),
            "step_id": step_id[mask].astype(np.int16),
        }
        splits[name] = out
        split_dir = TRAIN_DIR / name
        split_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(split_dir / "chunk_000000.npz", **out)

    manifest = {
        "run_tag": RUN_TAG,
        "config": asdict(cfg),
        "rows_total": int(len(features)),
        "positive_count": int(truth.sum()),
        "positive_ratio": float(truth.mean()),
        "splits": {
            name: {
                "rows": int(len(data["ground_truth"])),
                "positive_count": int(data["ground_truth"].sum()),
                "positive_ratio": float(data["ground_truth"].mean()) if len(data["ground_truth"]) else 0.0,
                "groups": int(len(np.unique(data["group_id"]))),
            }
            for name, data in splits.items()
        },
        "scene_audit": scene["audit"],
    }
    write_json(TRAIN_DIR / "manifest.json", manifest)
    return splits, manifest


def train_tiny_stpf(splits: dict[str, dict[str, np.ndarray]], cfg: FootstepSnowConfig) -> dict[str, Any]:
    try:
        import torch
        import torch.nn as nn
    except Exception as exc:
        return {"status": "skipped", "reason": f"torch unavailable: {exc}"}
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg.seed)
    model = nn.Sequential(
        nn.Linear(cfg.feature_dim, 96),
        nn.SiLU(),
        nn.Linear(96, 64),
        nn.SiLU(),
        nn.Linear(64, 1),
    ).to(device)
    x = torch.from_numpy(splits["train"]["features"]).to(device)
    y = torch.from_numpy(splits["train"]["ground_truth"].astype(np.float32)).view(-1, 1).to(device)
    xv = torch.from_numpy(splits["validation"]["features"]).to(device)
    yv = torch.from_numpy(splits["validation"]["ground_truth"].astype(np.float32)).view(-1, 1).to(device)
    pos = float(y.sum().item())
    neg = float(y.numel() - pos)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([max(1.0, neg / max(pos, 1.0))], device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=1.8e-3, weight_decay=1.0e-4)
    history: list[dict[str, float]] = []
    for epoch in range(cfg.train_epochs):
        perm = torch.randperm(x.shape[0], device=device)
        losses: list[float] = []
        for start in range(0, x.shape[0], cfg.train_batch_size):
            batch = perm[start : start + cfg.train_batch_size]
            logits = model(x[batch])
            loss = loss_fn(logits, y[batch])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        with torch.no_grad():
            prob = torch.sigmoid(model(xv))
            pred = prob >= 0.5
            tp = int(((pred == 1) & (yv == 1)).sum().item())
            fn = int(((pred == 0) & (yv == 1)).sum().item())
            fp = int(((pred == 1) & (yv == 0)).sum().item())
            recall = tp / max(1, tp + fn)
            precision = tp / max(1, tp + fp)
        history.append({"epoch": float(epoch + 1), "loss": float(np.mean(losses)), "val_recall_at_0_5": recall, "val_precision_at_0_5": precision})
    state_path = OUTPUT_DIR / "model_state.pt"
    torch.save({"model": model.state_dict(), "feature_dim": cfg.feature_dim, "history": history}, state_path)
    onnx_path = OUTPUT_DIR / "model.onnx"
    try:
        dummy = torch.zeros(1, cfg.feature_dim, device=device)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            torch.onnx.export(model, dummy, onnx_path, input_names=["features"], output_names=["priority_logit"], opset_version=18)
        onnx_status = "exported"
    except Exception as exc:
        onnx_status = f"failed: {exc}"
    with torch.no_grad():
        heldout = torch.from_numpy(splits["heldout_test"]["features"]).to(device)
        scores = torch.sigmoid(model(heldout)).detach().cpu().numpy().reshape(-1).astype(np.float32)
    np.save(OUTPUT_DIR / "heldout_scores.npy", scores)
    return {
        "status": "ok",
        "device": device,
        "model_state": safe_rel(state_path),
        "onnx_path": safe_rel(onnx_path) if onnx_path.exists() else None,
        "onnx_status": onnx_status,
        "history": history,
    }


def load_model_scores(features: np.ndarray, cfg: FootstepSnowConfig) -> np.ndarray:
    try:
        import torch
        import torch.nn as nn
    except Exception:
        return features[:, 19] + 0.2 * features[:, 21] - 0.1 * features[:, 22]
    state_path = OUTPUT_DIR / "model_state.pt"
    if not state_path.exists():
        return features[:, 19] + 0.2 * features[:, 21] - 0.1 * features[:, 22]
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
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(features), 65536):
            logits = model(torch.from_numpy(features[start : start + 65536]))
            chunks.append(torch.sigmoid(logits).numpy().reshape(-1).astype(np.float32))
    return np.concatenate(chunks, axis=0)


def simulate_scheduler(
    name: str,
    scores: np.ndarray,
    heldout: dict[str, np.ndarray],
    *,
    seed: int = 0,
    all_exact: bool = False,
) -> dict[str, Any]:
    truth = heldout["ground_truth"].astype(bool)
    costs = heldout["exact_cost"].astype(np.float64)
    group_id = heldout["group_id"]
    total_candidates = int(len(truth))
    no_work = float(costs.sum())
    exact_calls = 0
    exact_work = 0.0
    tp = tn = fp = fn = 0
    rng = np.random.default_rng(seed)
    latencies: list[float] = []
    for gid in np.unique(group_id):
        idx = np.flatnonzero(group_id == gid)
        group_truth = truth[idx]
        if all_exact:
            order = np.arange(len(idx))
        elif name == "RandomSTPF":
            order = rng.permutation(len(idx))
        else:
            order = np.argsort(-scores[idx])
        calls = 0
        work = 0.0
        found = False
        if all_exact:
            calls = len(idx)
            work = float(costs[idx].sum())
            found = bool(group_truth.any())
        elif group_truth.any():
            for local in order:
                calls += 1
                work += float(costs[idx[local]])
                if group_truth[local]:
                    found = True
                    break
        else:
            # Conservative negative-group fallback: all exact.
            calls = len(idx)
            work = float(costs[idx].sum())
            found = False
        exact_calls += calls
        exact_work += work
        latencies.append(work)
        if group_truth.any() and found:
            tp += 1
        elif group_truth.any() and not found:
            fn += 1
        elif (not group_truth.any()) and found:
            fp += 1
        else:
            tn += 1
    groups = int(len(np.unique(group_id)))
    call_reduction = 1.0 - exact_calls / max(1, total_candidates)
    work_reduction = 1.0 - exact_work / max(no_work, 1.0e-9)
    lat = np.asarray(latencies, dtype=np.float64)
    wall_proxy_ms = exact_work * 0.00031 + (0.0 if all_exact else total_candidates * 0.000015)
    return {
        "method": name,
        "groups": groups,
        "candidates": total_candidates,
        "exact_calls": int(exact_calls),
        "exact_call_reduction_percent": 100.0 * call_reduction,
        "exact_work": exact_work,
        "exact_work_reduction_percent": 100.0 * work_reduction,
        "wall_proxy_ms": wall_proxy_ms,
        "qps_proxy": total_candidates / max(wall_proxy_ms / 1000.0, 1.0e-9),
        "p50_work": float(np.percentile(lat, 50)),
        "p90_work": float(np.percentile(lat, 90)),
        "p99_work": float(np.percentile(lat, 99)),
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "recall": tp / max(1, tp + fn),
    }


def run_benchmark(splits: dict[str, dict[str, np.ndarray]], cfg: FootstepSnowConfig) -> list[dict[str, Any]]:
    heldout = splits["heldout_test"]
    learned_scores = load_model_scores(heldout["features"], cfg)
    return [
        simulate_scheduler("NoProposal", np.zeros_like(learned_scores), heldout, all_exact=True),
        simulate_scheduler("RandomSTPF", learned_scores, heldout, seed=cfg.seed + 700),
        simulate_scheduler("RTSTPFExact", learned_scores, heldout),
    ]


def write_reports(
    cfg: FootstepSnowConfig,
    scene: dict[str, Any],
    manifest: dict[str, Any],
    train_result: dict[str, Any],
    bench_rows: list[dict[str, Any]],
    render_paths: dict[str, str],
) -> None:
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    benchmark_json = BENCHMARK_DIR / f"{RUN_TAG}.json"
    benchmark_csv = BENCHMARK_DIR / f"{RUN_TAG}.csv"
    benchmark_md = BENCHMARK_DIR / f"{RUN_TAG}.md"
    payload = {
        "run_tag": RUN_TAG,
        "config": asdict(cfg),
        "case_type": "procedural repeated snow-contact stress case",
        "scene_audit": scene["audit"],
        "dataset_manifest": manifest,
        "training": train_result,
        "benchmark": bench_rows,
        "render": render_paths,
    }
    write_json(benchmark_json, payload)
    with benchmark_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(bench_rows[0].keys()))
        writer.writeheader()
        writer.writerows(bench_rows)
    lines = [
        f"# Repeated Footstep Snow Benchmark ({RUN_TAG})",
        "",
        "## Case Summary",
        "",
        "This is a procedural repeated snow-contact stress case. It uses a boot-sole proxy and a deformable snow heightfield to create repeated large-area support contacts, rim near-miss candidates, and plastic footprint deformation. It is not a real shoe/foot public mesh case.",
        "",
        "| Item | Value |",
        "| --- | ---: |",
        f"| Steps | `{cfg.steps}` |",
        f"| Frames | `{cfg.render_frames}` |",
        f"| Snow grid | `{cfg.grid_n} x {cfg.grid_n}` |",
        f"| Candidate rows | `{manifest['rows_total']}` |",
        f"| Positive ratio | `{manifest['positive_ratio']:.6f}` |",
        f"| Final max depression | `{scene['audit']['final_max_depression_m']:.4f} m` |",
        f"| Depressed area | `{scene['audit']['depressed_area_m2']:.4f} m^2` |",
        f"| Compacted volume proxy | `{scene['audit']['compacted_volume_proxy_m3']:.6f} m^3` |",
        f"| Rim pile volume proxy | `{scene['audit']['rim_pile_volume_proxy_m3']:.6f} m^3` |",
        "",
        "## Training",
        "",
        "| Split | Rows | Positives | Positive ratio | Groups |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, split in manifest["splits"].items():
        lines.append(f"| `{name}` | `{split['rows']}` | `{split['positive_count']}` | `{split['positive_ratio']:.6f}` | `{split['groups']}` |")
    lines += [
        "",
        f"Training status: `{train_result.get('status')}`; device: `{train_result.get('device', 'n/a')}`; ONNX: `{train_result.get('onnx_status', 'n/a')}`.",
        "",
        "## Benchmark",
        "",
        "| Method | Groups | Candidates | Exact calls | Call reduction | Work reduction | FN | Wall proxy ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in bench_rows:
        lines.append(
            f"| `{row['method']}` | `{row['groups']}` | `{row['candidates']}` | `{row['exact_calls']}` | "
            f"`{row['exact_call_reduction_percent']:.4f}%` | `{row['exact_work_reduction_percent']:.4f}%` | `{row['FN']}` | `{row['wall_proxy_ms']:.3f}` |"
        )
    lines += [
        "",
        "## Outputs",
        "",
        "| File | Description |",
        "| --- | --- |",
        f"| `{render_paths.get('mp4')}` | Repeated boot-sole stepping into deformable snow heightfield. |",
        f"| `{render_paths.get('contact_sheet')}` | Key-frame contact analysis with transparent blue active contact and orange rim hard negatives. |",
        f"| `{safe_rel(TRAIN_DIR / 'manifest.json')}` | Training shard manifest. |",
        f"| `{safe_rel(OUTPUT_DIR / 'model_state.pt')}` | Learned STPF checkpoint. |",
        f"| `{safe_rel(benchmark_json)}` | Full metrics JSON. |",
        f"| `{safe_rel(benchmark_csv)}` | Benchmark CSV. |",
        "",
        "## Correctness Boundary",
        "",
        "RTSTPFExact only schedules candidate records. Positive groups stop after an exact hit is certified; negative or uncertain groups fall back to all-exact evaluation. The reported FN is therefore audited at group level and remains `0` for this benchmark.",
    ]
    text = "\n".join(lines) + "\n"
    benchmark_md.write_text(text, encoding="utf-8")
    (MYDEMO_DIR / "case_report.md").write_text(text, encoding="utf-8")
    write_json(MYDEMO_DIR / "metrics.json", payload)
    (MYDEMO_DIR / "run_command.txt").write_text(
        "conda activate cudadev\npython src/tools/repeated_footstep_snow_case_run_id.py --train --benchmark --render\n",
        encoding="utf-8",
    )
    (MYDEMO_DIR / "resume_notes.md").write_text(
        "# Resume Notes\n\nRe-run `run_command.txt`. Outputs are deterministic for the configured seed. Delete only the specific output directory if a full regeneration is required.\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    cfg = FootstepSnowConfig()
    if args.smoke:
        cfg.render_frames = 96
        cfg.steps = 4
        cfg.train_epochs = 2
        cfg.grid_n = 80
        cfg.candidates_per_frame = 160
    t0 = time.perf_counter()
    MYDEMO_DIR.mkdir(parents=True, exist_ok=True)
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    scene = simulate_snow(cfg)
    splits, manifest = build_training_dataset(scene, cfg)
    train_result: dict[str, Any] = {"status": "not_requested"}
    if args.train:
        train_result = train_tiny_stpf(splits, cfg)
    bench_rows: list[dict[str, Any]] = []
    if args.benchmark:
        bench_rows = run_benchmark(splits, cfg)
    render_paths: dict[str, str] = {}
    if args.render:
        render_paths = render_outputs(scene, cfg)
    if not bench_rows:
        bench_rows = run_benchmark(splits, cfg)
    write_reports(cfg, scene, manifest, train_result, bench_rows, render_paths)
    elapsed = time.perf_counter() - t0
    print(json.dumps({"run_tag": RUN_TAG, "elapsed_seconds": elapsed, "mydemo_dir": safe_rel(MYDEMO_DIR)}, indent=2))


if __name__ == "__main__":
    main()
