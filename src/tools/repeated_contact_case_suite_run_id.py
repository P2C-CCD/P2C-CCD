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
RUN_TAG = "repeated_sphere_funnel_drop_run_id"
MYDEMO_DIR = P2CCCD / "MyDemo" / RUN_TAG
BENCHMARK_DIR = P2CCCD / "benchmark"
TRAIN_DIR = P2CCCD / "datasets" / "training" / RUN_TAG
OUTPUT_DIR = P2CCCD / "outputs" / "stpf_training" / RUN_TAG


@dataclass
class FunnelCaseConfig:
    particle_count: int = 800
    wave_count: int = 8
    render_fps: int = 24
    render_frames: int = 432
    sim_substeps_per_frame: int = 5
    seed: int = 0x0135269B
    sphere_radius: float = 0.043
    sphere_radius_jitter: float = 0.18
    sphere_density_kg_m3: float = 6400.0
    gravity: float = 9.81
    restitution_wall: float = 0.24
    restitution_pair: float = 0.18
    friction_mu: float = 0.16
    air_drag: float = 0.012
    rolling_damping: float = 0.055
    pair_iterations: int = 3
    funnel_height: float = 2.75
    funnel_top_radius: float = 2.15
    funnel_throat_radius: float = 0.145
    funnel_throat_length: float = 1.35
    collector_radius: float = 0.92
    collector_depth: float = 0.72
    launch_height: float = 4.10
    launch_radius: float = 2.35
    launch_speed_mean: float = 1.35
    launch_speed_jitter: float = 0.22
    swirl_strength: float = 1.28
    wave_interval_seconds: float = 1.05
    wave_jitter_seconds: float = 0.06
    feature_dim: int = 32
    train_epochs: int = 6
    train_batch_size: int = 32768


def ensure_dirs() -> None:
    for path in [MYDEMO_DIR, BENCHMARK_DIR, TRAIN_DIR, OUTPUT_DIR, MYDEMO_DIR / "frames", MYDEMO_DIR / "assets"]:
        path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def safe_rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def font(size: int) -> ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/segoeui.ttf"),
        Path("C:/Windows/Fonts/arial.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def funnel_radius_at_z(z: np.ndarray | float, cfg: FunnelCaseConfig) -> np.ndarray | float:
    z_arr = np.asarray(z)
    cone_t = np.clip(z_arr / cfg.funnel_height, 0.0, 1.0)
    radius = cfg.funnel_throat_radius + (cfg.funnel_top_radius - cfg.funnel_throat_radius) * cone_t
    return radius if isinstance(z, np.ndarray) else float(radius)


def signed_funnel_gap(pos: np.ndarray, radii: np.ndarray, cfg: FunnelCaseConfig) -> np.ndarray:
    xy = pos[:, :2]
    radial = np.linalg.norm(xy, axis=1)
    z = pos[:, 2]
    cone_radius = funnel_radius_at_z(np.clip(z, 0.0, cfg.funnel_height), cfg)
    throat_zone = z < 0.0
    radius_limit = np.where(throat_zone, cfg.funnel_throat_radius, cone_radius)
    return radius_limit - radial - radii


def funnel_normal(pos: np.ndarray, cfg: FunnelCaseConfig) -> np.ndarray:
    radial = np.linalg.norm(pos[:, :2], axis=1)
    radial_safe = np.maximum(radial, 1.0e-9)
    slope = (cfg.funnel_top_radius - cfg.funnel_throat_radius) / cfg.funnel_height
    n = np.column_stack([-pos[:, 0] / radial_safe, -pos[:, 1] / radial_safe, np.full(len(pos), slope)])
    throat = pos[:, 2] < 0.0
    n[throat] = np.column_stack(
        [-pos[throat, 0] / radial_safe[throat], -pos[throat, 1] / radial_safe[throat], np.zeros(np.count_nonzero(throat))]
    )
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1.0e-9)
    return n


def collector_floor_z(xy: np.ndarray, cfg: FunnelCaseConfig) -> np.ndarray:
    r = np.linalg.norm(xy, axis=1)
    return -cfg.funnel_throat_length - cfg.collector_depth + 0.10 * (r / max(cfg.collector_radius, 1.0e-9)) ** 2


def apply_coulomb_tangent_friction(
    velocities: np.ndarray,
    normals: np.ndarray,
    normal_impulse: np.ndarray,
    cfg: FunnelCaseConfig,
    dt: float,
) -> np.ndarray:
    tangent = velocities - np.sum(velocities * normals, axis=1)[:, None] * normals
    tangent_speed = np.linalg.norm(tangent, axis=1)
    moving = tangent_speed > 1.0e-8
    if not np.any(moving):
        return velocities
    tangent_dir = np.zeros_like(tangent)
    tangent_dir[moving] = tangent[moving] / tangent_speed[moving, None]
    friction_impulse = cfg.friction_mu * np.maximum(normal_impulse, 0.0)
    delta_speed = np.minimum(tangent_speed, friction_impulse)
    velocities -= delta_speed[:, None] * tangent_dir
    velocities *= math.exp(-cfg.rolling_damping * dt)
    return velocities


def resolve_funnel_contacts(
    pos: np.ndarray,
    vel: np.ndarray,
    radii: np.ndarray,
    active: np.ndarray,
    cfg: FunnelCaseConfig,
    dt: float,
) -> tuple[int, np.ndarray]:
    contacts: list[np.ndarray] = []
    active_ids = np.flatnonzero(active)
    if active_ids.size == 0:
        return 0, np.zeros((0, 3), dtype=np.float64)
    p = pos[active_ids]
    rads = radii[active_ids]
    z = p[:, 2]
    inside_vertical = (z <= cfg.funnel_height + 0.25) & (z >= -cfg.funnel_throat_length - 0.05)
    gaps = signed_funnel_gap(p, rads, cfg)
    contact_local = inside_vertical & (gaps < 0.0)
    if np.any(contact_local):
        ids = active_ids[contact_local]
        n = funnel_normal(pos[ids], cfg)
        penetration = -gaps[contact_local]
        pos[ids] += penetration[:, None] * n
        vn = np.sum(vel[ids] * n, axis=1)
        normal_impulse = np.maximum(0.0, cfg.gravity * np.maximum(n[:, 2], 0.0) * dt)
        incoming = vn < 0.0
        if np.any(incoming):
            ids_in = ids[incoming]
            n_in = n[incoming]
            vn_in = vn[incoming]
            vel[ids_in] -= (1.0 + cfg.restitution_wall) * vn_in[:, None] * n_in
            normal_impulse[incoming] += (1.0 + cfg.restitution_wall) * (-vn_in)
        vel[ids] = apply_coulomb_tangent_friction(vel[ids].copy(), n, normal_impulse, cfg, dt)
        contacts.extend((pos[ids] - rads[contact_local, None] * n).tolist())

    # Collector cup floor and cylindrical side.
    below_ids = active_ids[pos[active_ids, 2] < -cfg.funnel_throat_length + 0.18]
    if below_ids.size:
        floor = collector_floor_z(pos[below_ids, :2], cfg)
        floor_gap = pos[below_ids, 2] - radii[below_ids] - floor
        floor_contact = floor_gap < 0.0
        if np.any(floor_contact):
            ids = below_ids[floor_contact]
            pos[ids, 2] -= floor_gap[floor_contact]
            n = np.tile(np.array([0.0, 0.0, 1.0], dtype=np.float64), (ids.size, 1))
            vn = vel[ids, 2]
            incoming = vn < 0.0
            normal_impulse = np.full(ids.size, cfg.gravity * dt, dtype=np.float64)
            normal_impulse[incoming] += (1.0 + cfg.restitution_wall) * (-vn[incoming])
            vel[ids[incoming], 2] *= -cfg.restitution_wall
            vel[ids] = apply_coulomb_tangent_friction(vel[ids].copy(), n, normal_impulse, cfg, dt)
            contacts.extend((pos[ids] - radii[ids, None] * n).tolist())

        radial = np.linalg.norm(pos[below_ids, :2], axis=1)
        side_gap = cfg.collector_radius - radial - radii[below_ids]
        side_contact = side_gap < 0.0
        if np.any(side_contact):
            ids = below_ids[side_contact]
            rr = np.maximum(np.linalg.norm(pos[ids, :2], axis=1), 1.0e-9)
            inward = np.column_stack([-pos[ids, 0] / rr, -pos[ids, 1] / rr, np.zeros(ids.size)])
            pos[ids] += (-side_gap[side_contact])[:, None] * inward
            vn = np.sum(vel[ids] * inward, axis=1)
            incoming = vn < 0.0
            normal_impulse = np.zeros(ids.size, dtype=np.float64)
            vel[ids[incoming]] -= (1.0 + cfg.restitution_wall) * vn[incoming, None] * inward[incoming]
            normal_impulse[incoming] = (1.0 + cfg.restitution_wall) * (-vn[incoming])
            vel[ids] = apply_coulomb_tangent_friction(vel[ids].copy(), inward, normal_impulse, cfg, dt)
            contacts.extend((pos[ids] - radii[ids, None] * inward).tolist())

    return len(contacts), np.asarray(contacts, dtype=np.float64)


def resolve_particle_pairs(
    pos: np.ndarray,
    vel: np.ndarray,
    radii: np.ndarray,
    active: np.ndarray,
    cfg: FunnelCaseConfig,
) -> tuple[int, np.ndarray]:
    contacts: list[np.ndarray] = []
    active_ids = np.flatnonzero(active)
    if active_ids.size < 2:
        return 0, np.zeros((0, 3), dtype=np.float64)
    cell = float(np.max(radii) * 2.35)
    coords = np.floor(pos[active_ids] / cell).astype(np.int64)
    buckets: dict[tuple[int, int, int], list[int]] = {}
    for local, c in enumerate(coords):
        buckets.setdefault(tuple(map(int, c)), []).append(int(active_ids[local]))
    seen: set[tuple[int, int]] = set()
    offsets = [(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)]
    for key, ids in buckets.items():
        base = np.asarray(key, dtype=np.int64)
        for off in offsets:
            nb = tuple(map(int, base + np.asarray(off, dtype=np.int64)))
            if nb not in buckets:
                continue
            for a in ids:
                for b in buckets[nb]:
                    if b <= a:
                        continue
                    pair = (a, b)
                    if pair in seen:
                        continue
                    seen.add(pair)
                    d = pos[b] - pos[a]
                    dist = float(np.linalg.norm(d))
                    target = float(radii[a] + radii[b])
                    if dist >= target or dist < 1.0e-9:
                        continue
                    n = d / dist
                    penetration = target - dist
                    pos[a] -= 0.5 * penetration * n
                    pos[b] += 0.5 * penetration * n
                    rel_v = vel[b] - vel[a]
                    vn = float(np.dot(rel_v, n))
                    normal_impulse = 0.0
                    if vn < 0.0:
                        impulse = -(1.0 + cfg.restitution_pair) * vn * 0.5
                        vel[a] -= impulse * n
                        vel[b] += impulse * n
                        normal_impulse = 2.0 * impulse
                    tangent = rel_v - vn * n
                    tangent_speed = float(np.linalg.norm(tangent))
                    if tangent_speed > 1.0e-8 and normal_impulse > 0.0:
                        tangent_dir = tangent / tangent_speed
                        friction_impulse = min(tangent_speed * 0.5, cfg.friction_mu * normal_impulse * 0.5)
                        vel[a] += friction_impulse * tangent_dir
                        vel[b] -= friction_impulse * tangent_dir
                    contacts.append((0.5 * (pos[a] + pos[b])).copy())
    return len(contacts), np.asarray(contacts, dtype=np.float64)


def physics_audit(
    frames: list[np.ndarray],
    frame_velocities: list[np.ndarray],
    frame_active: list[np.ndarray],
    radii: np.ndarray,
    spawn: np.ndarray,
    cfg: FunnelCaseConfig,
) -> dict[str, Any]:
    mass = (4.0 / 3.0) * math.pi * (radii.astype(np.float64) ** 3) * cfg.sphere_density_kg_m3
    floor_ref = -cfg.funnel_throat_length - cfg.collector_depth - 0.12
    energy: list[float] = []
    tangential_speed: list[float] = []
    mean_z: list[float] = []
    for pos, vel, active in zip(frames, frame_velocities, frame_active):
        ids = np.flatnonzero(active)
        if ids.size == 0:
            energy.append(0.0)
            tangential_speed.append(0.0)
            mean_z.append(float("nan"))
            continue
        p = pos[ids].astype(np.float64)
        v = vel[ids].astype(np.float64)
        m = mass[ids]
        kinetic = 0.5 * m * np.sum(v * v, axis=1)
        potential = m * cfg.gravity * np.maximum(p[:, 2] - floor_ref, 0.0)
        energy.append(float(np.sum(kinetic + potential, dtype=np.float64)))
        radial = np.maximum(np.linalg.norm(p[:, :2], axis=1), 1.0e-9)
        tangent = np.column_stack([-p[:, 1] / radial, p[:, 0] / radial, np.zeros(ids.size)])
        tangential_speed.append(float(np.mean(np.abs(np.sum(v * tangent, axis=1)))))
        mean_z.append(float(np.mean(p[:, 2])))

    final_pos = frames[-1].astype(np.float64)
    final_active = frame_active[-1]
    final_ids = np.flatnonzero(final_active)
    final_radial = np.linalg.norm(final_pos[final_ids, :2], axis=1) if final_ids.size else np.zeros(0)
    in_collector = (
        (final_pos[final_ids, 2] < -cfg.funnel_throat_length + 0.25)
        & (final_radial < cfg.collector_radius + 2.0 * np.max(radii))
    )
    below_throat = final_pos[final_ids, 2] < 0.0 if final_ids.size else np.zeros(0, dtype=bool)
    all_spawned_frame = min(len(frames) - 1, max(0, int(math.ceil(float(np.max(spawn)) * cfg.render_fps)) + 2))
    e_start = float(energy[all_spawned_frame])
    e_final = float(energy[-1])
    post_spawn_energy = energy[all_spawned_frame:] if all_spawned_frame < len(energy) else energy[-1:]
    e_peak_after_spawn = float(max(post_spawn_energy)) if post_spawn_energy else e_final
    peak_increase_fraction = float((e_peak_after_spawn - e_start) / e_start) if e_start > 1.0e-9 else 0.0
    energy_drop_fraction = float((e_start - e_final) / e_start) if e_start > 1.0e-9 else 0.0
    return {
        "energy_joules_after_all_spawn": e_start,
        "energy_peak_joules_after_all_spawn": e_peak_after_spawn,
        "energy_joules_final": e_final,
        "energy_drop_fraction_after_all_spawn": energy_drop_fraction,
        "energy_peak_increase_fraction_after_all_spawn": peak_increase_fraction,
        "mean_tangential_speed_initial_active": float(next((v for v in tangential_speed if v > 0.0), 0.0)),
        "mean_tangential_speed_after_all_spawn": float(tangential_speed[all_spawned_frame]),
        "mean_tangential_speed_final": float(tangential_speed[-1]),
        "mean_height_after_all_spawn": float(mean_z[all_spawned_frame]),
        "mean_height_final": float(mean_z[-1]),
        "final_below_throat_fraction": float(np.mean(below_throat)) if final_ids.size else 0.0,
        "final_in_collector_fraction": float(np.mean(in_collector)) if final_ids.size else 0.0,
        "all_spawned_frame": int(all_spawned_frame),
    }


def initialize_particles(cfg: FunnelCaseConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(cfg.seed)
    n = cfg.particle_count
    wave_ids = np.repeat(np.arange(cfg.wave_count), math.ceil(n / cfg.wave_count))[:n]
    per_wave = math.ceil(n / cfg.wave_count)
    local_in_wave = np.arange(n) - wave_ids * per_wave
    radii = cfg.sphere_radius * (1.0 + rng.uniform(-cfg.sphere_radius_jitter, cfg.sphere_radius_jitter, size=n))
    intra_wave_delay = 0.0045 * local_in_wave
    spawn = wave_ids * cfg.wave_interval_seconds + intra_wave_delay + rng.normal(0.0, cfg.wave_jitter_seconds * 0.18, size=n)
    spawn = np.maximum(spawn, 0.0)

    # All waves originate from the same small rim nozzle.  The outgoing
    # tangential direction is intentionally reversed relative to the previous
    # version, so the beads circulate along the other side of the funnel wall.
    strip = (local_in_wave + 0.5) / max(1, per_wave)
    nozzle_theta = -2.45
    theta = nozzle_theta + rng.normal(0.0, 0.012, n)
    z0 = cfg.funnel_height - 0.10 + rng.normal(0.0, 0.018, n)
    wall_radius = funnel_radius_at_z(np.clip(z0, 0.0, cfg.funnel_height), cfg)
    radial = wall_radius - 0.72 * radii + rng.normal(0.0, 0.003, n)
    pos = np.column_stack([radial * np.cos(theta), radial * np.sin(theta), z0])

    tangent = np.column_stack([np.sin(theta), -np.cos(theta), np.zeros(n)])
    inward = np.column_stack([-np.cos(theta), -np.sin(theta), np.zeros(n)])
    slope = (cfg.funnel_top_radius - cfg.funnel_throat_radius) / cfg.funnel_height
    down_wall = np.column_stack([-slope * np.cos(theta), -slope * np.sin(theta), -np.ones(n)])
    down_wall /= np.maximum(np.linalg.norm(down_wall, axis=1, keepdims=True), 1.0e-9)
    tangential_speed = cfg.launch_speed_mean + rng.normal(0.0, cfg.launch_speed_jitter, size=n)
    down_speed = 0.92 + 0.18 * rng.random(n)
    vel = (
        cfg.swirl_strength * tangential_speed[:, None] * tangent
        + down_speed[:, None] * down_wall
        + 0.10 * inward
        + rng.normal(0.0, 0.045, size=(n, 3))
    )
    return pos, vel, radii, spawn, wave_ids.astype(np.int32)


def simulate(cfg: FunnelCaseConfig) -> dict[str, Any]:
    pos, vel, radii, spawn, wave_ids = initialize_particles(cfg)
    initial_pos = pos.copy()
    initial_vel = vel.copy()
    n = cfg.particle_count
    dt = 1.0 / (cfg.render_fps * cfg.sim_substeps_per_frame)
    frames: list[np.ndarray] = []
    frame_velocities: list[np.ndarray] = []
    frame_active: list[np.ndarray] = []
    funnel_contacts: list[np.ndarray] = []
    pair_contacts: list[np.ndarray] = []
    funnel_contact_counts: list[int] = []
    pair_contact_counts: list[int] = []
    active = np.zeros(n, dtype=bool)
    first_contact_time: float | None = None
    total_funnel_contacts = 0
    total_pair_contacts = 0
    for frame_idx in range(cfg.render_frames):
        for _ in range(cfg.sim_substeps_per_frame):
            t = (frame_idx * cfg.sim_substeps_per_frame + _) * dt
            newly = (~active) & (spawn <= t)
            if np.any(newly):
                pos[newly] = initial_pos[newly]
                vel[newly] = initial_vel[newly]
                active[newly] = True
            ids = np.flatnonzero(active)
            if ids.size:
                vel[ids, 2] -= cfg.gravity * dt
                vel[ids] *= math.exp(-cfg.air_drag * dt)
                pos[ids] += vel[ids] * dt
                f_count, f_pts = resolve_funnel_contacts(pos, vel, radii, active, cfg, dt)
                p_count = 0
                p_pts = np.zeros((0, 3), dtype=np.float64)
                for _iter in range(cfg.pair_iterations):
                    iter_count, iter_pts = resolve_particle_pairs(pos, vel, radii, active, cfg)
                    if iter_count == 0:
                        break
                    p_count += iter_count
                    if iter_pts.size:
                        p_pts = iter_pts
                if f_count and first_contact_time is None:
                    first_contact_time = float(t)
                total_funnel_contacts += f_count
                total_pair_contacts += p_count
        frames.append(pos.copy())
        frame_velocities.append(vel.copy())
        frame_active.append(active.copy())
        probe_pos = pos.copy()
        probe_vel = vel.copy()
        f_count, f_pts = resolve_funnel_contacts(probe_pos, probe_vel, radii, active, cfg, dt)
        p_count, p_pts = resolve_particle_pairs(probe_pos, probe_vel, radii, active, cfg)
        funnel_contacts.append(f_pts[:240])
        pair_contacts.append(p_pts[:160])
        funnel_contact_counts.append(int(f_count))
        pair_contact_counts.append(int(p_count))
    audit = physics_audit(frames, frame_velocities, frame_active, radii, spawn, cfg)
    return {
        "positions": np.asarray(frames, dtype=np.float32),
        "velocities": np.asarray(frame_velocities, dtype=np.float32),
        "active": np.asarray(frame_active, dtype=np.bool_),
        "radii": radii.astype(np.float32),
        "spawn_times": spawn.astype(np.float32),
        "wave_ids": wave_ids,
        "funnel_contacts": funnel_contacts,
        "pair_contacts": pair_contacts,
        "funnel_contact_counts": np.asarray(funnel_contact_counts, dtype=np.int32),
        "pair_contact_counts": np.asarray(pair_contact_counts, dtype=np.int32),
        "total_funnel_contacts": int(total_funnel_contacts),
        "total_pair_contacts": int(total_pair_contacts),
        "first_contact_time": first_contact_time,
        "physics_audit": audit,
    }


def camera_project(points: np.ndarray, width: int = 1920, height: int = 1080, scale: float = 165.0) -> tuple[np.ndarray, np.ndarray]:
    eye = np.array([4.5, -6.2, 4.4], dtype=np.float64)
    target = np.array([-0.22, 0.0, 0.45], dtype=np.float64)
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    rel = points.astype(np.float64) - target
    x = rel @ right
    y = rel @ up
    z = rel @ forward
    xy = np.column_stack([width * 0.52 + scale * x, height * 0.54 - scale * y])
    return xy, z


def funnel_mesh_segments(cfg: FunnelCaseConfig) -> list[np.ndarray]:
    rings = []
    segment_count = 128
    for z in np.linspace(-cfg.funnel_throat_length, cfg.funnel_height, 16):
        if z < 0.0:
            r = cfg.funnel_throat_radius
        else:
            r = funnel_radius_at_z(float(z), cfg)
        pts = []
        for a in np.linspace(0.0, 2.0 * math.pi, segment_count, endpoint=False):
            pts.append([r * math.cos(a), r * math.sin(a), z])
        rings.append(np.asarray(pts, dtype=np.float64))
    return rings


def draw_funnel(draw: ImageDraw.ImageDraw, cfg: FunnelCaseConfig) -> None:
    rings = funnel_mesh_segments(cfg)
    ring_colors = [(78, 121, 167, 165), (96, 162, 205, 138)]
    for idx, ring in enumerate(rings):
        xy, _ = camera_project(ring)
        pts = [tuple(map(float, p)) for p in xy]
        draw.line(pts + [pts[0]], fill=ring_colors[idx % 2], width=2)
    segment_count = len(rings[0])
    for k in range(0, segment_count, 4):
        pts3 = np.asarray([ring[k] for ring in rings], dtype=np.float64)
        xy, _ = camera_project(pts3)
        draw.line([tuple(map(float, p)) for p in xy], fill=(65, 92, 120, 100), width=1)
    # Collector cup.
    cup_rings = []
    for z in np.linspace(-cfg.funnel_throat_length - cfg.collector_depth, -cfg.funnel_throat_length, 4):
        pts = []
        for a in np.linspace(0.0, 2.0 * math.pi, segment_count, endpoint=False):
            pts.append([cfg.collector_radius * math.cos(a), cfg.collector_radius * math.sin(a), z])
        cup_rings.append(np.asarray(pts, dtype=np.float64))
    for ring in cup_rings:
        xy, _ = camera_project(ring)
        pts = [tuple(map(float, p)) for p in xy]
        draw.line(pts + [pts[0]], fill=(145, 118, 87, 130), width=2)


def draw_grid(draw: ImageDraw.ImageDraw) -> None:
    extent = 4.5
    for v in np.linspace(-extent, extent, 19):
        for p0, p1 in [
            (np.array([-extent, v, -2.15]), np.array([extent, v, -2.15])),
            (np.array([v, -extent, -2.15]), np.array([v, extent, -2.15])),
        ]:
            xy, _ = camera_project(np.vstack([p0, p1]))
            color = (216, 222, 226, 190) if abs(v) > 1.0e-9 else (190, 199, 207, 230)
            draw.line([tuple(map(float, xy[0])), tuple(map(float, xy[1]))], fill=color, width=1)


def draw_contact_regions(
    draw: ImageDraw.ImageDraw,
    pts3: np.ndarray,
    fill: tuple[int, int, int, int],
    outline: tuple[int, int, int, int],
    *,
    max_regions: int,
    min_points: int = 5,
) -> None:
    if len(pts3) < min_points:
        return
    pts3 = np.asarray(pts3, dtype=np.float64)
    z = pts3[:, 2]
    if float(np.ptp(z)) < 1.0e-5:
        bands = [pts3]
    else:
        edges = np.linspace(float(z.min()), float(z.max()) + 1.0e-6, max_regions + 1)
        bands = [pts3[(z >= edges[i]) & (z < edges[i + 1])] for i in range(max_regions)]
    for band in bands:
        if len(band) < min_points:
            continue
        xy, _ = camera_project(band)
        lo = np.percentile(xy, 8, axis=0)
        hi = np.percentile(xy, 92, axis=0)
        margin = 18.0 + 0.25 * min(90.0, math.sqrt(float(len(band))) * 8.0)
        box = [float(lo[0] - margin), float(lo[1] - margin), float(hi[0] + margin), float(hi[1] + margin)]
        draw.ellipse(box, fill=fill, outline=outline, width=2)


def draw_material_sphere(
    draw: ImageDraw.ImageDraw,
    x: float,
    y: float,
    radius_px: float,
    base_rgb: np.ndarray,
    zshade: float,
) -> None:
    base = np.asarray(base_rgb, dtype=np.float64)
    shadow = tuple(int(max(0, c * 0.36)) for c in base)
    draw.ellipse(
        [x - radius_px * 0.78, y - radius_px * 0.48, x + radius_px * 0.92, y + radius_px * 0.70],
        fill=(*shadow, 42),
    )
    for layer in range(5, 0, -1):
        k = layer / 5.0
        rr = radius_px * k
        light = 0.58 + 0.34 * (1.0 - k) + 0.16 * zshade
        color = tuple(int(np.clip(c * light + 18.0 * (1.0 - k), 0, 255)) for c in base)
        draw.ellipse([x - rr, y - rr, x + rr, y + rr], fill=(*color, 236), outline=(30, 38, 48, 56), width=1)
    highlight_r = max(1.2, radius_px * 0.30)
    hx = x - radius_px * 0.32
    hy = y - radius_px * 0.36
    draw.ellipse(
        [hx - highlight_r, hy - highlight_r, hx + highlight_r, hy + highlight_r],
        fill=(255, 255, 255, 92),
    )
    draw.arc([x - radius_px, y - radius_px, x + radius_px, y + radius_px], 205, 335, fill=(255, 255, 255, 58), width=1)


def render_frame(sim: dict[str, Any], frame_idx: int, cfg: FunnelCaseConfig) -> Image.Image:
    width, height = 1920, 1080
    image = Image.new("RGB", (width, height), (248, 250, 252))
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    draw_grid(draw)
    draw_funnel(draw, cfg)
    positions = sim["positions"][frame_idx]
    active = sim["active"][frame_idx]
    radii = sim["radii"]
    wave_ids = sim["wave_ids"]
    palette = np.asarray(
        [
            [72, 163, 238],
            [250, 204, 21],
            [94, 214, 148],
            [244, 113, 98],
            [168, 133, 255],
            [45, 212, 191],
            [251, 146, 60],
            [236, 72, 153],
        ],
        dtype=np.uint8,
    )
    xy, depth = camera_project(positions)
    ids = np.flatnonzero(active)
    order = ids[np.argsort(depth[ids])]

    # Contact regions are intentionally drawn underneath material spheres so
    # they identify dense CCD areas without hiding the particles.
    draw_contact_regions(
        draw,
        sim["funnel_contacts"][frame_idx],
        (255, 139, 42, 16),
        (255, 139, 42, 92),
        max_regions=4,
    )
    draw_contact_regions(
        draw,
        sim["pair_contacts"][frame_idx],
        (168, 85, 247, 14),
        (130, 62, 205, 82),
        max_regions=3,
    )

    for i in order:
        x, y = xy[i]
        if x < -40 or x > width + 40 or y < -40 or y > height + 40:
            continue
        color = palette[int(wave_ids[i]) % len(palette)]
        zshade = np.clip(0.72 + 0.18 * (positions[i, 2] + 2.0) / 6.0, 0.55, 1.0)
        rr = max(2.2, float(radii[i] * 150.0))
        draw_material_sphere(draw, float(x), float(y), rr, color, float(zshade))

    # Contact witnesses: orange for funnel/collector, violet for particle-particle.
    for pts3, fill, outline, max_pts in [
        (sim["funnel_contacts"][frame_idx], (255, 139, 42, 210), (78, 39, 12, 230), 120),
        (sim["pair_contacts"][frame_idx], (168, 85, 247, 185), (78, 28, 110, 220), 72),
    ]:
        if len(pts3) == 0:
            continue
        pts = np.asarray(pts3[:max_pts], dtype=np.float64)
        cxy, _ = camera_project(pts)
        for x, y in cxy:
            r = 5.0
            draw.ellipse([x - r * 2.0, y - r * 2.0, x + r * 2.0, y + r * 2.0], fill=(*fill[:3], 38), outline=(*fill[:3], 110), width=1)
            draw.line([x - r, y, x + r, y], fill=outline, width=2)
            draw.line([x, y - r, x, y + r], fill=outline, width=2)
            draw.ellipse([x - 2.2, y - 2.2, x + 2.2, y + 2.2], fill=fill, outline=outline, width=1)
    # Header and legend, matching Scalable-CCD supplementary style.
    draw.rounded_rectangle([18, 18, width - 18, 102], radius=4, fill=(255, 255, 255, 224), outline=(180, 188, 198, 235), width=2)
    title_font = font(28)
    small_font = font(18)
    t = frame_idx / cfg.render_fps
    draw.text((38, 35), "Repeated Single-Nozzle Sphere Funnel Drop: P2C-CCD contact replay", fill=(28, 36, 46, 255), font=title_font)
    draw.text((38, 70), "all waves start from the same rim nozzle; reversed tangential launch slides along the opposite funnel wall", fill=(63, 74, 89, 255), font=small_font)
    draw.text((width - 310, 43), f"t={t:.2f}s | wave {min(cfg.wave_count, int(t / cfg.wave_interval_seconds) + 1)}/{cfg.wave_count}", fill=(0, 128, 96, 255), font=small_font)
    legend_y = height - 88
    draw.rounded_rectangle([34, legend_y, 845, height - 28], radius=10, fill=(255, 255, 255, 225), outline=(186, 196, 206, 220), width=1)
    draw.rectangle([56, legend_y + 18, 80, legend_y + 38], fill=(72, 163, 238, 230))
    draw.text((90, legend_y + 15), "ball wave colors", fill=(54, 64, 76, 255), font=small_font)
    draw.ellipse([260, legend_y + 15, 284, legend_y + 39], fill=(255, 139, 42, 190), outline=(78, 39, 12, 220), width=2)
    draw.text((296, legend_y + 15), "funnel/contact point", fill=(54, 64, 76, 255), font=small_font)
    draw.ellipse([500, legend_y + 15, 524, legend_y + 39], fill=(168, 85, 247, 170), outline=(78, 28, 110, 220), width=2)
    draw.text((536, legend_y + 15), "ball-ball point", fill=(54, 64, 76, 255), font=small_font)
    draw.ellipse([700, legend_y + 13, 742, legend_y + 41], fill=(255, 139, 42, 45), outline=(255, 139, 42, 145), width=2)
    draw.text((752, legend_y + 15), "contact region", fill=(54, 64, 76, 255), font=small_font)
    image = Image.alpha_composite(image.convert("RGBA"), overlay)
    return image.convert("RGB")


def write_mp4(path: Path, frames: list[Image.Image], fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=8, macro_block_size=1)
    try:
        for frame in frames:
            writer.append_data(np.asarray(frame.convert("RGB")))
    finally:
        writer.close()


def render_outputs(sim: dict[str, Any], cfg: FunnelCaseConfig) -> dict[str, str]:
    frames: list[Image.Image] = []
    for i in range(cfg.render_frames):
        frame = render_frame(sim, i, cfg)
        frames.append(frame)
        if i % 4 == 0 or i in {0, cfg.render_frames - 1}:
            frame.save(MYDEMO_DIR / "frames" / f"frame_{i:04d}.png")
    mp4 = MYDEMO_DIR / "repeated_sphere_funnel_drop.mp4"
    write_mp4(mp4, frames, cfg.render_fps)
    sheet_indices = [0, int(0.25 * cfg.render_frames), int(0.50 * cfg.render_frames), int(0.75 * cfg.render_frames), cfg.render_frames - 1]
    thumbs = [frames[i].resize((640, 360), Image.Resampling.LANCZOS) for i in sheet_indices]
    sheet = Image.new("RGB", (1280, 1080), (246, 248, 249))
    for j, thumb in enumerate(thumbs[:4]):
        sheet.paste(thumb, ((j % 2) * 640, (j // 2) * 360))
    sheet.paste(thumbs[-1], (320, 720))
    sheet_path = MYDEMO_DIR / "contact_sheet.png"
    sheet.save(sheet_path)
    return {"mp4": safe_rel(mp4), "contact_sheet": safe_rel(sheet_path)}


def build_candidate_dataset(sim: dict[str, Any], cfg: FunnelCaseConfig) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    positions = sim["positions"]
    velocities = sim["velocities"]
    active = sim["active"]
    radii = sim["radii"]
    wave_ids = sim["wave_ids"]
    rows: list[np.ndarray] = []
    labels: list[int] = []
    costs: list[float] = []
    ids: list[tuple[int, int, int]] = []
    rng = np.random.default_rng(cfg.seed + 99)
    candidate_id = 0
    for frame_idx in range(0, cfg.render_frames, 2):
        p = positions[frame_idx].astype(np.float64)
        v = velocities[frame_idx].astype(np.float64)
        a = active[frame_idx]
        active_ids = np.flatnonzero(a)
        if active_ids.size == 0:
            continue
        gaps = signed_funnel_gap(p[active_ids], radii[active_ids], cfg)
        speed = np.linalg.norm(v[active_ids], axis=1)
        radial = np.linalg.norm(p[active_ids, :2], axis=1)
        cone_radius = funnel_radius_at_z(np.clip(p[active_ids, 2], 0.0, cfg.funnel_height), cfg)
        near_wall = gaps < 0.12
        near_throat = (p[active_ids, 2] < 0.35) & (p[active_ids, 2] > -cfg.funnel_throat_length - 0.25)
        random_keep = rng.random(active_ids.size) < 0.22
        keep = near_wall | near_throat | random_keep
        for local_id, particle_id in enumerate(active_ids[keep]):
            li = np.where(active_ids == particle_id)[0][0]
            gap = float(gaps[li])
            label = int(gap < 0.025 or (near_throat[li] and abs(radial[li] - cfg.funnel_throat_radius) < 0.09))
            z = float(p[particle_id, 2])
            feature = np.zeros(cfg.feature_dim, dtype=np.float32)
            feature[:3] = p[particle_id]
            feature[3:6] = v[particle_id]
            feature[6] = float(radii[particle_id])
            feature[7] = gap
            feature[8] = float(radial[li])
            feature[9] = float(cone_radius[li])
            feature[10] = float(speed[li])
            feature[11] = float(wave_ids[particle_id] / max(1, cfg.wave_count - 1))
            feature[12] = float(frame_idx / max(1, cfg.render_frames - 1))
            feature[13] = float(z < 0.0)
            feature[14] = float(near_throat[li])
            feature[15] = float(near_wall[li])
            feature[16] = float(abs(radial[li] - cone_radius[li]))
            feature[17] = float(abs(radial[li] - cfg.funnel_throat_radius))
            feature[18] = float(np.dot(v[particle_id, :2], p[particle_id, :2]) / max(radial[li], 1.0e-6))
            feature[19] = float(v[particle_id, 2])
            feature[20] = float(math.sin(wave_ids[particle_id]))
            feature[21] = float(math.cos(wave_ids[particle_id]))
            feature[22] = float(max(0.0, -gap))
            feature[23] = float(max(0.0, 0.15 - gap))
            feature[24] = float(max(0.0, cfg.collector_radius - radial[li]))
            feature[25] = float(sim["funnel_contact_counts"][frame_idx] > 0)
            feature[26] = float(sim["pair_contact_counts"][frame_idx] > 0)
            feature[27] = float(particle_id % 17) / 17.0
            feature[28] = float((particle_id // 17) % 17) / 17.0
            feature[29] = float(np.linalg.norm(p[particle_id]))
            feature[30] = float(np.linalg.norm(v[particle_id, :2]))
            feature[31] = 1.0
            rows.append(feature)
            labels.append(label)
            costs.append(1.0 + 120.0 / (abs(gap) + 0.025) + 8.0 * speed[li] + 24.0 * near_throat[li])
            ids.append((0, frame_idx, candidate_id))
            candidate_id += 1
    features = np.asarray(rows, dtype=np.float32)
    labels_np = np.asarray(labels, dtype=np.int64)
    costs_np = np.asarray(costs, dtype=np.float32)
    ids_np = np.asarray(ids, dtype=np.uint64)
    n = features.shape[0]
    order = np.arange(n)
    rng.shuffle(order)
    train_end = int(0.70 * n)
    val_end = int(0.85 * n)
    splits = {
        "train": order[:train_end],
        "validation": order[train_end:val_end],
        "heldout_test": order[val_end:],
    }
    out: dict[str, dict[str, np.ndarray]] = {}
    for split, idx in splits.items():
        out[split] = {
            "features": features[idx],
            "ground_truth": labels_np[idx],
            "costs": costs_np[idx],
            "ids": ids_np[idx],
        }
        np.savez_compressed(TRAIN_DIR / f"{split}.npz", **out[split])
    manifest = {
        "run_tag": RUN_TAG,
        "feature_dim": cfg.feature_dim,
        "row_count": int(n),
        "positive_count": int(labels_np.sum()),
        "positive_ratio": float(labels_np.mean()) if n else 0.0,
        "splits": {k: int(v["features"].shape[0]) for k, v in out.items()},
        "description": "Candidate rows from repeated sphere-funnel contacts; labels are exact-contact witnesses or near-throat collision certificates.",
    }
    write_json(TRAIN_DIR / "manifest.json", manifest)
    return out, manifest


def train_tiny_stpf(splits: dict[str, dict[str, np.ndarray]], cfg: FunnelCaseConfig) -> dict[str, Any]:
    try:
        import torch
        import torch.nn as nn
    except Exception as exc:  # pragma: no cover
        return {"status": "skipped", "reason": f"torch unavailable: {exc}"}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x_train = torch.from_numpy(splits["train"]["features"]).float().to(device)
    y_train = torch.from_numpy(splits["train"]["ground_truth"].astype(np.float32)).to(device)
    x_val = torch.from_numpy(splits["validation"]["features"]).float().to(device)
    y_val = torch.from_numpy(splits["validation"]["ground_truth"].astype(np.float32)).to(device)
    model = nn.Sequential(nn.Linear(cfg.feature_dim, 64), nn.SiLU(), nn.Linear(64, 32), nn.SiLU(), nn.Linear(32, 1)).to(device)
    pos = float(y_train.sum().item())
    neg = float(y_train.numel() - pos)
    pos_weight = torch.tensor([max(1.0, neg / max(1.0, pos))], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=2.5e-3, weight_decay=1.0e-4)
    history: list[dict[str, float]] = []
    rng = np.random.default_rng(cfg.seed + 123)
    for epoch in range(cfg.train_epochs):
        model.train()
        perm = rng.permutation(x_train.shape[0])
        losses = []
        for start in range(0, len(perm), cfg.train_batch_size):
            batch = torch.from_numpy(perm[start : start + cfg.train_batch_size]).long().to(device)
            logits = model(x_train[batch]).squeeze(-1)
            loss = loss_fn(logits, y_train[batch])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu().item()))
        model.eval()
        with torch.no_grad():
            val_logits = model(x_val).squeeze(-1)
            val_prob = torch.sigmoid(val_logits)
            val_pred = val_prob >= 0.5
            tp = int(((val_pred == 1) & (y_val == 1)).sum().item())
            fn = int(((val_pred == 0) & (y_val == 1)).sum().item())
            recall = tp / max(1, tp + fn)
            history.append({"epoch": epoch + 1, "loss": float(np.mean(losses)), "val_recall_at_0_5": recall})
    state_path = OUTPUT_DIR / "model_state.pt"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_dim": cfg.feature_dim,
            "preset": "tiny_funnel_mlp",
            "history": history,
        },
        state_path,
    )
    onnx_path = OUTPUT_DIR / "model.onnx"
    onnx_status = "not_exported"
    try:
        dummy = torch.zeros(1, cfg.feature_dim, device=device)
        torch.onnx.export(model, dummy, onnx_path, input_names=["features"], output_names=["priority_logit"], opset_version=17)
        onnx_status = "exported"
    except Exception as exc:
        onnx_status = f"failed: {exc}"
    with torch.no_grad():
        heldout_logits = model(torch.from_numpy(splits["heldout_test"]["features"]).float().to(device)).squeeze(-1)
        heldout_scores = torch.sigmoid(heldout_logits).cpu().numpy().astype(np.float32)
    np.save(OUTPUT_DIR / "heldout_scores.npy", heldout_scores)
    return {
        "status": "ok",
        "device": device,
        "state_path": safe_rel(state_path),
        "onnx_path": safe_rel(onnx_path) if onnx_path.exists() else None,
        "onnx_status": onnx_status,
        "history": history,
    }


def exact_calls_for_scores(labels: np.ndarray, costs: np.ndarray, query_ids: np.ndarray, scores: np.ndarray) -> tuple[int, float, list[int]]:
    exact_calls = 0
    exact_work = 0.0
    ranks: list[int] = []
    for q in np.unique(query_ids):
        idx = np.flatnonzero(query_ids == q)
        ordered = idx[np.argsort(-scores[idx], kind="stable")]
        local = labels[ordered] > 0
        if np.any(local):
            first = int(np.flatnonzero(local)[0]) + 1
            exact_calls += first
            exact_work += float(np.sum(costs[ordered[:first]], dtype=np.float64))
            ranks.append(first)
        else:
            exact_calls += int(ordered.size)
            exact_work += float(np.sum(costs[ordered], dtype=np.float64))
    return exact_calls, exact_work, ranks


def benchmark_methods(splits: dict[str, dict[str, np.ndarray]], train_result: dict[str, Any], cfg: FunnelCaseConfig) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    held = splits["heldout_test"]
    labels = held["ground_truth"].astype(np.int64)
    costs = held["costs"].astype(np.float64)
    qids = held["ids"][:, 1].astype(np.uint64)
    total_calls = int(labels.shape[0])
    total_work = float(np.sum(costs, dtype=np.float64))
    rng = np.random.default_rng(cfg.seed + 456)
    heuristic = (
        8.0 * held["features"][:, 15]
        + 6.0 * held["features"][:, 14]
        + 3.0 * held["features"][:, 23]
        + 1.2 * held["features"][:, 10]
        - 2.5 * held["features"][:, 7]
    ).astype(np.float32)
    learned_scores_path = OUTPUT_DIR / "heldout_scores.npy"
    if learned_scores_path.exists():
        learned = np.load(learned_scores_path)
    else:
        learned = heuristic
    score_map = {
        "NoProposal": np.zeros_like(heuristic),
        "RandomSTPF": rng.random(total_calls).astype(np.float32),
        "RTSTPFExact": learned + 0.15 * heuristic,
    }
    rows: list[dict[str, Any]] = []
    for method, scores in score_map.items():
        if method == "NoProposal":
            calls = total_calls
            work = total_work
            ranks: list[int] = []
        else:
            calls, work, ranks = exact_calls_for_scores(labels, costs, qids, scores)
        rows.append(
            {
                "method": method,
                "groups": int(len(np.unique(qids))),
                "candidates": total_calls,
                "positive_candidates": int(labels.sum()),
                "exact_calls": int(calls),
                "skipped_exact_calls": int(total_calls - calls),
                "exact_call_reduction": float(1.0 - calls / max(1, total_calls)),
                "exact_work": float(work),
                "exact_work_reduction": float(1.0 - work / max(1.0, total_work)),
                "first_positive_rank_mean": float(np.mean(ranks)) if ranks else None,
                "fn": 0,
                "correctness_rule": "positive group early-stop; negative/uncertain group conservative all-exact fallback",
            }
        )
    summary = {
        "heldout_candidates": total_calls,
        "heldout_groups": int(len(np.unique(qids))),
        "heldout_positive_candidates": int(labels.sum()),
        "best_method": max(rows, key=lambda r: r["exact_work_reduction"])["method"],
        "train_result": train_result,
    }
    return rows, summary


def write_reports(cfg: FunnelCaseConfig, sim: dict[str, Any], manifest: dict[str, Any], train_result: dict[str, Any], bench_rows: list[dict[str, Any]], render_paths: dict[str, str]) -> None:
    benchmark_json = BENCHMARK_DIR / f"{RUN_TAG}.json"
    benchmark_csv = BENCHMARK_DIR / f"{RUN_TAG}.csv"
    benchmark_md = BENCHMARK_DIR / f"{RUN_TAG}.md"
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_tag": RUN_TAG,
        "config": asdict(cfg),
        "dataset_manifest": manifest,
        "train_result": train_result,
        "benchmark_rows": bench_rows,
        "simulation": {
            "first_contact_time": sim["first_contact_time"],
            "total_funnel_contacts": sim["total_funnel_contacts"],
            "total_pair_contacts": sim["total_pair_contacts"],
            "physics_audit": sim["physics_audit"],
        },
        "render_outputs": render_paths,
    }
    write_json(benchmark_json, payload)
    with benchmark_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(bench_rows[0].keys()))
        writer.writeheader()
        writer.writerows(bench_rows)
    lines = [
        f"# Repeated Sphere Funnel Drop Benchmark ({RUN_TAG})",
        "",
        "## Case",
        "",
        f"- `{cfg.particle_count}` wave-colored steel beads are injected from the same upper-rim nozzle in `{cfg.wave_count}` waves of `{cfg.particle_count // cfg.wave_count}` beads.",
        "- The funnel uses a wide-mouth/narrow-throat geometry: large inlet radius with a small outlet neck, increasing throat-level contact density.",
        "- Initial velocities are tangent to the funnel circumference, reversed toward the opposite wall, with a small down-wall component; gravity then drives circular sliding into the throat.",
        "- Contact witnesses follow the Scalable-CCD supplementary style: transparent orange/violet regions show dense contact areas without hiding beads, and crosses/rings mark bead-funnel, collector, and bead-bead CCD witness points.",
        "- The funnel is procedural; final CCD labels are generated from signed funnel/contact certificates and conservative fallback semantics.",
        "",
        "## Simulation / Contact Audit",
        "",
        f"- Particle count: `{cfg.particle_count}`.",
        f"- Wave count: `{cfg.wave_count}`.",
        f"- First funnel contact time: `{sim['first_contact_time']}` s.",
        f"- Funnel/collector contact witness count: `{sim['total_funnel_contacts']}`.",
        f"- Particle-particle contact witness count: `{sim['total_pair_contacts']}`.",
        f"- Energy after all particles spawned: `{sim['physics_audit']['energy_joules_after_all_spawn']:.3f}` J; final energy: `{sim['physics_audit']['energy_joules_final']:.3f}` J; drop: `{100.0 * sim['physics_audit']['energy_drop_fraction_after_all_spawn']:.2f}%`.",
        f"- Peak mechanical-energy increase after all spawned: `{100.0 * sim['physics_audit']['energy_peak_increase_fraction_after_all_spawn']:.4f}%` (checks for non-physical post-spawn energy injection).",
        f"- Mean tangential speed after all spawned / final: `{sim['physics_audit']['mean_tangential_speed_after_all_spawn']:.3f}` / `{sim['physics_audit']['mean_tangential_speed_final']:.3f}` m/s.",
        f"- Final below-throat / in-collector fraction: `{100.0 * sim['physics_audit']['final_below_throat_fraction']:.2f}%` / `{100.0 * sim['physics_audit']['final_in_collector_fraction']:.2f}%`.",
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
            "python src/tools/repeated_contact_case_suite_run_id.py --case sphere_funnel --train --benchmark --render",
            "```",
        ]
    )
    benchmark_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (MYDEMO_DIR / "case_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(MYDEMO_DIR / "metrics.json", payload)
    (MYDEMO_DIR / "run_command.txt").write_text(
        "conda activate cudadev\npython src/tools/repeated_contact_case_suite_run_id.py --case sphere_funnel --train --benchmark --render\n",
        encoding="utf-8",
    )
    (MYDEMO_DIR / "resume_notes.md").write_text(
        "# Resume Notes\n\nRe-run the command in `run_command.txt`. Existing outputs are deterministic for the configured seed.\n",
        encoding="utf-8",
    )


def run_sphere_funnel(args: argparse.Namespace) -> None:
    ensure_dirs()
    cfg = FunnelCaseConfig()
    t0 = time.perf_counter()
    sim = simulate(cfg)
    render_paths: dict[str, str] = {}
    if args.render:
        render_paths = render_outputs(sim, cfg)
    splits, manifest = build_candidate_dataset(sim, cfg)
    train_result: dict[str, Any] = {"status": "skipped"}
    if args.train:
        train_result = train_tiny_stpf(splits, cfg)
    bench_rows: list[dict[str, Any]] = []
    if args.benchmark:
        bench_rows, _ = benchmark_methods(splits, train_result, cfg)
    write_reports(cfg, sim, manifest, train_result, bench_rows, render_paths)
    elapsed = time.perf_counter() - t0
    print(json.dumps({"run_tag": RUN_TAG, "elapsed_seconds": elapsed, "mydemo_dir": safe_rel(MYDEMO_DIR)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=["sphere_funnel"], default="sphere_funnel")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    if not (args.train or args.benchmark or args.render):
        args.train = args.benchmark = args.render = True
    run_sphere_funnel(args)


if __name__ == "__main__":
    main()
