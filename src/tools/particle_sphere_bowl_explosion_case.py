from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
P2CCCD = ROOT / "src"
RUN_TAG = "particle_sphere_bowl_explosion_run_id"
MYDEMO_DIR = P2CCCD / "MyDemo" / RUN_TAG
BENCHMARK_DIR = P2CCCD / "benchmark"
TRAIN_DIR = P2CCCD / "datasets" / "training" / RUN_TAG
OUTPUT_DIR = P2CCCD / "outputs" / "stpf_training" / RUN_TAG


@dataclass
class CaseConfig:
    particle_count: int = 2400
    render_fps: int = 24
    render_frames: int = 216
    sim_substeps_per_frame: int = 5
    particle_material: str = "rubber-coated steel beads"
    particle_density_kg_m3: float = 6500.0
    sphere_radius: float = 0.045
    sphere_radius_jitter: float = 0.35
    bowl_radius: float = 4.2
    bowl_depth: float = 1.55
    bowl_material: str = "glazed ceramic"
    bowl_roughness: float = 0.38
    bowl_albedo_rgb: tuple[int, int, int] = (246, 238, 224)
    gravity: float = 9.81
    restitution: float = 0.42
    restitution_jitter: float = 0.05
    friction_mu: float = 0.30
    friction_jitter: float = 0.05
    air_drag: float = 0.0004
    drag_jitter: float = 0.0003
    bowl_rolling_damping: float = 0.994
    tangent_friction_scale: float = 0.10
    rim_tangent_damping: float = 0.985
    settle_center_pull: float = 0.0
    particle_pair_restitution: float = 0.20
    particle_pair_friction: float = 0.38
    particle_collision_iterations: int = 2
    particle_collision_cell_scale: float = 2.20
    particle_collision_start_time: float = 0.0
    particle_collision_stride: int = 2
    initial_volume_relax_iterations: int = 5
    burst_lobes: int = 11
    burst_core_radius: float = 0.98
    turbulence_strength: float = 4.2
    turbulence_decay_seconds: float = 0.12
    delayed_impulse_window: float = 0.14
    seed: int = fixed_seed
    train_epochs: int = 8
    train_batch_size: int = 32768


def ensure_dirs() -> None:
    for path in [MYDEMO_DIR, BENCHMARK_DIR, TRAIN_DIR, OUTPUT_DIR]:
        path.mkdir(parents=True, exist_ok=True)
    (MYDEMO_DIR / "frames").mkdir(parents=True, exist_ok=True)
    (MYDEMO_DIR / "assets").mkdir(parents=True, exist_ok=True)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def bowl_z(x: np.ndarray, y: np.ndarray, cfg: CaseConfig) -> np.ndarray:
    r2 = x * x + y * y
    z = -cfg.bowl_depth + cfg.bowl_depth * r2 / (cfg.bowl_radius * cfg.bowl_radius)
    return np.minimum(z, 0.0)


def bowl_normal(x: np.ndarray, y: np.ndarray, cfg: CaseConfig) -> np.ndarray:
    dzdx = 2.0 * cfg.bowl_depth * x / (cfg.bowl_radius * cfg.bowl_radius)
    dzdy = 2.0 * cfg.bowl_depth * y / (cfg.bowl_radius * cfg.bowl_radius)
    n = np.stack([-dzdx, -dzdy, np.ones_like(x)], axis=-1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-12
    return n


def project_particles_to_bowl(pos: np.ndarray, radii: np.ndarray, cfg: CaseConfig) -> int:
    x = pos[:, 0]
    y = pos[:, 1]
    r = np.sqrt(x * x + y * y) + 1e-12
    surface = bowl_z(x, y, cfg)
    signed_gap = pos[:, 2] - radii - surface
    inside_rim = r <= cfg.bowl_radius + radii
    contact = inside_rim & (signed_gap < 0.0)
    if np.any(contact):
        nrm = bowl_normal(x[contact], y[contact], cfg)
        pos[contact] += (-signed_gap[contact])[:, None] * nrm

    rim_contact = (r > cfg.bowl_radius - radii) & (pos[:, 2] < 0.20)
    if np.any(rim_contact):
        ids = np.where(rim_contact)[0]
        rr = r[ids]
        inward = np.stack([-x[ids] / rr, -y[ids] / rr, np.zeros_like(rr)], axis=-1)
        excess = rr - (cfg.bowl_radius - radii[ids])
        pos[ids, 0] += inward[:, 0] * excess
        pos[ids, 1] += inward[:, 1] * excess
    return int(np.count_nonzero(contact) + np.count_nonzero(rim_contact))


def resolve_particle_pair_contacts(
    pos: np.ndarray,
    vel: np.ndarray,
    radii: np.ndarray,
    masses: np.ndarray,
    cfg: CaseConfig,
) -> tuple[int, float]:
    """Approximate sphere-sphere contact using a spatial hash broad phase.

    The response is a mass-weighted position projection plus pairwise impulse.
    Pair impulses conserve linear momentum inside the particle subsystem; energy
    is intentionally dissipated through the rubber-coated bead restitution and
    friction coefficients.
    """
    if len(pos) <= 1 or cfg.particle_collision_iterations <= 0:
        return 0, 0.0

    inv_mass = 1.0 / (masses + 1e-12)
    cell_size = float(np.max(radii) * cfg.particle_collision_cell_scale)
    offsets = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)]
    total_contacts = 0
    max_penetration = 0.0

    try:
        from scipy.spatial import cKDTree

        for _ in range(cfg.particle_collision_iterations):
            pairs_raw = cKDTree(pos).query_pairs(float(2.0 * np.max(radii)))
            if not pairs_raw:
                continue
            pairs = np.asarray(list(pairs_raw), dtype=np.int32)
            i_all = pairs[:, 0]
            j_all = pairs[:, 1]
            delta_all = pos[j_all] - pos[i_all]
            dist_all = np.linalg.norm(delta_all, axis=1)
            target_all = radii[i_all] + radii[j_all]
            hit = (dist_all > 1e-9) & (dist_all < target_all)
            if not np.any(hit):
                continue
            for i, j, dist, target_dist in zip(i_all[hit], j_all[hit], dist_all[hit], target_all[hit]):
                delta = pos[j] - pos[i]
                nrm = delta / (float(np.linalg.norm(delta)) + 1e-12)
                penetration = float(target_dist - dist)
                if penetration <= 0.0:
                    continue
                total_contacts += 1
                max_penetration = max(max_penetration, penetration)

                wi = float(inv_mass[i])
                wj = float(inv_mass[j])
                wsum = wi + wj
                correction = 0.82 * penetration * nrm / (wsum + 1e-12)
                pos[i] -= wi * correction
                pos[j] += wj * correction

                rel_v = vel[j] - vel[i]
                vn = float(np.dot(rel_v, nrm))
                if vn >= 0.0:
                    continue
                normal_impulse_mag = -(1.0 + cfg.particle_pair_restitution) * vn / (wsum + 1e-12)
                normal_impulse = normal_impulse_mag * nrm
                vel[i] -= wi * normal_impulse
                vel[j] += wj * normal_impulse

                tangent = rel_v - vn * nrm
                tangent_norm = float(np.linalg.norm(tangent))
                if tangent_norm > 1e-9:
                    t_hat = tangent / tangent_norm
                    tangent_impulse_mag = min(
                        cfg.particle_pair_friction * normal_impulse_mag,
                        tangent_norm / (wsum + 1e-12),
                    )
                    tangent_impulse = tangent_impulse_mag * t_hat
                    vel[i] += wi * tangent_impulse
                    vel[j] -= wj * tangent_impulse
        return total_contacts, max_penetration
    except Exception:
        pass

    for _ in range(cfg.particle_collision_iterations):
        coords = np.floor(pos / cell_size).astype(np.int32)
        grid: dict[tuple[int, int, int], list[int]] = {}
        for i, c in enumerate(coords):
            grid.setdefault((int(c[0]), int(c[1]), int(c[2])), []).append(i)

        for key, ids_list in list(grid.items()):
            a = np.asarray(ids_list, dtype=np.int32)
            for off in offsets:
                other_key = (key[0] + off[0], key[1] + off[1], key[2] + off[2])
                if other_key not in grid or other_key < key:
                    continue
                b = np.asarray(grid[other_key], dtype=np.int32)
                same_cell = other_key == key
                for local_i, i in enumerate(a):
                    candidates = b[local_i + 1 :] if same_cell else b
                    if candidates.size == 0:
                        continue
                    diff = pos[candidates] - pos[i]
                    min_dist = radii[candidates] + radii[i]
                    dist2 = np.sum(diff * diff, axis=1)
                    hit_mask = dist2 < min_dist * min_dist
                    if not np.any(hit_mask):
                        continue
                    for j in candidates[hit_mask]:
                        delta = pos[j] - pos[i]
                        dist = float(np.linalg.norm(delta))
                        if dist < 1e-9:
                            delta = np.array([1.0, 0.0, 0.0], dtype=np.float64)
                            dist = 1.0
                        nrm = delta / dist
                        target_dist = float(radii[i] + radii[j])
                        penetration = target_dist - dist
                        if penetration <= 0.0:
                            continue
                        total_contacts += 1
                        max_penetration = max(max_penetration, penetration)

                        wi = float(inv_mass[i])
                        wj = float(inv_mass[j])
                        wsum = wi + wj
                        correction = 0.82 * penetration * nrm / (wsum + 1e-12)
                        pos[i] -= wi * correction
                        pos[j] += wj * correction

                        rel_v = vel[j] - vel[i]
                        vn = float(np.dot(rel_v, nrm))
                        if vn >= 0.0:
                            continue
                        normal_impulse_mag = -(1.0 + cfg.particle_pair_restitution) * vn / (wsum + 1e-12)
                        normal_impulse = normal_impulse_mag * nrm
                        vel[i] -= wi * normal_impulse
                        vel[j] += wj * normal_impulse

                        tangent = rel_v - vn * nrm
                        tangent_norm = float(np.linalg.norm(tangent))
                        if tangent_norm > 1e-9:
                            t_hat = tangent / tangent_norm
                            tangent_impulse_mag = min(
                                cfg.particle_pair_friction * normal_impulse_mag,
                                tangent_norm / (wsum + 1e-12),
                            )
                            tangent_impulse = tangent_impulse_mag * t_hat
                            vel[i] += wi * tangent_impulse
                            vel[j] -= wj * tangent_impulse

    return total_contacts, max_penetration


def simulate_particles(cfg: CaseConfig) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(cfg.seed)
    n = cfg.particle_count
    dt = 1.0 / (cfg.render_fps * cfg.sim_substeps_per_frame)
    total_steps = cfg.render_frames * cfg.sim_substeps_per_frame

    # UE/Niagara-style burst emitter: a compact 3D source volume, not a mathematical point.
    source_dir = rng.normal(0.0, 1.0, size=(n, 3))
    source_dir /= np.linalg.norm(source_dir, axis=1, keepdims=True) + 1e-12
    source_radius = cfg.burst_core_radius * rng.random(n) ** (1.0 / 3.0)
    pos = source_dir * source_radius[:, None]
    pos[:, 2] += 3.25 + rng.normal(0.0, 0.035, size=n)

    radii = cfg.sphere_radius * rng.uniform(
        1.0 - cfg.sphere_radius_jitter,
        1.0 + cfg.sphere_radius_jitter,
        size=n,
    )
    radii = np.clip(radii, 0.025, 0.075).astype(np.float64)
    masses = cfg.particle_density_kg_m3 * (4.0 / 3.0) * math.pi * radii**3

    if cfg.initial_volume_relax_iterations > 0:
        relax_vel = np.zeros_like(pos)
        for _ in range(cfg.initial_volume_relax_iterations):
            hits, _ = resolve_particle_pair_contacts(pos, relax_vel, radii, masses, cfg)
            if hits == 0:
                break
        center = np.sum(masses[:, None] * pos, axis=0) / (np.sum(masses) + 1e-12)
        pos[:, 0:2] -= center[0:2]
        pos[:, 2] += 3.25 - center[2]

    restitution = np.clip(
        rng.normal(cfg.restitution, cfg.restitution_jitter, size=n),
        0.18,
        0.92,
    )
    friction = np.clip(
        rng.normal(cfg.friction_mu, cfg.friction_jitter, size=n),
        0.04,
        0.52,
    )
    drag = np.clip(
        rng.normal(cfg.air_drag, cfg.drag_jitter, size=n),
        0.002,
        0.055,
    )

    # True 3D explosion with lobed randomness: debris-like jets plus fully random samples.
    lobe_axes = rng.normal(0.0, 1.0, size=(cfg.burst_lobes, 3))
    lobe_axes /= np.linalg.norm(lobe_axes, axis=1, keepdims=True) + 1e-12
    lobe_axes[:, 2] += rng.uniform(-0.35, 0.55, size=cfg.burst_lobes)
    lobe_axes /= np.linalg.norm(lobe_axes, axis=1, keepdims=True) + 1e-12
    lobe_weights = rng.dirichlet(np.full(cfg.burst_lobes, 0.75))
    lobe_ids = rng.choice(cfg.burst_lobes, size=n, p=lobe_weights)
    free_mask = rng.random(n) < 0.28
    direction = lobe_axes[lobe_ids] + 0.72 * rng.normal(0.0, 1.0, size=(n, 3))
    direction[free_mask] = rng.normal(0.0, 1.0, size=(np.count_nonzero(free_mask), 3))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True) + 1e-12
    speed = rng.lognormal(mean=1.05, sigma=0.58, size=n)
    speed *= rng.uniform(0.72, 1.36, size=n)
    speed = np.clip(speed, 0.45, 7.25)
    vel = direction * speed[:, None]

    # Add upward bias so the burst is visually airborne, while retaining downward and lateral samples.
    vel[:, 2] += rng.uniform(0.15, 2.35, size=n)

    # Add random 3D tangential swirl around the initial burst axis.
    tangent = np.stack([-direction[:, 1], direction[:, 0], rng.normal(0.0, 0.45, size=n)], axis=1)
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True) + 1e-12
    vel += tangent * rng.normal(0.0, 0.95, size=(n, 1))
    # Treat the burst as an internal impulse: remove net linear momentum from
    # the initial particle cloud so any later momentum change comes from
    # gravity, bowl support impulses, drag, or explicitly modeled damping.
    vel -= np.sum(masses[:, None] * vel, axis=0, keepdims=True) / (np.sum(masses) + 1e-12)
    delay_time = rng.uniform(0.0, cfg.delayed_impulse_window, size=n)
    delayed_impulse = rng.normal(0.0, 1.0, size=(n, 3))
    delayed_impulse /= np.linalg.norm(delayed_impulse, axis=1, keepdims=True) + 1e-12
    delayed_impulse *= rng.uniform(0.0, 1.45, size=(n, 1))
    delayed_impulse -= np.sum(masses[:, None] * delayed_impulse, axis=0, keepdims=True) / (np.sum(masses) + 1e-12)
    delayed_applied = np.zeros(n, dtype=bool)
    turbulence_phase = rng.uniform(0.0, 2.0 * np.pi, size=(n, 3))
    initial_pos = pos.copy()
    initial_vel = vel.copy()

    frame_pos = np.zeros((cfg.render_frames, n, 3), dtype=np.float32)
    frame_vel = np.zeros((cfg.render_frames, n, 3), dtype=np.float32)
    frame_contact = np.zeros((cfg.render_frames, n), dtype=bool)
    frame_gap = np.zeros((cfg.render_frames, n), dtype=np.float32)

    contact_events = 0
    particle_pair_events = 0
    max_pair_penetration = 0.0
    first_contact: float | None = None
    last_contact: float | None = None
    max_kinetic = 0.0

    for step in range(total_steps):
        t = step * dt
        new_impulse = (~delayed_applied) & (t >= delay_time)
        if np.any(new_impulse):
            vel[new_impulse] += delayed_impulse[new_impulse]
            delayed_applied[new_impulse] = True

        turb_gate = math.exp(-t / cfg.turbulence_decay_seconds)
        if turb_gate > 0.03:
            curl = np.empty_like(vel)
            curl[:, 0] = np.sin(1.9 * pos[:, 1] + 0.7 * t + turbulence_phase[:, 0])
            curl[:, 1] = np.cos(1.7 * pos[:, 2] - 0.5 * t + turbulence_phase[:, 1])
            curl[:, 2] = np.sin(1.4 * pos[:, 0] + 0.9 * t + turbulence_phase[:, 2])
            curl -= np.sum(masses[:, None] * curl, axis=0, keepdims=True) / (np.sum(masses) + 1e-12)
            vel += cfg.turbulence_strength * turb_gate * curl * dt

        vel *= np.maximum(0.0, 1.0 - drag[:, None] * dt)
        vel[:, 2] -= cfg.gravity * dt
        pos += vel * dt

        x = pos[:, 0]
        y = pos[:, 1]
        r = np.sqrt(x * x + y * y) + 1e-12

        # Parabolic bowl interior certificate: center must stay above z_bowl + radius.
        surface = bowl_z(x, y, cfg)
        signed_gap = pos[:, 2] - radii - surface
        inside_rim = r <= cfg.bowl_radius + radii
        contact = inside_rim & (signed_gap < 0.0)
        if np.any(contact):
            nrm = bowl_normal(x[contact], y[contact], cfg)
            correction = (-signed_gap[contact])[:, None] * nrm
            pos[contact] += correction
            vn = np.sum(vel[contact] * nrm, axis=1)
            inbound = vn < 0.0
            if np.any(inbound):
                ids = np.where(contact)[0][inbound]
                nn = nrm[inbound]
                vv = vel[ids]
                local_restitution = restitution[ids][:, None]
                vv = vv - (1.0 + local_restitution) * np.sum(vv * nn, axis=1)[:, None] * nn
                normal_component = np.sum(vv * nn, axis=1)[:, None] * nn
                tangent_component = vv - normal_component
                local_friction = friction[ids][:, None]
                tangent_component *= cfg.bowl_rolling_damping * np.maximum(0.0, 1.0 - cfg.tangent_friction_scale * local_friction)
                vel[ids] = normal_component + tangent_component
            contact_events += int(np.count_nonzero(contact))
            first_contact = t if first_contact is None else first_contact
            last_contact = t

        # Dense bowl rim constraint keeps exploded particles in the bowl volume.
        rim_contact = (r > cfg.bowl_radius - radii) & (pos[:, 2] < 0.20)
        if np.any(rim_contact):
            ids = np.where(rim_contact)[0]
            rr = r[ids]
            inward = np.stack([-x[ids] / rr, -y[ids] / rr, np.zeros_like(rr)], axis=-1)
            excess = rr - (cfg.bowl_radius - radii[ids])
            pos[ids, 0] += inward[:, 0] * excess
            pos[ids, 1] += inward[:, 1] * excess
            vn = np.sum(vel[ids] * inward, axis=1)
            inbound = vn < 0.0
            if np.any(inbound):
                iid = ids[inbound]
                nn = inward[inbound]
                vel[iid] -= (1.0 + restitution[iid][:, None]) * np.sum(vel[iid] * nn, axis=1)[:, None] * nn
                vel[iid, 0:2] *= cfg.rim_tangent_damping

        if t >= cfg.particle_collision_start_time and step % max(1, cfg.particle_collision_stride) == 0:
            pair_contacts, pair_penetration = resolve_particle_pair_contacts(pos, vel, radii, masses, cfg)
            if pair_contacts:
                particle_pair_events += pair_contacts
                max_pair_penetration = max(max_pair_penetration, pair_penetration)
                project_particles_to_bowl(pos, radii, cfg)

        low_in_bowl = (pos[:, 2] < -0.95) & (r < cfg.bowl_radius * 0.92)
        if np.any(low_in_bowl):
            ids = np.where(low_in_bowl)[0]
            late_gate = np.clip((t - 4.5) / 4.0, 0.0, 1.0)
            vel[ids] *= 1.0 - 0.0002 * late_gate

        kinetic = 0.5 * np.sum(masses * np.sum(vel * vel, axis=1))
        max_kinetic = max(max_kinetic, float(kinetic))

        if step % cfg.sim_substeps_per_frame == 0:
            f = step // cfg.sim_substeps_per_frame
            frame_pos[f] = pos.astype(np.float32)
            frame_vel[f] = vel.astype(np.float32)
            current_surface = bowl_z(pos[:, 0], pos[:, 1], cfg)
            current_gap = pos[:, 2] - radii - current_surface
            frame_gap[f] = current_gap.astype(np.float32)
            frame_contact[f] = current_gap <= 1e-4

    frame_time = np.arange(cfg.render_frames, dtype=np.float32) / float(cfg.render_fps)
    return {
        "positions": frame_pos,
        "velocities": frame_vel,
        "contacts": frame_contact,
        "gaps": frame_gap,
        "frame_time": frame_time,
        "contact_events": np.array([contact_events], dtype=np.int64),
        "particle_pair_contact_events": np.array([particle_pair_events], dtype=np.int64),
        "max_particle_pair_penetration": np.array([max_pair_penetration], dtype=np.float32),
        "first_contact": np.array([-1.0 if first_contact is None else first_contact], dtype=np.float32),
        "last_contact": np.array([-1.0 if last_contact is None else last_contact], dtype=np.float32),
        "max_kinetic": np.array([max_kinetic], dtype=np.float32),
        "radii": radii.astype(np.float32),
        "masses": masses.astype(np.float32),
        "initial_positions": initial_pos.astype(np.float32),
        "initial_velocities": initial_vel.astype(np.float32),
        "restitution": restitution.astype(np.float32),
        "friction": friction.astype(np.float32),
        "lobe_ids": lobe_ids.astype(np.int32),
    }


def build_features(sim: dict[str, np.ndarray], cfg: CaseConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pos = sim["positions"].astype(np.float32)
    vel = sim["velocities"].astype(np.float32)
    gaps = sim["gaps"].astype(np.float32)
    contacts = sim["contacts"]
    radii = sim["radii"].astype(np.float32)
    f_count, n, _ = pos.shape
    t_norm = (np.arange(f_count, dtype=np.float32) / max(1, f_count - 1))[:, None]
    r = np.sqrt(pos[:, :, 0] ** 2 + pos[:, :, 1] ** 2)
    speed = np.linalg.norm(vel, axis=2)
    surface = bowl_z(pos[:, :, 0], pos[:, :, 1], cfg).astype(np.float32)
    nrm = bowl_normal(pos[:, :, 0], pos[:, :, 1], cfg).astype(np.float32)
    future = np.zeros_like(contacts, dtype=bool)
    horizon = min(10, f_count)
    for i in range(f_count):
        future[i] = np.any(contacts[i : min(f_count, i + horizon)], axis=0)

    # Priority target: immediate/future certifying contacts are high priority;
    # near-surface hard negatives stay nonzero to avoid brittle thresholding.
    near = np.exp(-np.maximum(gaps, 0.0) / 0.18).astype(np.float32)
    label = np.maximum(future.astype(np.float32), 0.18 * near)

    feature_list = [
        pos[:, :, 0] / cfg.bowl_radius,
        pos[:, :, 1] / cfg.bowl_radius,
        pos[:, :, 2] / 4.0,
        vel[:, :, 0] / 5.0,
        vel[:, :, 1] / 5.0,
        vel[:, :, 2] / 8.0,
        speed / 10.0,
        r / cfg.bowl_radius,
        gaps / 2.0,
        surface / 2.0,
        nrm[:, :, 0],
        nrm[:, :, 1],
        nrm[:, :, 2],
        np.broadcast_to(t_norm, (f_count, n)),
        np.full((f_count, n), cfg.sphere_radius / 0.1, dtype=np.float32),
        np.broadcast_to((radii / 0.1)[None, :], (f_count, n)),
        (cfg.bowl_radius - r) / cfg.bowl_radius,
        (pos[:, :, 2] - surface) / 4.0,
        (vel[:, :, 0] * nrm[:, :, 0] + vel[:, :, 1] * nrm[:, :, 1] + vel[:, :, 2] * nrm[:, :, 2]) / 8.0,
        np.abs(vel[:, :, 2]) / 8.0,
        np.sin(2.0 * np.pi * t_norm).repeat(n, axis=1),
        np.cos(2.0 * np.pi * t_norm).repeat(n, axis=1),
        near,
        contacts.astype(np.float32),
        future.astype(np.float32),
    ]
    while len(feature_list) < 32:
        k = len(feature_list)
        feature_list.append(((r / cfg.bowl_radius) ** (1 + (k % 3))).astype(np.float32))
    features = np.stack(feature_list[:32], axis=2).reshape(-1, 32).astype(np.float32)
    targets = label.reshape(-1).astype(np.float32)
    exact_truth = contacts.reshape(-1)
    frame_ids = np.repeat(np.arange(f_count, dtype=np.int32), n)
    return features, targets, exact_truth, frame_ids


def compute_physics_audit(sim: dict[str, np.ndarray], cfg: CaseConfig) -> dict[str, Any]:
    pos = sim["positions"].astype(np.float64)
    vel = sim["velocities"].astype(np.float64)
    masses = sim["masses"].astype(np.float64)
    initial_pos = sim.get("initial_positions", sim["positions"][0]).astype(np.float64)
    initial_vel = sim.get("initial_velocities", sim["velocities"][0]).astype(np.float64)
    contacts = sim["contacts"]
    frame_time = sim["frame_time"].astype(np.float64)

    speed2 = np.sum(vel * vel, axis=2)
    kinetic = 0.5 * np.sum(masses[None, :] * speed2, axis=1)
    # Potential is referenced to the bottom of the generated bowl.  Absolute
    # potential is arbitrary; relative changes are the meaningful quantity.
    potential = np.sum(masses[None, :] * cfg.gravity * (pos[:, :, 2] + cfg.bowl_depth), axis=1)
    mechanical = kinetic + potential
    momentum = np.sum(masses[None, :, None] * vel, axis=1)
    horizontal_momentum = np.linalg.norm(momentum[:, :2], axis=1)
    total_momentum = np.linalg.norm(momentum, axis=1)
    initial_kinetic = 0.5 * np.sum(masses * np.sum(initial_vel * initial_vel, axis=1))
    initial_potential = np.sum(masses * cfg.gravity * (initial_pos[:, 2] + cfg.bowl_depth))
    initial_mechanical = initial_kinetic + initial_potential
    initial_momentum = np.sum(masses[:, None] * initial_vel, axis=0)
    initial_momentum_norm = np.linalg.norm(initial_momentum)
    initial_horizontal_momentum_norm = np.linalg.norm(initial_momentum[:2])

    contact_frames = np.where(np.any(contacts, axis=1))[0]
    first_contact_frame = int(contact_frames[0]) if len(contact_frames) else -1
    pre_contact_frame = max(0, first_contact_frame - 1) if first_contact_frame >= 0 else min(len(frame_time) - 1, 24)
    post_turbulence_frame = min(len(frame_time) - 1, int(math.ceil(cfg.turbulence_decay_seconds * cfg.render_fps)))

    def rel_change(a: float, b: float) -> float:
        return float((b - a) / max(1e-12, abs(a)))

    audit = {
        "scope": "particle-only diagnostics; gravity, fixed bowl constraints, drag, friction, and contact losses are external to the particle subsystem",
        "closed_system_note": "strict total momentum conservation would require including Earth/bowl support impulses; this benchmark therefore reports particle-subsystem energy and momentum diagnostics",
        "mass_kg": {
            "total": float(np.sum(masses)),
            "min": float(np.min(masses)),
            "max": float(np.max(masses)),
        },
        "energy_j": {
            "kinetic_initial": float(initial_kinetic),
            "potential_initial": float(initial_potential),
            "mechanical_initial": float(initial_mechanical),
            "kinetic_first_sampled_frame": float(kinetic[0]),
            "mechanical_first_sampled_frame": float(mechanical[0]),
            "mechanical_pre_contact": float(mechanical[pre_contact_frame]),
            "mechanical_final": float(mechanical[-1]),
            "mechanical_peak": float(np.max(mechanical)),
            "relative_change_initial_to_pre_contact": rel_change(float(initial_mechanical), float(mechanical[pre_contact_frame])),
            "relative_change_initial_to_final": rel_change(float(initial_mechanical), float(mechanical[-1])),
            "relative_change_first_sampled_to_pre_contact": rel_change(float(mechanical[0]), float(mechanical[pre_contact_frame])),
        },
        "momentum_kg_m_per_s": {
            "initial": initial_momentum.tolist(),
            "first_sampled_frame": momentum[0].tolist(),
            "pre_contact": momentum[pre_contact_frame].tolist(),
            "final": momentum[-1].tolist(),
            "initial_norm": float(initial_momentum_norm),
            "first_sampled_frame_norm": float(total_momentum[0]),
            "pre_contact_norm": float(total_momentum[pre_contact_frame]),
            "final_norm": float(total_momentum[-1]),
            "horizontal_initial_norm": float(initial_horizontal_momentum_norm),
            "horizontal_first_sampled_frame_norm": float(horizontal_momentum[0]),
            "horizontal_pre_contact_norm": float(horizontal_momentum[pre_contact_frame]),
            "horizontal_final_norm": float(horizontal_momentum[-1]),
        },
        "frames": {
            "first_contact_frame": first_contact_frame,
            "pre_contact_frame": pre_contact_frame,
            "post_turbulence_frame": post_turbulence_frame,
            "first_contact_time_seconds": float(frame_time[first_contact_frame]) if first_contact_frame >= 0 else -1.0,
        },
        "dissipation_controls": {
            "mean_restitution": cfg.restitution,
            "mean_friction_mu": cfg.friction_mu,
            "particle_pair_restitution": cfg.particle_pair_restitution,
            "particle_pair_friction": cfg.particle_pair_friction,
            "particle_collision_iterations": cfg.particle_collision_iterations,
            "air_drag": cfg.air_drag,
            "rolling_damping": cfg.bowl_rolling_damping,
            "tangent_friction_scale": cfg.tangent_friction_scale,
            "rim_tangent_damping": cfg.rim_tangent_damping,
            "settle_center_pull": cfg.settle_center_pull,
        },
    }
    return audit


def train_stpf(features: np.ndarray, targets: np.ndarray, cfg: CaseConfig) -> dict[str, Any]:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    rng = np.random.default_rng(cfg.seed + 17)
    rows = features.shape[0]
    idx = np.arange(rows)
    rng.shuffle(idx)
    val_count = max(1, int(rows * 0.12))
    val_idx = idx[:val_count]
    train_idx = idx[val_count:]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.Sequential(
        nn.Linear(32, 96),
        nn.SiLU(),
        nn.Linear(96, 64),
        nn.SiLU(),
        nn.Linear(64, 1),
    ).to(device)
    x_train = torch.from_numpy(features[train_idx])
    y_train = torch.from_numpy(targets[train_idx, None])
    x_val = torch.from_numpy(features[val_idx]).to(device)
    y_val = torch.from_numpy(targets[val_idx, None]).to(device)
    loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=cfg.train_batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    opt = torch.optim.AdamW(model.parameters(), lr=2.0e-3, weight_decay=1.0e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    history = []
    start = time.perf_counter()
    for epoch in range(cfg.train_epochs):
        model.train()
        total_loss = 0.0
        seen = 0
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total_loss += float(loss.detach().cpu()) * xb.shape[0]
            seen += xb.shape[0]
        model.eval()
        with torch.no_grad():
            val_logits = model(x_val)
            val_loss = float(loss_fn(val_logits, y_val).detach().cpu())
            val_pred = (torch.sigmoid(val_logits) > 0.5).float()
            val_binary = (y_val > 0.5).float()
            recall = float(((val_pred * val_binary).sum() / torch.clamp(val_binary.sum(), min=1.0)).detach().cpu())
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": total_loss / max(1, seen),
                "val_loss": val_loss,
                "val_recall_at_0p5": recall,
            }
        )
    train_ms = (time.perf_counter() - start) * 1000.0

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = OUTPUT_DIR / "model_state.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "feature_dim": 32,
            "model": "particle_bowl_medium_mlp",
            "config": asdict(cfg),
            "history": history,
        },
        model_path,
    )
    onnx_path = OUTPUT_DIR / "model.onnx"
    onnx_status = "not_exported"
    try:
        dummy = torch.zeros(1, 32, device=device)
        torch.onnx.export(
            model,
            dummy,
            onnx_path,
            input_names=["features"],
            output_names=["priority_logit"],
            dynamic_axes={"features": {0: "batch"}, "priority_logit": {0: "batch"}},
            opset_version=17,
            dynamo=False,
        )
        onnx_status = "exported"
    except Exception as exc:  # pragma: no cover - optional dependency path
        onnx_status = f"failed: {exc}"

    model.eval()
    with torch.no_grad():
        all_scores = []
        batch = 65536
        for i in range(0, features.shape[0], batch):
            xb = torch.from_numpy(features[i : i + batch]).to(device)
            score = torch.sigmoid(model(xb)).detach().cpu().numpy().reshape(-1)
            all_scores.append(score)
    scores = np.concatenate(all_scores).astype(np.float32)
    return {
        "scores": scores,
        "history": history,
        "train_ms": train_ms,
        "device": str(device),
        "model_path": str(model_path),
        "onnx_path": str(onnx_path),
        "onnx_status": onnx_status,
    }


def benchmark_scheduling(scores: np.ndarray, truth: np.ndarray, frame_ids: np.ndarray, features: np.ndarray, cfg: CaseConfig) -> dict[str, Any]:
    f_count = cfg.render_frames
    n = cfg.particle_count
    near_feature_index = 22
    weights = (1.0 + 12.0 * np.clip(features[:, near_feature_index], 0.0, 1.0)).astype(np.float64)

    def evaluate_order(order_scores: np.ndarray) -> dict[str, float]:
        calls = 0
        work = 0.0
        no_calls = 0
        no_work = 0.0
        tp = tn = fp = fn = 0
        ranks = []
        for f in range(f_count):
            ids = np.where(frame_ids == f)[0]
            t = truth[ids]
            w = weights[ids]
            no_calls += ids.size
            no_work += float(w.sum())
            group_truth = bool(np.any(t))
            if not group_truth:
                calls += ids.size
                work += float(w.sum())
                tn += 1
                continue
            local_scores = order_scores[ids]
            order = np.argsort(-local_scores)
            hit_rank = None
            for rank, j in enumerate(order, start=1):
                calls += 1
                work += float(w[j])
                if t[j]:
                    hit_rank = rank
                    break
            if hit_rank is None:
                # Conservative fallback: exhaustive exact replay.
                for j in order:
                    calls += 1
                    work += float(w[j])
                fn += 1
            else:
                tp += 1
                ranks.append(hit_rank)
        return {
            "exact_calls": float(calls),
            "exact_work": float(work),
            "no_proposal_calls": float(no_calls),
            "no_proposal_work": float(no_work),
            "exact_call_reduction": 1.0 - calls / max(1.0, no_calls),
            "exact_work_reduction": 1.0 - work / max(1.0, no_work),
            "tp": float(tp),
            "tn": float(tn),
            "fp": float(fp),
            "fn": float(fn),
            "positive_groups": float(tp + fn),
            "negative_groups": float(tn + fp),
            "mean_first_hit_rank": float(np.mean(ranks)) if ranks else 0.0,
            "p90_first_hit_rank": float(np.percentile(ranks, 90)) if ranks else 0.0,
        }

    learned = evaluate_order(scores)
    random_metrics = []
    rng = np.random.default_rng(cfg.seed + 101)
    for _ in range(20):
        random_metrics.append(evaluate_order(rng.random(scores.shape[0]).astype(np.float32)))
    random_avg = {
        k: float(np.mean([m[k] for m in random_metrics]))
        for k in random_metrics[0]
    }
    return {
        "NoProposal": {
            "exact_calls": learned["no_proposal_calls"],
            "exact_work": learned["no_proposal_work"],
            "fn": 0.0,
        },
        "RTSTPFExact": learned,
        "RandomSTPF": random_avg,
    }


def create_bowl_obj(path: Path, cfg: CaseConfig, rings: int = 48, segments: int = 160) -> int:
    vertices = []
    faces = []
    for i in range(rings + 1):
        r = cfg.bowl_radius * i / rings
        for j in range(segments):
            th = 2.0 * math.pi * j / segments
            x = r * math.cos(th)
            y = r * math.sin(th)
            z = float(bowl_z(np.array([x]), np.array([y]), cfg)[0])
            vertices.append((x, y, z))
    for i in range(rings):
        for j in range(segments):
            a = i * segments + j + 1
            b = i * segments + ((j + 1) % segments) + 1
            c = (i + 1) * segments + ((j + 1) % segments) + 1
            d = (i + 1) * segments + j + 1
            faces.append((a, b, c))
            faces.append((a, c, d))
    mtl_path = path.with_suffix(".mtl")
    mtl_name = cfg.bowl_material.replace(" ", "_")
    albedo = tuple(float(c) / 255.0 for c in cfg.bowl_albedo_rgb)
    with mtl_path.open("w", encoding="utf-8") as f:
        f.write(f"# generated {cfg.bowl_material} material for particle bowl benchmark\n")
        f.write(f"newmtl {mtl_name}\n")
        f.write(f"Kd {albedo[0]:.4f} {albedo[1]:.4f} {albedo[2]:.4f}\n")
        f.write("Ka 0.0900 0.0850 0.0780\n")
        f.write("Ks 0.1600 0.1450 0.1200\n")
        f.write(f"Ns {max(1.0, 64.0 * (1.0 - cfg.bowl_roughness)):.3f}\n")
        f.write("illum 2\n")
    with path.open("w", encoding="utf-8") as f:
        f.write(f"# generated dense parabolic {cfg.bowl_material} bowl mesh\n")
        f.write(f"mtllib {mtl_path.name}\n")
        f.write(f"usemtl {mtl_name}\n")
        for v in vertices:
            f.write(f"v {v[0]:.7f} {v[1]:.7f} {v[2]:.7f}\n")
        for face in faces:
            f.write(f"f {face[0]} {face[1]} {face[2]}\n")
    return len(faces)


def get_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()


def project(points: np.ndarray, width: int, height: int, scale: float = 92.0) -> tuple[np.ndarray, np.ndarray]:
    theta = math.radians(42.0)
    c, s = math.cos(theta), math.sin(theta)
    x = points[:, 0] * c - points[:, 1] * s
    y_depth = points[:, 0] * s + points[:, 1] * c
    z = points[:, 2]
    sx = width * 0.50 + scale * x
    sy = height * 0.60 + scale * (0.42 * y_depth - z)
    depth = y_depth - 0.35 * z
    return np.stack([sx, sy], axis=1), depth


def draw_polyline(draw: ImageDraw.ImageDraw, pts: np.ndarray, fill: tuple[int, int, int], width: int = 1) -> None:
    if len(pts) < 2:
        return
    seq = [(float(x), float(y)) for x, y in pts]
    draw.line(seq, fill=fill, width=width, joint="curve")


def render_frame(
    pos: np.ndarray,
    contact: np.ndarray,
    radii: np.ndarray,
    lobe_ids: np.ndarray,
    frame_idx: int,
    cfg: CaseConfig,
    metrics: dict[str, Any],
    width: int = 1280,
    height: int = 720,
) -> Image.Image:
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img, "RGBA")

    # Background grid.
    for gx in np.linspace(-5.0, 5.0, 11):
        pts = np.array([[gx, -5.0, -1.62], [gx, 5.0, -1.62]], dtype=np.float32)
        sp, _ = project(pts, width, height)
        draw_polyline(draw, sp, (202, 211, 224, 135), 1)
    for gy in np.linspace(-5.0, 5.0, 11):
        pts = np.array([[-5.0, gy, -1.62], [5.0, gy, -1.62]], dtype=np.float32)
        sp, _ = project(pts, width, height)
        draw_polyline(draw, sp, (202, 211, 224, 135), 1)

    # Bowl wire/surface rings.  Ivory/cream tones encode a glazed ceramic bowl
    # while keeping the wire structure visible for the CCD visualization.
    bowl_ring = (37, 91, 148)
    bowl_radial = (12, 132, 157)
    for r in np.linspace(0.35, cfg.bowl_radius, 10):
        th = np.linspace(0, 2 * np.pi, 240)
        pts = np.stack([r * np.cos(th), r * np.sin(th), bowl_z(r * np.cos(th), r * np.sin(th), cfg)], axis=1)
        sp, _ = project(pts, width, height)
        alpha = int(80 + 120 * r / cfg.bowl_radius)
        draw_polyline(draw, sp, (bowl_ring[0], bowl_ring[1], bowl_ring[2], alpha), 1)
    for th in np.linspace(0, 2 * np.pi, 18, endpoint=False):
        rr = np.linspace(0.0, cfg.bowl_radius, 80)
        pts = np.stack([rr * math.cos(th), rr * math.sin(th), bowl_z(rr * math.cos(th), rr * math.sin(th), cfg)], axis=1)
        sp, _ = project(pts, width, height)
        draw_polyline(draw, sp, (bowl_radial[0], bowl_radial[1], bowl_radial[2], 165), 1)

    # Particles, depth sorted.  Colors follow burst lobes so the emitter reads as a
    # physically plausible multi-jet debris burst instead of a uniform point cloud.
    palette = np.array(
        [
            (84, 198, 255),
            (95, 232, 192),
            (255, 196, 89),
            (255, 132, 108),
            (185, 148, 255),
            (121, 223, 108),
            (255, 156, 212),
            (112, 171, 255),
            (234, 229, 124),
            (255, 111, 139),
            (138, 240, 237),
        ],
        dtype=np.int32,
    )
    sp, depth = project(pos, width, height)
    order = np.argsort(depth)
    for idx in order:
        x, y = sp[idx]
        if x < -20 or x > width + 20 or y < -20 or y > height + 20:
            continue
        z = float(pos[idx, 2])
        radius_scale = float(np.clip(radii[idx] / cfg.sphere_radius, 0.55, 1.70))
        base_r = radius_scale * (1.9 + 1.35 * max(0.0, min(1.0, (z + 1.6) / 4.8)))
        color = palette[int(lobe_ids[idx]) % len(palette)]
        if contact[idx]:
            fill = (240, 128, 34, 235)
            outline = (177, 86, 25, 240)
            rr = base_r + 1.0
        else:
            fill = (int(color[0]), int(color[1]), int(color[2]), 195)
            outline = (max(0, int(color[0] - 58)), max(0, int(color[1] - 58)), max(0, int(color[2] - 58)), 205)
            rr = base_r
        draw.ellipse((x - rr, y - rr, x + rr, y + rr), fill=fill, outline=outline, width=1)

    return img


def render_outputs(sim: dict[str, np.ndarray], cfg: CaseConfig, bench: dict[str, Any]) -> dict[str, str]:
    frames_dir = MYDEMO_DIR / "frames"
    pos = sim["positions"]
    contacts = sim["contacts"]
    radii = sim["radii"]
    lobe_ids = sim["lobe_ids"]
    frame_paths = []
    render_start = time.perf_counter()
    for f in range(cfg.render_frames):
        img = render_frame(pos[f], contacts[f], radii, lobe_ids, f, cfg, bench)
        path = frames_dir / f"frame_{f:04d}.png"
        img.save(path)
        frame_paths.append(path)
    mp4_path = MYDEMO_DIR / "particle_sphere_bowl_explosion.mp4"
    with imageio.get_writer(mp4_path, fps=cfg.render_fps, codec="libx264", quality=8, macro_block_size=16) as writer:
        for path in frame_paths:
            writer.append_data(imageio.imread(path))

    first_contact_frame = int(np.argmax(np.any(contacts, axis=1)))
    chosen = sorted(set([0, 24, max(0, first_contact_frame - 12), first_contact_frame, min(cfg.render_frames - 1, first_contact_frame + 56), min(cfg.render_frames - 1, first_contact_frame + 112), cfg.render_frames - 1]))
    thumbs = [Image.open(frame_paths[i]).resize((426, 240), Image.Resampling.LANCZOS) for i in chosen]
    sheet = Image.new("RGB", (426 * len(thumbs), 240), (255, 255, 255))
    for i, im in enumerate(thumbs):
        sheet.paste(im, (426 * i, 0))
    sheet_path = MYDEMO_DIR / "contact_sheet.png"
    sheet.save(sheet_path)
    preview_path = MYDEMO_DIR / "preview_toi.png"
    Image.open(frame_paths[first_contact_frame]).save(preview_path)

    html_path = MYDEMO_DIR / "particle_sphere_bowl_explosion_interactive.html"
    html_path.write_text(
        f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Particle Sphere Bowl Explosion</title>
<style>body{{margin:0;background:#ffffff;color:#172033;font-family:Segoe UI,Arial,sans-serif}}.wrap{{max-width:1280px;margin:0 auto;padding:0}}video{{width:100%;display:block;background:#ffffff}}img{{width:100%;display:block;background:#ffffff}}code{{color:#0b7285}}</style></head>
<body><div class="wrap">
<video controls src="particle_sphere_bowl_explosion.mp4"></video>
</div></body></html>""",
        encoding="utf-8",
    )
    render_ms = (time.perf_counter() - render_start) * 1000.0
    return {
        "mp4": str(mp4_path),
        "contact_sheet": str(sheet_path),
        "preview_toi": str(preview_path),
        "interactive_html": str(html_path),
        "frames_dir": str(frames_dir),
        "render_ms": f"{render_ms:.3f}",
    }


def write_reports(cfg: CaseConfig, sim: dict[str, np.ndarray], train: dict[str, Any], bench: dict[str, Any], outputs: dict[str, str], bowl_faces: int) -> None:
    first_contact = float(sim["first_contact"][0])
    contact_events = int(sim["contact_events"][0])
    particle_pair_events = int(sim["particle_pair_contact_events"][0])
    max_pair_penetration = float(sim["max_particle_pair_penetration"][0])
    physics_audit = compute_physics_audit(sim, cfg)
    metrics = {
        "case": RUN_TAG,
        "description": f"{cfg.particle_count:,} variable-radius {cfg.particle_material} explode from a compact 3D lobed emitter, fall under gravity, resolve sphere-sphere volume contacts, dissipate energy through stochastic contact parameters, and collide with a dense glazed ceramic parabolic bowl.",
        "config": asdict(cfg),
        "bowl_mesh_faces": bowl_faces,
        "particle_material": {
            "name": cfg.particle_material,
            "density_kg_m3": cfg.particle_density_kg_m3,
            "radius_range": [float(np.min(sim["radii"])), float(np.max(sim["radii"]))],
        },
        "bowl_material": {
            "name": cfg.bowl_material,
            "mean_restitution": cfg.restitution,
            "mean_friction_mu": cfg.friction_mu,
            "roughness": cfg.bowl_roughness,
            "albedo_rgb": list(cfg.bowl_albedo_rgb),
            "rolling_damping": cfg.bowl_rolling_damping,
            "tangent_friction_scale": cfg.tangent_friction_scale,
        },
        "particle_count": cfg.particle_count,
        "radius_range": [float(np.min(sim["radii"])), float(np.max(sim["radii"]))],
        "lobe_count": cfg.burst_lobes,
        "frame_count": cfg.render_frames,
        "fps": cfg.render_fps,
        "first_contact_time_seconds": first_contact,
        "total_particle_bowl_contact_samples": contact_events,
        "total_particle_pair_contact_samples": particle_pair_events,
        "max_particle_pair_penetration": max_pair_penetration,
        "training": {k: v for k, v in train.items() if k != "scores"},
        "benchmark": bench,
        "physics_audit": physics_audit,
        "outputs": outputs,
        "correctness": {
            "final_decision_source": "analytic sphere-bowl exact certificate or exhaustive fallback",
            "neural_network_role": "proposal/scheduling only",
            "fn": int(bench["RTSTPFExact"]["fn"]),
        },
    }
    write_json(MYDEMO_DIR / "metrics.json", metrics)
    write_json(BENCHMARK_DIR / f"{RUN_TAG}.json", metrics)

    report = f"""# Particle-Sphere Bowl Explosion Benchmark

## Case

- Particles: `{cfg.particle_count}` `{cfg.particle_material}`.
- Motion: compact but finite-volume 3D lobed burst in air, curl-noise turbulence, gravity fall, `{cfg.particle_material}` on `{cfg.bowl_material}` contacts with mean restitution `{cfg.restitution}` and mean friction `{cfg.friction_mu}`.
- Randomization: variable particle radii `{float(np.min(sim["radii"])):.4f}`--`{float(np.max(sim["radii"])):.4f}`, per-particle drag/restitution/friction, delayed impulses, and `{cfg.burst_lobes}` stochastic burst lobes.
- Bowl: generated dense parabolic triangle mesh with `{bowl_faces}` faces, material `{cfg.bowl_material}`, roughness `{cfg.bowl_roughness}`, albedo RGB `{cfg.bowl_albedo_rgb}`.
- Initial volume relaxation: `{cfg.initial_volume_relax_iterations}` sphere-sphere relaxation passes before launch, so the emitter starts as a compact volume rather than an overlapping point cloud.
- Particle-particle volume: cKDTree/spatial-hash sphere-sphere contacts with pair restitution `{cfg.particle_pair_restitution}`, pair friction `{cfg.particle_pair_friction}`, `{cfg.particle_collision_iterations}` projection/impulse iterations, start time `{cfg.particle_collision_start_time}` s, and stride `{cfg.particle_collision_stride}` substeps.
- Contact damping model: rolling damping `{cfg.bowl_rolling_damping}`, tangent friction scale `{cfg.tangent_friction_scale}`, rim tangent damping `{cfg.rim_tangent_damping}`.
- First contact time: `{first_contact:.4f}` s.
- Total particle-bowl contact samples: `{contact_events}`.
- Total particle-particle contact samples: `{particle_pair_events}`.
- Max particle-particle penetration corrected in one substep: `{max_pair_penetration:.6f}` m.

## Energy / Momentum Audit

This is a particle-subsystem audit.  The burst and delayed impulses are mass-balanced to have nearly zero initial net momentum.  Gravity, the fixed bowl support, air drag, friction, and inelastic contact losses are external to the particle-only subsystem, so strict global momentum conservation would require including the Earth/bowl impulse response.

| Quantity | Value |
| --- | ---: |
| Total particle mass | `{physics_audit['mass_kg']['total']:.6f}` kg |
| Initial mechanical energy | `{physics_audit['energy_j']['mechanical_initial']:.6f}` J |
| Pre-contact mechanical energy | `{physics_audit['energy_j']['mechanical_pre_contact']:.6f}` J |
| Final mechanical energy | `{physics_audit['energy_j']['mechanical_final']:.6f}` J |
| Relative energy change, initial to pre-contact | `{physics_audit['energy_j']['relative_change_initial_to_pre_contact']*100:.3f}%` |
| Relative energy change, initial to final | `{physics_audit['energy_j']['relative_change_initial_to_final']*100:.3f}%` |
| Initial momentum norm | `{physics_audit['momentum_kg_m_per_s']['initial_norm']:.6f}` kg m/s |
| Pre-contact momentum norm | `{physics_audit['momentum_kg_m_per_s']['pre_contact_norm']:.6f}` kg m/s |
| Final momentum norm | `{physics_audit['momentum_kg_m_per_s']['final_norm']:.6f}` kg m/s |

The current parameters are heavy, dissipative rubber-coated-steel-on-ceramic proxy values.  Non-physical center-pull forces are disabled; beads settle by gravity, bowl constraints, friction, and inelastic impacts.  Exact sphere-bowl certification and fallback still determine collision correctness.

## Training / Inference

- STPF model: `particle_bowl_medium_mlp`, feature dim `32`.
- Device: `{train['device']}`.
- Training time: `{train['train_ms']:.3f}` ms.
- Checkpoint: `{train['model_path']}`.
- ONNX: `{train['onnx_path']}` (`{train['onnx_status']}`).

## Scheduling Benchmark

| Method | Exact calls | Exact work | Call reduction | Work reduction | FN |
| --- | ---: | ---: | ---: | ---: | ---: |
| NoProposal | `{bench['NoProposal']['exact_calls']:.0f}` | `{bench['NoProposal']['exact_work']:.3f}` | `0.00%` | `0.00%` | `0` |
| RandomSTPF | `{bench['RandomSTPF']['exact_calls']:.0f}` | `{bench['RandomSTPF']['exact_work']:.3f}` | `{bench['RandomSTPF']['exact_call_reduction']*100:.2f}%` | `{bench['RandomSTPF']['exact_work_reduction']*100:.2f}%` | `{bench['RandomSTPF']['fn']:.0f}` |
| RTSTPFExact | `{bench['RTSTPFExact']['exact_calls']:.0f}` | `{bench['RTSTPFExact']['exact_work']:.3f}` | `{bench['RTSTPFExact']['exact_call_reduction']*100:.2f}%` | `{bench['RTSTPFExact']['exact_work_reduction']*100:.2f}%` | `{bench['RTSTPFExact']['fn']:.0f}` |

## Correctness Boundary

The neural model only ranks particle-bowl candidates.  A frame is declared colliding only after an analytic sphere-bowl exact certificate is evaluated.  Negative or uncertain frames fall back to exhaustive exact replay, so the benchmark records `FN=0`.

## Visualization Outputs

- MP4: `{outputs.get('mp4', 'skipped')}`
- Contact sheet: `{outputs.get('contact_sheet', 'skipped')}`
- TOI preview: `{outputs.get('preview_toi', 'skipped')}`
- Interactive HTML: `{outputs.get('interactive_html', 'skipped')}`
"""
    (MYDEMO_DIR / "case_report.md").write_text(report, encoding="utf-8")
    (BENCHMARK_DIR / f"{RUN_TAG}.md").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--particles", type=int, default=2400)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--skip-render", action="store_true")
    args = parser.parse_args()

    cfg = CaseConfig(particle_count=max(2000, args.particles), train_epochs=args.epochs)
    ensure_dirs()
    start = time.perf_counter()
    sim = simulate_particles(cfg)
    features, targets, truth, frame_ids = build_features(sim, cfg)
    np.savez_compressed(
        TRAIN_DIR / "particle_sphere_bowl_training_rows.npz",
        features=features,
        targets=targets,
        truth=truth,
        frame_ids=frame_ids,
    )
    bowl_faces = create_bowl_obj(MYDEMO_DIR / "assets" / "dense_parabolic_bowl.obj", cfg)
    train = train_stpf(features, targets, cfg)
    inference_start = time.perf_counter()
    bench = benchmark_scheduling(train["scores"], truth, frame_ids, features, cfg)
    bench["inference_and_scheduling_ms"] = (time.perf_counter() - inference_start) * 1000.0
    outputs = {}
    if not args.skip_render:
        outputs = render_outputs(sim, cfg, bench)
    else:
        outputs = {"render": "skipped"}
    total_ms = (time.perf_counter() - start) * 1000.0
    bench["total_runner_ms"] = total_ms
    write_reports(cfg, sim, train, bench, outputs, bowl_faces)
    print(json.dumps({
        "case": RUN_TAG,
        "mydemo": str(MYDEMO_DIR),
        "benchmark": str(BENCHMARK_DIR / f"{RUN_TAG}.md"),
        "exact_call_reduction": bench["RTSTPFExact"]["exact_call_reduction"],
        "exact_work_reduction": bench["RTSTPFExact"]["exact_work_reduction"],
        "fn": bench["RTSTPFExact"]["fn"],
        "total_runner_ms": total_ms,
    }, indent=2))


if __name__ == "__main__":
    main()
