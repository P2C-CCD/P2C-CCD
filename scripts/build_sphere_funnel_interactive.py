from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO = Path(__file__).resolve().parents[1]
CASE_SCRIPT = REPO / "src" / "tools" / "repeated_contact_case_suite_run_id.py"
OUTPUT = REPO / "assets" / "interactive" / "sphere_funnel" / "space_data.js"


def load_case_module() -> Any:
    module_name = "sphere_funnel_case_source"
    spec = importlib.util.spec_from_file_location(module_name, CASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load case module: {CASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def q(value: float, digits: int = 4) -> float:
    return round(float(value), digits)


def map_point(point: np.ndarray | list[float] | tuple[float, float, float]) -> list[float]:
    return [q(point[0]), q(point[2]), q(point[1])]


def flat_points(points: np.ndarray) -> list[float]:
    out: list[float] = []
    for point in points:
        out.extend(map_point(point))
    return out


def ring(radius: float, z: float, segments: int) -> np.ndarray:
    return np.asarray(
        [[radius * math.cos(a), radius * math.sin(a), z] for a in np.linspace(0.0, 2.0 * math.pi, segments, endpoint=False)],
        dtype=np.float64,
    )


def add_ring_mesh(vertices: list[list[float]], triangles: list[int], rings: list[np.ndarray]) -> None:
    start = len(vertices)
    for ring_points in rings:
        for point in ring_points:
            vertices.append(map_point(point))
    segment_count = len(rings[0])
    for ring_id in range(len(rings) - 1):
        base0 = start + ring_id * segment_count
        base1 = start + (ring_id + 1) * segment_count
        for i in range(segment_count):
            a = base0 + i
            b = base0 + (i + 1) % segment_count
            c = base1 + i
            d = base1 + (i + 1) % segment_count
            triangles.extend([a, c, b, b, c, d])


def build_funnel_mesh(case: Any, cfg: Any) -> dict[str, list[float] | list[int]]:
    segment_count = 128
    vertices: list[list[float]] = []
    triangles: list[int] = []

    funnel_rings: list[np.ndarray] = []
    for z in np.linspace(-cfg.funnel_throat_length, cfg.funnel_height, 16):
        radius = cfg.funnel_throat_radius if z < 0.0 else case.funnel_radius_at_z(float(z), cfg)
        funnel_rings.append(ring(float(radius), float(z), segment_count))
    add_ring_mesh(vertices, triangles, funnel_rings)

    cup_rings: list[np.ndarray] = []
    for z in np.linspace(-cfg.funnel_throat_length - cfg.collector_depth, -cfg.funnel_throat_length, 5):
        cup_rings.append(ring(cfg.collector_radius, float(z), segment_count))
    add_ring_mesh(vertices, triangles, cup_rings)

    floor_start = len(vertices)
    radial_steps = 8
    for radial_id in range(radial_steps + 1):
        radius = cfg.collector_radius * radial_id / radial_steps
        for angle in np.linspace(0.0, 2.0 * math.pi, segment_count, endpoint=False):
            xy = np.asarray([[radius * math.cos(angle), radius * math.sin(angle)]], dtype=np.float64)
            z = float(case.collector_floor_z(xy, cfg)[0])
            vertices.append(map_point([xy[0, 0], xy[0, 1], z]))

    for radial_id in range(radial_steps):
        base0 = floor_start + radial_id * segment_count
        base1 = floor_start + (radial_id + 1) * segment_count
        for i in range(segment_count):
            a = base0 + i
            b = base0 + (i + 1) % segment_count
            c = base1 + i
            d = base1 + (i + 1) % segment_count
            if radial_id == 0:
                triangles.extend([a, c, d])
            else:
                triangles.extend([a, c, b, b, c, d])

    return {"vertices": [coord for point in vertices for coord in point], "triangles": triangles}


def simulate_with_progress(case: Any, cfg: Any) -> dict[str, Any]:
    pos, vel, radii, spawn, wave_ids = case.initialize_particles(cfg)
    initial_pos = pos.copy()
    initial_vel = vel.copy()
    dt = 1.0 / (cfg.render_fps * cfg.sim_substeps_per_frame)
    frames: list[np.ndarray] = []
    frame_velocities: list[np.ndarray] = []
    frame_active: list[np.ndarray] = []
    funnel_contacts: list[np.ndarray] = []
    pair_contacts: list[np.ndarray] = []
    funnel_contact_counts: list[int] = []
    pair_contact_counts: list[int] = []
    active = np.zeros(cfg.particle_count, dtype=bool)
    first_contact_time: float | None = None
    total_funnel_contacts = 0
    total_pair_contacts = 0

    for frame_idx in range(cfg.render_frames):
        for substep in range(cfg.sim_substeps_per_frame):
            t = (frame_idx * cfg.sim_substeps_per_frame + substep) * dt
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
                f_count, _ = case.resolve_funnel_contacts(pos, vel, radii, active, cfg, dt)
                p_count = 0
                for _ in range(cfg.pair_iterations):
                    iter_count, _ = case.resolve_particle_pairs(pos, vel, radii, active, cfg)
                    if iter_count == 0:
                        break
                    p_count += iter_count
                if f_count and first_contact_time is None:
                    first_contact_time = float(t)
                total_funnel_contacts += f_count
                total_pair_contacts += p_count

        frames.append(pos.copy())
        frame_velocities.append(vel.copy())
        frame_active.append(active.copy())
        probe_pos = pos.copy()
        probe_vel = vel.copy()
        f_count, f_pts = case.resolve_funnel_contacts(probe_pos, probe_vel, radii, active, cfg, dt)
        p_count, p_pts = case.resolve_particle_pairs(probe_pos, probe_vel, radii, active, cfg)
        funnel_contacts.append(f_pts[:240])
        pair_contacts.append(p_pts[:160])
        funnel_contact_counts.append(int(f_count))
        pair_contact_counts.append(int(p_count))

        if frame_idx == 0 or (frame_idx + 1) % 24 == 0 or frame_idx + 1 == cfg.render_frames:
            print(f"sim frame {frame_idx + 1}/{cfg.render_frames}", flush=True)

    audit = case.physics_audit(frames, frame_velocities, frame_active, radii, spawn, cfg)
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


def build_frames(sim: dict[str, Any], cfg: Any) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    positions = sim["positions"]
    active = sim["active"]
    for frame_idx in range(cfg.render_frames):
        ids = np.flatnonzero(active[frame_idx]).astype(np.uint32)
        clouds: list[dict[str, Any]] = []
        fpts = np.asarray(sim["funnel_contacts"][frame_idx], dtype=np.float64)
        if fpts.size:
            clouds.append({"points": flat_points(fpts[:120]), "color": [255, 139, 42], "alpha": 0.88, "size": 5.2, "shape": "circle"})
        ppts = np.asarray(sim["pair_contacts"][frame_idx], dtype=np.float64)
        if ppts.size:
            clouds.append({"points": flat_points(ppts[:72]), "color": [168, 85, 247], "alpha": 0.76, "size": 4.6, "shape": "circle"})

        frame: dict[str, Any] = {
            "label": f"{frame_idx / cfg.render_fps:.2f} s | {len(ids)}/{cfg.particle_count} particles",
            "surfaces": [
                {
                    "mesh": "funnel_collector",
                    "color": [178, 190, 205],
                    "alpha": 0.32,
                    "wireAlpha": 0.24,
                    "strokeWidth": 0.55,
                    "cull": False,
                }
            ],
            "packedBalls": {
                "ids": ids.tolist(),
                "centers": flat_points(positions[frame_idx, ids]) if ids.size else [],
                "alpha": 0.9,
                "strokeAlpha": 0.24,
                "strokeWidth": 0.75,
            },
        }
        if clouds:
            frame["clouds"] = clouds
        frames.append(frame)
        if frame_idx == 0 or (frame_idx + 1) % 72 == 0 or frame_idx + 1 == cfg.render_frames:
            print(f"pack frame {frame_idx + 1}/{cfg.render_frames}", flush=True)
    return frames


def main() -> None:
    case = load_case_module()
    cfg = case.FunnelCaseConfig()
    sim = simulate_with_progress(case, cfg)
    data = {
        "title": "Sphere-Funnel Accumulation",
        "subtitle": "Interactive replay of the original 800-particle, 8-wave procedural funnel case.",
        "hint": "Drag to orbit 360 deg | wheel to zoom | slider to scrub",
        "help": "The viewer keeps the original bead radii, wave IDs, launch schedule, funnel throat, and collector geometry while packing frame data for browser playback.",
        "accent": "#e07a1f",
        "background": "#f7f9fd",
        "camera": {"yaw": 0.72, "pitch": -0.22, "radius": 6.6, "minRadius": 3.4, "maxRadius": 12.5},
        "grid": {"size": 3.1, "divisions": 24, "majorEvery": 2, "y": q(-cfg.funnel_throat_length - cfg.collector_depth)},
        "focalScale": 0.92,
        "metadata": {
            "particleCount": int(cfg.particle_count),
            "waveCount": int(cfg.wave_count),
            "renderFrames": int(cfg.render_frames),
            "renderFps": int(cfg.render_fps),
            "allSpawnedFrame": int(sim["physics_audit"]["all_spawned_frame"]),
        },
        "ballPalette": [
            [72, 163, 238],
            [250, 204, 21],
            [94, 214, 148],
            [244, 113, 98],
            [168, 133, 255],
            [45, 212, 191],
            [251, 146, 60],
            [236, 72, 153],
        ],
        "ballRadii": [q(v, 5) for v in sim["radii"]],
        "ballWaveIds": [int(v) for v in sim["wave_ids"]],
        "meshes": {"funnel_collector": build_funnel_mesh(case, cfg)},
        "frames": build_frames(sim, cfg),
        "initialFrame": int(sim["physics_audit"]["all_spawned_frame"]),
        "playbackMs": 42,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("window.P2C_SPACE_DATA = " + json.dumps(data, ensure_ascii=True, separators=(",", ":")) + ";\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "frames": len(data["frames"]), "particles": cfg.particle_count, "bytes": OUTPUT.stat().st_size}, indent=2), flush=True)


if __name__ == "__main__":
    main()
