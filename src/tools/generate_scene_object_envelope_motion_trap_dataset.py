#!/usr/bin/env python3
"""Generate a native object-envelope motion-trap CCD dataset.

The scenes are full binary PLY frame pairs. Each disconnected component is a
rigid triangulated sphere. Paired objects exchange positions between adjacent
frames, so contact occurs during the sweep even though endpoint proximity is a
weak cue. This is intended to stress native scene/object-envelope schedulers,
not Scalable-CCD query replay.
"""

from __future__ import annotations

import argparse
from math import cos, pi, sin
from pathlib import Path
import struct
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "datasets" / "native_scene_object_envelope_motion_trap_run_id"


def sphere_template(lat: int, lon: int) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]]:
    vertices: list[tuple[float, float, float]] = [(0.0, 0.0, 1.0)]
    for i in range(1, lat):
        theta = pi * i / lat
        z = cos(theta)
        r = sin(theta)
        for j in range(lon):
            phi = 2.0 * pi * j / lon
            vertices.append((r * cos(phi), r * sin(phi), z))
    bottom = len(vertices)
    vertices.append((0.0, 0.0, -1.0))

    faces: list[tuple[int, int, int]] = []
    first = 1
    for j in range(lon):
        faces.append((0, first + (j + 1) % lon, first + j))
    for i in range(lat - 2):
        row0 = 1 + i * lon
        row1 = row0 + lon
        for j in range(lon):
            a = row0 + j
            b = row0 + (j + 1) % lon
            c = row1 + j
            d = row1 + (j + 1) % lon
            faces.append((a, c, b))
            faces.append((b, c, d))
    last_row = 1 + (lat - 2) * lon
    for j in range(lon):
        faces.append((bottom, last_row + j, last_row + (j + 1) % lon))
    return vertices, faces


def transform(
    vertices: Iterable[tuple[float, float, float]],
    *,
    center: tuple[float, float, float],
    radius: float,
) -> list[tuple[float, float, float]]:
    cx, cy, cz = center
    return [(cx + radius * x, cy + radius * y, cz + radius * z) for x, y, z in vertices]


def write_binary_ply(path: Path, vertices: list[tuple[float, float, float]], faces: list[tuple[int, int, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(vertices)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        f"element face {len(faces)}\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
    ).encode("ascii")
    with path.open("wb") as handle:
        handle.write(header)
        for vertex in vertices:
            handle.write(struct.pack("<fff", *vertex))
        for face in faces:
            handle.write(struct.pack("<Biii", 3, *face))


def build_scene(
    scene_dir: Path,
    *,
    pair_count: int,
    y_offset: float,
    radius_jitter: float,
    distance_jitter: float,
    lat: int,
    lon: int,
) -> dict[str, int]:
    base_vertices, base_faces = sphere_template(lat=lat, lon=lon)
    frames = [([], []), ([], [])]  # (vertices, faces)
    for pair in range(pair_count):
        radius = 0.45 + radius_jitter * ((pair % 5) - 2)
        distance = 1.85 + distance_jitter * ((pair % 4) - 1.5)
        y = y_offset + pair * 3.2
        z = 0.15 * (pair % 3)
        centers0 = [(-distance, y, z), (distance, y, z)]
        centers1 = [(distance, y, z), (-distance, y, z)]
        for obj in range(2):
            for frame_index, centers in enumerate((centers0, centers1)):
                vertices_out, faces_out = frames[frame_index]
                start = len(vertices_out)
                vertices_out.extend(transform(base_vertices, center=centers[obj], radius=radius))
                faces_out.extend((a + start, b + start, c + start) for a, b, c in base_faces)
    frames_dir = scene_dir / "frames"
    write_binary_ply(frames_dir / "0.ply", frames[0][0], frames[0][1])
    write_binary_ply(frames_dir / "1.ply", frames[1][0], frames[1][1])
    return {
        "objects": pair_count * 2,
        "vertices": len(frames[0][0]),
        "faces": len(frames[0][1]),
        "pairs": pair_count,
        "sphere_lat": lat,
        "sphere_lon": lon,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--train-pairs", type=int, default=6)
    parser.add_argument("--heldout-pairs", type=int, default=8)
    parser.add_argument("--lat", type=int, default=8)
    parser.add_argument("--lon", type=int, default=16)
    args = parser.parse_args()

    output_root = args.output_root
    train = build_scene(
        output_root / "motion-trap-train",
        pair_count=args.train_pairs,
        y_offset=0.0,
        radius_jitter=0.015,
        distance_jitter=0.06,
        lat=args.lat,
        lon=args.lon,
    )
    heldout = build_scene(
        output_root / "motion-trap-heldout",
        pair_count=args.heldout_pairs,
        y_offset=0.4,
        radius_jitter=0.02,
        distance_jitter=0.08,
        lat=args.lat,
        lon=args.lon,
    )
    summary = {
        "output_root": output_root.as_posix(),
        "train_scene": "motion-trap-train",
        "heldout_scene": "motion-trap-heldout",
        "train": train,
        "heldout": heldout,
        "format": "binary_little_endian PLY adjacent full-scene frames",
        "protocol": "object-object native swept envelope; self-object pairs are excluded by the benchmark runner",
    }
    (output_root / "dataset_summary.json").write_text(__import__("json").dumps(summary, indent=2), encoding="utf-8")
    print(output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
