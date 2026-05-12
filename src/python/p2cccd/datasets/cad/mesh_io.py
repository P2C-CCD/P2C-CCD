from __future__ import annotations

import hashlib
import math
from pathlib import Path
import struct
from typing import Iterable

from .contracts import CadMeshStats, Vec3


SUPPORTED_MESH_EXTENSIONS = (".obj", ".off", ".stl", ".ply")


def is_supported_mesh_path(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_MESH_EXTENSIONS


def stable_asset_id(path: Path, root: Path) -> str:
    relative = path.resolve().relative_to(root.resolve()).as_posix()
    digest = hashlib.sha1(relative.encode("utf-8")).hexdigest()[:12]
    stem = path.stem.replace(" ", "_")
    return f"{stem}-{digest}"


def _empty_stats(path: Path, *, vertex_count: int = 0, face_count: int = 0) -> CadMeshStats:
    return CadMeshStats(
        vertex_count=vertex_count,
        face_count=face_count,
        bounds_min=(0.0, 0.0, 0.0),
        bounds_max=(0.0, 0.0, 0.0),
        diagonal=0.0,
        file_size_bytes=path.stat().st_size if path.exists() else 0,
    )


def _stats_from_vertices(path: Path, vertices: Iterable[Vec3], face_count: int) -> CadMeshStats:
    vertex_list = list(vertices)
    if not vertex_list:
        return _empty_stats(path, vertex_count=0, face_count=face_count)
    mins = tuple(min(vertex[index] for vertex in vertex_list) for index in range(3))
    maxs = tuple(max(vertex[index] for vertex in vertex_list) for index in range(3))
    diagonal = math.sqrt(sum((maxs[index] - mins[index]) ** 2 for index in range(3)))
    return CadMeshStats(
        vertex_count=len(vertex_list),
        face_count=face_count,
        bounds_min=(float(mins[0]), float(mins[1]), float(mins[2])),
        bounds_max=(float(maxs[0]), float(maxs[1]), float(maxs[2])),
        diagonal=float(diagonal),
        file_size_bytes=path.stat().st_size,
    )


def _parse_obj(path: Path) -> CadMeshStats:
    vertices: list[Vec3] = []
    face_count = 0
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if parts[0] == "v" and len(parts) >= 4:
                vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif parts[0] == "f" and len(parts) >= 4:
                face_count += 1
    return _stats_from_vertices(path, vertices, face_count)


def _parse_off(path: Path) -> CadMeshStats:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        lines = [line.strip() for line in handle if line.strip() and not line.lstrip().startswith("#")]
    if not lines or lines[0] != "OFF":
        raise ValueError(f"unsupported OFF header: {path}")
    if len(lines) < 2:
        raise ValueError(f"missing OFF counts: {path}")
    counts = lines[1].split()
    if len(counts) < 2:
        raise ValueError(f"invalid OFF counts: {path}")
    vertex_count = int(counts[0])
    face_count = int(counts[1])
    vertices: list[Vec3] = []
    for line in lines[2 : 2 + vertex_count]:
        parts = line.split()
        if len(parts) < 3:
            raise ValueError(f"invalid OFF vertex row in {path}")
        vertices.append((float(parts[0]), float(parts[1]), float(parts[2])))
    return _stats_from_vertices(path, vertices, face_count)


def _parse_ascii_stl(path: Path) -> CadMeshStats:
    vertices: list[Vec3] = []
    face_count = 0
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            parts = raw_line.strip().split()
            if len(parts) == 4 and parts[0].lower() == "vertex":
                vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif len(parts) >= 2 and parts[0].lower() == "facet" and parts[1].lower() == "normal":
                face_count += 1
    if not vertices:
        return _empty_stats(path, vertex_count=0, face_count=face_count)
    return _stats_from_vertices(path, vertices, face_count)


def _parse_binary_stl(path: Path) -> CadMeshStats:
    with path.open("rb") as handle:
        header = handle.read(84)
        if len(header) < 84:
            return _empty_stats(path)
        face_count = struct.unpack("<I", header[80:84])[0]
        vertices: list[Vec3] = []
        for _ in range(face_count):
            record = handle.read(50)
            if len(record) < 50:
                break
            values = struct.unpack("<12fH", record)
            vertices.append((float(values[3]), float(values[4]), float(values[5])))
            vertices.append((float(values[6]), float(values[7]), float(values[8])))
            vertices.append((float(values[9]), float(values[10]), float(values[11])))
    if not vertices:
        return _empty_stats(path, vertex_count=0, face_count=face_count)
    return _stats_from_vertices(path, vertices, face_count)


def _parse_stl(path: Path) -> CadMeshStats:
    stats = _parse_ascii_stl(path)
    if stats.vertex_count > 0 or stats.face_count > 0:
        return stats
    return _parse_binary_stl(path)


def _parse_ascii_ply(path: Path) -> CadMeshStats:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        header: list[str] = []
        for raw_line in handle:
            line = raw_line.strip()
            header.append(line)
            if line == "end_header":
                break
        if not header or header[0] != "ply":
            raise ValueError(f"unsupported PLY header: {path}")
        if not any(line == "format ascii 1.0" for line in header):
            return _empty_stats(path)
        vertex_count = 0
        face_count = 0
        for line in header:
            parts = line.split()
            if len(parts) == 3 and parts[:2] == ["element", "vertex"]:
                vertex_count = int(parts[2])
            elif len(parts) == 3 and parts[:2] == ["element", "face"]:
                face_count = int(parts[2])
        vertices: list[Vec3] = []
        for _ in range(vertex_count):
            parts = handle.readline().strip().split()
            if len(parts) < 3:
                raise ValueError(f"invalid PLY vertex row in {path}")
            vertices.append((float(parts[0]), float(parts[1]), float(parts[2])))
    return _stats_from_vertices(path, vertices, face_count)


def mesh_stats_from_file(path: str | Path) -> CadMeshStats:
    mesh_path = Path(path)
    suffix = mesh_path.suffix.lower()
    if suffix == ".obj":
        return _parse_obj(mesh_path)
    if suffix == ".off":
        return _parse_off(mesh_path)
    if suffix == ".stl":
        return _parse_stl(mesh_path)
    if suffix == ".ply":
        return _parse_ascii_ply(mesh_path)
    raise ValueError(f"unsupported CAD mesh format: {mesh_path.suffix}")
