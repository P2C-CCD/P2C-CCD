from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from p2cccd.datasets.cad.contracts import Vec3
from p2cccd.datasets.cad.mesh_io import is_supported_mesh_path, mesh_stats_from_file, stable_asset_id

from .contracts import OBJECT_ADAPTER_SCHEMA_VERSION, ObjectMeshAsset, ObjectMotionSample


def load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object in {path}")
    return data


def center_from_asset(asset: ObjectMeshAsset) -> Vec3:
    return tuple(
        0.5 * (asset.stats.bounds_min[index] + asset.stats.bounds_max[index])
        for index in range(3)
    )  # type: ignore[return-value]


def add_vec(lhs: Vec3, rhs: Vec3) -> Vec3:
    return (lhs[0] + rhs[0], lhs[1] + rhs[1], lhs[2] + rhs[2])


def sub_vec(lhs: Vec3, rhs: Vec3) -> Vec3:
    return (lhs[0] - rhs[0], lhs[1] - rhs[1], lhs[2] - rhs[2])


def scale_vec(vec: Vec3, scalar: float) -> Vec3:
    return (vec[0] * scalar, vec[1] * scalar, vec[2] * scalar)


def unit_or_x_axis(vec: Vec3) -> Vec3:
    length = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2])
    if length <= 1.0e-12:
        return (1.0, 0.0, 0.0)
    return (vec[0] / length, vec[1] / length, vec[2] / length)


def mesh_paths_under(root: Path, *, limit: int | None = None) -> tuple[Path, ...]:
    paths: list[Path] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and is_supported_mesh_path(path):
            paths.append(path)
            if limit is not None and len(paths) >= limit:
                break
    return tuple(paths)


def object_asset_from_mesh(
    *,
    source_name: str,
    root: Path,
    mesh_path: Path,
    object_name: str | None = None,
    category: str = "unknown",
    part_count: int = 1,
    dirty_score: float = 0.0,
    metadata: dict[str, Any] | None = None,
) -> ObjectMeshAsset:
    stats = mesh_stats_from_file(mesh_path)
    return ObjectMeshAsset(
        schema_version=OBJECT_ADAPTER_SCHEMA_VERSION,
        source_name=source_name,
        object_id=stable_asset_id(mesh_path, root),
        object_name=object_name or mesh_path.parent.name or mesh_path.stem,
        mesh_path=mesh_path,
        mesh_format=mesh_path.suffix.lower().lstrip("."),
        stats=stats,
        category=category,
        part_count=max(1, part_count),
        dirty_score=dirty_score,
        metadata={
            "source_relative_path": mesh_path.relative_to(root).as_posix(),
            **dict(metadata or {}),
        },
    )


def approach_motion_sample(
    *,
    source_name: str,
    sample_id: str,
    asset_a: ObjectMeshAsset,
    asset_b: ObjectMeshAsset,
    motion_type: str,
    approach_fraction: float,
    metadata: dict[str, Any] | None = None,
) -> ObjectMotionSample:
    center_a = center_from_asset(asset_a)
    center_b = center_from_asset(asset_b)
    direction = unit_or_x_axis(sub_vec(center_b, center_a))
    step = approach_fraction * max(asset_a.stats.diagonal, asset_b.stats.diagonal, 1.0)
    return ObjectMotionSample(
        schema_version=OBJECT_ADAPTER_SCHEMA_VERSION,
        source_name=source_name,
        sample_id=sample_id,
        asset_a=asset_a,
        asset_b=asset_b,
        center_a_t0=center_a,
        center_a_t1=add_vec(center_a, scale_vec(direction, step)),
        center_b_t0=center_b,
        center_b_t1=center_b,
        motion_type=motion_type,
        metadata=dict(metadata or {}),
    )
