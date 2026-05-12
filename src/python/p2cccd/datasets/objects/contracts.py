from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from p2cccd.datasets.cad.contracts import CadMeshStats, Vec3


OBJECT_ADAPTER_SCHEMA_VERSION = 1


class ObjectSourceKind(StrEnum):
    YCB = "ycb"
    GOOGLE_SCANNED_OBJECTS = "google_scanned_objects"
    THINGI10K = "thingi10k"
    PARTNET = "partnet"
    PARTNET_MOBILITY = "partnet_mobility"
    SHAPENET = "shapenet"
    OBJAVERSE_XL = "objaverse_xl"


@dataclass(frozen=True, slots=True)
class ObjectMeshAsset:
    schema_version: int
    source_name: str
    object_id: str
    object_name: str
    mesh_path: Path
    mesh_format: str
    stats: CadMeshStats
    category: str = "unknown"
    part_count: int = 1
    dirty_score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema_version != OBJECT_ADAPTER_SCHEMA_VERSION:
            raise ValueError("unsupported ObjectMeshAsset schema_version")
        if not self.source_name:
            raise ValueError("ObjectMeshAsset.source_name is required")
        if not self.object_id:
            raise ValueError("ObjectMeshAsset.object_id is required")
        if not self.object_name:
            raise ValueError("ObjectMeshAsset.object_name is required")
        if not self.mesh_format:
            raise ValueError("ObjectMeshAsset.mesh_format is required")
        if self.part_count <= 0:
            raise ValueError("ObjectMeshAsset.part_count must be positive")
        if self.dirty_score < 0.0 or self.dirty_score > 1.0:
            raise ValueError("ObjectMeshAsset.dirty_score must lie in [0, 1]")


@dataclass(frozen=True, slots=True)
class ObjectMotionSample:
    schema_version: int
    source_name: str
    sample_id: str
    asset_a: ObjectMeshAsset
    asset_b: ObjectMeshAsset
    center_a_t0: Vec3
    center_a_t1: Vec3
    center_b_t0: Vec3
    center_b_t1: Vec3
    motion_type: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema_version != OBJECT_ADAPTER_SCHEMA_VERSION:
            raise ValueError("unsupported ObjectMotionSample schema_version")
        if not self.source_name:
            raise ValueError("ObjectMotionSample.source_name is required")
        if not self.sample_id:
            raise ValueError("ObjectMotionSample.sample_id is required")
        if self.asset_a.object_id == self.asset_b.object_id:
            raise ValueError("ObjectMotionSample requires two distinct assets")
        if not self.motion_type:
            raise ValueError("ObjectMotionSample.motion_type is required")


@dataclass(frozen=True, slots=True)
class ArticulatedObjectScene:
    schema_version: int
    source_name: str
    scene_id: str
    scene_path: Path
    assets: tuple[ObjectMeshAsset, ...]
    joint_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema_version != OBJECT_ADAPTER_SCHEMA_VERSION:
            raise ValueError("unsupported ArticulatedObjectScene schema_version")
        if not self.source_name:
            raise ValueError("ArticulatedObjectScene.source_name is required")
        if not self.scene_id:
            raise ValueError("ArticulatedObjectScene.scene_id is required")
        if self.joint_count < 0:
            raise ValueError("ArticulatedObjectScene.joint_count must be non-negative")
