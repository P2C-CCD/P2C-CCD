from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any


CAD_ADAPTER_SCHEMA_VERSION = 1
Vec3 = tuple[float, float, float]


class CADSourceKind(StrEnum):
    ABC = "abc"
    BETTER_STEP = "better_step"
    FUSION_360 = "fusion360"


@dataclass(frozen=True, slots=True)
class CadMeshStats:
    vertex_count: int
    face_count: int
    bounds_min: Vec3
    bounds_max: Vec3
    diagonal: float
    file_size_bytes: int

    def __post_init__(self) -> None:
        if self.vertex_count < 0:
            raise ValueError("CadMeshStats.vertex_count must be non-negative")
        if self.face_count < 0:
            raise ValueError("CadMeshStats.face_count must be non-negative")
        if self.diagonal < 0.0:
            raise ValueError("CadMeshStats.diagonal must be non-negative")
        if self.file_size_bytes < 0:
            raise ValueError("CadMeshStats.file_size_bytes must be non-negative")


@dataclass(frozen=True, slots=True)
class CadMeshAsset:
    schema_version: int
    source_name: str
    asset_id: str
    asset_path: Path
    mesh_format: str
    stats: CadMeshStats
    patch_metadata: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema_version != CAD_ADAPTER_SCHEMA_VERSION:
            raise ValueError("unsupported CadMeshAsset schema_version")
        if not self.source_name:
            raise ValueError("CadMeshAsset.source_name is required")
        if not self.asset_id:
            raise ValueError("CadMeshAsset.asset_id is required")
        if not self.mesh_format:
            raise ValueError("CadMeshAsset.mesh_format is required")

    @property
    def has_patch_metadata(self) -> bool:
        return bool(self.patch_metadata)


@dataclass(frozen=True, slots=True)
class CadMeshPair:
    schema_version: int
    source_name: str
    pair_id: str
    asset_a: CadMeshAsset
    asset_b: CadMeshAsset
    hardness_score: float
    patch_pair_count: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema_version != CAD_ADAPTER_SCHEMA_VERSION:
            raise ValueError("unsupported CadMeshPair schema_version")
        if not self.source_name:
            raise ValueError("CadMeshPair.source_name is required")
        if not self.pair_id:
            raise ValueError("CadMeshPair.pair_id is required")
        if self.asset_a.asset_id == self.asset_b.asset_id:
            raise ValueError("CadMeshPair requires two distinct assets")
        if self.hardness_score < 0.0 or self.hardness_score > 1.0:
            raise ValueError("CadMeshPair.hardness_score must lie in [0, 1]")
        if self.patch_pair_count < 0:
            raise ValueError("CadMeshPair.patch_pair_count must be non-negative")


@dataclass(frozen=True, slots=True)
class CadSequence:
    schema_version: int
    source_name: str
    sequence_name: str
    sequence_path: Path
    assets: tuple[CadMeshAsset, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema_version != CAD_ADAPTER_SCHEMA_VERSION:
            raise ValueError("unsupported CadSequence schema_version")
        if not self.source_name:
            raise ValueError("CadSequence.source_name is required")
        if not self.sequence_name:
            raise ValueError("CadSequence.sequence_name is required")


@dataclass(frozen=True, slots=True)
class CadAssemblyMotionSample:
    schema_version: int
    source_name: str
    sequence_name: str
    sample_id: str
    pair: CadMeshPair
    center_a_t0: Vec3
    center_a_t1: Vec3
    center_b_t0: Vec3
    center_b_t1: Vec3
    motion_type: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema_version != CAD_ADAPTER_SCHEMA_VERSION:
            raise ValueError("unsupported CadAssemblyMotionSample schema_version")
        if not self.source_name:
            raise ValueError("CadAssemblyMotionSample.source_name is required")
        if not self.sequence_name:
            raise ValueError("CadAssemblyMotionSample.sequence_name is required")
        if not self.sample_id:
            raise ValueError("CadAssemblyMotionSample.sample_id is required")
        if not self.motion_type:
            raise ValueError("CadAssemblyMotionSample.motion_type is required")


@dataclass(frozen=True, slots=True)
class StepNativeAsset:
    schema_version: int
    source_name: str
    asset_id: str
    step_path: Path
    file_size_bytes: int
    entity_count: int
    product_names: tuple[str, ...] = ()
    schema_names: tuple[str, ...] = ()
    unit_names: tuple[str, ...] = ()
    sidecar_metadata: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema_version != CAD_ADAPTER_SCHEMA_VERSION:
            raise ValueError("unsupported StepNativeAsset schema_version")
        if not self.source_name:
            raise ValueError("StepNativeAsset.source_name is required")
        if not self.asset_id:
            raise ValueError("StepNativeAsset.asset_id is required")
        if self.file_size_bytes < 0:
            raise ValueError("StepNativeAsset.file_size_bytes must be non-negative")
        if self.entity_count < 0:
            raise ValueError("StepNativeAsset.entity_count must be non-negative")


@dataclass(frozen=True, slots=True)
class StepPreprocessRecord:
    schema_version: int
    source_name: str
    record_id: str
    asset: StepNativeAsset
    output_stem: str
    target_mesh_format: str = "obj"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema_version != CAD_ADAPTER_SCHEMA_VERSION:
            raise ValueError("unsupported StepPreprocessRecord schema_version")
        if not self.source_name:
            raise ValueError("StepPreprocessRecord.source_name is required")
        if not self.record_id:
            raise ValueError("StepPreprocessRecord.record_id is required")
        if not self.output_stem:
            raise ValueError("StepPreprocessRecord.output_stem is required")
        if self.target_mesh_format not in {"obj", "ply", "stl", "off"}:
            raise ValueError("StepPreprocessRecord.target_mesh_format is unsupported")
