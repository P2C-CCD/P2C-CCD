from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any


CCD_ADAPTER_SCHEMA_VERSION = 1
Vec3 = tuple[float, float, float]


class CCDQueryFamily(StrEnum):
    VERTEX_FACE = "vf"
    EDGE_EDGE = "ee"

    @property
    def p2cccd_witness_family(self) -> str:
        if self is CCDQueryFamily.VERTEX_FACE:
            return "point_triangle"
        if self is CCDQueryFamily.EDGE_EDGE:
            return "edge_edge"
        raise ValueError(f"unknown CCD query family: {self}")


@dataclass(frozen=True, slots=True)
class SourceLicense:
    name: str
    license_path: Path | None
    url: str
    terms: str = ""

    def available(self) -> bool:
        return self.license_path is not None and self.license_path.exists()

    def require_available(self) -> SourceLicense:
        if not self.name:
            raise ValueError("SourceLicense.name is required")
        if not self.url:
            raise ValueError("SourceLicense.url is required")
        if not self.terms:
            raise ValueError("SourceLicense.terms is required")
        if self.license_path is None:
            raise FileNotFoundError(f"{self.name} license path is not configured")
        if not self.license_path.exists():
            raise FileNotFoundError(f"{self.name} license file is missing: {self.license_path}")
        return self


@dataclass(frozen=True, slots=True)
class DatasetScene:
    schema_version: int
    source_name: str
    scene_name: str
    scene_path: Path
    frames: tuple[Path, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema_version != CCD_ADAPTER_SCHEMA_VERSION:
            raise ValueError("unsupported DatasetScene schema_version")
        if not self.source_name:
            raise ValueError("DatasetScene.source_name is required")
        if not self.scene_name:
            raise ValueError("DatasetScene.scene_name is required")


@dataclass(frozen=True, slots=True)
class ExternalCCDQuery:
    schema_version: int
    source_name: str
    scene_name: str
    batch_id: str
    query_id: int
    source_query_index: int
    family: CCDQueryFamily
    vertices_t0: tuple[Vec3, Vec3, Vec3, Vec3]
    vertices_t1: tuple[Vec3, Vec3, Vec3, Vec3]
    ground_truth_collides: bool | None = None
    box_pair: tuple[int, int] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema_version != CCD_ADAPTER_SCHEMA_VERSION:
            raise ValueError("unsupported ExternalCCDQuery schema_version")
        if self.query_id < 0:
            raise ValueError("ExternalCCDQuery.query_id must be non-negative")
        if self.source_query_index < 0:
            raise ValueError("ExternalCCDQuery.source_query_index must be non-negative")
        if len(self.vertices_t0) != 4 or len(self.vertices_t1) != 4:
            raise ValueError("CCD queries must store four vertices at t0 and t1")
        for vertex in (*self.vertices_t0, *self.vertices_t1):
            if len(vertex) != 3:
                raise ValueError("CCD query vertices must be 3D")


@dataclass(frozen=True, slots=True)
class DatasetQueryBatch:
    schema_version: int
    source_name: str
    scene_name: str
    batch_id: str
    family: CCDQueryFamily
    queries: tuple[ExternalCCDQuery, ...]
    frame_t0: Path | None = None
    frame_t1: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_query_batch(self)

    @property
    def query_count(self) -> int:
        return len(self.queries)

    @property
    def collision_count(self) -> int:
        return sum(1 for query in self.queries if query.ground_truth_collides is True)

    @property
    def known_label_count(self) -> int:
        return sum(1 for query in self.queries if query.ground_truth_collides is not None)


def validate_query_batch(batch: DatasetQueryBatch) -> DatasetQueryBatch:
    if batch.schema_version != CCD_ADAPTER_SCHEMA_VERSION:
        raise ValueError("unsupported DatasetQueryBatch schema_version")
    if not batch.source_name:
        raise ValueError("DatasetQueryBatch.source_name is required")
    if not batch.scene_name:
        raise ValueError("DatasetQueryBatch.scene_name is required")
    if not batch.batch_id:
        raise ValueError("DatasetQueryBatch.batch_id is required")
    seen_query_ids: set[int] = set()
    for query in batch.queries:
        if query.schema_version != batch.schema_version:
            raise ValueError("query schema_version must match batch schema_version")
        if query.source_name != batch.source_name:
            raise ValueError("query source_name must match batch source_name")
        if query.scene_name != batch.scene_name:
            raise ValueError("query scene_name must match batch scene_name")
        if query.batch_id != batch.batch_id:
            raise ValueError("query batch_id must match batch batch_id")
        if query.family != batch.family:
            raise ValueError("query family must match batch family")
        if query.query_id in seen_query_ids:
            raise ValueError("query_id values must be unique within a batch")
        seen_query_ids.add(query.query_id)
    return batch
