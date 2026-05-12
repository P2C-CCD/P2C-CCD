from __future__ import annotations

import json
import math
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Sequence

from p2cccd.datasets.ccd.contracts import SourceLicense

from .contracts import (
    CAD_ADAPTER_SCHEMA_VERSION,
    CadAssemblyMotionSample,
    CadMeshAsset,
    CadMeshPair,
    CadSequence,
    Vec3,
)
from .mesh_io import is_supported_mesh_path, mesh_stats_from_file, stable_asset_id


FUSION360_SOURCE_NAME = "Fusion 360 Gallery"


def default_fusion360_root() -> Path:
    return Path(__file__).resolve().parents[5] / "datasets" / "fusion360"


def _load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object in {path}")
    return data


def _center(asset: CadMeshAsset) -> Vec3:
    return tuple(
        0.5 * (asset.stats.bounds_min[index] + asset.stats.bounds_max[index])
        for index in range(3)
    )  # type: ignore[return-value]


def _add(lhs: Vec3, rhs: Vec3) -> Vec3:
    return (lhs[0] + rhs[0], lhs[1] + rhs[1], lhs[2] + rhs[2])


def _sub(lhs: Vec3, rhs: Vec3) -> Vec3:
    return (lhs[0] - rhs[0], lhs[1] - rhs[1], lhs[2] - rhs[2])


def _scale(vec: Vec3, scalar: float) -> Vec3:
    return (vec[0] * scalar, vec[1] * scalar, vec[2] * scalar)


def _norm(vec: Vec3) -> float:
    return math.sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2])


def _unit_or_x_axis(vec: Vec3) -> Vec3:
    length = _norm(vec)
    if length <= 1.0e-12:
        return (1.0, 0.0, 0.0)
    return (vec[0] / length, vec[1] / length, vec[2] / length)


def _patch_count(asset: CadMeshAsset) -> int:
    patches = asset.patch_metadata.get("patches")
    if isinstance(patches, list):
        return len(patches)
    return max(1, asset.stats.face_count)


def _pair_score(asset_a: CadMeshAsset, asset_b: CadMeshAsset) -> float:
    diag_a = max(asset_a.stats.diagonal, 1.0e-12)
    diag_b = max(asset_b.stats.diagonal, 1.0e-12)
    return min(1.0, math.exp(-abs(math.log(diag_a / diag_b))))


@dataclass(frozen=True, slots=True)
class Fusion360GalleryAdapter:
    root: Path | None = None

    @property
    def source_root(self) -> Path:
        return self.root if self.root is not None else default_fusion360_root()

    def require_available(self) -> None:
        self.license().require_available()
        if not self.source_root.exists():
            raise FileNotFoundError(f"Fusion 360 Gallery root not found: {self.source_root}")

    def license(self) -> SourceLicense:
        return SourceLicense(
            name=FUSION360_SOURCE_NAME,
            license_path=self.source_root / "LICENSE",
            url="https://github.com/AutodeskAILab/Fusion360GalleryDataset",
            terms="Human-designed CAD assemblies; preserve upstream dataset terms for sequence-derived shards.",
        )

    def _load_asset(self, mesh_path: Path, *, sequence_path: Path) -> CadMeshAsset:
        metadata = _load_json_if_exists(mesh_path.with_suffix(".json"))
        return CadMeshAsset(
            schema_version=CAD_ADAPTER_SCHEMA_VERSION,
            source_name=FUSION360_SOURCE_NAME,
            asset_id=stable_asset_id(mesh_path, self.source_root),
            asset_path=mesh_path,
            mesh_format=mesh_path.suffix.lower().lstrip("."),
            stats=mesh_stats_from_file(mesh_path),
            patch_metadata=metadata.get("patch_metadata", {}) if isinstance(metadata.get("patch_metadata"), dict) else {},
            metadata={
                "source_relative_path": mesh_path.relative_to(self.source_root).as_posix(),
                "sequence_relative_path": sequence_path.relative_to(self.source_root).as_posix(),
                **{key: value for key, value in metadata.items() if key != "patch_metadata"},
            },
        )

    def list_sequences(self, *, limit: int | None = None) -> tuple[CadSequence, ...]:
        self.require_available()
        candidate_dirs = [self.source_root]
        candidate_dirs.extend(path for path in sorted(self.source_root.iterdir()) if path.is_dir())
        sequences: list[CadSequence] = []
        seen: set[Path] = set()
        for sequence_path in candidate_dirs:
            if sequence_path in seen:
                continue
            seen.add(sequence_path)
            mesh_iter = sequence_path.glob("*") if sequence_path == self.source_root else sequence_path.rglob("*")
            mesh_paths = tuple(
                path
                for path in sorted(mesh_iter)
                if path.is_file() and is_supported_mesh_path(path)
            )
            if not mesh_paths:
                continue
            assets = tuple(self._load_asset(path, sequence_path=sequence_path) for path in mesh_paths)
            metadata = _load_json_if_exists(sequence_path / "assembly.json")
            sequences.append(
                CadSequence(
                    schema_version=CAD_ADAPTER_SCHEMA_VERSION,
                    source_name=FUSION360_SOURCE_NAME,
                    sequence_name=sequence_path.relative_to(self.source_root).as_posix()
                    if sequence_path != self.source_root
                    else "root",
                    sequence_path=sequence_path,
                    assets=assets,
                    metadata={
                        "asset_count": len(assets),
                        "has_assembly_json": bool(metadata),
                        **metadata,
                    },
                )
            )
            if limit is not None and len(sequences) >= limit:
                break
        return tuple(sequences)

    def load_sequence(self, sequence_name: str) -> CadSequence:
        for sequence in self.list_sequences():
            if sequence.sequence_name == sequence_name:
                return sequence
        raise FileNotFoundError(f"Fusion 360 sequence not found: {sequence_name}")

    def generate_mesh_pairs(
        self,
        sequence: CadSequence,
        *,
        limit: int | None = None,
    ) -> tuple[CadMeshPair, ...]:
        pairs: list[CadMeshPair] = []
        for asset_a, asset_b in combinations(sequence.assets, 2):
            pairs.append(
                CadMeshPair(
                    schema_version=CAD_ADAPTER_SCHEMA_VERSION,
                    source_name=FUSION360_SOURCE_NAME,
                    pair_id=f"{sequence.sequence_name}:{asset_a.asset_id}__{asset_b.asset_id}",
                    asset_a=asset_a,
                    asset_b=asset_b,
                    hardness_score=_pair_score(asset_a, asset_b),
                    patch_pair_count=_patch_count(asset_a) * _patch_count(asset_b),
                    metadata={
                        "sampling_role": "assembly_pair",
                        "sequence_name": sequence.sequence_name,
                    },
                )
            )
        pairs.sort(key=lambda pair: (-pair.hardness_score, pair.pair_id))
        return tuple(pairs[:limit] if limit is not None else pairs)

    def generate_assembly_motion_samples(
        self,
        *,
        sequence_name: str | None = None,
        limit: int | None = None,
        approach_fraction: float = 0.15,
    ) -> tuple[CadAssemblyMotionSample, ...]:
        if approach_fraction <= 0.0:
            raise ValueError("approach_fraction must be positive")
        sequences = (self.load_sequence(sequence_name),) if sequence_name is not None else self.list_sequences()
        samples: list[CadAssemblyMotionSample] = []
        for sequence in sequences:
            for pair in self.generate_mesh_pairs(sequence):
                center_a = _center(pair.asset_a)
                center_b = _center(pair.asset_b)
                direction = _unit_or_x_axis(_sub(center_b, center_a))
                step = approach_fraction * max(pair.asset_a.stats.diagonal, pair.asset_b.stats.diagonal, 1.0)
                sample = CadAssemblyMotionSample(
                    schema_version=CAD_ADAPTER_SCHEMA_VERSION,
                    source_name=FUSION360_SOURCE_NAME,
                    sequence_name=sequence.sequence_name,
                    sample_id=f"{sequence.sequence_name}:{len(samples)}",
                    pair=pair,
                    center_a_t0=center_a,
                    center_a_t1=_add(center_a, _scale(direction, step)),
                    center_b_t0=center_b,
                    center_b_t1=center_b,
                    motion_type="linear_assembly_approach",
                    metadata={
                        "approach_fraction": approach_fraction,
                        "human_designed_sequence": True,
                    },
                )
                samples.append(sample)
                if limit is not None and len(samples) >= limit:
                    return tuple(samples)
        return tuple(samples)
