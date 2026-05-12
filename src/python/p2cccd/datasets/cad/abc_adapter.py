from __future__ import annotations

import json
import math
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Sequence

from p2cccd.datasets.ccd.contracts import SourceLicense

from .contracts import CAD_ADAPTER_SCHEMA_VERSION, CadMeshAsset, CadMeshPair
from .mesh_io import is_supported_mesh_path, mesh_stats_from_file, stable_asset_id


ABC_SOURCE_NAME = "ABC Dataset"


def _default_dataset_root(name: str) -> Path:
    file_path = Path(__file__).resolve()
    project_local = file_path.parents[4] / "datasets" / name
    legacy_root = file_path.parents[5] / "datasets" / name
    return project_local if project_local.exists() else legacy_root


def default_abc_root() -> Path:
    return _default_dataset_root("abc")


def _load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object in {path}")
    return data


def _patch_metadata_paths(mesh_path: Path) -> tuple[Path, ...]:
    return (
        mesh_path.with_suffix(".patch.json"),
        mesh_path.with_suffix(".patches.json"),
        mesh_path.parent / f"{mesh_path.stem}.patch.json",
        mesh_path.parent / f"{mesh_path.stem}.patches.json",
        mesh_path.parent / "metadata" / f"{mesh_path.stem}.json",
    )


def _load_patch_metadata(mesh_path: Path) -> dict[str, Any]:
    for path in _patch_metadata_paths(mesh_path):
        data = _load_json_if_exists(path)
        if data:
            return {**data, "metadata_path": str(path)}
    return {}


def _patch_count(asset: CadMeshAsset) -> int:
    patches = asset.patch_metadata.get("patches")
    if isinstance(patches, list):
        return len(patches)
    patch_count = asset.patch_metadata.get("patch_count")
    if isinstance(patch_count, int):
        return max(0, patch_count)
    return max(1, asset.stats.face_count)


def _hardness_score(asset_a: CadMeshAsset, asset_b: CadMeshAsset) -> float:
    diag_a = max(asset_a.stats.diagonal, 1.0e-12)
    diag_b = max(asset_b.stats.diagonal, 1.0e-12)
    scale_similarity = math.exp(-abs(math.log(diag_a / diag_b)))
    face_a = max(asset_a.stats.face_count, 1)
    face_b = max(asset_b.stats.face_count, 1)
    topology_similarity = math.exp(-abs(math.log(face_a / face_b)))
    patch_bonus = 0.15 if asset_a.has_patch_metadata and asset_b.has_patch_metadata else 0.0
    return min(1.0, 0.55 * scale_similarity + 0.30 * topology_similarity + patch_bonus)


@dataclass(frozen=True, slots=True)
class ABCDatasetAdapter:
    root: Path | None = None

    @property
    def source_root(self) -> Path:
        return self.root if self.root is not None else default_abc_root()

    def require_available(self) -> None:
        self.license().require_available()
        if not self.source_root.exists():
            raise FileNotFoundError(f"ABC Dataset root not found: {self.source_root}")

    def license(self) -> SourceLicense:
        return SourceLicense(
            name=ABC_SOURCE_NAME,
            license_path=self.source_root / "LICENSE",
            url="https://deep-geometry.github.io/abc-dataset/",
            terms="CAD-derived mesh ingestion; preserve upstream ABC terms for redistributed shards.",
        )

    def list_mesh_paths(self, *, limit: int | None = None) -> tuple[Path, ...]:
        self.require_available()
        paths: list[Path] = []
        for path in sorted(self.source_root.rglob("*")):
            if path.is_file() and is_supported_mesh_path(path):
                paths.append(path)
                if limit is not None and len(paths) >= limit:
                    break
        return tuple(paths)

    def load_asset(self, path: str | Path) -> CadMeshAsset:
        self.require_available()
        mesh_path = Path(path)
        if not mesh_path.is_absolute():
            if not mesh_path.exists():
                mesh_path = self.source_root / mesh_path
        if not mesh_path.exists():
            raise FileNotFoundError(f"ABC mesh asset not found: {mesh_path}")
        if not is_supported_mesh_path(mesh_path):
            raise ValueError(f"unsupported ABC mesh asset format: {mesh_path.suffix}")

        metadata = _load_json_if_exists(mesh_path.with_suffix(".json"))
        patch_metadata = _load_patch_metadata(mesh_path)
        return CadMeshAsset(
            schema_version=CAD_ADAPTER_SCHEMA_VERSION,
            source_name=ABC_SOURCE_NAME,
            asset_id=stable_asset_id(mesh_path, self.source_root),
            asset_path=mesh_path,
            mesh_format=mesh_path.suffix.lower().lstrip("."),
            stats=mesh_stats_from_file(mesh_path),
            patch_metadata=patch_metadata,
            metadata={
                "source_relative_path": mesh_path.relative_to(self.source_root).as_posix(),
                **metadata,
            },
        )

    def list_assets(self, *, limit: int | None = None) -> tuple[CadMeshAsset, ...]:
        return tuple(self.load_asset(path) for path in self.list_mesh_paths(limit=limit))

    def generate_mesh_pairs(
        self,
        *,
        assets: Sequence[CadMeshAsset] | None = None,
        limit: int | None = None,
    ) -> tuple[CadMeshPair, ...]:
        asset_list = list(assets) if assets is not None else list(self.list_assets())
        pairs: list[CadMeshPair] = []
        for asset_a, asset_b in combinations(asset_list, 2):
            patch_pair_count = _patch_count(asset_a) * _patch_count(asset_b)
            score = _hardness_score(asset_a, asset_b)
            pair_id = f"{asset_a.asset_id}__{asset_b.asset_id}"
            pairs.append(
                CadMeshPair(
                    schema_version=CAD_ADAPTER_SCHEMA_VERSION,
                    source_name=ABC_SOURCE_NAME,
                    pair_id=pair_id,
                    asset_a=asset_a,
                    asset_b=asset_b,
                    hardness_score=score,
                    patch_pair_count=patch_pair_count,
                    metadata={
                        "sampling_role": "industrial_hard_negative",
                        "scale_diagonal_a": asset_a.stats.diagonal,
                        "scale_diagonal_b": asset_b.stats.diagonal,
                    },
                )
            )
        pairs.sort(key=lambda pair: (-pair.hardness_score, pair.pair_id))
        return tuple(pairs[:limit] if limit is not None else pairs)

    def industrial_hard_negative_pairs(self, *, limit: int) -> tuple[CadMeshPair, ...]:
        if limit <= 0:
            raise ValueError("limit must be positive")
        return self.generate_mesh_pairs(limit=limit)
