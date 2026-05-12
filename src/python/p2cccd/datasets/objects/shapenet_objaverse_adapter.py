from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from p2cccd.datasets.ccd.contracts import SourceLicense

from .common import approach_motion_sample, load_json_if_exists, mesh_paths_under, object_asset_from_mesh
from .contracts import ObjectMeshAsset, ObjectMotionSample


SHAPENET_SOURCE_NAME = "ShapeNet"
OBJAVERSE_XL_SOURCE_NAME = "Objaverse-XL"


def default_shapenet_root() -> Path:
    return Path(__file__).resolve().parents[5] / "datasets" / "shapenet"


def default_objaverse_xl_root() -> Path:
    return Path(__file__).resolve().parents[5] / "datasets" / "objaverse_xl"


@dataclass(frozen=True, slots=True)
class _LargeScaleObjectAdapterBase(ABC):
    root: Path | None = None

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Dataset display name used in manifests and benchmark rows."""

    @property
    @abstractmethod
    def source_root(self) -> Path:
        """Resolved dataset root directory."""

    def require_available(self) -> None:
        self.license().require_available()
        if not self.source_root.exists():
            raise FileNotFoundError(f"{self.source_name} root not found: {self.source_root}")

    @abstractmethod
    def license(self) -> SourceLicense:
        """License metadata gate for this dataset."""

    def list_mesh_paths(self, *, limit: int | None = None) -> tuple[Path, ...]:
        self.require_available()
        return mesh_paths_under(self.source_root, limit=limit)

    def load_asset(self, path: str | Path) -> ObjectMeshAsset:
        self.require_available()
        mesh_path = Path(path)
        if not mesh_path.is_absolute():
            mesh_path = self.source_root / mesh_path
        if not mesh_path.exists():
            raise FileNotFoundError(f"{self.source_name} mesh not found: {mesh_path}")
        metadata = load_json_if_exists(mesh_path.with_suffix(".json"))
        category = str(
            metadata.get(
                "category",
                mesh_path.parent.parent.name if mesh_path.parent.parent != self.source_root else mesh_path.parent.name,
            )
        )
        return object_asset_from_mesh(
            source_name=self.source_name,
            root=self.source_root,
            mesh_path=mesh_path,
            object_name=str(metadata.get("name", mesh_path.parent.name or mesh_path.stem)),
            category=category,
            dirty_score=float(metadata.get("dirty_score", 0.0) or 0.0),
            metadata={
                "large_scale_object_source": True,
                **metadata,
            },
        )

    def list_assets(self, *, limit: int | None = None) -> tuple[ObjectMeshAsset, ...]:
        return tuple(self.load_asset(path) for path in self.list_mesh_paths(limit=limit))

    def generate_ood_subset_samples(
        self,
        *,
        assets: Sequence[ObjectMeshAsset] | None = None,
        limit: int | None = None,
        approach_fraction: float = 0.18,
    ) -> tuple[ObjectMotionSample, ...]:
        if approach_fraction <= 0.0:
            raise ValueError("approach_fraction must be positive")
        asset_list = list(assets) if assets is not None else list(self.list_assets())
        samples: list[ObjectMotionSample] = []
        for index, asset_a in enumerate(asset_list):
            for asset_b in asset_list[index + 1 :]:
                samples.append(
                    approach_motion_sample(
                        source_name=self.source_name,
                        sample_id=f"{self.source_name}:ood_subset:{len(samples)}",
                        asset_a=asset_a,
                        asset_b=asset_b,
                        motion_type="large_scale_ood_object_approach",
                        approach_fraction=approach_fraction,
                        metadata={
                            "ood_subset": True,
                            "adapter_query_type": "large_scale_object_pair",
                            "allow_slowdown_not_false_negative": True,
                        },
                    )
                )
                if limit is not None and len(samples) >= limit:
                    return tuple(samples)
        return tuple(samples)


@dataclass(frozen=True, slots=True)
class ShapeNetAdapter(_LargeScaleObjectAdapterBase):
    @property
    def source_name(self) -> str:
        return SHAPENET_SOURCE_NAME

    @property
    def source_root(self) -> Path:
        return self.root if self.root is not None else default_shapenet_root()

    def license(self) -> SourceLicense:
        return SourceLicense(
            name=SHAPENET_SOURCE_NAME,
            license_path=self.source_root / "LICENSE",
            url="https://shapenet.org/",
            terms="Large-scale object OOD subsets; preserve ShapeNet terms, categories, and model identifiers.",
        )


@dataclass(frozen=True, slots=True)
class ObjaverseXLAdapter(_LargeScaleObjectAdapterBase):
    @property
    def source_name(self) -> str:
        return OBJAVERSE_XL_SOURCE_NAME

    @property
    def source_root(self) -> Path:
        return self.root if self.root is not None else default_objaverse_xl_root()

    def license(self) -> SourceLicense:
        return SourceLicense(
            name=OBJAVERSE_XL_SOURCE_NAME,
            license_path=self.source_root / "LICENSE",
            url="https://objaverse.allenai.org/",
            terms="Large-scale object OOD subsets; preserve Objaverse-XL asset licenses and metadata.",
        )
