from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from p2cccd.datasets.ccd.contracts import SourceLicense

from .common import approach_motion_sample, mesh_paths_under, object_asset_from_mesh
from .contracts import ObjectMeshAsset, ObjectMotionSample


YCB_SOURCE_NAME = "YCB Object and Model Set"
GOOGLE_SCANNED_OBJECTS_SOURCE_NAME = "Google Scanned Objects"


def default_ycb_root() -> Path:
    return Path(__file__).resolve().parents[5] / "datasets" / "ycb"


def default_google_scanned_objects_root() -> Path:
    return Path(__file__).resolve().parents[5] / "datasets" / "google_scanned_objects"


@dataclass(frozen=True, slots=True)
class _ObjectDatasetAdapterBase(ABC):
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
        category = mesh_path.parent.parent.name if mesh_path.parent.parent != self.source_root else mesh_path.parent.name
        return object_asset_from_mesh(
            source_name=self.source_name,
            root=self.source_root,
            mesh_path=mesh_path,
            object_name=mesh_path.parent.name,
            category=category,
            metadata={"robot_validation_role": "manipulation_object"},
        )

    def list_assets(self, *, limit: int | None = None) -> tuple[ObjectMeshAsset, ...]:
        return tuple(self.load_asset(path) for path in self.list_mesh_paths(limit=limit))

    def generate_robot_validation_samples(
        self,
        *,
        assets: Sequence[ObjectMeshAsset] | None = None,
        limit: int | None = None,
        approach_fraction: float = 0.20,
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
                        sample_id=f"{self.source_name}:{len(samples)}",
                        asset_a=asset_a,
                        asset_b=asset_b,
                        motion_type="robot_gripper_object_approach",
                        approach_fraction=approach_fraction,
                        metadata={
                            "adapter_query_type": "manipulation_object_pair",
                            "allow_slowdown_not_false_negative": True,
                        },
                    )
                )
                if limit is not None and len(samples) >= limit:
                    return tuple(samples)
        return tuple(samples)


@dataclass(frozen=True, slots=True)
class YCBObjectSetAdapter(_ObjectDatasetAdapterBase):
    @property
    def source_name(self) -> str:
        return YCB_SOURCE_NAME

    @property
    def source_root(self) -> Path:
        return self.root if self.root is not None else default_ycb_root()

    def license(self) -> SourceLicense:
        return SourceLicense(
            name=YCB_SOURCE_NAME,
            license_path=self.source_root / "LICENSE",
            url="https://www.ycbbenchmarks.com/",
            terms="Manipulation object meshes for robot validation; preserve YCB dataset license and attribution.",
        )


@dataclass(frozen=True, slots=True)
class GoogleScannedObjectsAdapter(_ObjectDatasetAdapterBase):
    @property
    def source_name(self) -> str:
        return GOOGLE_SCANNED_OBJECTS_SOURCE_NAME

    @property
    def source_root(self) -> Path:
        return self.root if self.root is not None else default_google_scanned_objects_root()

    def license(self) -> SourceLicense:
        return SourceLicense(
            name=GOOGLE_SCANNED_OBJECTS_SOURCE_NAME,
            license_path=self.source_root / "LICENSE",
            url="https://research.google/blog/scanned-objects-by-google-research-a-dataset-of-3d-scanned-common-household-items/",
            terms="Scanned object meshes for OOD robot validation; preserve upstream terms and object metadata.",
        )
