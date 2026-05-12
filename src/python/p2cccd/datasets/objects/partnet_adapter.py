from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from p2cccd.datasets.ccd.contracts import SourceLicense

from .common import approach_motion_sample, load_json_if_exists, mesh_paths_under, object_asset_from_mesh
from .contracts import (
    OBJECT_ADAPTER_SCHEMA_VERSION,
    ArticulatedObjectScene,
    ObjectMeshAsset,
    ObjectMotionSample,
)


PARTNET_SOURCE_NAME = "PartNet"
PARTNET_MOBILITY_SOURCE_NAME = "PartNet-Mobility"


def default_partnet_root() -> Path:
    return Path(__file__).resolve().parents[5] / "datasets" / "partnet"


def default_partnet_mobility_root() -> Path:
    return Path(__file__).resolve().parents[5] / "datasets" / "partnet_mobility"


@dataclass(frozen=True, slots=True)
class PartNetAdapter:
    root: Path | None = None

    @property
    def source_root(self) -> Path:
        return self.root if self.root is not None else default_partnet_root()

    @property
    def source_name(self) -> str:
        return PARTNET_SOURCE_NAME

    def require_available(self) -> None:
        self.license().require_available()
        if not self.source_root.exists():
            raise FileNotFoundError(f"{self.source_name} root not found: {self.source_root}")

    def license(self) -> SourceLicense:
        return SourceLicense(
            name=self.source_name,
            license_path=self.source_root / "LICENSE",
            url="https://partnet.cs.stanford.edu/",
            terms="Part-aware mesh scenes; preserve PartNet license and semantic annotations.",
        )

    def list_assets(self, *, limit: int | None = None) -> tuple[ObjectMeshAsset, ...]:
        self.require_available()
        assets: list[ObjectMeshAsset] = []
        for path in mesh_paths_under(self.source_root, limit=limit):
            metadata = load_json_if_exists(path.with_suffix(".json"))
            part_count = int(metadata.get("part_count", 1)) if isinstance(metadata.get("part_count", 1), int) else 1
            assets.append(
                object_asset_from_mesh(
                    source_name=self.source_name,
                    root=self.source_root,
                    mesh_path=path,
                    object_name=path.parent.name,
                    category=str(metadata.get("category", path.parent.parent.name if path.parent.parent != self.source_root else "unknown")),
                    part_count=part_count,
                    metadata={
                        "part_aware": True,
                        **metadata,
                    },
                )
            )
        return tuple(assets)


@dataclass(frozen=True, slots=True)
class PartNetMobilityAdapter(PartNetAdapter):
    @property
    def source_root(self) -> Path:
        return self.root if self.root is not None else default_partnet_mobility_root()

    @property
    def source_name(self) -> str:
        return PARTNET_MOBILITY_SOURCE_NAME

    def license(self) -> SourceLicense:
        return SourceLicense(
            name=self.source_name,
            license_path=self.source_root / "LICENSE",
            url="https://sapien.ucsd.edu/browse",
            terms="Articulated object scenes; preserve PartNet-Mobility license, URDF, and joint metadata.",
        )

    def list_scenes(self, *, limit: int | None = None) -> tuple[ArticulatedObjectScene, ...]:
        self.require_available()
        scenes: list[ArticulatedObjectScene] = []
        candidate_dirs = [path for path in sorted(self.source_root.iterdir()) if path.is_dir()]
        if not candidate_dirs:
            candidate_dirs = [self.source_root]
        for scene_path in candidate_dirs:
            mesh_paths = mesh_paths_under(scene_path)
            if not mesh_paths:
                continue
            metadata = load_json_if_exists(scene_path / "mobility.json")
            if not metadata:
                metadata = load_json_if_exists(scene_path / "meta.json")
            joints = metadata.get("joints", [])
            joint_count = len(joints) if isinstance(joints, list) else int(metadata.get("joint_count", 0) or 0)
            assets = tuple(
                object_asset_from_mesh(
                    source_name=self.source_name,
                    root=self.source_root,
                    mesh_path=path,
                    object_name=path.parent.name,
                    category=str(metadata.get("category", scene_path.name)),
                    part_count=max(1, joint_count + 1),
                    metadata={
                        "articulated": True,
                        "scene_id": scene_path.relative_to(self.source_root).as_posix() if scene_path != self.source_root else "root",
                    },
                )
                for path in mesh_paths
            )
            scenes.append(
                ArticulatedObjectScene(
                    schema_version=OBJECT_ADAPTER_SCHEMA_VERSION,
                    source_name=self.source_name,
                    scene_id=scene_path.relative_to(self.source_root).as_posix() if scene_path != self.source_root else "root",
                    scene_path=scene_path,
                    assets=assets,
                    joint_count=joint_count,
                    metadata={
                        "has_mobility_metadata": bool(metadata),
                        **metadata,
                    },
                )
            )
            if limit is not None and len(scenes) >= limit:
                break
        return tuple(scenes)

    def generate_articulated_motion_samples(
        self,
        *,
        scene_id: str | None = None,
        limit: int | None = None,
    ) -> tuple[ObjectMotionSample, ...]:
        scenes = self.list_scenes()
        if scene_id is not None:
            scenes = tuple(scene for scene in scenes if scene.scene_id == scene_id)
            if not scenes:
                raise FileNotFoundError(f"PartNet-Mobility scene not found: {scene_id}")
        samples: list[ObjectMotionSample] = []
        for scene in scenes:
            assets = scene.assets
            for index, asset_a in enumerate(assets):
                for asset_b in assets[index + 1 :]:
                    samples.append(
                        approach_motion_sample(
                            source_name=self.source_name,
                            sample_id=f"{scene.scene_id}:{len(samples)}",
                            asset_a=asset_a,
                            asset_b=asset_b,
                            motion_type="articulated_part_motion",
                            approach_fraction=0.10,
                            metadata={
                                "scene_id": scene.scene_id,
                                "joint_count": scene.joint_count,
                                "part_aware": True,
                            },
                        )
                    )
                    if limit is not None and len(samples) >= limit:
                        return tuple(samples)
        return tuple(samples)
