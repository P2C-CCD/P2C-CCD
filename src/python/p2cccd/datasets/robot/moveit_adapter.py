from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from p2cccd.datasets.cad.mesh_io import is_supported_mesh_path
from p2cccd.datasets.ccd.contracts import SourceLicense

from .contracts import ROBOT_ADAPTER_SCHEMA_VERSION, PlanningSceneAsset, RobotMotionQuery


MOVEIT_RESOURCES_SOURCE_NAME = "MoveIt resources"


def default_moveit_resources_root() -> Path:
    return Path(__file__).resolve().parents[5] / "datasets" / "moveit_resources"


def _load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object in {path}")
    return data


def _scene_files(root: Path) -> tuple[Path, ...]:
    suffixes = {".json", ".scene", ".yaml", ".yml", ".srdf", ".urdf"}
    return tuple(sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in suffixes))


def _mesh_files_near(scene_path: Path, root: Path) -> tuple[Path, ...]:
    scene_dir = scene_path.parent
    candidates = set()
    for base in (scene_dir, root):
        for path in base.rglob("*"):
            if path.is_file() and is_supported_mesh_path(path):
                candidates.add(path)
    return tuple(sorted(candidates))


@dataclass(frozen=True, slots=True)
class MoveItResourcesAdapter:
    root: Path | None = None

    @property
    def source_root(self) -> Path:
        return self.root if self.root is not None else default_moveit_resources_root()

    def require_available(self) -> None:
        self.license().require_available()
        if not self.source_root.exists():
            raise FileNotFoundError(f"MoveIt resources root not found: {self.source_root}")

    def license(self) -> SourceLicense:
        return SourceLicense(
            name=MOVEIT_RESOURCES_SOURCE_NAME,
            license_path=self.source_root / "LICENSE",
            url="https://github.com/moveit/moveit_resources",
            terms="Robot planning scenes and benchmark resources; preserve MoveIt package and mesh licenses.",
        )

    def list_scene_assets(self, *, limit: int | None = None) -> tuple[PlanningSceneAsset, ...]:
        self.require_available()
        scenes: list[PlanningSceneAsset] = []
        for scene_path in _scene_files(self.source_root):
            metadata = _load_json_if_exists(scene_path) if scene_path.suffix.lower() == ".json" else {}
            robot_name = str(metadata.get("robot_name", scene_path.parent.name or "robot"))
            scene_id = scene_path.relative_to(self.source_root).with_suffix("").as_posix()
            scenes.append(
                PlanningSceneAsset(
                    schema_version=ROBOT_ADAPTER_SCHEMA_VERSION,
                    source_name=MOVEIT_RESOURCES_SOURCE_NAME,
                    scene_id=scene_id,
                    scene_path=scene_path,
                    robot_name=robot_name,
                    mesh_paths=_mesh_files_near(scene_path, self.source_root),
                    metadata={
                        "scene_format": scene_path.suffix.lower().lstrip("."),
                        "mesh_count": len(_mesh_files_near(scene_path, self.source_root)),
                        **metadata,
                    },
                )
            )
            if limit is not None and len(scenes) >= limit:
                break
        return tuple(scenes)

    def generate_motion_queries(
        self,
        *,
        scene_id: str | None = None,
        limit: int | None = None,
    ) -> tuple[RobotMotionQuery, ...]:
        scenes = self.list_scene_assets()
        if scene_id is not None:
            scenes = tuple(scene for scene in scenes if scene.scene_id == scene_id)
            if not scenes:
                raise FileNotFoundError(f"MoveIt planning scene not found: {scene_id}")
        queries: list[RobotMotionQuery] = []
        for scene in scenes:
            link_names = scene.metadata.get("links", ["link_0", "link_1"])
            if not isinstance(link_names, list) or len(link_names) < 2:
                link_names = ["link_0", "link_1"]
            for index in range(len(link_names) - 1):
                link_a = str(link_names[index])
                link_b = str(link_names[index + 1])
                queries.append(
                    RobotMotionQuery(
                        schema_version=ROBOT_ADAPTER_SCHEMA_VERSION,
                        source_name=MOVEIT_RESOURCES_SOURCE_NAME,
                        query_id=f"{scene.scene_id}:{len(queries)}",
                        scene=scene,
                        link_a=link_a,
                        link_b=link_b,
                        center_a_t0=(0.0, 0.0, 0.0),
                        center_a_t1=(0.05 * (index + 1), 0.0, 0.0),
                        center_b_t0=(0.2 * (index + 1), 0.0, 0.0),
                        center_b_t1=(0.2 * (index + 1), 0.0, 0.0),
                        metadata={
                            "robot_name": scene.robot_name,
                            "planning_scene_query": True,
                            "benchmark_resource": True,
                        },
                    )
                )
                if limit is not None and len(queries) >= limit:
                    return tuple(queries)
        return tuple(queries)
