from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from p2cccd.datasets.cad.contracts import Vec3


ROBOT_ADAPTER_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class PlanningSceneAsset:
    schema_version: int
    source_name: str
    scene_id: str
    scene_path: Path
    robot_name: str
    mesh_paths: tuple[Path, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema_version != ROBOT_ADAPTER_SCHEMA_VERSION:
            raise ValueError("unsupported PlanningSceneAsset schema_version")
        if not self.source_name:
            raise ValueError("PlanningSceneAsset.source_name is required")
        if not self.scene_id:
            raise ValueError("PlanningSceneAsset.scene_id is required")
        if not self.robot_name:
            raise ValueError("PlanningSceneAsset.robot_name is required")


@dataclass(frozen=True, slots=True)
class RobotMotionQuery:
    schema_version: int
    source_name: str
    query_id: str
    scene: PlanningSceneAsset
    link_a: str
    link_b: str
    center_a_t0: Vec3
    center_a_t1: Vec3
    center_b_t0: Vec3
    center_b_t1: Vec3
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema_version != ROBOT_ADAPTER_SCHEMA_VERSION:
            raise ValueError("unsupported RobotMotionQuery schema_version")
        if not self.source_name:
            raise ValueError("RobotMotionQuery.source_name is required")
        if not self.query_id:
            raise ValueError("RobotMotionQuery.query_id is required")
        if not self.link_a or not self.link_b:
            raise ValueError("RobotMotionQuery links are required")
        if self.link_a == self.link_b:
            raise ValueError("RobotMotionQuery requires two distinct links")
