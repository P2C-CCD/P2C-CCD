from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from p2cccd.datasets.cad.mesh_io import is_supported_mesh_path, mesh_stats_from_file

from .baseline_registry import default_baseline_root
from .contracts import (
    CCD_ADAPTER_SCHEMA_VERSION,
    CCDQueryFamily,
    DatasetQueryBatch,
    DatasetScene,
    ExternalCCDQuery,
    SourceLicense,
    Vec3,
)


RIGID_IPC_SOURCE_NAME = "Rigid IPC scenes"


@dataclass(frozen=True, slots=True)
class RigidIPCBody:
    body_id: int
    mesh: str | None
    mesh_path: Path | None
    position: Vec3
    rotation: tuple[float, ...]
    linear_velocity: Vec3
    angular_velocity: tuple[float, ...]
    is_fixed: bool
    body_type: str
    radius: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RigidIPCFixtureInfo:
    scene_name: str
    fixture_path: Path
    dimension: str
    timestep: float
    max_time: float | None
    body_count: int
    mesh_body_count: int
    inline_body_count: int
    moving_body_count: int


@dataclass(frozen=True, slots=True)
class RigidIPCScene:
    source_name: str
    scene_name: str
    fixture_path: Path
    timestep: float
    max_time: float | None
    bodies: tuple[RigidIPCBody, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def body_count(self) -> int:
        return len(self.bodies)

    @property
    def moving_body_count(self) -> int:
        return sum(1 for body in self.bodies if not body.is_fixed or _norm(body.linear_velocity) > 0.0)


def _as_vec3(values: Any, *, default_z: float = 0.0) -> Vec3:
    if not isinstance(values, list):
        return (0.0, 0.0, default_z)
    numeric = [float(value) for value in values[:3]]
    while len(numeric) < 3:
        numeric.append(default_z if len(numeric) == 2 else 0.0)
    return (numeric[0], numeric[1], numeric[2])


def _as_float_tuple(values: Any) -> tuple[float, ...]:
    if not isinstance(values, list):
        return ()
    return tuple(float(value) for value in values)


def _norm(vec: Vec3) -> float:
    return math.sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2])


def _add(lhs: Vec3, rhs: Vec3) -> Vec3:
    return (lhs[0] + rhs[0], lhs[1] + rhs[1], lhs[2] + rhs[2])


def _scale(vec: Vec3, scalar: float) -> Vec3:
    return (vec[0] * scalar, vec[1] * scalar, vec[2] * scalar)


def _is_fixed(raw: dict[str, Any]) -> bool:
    body_type = str(raw.get("type", "")).lower()
    if body_type == "static":
        return True
    fixed = raw.get("is_dof_fixed", False)
    if isinstance(fixed, bool):
        return fixed
    if isinstance(fixed, list) and fixed:
        return all(bool(value) for value in fixed)
    return False


def _scale_factor(raw: dict[str, Any]) -> float:
    scale = raw.get("scale", 1.0)
    if isinstance(scale, (int, float)):
        return abs(float(scale))
    if isinstance(scale, list) and scale:
        return max(abs(float(value)) for value in scale)
    return 1.0


def _inline_vertices(raw: dict[str, Any]) -> list[Vec3]:
    vertices = raw.get("vertices", [])
    if not isinstance(vertices, list):
        return []
    result: list[Vec3] = []
    for vertex in vertices:
        if isinstance(vertex, list) and len(vertex) >= 2:
            result.append(_as_vec3(vertex))
    return result


def _radius_from_inline_vertices(vertices: Iterable[Vec3], scale: float) -> float:
    vertex_list = list(vertices)
    if not vertex_list:
        return 0.5 * scale
    mins = tuple(min(vertex[index] for vertex in vertex_list) for index in range(3))
    maxs = tuple(max(vertex[index] for vertex in vertex_list) for index in range(3))
    diagonal = math.sqrt(sum((maxs[index] - mins[index]) ** 2 for index in range(3)))
    return max(1.0e-6, 0.5 * diagonal * scale)


def _radius_from_mesh(mesh_path: Path | None, scale: float) -> float:
    if mesh_path is None or not mesh_path.exists() or not is_supported_mesh_path(mesh_path):
        return 0.5 * scale
    try:
        return max(1.0e-6, 0.5 * mesh_stats_from_file(mesh_path).diagonal * scale)
    except (OSError, ValueError):
        return 0.5 * scale


class RigidIPCSceneAdapter:
    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root) if root is not None else default_baseline_root() / "rigid-ipc"

    @property
    def fixtures_root(self) -> Path:
        return self.root / "fixtures"

    @property
    def meshes_root(self) -> Path:
        return self.root / "meshes"

    def require_available(self) -> None:
        self.license().require_available()
        if not self.root.exists():
            raise FileNotFoundError(f"Rigid-IPC source root not found: {self.root}")
        if not self.fixtures_root.exists():
            raise FileNotFoundError(f"Rigid-IPC fixtures directory missing: {self.fixtures_root}")
        if not self.meshes_root.exists():
            raise FileNotFoundError(f"Rigid-IPC meshes directory missing: {self.meshes_root}")

    def license(self) -> SourceLicense:
        return SourceLicense(
            name=RIGID_IPC_SOURCE_NAME,
            license_path=self.root / "LICENSE",
            url="https://ipc-sim.github.io/rigid-ipc/",
            terms="Rigid-body correctness scenes and meshes; preserve upstream Rigid-IPC license and mesh notices.",
        )

    def fixture_entry_points(self) -> dict[str, str]:
        self.require_available()
        return {
            "fixtures": str(self.fixtures_root),
            "meshes": str(self.meshes_root),
            "simulator_readme": str(self.root / "README.md"),
            "ipc_comparison_scripts": str(self.root / "comparisons" / "IPC"),
        }

    def _fixture_path(self, scene_name: str) -> Path:
        relative = Path(scene_name)
        if relative.suffix != ".json":
            relative = relative.with_suffix(".json")
        path = self.fixtures_root / relative
        if not path.exists():
            raise FileNotFoundError(f"Rigid-IPC fixture not found: {path}")
        return path

    def _scene_name(self, fixture_path: Path) -> str:
        return fixture_path.relative_to(self.fixtures_root).with_suffix("").as_posix()

    def _load_json(self, fixture_path: Path) -> dict[str, Any]:
        data = json.loads(fixture_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"Rigid-IPC fixture must be a JSON object: {fixture_path}")
        return data

    def _resolve_mesh_path(self, mesh_name: str | None) -> Path | None:
        if not mesh_name:
            return None
        direct = self.meshes_root / mesh_name
        if direct.exists():
            return direct
        matches = tuple(self.meshes_root.rglob(mesh_name))
        return matches[0] if matches else direct

    def _bodies_from_fixture(self, fixture_path: Path, data: dict[str, Any]) -> tuple[RigidIPCBody, ...]:
        problem = data.get("rigid_body_problem", {})
        if not isinstance(problem, dict):
            raise ValueError(f"Rigid-IPC fixture missing rigid_body_problem: {fixture_path}")
        raw_bodies = problem.get("rigid_bodies", [])
        if not isinstance(raw_bodies, list):
            raise ValueError(f"Rigid-IPC rigid_bodies must be a list: {fixture_path}")
        bodies: list[RigidIPCBody] = []
        for index, raw in enumerate(raw_bodies):
            if not isinstance(raw, dict):
                raise ValueError(f"Rigid-IPC body must be an object in {fixture_path}")
            mesh_name = str(raw["mesh"]) if "mesh" in raw else None
            mesh_path = self._resolve_mesh_path(mesh_name)
            scale = _scale_factor(raw)
            inline_vertices = _inline_vertices(raw)
            radius = (
                _radius_from_mesh(mesh_path, scale)
                if mesh_name is not None
                else _radius_from_inline_vertices(inline_vertices, scale)
            )
            bodies.append(
                RigidIPCBody(
                    body_id=index,
                    mesh=mesh_name,
                    mesh_path=mesh_path,
                    position=_as_vec3(raw.get("position", [0.0, 0.0, 0.0])),
                    rotation=_as_float_tuple(raw.get("rotation", [])),
                    linear_velocity=_as_vec3(raw.get("linear_velocity", [0.0, 0.0, 0.0])),
                    angular_velocity=_as_float_tuple(raw.get("angular_velocity", [])),
                    is_fixed=_is_fixed(raw),
                    body_type=str(raw.get("type", "dynamic")),
                    radius=radius,
                    metadata={
                        "has_inline_vertices": bool(inline_vertices),
                        "scale": raw.get("scale", 1.0),
                        "source_keys": sorted(str(key) for key in raw.keys()),
                    },
                )
            )
        return tuple(bodies)

    def list_fixture_infos(
        self,
        *,
        dimension: str | None = None,
        limit: int | None = None,
    ) -> tuple[RigidIPCFixtureInfo, ...]:
        self.require_available()
        infos: list[RigidIPCFixtureInfo] = []
        for fixture_path in sorted(self.fixtures_root.rglob("*.json")):
            scene_name = self._scene_name(fixture_path)
            scene_dimension = scene_name.split("/", 1)[0]
            if dimension is not None and scene_dimension != dimension:
                continue
            data = self._load_json(fixture_path)
            bodies = self._bodies_from_fixture(fixture_path, data)
            timestep = float(data.get("timestep", 1.0))
            max_time = float(data["max_time"]) if "max_time" in data else None
            infos.append(
                RigidIPCFixtureInfo(
                    scene_name=scene_name,
                    fixture_path=fixture_path,
                    dimension=scene_dimension,
                    timestep=timestep,
                    max_time=max_time,
                    body_count=len(bodies),
                    mesh_body_count=sum(1 for body in bodies if body.mesh is not None),
                    inline_body_count=sum(1 for body in bodies if body.metadata["has_inline_vertices"]),
                    moving_body_count=sum(1 for body in bodies if not body.is_fixed or _norm(body.linear_velocity) > 0.0),
                )
            )
            if limit is not None and len(infos) >= limit:
                break
        return tuple(infos)

    def list_scenes(
        self,
        *,
        dimension: str | None = None,
        limit: int | None = None,
    ) -> tuple[DatasetScene, ...]:
        scenes: list[DatasetScene] = []
        for info in self.list_fixture_infos(dimension=dimension, limit=limit):
            scenes.append(
                DatasetScene(
                    schema_version=CCD_ADAPTER_SCHEMA_VERSION,
                    source_name=RIGID_IPC_SOURCE_NAME,
                    scene_name=info.scene_name,
                    scene_path=info.fixture_path,
                    frames=(),
                    metadata={
                        "dimension": info.dimension,
                        "timestep": info.timestep,
                        "max_time": info.max_time,
                        "body_count": info.body_count,
                        "mesh_body_count": info.mesh_body_count,
                        "inline_body_count": info.inline_body_count,
                        "moving_body_count": info.moving_body_count,
                    },
                )
            )
        return tuple(scenes)

    def load_scene(self, scene_name: str) -> RigidIPCScene:
        self.require_available()
        fixture_path = self._fixture_path(scene_name)
        data = self._load_json(fixture_path)
        timestep = float(data.get("timestep", 1.0))
        return RigidIPCScene(
            source_name=RIGID_IPC_SOURCE_NAME,
            scene_name=self._scene_name(fixture_path),
            fixture_path=fixture_path,
            timestep=timestep,
            max_time=float(data["max_time"]) if "max_time" in data else None,
            bodies=self._bodies_from_fixture(fixture_path, data),
            metadata={
                "scene_type": data.get("scene_type", ""),
                "solver": data.get("solver", ""),
            },
        )

    def _vertices_for_body_pair(
        self,
        body_a: RigidIPCBody,
        body_b: RigidIPCBody,
        timestep: float,
        family: CCDQueryFamily,
    ) -> tuple[tuple[Vec3, Vec3, Vec3, Vec3], tuple[Vec3, Vec3, Vec3, Vec3]]:
        center_a_t0 = body_a.position
        center_b_t0 = body_b.position
        center_a_t1 = _add(center_a_t0, _scale(body_a.linear_velocity, timestep))
        center_b_t1 = _add(center_b_t0, _scale(body_b.linear_velocity, timestep))
        radius_a = max(body_a.radius, 1.0e-6)
        radius_b = max(body_b.radius, 1.0e-6)
        if family is CCDQueryFamily.VERTEX_FACE:
            t0 = (
                center_a_t0,
                _add(center_b_t0, (radius_b, 0.0, 0.0)),
                _add(center_b_t0, (0.0, radius_b, 0.0)),
                _add(center_b_t0, (0.0, 0.0, radius_b)),
            )
            t1 = (
                center_a_t1,
                _add(center_b_t1, (radius_b, 0.0, 0.0)),
                _add(center_b_t1, (0.0, radius_b, 0.0)),
                _add(center_b_t1, (0.0, 0.0, radius_b)),
            )
            return t0, t1
        t0 = (
            _add(center_a_t0, (-radius_a, 0.0, 0.0)),
            _add(center_a_t0, (radius_a, 0.0, 0.0)),
            _add(center_b_t0, (0.0, -radius_b, 0.0)),
            _add(center_b_t0, (0.0, radius_b, 0.0)),
        )
        t1 = (
            _add(center_a_t1, (-radius_a, 0.0, 0.0)),
            _add(center_a_t1, (radius_a, 0.0, 0.0)),
            _add(center_b_t1, (0.0, -radius_b, 0.0)),
            _add(center_b_t1, (0.0, radius_b, 0.0)),
        )
        return t0, t1

    def iter_body_pairs(
        self,
        scene: RigidIPCScene,
        *,
        include_fixed_fixed_pairs: bool = False,
    ) -> tuple[tuple[RigidIPCBody, RigidIPCBody], ...]:
        pairs: list[tuple[RigidIPCBody, RigidIPCBody]] = []
        for body_a_index, body_a in enumerate(scene.bodies):
            for body_b in scene.bodies[body_a_index + 1 :]:
                if not include_fixed_fixed_pairs and body_a.is_fixed and body_b.is_fixed:
                    continue
                pairs.append((body_a, body_b))
        return tuple(pairs)

    def make_body_pair_query(
        self,
        scene: RigidIPCScene,
        body_a: RigidIPCBody,
        body_b: RigidIPCBody,
        *,
        family: CCDQueryFamily | str = CCDQueryFamily.VERTEX_FACE,
        query_id: int,
        source_query_index: int,
        batch_id: str,
    ) -> ExternalCCDQuery:
        query_family = family if isinstance(family, CCDQueryFamily) else CCDQueryFamily(str(family))
        vertices_t0, vertices_t1 = self._vertices_for_body_pair(
            body_a,
            body_b,
            scene.timestep,
            query_family,
        )
        return ExternalCCDQuery(
            schema_version=CCD_ADAPTER_SCHEMA_VERSION,
            source_name=RIGID_IPC_SOURCE_NAME,
            scene_name=scene.scene_name,
            batch_id=batch_id,
            query_id=int(query_id),
            source_query_index=int(source_query_index),
            family=query_family,
            vertices_t0=vertices_t0,
            vertices_t1=vertices_t1,
            ground_truth_collides=None,
            box_pair=(body_a.body_id, body_b.body_id),
            metadata={
                "adapter_query_type": "rigid_ipc_body_pair_proxy",
                "body_a_mesh": body_a.mesh,
                "body_b_mesh": body_b.mesh,
                "body_a_fixed": body_a.is_fixed,
                "body_b_fixed": body_b.is_fixed,
                "timestep": scene.timestep,
                "witness_family": query_family.p2cccd_witness_family,
            },
        )

    def load_body_pair_query_batch(
        self,
        scene_name: str,
        *,
        family: CCDQueryFamily | str = CCDQueryFamily.VERTEX_FACE,
        limit: int | None = None,
        include_fixed_fixed_pairs: bool = False,
        body_pairs: Sequence[tuple[int, int]] | None = None,
    ) -> DatasetQueryBatch:
        scene = self.load_scene(scene_name)
        query_family = family if isinstance(family, CCDQueryFamily) else CCDQueryFamily(str(family))
        batch_id = f"{scene.scene_name}:{query_family.value}:body_pairs"
        queries: list[ExternalCCDQuery] = []
        if body_pairs is None:
            selected_pairs = self.iter_body_pairs(
                scene,
                include_fixed_fixed_pairs=include_fixed_fixed_pairs,
            )
        else:
            bodies_by_id = {body.body_id: body for body in scene.bodies}
            selected_pairs = []
            for body_a_id, body_b_id in body_pairs:
                if body_a_id == body_b_id:
                    continue
                ordered_pair = (min(int(body_a_id), int(body_b_id)), max(int(body_a_id), int(body_b_id)))
                body_a = bodies_by_id.get(ordered_pair[0])
                body_b = bodies_by_id.get(ordered_pair[1])
                if body_a is None or body_b is None:
                    raise ValueError("Rigid-IPC body_pairs references an unknown body id")
                if not include_fixed_fixed_pairs and body_a.is_fixed and body_b.is_fixed:
                    continue
                selected_pairs.append((body_a, body_b))
        for source_index, (body_a, body_b) in enumerate(selected_pairs):
            queries.append(
                self.make_body_pair_query(
                    scene,
                    body_a,
                    body_b,
                    family=query_family,
                    query_id=source_index,
                    source_query_index=source_index,
                    batch_id=batch_id,
                )
            )
            if limit is not None and len(queries) >= limit:
                break
        return DatasetQueryBatch(
            schema_version=CCD_ADAPTER_SCHEMA_VERSION,
            source_name=RIGID_IPC_SOURCE_NAME,
            scene_name=scene.scene_name,
            batch_id=batch_id,
            family=query_family,
            queries=tuple(queries),
            metadata={
                "fixture_path": str(scene.fixture_path),
                "query_generation": "body_pair_proxy_from_rigid_ipc_fixture",
                "ground_truth_labels": "unknown",
                "body_count": scene.body_count,
                "moving_body_count": scene.moving_body_count,
                "limit": limit,
            },
        )
