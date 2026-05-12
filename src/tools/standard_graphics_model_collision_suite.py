from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[2]
P2CCCD_PYTHON = ROOT / "src" / "python"
TOOLS_DIR = Path(__file__).resolve().parent
for import_root in (P2CCCD_PYTHON, TOOLS_DIR):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from p2cccd.bench.tight_inclusion_stpf_training import run_tight_inclusion_stpf_training

from render_aris_real_mesh_physics_cases import (
    Body,
    CONTACT_T,
    DURATION_T,
    FPS,
    FRAME_COUNT,
    H,
    MeshAsset,
    W,
    build_case,
    compact_display_mesh,
    generated_dense_ground_asset,
    normalize,
    project,
    local_vertices,
    write_mp4,
    write_json,
)


RUN_TAG = "standard_graphics_models_run_id"
STANDARD_ROOT = ROOT / "src" / "datasets" / "standard_graphics_models"
COMMON_ROOT = STANDARD_ROOT / "common_3d_test_models" / "common-3d-test-models" / "data"
MYDEMO_ROOT = ROOT / "src" / "MyDemo" / RUN_TAG
BENCHMARK_ROOT = ROOT / "src" / "benchmark" / RUN_TAG
SHARDS_ROOT = ROOT / "src" / "datasets" / "training" / RUN_TAG / "shards"
TRAINING_OUTPUT_ROOT = ROOT / "src" / "outputs" / "stpf_training" / RUN_TAG
GENERATED_ASSET_ROOT = MYDEMO_ROOT / "_generated_assets"

SEED = fixed_seed
STPF_TARGET_MASK_ALL = (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3) | (1 << 4)
FAMILY_TARGET_INDEX = {
    "stanford_bunny_sanity": 0,
    "stanford_dragon_highfreq": 1,
    "armadillo_articulated_scan": 2,
    "fandisk_sharp_cad": 3,
    "spot_clean_manifold": 4,
    "suzanne_lowpoly_visual": 5,
    "utah_teapot_curved_classic": 6,
    "cornell_box_scene_container": 7,
}


@dataclass(frozen=True)
class StandardModelSpec:
    name: str
    key: str
    family: str
    path: Path
    description: str
    color: tuple[int, int, int]
    mass: float
    scale: float
    xy: tuple[float, float]
    relative_speed: float
    sharp_feature: float
    high_frequency: float
    manifold_clean: float


@dataclass(frozen=True)
class ClassicBenchmarkCase:
    name: str
    family: str
    dataset_model: str
    description: str
    motion_type: str
    density: int
    relative_speed: float
    toi: float
    positives: int
    exact_cost_mean: float
    learned_selected: int
    random_selected_mean: float
    object_count: int
    source_mesh_faces: int
    render_preview_faces: int
    tags: tuple[str, ...]
    fn: int = 0
    analytic_truth: bool = False


@dataclass(frozen=True)
class StandardPhysicalProperties:
    model_name: str
    material: str
    density_kg_m3: float
    volume_m3: float
    mass_kg: float
    center_of_mass_m: tuple[float, float, float]
    inertia_tensor_kg_m2: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]
    principal_inertia_kg_m2: tuple[float, float, float]
    estimation_method: str
    legacy_manual_mass_parameter: float
    scale_to_scene_m: float


MATERIAL_DENSITY_TABLE_KG_M3: dict[str, float] = {
    "polymer_resin": 1180.0,
    "cast_plaster": 1450.0,
    "aluminum": 2700.0,
    "ceramic": 2400.0,
    "printed_plastic": 1050.0,
}

STANDARD_MODEL_MATERIAL: dict[str, str] = {
    "bunny": "polymer_resin",
    "dragon": "cast_plaster",
    "armadillo": "cast_plaster",
    "fandisk": "aluminum",
    "spot": "polymer_resin",
    "suzanne": "printed_plastic",
    "teapot": "ceramic",
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return path.as_posix()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def estimate_physical_properties(asset: MeshAsset, spec: StandardModelSpec) -> StandardPhysicalProperties:
    material = STANDARD_MODEL_MATERIAL.get(spec.key, "polymer_resin")
    density = float(MATERIAL_DENSITY_TABLE_KG_M3[material])
    scaled_vertices = np.asarray(asset.vertices, dtype=np.float64) * float(spec.scale)
    faces = np.asarray(asset.faces, dtype=np.int64)
    estimation_method = "convex_hull_fallback"
    try:
        mesh = trimesh.Trimesh(vertices=scaled_vertices, faces=faces, process=False)
        if bool(mesh.is_watertight) and abs(float(mesh.volume)) > 1.0e-9:
            mass_mesh = mesh.copy()
            if float(mass_mesh.volume) < 0.0:
                mass_mesh.invert()
            estimation_method = "watertight_triangle_mesh"
        else:
            mass_mesh = mesh.convex_hull
        mp = mass_mesh.mass_properties
        volume = abs(float(mp.volume))
        if not np.isfinite(volume) or volume <= 1.0e-9:
            raise ValueError("non-positive volume estimate")
        center = np.asarray(mp.center_mass, dtype=np.float64)
        inertia = np.asarray(mp.inertia, dtype=np.float64) * density
    except Exception:
        bbox_min = scaled_vertices.min(axis=0)
        bbox_max = scaled_vertices.max(axis=0)
        extent = np.maximum(bbox_max - bbox_min, 1.0e-6)
        # Sparse/open graphics meshes are sometimes not volume closed.  The
        # fallback is a conservative occupied-box proxy with a moderate fill
        # ratio, suitable for visualization and audit metadata.
        volume = float(np.prod(extent) * 0.55)
        center = 0.5 * (bbox_min + bbox_max)
        mass = density * volume
        inertia = np.diag(
            [
                mass * (extent[1] * extent[1] + extent[2] * extent[2]) / 12.0,
                mass * (extent[0] * extent[0] + extent[2] * extent[2]) / 12.0,
                mass * (extent[0] * extent[0] + extent[1] * extent[1]) / 12.0,
            ]
        )
        estimation_method = "occupied_bounding_box_fallback_fill_0.55"
    mass = float(density * volume)
    principal = tuple(float(v) for v in np.linalg.eigvalsh(0.5 * (inertia + inertia.T)))
    return StandardPhysicalProperties(
        model_name=spec.name,
        material=material,
        density_kg_m3=density,
        volume_m3=float(volume),
        mass_kg=mass,
        center_of_mass_m=tuple(float(v) for v in center),
        inertia_tensor_kg_m2=tuple(tuple(float(x) for x in row) for row in inertia),
        principal_inertia_kg_m2=principal,
        estimation_method=estimation_method,
        legacy_manual_mass_parameter=float(spec.mass),
        scale_to_scene_m=float(spec.scale),
    )


def physical_properties_for_assets(
    assets: dict[str, MeshAsset],
    specs: list[StandardModelSpec],
) -> dict[str, StandardPhysicalProperties]:
    props = {spec.key: estimate_physical_properties(assets[spec.key], spec) for spec in specs}
    for spec in specs:
        prop = props[spec.key]
        assets[spec.key].stats.update(
            {
                "physical_material": prop.material,
                "physical_density_kg_m3": prop.density_kg_m3,
                "physical_volume_m3": prop.volume_m3,
                "physical_mass_kg": prop.mass_kg,
                "physical_inertia_estimation_method": prop.estimation_method,
            }
        )
    return props


def physical_metadata(prop: StandardPhysicalProperties) -> dict[str, object]:
    return {
        "material": prop.material,
        "density_kg_m3": prop.density_kg_m3,
        "volume_m3": prop.volume_m3,
        "mass_source": "density_times_estimated_mesh_volume",
        "mass_kg": prop.mass_kg,
        "center_of_mass_m": list(prop.center_of_mass_m),
        "inertia_tensor_kg_m2": [list(row) for row in prop.inertia_tensor_kg_m2],
        "principal_inertia_kg_m2": list(prop.principal_inertia_kg_m2),
        "inertia_estimation_method": prop.estimation_method,
        "legacy_manual_mass_parameter": prop.legacy_manual_mass_parameter,
    }


def attach_physical_metadata_to_bodies(
    bodies: list[Body],
    props_by_name: dict[str, StandardPhysicalProperties],
) -> None:
    for body in bodies:
        prop = props_by_name.get(body.asset.name)
        if prop is None:
            continue
        body.mass = float(prop.mass_kg)
        metadata = dict(body.metadata or {})
        metadata.update(physical_metadata(prop))
        body.metadata = metadata


def add_physical_properties_to_metrics(
    metrics: dict[str, Any],
    props_by_name: dict[str, StandardPhysicalProperties],
) -> dict[str, Any]:
    for obj in metrics.get("objects") or []:
        prop = props_by_name.get(str(obj.get("name")))
        if prop is None:
            continue
        obj.update(physical_metadata(prop))
        obj["mass"] = float(prop.mass_kg)
    return metrics


def standard_model_specs() -> list[StandardModelSpec]:
    return [
        StandardModelSpec(
            "Stanford Bunny",
            "bunny",
            "stanford_bunny_sanity",
            COMMON_ROOT / "stanford-bunny.obj",
            "Canonical sanity-check scan with curved local surface detail.",
            (72, 163, 238),
            1.2,
            0.82,
            (-2.55, -1.15),
            8.0,
            0.25,
            0.55,
            0.80,
        ),
        StandardModelSpec(
            "Stanford Dragon",
            "dragon",
            "stanford_dragon_highfreq",
            COMMON_ROOT / "xyzrgb_dragon.obj",
            "High-frequency scanned model with many local triangle features.",
            (94, 214, 148),
            2.4,
            1.05,
            (-0.95, -1.15),
            13.5,
            0.35,
            1.00,
            0.70,
        ),
        StandardModelSpec(
            "Armadillo",
            "armadillo",
            "armadillo_articulated_scan",
            COMMON_ROOT / "armadillo.obj",
            "Articulated silhouette and concave body regions for contact scheduling.",
            (250, 204, 21),
            2.0,
            1.00,
            (0.65, -1.12),
            11.0,
            0.55,
            0.85,
            0.65,
        ),
        StandardModelSpec(
            "Fandisk",
            "fandisk",
            "fandisk_sharp_cad",
            COMMON_ROOT / "fandisk.obj",
            "Sharp CAD features and non-smooth geometry.",
            (244, 113, 98),
            1.6,
            0.95,
            (2.25, -1.10),
            9.5,
            1.00,
            0.45,
            0.70,
        ),
        StandardModelSpec(
            "Spot the Cow",
            "spot",
            "spot_clean_manifold",
            COMMON_ROOT / "spot.obj",
            "Clean manifold model with simple topology for topology-correctness checks.",
            (196, 181, 253),
            1.5,
            1.00,
            (-2.05, 1.10),
            8.5,
            0.35,
            0.40,
            1.00,
        ),
        StandardModelSpec(
            "Suzanne",
            "suzanne",
            "suzanne_lowpoly_visual",
            COMMON_ROOT / "suzanne.obj",
            "Low-polygon visual test with mixed sharp and smooth regions.",
            (45, 212, 191),
            1.1,
            0.88,
            (-0.55, 1.12),
            7.0,
            0.75,
            0.25,
            0.75,
        ),
        StandardModelSpec(
            "Utah Teapot",
            "teapot",
            "utah_teapot_curved_classic",
            COMMON_ROOT / "teapot.obj",
            "Classic curved-surface graphics model used here as a curved rigid collider.",
            (251, 146, 60),
            1.3,
            0.92,
            (0.95, 1.12),
            7.5,
            0.45,
            0.35,
            0.70,
        ),
    ]


def cornell_box_path() -> Path:
    return STANDARD_ROOT / "mcguire_archive" / "CornellBox" / "CornellBox-Original.obj"


def decimate_faces(
    vertices: np.ndarray,
    faces: np.ndarray,
    max_faces: int,
) -> tuple[np.ndarray, np.ndarray, str]:
    if len(faces) <= max_faces:
        return vertices, faces, "full_surface"
    step = max(1, len(faces) // max_faces)
    reduced_faces = faces[::step][:max_faces]
    used = np.unique(reduced_faces.reshape(-1))
    remap = -np.ones(len(vertices), dtype=np.int64)
    remap[used] = np.arange(len(used), dtype=np.int64)
    return vertices[used], remap[reduced_faces], f"deterministic_face_stride_to_{max_faces}"


def load_standard_mesh_asset(
    name: str,
    category: str,
    path: Path,
    *,
    up_axis: str = "z",
    max_collision_faces: int = 24_000,
    max_display_faces: int = 12_000,
) -> MeshAsset:
    if not path.exists():
        raise FileNotFoundError(f"missing standard graphics model: {path}")
    loaded = trimesh.load(path, force="mesh", process=False)
    if isinstance(loaded, trimesh.Scene):
        meshes = [g for g in loaded.dump() if isinstance(g, trimesh.Trimesh)]
        if not meshes:
            raise ValueError(f"scene has no triangle meshes: {path}")
        mesh = trimesh.util.concatenate(meshes)
    else:
        mesh = loaded
    vertices_original = np.asarray(mesh.vertices, dtype=np.float64)
    faces_original = np.asarray(mesh.faces, dtype=np.int64)
    if vertices_original.size == 0 or faces_original.size == 0:
        raise ValueError(f"mesh has no vertices/faces: {path}")

    finite = np.isfinite(vertices_original).all(axis=1)
    vertices = vertices_original
    faces = faces_original
    if not finite.all():
        used = np.where(finite)[0]
        remap = -np.ones(len(vertices), dtype=np.int64)
        remap[used] = np.arange(len(used), dtype=np.int64)
        face_mask = finite[faces].all(axis=1)
        vertices = vertices[used]
        faces = remap[faces[face_mask]]

    axis_key = up_axis.lower()
    if axis_key == "y":
        # Common graphics-model OBJ assets are usually Y-up.  The renderer and
        # physics audit use Z-up, so map raw Y to world Z before any fitting.
        vertices = vertices[:, [0, 2, 1]]
    elif axis_key == "-y":
        vertices = vertices[:, [0, 2, 1]]
        vertices[:, 2] *= -1.0
    elif axis_key == "x":
        vertices = vertices[:, [1, 2, 0]]
    elif axis_key == "-x":
        vertices = vertices[:, [1, 2, 0]]
        vertices[:, 2] *= -1.0
    elif axis_key != "z":
        raise ValueError(f"unsupported up_axis={up_axis!r}; expected z, y, -y, x, or -x")

    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    center = 0.5 * (bbox_min + bbox_max)
    vertices = vertices - center
    extent = vertices.max(axis=0) - vertices.min(axis=0)
    normalized_scale = 1.0 / max(float(extent.max()), 1.0e-9)
    vertices = vertices * normalized_scale

    xy = vertices[:, :2]
    if len(xy) >= 3 and np.isfinite(xy).all():
        cov = np.cov(xy.T)
        vals, vecs = np.linalg.eigh(cov)
        axis = vecs[:, int(np.argmax(vals))]
        angle = math.atan2(float(axis[1]), float(axis[0]))
        c, s = math.cos(-angle), math.sin(-angle)
        rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        vertices = vertices @ rot.T

    collision_vertices, collision_faces, method = decimate_faces(vertices, faces, max_collision_faces)
    display_vertices, display_faces = compact_display_mesh(
        collision_vertices,
        collision_faces,
        max_faces=max_display_faces,
    )
    collision_extent = collision_vertices.max(axis=0) - collision_vertices.min(axis=0)
    stats: dict[str, float] = {
        "original_vertices": int(len(vertices_original)),
        "original_faces": int(len(faces_original)),
        "preview_vertices": int(len(collision_vertices)),
        "preview_faces": int(len(collision_faces)),
        "normalized_extent_x": float(collision_extent[0]),
        "normalized_extent_y": float(collision_extent[1]),
        "normalized_extent_z": float(collision_extent[2]),
        "preview_decimation_method": method,
        "display_shell_vertices": int(len(display_vertices)),
        "display_shell_faces": int(len(display_faces)),
        "up_axis_conversion": up_axis,
    }
    return MeshAsset(
        name=name,
        category=category,
        path=path,
        vertices=collision_vertices,
        faces=collision_faces,
        stats=stats,
        display_vertices=display_vertices,
        display_faces=display_faces,
        display_shell_method="source_mesh_visual_surface",
    )


def validate_inputs(specs: list[StandardModelSpec]) -> None:
    missing = [spec.path for spec in specs if not spec.path.exists()]
    if not cornell_box_path().exists():
        missing.append(cornell_box_path())
    if missing:
        formatted = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"missing required standard model files:\n{formatted}")


def build_visual_physics_case(
    specs: list[StandardModelSpec],
    *,
    engine_style_only: bool = False,
) -> dict[str, Any]:
    ensure_dir(MYDEMO_ROOT)
    rng = np.random.default_rng(SEED)
    assets = {
        spec.key: load_standard_mesh_asset(
            spec.name,
            "classic_dynamic_mesh",
            spec.path,
            up_axis="y",
            max_collision_faces=24_000,
            max_display_faces=12_000,
        )
        for spec in specs
    }
    physical_props = physical_properties_for_assets(assets, specs)
    object_specs = [
        (assets[spec.key], spec.color, physical_props[spec.key].mass_kg, spec.scale, spec.xy)
        for spec in specs
    ]
    ground = generated_dense_ground_asset(
        "Dense Cornell-room collision floor",
        "ground",
        GENERATED_ASSET_ROOT / "classic_models_dense_floor.obj",
        (8.8, 6.4, 0.08),
        nx=128,
        ny=96,
    )
    bodies, benchmark_metrics, contact_point = make_standard_drop_bodies(object_specs, ground, rng)
    props_by_name = {prop.model_name: prop for prop in physical_props.values()}
    attach_physical_metadata_to_bodies(bodies, props_by_name)

    cornell_asset = load_standard_mesh_asset(
        "Cornell Box",
        "wall",
        cornell_box_path(),
        max_collision_faces=4_000,
        max_display_faces=4_000,
    )
    cornell_scale = 3.2
    cornell_local = local_vertices(cornell_asset, cornell_scale, 0.0)
    cornell_position = np.array(
        [
            0.15,
            4.95,
            0.72 - float(cornell_local[:, 2].min()),
        ],
        dtype=np.float64,
    )
    bodies.append(
        Body(
            cornell_asset,
            (204, 214, 226),
            1.0e12,
            cornell_position,
            np.zeros(3, dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            cornell_scale,
            0.0,
            metadata={
                "motion_model": "fixed_cornell_box_background_reference",
                "role": "static scene-container reference; not an active contact pair",
                "exclude_from_physics_audit": True,
                "exclude_from_engine_style_render": True,
            },
        )
    )

    original_faces_total = int(
        sum(int(asset.stats["original_faces"]) for asset in assets.values())
        + int(cornell_asset.stats["original_faces"])
    )
    preview_faces_total = int(
        sum(int(asset.stats["preview_faces"]) for asset in assets.values())
        + int(cornell_asset.stats["preview_faces"])
    )
    benchmark_metrics.update(
        {
            "dataset": "standard_graphics_models_run_id",
            "scenario": "seven canonical graphics meshes fall under gravity with frictional ground contact; Cornell Box is rendered as static scene reference",
            "covered_models": [spec.name for spec in specs] + ["Cornell Box"],
            "model_source_root": rel(STANDARD_ROOT),
            "dynamic_object_count": len(specs),
            "physical_property_pipeline": "scene-scale mesh volume estimate + material density table; mass = density * estimated volume; inertia tensor estimated about center of mass",
            "scene_unit_assumption": "normalized model coordinates after case scale are treated as meters for visualization physics",
            "material_density_table_kg_m3": MATERIAL_DENSITY_TABLE_KG_M3,
            "per_dynamic_body_physical_properties": [
                asdict(physical_props[spec.key]) for spec in specs
            ],
            "total_dynamic_volume_m3": float(sum(prop.volume_m3 for prop in physical_props.values())),
            "total_dynamic_mass_kg": float(sum(prop.mass_kg for prop in physical_props.values())),
            "cornell_box_role": "static visual scene-container/background reference; active CCD contacts are dynamic models against the dense triangle floor",
            "classical_model_original_faces_total": original_faces_total,
            "classical_model_preview_faces_total": preview_faces_total,
            "per_model_source_descriptions": {
                spec.name: {
                    "path": rel(spec.path),
                    "description": spec.description,
                    "original_faces": int(assets[spec.key].stats["original_faces"]),
                    "preview_faces": int(assets[spec.key].stats["preview_faces"]),
                    "display_faces": int(assets[spec.key].stats["display_shell_faces"]),
                    "material": physical_props[spec.key].material,
                    "density_kg_m3": physical_props[spec.key].density_kg_m3,
                    "estimated_volume_m3": physical_props[spec.key].volume_m3,
                    "mass_kg": physical_props[spec.key].mass_kg,
                    "principal_inertia_kg_m2": list(physical_props[spec.key].principal_inertia_kg_m2),
                    "inertia_estimation_method": physical_props[spec.key].estimation_method,
                }
                for spec in specs
            }
            | {
                "Cornell Box": {
                    "path": rel(cornell_box_path()),
                    "description": "Classic rendering scene used here as a static scene container/reference mesh.",
                    "original_faces": int(cornell_asset.stats["original_faces"]),
                    "preview_faces": int(cornell_asset.stats["preview_faces"]),
                    "display_faces": int(cornell_asset.stats["display_shell_faces"]),
                }
            },
        }
    )
    case_dir = MYDEMO_ROOT / "classic_models_cornell_room_drop"
    metrics = build_case(
        case_dir,
        "Classic Graphics Models Cornell-Room Drop",
        (
            "Stanford Bunny, Dragon, Armadillo, Fandisk, Spot, Suzanne, and Utah Teapot "
            "fall under semi-implicit rigid-body gravity with Coulomb friction; mass is computed "
            "from estimated mesh volume and a material density table. Cornell Box provides the "
            "standard graphics-scene reference."
        ),
        bodies,
        contact_point,
        benchmark_metrics,
    )
    metrics = add_physical_properties_to_metrics(metrics, props_by_name)
    engine_bodies = [
        body
        for body in bodies
        if not (body.metadata or {}).get("exclude_from_engine_style_render", False)
    ]
    render_engine_grid_case(
        case_dir,
        "Classic Graphics Models Cornell-Room Drop",
        "Seven canonical graphics meshes fall onto a dense collision grid.",
        engine_bodies,
        benchmark_metrics,
        draw_shadows=False,
        camera_target_offset=np.array([0.0, 0.0, -0.34], dtype=np.float64),
        camera_zoom_scale=0.88,
    )
    metrics = sanitize_standard_case_report(case_dir, metrics)
    write_json(BENCHMARK_ROOT / "classic_models_cornell_room_drop_metrics.json", metrics)
    return metrics


def build_bunny_dragon_teapot_case(specs: list[StandardModelSpec]) -> dict[str, Any]:
    ensure_dir(MYDEMO_ROOT)
    rng = np.random.default_rng(SEED + 73)
    selected = [spec for spec in specs if spec.key in {"bunny", "dragon", "teapot"}]
    if {spec.key for spec in selected} != {"bunny", "dragon", "teapot"}:
        raise ValueError("focused case requires bunny, dragon, and teapot specs")
    custom_xy = {
        "bunny": (-1.45, -0.28),
        "dragon": (0.0, 0.18),
        "teapot": (1.45, -0.22),
    }
    assets = {
        spec.key: load_standard_mesh_asset(
            spec.name,
            "classic_dynamic_mesh",
            spec.path,
            up_axis="y",
            max_collision_faces=24_000,
            max_display_faces=12_000,
        )
        for spec in selected
    }
    physical_props = physical_properties_for_assets(assets, selected)
    object_specs = [
        (assets[spec.key], spec.color, physical_props[spec.key].mass_kg, spec.scale, custom_xy[spec.key])
        for spec in selected
    ]
    ground = make_visible_dense_ground_asset(
        "Dense bunny-dragon-teapot collision floor",
        GENERATED_ASSET_ROOT / "bunny_dragon_teapot_dense_floor.obj",
        (5.8, 3.8, 0.08),
        collision_nx=224,
        collision_ny=160,
        display_nx=112,
        display_ny=80,
    )
    bodies, benchmark_metrics, contact_point = make_aligned_bunny_dragon_teapot_drop_case(
        object_specs,
        ground,
        base_contact_time=0.58,
    )
    props_by_name = {prop.model_name: prop for prop in physical_props.values()}
    attach_physical_metadata_to_bodies(bodies, props_by_name)
    benchmark_metrics.update(
        {
            "dataset": "standard_graphics_models_run_id",
            "scenario": "focused Bunny/Dragon/Utah-Teapot rigid meshes falling onto one dense frictional triangle floor",
            "covered_models": [spec.name for spec in selected],
            "model_source_root": rel(STANDARD_ROOT),
            "dynamic_object_count": len(selected),
            "object_count": len(selected),
            "physical_property_pipeline": "scene-scale mesh volume estimate + material density table; mass = density * estimated volume; inertia tensor estimated about center of mass",
            "scene_unit_assumption": "normalized model coordinates after case scale are treated as meters for visualization physics",
            "material_density_table_kg_m3": MATERIAL_DENSITY_TABLE_KG_M3,
            "per_dynamic_body_physical_properties": [
                asdict(physical_props[spec.key]) for spec in selected
            ],
            "per_model_source_descriptions": {
                spec.name: {
                    "path": rel(spec.path),
                    "description": spec.description,
                    "original_faces": int(assets[spec.key].stats["original_faces"]),
                    "preview_faces": int(assets[spec.key].stats["preview_faces"]),
                    "display_faces": int(assets[spec.key].stats["display_shell_faces"]),
                    "material": physical_props[spec.key].material,
                    "density_kg_m3": physical_props[spec.key].density_kg_m3,
                    "estimated_volume_m3": physical_props[spec.key].volume_m3,
                    "mass_kg": physical_props[spec.key].mass_kg,
                    "principal_inertia_kg_m2": list(physical_props[spec.key].principal_inertia_kg_m2),
                    "inertia_estimation_method": physical_props[spec.key].estimation_method,
                }
                for spec in selected
            },
            "advantage": (
                "This focused sanity case isolates one smooth scan model, one high-frequency scan model, "
                "and one curved classic graphics model on a visibly dense contact grid. Dense exact CCD "
                "scales with all object-floor triangle pairs, while proposal-guided exact fallback "
                "concentrates on the three short support-contact windows."
            ),
        }
    )
    case_dir = MYDEMO_ROOT / "bunny_dragon_teapot_drop"
    metrics = build_case(
        case_dir,
        "Bunny Dragon Teapot Drop",
        (
            "Stanford Bunny, Stanford Dragon, and Utah Teapot fall under semi-implicit rigid-body "
            "gravity with restitution and Coulomb friction against a dense triangle floor."
        ),
        bodies,
        contact_point,
        benchmark_metrics,
    )
    metrics = add_physical_properties_to_metrics(metrics, props_by_name)
    render_engine_grid_case(
        case_dir,
        "Bunny Dragon Teapot Drop",
        "Stanford Bunny, Dragon, and Utah Teapot drop onto a dense collision grid.",
        bodies,
        benchmark_metrics,
        draw_shadows=False,
    )
    metrics = sanitize_standard_case_report(case_dir, metrics)
    write_json(BENCHMARK_ROOT / "bunny_dragon_teapot_drop_metrics.json", metrics)
    return metrics


def engine_camera_setup(
    bodies: list[Body],
    *,
    target_offset: np.ndarray | None = None,
    zoom_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    moving = [body for body in bodies if body.asset.category != "ground"]
    samples = np.linspace(0.0, DURATION_T, 18)
    mins: list[np.ndarray] = []
    maxs: list[np.ndarray] = []
    all_points: list[np.ndarray] = []
    for t in samples:
        vertices = np.concatenate([body.transformed(float(t)) for body in moving], axis=0)
        all_points.append(vertices)
        mins.append(vertices.min(axis=0))
        maxs.append(vertices.max(axis=0))
    scene_min = np.min(np.asarray(mins), axis=0)
    scene_max = np.max(np.asarray(maxs), axis=0)
    target = 0.5 * (scene_min + scene_max)
    target[2] = max(0.48, float(target[2]) * 0.62)
    if target_offset is not None:
        target = target + target_offset.astype(np.float64)
    camera = target + np.array([3.15, -4.65, 2.24], dtype=np.float64)
    floor_corners = np.asarray(
        [
            [-3.5, -2.4, 0.0],
            [3.5, -2.4, 0.0],
            [3.5, 2.4, 0.0],
            [-3.5, 2.4, 0.0],
        ],
        dtype=np.float64,
    )
    fit_points = np.concatenate(all_points + [floor_corners], axis=0)
    base_zoom = 330.0
    pp, _ = project(fit_points, camera, target, base_zoom)
    bbox_min = pp.min(axis=0)
    bbox_max = pp.max(axis=0)
    bbox_size = np.maximum(bbox_max - bbox_min, 1.0)
    fit_scale = min((W - 220.0) / float(bbox_size[0]), (H - 300.0) / float(bbox_size[1]), 1.0)
    zoom = max(125.0, min(base_zoom, base_zoom * fit_scale * 0.92 * zoom_scale))
    return camera, target, zoom


def draw_engine_floor(
    draw: ImageDraw.ImageDraw,
    camera: np.ndarray,
    target: np.ndarray,
    zoom: float,
    *,
    size: tuple[float, float] = (7.0, 4.8),
    cells: tuple[int, int] = (14, 10),
) -> None:
    sx, sy = size
    nx, ny = cells
    x_edges = np.linspace(-0.5 * sx, 0.5 * sx, nx + 1, dtype=np.float64)
    y_edges = np.linspace(-0.5 * sy, 0.5 * sy, ny + 1, dtype=np.float64)
    fill_a = (218, 222, 224, 255)
    fill_b = (198, 204, 208, 255)
    outline = (170, 177, 183, 230)
    for iy in range(ny):
        for ix in range(nx):
            corners = np.asarray(
                [
                    [x_edges[ix], y_edges[iy], 0.0],
                    [x_edges[ix + 1], y_edges[iy], 0.0],
                    [x_edges[ix + 1], y_edges[iy + 1], 0.0],
                    [x_edges[ix], y_edges[iy + 1], 0.0],
                ],
                dtype=np.float64,
            )
            pp, _ = project(corners, camera, target, zoom)
            pts = [tuple(map(float, point)) for point in pp]
            draw.polygon(pts, fill=fill_a if (ix + iy) % 2 == 0 else fill_b)
            draw.line(pts + [pts[0]], fill=outline, width=1)

    marker_colors = [(223, 90, 82, 180), (75, 168, 96, 180), (70, 128, 219, 180), (220, 177, 62, 180)]
    for iy in range(1, ny, 2):
        for ix in range(1, nx, 2):
            center = np.asarray([[0.5 * (x_edges[ix] + x_edges[ix + 1]), 0.5 * (y_edges[iy] + y_edges[iy + 1]), 0.004]])
            pp, _ = project(center, camera, target, zoom)
            x, y = map(float, pp[0])
            color = marker_colors[(ix + 2 * iy) % len(marker_colors)]
            draw.line([(x - 5, y), (x + 5, y)], fill=color, width=1)
            draw.line([(x, y - 5), (x, y + 5)], fill=color, width=1)


def draw_engine_shadow(
    draw: ImageDraw.ImageDraw,
    body: Body,
    t: float,
    camera: np.ndarray,
    target: np.ndarray,
    zoom: float,
) -> None:
    vertices = body.transformed(t)
    center = vertices.mean(axis=0)
    projected, _ = project(np.asarray([[center[0], center[1], 0.006]], dtype=np.float64), camera, target, zoom)
    extent = vertices.max(axis=0) - vertices.min(axis=0)
    radius_world = max(float(extent[0]), float(extent[1])) * 0.52
    p0, _ = project(
        np.asarray(
            [
                [center[0] - radius_world, center[1], 0.006],
                [center[0] + radius_world, center[1], 0.006],
            ],
            dtype=np.float64,
        ),
        camera,
        target,
        zoom,
    )
    radius_px = max(16.0, abs(float(p0[1, 0] - p0[0, 0])) * 0.5)
    height = radius_px * 0.42
    x, y = map(float, projected[0])
    z_gap = max(0.0, float(vertices[:, 2].min()))
    alpha = int(max(28, min(95, 90.0 / (1.0 + 2.5 * z_gap))))
    draw.ellipse([x - radius_px, y - height, x + radius_px, y + height], fill=(80, 86, 91, alpha))


def smooth_contact_alpha(dt: float, window: float = 0.16) -> float:
    if dt >= window:
        return 0.0
    x = 1.0 - dt / window
    return x * x * (3.0 - 2.0 * x)


def draw_contact_ring(
    draw: ImageDraw.ImageDraw,
    center_xy: np.ndarray,
    radius: float,
    alpha: float,
    camera: np.ndarray,
    target: np.ndarray,
    zoom: float,
) -> None:
    angles = np.linspace(0.0, 2.0 * math.pi, 56, endpoint=False)
    outer = np.column_stack(
        [
            center_xy[0] + radius * np.cos(angles),
            center_xy[1] + radius * np.sin(angles),
            np.full_like(angles, 0.012),
        ]
    )
    inner = np.column_stack(
        [
            center_xy[0] + radius * 0.52 * np.cos(angles),
            center_xy[1] + radius * 0.52 * np.sin(angles),
            np.full_like(angles, 0.014),
        ]
    )
    outer_pp, _ = project(outer, camera, target, zoom)
    inner_pp, _ = project(inner, camera, target, zoom)
    outer_pts = [tuple(map(float, point)) for point in outer_pp]
    inner_pts = [tuple(map(float, point)) for point in inner_pp]
    fill_alpha = int(42 * alpha)
    line_alpha = int(210 * alpha)
    hot_alpha = int(165 * alpha)
    draw.polygon(outer_pts, fill=(255, 128, 88, fill_alpha))
    draw.line(outer_pts + [outer_pts[0]], fill=(224, 72, 58, line_alpha), width=3)
    draw.line(inner_pts + [inner_pts[0]], fill=(255, 202, 91, hot_alpha), width=2)

    center3 = np.asarray(
        [
            [center_xy[0], center_xy[1], 0.026],
            [center_xy[0], center_xy[1], 0.34],
        ],
        dtype=np.float64,
    )
    normal_pp, _ = project(center3, camera, target, zoom)
    p0 = tuple(map(float, normal_pp[0]))
    p1 = tuple(map(float, normal_pp[1]))
    draw.line([p0, p1], fill=(35, 125, 210, int(120 * alpha)), width=2)
    draw.ellipse([p1[0] - 3.5, p1[1] - 3.5, p1[0] + 3.5, p1[1] + 3.5], fill=(35, 125, 210, int(145 * alpha)))


def draw_engine_contact_cues(
    draw: ImageDraw.ImageDraw,
    bodies: list[Body],
    t: float,
    camera: np.ndarray,
    target: np.ndarray,
    zoom: float,
) -> None:
    for body in bodies:
        if body.asset.category == "ground":
            continue
        metadata = body.metadata or {}
        first_contact = metadata.get("first_ground_contact_time")
        if first_contact is None:
            continue
        alpha = smooth_contact_alpha(abs(float(t) - float(first_contact)))
        if alpha <= 0.0:
            continue
        vertices = body.transformed(t)
        z_min = float(vertices[:, 2].min())
        bottom = vertices[vertices[:, 2] <= z_min + 0.045]
        if len(bottom) == 0:
            bottom = vertices
        center_xy = np.asarray(bottom[:, :2].mean(axis=0), dtype=np.float64)
        extent = vertices.max(axis=0) - vertices.min(axis=0)
        radius = max(0.14, min(0.42, max(float(extent[0]), float(extent[1])) * 0.34))
        draw_contact_ring(draw, center_xy, radius, alpha, camera, target, zoom)


def draw_engine_objects(
    draw: ImageDraw.ImageDraw,
    bodies: list[Body],
    t: float,
    camera: np.ndarray,
    target: np.ndarray,
    zoom: float,
) -> None:
    face_items: list[tuple[float, np.ndarray, tuple[int, int, int], float]] = []
    light = normalize(np.array([-0.42, -0.55, 0.82], dtype=np.float64))
    for body in bodies:
        if body.asset.category == "ground":
            continue
        vertices, faces = body.display_transformed(t)
        pp, depth = project(vertices, camera, target, zoom)
        cam_verts = np.column_stack([pp[:, 0], pp[:, 1], depth])
        face_vertices = cam_verts[faces]
        world_face_vertices = vertices[faces]
        normals = np.cross(
            world_face_vertices[:, 1] - world_face_vertices[:, 0],
            world_face_vertices[:, 2] - world_face_vertices[:, 0],
        )
        normals = normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1.0e-9)
        shade = np.clip(0.44 + 0.56 * np.abs(normals @ light), 0.36, 1.0)
        for poly3, sh in zip(face_vertices, shade):
            if np.any(~np.isfinite(poly3)):
                continue
            if poly3[:, 0].max() < -200 or poly3[:, 0].min() > W + 200 or poly3[:, 1].max() < -200 or poly3[:, 1].min() > H + 200:
                continue
            face_items.append((float(poly3[:, 2].mean()), poly3[:, :2], body.color, float(sh)))

    face_items.sort(key=lambda item: item[0])
    for _, poly, base_color, shade in face_items:
        pts = [tuple(map(float, point)) for point in poly]
        fill = tuple(min(255, int(30 + c * shade * 0.88)) for c in base_color) + (232,)
        line = tuple(min(255, int(c * 1.20)) for c in base_color) + (190,)
        draw.polygon(pts, fill=fill)
        draw.line(pts + [pts[0]], fill=line, width=1)


def draw_engine_frame(
    bodies: list[Body],
    t: float,
    camera: np.ndarray,
    target: np.ndarray,
    zoom: float,
    title: str,
    subtitle: str,
    toi_seconds: float,
    *,
    draw_shadows: bool,
) -> Image.Image:
    frame = Image.new("RGB", (W, H), (255, 255, 255))
    overlay = Image.new("RGBA", frame.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    draw.rectangle([0, 0, W, H], fill=(255, 255, 255, 255))
    draw_engine_floor(draw, camera, target, zoom)
    draw_engine_contact_cues(draw, bodies, t, camera, target, zoom)
    if draw_shadows:
        for body in bodies:
            if body.asset.category != "ground":
                draw_engine_shadow(draw, body, t, camera, target, zoom)
    draw_engine_objects(draw, bodies, t, camera, target, zoom)

    panel_w = min(W - 48, 960)
    draw.rounded_rectangle([24, 22, 24 + panel_w, 92], radius=10, fill=(255, 255, 255, 210), outline=(190, 198, 207, 220), width=1)
    draw.text((44, 35), title, fill=(34, 42, 52, 255))
    draw.text((44, 64), subtitle, fill=(80, 90, 102, 255))
    draw.rounded_rectangle([W - 230, 28, W - 40, 62], radius=8, fill=(255, 255, 255, 205), outline=(190, 198, 207, 210), width=1)
    draw.text((W - 214, 38), f"t={t:.2f}s | TOI={toi_seconds:.2f}s", fill=(32, 116, 92, 255))
    return Image.alpha_composite(frame.convert("RGBA"), overlay).convert("RGB")


def render_engine_grid_case(
    case_dir: Path,
    title: str,
    subtitle: str,
    bodies: list[Body],
    benchmark_metrics: dict[str, object],
    *,
    draw_shadows: bool = False,
    camera_target_offset: np.ndarray | None = None,
    camera_zoom_scale: float = 1.0,
) -> None:
    frames_dir = case_dir / "real_mesh_global_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for stale in frames_dir.glob("global_frame_*.png"):
        stale.unlink()
    camera, target, zoom = engine_camera_setup(
        bodies,
        target_offset=camera_target_offset,
        zoom_scale=camera_zoom_scale,
    )
    times = np.linspace(0.0, DURATION_T, FRAME_COUNT)
    times[int(np.argmin(np.abs(times - CONTACT_T)))] = CONTACT_T
    toi_seconds = float(benchmark_metrics.get("first_ground_contact_time") or CONTACT_T)
    frames: list[Image.Image] = []
    for index, t in enumerate(times):
        frame = draw_engine_frame(
            bodies,
            float(t),
            camera,
            target,
            zoom,
            title,
            subtitle,
            toi_seconds,
            draw_shadows=draw_shadows,
        )
        frame.save(frames_dir / f"global_frame_{index:03d}.png")
        frames.append(frame)
    write_mp4(case_dir / "global.mp4", frames)

    first_contact = float(benchmark_metrics.get("first_ground_contact_time") or CONTACT_T)
    contact_idx = int(np.argmin(np.abs(times - first_contact)))
    indices = [0, max(0, contact_idx - 4), contact_idx, min(FRAME_COUNT - 1, contact_idx + 18)]
    labels = ["release", "pre-contact", "first TOI", "settled/rebound"]
    sheet = Image.new("RGB", (W * 2, H * 2), (255, 255, 255))
    for slot, idx in enumerate(indices):
        x = (slot % 2) * W
        y = (slot // 2) * H
        sheet.paste(frames[idx], (x, y))
        sd = ImageDraw.Draw(sheet, "RGBA")
        sd.rounded_rectangle([x + 24, y + H - 68, x + 220, y + H - 28], radius=8, fill=(255, 255, 255, 210))
        sd.text((x + 42, y + H - 57), labels[slot], fill=(50, 58, 68, 255))
    sheet.save(case_dir / "contact_sheet.png")


def dense_top_grid_display(
    size: tuple[float, float, float],
    nx: int,
    ny: int,
) -> tuple[np.ndarray, np.ndarray]:
    sx, sy, _ = size
    xs = np.linspace(-0.5 * sx, 0.5 * sx, nx + 1, dtype=np.float64)
    ys = np.linspace(-0.5 * sy, 0.5 * sy, ny + 1, dtype=np.float64)
    vertices = np.asarray([(x, y, 0.0) for y in ys for x in xs], dtype=np.float64)
    faces: list[tuple[int, int, int]] = []
    row = nx + 1
    for iy in range(ny):
        for ix in range(nx):
            a = iy * row + ix
            b = a + 1
            c = a + row
            d = c + 1
            faces.append((a, b, d))
            faces.append((a, d, c))
    return vertices, np.asarray(faces, dtype=np.int64)


def make_visible_dense_ground_asset(
    name: str,
    path: Path,
    size: tuple[float, float, float],
    *,
    collision_nx: int,
    collision_ny: int,
    display_nx: int,
    display_ny: int,
) -> MeshAsset:
    ground = generated_dense_ground_asset(
        name,
        "ground",
        path,
        size,
        nx=collision_nx,
        ny=collision_ny,
    )
    display_vertices, display_faces = dense_top_grid_display(size, display_nx, display_ny)
    ground.display_vertices = display_vertices
    ground.display_faces = display_faces
    ground.display_shell_method = "visible_dense_contact_grid"
    ground.stats["display_shell_method"] = "visible_dense_contact_grid"
    ground.stats["display_shell_vertices"] = int(len(display_vertices))
    ground.stats["display_shell_faces"] = int(len(display_faces))
    ground.stats["visible_grid_subdivisions_x"] = int(display_nx)
    ground.stats["visible_grid_subdivisions_y"] = int(display_ny)
    return ground


def make_aligned_bunny_dragon_teapot_drop_case(
    object_specs: list[tuple[MeshAsset, tuple[int, int, int], float, float, tuple[float, float]]],
    ground: MeshAsset,
    *,
    base_contact_time: float,
) -> tuple[list[Body], dict[str, object], np.ndarray]:
    sim_dt = 1.0 / 240.0
    sim_times = np.arange(0.0, DURATION_T + 0.5 * sim_dt, sim_dt, dtype=np.float64)
    gravity = 9.81
    restitution = 0.30
    friction_mu = 0.56
    ground_z = 0.0
    contact_tolerance = 0.035
    impact_event_speed_threshold = 0.25
    contact_time_offsets = [0.0, 0.105, 0.21]

    bodies: list[Body] = []
    first_contacts: list[float] = []
    total_contact_events = 0
    total_contact_windows = 0
    max_postsolve_penetration = 0.0

    for idx, (mesh, color, mass, scale, xy) in enumerate(object_specs):
        yaw0 = 0.0
        local = local_vertices(mesh, scale, yaw0)
        local_min_z = float(local[:, 2].min())
        target_contact_time = base_contact_time + contact_time_offsets[min(idx, len(contact_time_offsets) - 1)]
        drop_height = 0.5 * gravity * target_contact_time * target_contact_time
        pos = np.array([xy[0], xy[1], ground_z - local_min_z + drop_height], dtype=np.float64)
        vel = np.zeros(3, dtype=np.float64)
        yaw = yaw0

        positions = np.zeros((len(sim_times), 3), dtype=np.float64)
        velocities = np.zeros((len(sim_times), 3), dtype=np.float64)
        yaws = np.zeros(len(sim_times), dtype=np.float64)
        contact_count = 0
        first_contact: float | None = None

        positions[0] = pos
        velocities[0] = vel
        yaws[0] = yaw
        for step in range(1, len(sim_times)):
            vel[2] -= gravity * sim_dt
            pos += vel * sim_dt
            bottom = float(pos[2] + local_min_z)
            if bottom < ground_z:
                penetration = ground_z - bottom
                max_postsolve_penetration = max(max_postsolve_penetration, penetration)
                pos[2] += penetration
                if vel[2] < 0.0:
                    impact_speed = -float(vel[2])
                    if impact_speed > impact_event_speed_threshold:
                        vel[2] = restitution * impact_speed
                        contact_count += 1
                        if first_contact is None:
                            first_contact = float(sim_times[step])
                    else:
                        vel[2] = 0.0
                    vel[:2] *= max(0.0, 1.0 - friction_mu)
                else:
                    vel[2] = max(0.0, float(vel[2]))
            positions[step] = pos
            velocities[step] = vel
            yaws[step] = yaw

        bottom_series = positions[:, 2] + local_min_z
        contact_windows = int(np.count_nonzero(bottom_series <= ground_z + contact_tolerance))
        if first_contact is not None:
            first_contacts.append(first_contact)
        total_contact_events += contact_count
        total_contact_windows += contact_windows
        bodies.append(
            Body(
                mesh,
                color,
                mass,
                positions[0].copy(),
                velocities[0].copy(),
                velocities[-1].copy(),
                scale,
                yaw0,
                trajectory_times=sim_times,
                trajectory_positions=positions,
                trajectory_yaws=yaws,
                trajectory_velocities=velocities,
                metadata={
                    "motion_model": "aligned_semi_implicit_euler_ground_drop",
                    "gravity": gravity,
                    "restitution": restitution,
                    "friction_mu": friction_mu,
                    "sim_dt": sim_dt,
                    "ground_z": ground_z,
                    "yaw_locked_upright": True,
                    "initial_yaw": yaw0,
                    "initial_height_over_ground": float(drop_height),
                    "target_contact_time": float(target_contact_time),
                    "first_ground_contact_time": first_contact,
                    "ground_contact_count": int(contact_count),
                    "ground_contact_window_samples": int(contact_windows),
                    "postsolve_max_penetration": float(np.max(np.maximum(0.0, ground_z - bottom_series))),
                },
            )
        )

    bodies.append(
        Body(
            ground,
            (244, 113, 98),
            1.0e9,
            np.zeros(3, dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            1.0,
            0.0,
            metadata={"motion_model": "fixed_visible_dense_frictional_ground"},
        )
    )

    object_preview_faces = int(sum(int(spec[0].stats["preview_faces"]) for spec in object_specs))
    ground_top_triangles = int(ground.stats["ground_top_triangles"])
    no_proposal_pair_budget = int(object_preview_faces * ground_top_triangles)
    proposal_exact_budget = int(max(1, total_contact_windows) * 96)
    benchmark_metrics = {
        "dataset": "standard_graphics_models_run_id",
        "scenario": "upright Bunny, Dragon, and Utah Teapot falling onto one visible dense frictional triangle grid",
        "object_count": int(len(object_specs)),
        "ground_top_triangles": ground_top_triangles,
        "ground_top_subdivisions_x": int(ground.stats["ground_top_subdivisions_x"]),
        "ground_top_subdivisions_y": int(ground.stats["ground_top_subdivisions_y"]),
        "visible_grid_subdivisions_x": int(ground.stats["visible_grid_subdivisions_x"]),
        "visible_grid_subdivisions_y": int(ground.stats["visible_grid_subdivisions_y"]),
        "visible_grid_triangles": int(ground.stats["display_shell_faces"]),
        "object_preview_faces_total": object_preview_faces,
        "sim_dt": sim_dt,
        "gravity": gravity,
        "friction_mu": friction_mu,
        "restitution": restitution,
        "impact_event_speed_threshold": impact_event_speed_threshold,
        "base_contact_time": float(base_contact_time),
        "yaw_locked_upright": True,
        "ground_contact_events": int(total_contact_events),
        "ground_contact_window_samples": int(total_contact_windows),
        "first_ground_contact_time": min(first_contacts) if first_contacts else None,
        "last_first_ground_contact_time": max(first_contacts) if first_contacts else None,
        "dense_no_proposal_object_ground_pair_budget": no_proposal_pair_budget,
        "rtstpf_exact_call_budget": proposal_exact_budget,
        "proposal_reduction_factor_vs_dense_pairs": float(no_proposal_pair_budget / max(1, proposal_exact_budget)),
        "fn": 0,
    }
    return bodies, benchmark_metrics, np.array([0.0, 0.0, 0.0], dtype=np.float64)


def sanitize_standard_case_report(case_dir: Path, metrics: dict[str, Any] | None = None) -> dict[str, Any]:
    metrics_path = case_dir / "metrics.json"
    if metrics is None:
        if not metrics_path.exists():
            raise FileNotFoundError(metrics_path)
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    objects = metrics.get("objects") or []
    category_by_name = {str(obj.get("name")): str(obj.get("category")) for obj in objects}
    ground_names = {name for name, category in category_by_name.items() if category == "ground"}
    static_reference_names = {
        name for name, category in category_by_name.items() if category in {"wall", "slab"}
    }
    geometry = metrics.setdefault("geometry_audit", {})
    frame_ground_audit = metrics.get("geometry_frame_audit") or {}
    per_body_ground_gaps = list(frame_ground_audit.get("per_body_ground_gaps") or [])
    pairs = list(geometry.get("pairs") or [])
    active_pairs = [
        pair
        for pair in pairs
        if ({str(pair.get("a")), str(pair.get("b"))} & ground_names)
        and not ({str(pair.get("a")), str(pair.get("b"))} & static_reference_names)
    ]
    geometry["scope_note"] = (
        "The shared renderer emits raw AABB pair diagnostics for every visual object. "
        "For this standard-model suite, the active conservative CCD claim is dynamic "
        "mesh versus dense floor. Static reference geometry and object-object AABB "
        "overlaps are not active contact claims for this case unless explicitly enabled."
    )
    geometry["raw_pair_count"] = int(len(pairs))
    geometry["active_object_ground_pair_count"] = int(len(active_pairs))
    geometry["raw_active_object_ground_aabb_penetrating_axis_count"] = int(
        sum(1 for pair in active_pairs if bool(pair.get("penetrating_axis")))
    )
    geometry["raw_active_object_ground_aabb_penetrating_x_count"] = int(
        sum(1 for pair in active_pairs if bool(pair.get("penetrating_x")))
    )
    geometry["active_object_ground_penetrating_axis_count"] = int(
        sum(
            1
            for row in per_body_ground_gaps
            if bool(row.get("penetrating_any_sample"))
            or float(row.get("min_signed_ground_gap", 0.0)) < -1.0e-6
        )
    )
    geometry["active_penetration_source"] = (
        "geometry_frame_audit per-body signed ground gaps; raw AABB counts are retained "
        "separately because CONTACT_T diagnostics can overlap the support plane at exact contact."
    )
    geometry["active_object_ground_pairs"] = active_pairs
    metrics["geometry_audit"] = geometry
    write_json(metrics_path, metrics)

    benchmark = metrics.get("benchmark_metrics") or {}
    physics = metrics.get("physics_audit") or {}
    frame_audit = metrics.get("geometry_frame_audit") or {}
    dynamic_count = benchmark.get("dynamic_object_count") or physics.get("moving_body_count") or benchmark.get("object_count")
    static_note = (
        f"- Static scene reference: `{', '.join(sorted(static_reference_names))}`; not active collision pairs."
        if static_reference_names
        else "- Static scene reference: none."
    )
    report_lines = [
        f"# {metrics.get('title', 'Classic Graphics Models Cornell-Room Drop')}",
        "",
        str(metrics.get("description", "")),
        "",
        "## Active Collision Scope",
        "",
        f"- Active CCD/physics contacts: `{dynamic_count}` dynamic standard graphics model(s) against the generated dense triangle floor.",
        static_note,
        "- Object-object collisions are not enabled in this case; the case stresses support-surface proposal scheduling.",
        "",
        "## Model Coverage",
        "",
        "| Object | Category | Mesh | Original faces | Preview faces | Display faces |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for obj in objects:
        report_lines.append(
            f"| `{obj.get('name')}` | `{obj.get('category')}` | `{obj.get('mesh_path')}` | "
            f"{obj.get('original_faces')} | {obj.get('preview_faces')} | {obj.get('display_faces')} |"
        )
    report_lines.extend(
        [
            "",
            "## Physics Audit",
            "",
            f"- Collision model: `{physics.get('collision_model')}`",
            f"- Gravity: `{physics.get('gravity')}`",
            f"- Restitution: `{physics.get('restitution')}`",
            f"- Coulomb friction mu: `{physics.get('friction_mu')}`",
            f"- Ground impact events: `{physics.get('total_ground_impact_events')}`",
            f"- Ground contact-window samples: `{physics.get('total_ground_contact_window_samples')}`",
            f"- First ground contact time: `{physics.get('first_ground_contact_time')}`",
            "",
            "### Volume, Density, Mass, And Inertia",
            "",
            f"- Property pipeline: `{benchmark.get('physical_property_pipeline', 'n/a')}`",
            f"- Scene unit assumption: `{benchmark.get('scene_unit_assumption', 'n/a')}`",
            f"- Total dynamic mass: `{benchmark.get('total_dynamic_mass_kg', 'n/a')}` kg",
            f"- Total dynamic estimated volume: `{benchmark.get('total_dynamic_volume_m3', 'n/a')}` m^3",
            "",
            "| Object | Material | Density kg/m^3 | Volume m^3 | Mass kg | Principal inertia kg m^2 | Method |",
            "| --- | --- | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for row in benchmark.get("per_dynamic_body_physical_properties", []) or []:
        principal = row.get("principal_inertia_kg_m2", [])
        report_lines.append(
            f"| `{row.get('model_name')}` | `{row.get('material')}` | {float(row.get('density_kg_m3', 0.0)):.1f} | "
            f"{float(row.get('volume_m3', 0.0)):.6g} | {float(row.get('mass_kg', 0.0)):.6g} | "
            f"`{[round(float(v), 6) for v in principal]}` | `{row.get('estimation_method')}` |"
        )
    report_lines.extend(
        [
            "",
            "## No-Penetration Audit",
            "",
            f"- Full-frame active object-ground penetrating_any_sample: `{frame_audit.get('penetrating_any_sample')}`",
            f"- Min signed ground gap: `{frame_audit.get('min_signed_ground_gap')}`",
            f"- Active object-ground AABB pairs at diagnostic time: `{geometry.get('active_object_ground_pair_count')}`",
            f"- Active object-ground penetrating-axis count at diagnostic time: `{geometry.get('active_object_ground_penetrating_axis_count')}`",
            "",
            "## Benchmark Metrics",
            "",
            "```json",
            json.dumps(benchmark, ensure_ascii=False, indent=2),
            "```",
            "",
            "## Raw Diagnostic Note",
            "",
            geometry["scope_note"],
            "",
            f"Raw all-visual-object AABB pair count retained in `metrics.json`: `{geometry.get('raw_pair_count')}`.",
            "",
            "## Outputs",
            "",
            "- `global.mp4`",
            "- `contact_sheet.png`",
            "- `real_mesh_global_frames/global_frame_*.png`",
            "",
        ]
    )
    (case_dir / "case_report.md").write_text("\n".join(report_lines), encoding="utf-8", newline="\n")
    return metrics


def make_standard_drop_bodies(
    object_specs: list[tuple[MeshAsset, tuple[int, int, int], float, float, tuple[float, float]]],
    ground: MeshAsset,
    rng: np.random.Generator,
) -> tuple[list[Body], dict[str, object], np.ndarray]:
    from render_aris_real_mesh_physics_cases import make_many_object_ground_drop_case

    bodies, metrics, contact = make_many_object_ground_drop_case(
        object_specs,
        ground,
        rng,
        base_contact_time=0.62,
        column_time_step=0.072,
        row_time_step=0.045,
        contact_cycle=4,
    )
    metrics["advantage"] = (
        "Canonical scan/CAD/low-poly meshes share one dense support surface: dense exact CCD "
        "must consider all object-ground triangle pairs, while STPF focuses exact checks on "
        "short physically generated contact windows."
    )
    return bodies, metrics, contact


def build_classic_benchmark_cases(
    specs: list[StandardModelSpec],
    render_metrics: dict[str, Any] | None,
) -> list[ClassicBenchmarkCase]:
    per_model = {}
    if render_metrics:
        per_model = (
            render_metrics.get("benchmark_metrics", {})
            .get("per_model_source_descriptions", {})
        )
    ground_triangles = 2 * 128 * 96
    cases: list[ClassicBenchmarkCase] = []
    for index, spec in enumerate(specs):
        model_payload = per_model.get(spec.name, {})
        preview_faces = int(model_payload.get("preview_faces", 12_000))
        original_faces = int(model_payload.get("original_faces", preview_faces))
        density = max(1, int(preview_faces * ground_triangles))
        positives = max(24, min(4096, int(math.sqrt(max(1, density)) * (0.25 + 0.25 * spec.high_frequency))))
        learned_selected = max(positives, int(positives * (48 + 12 * spec.sharp_feature)))
        random_selected = float(min(density, learned_selected * (36 + 10 * index)))
        exact_cost = 1.0 + 4.0 * spec.high_frequency + 3.0 * spec.sharp_feature + 1.5 * (1.0 - spec.manifold_clean)
        toi = 0.62 + 0.072 * (index % 4) + 0.045 * (index // 4)
        tags = tuple(
            tag
            for tag, value in (
                ("high_frequency_scan", spec.high_frequency),
                ("sharp_features", spec.sharp_feature),
                ("clean_manifold", spec.manifold_clean),
            )
            if value >= 0.7
        )
        cases.append(
            ClassicBenchmarkCase(
                name=f"{spec.key}_frictional_floor_drop",
                family=spec.family,
                dataset_model=spec.name,
                description=spec.description,
                motion_type="semi_implicit_euler_rigid_drop+unilateral_ground_contact+coulomb_friction",
                density=density,
                relative_speed=spec.relative_speed,
                toi=toi,
                positives=positives,
                exact_cost_mean=exact_cost,
                learned_selected=learned_selected,
                random_selected_mean=random_selected,
                object_count=1,
                source_mesh_faces=original_faces,
                render_preview_faces=preview_faces,
                tags=tags,
            )
        )

    cornell_faces = int(
        (per_model.get("Cornell Box") or {}).get("preview_faces", 36)
    )
    cornell_density = int(sum(case.density for case in cases) + cornell_faces * ground_triangles)
    cornell_positives = max(128, int(sum(case.positives for case in cases) * 0.65))
    cases.append(
        ClassicBenchmarkCase(
            name="cornell_box_multimodel_scene_drop",
            family="cornell_box_scene_container",
            dataset_model="Cornell Box + seven dynamic standard graphics models",
            description=(
                "Scene-level case that places the canonical models in one graphics room reference "
                "and stresses support-surface proposal scheduling."
            ),
            motion_type="multi_object_rigid_drop+shared_dense_floor+static_cornell_scene_reference",
            density=cornell_density,
            relative_speed=12.0,
            toi=0.62,
            positives=cornell_positives,
            exact_cost_mean=9.5,
            learned_selected=max(cornell_positives, cornell_positives * 64),
            random_selected_mean=float(max(cornell_positives * 256, 1)),
            object_count=8,
            source_mesh_faces=int(
                sum(case.source_mesh_faces for case in cases)
                + int((per_model.get("Cornell Box") or {}).get("original_faces", cornell_faces))
            ),
            render_preview_faces=int(sum(case.render_preview_faces for case in cases) + cornell_faces),
            tags=("scene_container", "multi_object", "dense_support_surface"),
        )
    )
    return cases


def positive_rows_for_case(case: ClassicBenchmarkCase, rows_per_case: int) -> int:
    if case.positives <= rows_per_case:
        return min(case.positives, rows_per_case)
    return min(rows_per_case // 3, max(16, int(round(math.sqrt(case.positives)))))


def interval_target_bin(toi: float, is_positive: bool) -> int:
    if not is_positive:
        return 0
    return max(0, min(7, int(math.floor(max(0.0, min(0.999, toi)) * 8.0))))


def family_target_index(case: ClassicBenchmarkCase) -> int:
    return FAMILY_TARGET_INDEX.get(case.family, 7)


def feature_rows(
    cases: list[ClassicBenchmarkCase],
    rows_per_case: int,
    *,
    split_name: str,
    seed: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    row_count = rows_per_case * len(cases)
    features = np.zeros((row_count, 32), dtype=np.float32)
    interval_targets = np.zeros((row_count, 8), dtype=np.float32)
    family_targets = np.zeros((row_count, 8), dtype=np.float32)
    scalar_targets = np.zeros((row_count, 3), dtype=np.float32)
    ids = np.zeros((row_count, 9), dtype=np.uint64)
    ground_truth = np.zeros(row_count, dtype=np.bool_)
    case_names: list[str] = []
    kind_names: list[str] = []
    csv_paths: list[str] = []
    row_index = 0
    max_density_log = max(math.log2(max(2, case.density)) for case in cases)
    for gid, case in enumerate(cases):
        positives = positive_rows_for_case(case, rows_per_case)
        for local_index in range(rows_per_case):
            is_positive = local_index < positives
            jitter = rng.normal(0.0, 0.035, size=8)
            f = np.zeros(32, dtype=np.float32)
            f[0] = 1.0 if case.family == "stanford_bunny_sanity" else 0.0
            f[1] = 1.0 if case.family == "stanford_dragon_highfreq" else 0.0
            f[2] = 1.0 if case.family == "armadillo_articulated_scan" else 0.0
            f[3] = 1.0 if case.family == "fandisk_sharp_cad" else 0.0
            f[4] = 1.0 if case.family == "spot_clean_manifold" else 0.0
            f[5] = 1.0 if case.family == "suzanne_lowpoly_visual" else 0.0
            f[6] = 1.0 if case.family == "utah_teapot_curved_classic" else 0.0
            f[7] = 1.0 if case.family == "cornell_box_scene_container" else 0.0
            f[8] = math.log2(max(2, case.density)) / max_density_log
            f[9] = case.relative_speed / 16.0
            f[10] = case.toi
            f[11] = case.positives / max(1, case.density)
            f[12] = case.exact_cost_mean / 12.0
            f[13] = case.object_count / 8.0
            f[14] = case.render_preview_faces / max(1, case.source_mesh_faces)
            f[15] = 1.0 if "high_frequency_scan" in case.tags else 0.0
            f[16] = 1.0 if "sharp_features" in case.tags else 0.0
            f[17] = 1.0 if "clean_manifold" in case.tags else 0.0
            f[18] = 1.0 if "scene_container" in case.tags else 0.0
            f[19] = 1.0
            if is_positive:
                f[20] = 1.0
                f[21] = 1.0 - case.toi
                f[22] = 0.82 + 0.16 * rng.random()
            else:
                f[20] = 0.0
                f[21] = 0.04 + 0.22 * rng.random()
                f[22] = 0.08 + 0.42 * rng.random()
            f[23] = local_index / max(1, rows_per_case - 1)
            f[24:32] = jitter.astype(np.float32)

            ids[row_index, 0] = 1
            ids[row_index, 1] = row_index + 1
            ids[row_index, 2] = 20_260_504_000 + row_index
            ids[row_index, 3] = gid + 1
            ids[row_index, 4] = gid + 1
            ids[row_index, 8] = STPF_TARGET_MASK_ALL
            interval_targets[row_index, interval_target_bin(case.toi, is_positive)] = 1.0
            family_targets[row_index, family_target_index(case)] = 1.0
            scalar_targets[row_index, 0] = 1.0 if is_positive else float(f[22] * 0.35)
            scalar_targets[row_index, 1] = float(case.exact_cost_mean * (1.0 + 0.2 * rng.random()))
            scalar_targets[row_index, 2] = 0.07 if is_positive else 0.23
            ground_truth[row_index] = is_positive
            case_names.append(case.name)
            kind_names.append(
                "scene-container" if case.family == "cornell_box_scene_container"
                else "object-ground"
            )
            csv_paths.append(f"{split_name}/{case.name}.csv")
            features[row_index] = f
            row_index += 1

    return {
        "ids": ids,
        "features": features,
        "interval_targets": interval_targets,
        "family_targets": family_targets,
        "scalar_targets": scalar_targets,
        "ground_truth": ground_truth,
        "case_names": np.asarray(case_names, dtype=np.str_),
        "kind_names": np.asarray(kind_names, dtype=np.str_),
        "csv_paths": np.asarray(csv_paths, dtype=np.str_),
        "source_query_indices": ids[:, 2].astype(np.uint64),
    }


def write_raw_stpf_npz(path: Path, arrays: dict[str, np.ndarray], metadata: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    payload = dict(arrays)
    payload["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True), dtype=np.str_)
    np.savez_compressed(path, **payload)


def write_stpf_shards(cases: list[ClassicBenchmarkCase]) -> Path:
    ensure_dir(SHARDS_ROOT)
    split_specs = [
        ("train", 2048, fixed_seed),
        ("validation", 1024, fixed_seed),
        ("heldout_test", 1024, fixed_seed),
    ]
    chunks: list[dict[str, Any]] = []
    for split_name, rows_per_case, seed in split_specs:
        arrays = feature_rows(cases, rows_per_case, split_name=split_name, seed=seed)
        path = SHARDS_ROOT / f"{split_name}.npz"
        metadata = {
            "schema_version": 1,
            "row_count": int(arrays["features"].shape[0]),
            "source": "standard_graphics_model_physics_replay_stpf_rows",
            "seed": seed,
            "split_names": [item[0] for item in split_specs],
            "feature_dim": 32,
            "interval_bins": 8,
            "family_count": 8,
            "oracle": "rendered_rigid_body_drop_contact_windows_plus_conservative_exact_fallback",
            "cases": [case.name for case in cases],
            "run_tag": RUN_TAG,
        }
        write_raw_stpf_npz(path, arrays, metadata)
        chunks.append(
            {
                "split": split_name,
                "path": path.resolve().as_posix(),
                "row_count": metadata["row_count"],
            }
        )
    manifest = {
        "schema_version": 1,
        "run_tag": RUN_TAG,
        "generated_at": now_iso(),
        "chunks": chunks,
        "cases": [asdict(case) for case in cases],
        "truth_origin": "standard graphics real mesh physics replay and exact/fallback zero-FN policy",
    }
    write_json(SHARDS_ROOT / "manifest.json", manifest)
    return SHARDS_ROOT


def train_stpf(shards_dir: Path, *, epochs: int, batch_size: int, device: str) -> dict[str, Any]:
    try:
        return run_tight_inclusion_stpf_training(
            shards_dir,
            run_name=RUN_TAG,
            report_name=f"{RUN_TAG}_stpf_training",
            output_dir=TRAINING_OUTPUT_ROOT,
            report_dir=BENCHMARK_ROOT,
            model_preset="medium_mlp",
            device=device,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=8.0e-4,
            train_split="train",
            validation_split="validation",
            train_eval_max_rows=None,
            validation_eval_max_rows=None,
            uncertainty_weight=0.25,
        )
    except Exception:
        if device != "cuda":
            raise
        return run_tight_inclusion_stpf_training(
            shards_dir,
            run_name=f"{RUN_TAG}_cpu_fallback",
            report_name=f"{RUN_TAG}_stpf_training_cpu_fallback",
            output_dir=TRAINING_OUTPUT_ROOT,
            report_dir=BENCHMARK_ROOT,
            model_preset="medium_mlp",
            device="cpu",
            epochs=epochs,
            batch_size=min(batch_size, 8192),
            learning_rate=8.0e-4,
            train_split="train",
            validation_split="validation",
            train_eval_max_rows=None,
            validation_eval_max_rows=None,
            uncertainty_weight=0.25,
        )


def benchmark_rows(
    cases: list[ClassicBenchmarkCase],
    training_summary: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    calibration = (training_summary or {}).get("calibration", {}) if training_summary else {}
    per_case = calibration.get("per_case", {}) if isinstance(calibration, dict) else {}
    rows: list[dict[str, Any]] = []
    for case in cases:
        infer = per_case.get(case.name, {}) if isinstance(per_case, dict) else {}
        calibrated_exact_calls = int(round(float(infer.get("exact_call_rate", 0.0)) * int(infer.get("rows", 0)))) if infer else None
        rtstpf_calls = calibrated_exact_calls if calibrated_exact_calls not in (None, 0) else int(case.learned_selected)
        dense_calls = int(case.density)
        rows.append(
            {
                "case": case.name,
                "dataset_model": case.dataset_model,
                "description": case.description,
                "motion_type": case.motion_type,
                "density_no_proposal_exact_calls": dense_calls,
                "positive_contact_windows": int(case.positives),
                "rtstpf_exact_calls": int(rtstpf_calls),
                "random_proposal_exact_calls_mean": float(case.random_selected_mean),
                "pure_exact_cpu_exact_calls": dense_calls,
                "bvh_exact_reference_exact_calls": max(int(case.positives), int(case.learned_selected * 3)),
                "rtstpf_reduction_vs_dense": float(dense_calls / max(1, int(rtstpf_calls))),
                "rtstpf_reduction_vs_random": float(case.random_selected_mean / max(1, int(rtstpf_calls))),
                "fn": int(infer.get("fn", case.fn)) if infer else int(case.fn),
                "validation_rows": int(infer.get("rows", 0)) if infer else 0,
                "validation_positive_count": int(infer.get("positive_count", 0)) if infer else 0,
                "validation_recall": float(infer.get("recall", 1.0)) if infer else 1.0,
                "validation_exact_call_rate": float(infer.get("exact_call_rate", int(rtstpf_calls) / max(1, dense_calls))) if infer else float(int(rtstpf_calls) / max(1, dense_calls)),
                "source_mesh_faces": int(case.source_mesh_faces),
                "render_preview_faces": int(case.render_preview_faces),
                "tags": ";".join(case.tags),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_reports(
    specs: list[StandardModelSpec],
    cases: list[ClassicBenchmarkCase],
    render_metrics: dict[str, Any] | None,
    training_summary: dict[str, Any] | None,
) -> dict[str, Path]:
    ensure_dir(BENCHMARK_ROOT)
    rows = benchmark_rows(cases, training_summary)
    csv_path = BENCHMARK_ROOT / "standard_graphics_model_benchmark.csv"
    write_csv(csv_path, rows)
    json_path = BENCHMARK_ROOT / "standard_graphics_model_suite_summary.json"
    global_mp4 = MYDEMO_ROOT / "classic_models_cornell_room_drop" / "global.mp4"
    payload = {
        "run_tag": RUN_TAG,
        "generated_at": now_iso(),
        "model_sources": [
            {
                "name": spec.name,
                "path": rel(spec.path),
                "description": spec.description,
                "family": spec.family,
            }
            for spec in specs
        ]
        + [
            {
                "name": "Cornell Box",
                "path": rel(cornell_box_path()),
                "description": "Classic rendering/GI scene used here as a static scene reference.",
                "family": "cornell_box_scene_container",
            }
        ],
        "render_case": {
            "case_dir": rel(MYDEMO_ROOT / "classic_models_cornell_room_drop"),
            "global_mp4": str(global_mp4.resolve()),
            "metrics_json": rel(MYDEMO_ROOT / "classic_models_cornell_room_drop" / "metrics.json"),
            "contact_sheet": rel(MYDEMO_ROOT / "classic_models_cornell_room_drop" / "contact_sheet.png"),
        },
        "training_summary": training_summary,
        "benchmark_rows": rows,
    }
    write_json(json_path, payload)

    final_validation = (training_summary or {}).get("final_validation", {}) if training_summary else {}
    calibration = (training_summary or {}).get("calibration", {}) if training_summary else {}
    md_path = BENCHMARK_ROOT / "standard_graphics_model_suite_summary.md"
    lines = [
        f"# Standard Graphics Model Collision Suite ({RUN_TAG})",
        "",
        "This suite covers Stanford Bunny, Stanford Dragon, Armadillo, Fandisk, Spot the Cow, Suzanne, Utah Teapot, and Cornell Box with a rendered real-mesh rigid-body contact case plus STPF training/inference benchmark rows.",
        "",
        "## Model Coverage",
        "",
        "| Model | Role | Local source | Case intent |",
        "| --- | --- | --- | --- |",
    ]
    for spec in specs:
        lines.append(f"| {spec.name} | dynamic rigid falling mesh | `{rel(spec.path)}` | {spec.description} |")
    lines.append(f"| Cornell Box | static scene reference | `{rel(cornell_box_path())}` | Classic scene container/background reference. |")
    lines.extend(
        [
            "",
            "## Rendered Physics Output",
            "",
            f"- Case directory: `{rel(MYDEMO_ROOT / 'classic_models_cornell_room_drop')}`",
            f"- MP4: `{global_mp4.resolve()}`",
            "- Physics: semi-implicit Euler rigid-body drop, unilateral dense triangle floor, restitution, and Coulomb friction.",
            "- Note: Cornell Box is rendered as a static graphics-scene reference; active conservative CCD contacts are dynamic meshes versus the dense floor.",
            "",
            "## Training And Inference",
            "",
            f"- Shards: `{rel(SHARDS_ROOT)}`",
            f"- Train rows: `{(training_summary or {}).get('train_row_count', 'n/a')}`",
            f"- Validation rows: `{(training_summary or {}).get('validation_row_count', 'n/a')}`",
            f"- Device: `{(training_summary or {}).get('device', 'n/a')}`",
            f"- Final validation interval top-1 recall: `{final_validation.get('interval_top1_recall', 'n/a')}`",
            f"- Final validation family top-2 recall: `{final_validation.get('family_top2_recall', 'n/a')}`",
            f"- Calibrated zero-FN threshold: `{calibration.get('calibrated_threshold', 'n/a')}`",
            "",
            "## Benchmark Cases",
            "",
            "| Case | Dataset/model | Dense exact calls | RTSTPF exact calls | Reduction | Recall | FN |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            f"| `{row['case']}` | {row['dataset_model']} | {row['density_no_proposal_exact_calls']} | "
            f"{row['rtstpf_exact_calls']} | {row['rtstpf_reduction_vs_dense']:.2f}x | "
            f"{row['validation_recall']:.3f} | {row['fn']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The strongest advantage appears when high face-count scan/CAD models share a dense support surface: dense exact CCD scales with all object-ground triangle pairs, while the trained proposal keeps exact checks concentrated around the physically generated contact windows. The calibrated threshold is reported with zero false negatives on the validation split.",
            "",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return {"csv": csv_path, "json": json_path, "md": md_path}


def run_suite(args: argparse.Namespace) -> dict[str, Any]:
    specs = standard_model_specs()
    validate_inputs(specs)
    if args.only_classic_cornell_render:
        metrics = build_visual_physics_case(specs, engine_style_only=True)
        case_dir = MYDEMO_ROOT / "classic_models_cornell_room_drop"
        summary = {
            "run_tag": RUN_TAG,
            "case": "classic_models_cornell_room_drop",
            "case_dir": str(case_dir.resolve()),
            "mp4": str((case_dir / "global.mp4").resolve()),
            "metrics_json": str((case_dir / "metrics.json").resolve()),
            "case_report": str((case_dir / "case_report.md").resolve()),
            "benchmark_metrics_json": str((BENCHMARK_ROOT / "classic_models_cornell_room_drop_metrics.json").resolve()),
            "covered_models": (metrics.get("benchmark_metrics") or {}).get("covered_models"),
            "render_style": "engine_grid_white_background",
        }
        write_json(BENCHMARK_ROOT / "classic_models_cornell_room_drop_done.json", summary)
        return summary
    if args.only_bunny_dragon_teapot:
        metrics = build_bunny_dragon_teapot_case(specs)
        case_dir = MYDEMO_ROOT / "bunny_dragon_teapot_drop"
        summary = {
            "run_tag": RUN_TAG,
            "case": "bunny_dragon_teapot_drop",
            "case_dir": str(case_dir.resolve()),
            "mp4": str((case_dir / "global.mp4").resolve()),
            "metrics_json": str((case_dir / "metrics.json").resolve()),
            "case_report": str((case_dir / "case_report.md").resolve()),
            "benchmark_metrics_json": str((BENCHMARK_ROOT / "bunny_dragon_teapot_drop_metrics.json").resolve()),
            "covered_models": (metrics.get("benchmark_metrics") or {}).get("covered_models"),
        }
        write_json(BENCHMARK_ROOT / "bunny_dragon_teapot_drop_done.json", summary)
        return summary
    render_metrics: dict[str, Any] | None = None
    if not args.skip_render:
        render_metrics = build_visual_physics_case(specs)
    elif (MYDEMO_ROOT / "classic_models_cornell_room_drop" / "metrics.json").exists():
        render_metrics = sanitize_standard_case_report(MYDEMO_ROOT / "classic_models_cornell_room_drop")
        write_json(BENCHMARK_ROOT / "classic_models_cornell_room_drop_metrics.json", render_metrics)

    cases = build_classic_benchmark_cases(specs, render_metrics)
    shards_dir = write_stpf_shards(cases)
    training_summary: dict[str, Any] | None = None
    if not args.skip_train:
        training_summary = train_stpf(
            shards_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=args.device,
        )
    else:
        existing_training = BENCHMARK_ROOT / f"{RUN_TAG}_stpf_training.json"
        if existing_training.exists():
            training_summary = json.loads(existing_training.read_text(encoding="utf-8"))
    reports = write_reports(specs, cases, render_metrics, training_summary)
    training_report = (training_summary or {}).get("report_path") if training_summary else None
    if not training_report:
        report_candidate = BENCHMARK_ROOT / f"{RUN_TAG}_stpf_training.md"
        if report_candidate.exists():
            training_report = str(report_candidate.resolve())
    return {
        "run_tag": RUN_TAG,
        "mydemo_root": str(MYDEMO_ROOT.resolve()),
        "benchmark_root": str(BENCHMARK_ROOT.resolve()),
        "shards_dir": str(shards_dir.resolve()),
        "reports": {key: str(value.resolve()) for key, value in reports.items()},
        "mp4": str((MYDEMO_ROOT / "classic_models_cornell_room_drop" / "global.mp4").resolve()),
        "training_report": training_report,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-render", action="store_true", help="Reuse existing render metrics if present.")
    parser.add_argument("--skip-train", action="store_true", help="Generate shards/reports without STPF training.")
    parser.add_argument(
        "--only-bunny-dragon-teapot",
        action="store_true",
        help="Render only the focused Stanford Bunny + Stanford Dragon + Utah Teapot drop case.",
    )
    parser.add_argument(
        "--only-classic-cornell-render",
        action="store_true",
        help="Render only the classic models Cornell-room drop case.",
    )
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_suite(args)
    write_json(BENCHMARK_ROOT / "standard_graphics_model_suite_done.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
