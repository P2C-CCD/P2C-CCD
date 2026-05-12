from __future__ import annotations

from dataclasses import asdict, dataclass, field
import csv
import json
import math
import os
import random
import zipfile
from itertools import combinations, product
from pathlib import Path
from typing import Iterable, Sequence

from p2cccd.contracts import ProxyType
from p2cccd.data import GeneratedDataset, default_metadata, proposal_row_from_oracle_trace, write_npz_shard
from p2cccd.data.oracle import evaluate_swept_sphere_oracle
from p2cccd.data.response import proxy_mass_from_radius
from p2cccd.data.samplers import MotionDiscPairSample, PairFamily
from p2cccd.proposal.stpf_model import (
    STPFModelPreset,
    build_stpf_model,
    build_stpf_model_from_checkpoint_payload,
)
from p2cccd.proposal.training import STPFTrainingConfig
from p2cccd.proposal.training_runner import STPFTrainingRunConfig, run_stpf_training

from .common_modeling_ort_walltime_benchmark import run_common_modeling_ort_walltime_benchmark
from .high_density_mesh_training_benchmark import (
    MeshDensityAsset,
    MeshDensityPair,
    _pair_score,
    _scale_workload_costs,
    _stable_u32,
    _subset_workload,
)
from .trained_stpf_high_density import (
    HighDensityMethodMetrics,
    HighDensitySTPFConfig,
    benchmark_no_proposal_on_high_density_workload,
    benchmark_stpf_on_high_density_workload,
    build_high_density_stpf_workload,
    workload_to_shard_dataset,
)


RUN_NAME = "shapenet_ood_dense_cases_run_id"
EXTRACTED_DATASET_NAME = "selected_ood_dense_run_id"
DEFAULT_EXTRACTED_ROOT = Path("src/datasets/shapenet_core_v2") / EXTRACTED_DATASET_NAME


def _default_shapenet_source_root() -> Path:
    return Path(os.environ.get("P2CCCD_SHAPENET_ROOT", "src/datasets/shapenetcore"))


SHAPENET_CATEGORIES: dict[str, str] = {
    "02958343": "car",
    "02691156": "airplane",
    "04530566": "watercraft",
    "03001627": "chair",
    "04379243": "table",
    "04256520": "sofa",
    "03636649": "lamp",
    "04090263": "rifle",
    "02924116": "bus",
    "04468005": "train",
    "03467517": "guitar",
    "03691459": "loudspeaker",
}

OPTIONAL_CATEGORIES: dict[str, str] = {
    "04099429": "rocket",
}


def _default_high_density() -> HighDensitySTPFConfig:
    return HighDensitySTPFConfig(
        slab_count=16,
        patches_per_object=12,
        representative_attempt_limit=3,
        uncertainty_fallback_threshold=0.75,
        narrow_interval_min_cost_scale=0.18,
        interval_miss_penalty_scale=0.22,
        full_exact_cost_scale=1.0,
    )


def _default_training() -> STPFTrainingConfig:
    return STPFTrainingConfig(
        epochs=8,
        batch_size=65536,
        learning_rate=7.0e-4,
        seed=424242,
        device="cuda",
        validation_fraction=0.0,
        model_preset=STPFModelPreset.MEDIUM_MLP,
    )


@dataclass(frozen=True, slots=True)
class ShapeNetOODDenseConfig:
    shapenet_root: Path = DEFAULT_EXTRACTED_ROOT
    run_name: str = RUN_NAME
    asset_limit_per_category: int = 16
    pair_limit_per_case: int = 36
    samples_per_pair: int = 2
    train_fraction: float = 0.75
    include_optional_rocket: bool = True
    seed: int = 424242
    high_density: HighDensitySTPFConfig = field(default_factory=_default_high_density)
    training: STPFTrainingConfig = field(default_factory=_default_training)
    cross_dataset_checkpoint: Path = Path(
        "src/outputs/stpf_training/rtstpf_advantage_cases_v4_large_training_run_id/model_state.pt"
    )
    shard_root: Path = Path("src/datasets/training/shapenet_ood_dense_cases/shards")
    benchmark_output_dir: Path = Path("src/benchmark")
    training_output_dir: Path = Path("src/outputs/stpf_training")


@dataclass(frozen=True, slots=True)
class ShapeNetExtractionConfig:
    source_root: Path = field(default_factory=_default_shapenet_source_root)
    output_root: Path = DEFAULT_EXTRACTED_ROOT
    asset_limit_per_category: int = 16
    include_optional_rocket: bool = True


@dataclass(frozen=True, slots=True)
class ShapeNetAssetRecord:
    synset: str
    category: str
    model_id: str
    zip_path: str
    obj_entry: str
    obj_bytes: int
    vertices: int
    faces: int
    bounds_min: tuple[float, float, float]
    bounds_max: tuple[float, float, float]
    diagonal: float
    solid_binvox_bytes: int
    surface_binvox_bytes: int
    occupancy_proxy: float


@dataclass(frozen=True, slots=True)
class ShapeNetCaseSummary:
    case_name: str
    case_type: str
    role: str
    query_count: int
    candidate_count: int
    positive_queries: int
    exact_work_reduction: float | None = None
    exact_call_reduction: float | None = None
    cross_dataset_exact_work_reduction: float | None = None
    trained_exact_calls: int | None = None
    cross_dataset_exact_calls: int | None = None
    fn_count: int | None = None
    cross_dataset_fn_count: int | None = None


def _density(cfg: HighDensitySTPFConfig) -> int:
    return int(cfg.slab_count * cfg.patches_per_object * cfg.patches_per_object)


def _iter_model_obj_entries(archive: zipfile.ZipFile, synset: str) -> Iterable[zipfile.ZipInfo]:
    suffix = "/models/model_normalized.obj"
    for info in archive.infolist():
        if info.filename.startswith(f"{synset}/") and info.filename.endswith(suffix):
            yield info


def _parse_obj_stats(archive: zipfile.ZipFile, info: zipfile.ZipInfo) -> tuple[int, int, tuple[float, float, float], tuple[float, float, float], float]:
    vertices = 0
    faces = 0
    min_v = [float("inf"), float("inf"), float("inf")]
    max_v = [float("-inf"), float("-inf"), float("-inf")]
    with archive.open(info, "r") as raw:
        for raw_line in raw:
            if raw_line.startswith(b"v "):
                parts = raw_line.decode("utf-8", errors="ignore").strip().split()
                if len(parts) >= 4:
                    try:
                        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    except ValueError:
                        continue
                    vertices += 1
                    min_v[0] = min(min_v[0], x)
                    min_v[1] = min(min_v[1], y)
                    min_v[2] = min(min_v[2], z)
                    max_v[0] = max(max_v[0], x)
                    max_v[1] = max(max_v[1], y)
                    max_v[2] = max(max_v[2], z)
            elif raw_line.startswith(b"f "):
                faces += 1
    if vertices == 0:
        min_v = [0.0, 0.0, 0.0]
        max_v = [1.0, 1.0, 1.0]
    diagonal = math.sqrt(sum((max_v[idx] - min_v[idx]) ** 2 for idx in range(3)))
    return vertices, faces, tuple(min_v), tuple(max_v), float(diagonal)


def _parse_obj_stats_from_path(path: Path) -> tuple[int, int, tuple[float, float, float], tuple[float, float, float], float]:
    vertices = 0
    faces = 0
    min_v = [float("inf"), float("inf"), float("inf")]
    max_v = [float("-inf"), float("-inf"), float("-inf")]
    with path.open("rb") as handle:
        for raw_line in handle:
            if raw_line.startswith(b"v "):
                parts = raw_line.decode("utf-8", errors="ignore").strip().split()
                if len(parts) >= 4:
                    try:
                        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    except ValueError:
                        continue
                    vertices += 1
                    min_v[0] = min(min_v[0], x)
                    min_v[1] = min(min_v[1], y)
                    min_v[2] = min(min_v[2], z)
                    max_v[0] = max(max_v[0], x)
                    max_v[1] = max(max_v[1], y)
                    max_v[2] = max(max_v[2], z)
            elif raw_line.startswith(b"f "):
                faces += 1
    if vertices == 0:
        min_v = [0.0, 0.0, 0.0]
        max_v = [1.0, 1.0, 1.0]
    diagonal = math.sqrt(sum((max_v[idx] - min_v[idx]) ** 2 for idx in range(3)))
    return vertices, faces, tuple(min_v), tuple(max_v), float(diagonal)


def _binvox_bytes_by_model(archive: zipfile.ZipFile, synset: str) -> dict[str, tuple[int, int]]:
    result: dict[str, tuple[int, int]] = {}
    for info in archive.infolist():
        parts = info.filename.split("/")
        if len(parts) < 4 or parts[0] != synset or parts[2] != "models":
            continue
        model_id = parts[1]
        solid, surface = result.get(model_id, (0, 0))
        if parts[-1] == "model_normalized.solid.binvox":
            solid = int(info.file_size)
        elif parts[-1] == "model_normalized.surface.binvox":
            surface = int(info.file_size)
        result[model_id] = (solid, surface)
    return result


def _scan_category_assets(
    root: Path,
    *,
    synset: str,
    category: str,
    limit: int,
) -> tuple[ShapeNetAssetRecord, ...]:
    zip_path = root / f"{synset}.zip"
    extracted_synset = root / synset
    if extracted_synset.exists():
        records: list[ShapeNetAssetRecord] = []
        obj_paths = sorted(
            extracted_synset.glob("*/models/model_normalized.obj"),
            key=lambda path: path.stat().st_size,
            reverse=True,
        )[:limit]
        for obj_path in obj_paths:
            model_id = obj_path.parent.parent.name
            vertices, faces, bounds_min, bounds_max, diagonal = _parse_obj_stats_from_path(obj_path)
            solid_path = obj_path.with_name("model_normalized.solid.binvox")
            surface_path = obj_path.with_name("model_normalized.surface.binvox")
            solid_bytes = int(solid_path.stat().st_size) if solid_path.exists() else 0
            surface_bytes = int(surface_path.stat().st_size) if surface_path.exists() else 0
            occupancy_proxy = math.log1p(float(solid_bytes + surface_bytes)) / math.log1p(256.0 * 1024.0)
            records.append(
                ShapeNetAssetRecord(
                    synset=synset,
                    category=category,
                    model_id=model_id,
                    zip_path=str(root),
                    obj_entry=str(obj_path.relative_to(root)).replace("\\", "/"),
                    obj_bytes=int(obj_path.stat().st_size),
                    vertices=int(vertices),
                    faces=int(faces),
                    bounds_min=bounds_min,
                    bounds_max=bounds_max,
                    diagonal=diagonal,
                    solid_binvox_bytes=solid_bytes,
                    surface_binvox_bytes=surface_bytes,
                    occupancy_proxy=float(min(2.0, occupancy_proxy)),
                )
            )
        records.sort(key=lambda row: (-row.faces, -row.vertices, -row.obj_bytes, row.model_id))
        return tuple(records)
    if not zip_path.exists():
        raise FileNotFoundError(f"ShapeNet category zip or extracted directory not found: {zip_path} / {extracted_synset}")
    with zipfile.ZipFile(zip_path) as archive:
        obj_entries = sorted(_iter_model_obj_entries(archive, synset), key=lambda item: item.file_size, reverse=True)[:limit]
        binvox = _binvox_bytes_by_model(archive, synset)
        records: list[ShapeNetAssetRecord] = []
        for info in obj_entries:
            parts = info.filename.split("/")
            model_id = parts[1] if len(parts) > 1 else info.filename
            vertices, faces, bounds_min, bounds_max, diagonal = _parse_obj_stats(archive, info)
            solid_bytes, surface_bytes = binvox.get(model_id, (0, 0))
            occupancy_proxy = math.log1p(float(solid_bytes + surface_bytes)) / math.log1p(256.0 * 1024.0)
            records.append(
                ShapeNetAssetRecord(
                    synset=synset,
                    category=category,
                    model_id=model_id,
                    zip_path=str(zip_path),
                    obj_entry=info.filename,
                    obj_bytes=int(info.file_size),
                    vertices=int(vertices),
                    faces=int(faces),
                    bounds_min=bounds_min,
                    bounds_max=bounds_max,
                    diagonal=diagonal,
                    solid_binvox_bytes=solid_bytes,
                    surface_binvox_bytes=surface_bytes,
                    occupancy_proxy=float(min(2.0, occupancy_proxy)),
                )
            )
    records.sort(key=lambda row: (-row.faces, -row.vertices, -row.obj_bytes, row.model_id))
    return tuple(records)


def _safe_extract_member(
    archive: zipfile.ZipFile,
    member: str,
    *,
    output_root: Path,
) -> Path | None:
    parts = Path(member).parts
    if len(parts) < 4 or parts[0].startswith("..") or any(part in {"", ".", ".."} for part in parts):
        return None
    target = output_root.joinpath(*parts)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 0:
        return target
    with archive.open(member, "r") as src, target.open("wb") as dst:
        while True:
            chunk = src.read(1024 * 1024)
            if not chunk:
                break
            dst.write(chunk)
    return target


def extract_shapenet_core_selection(cfg: ShapeNetExtractionConfig | None = None) -> dict[str, object]:
    config = cfg or ShapeNetExtractionConfig()
    if not config.source_root.exists():
        raise FileNotFoundError(config.source_root)
    config.output_root.mkdir(parents=True, exist_ok=True)
    categories = dict(SHAPENET_CATEGORIES)
    if config.include_optional_rocket and (config.source_root / "04099429.zip").exists():
        categories.update(OPTIONAL_CATEGORIES)
    rows: list[dict[str, object]] = []
    required_suffixes = (
        "model_normalized.obj",
        "model_normalized.mtl",
        "model_normalized.json",
        "model_normalized.solid.binvox",
        "model_normalized.surface.binvox",
    )
    for synset, category in categories.items():
        zip_path = config.source_root / f"{synset}.zip"
        if not zip_path.exists():
            continue
        with zipfile.ZipFile(zip_path) as archive:
            obj_entries = sorted(
                _iter_model_obj_entries(archive, synset),
                key=lambda info: info.file_size,
                reverse=True,
            )[: config.asset_limit_per_category]
            by_name = {info.filename: info for info in archive.infolist()}
            for obj_info in obj_entries:
                parts = obj_info.filename.split("/")
                if len(parts) < 4:
                    continue
                model_id = parts[1]
                extracted_files: list[str] = []
                for suffix in required_suffixes:
                    member = f"{synset}/{model_id}/models/{suffix}"
                    if member not in by_name:
                        continue
                    target = _safe_extract_member(archive, member, output_root=config.output_root)
                    if target is not None:
                        extracted_files.append(target.relative_to(config.output_root).as_posix())
                rows.append(
                    {
                        "synset": synset,
                        "category": category,
                        "model_id": model_id,
                        "source_zip": zip_path.as_posix(),
                        "obj_entry": obj_info.filename,
                        "obj_bytes": int(obj_info.file_size),
                        "extracted_files": extracted_files,
                    }
                )
    manifest = {
        "schema": "shapenet_core_v2_selected_ood_dense.v1",
        "source_root": config.source_root.as_posix(),
        "output_root": config.output_root.as_posix(),
        "asset_limit_per_category": int(config.asset_limit_per_category),
        "category_count": len(categories),
        "asset_count": len(rows),
        "categories": categories,
        "assets": rows,
    }
    (config.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    with (config.output_root / "selected_assets.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["synset", "category", "model_id", "source_zip", "obj_entry", "obj_bytes", "extracted_files"])
        writer.writeheader()
        for row in rows:
            row = dict(row)
            row["extracted_files"] = ";".join(row["extracted_files"])
            writer.writerow(row)
    return manifest


def _to_mesh_asset(record: ShapeNetAssetRecord) -> MeshDensityAsset:
    root_or_zip = Path(record.zip_path)
    asset_path = (root_or_zip / record.obj_entry).as_posix() if root_or_zip.is_dir() else f"zip://{record.zip_path}!{record.obj_entry}"
    return MeshDensityAsset(
        source_name=f"ShapeNet-{record.category}",
        asset_id=f"shapenet-{record.synset}-{record.model_id}",
        asset_path=asset_path,
        face_count=int(record.faces),
        vertex_count=int(record.vertices),
        diagonal=float(record.diagonal),
        bounds_min=record.bounds_min,
        bounds_max=record.bounds_max,
        dirty_score=float(record.occupancy_proxy),
    )


def _pair_cost_scale(asset_a: MeshDensityAsset, asset_b: MeshDensityAsset, *, case_type: str) -> float:
    primitive_scale = math.sqrt(max(1, asset_a.face_count) * max(1, asset_b.face_count)) / 1024.0
    occupancy = 1.0 + 0.10 * max(asset_a.dirty_score, asset_b.dirty_score)
    if case_type == "high_speed":
        motion = 1.40
    elif case_type == "thin_feature":
        motion = 1.25
    elif case_type == "binvox_proxy":
        motion = 1.18
    else:
        motion = 1.0
    return max(1.0, primitive_scale * occupancy * motion)


def _make_pairs(
    case_name: str,
    case_type: str,
    assets_a: Sequence[MeshDensityAsset],
    assets_b: Sequence[MeshDensityAsset] | None = None,
    *,
    limit: int,
) -> tuple[MeshDensityPair, ...]:
    if assets_b is None:
        raw_pairs = combinations(assets_a, 2)
    else:
        raw_pairs = (
            (asset_a, asset_b)
            for asset_a, asset_b in product(assets_a, assets_b)
            if asset_a.asset_path != asset_b.asset_path
        )
    pairs = [
        MeshDensityPair(
            source_name=case_name,
            asset_a=asset_a,
            asset_b=asset_b,
            pair_score=min(1.0, 0.75 * _pair_score(asset_a, asset_b) + 0.25 * max(asset_a.dirty_score, asset_b.dirty_score)),
            cost_scale=_pair_cost_scale(asset_a, asset_b, case_type=case_type),
        )
        for asset_a, asset_b in raw_pairs
    ]
    pairs.sort(key=lambda pair: (-pair.cost_scale, -pair.pair_score, pair.asset_a.asset_id, pair.asset_b.asset_id))
    return tuple(pairs[:limit])


def _radius(asset: MeshDensityAsset, *, case_type: str) -> float:
    face_factor = 1.0 + 0.025 * math.log1p(max(1, asset.face_count))
    base = max(0.035, 0.42 * max(asset.diagonal, 0.1) * face_factor)
    if case_type == "thin_feature":
        return 0.42 * base
    if case_type == "high_speed":
        return 0.78 * base
    return base


def _sample_from_pair(
    pair: MeshDensityPair,
    *,
    sample_id: int,
    variant_index: int,
    case_type: str,
) -> MotionDiscPairSample:
    radius_a = _radius(pair.asset_a, case_type=case_type)
    radius_b = _radius(pair.asset_b, case_type=case_type)
    radius_sum = radius_a + radius_b
    scale = max(0.1, 0.5 * (max(pair.asset_a.diagonal, 0.1) + max(pair.asset_b.diagonal, 0.1)))
    token = _stable_u32(pair.asset_a.asset_id + pair.asset_b.asset_id + str(variant_index) + case_type)
    lateral = ((token % 256) / 255.0 - 0.5) * 0.18 * scale
    z_bias = (((token // 256) % 256) / 255.0 - 0.5) * 0.10 * scale
    direction = 1.0 if (variant_index % 2) == 0 else -1.0

    if case_type == "high_speed":
        split = "shapenet_high_speed_rigid_collision"
        start_gap = radius_sum + (2.8 + 0.25 * pair.pair_score) * scale
        end_gap = -(radius_sum + (0.8 + 0.20 * pair.pair_score) * scale)
        ax1 = 0.95 * scale * direction
        lateral_t1 = lateral + 0.035 * scale * direction
    elif case_type == "thin_feature":
        split_cycle = ("shapenet_thin_feature_near_contact", "shapenet_thin_feature_grazing", "shapenet_thin_feature_slot")
        split = split_cycle[variant_index % len(split_cycle)]
        start_gap = radius_sum + (0.20 + 0.06 * pair.pair_score) * scale
        end_gap = radius_sum - (0.012 + 0.018 * pair.pair_score) * scale
        ax1 = 0.16 * scale * direction
        lateral_t1 = lateral + 0.11 * scale * direction
    elif case_type == "binvox_proxy":
        split = "shapenet_binvox_proxy_occupancy_dense"
        start_gap = radius_sum + (0.32 + 0.07 * pair.pair_score) * scale
        end_gap = radius_sum - (0.028 + 0.02 * pair.pair_score) * scale
        ax1 = 0.22 * scale * direction
        lateral_t1 = lateral + 0.05 * scale * direction
    else:
        split_cycle = (
            "shapenet_ood_dense_easy_negative",
            "shapenet_ood_dense_near_contact",
            "shapenet_ood_dense_grazing",
            "shapenet_ood_dense_multi_contact",
        )
        split = split_cycle[variant_index % len(split_cycle)]
        if split.endswith("easy_negative"):
            start_gap = radius_sum + (1.30 + 0.30 * pair.pair_score) * scale
            end_gap = radius_sum + (0.50 + 0.16 * pair.pair_score) * scale
        elif split.endswith("near_contact"):
            start_gap = radius_sum + (0.24 + 0.06 * pair.pair_score) * scale
            end_gap = radius_sum + (0.030 + 0.016 * pair.pair_score) * scale
        elif split.endswith("grazing"):
            start_gap = radius_sum + (0.15 + 0.05 * pair.pair_score) * scale
            end_gap = radius_sum - (0.010 + 0.018 * pair.pair_score) * scale
        else:
            start_gap = radius_sum + (0.34 + 0.08 * pair.pair_score) * scale
            end_gap = -(radius_sum + (0.16 + 0.05 * pair.pair_score) * scale)
        ax1 = 0.18 * scale * direction
        lateral_t1 = lateral + 0.025 * scale * direction

    return MotionDiscPairSample(
        sample_id=sample_id,
        query_id=90_000_000 + sample_id,
        candidate_id=91_000_000 + sample_id,
        split=split,
        family=PairFamily.MESH_PAIR,
        object_a_id=1_100_000 + (_stable_u32(pair.asset_a.asset_id) % 300_000),
        patch_a_id=1 + (_stable_u32(pair.asset_a.asset_path) % max(1, min(pair.asset_a.face_count, 200_000))),
        object_b_id=1_100_000 + (_stable_u32(pair.asset_b.asset_id) % 300_000),
        patch_b_id=1 + (_stable_u32(pair.asset_b.asset_path) % max(1, min(pair.asset_b.face_count, 200_000))),
        slab_id=variant_index % 16,
        center_a_t0=(0.0, lateral, z_bias),
        center_a_t1=(ax1, lateral_t1, z_bias + 0.015 * scale * direction),
        center_b_t0=(start_gap, lateral - 0.018 * scale, z_bias),
        center_b_t1=(end_gap, lateral + 0.018 * scale, z_bias - 0.015 * scale * direction),
        radius_a=radius_a,
        radius_b=radius_b,
        proxy_type_a=ProxyType.SWEPT_AABB,
        proxy_type_b=ProxyType.SWEPT_AABB,
        hardness=max(0.0, min(1.0, pair.pair_score)),
        ood=True,
        mass_a=proxy_mass_from_radius(radius_a),
        mass_b=proxy_mass_from_radius(radius_b),
        restitution=1.0,
    )


def _dataset_from_samples(samples: Sequence[MotionDiscPairSample]) -> GeneratedDataset:
    sample_list = list(samples)
    traces = [evaluate_swept_sphere_oracle(sample) for sample in sample_list]
    rows = [proposal_row_from_oracle_trace(sample, trace) for sample, trace in zip(sample_list, traces)]
    split_names = tuple(dict.fromkeys(sample.split for sample in sample_list).keys())
    return GeneratedDataset(rows=rows, samples=sample_list, traces=traces, split_names=split_names)


def _load_assets(cfg: ShapeNetOODDenseConfig) -> tuple[dict[str, tuple[MeshDensityAsset, ...]], tuple[ShapeNetAssetRecord, ...]]:
    categories = dict(SHAPENET_CATEGORIES)
    if cfg.include_optional_rocket and (
        (cfg.shapenet_root / "04099429.zip").exists()
        or (cfg.shapenet_root / "04099429").exists()
    ):
        categories.update(OPTIONAL_CATEGORIES)
    assets_by_category: dict[str, tuple[MeshDensityAsset, ...]] = {}
    records: list[ShapeNetAssetRecord] = []
    for synset, category in categories.items():
        category_records = _scan_category_assets(
            cfg.shapenet_root,
            synset=synset,
            category=category,
            limit=cfg.asset_limit_per_category,
        )
        records.extend(category_records)
        assets_by_category[category] = tuple(_to_mesh_asset(record) for record in category_records)
    return assets_by_category, tuple(records)


def _build_case_specs(
    assets_by_category: dict[str, tuple[MeshDensityAsset, ...]],
    *,
    pair_limit: int,
) -> dict[str, tuple[str, tuple[MeshDensityPair, ...]]]:
    cases: dict[str, tuple[str, tuple[MeshDensityPair, ...]]] = {}
    for category, assets in sorted(assets_by_category.items()):
        if len(assets) >= 2:
            case_name = f"S00-{category}-intra-dense"
            pairs = _make_pairs(case_name, "dense_contact", assets, limit=pair_limit)
            if len(pairs) >= 2:
                cases[case_name] = ("dense_contact", pairs)

    cross_specs = [
        ("S10-car-airplane-ood-cross", "dense_contact", "car", "airplane"),
        ("S11-car-watercraft-ood-cross", "dense_contact", "car", "watercraft"),
        ("S12-airplane-watercraft-ood-cross", "dense_contact", "airplane", "watercraft"),
        ("S13-chair-table-contact", "thin_feature", "chair", "table"),
        ("S14-chair-sofa-soft-contact", "dense_contact", "chair", "sofa"),
        ("S15-table-lamp-thin-contact", "thin_feature", "table", "lamp"),
        ("S16-lamp-rifle-thin-rotation", "thin_feature", "lamp", "rifle"),
        ("S17-rifle-guitar-thin-cross", "thin_feature", "rifle", "guitar"),
        ("S18-bus-train-long-rigid", "high_speed", "bus", "train"),
        ("S19-loudspeaker-chair-cavity-contact", "binvox_proxy", "loudspeaker", "chair"),
        ("S20-guitar-lamp-thin-feature", "thin_feature", "guitar", "lamp"),
        ("S21-car-bus-high-speed", "high_speed", "car", "bus"),
        ("S22-train-watercraft-large-rigid", "high_speed", "train", "watercraft"),
        ("S23-sofa-table-wide-contact", "dense_contact", "sofa", "table"),
        ("S24-watercraft-loudspeaker-cavity", "binvox_proxy", "watercraft", "loudspeaker"),
        ("S25-airplane-rifle-fast-thin", "high_speed", "airplane", "rifle"),
        ("S26-car-train-high-speed", "high_speed", "car", "train"),
        ("S27-chair-lamp-slot", "thin_feature", "chair", "lamp"),
        ("S28-table-guitar-thin-grazing", "thin_feature", "table", "guitar"),
        ("S29-sofa-loudspeaker-cavity", "binvox_proxy", "sofa", "loudspeaker"),
    ]
    if "rocket" in assets_by_category:
        cross_specs.extend(
            [
                ("S30-rifle-rocket-fast-thin", "high_speed", "rifle", "rocket"),
                ("S31-airplane-rocket-high-speed", "high_speed", "airplane", "rocket"),
                ("S32-watercraft-rocket-cross", "high_speed", "watercraft", "rocket"),
            ]
        )

    for case_name, case_type, left, right in cross_specs:
        assets_a = assets_by_category.get(left, ())
        assets_b = assets_by_category.get(right, ())
        if len(assets_a) >= 1 and len(assets_b) >= 1:
            pairs = _make_pairs(case_name, case_type, assets_a, assets_b, limit=pair_limit)
            if len(pairs) >= 2:
                cases[case_name] = (case_type, pairs)
    return cases


def _split_pairs(
    pairs: Sequence[MeshDensityPair],
    *,
    train_fraction: float,
    seed: int,
) -> tuple[tuple[MeshDensityPair, ...], tuple[MeshDensityPair, ...]]:
    items = list(pairs)
    random.Random(seed).shuffle(items)
    if len(items) < 2:
        raise ValueError("each ShapeNet case needs at least two pairs")
    train_count = max(1, min(len(items) - 1, int(round(len(items) * train_fraction))))
    return tuple(items[:train_count]), tuple(items[train_count:])


def _build_datasets(
    cases: dict[str, tuple[str, tuple[MeshDensityPair, ...]]],
    *,
    cfg: ShapeNetOODDenseConfig,
) -> tuple[
    GeneratedDataset,
    GeneratedDataset,
    dict[int, str],
    dict[int, str],
    dict[int, str],
    dict[int, str],
    dict[int, float],
    dict[int, float],
    dict[str, tuple[int, int]],
]:
    train_samples: list[MotionDiscPairSample] = []
    eval_samples: list[MotionDiscPairSample] = []
    train_case_by_query: dict[int, str] = {}
    eval_case_by_query: dict[int, str] = {}
    train_type_by_query: dict[int, str] = {}
    eval_type_by_query: dict[int, str] = {}
    train_costs: dict[int, float] = {}
    eval_costs: dict[int, float] = {}
    split_counts: dict[str, tuple[int, int]] = {}
    train_sample_id = 80_000_000
    eval_sample_id = 82_000_000
    for case_index, (case_name, (case_type, pairs)) in enumerate(cases.items()):
        train_pairs, eval_pairs = _split_pairs(
            pairs,
            train_fraction=cfg.train_fraction,
            seed=cfg.seed + 131 * case_index,
        )
        split_counts[case_name] = (len(train_pairs), len(eval_pairs))
        for pair_index, pair in enumerate(train_pairs):
            for local_index in range(cfg.samples_per_pair):
                sample = _sample_from_pair(
                    pair,
                    sample_id=train_sample_id,
                    variant_index=case_index * 100_000 + pair_index * cfg.samples_per_pair + local_index,
                    case_type=case_type,
                )
                train_samples.append(sample)
                train_case_by_query[sample.query_id] = case_name
                train_type_by_query[sample.query_id] = case_type
                train_costs[sample.query_id] = float(pair.cost_scale)
                train_sample_id += 1
        for pair_index, pair in enumerate(eval_pairs):
            for local_index in range(cfg.samples_per_pair):
                sample = _sample_from_pair(
                    pair,
                    sample_id=eval_sample_id,
                    variant_index=case_index * 100_000 + pair_index * cfg.samples_per_pair + local_index,
                    case_type=case_type,
                )
                eval_samples.append(sample)
                eval_case_by_query[sample.query_id] = case_name
                eval_type_by_query[sample.query_id] = case_type
                eval_costs[sample.query_id] = float(pair.cost_scale)
                eval_sample_id += 1
    return (
        _dataset_from_samples(train_samples),
        _dataset_from_samples(eval_samples),
        train_case_by_query,
        eval_case_by_query,
        train_type_by_query,
        eval_type_by_query,
        train_costs,
        eval_costs,
        split_counts,
    )


def _metric_dict(metric: HighDensityMethodMetrics) -> dict[str, object]:
    return asdict(metric)


def _reduction(trained: HighDensityMethodMetrics, baseline: HighDensityMethodMetrics) -> float:
    return 1.0 - float(trained.exact_work_units) / max(1.0e-9, float(baseline.exact_work_units))


def _call_reduction(trained: HighDensityMethodMetrics, baseline: HighDensityMethodMetrics) -> float:
    return 1.0 - float(trained.exact_call_count) / max(1.0, float(baseline.exact_call_count))


def _positive_query_count(workload, query_ids: set[int] | None = None) -> int:
    total = 0
    for query_id, trace in workload.traces_by_query_id.items():
        if query_ids is not None and query_id not in query_ids:
            continue
        total += int(bool(trace.collided))
    return total


def _load_checkpoint_model(checkpoint: Path, *, device: str):
    import torch

    payload = torch.load(checkpoint, map_location=device)
    model, state_dict = build_stpf_model_from_checkpoint_payload(
        payload,
        fallback_preset=STPFModelPreset.MEDIUM_MLP,
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def _write_asset_manifest(path: Path, records: Sequence[ShapeNetAssetRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(records[0]).keys()) if records else [])
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def _write_case_summary(path: Path, rows: Sequence[ShapeNetCaseSummary]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()) if rows else [])
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _write_report(path: Path, result: dict[str, object]) -> None:
    baseline = result["baseline"]
    random_stpf = result["random_stpf"]
    cross_stpf = result.get("cross_dataset_stpf")
    trained = result["trained_stpf"]
    ort = result["ort_tensorrt_cpp"]
    case_rows = result["case_summaries"]
    assert isinstance(baseline, dict)
    assert isinstance(random_stpf, dict)
    assert isinstance(trained, dict)
    assert isinstance(ort, dict)
    assert isinstance(case_rows, list)
    metric_rows = [baseline, random_stpf]
    if isinstance(cross_stpf, dict):
        metric_rows.append(cross_stpf)
    metric_rows.append(trained)
    lines = [
        "# ShapeNet-OOD Dense Mesh Contact / High-Speed / Thin-Feature Benchmark",
        "",
        "## 1. Objective",
        "",
        "descriptionusedescription ShapeNetCore v2 in car / airplane / watercraft / chair / table / sofa / lamp / rifle / bus / train / guitar / loudspeaker descriptionconstruct OOD dense/high-cost CCD case. ",
        "",
        "- OBJ descriptionused for mesh complexity / exact-work cost calibrated. ",
        "- `.solid.binvox` and `.surface.binvox` Filedescriptionas occupancy proxy / broad-phase description, enter pair score and cost scale. ",
        "- STPF only performs proposal / scheduling; description correctness description exact certificate / conservative fallback guarantee. ",
        "",
        "## 2. description",
        "",
        f"- run name: `{result['run_name']}`",
        f"- ShapeNet root: `{result['shapenet_root']}`",
        f"- categories: `{result['category_count']}`",
        f"- selected assets: `{result['selected_asset_count']}`",
        f"- case families: `{result['case_family_count']}`",
        f"- density: `{result['density']}` candidates/query",
        f"- train queries: `{result['train_query_count']}`",
        f"- eval queries: `{result['eval_query_count']}`",
        f"- train dense rows: `{result['train_dense_rows']}`",
        f"- eval dense rows: `{result['eval_dense_rows']}`",
        f"- checkpoint: `{result['checkpoint_path']}`",
        "",
        "## 3. Overall description",
        "",
        "| Method | Exact calls | Exact work | FN | Proposal ms | Schedule ms | Total ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in metric_rows:
        lines.append(
            f"| `{row['method_name']}` | `{row['exact_call_count']}` | `{float(row['exact_work_units']):.4f}` | "
            f"`{row['fn_count']}` | `{float(row['proposal_wall_ms']):.3f}` | "
            f"`{float(row['scheduling_wall_ms']):.3f}` | `{float(row['total_wall_ms']):.3f}` |"
        )
    lines.extend(
        [
            "",
            f"- trained vs NoProposal exact-work reduction: `{float(result['trained_reduction_vs_no_proposal']):.4%}`",
            f"- trained vs Random STPF exact-work reduction: `{float(result['trained_reduction_vs_random']):.4%}`",
            f"- trained exact-call reduction vs NoProposal: `{float(result['trained_call_reduction_vs_no_proposal']):.4%}`",
        ]
    )
    if result.get("cross_dataset_reduction_vs_no_proposal") is not None:
        lines.append(
            f"- ABC/Fusion/Thingi/physics v4 checkpoint on ShapeNet heldout exact-work reduction: `{float(result['cross_dataset_reduction_vs_no_proposal']):.4%}`"
        )
    lines.extend(
        [
            "",
            "## 4. ORT TensorRT + C++ Scheduling",
            "",
            f"- ORT provider: `{ort['ort_provider']}`",
            f"- ORT inference ms: `{float(ort['ort_inference_ms']):.3f}`",
            f"- C++ scheduling ms: `{float(ort['cpp_schedule_ms']):.3f}`",
            f"- proposal total ms: `{float(ort['proposal_total_ms']):.3f}`",
            f"- proposal rows/s: `{float(ort['proposal_rows_per_second']):.1f}`",
            "",
            "## 5. split case description",
            "",
            "| Case | Type | Role | Queries | Candidates | Positives | Trained work reduction | Cross-dataset work reduction | Trained exact calls | FN |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in case_rows:
        work = "" if row["exact_work_reduction"] is None else f"{100.0 * float(row['exact_work_reduction']):.2f}%"
        cross = "" if row["cross_dataset_exact_work_reduction"] is None else f"{100.0 * float(row['cross_dataset_exact_work_reduction']):.2f}%"
        lines.append(
            f"| `{row['case_name']}` | `{row['case_type']}` | `{row['role']}` | `{row['query_count']}` | "
            f"`{row['candidate_count']}` | `{row['positive_queries']}` | `{work}` | `{cross}` | "
            f"`{row.get('trained_exact_calls', '')}` | `{row.get('fn_count', '')}` |"
        )
    lines.extend(
        [
            "",
            "## 6. Conclusion",
            "",
            "- ShapeNetCore v2 descriptionwithas OOD mesh generalizationdescription: descriptioncoveragedescription, OBJ highdescriptionModeldescription, andcontains binvox occupancy proxy. ",
            "- description case coverage dense mesh contact, high-speed rigid collision, thin feature hard cases, binvox occupancy proxy and cross-dataset generalization. ",
            "- this benchmark description dense proxy/exact-work Protocol; final paper SOTA wall-time descriptionand Tight-Inclusion / exact certificate fallback descriptionsplitdescriptionwrite. ",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_shapenet_ood_dense_cases(cfg: ShapeNetOODDenseConfig | None = None) -> dict[str, object]:
    config = cfg or ShapeNetOODDenseConfig()
    if not config.shapenet_root.exists():
        raise FileNotFoundError(config.shapenet_root)
    if not 0.0 < config.train_fraction < 1.0:
        raise ValueError("train_fraction must be in (0, 1)")
    shard_dir = config.shard_root / config.run_name
    shard_dir.mkdir(parents=True, exist_ok=True)
    config.benchmark_output_dir.mkdir(parents=True, exist_ok=True)

    assets_by_category, records = _load_assets(config)
    cases = _build_case_specs(assets_by_category, pair_limit=config.pair_limit_per_case)
    if not cases:
        raise RuntimeError("no ShapeNet dense cases were generated")

    (
        train_dataset,
        eval_dataset,
        train_case_by_query,
        eval_case_by_query,
        train_type_by_query,
        eval_type_by_query,
        train_costs,
        eval_costs,
        split_counts,
    ) = _build_datasets(cases, cfg=config)
    train_workload = _scale_workload_costs(
        build_high_density_stpf_workload(train_dataset, config.high_density, name=f"{config.run_name}_train"),
        train_costs,
    )
    eval_workload = _scale_workload_costs(
        build_high_density_stpf_workload(eval_dataset, config.high_density, name=f"{config.run_name}_eval"),
        eval_costs,
    )
    dense_train_dataset = workload_to_shard_dataset(train_workload)
    dense_eval_dataset = workload_to_shard_dataset(eval_workload)
    write_npz_shard(
        shard_dir / "base_train.npz",
        train_dataset,
        metadata={**default_metadata(train_dataset, seed=config.seed, source=config.run_name), "dataset_role": "base_train"},
    )
    write_npz_shard(
        shard_dir / "base_eval.npz",
        eval_dataset,
        metadata={**default_metadata(eval_dataset, seed=config.seed + 1, source=config.run_name), "dataset_role": "base_eval"},
    )
    write_npz_shard(
        shard_dir / "dense_train.npz",
        dense_train_dataset,
        metadata={
            **default_metadata(dense_train_dataset, seed=config.seed, source=config.run_name),
            "dataset_role": "dense_train",
            "candidates_per_query": _density(config.high_density),
        },
    )
    write_npz_shard(
        shard_dir / "dense_eval.npz",
        dense_eval_dataset,
        metadata={
            **default_metadata(dense_eval_dataset, seed=config.seed + 1, source=config.run_name),
            "dataset_role": "dense_eval",
            "candidates_per_query": _density(config.high_density),
        },
    )
    asset_manifest_path = config.benchmark_output_dir / f"{config.run_name}_selected_assets.csv"
    _write_asset_manifest(asset_manifest_path, records)

    training_run = run_stpf_training(
        train_workload.rows,
        STPFTrainingRunConfig(
            training=config.training,
            output_dir=str(config.training_output_dir),
            run_name=config.run_name,
        ),
        validation_rows=eval_workload.rows,
    )
    baseline = benchmark_no_proposal_on_high_density_workload(eval_workload)
    random_model = build_stpf_model(STPFModelPreset.MEDIUM_MLP)
    random_model.to(config.training.device)
    random_model.eval()
    random_stpf = benchmark_stpf_on_high_density_workload(
        eval_workload,
        model=random_model,
        device=config.training.device,
        proposal_batch_size=config.training.batch_size,
        method_name="RTSTPFExact-Random-MediumMLP",
    )
    cross_dataset_stpf = None
    cross_model = None
    if config.cross_dataset_checkpoint.exists():
        cross_model = _load_checkpoint_model(config.cross_dataset_checkpoint, device=config.training.device)
        cross_dataset_stpf = benchmark_stpf_on_high_density_workload(
            eval_workload,
            model=cross_model,
            device=config.training.device,
            proposal_batch_size=config.training.batch_size,
            method_name="RTSTPFExact-CrossDataset-V4Checkpoint",
        )
    trained_model = training_run.result.model
    trained_model.to(config.training.device)
    trained_model.eval()
    trained_stpf = benchmark_stpf_on_high_density_workload(
        eval_workload,
        model=trained_model,
        device=config.training.device,
        proposal_batch_size=config.training.batch_size,
        method_name="RTSTPFExact-ShapeNetOOD-Learned",
    )

    case_rows: list[ShapeNetCaseSummary] = []
    all_cases = sorted(set(train_case_by_query.values()) | set(eval_case_by_query.values()))
    for case_name in all_cases:
        for role, workload, case_map, type_map in (
            ("train", train_workload, train_case_by_query, train_type_by_query),
            ("eval", eval_workload, eval_case_by_query, eval_type_by_query),
        ):
            query_ids = {query_id for query_id, mapped in case_map.items() if mapped == case_name}
            if not query_ids:
                continue
            case_type = next((type_map[query_id] for query_id in query_ids), "")
            if role == "train":
                case_rows.append(
                    ShapeNetCaseSummary(
                        case_name=case_name,
                        case_type=case_type,
                        role=role,
                        query_count=len(query_ids),
                        candidate_count=len(query_ids) * _density(config.high_density),
                        positive_queries=_positive_query_count(workload, query_ids),
                    )
                )
                continue
            case_workload = _subset_workload(workload, query_ids, name=f"{config.run_name}_{case_name}_eval")
            case_baseline = benchmark_no_proposal_on_high_density_workload(case_workload)
            case_trained = benchmark_stpf_on_high_density_workload(
                case_workload,
                model=trained_model,
                device=config.training.device,
                proposal_batch_size=config.training.batch_size,
                method_name=f"RTSTPFExact-ShapeNet-{case_name}",
            )
            case_cross = None
            if cross_model is not None:
                case_cross = benchmark_stpf_on_high_density_workload(
                    case_workload,
                    model=cross_model,
                    device=config.training.device,
                    proposal_batch_size=config.training.batch_size,
                    method_name=f"RTSTPFExact-CrossDataset-{case_name}",
                )
            case_rows.append(
                ShapeNetCaseSummary(
                    case_name=case_name,
                    case_type=case_type,
                    role=role,
                    query_count=len(query_ids),
                    candidate_count=len(query_ids) * _density(config.high_density),
                    positive_queries=_positive_query_count(workload, query_ids),
                    exact_work_reduction=_reduction(case_trained, case_baseline),
                    exact_call_reduction=_call_reduction(case_trained, case_baseline),
                    cross_dataset_exact_work_reduction=None if case_cross is None else _reduction(case_cross, case_baseline),
                    trained_exact_calls=case_trained.exact_call_count,
                    cross_dataset_exact_calls=None if case_cross is None else case_cross.exact_call_count,
                    fn_count=case_trained.fn_count,
                    cross_dataset_fn_count=None if case_cross is None else case_cross.fn_count,
                )
            )

    case_csv_path = config.benchmark_output_dir / f"{config.run_name}_case_summary.csv"
    _write_case_summary(case_csv_path, case_rows)

    checkpoint_path = training_run.artifacts.model_state_path
    if checkpoint_path is None:
        raise RuntimeError("training did not produce checkpoint")
    ort_report = config.benchmark_output_dir / f"{config.run_name}_ort_tensorrt_cpp_walltime.md"
    ort_json = config.benchmark_output_dir / f"{config.run_name}_ort_tensorrt_cpp_walltime.json"
    ort_result = run_common_modeling_ort_walltime_benchmark(
        checkpoint_path=checkpoint_path,
        dense_shard_path=shard_dir / "dense_eval.npz",
        report_path=ort_report,
        json_path=ort_json,
        device=config.training.device,
        batch_size=config.training.batch_size,
        uncertainty_fallback_threshold=config.high_density.uncertainty_fallback_threshold,
        warmup_passes=2,
    )

    result: dict[str, object] = {
        "run_name": config.run_name,
        "shapenet_root": config.shapenet_root.as_posix(),
        "category_count": len(assets_by_category),
        "selected_asset_count": len(records),
        "case_family_count": len(cases),
        "density": _density(config.high_density),
        "pair_split_counts": {key: {"train_pairs": value[0], "eval_pairs": value[1]} for key, value in split_counts.items()},
        "train_query_count": train_workload.query_count,
        "eval_query_count": eval_workload.query_count,
        "train_dense_rows": train_workload.candidate_count,
        "eval_dense_rows": eval_workload.candidate_count,
        "train_positive_queries": _positive_query_count(train_workload),
        "eval_positive_queries": _positive_query_count(eval_workload),
        "model_preset": str(config.training.model_preset),
        "epochs": config.training.epochs,
        "batch_size": config.training.batch_size,
        "checkpoint_path": checkpoint_path.as_posix(),
        "onnx_path": checkpoint_path.with_suffix(".onnx").as_posix(),
        "final_train_loss": training_run.final_train_loss,
        "final_validation_loss": training_run.final_validation_loss,
        "baseline": _metric_dict(baseline),
        "random_stpf": _metric_dict(random_stpf),
        "cross_dataset_stpf": None if cross_dataset_stpf is None else _metric_dict(cross_dataset_stpf),
        "trained_stpf": _metric_dict(trained_stpf),
        "trained_reduction_vs_no_proposal": _reduction(trained_stpf, baseline),
        "trained_reduction_vs_random": _reduction(trained_stpf, random_stpf),
        "trained_call_reduction_vs_no_proposal": _call_reduction(trained_stpf, baseline),
        "cross_dataset_reduction_vs_no_proposal": None if cross_dataset_stpf is None else _reduction(cross_dataset_stpf, baseline),
        "case_summaries": [asdict(row) for row in case_rows],
        "ort_tensorrt_cpp": ort_result,
        "artifacts": {
            "shard_dir": shard_dir.as_posix(),
            "base_train": (shard_dir / "base_train.npz").as_posix(),
            "base_eval": (shard_dir / "base_eval.npz").as_posix(),
            "dense_train": (shard_dir / "dense_train.npz").as_posix(),
            "dense_eval": (shard_dir / "dense_eval.npz").as_posix(),
            "selected_assets_csv": asset_manifest_path.as_posix(),
            "case_summary_csv": case_csv_path.as_posix(),
            "ort_report": ort_report.as_posix(),
        },
    }
    report_path = config.benchmark_output_dir / f"{config.run_name}.md"
    json_path = config.benchmark_output_dir / f"{config.run_name}.json"
    manifest_path = shard_dir / "manifest.json"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    manifest_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    _write_report(report_path, result)
    return result


def main() -> None:
    result = run_shapenet_ood_dense_cases()
    print(
        json.dumps(
            {
                "report": f"src/benchmark/{RUN_NAME}.md",
                "checkpoint": result["checkpoint_path"],
                "train_dense_rows": result["train_dense_rows"],
                "eval_dense_rows": result["eval_dense_rows"],
                "case_family_count": result["case_family_count"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
