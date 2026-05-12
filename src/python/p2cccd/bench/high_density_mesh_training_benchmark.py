from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
import random
from itertools import combinations
from pathlib import Path
from typing import Sequence

from p2cccd.contracts import ProxyType
from p2cccd.data import GeneratedDataset, proposal_row_from_oracle_trace, write_npz_shard, default_metadata
from p2cccd.data.oracle import evaluate_swept_sphere_oracle
from p2cccd.data.response import proxy_mass_from_radius
from p2cccd.data.samplers import MotionDiscPairSample, PairFamily
from p2cccd.datasets.cad import ABCDatasetAdapter, Fusion360GalleryAdapter
from p2cccd.datasets.cad.contracts import CadMeshAsset, CadMeshStats
from p2cccd.datasets.cad.mesh_io import mesh_stats_from_file
from p2cccd.proposal.stpf_model import STPFModelPreset, build_stpf_model
from p2cccd.proposal.training import STPFTrainingConfig
from p2cccd.proposal.training_runner import STPFTrainingRunConfig, STPFTrainingRunResult, run_stpf_training

from .complete_example_scaled_training_benchmark import CompleteExampleScaledTrainingBenchmarkConfig, _benchmark_dataset
from .trained_stpf_high_density import (
    HighDensityMethodMetrics,
    HighDensitySTPFConfig,
    HighDensitySTPFWorkload,
    benchmark_no_proposal_on_high_density_workload,
    benchmark_stpf_on_high_density_workload,
    build_high_density_stpf_workload,
)


def _default_training() -> STPFTrainingConfig:
    return STPFTrainingConfig(
        epochs=4,
        batch_size=4096,
        learning_rate=1.0e-3,
        seed=424242,
        device="cuda",
        validation_fraction=0.0,
        model_preset=STPFModelPreset.MEDIUM_MLP,
    )


@dataclass(frozen=True, slots=True)
class HighDensityMeshBenchmarkConfig:
    abc_root: str = "src/datasets/abc_official"
    fusion360_root: str = "src/datasets/fusion360"
    thingi10k_root: str = "src/datasets/thingi10k"
    asset_limit_per_source: int = 96
    samples_per_source: int = 1000
    train_fraction: float = 0.75
    seed: int = 424242
    high_density: HighDensitySTPFConfig = HighDensitySTPFConfig(
        slab_count=8,
        patches_per_object=4,
        representative_attempt_limit=3,
        uncertainty_fallback_threshold=0.75,
        narrow_interval_min_cost_scale=0.18,
        interval_miss_penalty_scale=0.22,
        full_exact_cost_scale=1.0,
    )
    training: STPFTrainingConfig = _default_training()
    run_name: str = "high_density_mesh_training_benchmark_run_id"
    shard_root: str = "src/datasets/training/high_density_mesh/shards"
    training_output_dir: str = "src/outputs/stpf_training"
    benchmark_output_dir: str = "src/benchmark"


@dataclass(frozen=True, slots=True)
class MeshDensityAsset:
    source_name: str
    asset_id: str
    asset_path: str
    face_count: int
    vertex_count: int
    diagonal: float
    bounds_min: tuple[float, float, float]
    bounds_max: tuple[float, float, float]
    dirty_score: float = 0.0


@dataclass(frozen=True, slots=True)
class MeshDensityPair:
    source_name: str
    asset_a: MeshDensityAsset
    asset_b: MeshDensityAsset
    pair_score: float
    cost_scale: float


@dataclass(frozen=True, slots=True)
class HighDensityMeshSourceResult:
    source_name: str
    asset_count: int
    sample_count: int
    train_query_count: int
    eval_query_count: int
    min_face_count: int
    median_face_count: int
    max_face_count: int
    no_proposal: HighDensityMethodMetrics
    random_stpf: HighDensityMethodMetrics
    trained_stpf: HighDensityMethodMetrics


@dataclass(frozen=True, slots=True)
class HighDensityMeshBenchmarkResult:
    config: HighDensityMeshBenchmarkConfig
    assets_by_source: dict[str, tuple[MeshDensityAsset, ...]]
    train_dataset: GeneratedDataset
    eval_dataset: GeneratedDataset
    train_workload: HighDensitySTPFWorkload
    eval_workload: HighDensitySTPFWorkload
    training_run: STPFTrainingRunResult
    combined_no_proposal: HighDensityMethodMetrics
    combined_random_stpf: HighDensityMethodMetrics
    combined_trained_stpf: HighDensityMethodMetrics
    source_results: tuple[HighDensityMeshSourceResult, ...]
    query_level_rows: tuple[object, ...]
    report_path: Path
    summary_json_path: Path


def _stable_u32(token: str) -> int:
    return int(hashlib.sha1(token.encode("utf-8")).hexdigest()[:8], 16)


def _asset_from_cad(source_name: str, asset: CadMeshAsset) -> MeshDensityAsset:
    stats = asset.stats
    return MeshDensityAsset(
        source_name=source_name,
        asset_id=asset.asset_id,
        asset_path=str(asset.asset_path),
        face_count=int(stats.face_count),
        vertex_count=int(stats.vertex_count),
        diagonal=float(stats.diagonal),
        bounds_min=stats.bounds_min,
        bounds_max=stats.bounds_max,
        dirty_score=0.0,
    )


def _asset_from_path(
    *,
    source_name: str,
    root: Path,
    relative_path: str,
    asset_id: str,
    dirty_score: float = 0.0,
) -> MeshDensityAsset:
    path = root / relative_path
    stats = mesh_stats_from_file(path)
    return MeshDensityAsset(
        source_name=source_name,
        asset_id=asset_id,
        asset_path=str(path),
        face_count=int(stats.face_count),
        vertex_count=int(stats.vertex_count),
        diagonal=float(stats.diagonal),
        bounds_min=stats.bounds_min,
        bounds_max=stats.bounds_max,
        dirty_score=float(dirty_score),
    )


def _load_abc_assets(root: Path, limit: int) -> tuple[MeshDensityAsset, ...]:
    adapter = ABCDatasetAdapter(root)
    assets = [_asset_from_cad("ABC official", asset) for asset in adapter.list_assets(limit=1_000_000)]
    assets.sort(key=lambda asset: (-asset.face_count, -asset.vertex_count, asset.asset_id))
    return tuple(assets[:limit])


def _load_fusion360_assets(root: Path, limit: int) -> tuple[MeshDensityAsset, ...]:
    adapter = Fusion360GalleryAdapter(root)
    assets: list[MeshDensityAsset] = []
    # Parsing all Fusion assets is acceptable for an offline training job and avoids
    # accidentally biasing the high-density subset toward early archive directories.
    for sequence in adapter.list_sequences():
        for asset in sequence.assets:
            if Path(asset.asset_path).name.lower().startswith("assembly."):
                continue
            assets.append(_asset_from_cad("Fusion 360 Gallery", asset))
    assets.sort(key=lambda asset: (-asset.face_count, -asset.vertex_count, asset.asset_id))
    return tuple(assets[:limit])


def _load_thingi10k_assets(root: Path, limit: int) -> tuple[MeshDensityAsset, ...]:
    manifest_path = root / "official_subset_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Thingi10K manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = list(manifest.get("assets", []))
    rows.sort(key=lambda row: (-int(row.get("num_facets", 0)), -int(row.get("num_vertices", 0)), str(row.get("mesh_path"))))
    assets: list[MeshDensityAsset] = []
    for row in rows[:limit]:
        assets.append(
            _asset_from_path(
                source_name="Thingi10K",
                root=root,
                relative_path=str(row["mesh_path"]),
                asset_id=f"thingi10k-{int(row.get('file_id', len(assets)))}",
                dirty_score=float(row.get("dirty_score", 0.0)),
            )
        )
    assets.sort(key=lambda asset: (-asset.face_count, -asset.vertex_count, asset.asset_id))
    return tuple(assets)


def _center(asset: MeshDensityAsset) -> tuple[float, float, float]:
    return (
        0.5 * (asset.bounds_min[0] + asset.bounds_max[0]),
        0.5 * (asset.bounds_min[1] + asset.bounds_max[1]),
        0.5 * (asset.bounds_min[2] + asset.bounds_max[2]),
    )


def _pair_score(asset_a: MeshDensityAsset, asset_b: MeshDensityAsset) -> float:
    diag_a = max(asset_a.diagonal, 1.0e-12)
    diag_b = max(asset_b.diagonal, 1.0e-12)
    face_a = max(asset_a.face_count, 1)
    face_b = max(asset_b.face_count, 1)
    scale_similarity = math.exp(-abs(math.log(diag_a / diag_b)))
    face_similarity = math.exp(-abs(math.log(face_a / face_b)))
    face_complexity = min(1.0, math.log1p(math.sqrt(face_a * face_b)) / math.log1p(1_000_000.0))
    dirty = max(asset_a.dirty_score, asset_b.dirty_score)
    return min(1.0, 0.30 * scale_similarity + 0.25 * face_similarity + 0.35 * face_complexity + 0.10 * dirty)


def _cost_scale(asset_a: MeshDensityAsset, asset_b: MeshDensityAsset) -> float:
    primitive_scale = math.sqrt(max(1, asset_a.face_count) * max(1, asset_b.face_count))
    return max(1.0, primitive_scale / 1024.0)


def _make_pairs(assets: Sequence[MeshDensityAsset], limit: int) -> tuple[MeshDensityPair, ...]:
    pairs: list[MeshDensityPair] = []
    for asset_a, asset_b in combinations(assets, 2):
        if asset_a.source_name != asset_b.source_name:
            continue
        pairs.append(
            MeshDensityPair(
                source_name=asset_a.source_name,
                asset_a=asset_a,
                asset_b=asset_b,
                pair_score=_pair_score(asset_a, asset_b),
                cost_scale=_cost_scale(asset_a, asset_b),
            )
        )
    pairs.sort(key=lambda pair: (-pair.pair_score, -pair.cost_scale, pair.asset_a.asset_id, pair.asset_b.asset_id))
    return tuple(pairs[:limit])


def _radius(asset: MeshDensityAsset) -> float:
    face_factor = 1.0 + 0.03 * math.log1p(max(1, asset.face_count))
    return max(0.05, 0.5 * max(asset.diagonal, 0.1) * face_factor)


def _sample_from_pair(pair: MeshDensityPair, *, sample_id: int, variant_index: int) -> MotionDiscPairSample:
    radius_a = _radius(pair.asset_a)
    radius_b = _radius(pair.asset_b)
    radius_sum = radius_a + radius_b
    scale = max(0.1, 0.5 * (max(pair.asset_a.diagonal, 0.1) + max(pair.asset_b.diagonal, 0.1)))
    token = _stable_u32(pair.asset_a.asset_id + pair.asset_b.asset_id + str(variant_index))
    lateral = ((token % 200) / 199.0 - 0.5) * 0.16 * scale
    z_bias = (((token // 200) % 200) / 199.0 - 0.5) * 0.08 * scale
    split_cycle = (
        "mesh_high_density_easy_negative",
        "mesh_high_density_near_contact",
        "mesh_high_density_grazing",
        "mesh_high_density_multi_contact",
    )
    split = split_cycle[variant_index % len(split_cycle)]
    if split.endswith("easy_negative"):
        start_gap = radius_sum + (1.4 + 0.4 * pair.pair_score) * scale
        end_gap = radius_sum + (0.6 + 0.2 * pair.pair_score) * scale
        ax1 = 0.20 * scale
    elif split.endswith("near_contact"):
        start_gap = radius_sum + (0.25 + 0.08 * pair.pair_score) * scale
        end_gap = radius_sum + (0.035 + 0.02 * pair.pair_score) * scale
        ax1 = 0.24 * scale
    elif split.endswith("grazing"):
        start_gap = radius_sum + (0.16 + 0.06 * pair.pair_score) * scale
        end_gap = radius_sum - (0.010 + 0.02 * pair.pair_score) * scale
        ax1 = 0.18 * scale
    else:
        start_gap = radius_sum + (0.38 + 0.08 * pair.pair_score) * scale
        end_gap = -(radius_sum + (0.18 + 0.07 * pair.pair_score) * scale)
        ax1 = 0.10 * scale

    return MotionDiscPairSample(
        sample_id=sample_id,
        query_id=8_700_000 + sample_id,
        candidate_id=8_800_000 + sample_id,
        split=split,
        family=PairFamily.MESH_PAIR,
        object_a_id=700_000 + (_stable_u32(pair.asset_a.asset_id) % 200_000),
        patch_a_id=1 + (_stable_u32(pair.asset_a.asset_path) % max(1, min(pair.asset_a.face_count, 100_000))),
        object_b_id=700_000 + (_stable_u32(pair.asset_b.asset_id) % 200_000),
        patch_b_id=1 + (_stable_u32(pair.asset_b.asset_path) % max(1, min(pair.asset_b.face_count, 100_000))),
        slab_id=variant_index % 8,
        center_a_t0=(0.0, lateral, z_bias),
        center_a_t1=(ax1, lateral + 0.02 * scale * ((variant_index % 2) * 2 - 1), z_bias),
        center_b_t0=(start_gap, lateral - 0.015 * scale, z_bias),
        center_b_t1=(end_gap, lateral + 0.015 * scale, z_bias),
        radius_a=radius_a,
        radius_b=radius_b,
        proxy_type_a=ProxyType.SWEPT_AABB,
        proxy_type_b=ProxyType.SWEPT_AABB,
        hardness=max(0.0, min(1.0, pair.pair_score)),
        ood=pair.source_name == "Thingi10K",
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


def _scale_workload_costs(
    workload: HighDensitySTPFWorkload,
    cost_scale_by_query_id: dict[int, float],
) -> HighDensitySTPFWorkload:
    scaled_infos = {}
    for candidate_id, info in workload.candidate_infos.items():
        scale = cost_scale_by_query_id.get(info.query_id, 1.0)
        scaled_infos[candidate_id] = replace(
            info,
            full_exact_cost=info.full_exact_cost * scale,
            narrow_exact_cost=info.narrow_exact_cost * scale,
        )
    return HighDensitySTPFWorkload(
        name=workload.name,
        config=workload.config,
        samples=workload.samples,
        traces_by_query_id=workload.traces_by_query_id,
        candidates=workload.candidates,
        rows=workload.rows,
        candidate_infos=scaled_infos,
    )


def _subset_dataset(dataset: GeneratedDataset, indices: set[int]) -> GeneratedDataset:
    return GeneratedDataset(
        rows=[row for index, row in enumerate(dataset.rows) if index in indices],
        samples=[sample for index, sample in enumerate(dataset.samples) if index in indices],
        traces=[trace for index, trace in enumerate(dataset.traces) if index in indices],
        split_names=dataset.split_names,
    )


def _subset_workload(workload: HighDensitySTPFWorkload, query_ids: set[int], name: str) -> HighDensitySTPFWorkload:
    candidate_ids = {
        candidate.candidate_id
        for candidate in workload.candidates
        if candidate.query_id in query_ids
    }
    return HighDensitySTPFWorkload(
        name=name,
        config=workload.config,
        samples=tuple(sample for sample in workload.samples if sample.query_id in query_ids),
        traces_by_query_id={query_id: trace for query_id, trace in workload.traces_by_query_id.items() if query_id in query_ids},
        candidates=tuple(candidate for candidate in workload.candidates if candidate.query_id in query_ids),
        rows=tuple(row for row in workload.rows if row.query_id in query_ids),
        candidate_infos={cid: info for cid, info in workload.candidate_infos.items() if cid in candidate_ids},
    )


def run_high_density_mesh_training_benchmark(
    config: HighDensityMeshBenchmarkConfig | None = None,
) -> HighDensityMeshBenchmarkResult:
    cfg = config or HighDensityMeshBenchmarkConfig()
    if cfg.asset_limit_per_source < 2:
        raise ValueError("asset_limit_per_source must be at least 2")
    if cfg.samples_per_source < 2:
        raise ValueError("samples_per_source must be at least 2")
    if not 0.0 < cfg.train_fraction < 1.0:
        raise ValueError("train_fraction must lie in (0, 1)")

    assets_by_source = {
        "ABC official": _load_abc_assets(Path(cfg.abc_root), cfg.asset_limit_per_source),
        "Fusion 360 Gallery": _load_fusion360_assets(Path(cfg.fusion360_root), cfg.asset_limit_per_source),
        "Thingi10K": _load_thingi10k_assets(Path(cfg.thingi10k_root), cfg.asset_limit_per_source),
    }
    pairs_by_source = {
        source_name: _make_pairs(assets, cfg.samples_per_source)
        for source_name, assets in assets_by_source.items()
    }

    samples: list[MotionDiscPairSample] = []
    source_by_query_id: dict[int, str] = {}
    cost_scale_by_query_id: dict[int, float] = {}
    sample_id = 1
    for source_name, pairs in pairs_by_source.items():
        for index, pair in enumerate(pairs):
            sample = _sample_from_pair(pair, sample_id=sample_id, variant_index=index)
            samples.append(sample)
            source_by_query_id[sample.query_id] = source_name
            cost_scale_by_query_id[sample.query_id] = pair.cost_scale
            sample_id += 1

    dataset = _dataset_from_samples(samples)
    indices_by_source: dict[str, list[int]] = {}
    for index, sample in enumerate(dataset.samples):
        indices_by_source.setdefault(source_by_query_id[sample.query_id], []).append(index)

    rng = random.Random(cfg.seed)
    train_indices: set[int] = set()
    eval_indices: set[int] = set()
    for indices in indices_by_source.values():
        shuffled = list(indices)
        rng.shuffle(shuffled)
        train_count = max(1, min(len(shuffled) - 1, int(round(len(shuffled) * cfg.train_fraction))))
        train_indices.update(shuffled[:train_count])
        eval_indices.update(shuffled[train_count:])

    train_dataset = _subset_dataset(dataset, train_indices)
    eval_dataset = _subset_dataset(dataset, eval_indices)
    train_workload = _scale_workload_costs(
        build_high_density_stpf_workload(train_dataset, cfg.high_density, name="high_density_mesh_train"),
        cost_scale_by_query_id,
    )
    eval_workload = _scale_workload_costs(
        build_high_density_stpf_workload(eval_dataset, cfg.high_density, name="high_density_mesh_eval"),
        cost_scale_by_query_id,
    )

    shard_dir = Path(cfg.shard_root) / cfg.run_name
    shard_dir.mkdir(parents=True, exist_ok=True)
    write_npz_shard(
        shard_dir / "train.npz",
        train_dataset,
        metadata={**default_metadata(train_dataset, seed=cfg.seed, source="high_density_mesh"), "dataset_role": "train"},
    )
    write_npz_shard(
        shard_dir / "eval.npz",
        eval_dataset,
        metadata={**default_metadata(eval_dataset, seed=cfg.seed + 1, source="high_density_mesh"), "dataset_role": "eval"},
    )

    training_run = run_stpf_training(
        train_workload.rows,
        STPFTrainingRunConfig(
            training=cfg.training,
            output_dir=cfg.training_output_dir,
            run_name=cfg.run_name,
        ),
        validation_rows=eval_workload.rows,
    )
    trained_model = training_run.result.model
    trained_model.eval()
    trained_model.to(cfg.training.device)
    random_model = build_stpf_model(cfg.training.model_preset)
    random_model.eval()
    random_model.to(cfg.training.device)

    combined_no_proposal = benchmark_no_proposal_on_high_density_workload(eval_workload)
    combined_random_stpf = benchmark_stpf_on_high_density_workload(
        eval_workload,
        model=random_model,
        device=cfg.training.device,
        proposal_batch_size=cfg.training.batch_size,
        method_name="HighDensityMesh-RTSTPFExact-Random",
    )
    combined_trained_stpf = benchmark_stpf_on_high_density_workload(
        eval_workload,
        model=trained_model,
        device=cfg.training.device,
        proposal_batch_size=cfg.training.batch_size,
        method_name="HighDensityMesh-RTSTPFExact-Trained",
    )

    source_results: list[HighDensityMeshSourceResult] = []
    eval_query_ids_by_source: dict[str, set[int]] = {}
    for sample in eval_dataset.samples:
        eval_query_ids_by_source.setdefault(source_by_query_id[sample.query_id], set()).add(sample.query_id)
    for source_name, query_ids in sorted(eval_query_ids_by_source.items()):
        source_workload = _subset_workload(eval_workload, query_ids, f"high_density_mesh_eval_{source_name}")
        no_proposal = benchmark_no_proposal_on_high_density_workload(source_workload)
        random_stpf = benchmark_stpf_on_high_density_workload(
            source_workload,
            model=random_model,
            device=cfg.training.device,
            proposal_batch_size=cfg.training.batch_size,
            method_name=f"{source_name}-RTSTPFExact-Random",
        )
        trained_stpf = benchmark_stpf_on_high_density_workload(
            source_workload,
            model=trained_model,
            device=cfg.training.device,
            proposal_batch_size=cfg.training.batch_size,
            method_name=f"{source_name}-RTSTPFExact-Trained",
        )
        face_counts = sorted(asset.face_count for asset in assets_by_source[source_name])
        source_results.append(
            HighDensityMeshSourceResult(
                source_name=source_name,
                asset_count=len(assets_by_source[source_name]),
                sample_count=len(pairs_by_source[source_name]),
                train_query_count=sum(1 for sample in train_dataset.samples if source_by_query_id[sample.query_id] == source_name),
                eval_query_count=len(query_ids),
                min_face_count=face_counts[0],
                median_face_count=face_counts[len(face_counts) // 2],
                max_face_count=face_counts[-1],
                no_proposal=no_proposal,
                random_stpf=random_stpf,
                trained_stpf=trained_stpf,
            )
        )

    checkpoint_path = str(training_run.artifacts.model_state_path)
    bench_cfg = CompleteExampleScaledTrainingBenchmarkConfig(
        training=cfg.training,
        run_name=cfg.run_name,
        training_output_dir=cfg.training_output_dir,
        benchmark_output_dir=cfg.benchmark_output_dir,
        proposal_batch_size=cfg.training.batch_size,
    )
    query_level_rows = _benchmark_dataset(bench_cfg, dataset=eval_dataset, checkpoint_path=checkpoint_path)

    output_root = Path(cfg.benchmark_output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    report_path = output_root / f"{cfg.run_name}.md"
    summary_json_path = output_root / f"{cfg.run_name}.json"
    result = HighDensityMeshBenchmarkResult(
        config=cfg,
        assets_by_source=assets_by_source,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        train_workload=train_workload,
        eval_workload=eval_workload,
        training_run=training_run,
        combined_no_proposal=combined_no_proposal,
        combined_random_stpf=combined_random_stpf,
        combined_trained_stpf=combined_trained_stpf,
        source_results=tuple(source_results),
        query_level_rows=query_level_rows,
        report_path=report_path,
        summary_json_path=summary_json_path,
    )
    _write_summary_json(summary_json_path, result)
    _write_report(report_path, result)
    return result


def _metric_dict(metric: HighDensityMethodMetrics) -> dict[str, object]:
    return asdict(metric)


def _reduction(trained: HighDensityMethodMetrics, baseline: HighDensityMethodMetrics) -> float:
    return 1.0 - trained.exact_work_units / max(1.0e-9, baseline.exact_work_units)


def _write_summary_json(path: Path, result: HighDensityMeshBenchmarkResult) -> None:
    payload = {
        "config": asdict(result.config),
        "checkpoint_path": str(result.training_run.artifacts.model_state_path),
        "train_query_count": len(result.train_dataset.samples),
        "eval_query_count": len(result.eval_dataset.samples),
        "dense_train_rows": len(result.train_workload.rows),
        "dense_eval_rows": len(result.eval_workload.rows),
        "combined_no_proposal": _metric_dict(result.combined_no_proposal),
        "combined_random_stpf": _metric_dict(result.combined_random_stpf),
        "combined_trained_stpf": _metric_dict(result.combined_trained_stpf),
        "combined_exact_work_reduction_vs_no_proposal": _reduction(
            result.combined_trained_stpf,
            result.combined_no_proposal,
        ),
        "source_results": [
            {
                "source_name": item.source_name,
                "asset_count": item.asset_count,
                "sample_count": item.sample_count,
                "train_query_count": item.train_query_count,
                "eval_query_count": item.eval_query_count,
                "min_face_count": item.min_face_count,
                "median_face_count": item.median_face_count,
                "max_face_count": item.max_face_count,
                "no_proposal": _metric_dict(item.no_proposal),
                "random_stpf": _metric_dict(item.random_stpf),
                "trained_stpf": _metric_dict(item.trained_stpf),
                "exact_work_reduction_vs_no_proposal": _reduction(item.trained_stpf, item.no_proposal),
            }
            for item in result.source_results
        ],
        "query_level_rows": [asdict(row) for row in result.query_level_rows],
        "assets_by_source": {
            source_name: [asdict(asset) for asset in assets]
            for source_name, assets in result.assets_by_source.items()
        },
        "final_train_loss": result.training_run.final_train_loss,
        "final_validation_loss": result.training_run.final_validation_loss,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _fmt(value: float, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def _pct(value: float) -> str:
    return f"{100.0 * float(value):.2f}%"


def _write_report(path: Path, result: HighDensityMeshBenchmarkResult) -> None:
    lines = [
        "# High-density mesh STPF training and comparison",
        "",
        "## Protocol",
        "",
        "- data source: `ABC official`, `Fusion 360 Gallery Assembly`, `Thingi10K official subset`. ",
        "- Mesh select: eachdata sourceby face count / vertex count selecthigh-density mesh. ",
        "- description: description mesh pair description proxy CCD query, description `8 slabs x 4 patches x 4 patches = 128 candidates/query`  dense workload. ",
        "- exact work statistics: by mesh pair face count perform primitive-weighted scale, used fordescriptionhigh-density mesh  exact work description. ",
        "- RTSTPFExact constraint: descriptionandcomparedescriptionuse learned STPF, descriptionuse dummy policy. ",
        "",
        "## description",
        "",
        f"- Train queries: `{len(result.train_dataset.samples)}`",
        f"- Eval queries: `{len(result.eval_dataset.samples)}`",
        f"- Dense train rows: `{len(result.train_workload.rows)}`",
        f"- Dense eval rows: `{len(result.eval_workload.rows)}`",
        f"- Dataset shards: `{Path(result.config.shard_root) / result.config.run_name}`",
        f"- Checkpoint: `{result.training_run.artifacts.model_state_path}`",
        f"- Final train loss: `{result.training_run.final_train_loss:.6f}`",
        f"- Final validation loss: `{result.training_run.final_validation_loss:.6f}`",
        "",
        "## Per-Source Mesh coverage",
        "",
        "| Source | Assets | Samples | Train queries | Eval queries | Face min | Face median | Face max |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in result.source_results:
        lines.append(
            f"| `{item.source_name}` | `{item.asset_count}` | `{item.sample_count}` | `{item.train_query_count}` | "
            f"`{item.eval_query_count}` | `{item.min_face_count}` | `{item.median_face_count}` | `{item.max_face_count}` |"
        )
    lines.extend(
        [
            "",
            "## Dense Workload Exact-Work compare",
            "",
            "| Scope | NoProposal calls | Trained calls | NoProposal work | Trained work | Reduction | Random work | STPF proposal ms | STPF total ms | FN |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    combined = (
        "Combined",
        result.combined_no_proposal,
        result.combined_random_stpf,
        result.combined_trained_stpf,
    )
    rows = [combined] + [
        (item.source_name, item.no_proposal, item.random_stpf, item.trained_stpf)
        for item in result.source_results
    ]
    for scope, baseline, random_stpf, trained in rows:
        lines.append(
            f"| `{scope}` | `{baseline.exact_call_count}` | `{trained.exact_call_count}` | "
            f"`{_fmt(baseline.exact_work_units, 1)}` | `{_fmt(trained.exact_work_units, 1)}` | "
            f"`{_pct(_reduction(trained, baseline))}` | `{_fmt(random_stpf.exact_work_units, 1)}` | "
            f"`{_fmt(trained.proposal_wall_ms, 3)}` | `{_fmt(trained.total_wall_ms, 3)}` | `{trained.fn_count}` |"
        )
    lines.extend(
        [
            "",
            "## Query-Level descriptionMethodcompare",
            "",
            "| Method | Total ms | RT ms | Proposal ms | Exact ms | QPS | FN | Recall | Avg candidates | Avg exact evals |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in result.query_level_rows:
        lines.append(
            f"| `{row.method}` | `{_fmt(row.total_ms, 3)}` | `{_fmt(row.rt_ms, 3)}` | `{_fmt(row.proposal_ms, 3)}` | "
            f"`{_fmt(row.exact_ms, 3)}` | `{_fmt(row.qps, 1)}` | `{row.fn_count}` | `{_fmt(row.candidate_recall, 4)}` | "
            f"`{_fmt(row.avg_candidates, 3)}` | `{_fmt(row.avg_exact_evals, 3)}` |"
        )
    fastest = min(result.query_level_rows, key=lambda row: row.total_ms)
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            f"- Dense workload on trained STPF description NoProposal  combined primitive-weighted exact work reduction as `{_pct(_reduction(result.combined_trained_stpf, result.combined_no_proposal))}`. ",
            f"- Query-level descriptionMethodindescriptionis `{fastest.method}`, descriptionwhen `{_fmt(fastest.total_ms, 3)} ms`. ",
            "- this benchmark usedescriptionhigh-density mesh descriptiondistributionand proposal description exact work reduction; ifdescriptionreal mesh-mesh exact wall time, descriptionconnectdescriptionreal primitive-level exact hot path. ",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


__all__ = [
    "HighDensityMeshBenchmarkConfig",
    "HighDensityMeshBenchmarkResult",
    "HighDensityMeshSourceResult",
    "MeshDensityAsset",
    "MeshDensityPair",
    "run_high_density_mesh_training_benchmark",
]
