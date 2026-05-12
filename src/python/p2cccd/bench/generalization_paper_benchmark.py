from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import random
from pathlib import Path
from typing import Sequence

import numpy as np

from p2cccd.data import DatasetGenerationConfig, GeneratedDataset, default_metadata, generate_exact_oracle_dataset, write_npz_shard
from p2cccd.datasets.cad import ABCProxyDatasetConfig, default_abc_official_root, generate_abc_proxy_datasets
from p2cccd.datasets.objects.thingi10k_training import (
    Thingi10KOfficialSubsetConfig,
    Thingi10KProxyDatasetConfig,
    generate_thingi10k_proxy_datasets,
)
from p2cccd.proposal.features import PROPOSAL_FAMILY_COUNT, PROPOSAL_FEATURE_DIM, PROPOSAL_INTERVAL_BIN_COUNT, ProposalFeatureRow, validate_proposal_feature_row
from p2cccd.proposal.stpf_model import STPFModelPreset, build_stpf_model
from p2cccd.proposal.training import STPFEpochMetrics, STPFTrainingConfig, evaluate_stpf_model
from p2cccd.proposal.training_runner import STPFTrainingRunConfig, STPFTrainingRunResult, run_stpf_training

from .complete_example_scaled_training_benchmark import CompleteExampleScaledTrainingBenchmarkConfig, _MethodRow, _benchmark_dataset
from .high_density_mesh_training_benchmark import (
    _dataset_from_samples as _high_density_mesh_dataset_from_samples,
    _load_abc_assets,
    _load_fusion360_assets,
    _load_thingi10k_assets,
    _make_pairs as _make_high_density_mesh_pairs,
    _sample_from_pair as _sample_from_high_density_mesh_pair,
    _scale_workload_costs,
)
from .third_party_dataset_training import ThirdPartyTrainingConfig, _split_dataset, dataset_from_fusion360
from .trained_stpf_high_density import (
    HighDensityMethodMetrics,
    HighDensitySTPFConfig,
    HighDensitySTPFWorkload,
    benchmark_no_proposal_on_high_density_workload,
    benchmark_stpf_on_high_density_workload,
    build_high_density_stpf_workload,
    workload_to_shard_dataset,
)


def _default_training() -> STPFTrainingConfig:
    return STPFTrainingConfig(
        epochs=4,
        batch_size=8192,
        learning_rate=8.0e-4,
        seed=424242,
        device="cuda",
        validation_fraction=0.0,
        model_preset=STPFModelPreset.MEDIUM_MLP,
    )


@dataclass(frozen=True, slots=True)
class GeneralizationPaperBenchmarkConfig:
    run_name: str = "generalization_paper_benchmark_run_id"
    seed: int = 424242
    training: STPFTrainingConfig = _default_training()
    high_density: HighDensitySTPFConfig = HighDensitySTPFConfig(
        slab_count=8,
        patches_per_object=4,
        representative_attempt_limit=3,
        uncertainty_fallback_threshold=0.75,
        narrow_interval_min_cost_scale=0.18,
        interval_miss_penalty_scale=0.22,
        full_exact_cost_scale=1.0,
    )
    max_train_rows_per_source: int = 150_000
    max_validation_rows_per_source: int = 30_000
    benchmark_output_dir: str = "src/benchmark"
    training_output_dir: str = "src/outputs/stpf_training"
    shard_root: str = "src/datasets/training/generalization/shards"
    model_device: str = "cuda"
    rt_backend_name: str = "optix_rt"
    proposal_batch_size: int = 8192
    t0_train_mesh_count_per_split: int = 560
    t0_train_robot_link_count: int = 360
    t0_eval_mesh_count_per_split: int = 280
    t0_eval_robot_link_count: int = 180
    high_density_train_mesh_count_per_split: int = 520
    high_density_eval_mesh_count_per_split: int = 260
    abc_asset_limit: int = 128
    abc_pair_limit: int = 2400
    thingi10k_asset_limit: int = 96
    thingi10k_train_pair_limit: int = 1200
    thingi10k_eval_pair_limit: int = 600
    fusion360_sample_limit: int = 2400
    high_density_mesh_asset_limit_per_source: int = 96
    high_density_mesh_samples_per_source: int = 1200


@dataclass(frozen=True, slots=True)
class GeneralizationSourcePack:
    name: str
    source_name: str
    train_dataset: GeneratedDataset
    eval_dataset: GeneratedDataset
    train_workload: HighDensitySTPFWorkload
    eval_workload: HighDensitySTPFWorkload
    note: str


@dataclass(frozen=True, slots=True)
class GeneralizationDenseRow:
    example_name: str
    no_proposal: HighDensityMethodMetrics
    random_stpf: HighDensityMethodMetrics
    trained_stpf: HighDensityMethodMetrics


@dataclass(frozen=True, slots=True)
class GeneralizationQueryRow:
    example_name: str
    rows: tuple[_MethodRow, ...]


@dataclass(frozen=True, slots=True)
class GeneralizationPaperBenchmarkResult:
    config: GeneralizationPaperBenchmarkConfig
    sources: tuple[GeneralizationSourcePack, ...]
    train_row_count: int
    validation_row_count: int
    selected_train_rows_by_source: dict[str, int]
    selected_validation_rows_by_source: dict[str, int]
    training_run: STPFTrainingRunResult
    validation_metrics_by_source: tuple[STPFEpochMetrics, ...]
    dense_rows: tuple[GeneralizationDenseRow, ...]
    query_rows: tuple[GeneralizationQueryRow, ...]
    shard_dir: Path
    train_rows_path: Path
    validation_rows_path: Path
    report_path: Path
    summary_json_path: Path


def _subset_dataset(dataset: GeneratedDataset, count: int) -> GeneratedDataset:
    if count <= 0 or count >= len(dataset.samples):
        return dataset
    return GeneratedDataset(
        rows=list(dataset.rows[:count]),
        samples=list(dataset.samples[:count]),
        traces=list(dataset.traces[:count]),
        split_names=dataset.split_names,
    )


def _sample_rows(rows: Sequence[ProposalFeatureRow], limit: int, seed: int) -> list[ProposalFeatureRow]:
    row_list = list(rows)
    if limit <= 0 or len(row_list) <= limit:
        return row_list
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(row_list)), limit))
    return [row_list[index] for index in indices]


def _rows_only_npz(path: Path, rows: Sequence[ProposalFeatureRow], *, metadata: dict[str, object]) -> None:
    row_list = [validate_proposal_feature_row(row) for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    ids = np.asarray(
        [
            [
                row.schema_version,
                row.query_id,
                row.candidate_id,
                row.slab_id,
                row.object_a_id,
                row.patch_a_id,
                row.object_b_id,
                row.patch_b_id,
                row.target_mask,
            ]
            for row in row_list
        ],
        dtype=np.uint64,
    )
    arrays = {
        "ids": ids,
        "features": np.asarray([row.features for row in row_list], dtype=np.float32).reshape(len(row_list), PROPOSAL_FEATURE_DIM),
        "interval_targets": np.asarray([row.interval_targets for row in row_list], dtype=np.float32).reshape(len(row_list), PROPOSAL_INTERVAL_BIN_COUNT),
        "family_targets": np.asarray([row.family_targets for row in row_list], dtype=np.float32).reshape(len(row_list), PROPOSAL_FAMILY_COUNT),
        "scalar_targets": np.asarray(
            [[row.priority_target, row.cost_target, row.uncertainty_target] for row in row_list],
            dtype=np.float32,
        ).reshape(len(row_list), 3),
        "metadata_json": np.asarray(
            json.dumps({**metadata, "row_count": len(row_list)}, sort_keys=True),
            dtype=np.str_,
        ),
    }
    np.savez_compressed(path, **arrays)


def _write_source_shards(shard_dir: Path, pack: GeneralizationSourcePack, seed: int) -> None:
    source_dir = shard_dir / pack.name.lower().replace(" ", "_").replace("/", "_")
    source_dir.mkdir(parents=True, exist_ok=True)
    write_npz_shard(
        source_dir / "base_train.npz",
        pack.train_dataset,
        metadata={**default_metadata(pack.train_dataset, seed=seed, source=pack.name), "dataset_role": "base_train"},
    )
    write_npz_shard(
        source_dir / "base_eval.npz",
        pack.eval_dataset,
        metadata={**default_metadata(pack.eval_dataset, seed=seed + 1, source=pack.name), "dataset_role": "base_eval"},
    )
    write_npz_shard(
        source_dir / "dense_train.npz",
        workload_to_shard_dataset(pack.train_workload),
        metadata={
            **default_metadata(workload_to_shard_dataset(pack.train_workload), seed=seed, source=pack.name),
            "dataset_role": "dense_train",
        },
    )
    write_npz_shard(
        source_dir / "dense_eval.npz",
        workload_to_shard_dataset(pack.eval_workload),
        metadata={
            **default_metadata(workload_to_shard_dataset(pack.eval_workload), seed=seed + 1, source=pack.name),
            "dataset_role": "dense_eval",
        },
    )


def _pack_from_datasets(
    *,
    name: str,
    source_name: str,
    train_dataset: GeneratedDataset,
    eval_dataset: GeneratedDataset,
    high_density: HighDensitySTPFConfig,
    note: str,
) -> GeneralizationSourcePack:
    return GeneralizationSourcePack(
        name=name,
        source_name=source_name,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        train_workload=build_high_density_stpf_workload(train_dataset, high_density, name=f"{name}_dense_train"),
        eval_workload=build_high_density_stpf_workload(eval_dataset, high_density, name=f"{name}_dense_eval"),
        note=note,
    )


def _build_t0_pack(cfg: GeneralizationPaperBenchmarkConfig) -> GeneralizationSourcePack:
    train_dataset = generate_exact_oracle_dataset(
        DatasetGenerationConfig(
            mesh_count_per_split=cfg.t0_train_mesh_count_per_split,
            robot_link_count=cfg.t0_train_robot_link_count,
            seed=cfg.seed + 11,
            include_robot_links=True,
        )
    )
    eval_dataset = generate_exact_oracle_dataset(
        DatasetGenerationConfig(
            mesh_count_per_split=cfg.t0_eval_mesh_count_per_split,
            robot_link_count=cfg.t0_eval_robot_link_count,
            seed=cfg.seed + 12,
            include_robot_links=True,
        )
    )
    return _pack_from_datasets(
        name="T0 synthetic_proxy",
        source_name="analytic_swept_sphere_proxy",
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        high_density=cfg.high_density,
        note="Procedural analytic proxy data with mesh-pair and robot-link variants.",
    )


def _build_synthetic_high_density_pack(cfg: GeneralizationPaperBenchmarkConfig) -> GeneralizationSourcePack:
    train_dataset = generate_exact_oracle_dataset(
        DatasetGenerationConfig(
            mesh_count_per_split=cfg.high_density_train_mesh_count_per_split,
            robot_link_count=0,
            seed=cfg.seed + 21,
            include_robot_links=False,
        )
    )
    eval_dataset = generate_exact_oracle_dataset(
        DatasetGenerationConfig(
            mesh_count_per_split=cfg.high_density_eval_mesh_count_per_split,
            robot_link_count=0,
            seed=cfg.seed + 22,
            include_robot_links=False,
        )
    )
    return _pack_from_datasets(
        name="trained_stpf_high_density",
        source_name="analytic_high_candidate_density",
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        high_density=cfg.high_density,
        note="Synthetic hard-case distribution focused on dense candidate inflation.",
    )


def _build_abc_pack(cfg: GeneralizationPaperBenchmarkConfig) -> GeneralizationSourcePack:
    bundle = generate_abc_proxy_datasets(
        ABCProxyDatasetConfig(
            root=default_abc_official_root(),
            allow_demo_bootstrap=False,
            asset_limit=cfg.abc_asset_limit,
            pair_limit=cfg.abc_pair_limit,
            train_fraction=0.70,
            seed=cfg.seed + 31,
        )
    )
    return _pack_from_datasets(
        name="ABC CAD",
        source_name="ABC official CAD",
        train_dataset=bundle.train_dataset,
        eval_dataset=bundle.eval_dataset,
        high_density=cfg.high_density,
        note=f"Official ABC proxy CAD pairs; assets={len(bundle.assets)}, train_pairs={len(bundle.train_pairs)}, eval_pairs={len(bundle.eval_pairs)}.",
    )


def _build_thingi10k_pack(cfg: GeneralizationPaperBenchmarkConfig) -> GeneralizationSourcePack:
    bundle = generate_thingi10k_proxy_datasets(
        Thingi10KProxyDatasetConfig(
            subset=Thingi10KOfficialSubsetConfig(asset_limit=cfg.thingi10k_asset_limit),
            train_fraction=0.70,
            train_pair_limit=cfg.thingi10k_train_pair_limit,
            eval_pair_limit=cfg.thingi10k_eval_pair_limit,
            seed=cfg.seed + 41,
        )
    )
    return _pack_from_datasets(
        name="Thingi10K",
        source_name="Thingi10K official subset",
        train_dataset=bundle.train_dataset,
        eval_dataset=bundle.eval_dataset,
        high_density=cfg.high_density,
        note=f"Dirty/OOD mesh proxy data; assets={len(bundle.assets)}, train_pairs={len(bundle.train_pairs)}, eval_pairs={len(bundle.eval_pairs)}.",
    )


def _build_fusion360_pack(cfg: GeneralizationPaperBenchmarkConfig) -> GeneralizationSourcePack:
    dataset, stats = dataset_from_fusion360(
        ThirdPartyTrainingConfig(
            source_name="fusion360",
            root="src/datasets/fusion360",
            run_name=f"{cfg.run_name}_fusion360_source",
            sample_limit=cfg.fusion360_sample_limit,
            train_fraction=0.80,
            seed=cfg.seed + 51,
            training=cfg.training,
        )
    )
    train_dataset, eval_dataset = _split_dataset(dataset, 0.80, cfg.seed + 52)
    return _pack_from_datasets(
        name="Fusion 360 Gallery Assembly",
        source_name="Fusion 360 Gallery official assembly",
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        high_density=cfg.high_density,
        note=f"Official Fusion assembly motion proxy; sequences={stats.get('sequence_count')}, mesh_assets={stats.get('mesh_asset_count')}.",
    )


def _build_high_density_mesh_pack(cfg: GeneralizationPaperBenchmarkConfig) -> GeneralizationSourcePack:
    assets_by_source = {
        "ABC official": _load_abc_assets(Path("src/datasets/abc_official"), cfg.high_density_mesh_asset_limit_per_source),
        "Fusion 360 Gallery": _load_fusion360_assets(Path("src/datasets/fusion360"), cfg.high_density_mesh_asset_limit_per_source),
        "Thingi10K": _load_thingi10k_assets(Path("src/datasets/thingi10k"), cfg.high_density_mesh_asset_limit_per_source),
    }
    samples = []
    source_by_query_id: dict[int, str] = {}
    cost_scale_by_query_id: dict[int, float] = {}
    sample_id = 1
    for source_name, assets in assets_by_source.items():
        pairs = _make_high_density_mesh_pairs(assets, cfg.high_density_mesh_samples_per_source)
        for index, pair in enumerate(pairs):
            sample = _sample_from_high_density_mesh_pair(pair, sample_id=sample_id, variant_index=index)
            samples.append(sample)
            source_by_query_id[sample.query_id] = source_name
            cost_scale_by_query_id[sample.query_id] = pair.cost_scale
            sample_id += 1
    dataset = _high_density_mesh_dataset_from_samples(samples)
    train_ids: set[int] = set()
    eval_ids: set[int] = set()
    rng = random.Random(cfg.seed + 61)
    indices_by_source: dict[str, list[int]] = {}
    for index, sample in enumerate(dataset.samples):
        indices_by_source.setdefault(source_by_query_id[sample.query_id], []).append(index)
    for indices in indices_by_source.values():
        shuffled = list(indices)
        rng.shuffle(shuffled)
        split = max(1, min(len(shuffled) - 1, int(round(len(shuffled) * 0.75))))
        train_ids.update(shuffled[:split])
        eval_ids.update(shuffled[split:])
    train_dataset = GeneratedDataset(
        rows=[row for index, row in enumerate(dataset.rows) if index in train_ids],
        samples=[sample for index, sample in enumerate(dataset.samples) if index in train_ids],
        traces=[trace for index, trace in enumerate(dataset.traces) if index in train_ids],
        split_names=dataset.split_names,
    )
    eval_dataset = GeneratedDataset(
        rows=[row for index, row in enumerate(dataset.rows) if index in eval_ids],
        samples=[sample for index, sample in enumerate(dataset.samples) if index in eval_ids],
        traces=[trace for index, trace in enumerate(dataset.traces) if index in eval_ids],
        split_names=dataset.split_names,
    )
    train_workload = _scale_workload_costs(
        build_high_density_stpf_workload(train_dataset, cfg.high_density, name="high_density_mesh_train"),
        cost_scale_by_query_id,
    )
    eval_workload = _scale_workload_costs(
        build_high_density_stpf_workload(eval_dataset, cfg.high_density, name="high_density_mesh_eval"),
        cost_scale_by_query_id,
    )
    return GeneralizationSourcePack(
        name="high_density_mesh_multi_source",
        source_name="ABC official + Fusion 360 + Thingi10K high-density mesh",
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        train_workload=train_workload,
        eval_workload=eval_workload,
        note="High face-count mesh-pair proxy queries with primitive-weighted dense exact-work costs.",
    )


def _build_sources(cfg: GeneralizationPaperBenchmarkConfig) -> tuple[GeneralizationSourcePack, ...]:
    return (
        _build_t0_pack(cfg),
        _build_synthetic_high_density_pack(cfg),
        _build_abc_pack(cfg),
        _build_thingi10k_pack(cfg),
        _build_fusion360_pack(cfg),
        _build_high_density_mesh_pack(cfg),
    )


def _dense_reduction(trained: HighDensityMethodMetrics, baseline: HighDensityMethodMetrics) -> float:
    return 1.0 - trained.exact_work_units / max(1.0e-9, baseline.exact_work_units)


def _benchmark_dense_sources(
    sources: Sequence[GeneralizationSourcePack],
    *,
    model,
    random_model,
    device: str,
    batch_size: int,
) -> tuple[GeneralizationDenseRow, ...]:
    rows: list[GeneralizationDenseRow] = []
    for pack in sources:
        no_proposal = benchmark_no_proposal_on_high_density_workload(pack.eval_workload)
        random_stpf = benchmark_stpf_on_high_density_workload(
            pack.eval_workload,
            model=random_model,
            device=device,
            proposal_batch_size=batch_size,
            method_name=f"{pack.name}-RTSTPFExact-Random",
        )
        trained_stpf = benchmark_stpf_on_high_density_workload(
            pack.eval_workload,
            model=model,
            device=device,
            proposal_batch_size=batch_size,
            method_name=f"{pack.name}-RTSTPFExact-Generalized",
        )
        rows.append(GeneralizationDenseRow(pack.name, no_proposal, random_stpf, trained_stpf))
    return tuple(rows)


def _benchmark_query_sources(
    cfg: GeneralizationPaperBenchmarkConfig,
    sources: Sequence[GeneralizationSourcePack],
    *,
    checkpoint_path: str,
) -> tuple[GeneralizationQueryRow, ...]:
    bench_cfg = CompleteExampleScaledTrainingBenchmarkConfig(
        training=cfg.training,
        rt_backend_name=cfg.rt_backend_name,
        model_device=cfg.model_device,
        proposal_batch_size=cfg.proposal_batch_size,
        training_output_dir=cfg.training_output_dir,
        benchmark_output_dir=cfg.benchmark_output_dir,
        run_name=cfg.run_name,
    )
    rows: list[GeneralizationQueryRow] = []
    for pack in sources:
        rows.append(GeneralizationQueryRow(pack.name, _benchmark_dataset(bench_cfg, dataset=pack.eval_dataset, checkpoint_path=checkpoint_path)))
    return tuple(rows)


def _write_report(path: Path, result: GeneralizationPaperBenchmarkResult) -> None:
    checkpoint = result.training_run.artifacts.model_state_path
    final_validation = next((item for item in reversed(result.training_run.result.history) if item.split == "validation"), None)
    lines = [
        "# Generalization-enhanced large dataset training and paper benchmark",
        "",
        "## Protocol",
        "",
        "- Objective: descriptiondistribution, insteaddescription T0 synthetic, synthetic high-density, ABC CAD, Thingi10K, Fusion 360 andhigh-density mesh multi-source. ",
        "- RTSTPFExact constraint: descriptionuse learned STPF description, descriptionuse dummy policy. ",
        "- defaultdescriptionPath: `medium_mlp + ORT(TensorRT EP description) + optix_rt + cuda_exact`. ",
        "- Dense exact-work description proposal isdescriptionreduction primitive/candidate level exact work; Query-level descriptioncurrent runner descriptiontodescription wall time. ",
        "",
        "## description",
        "",
        f"- Shard dir: `{result.shard_dir}`",
        f"- Selected train rows: `{result.train_row_count}`",
        f"- Selected validation rows: `{result.validation_row_count}`",
        f"- Checkpoint: `{checkpoint}`",
        f"- Final train loss: `{result.training_run.final_train_loss:.6f}`",
        f"- Final validation loss: `{result.training_run.final_validation_loss:.6f}`",
    ]
    if final_validation is not None:
        lines.extend(
            [
                f"- Validation interval top1 recall: `{final_validation.interval_top1_recall:.4f}`",
                f"- Validation family top2 recall: `{final_validation.family_top2_recall:.4f}`",
                f"- Validation estimated exact-work reduction: `{final_validation.estimated_exact_work_reduction:.4f}`",
            ]
        )
    lines.extend(
        [
            "",
            "## data coverage",
            "",
            "| Example | Source | Train queries | Eval queries | Dense train rows | Dense eval rows | Selected train rows | Selected validation rows | Note |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for pack in result.sources:
        lines.append(
            f"| `{pack.name}` | `{pack.source_name}` | `{len(pack.train_dataset.samples)}` | `{len(pack.eval_dataset.samples)}` | "
            f"`{len(pack.train_workload.rows)}` | `{len(pack.eval_workload.rows)}` | "
            f"`{result.selected_train_rows_by_source.get(pack.name, 0)}` | `{result.selected_validation_rows_by_source.get(pack.name, 0)}` | {pack.note} |"
        )
    lines.extend(
        [
            "",
            "## Dense Exact-Work Reduction",
            "",
            "| Example | NoProposal calls | STPF calls | NoProposal work | STPF work | Reduction | Random STPF work | STPF proposal ms | STPF total ms | FN |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in result.dense_rows:
        lines.append(
            f"| `{row.example_name}` | `{row.no_proposal.exact_call_count}` | `{row.trained_stpf.exact_call_count}` | "
            f"`{row.no_proposal.exact_work_units:.1f}` | `{row.trained_stpf.exact_work_units:.1f}` | "
            f"`{100.0 * _dense_reduction(row.trained_stpf, row.no_proposal):.2f}%` | `{row.random_stpf.exact_work_units:.1f}` | "
            f"`{row.trained_stpf.proposal_wall_ms:.3f}` | `{row.trained_stpf.total_wall_ms:.3f}` | `{row.trained_stpf.fn_count}` |"
        )
    lines.extend(
        [
            "",
            "## Query-Level descriptionMethod Benchmark",
            "",
        ]
    )
    for query_group in result.query_rows:
        lines.extend(
            [
                f"### {query_group.example_name}",
                "",
                "| Method | Total(ms) | RT(ms) | Proposal(ms) | Exact(ms) | QPS | FN | Recall | AvgCandidates | AvgExactEvals | Backend |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for row in query_group.rows:
            backend = row.exact_backend_name
            if row.method == "RTSTPFExact":
                backend = f"{backend}; {row.inference_backend_name}/{row.inference_provider_name}; {row.resolved_execution_profile_name}"
            lines.append(
                f"| `{row.method}` | `{row.total_ms:.4f}` | `{row.rt_ms:.4f}` | `{row.proposal_ms:.4f}` | `{row.exact_ms:.4f}` | "
                f"`{row.qps:.2f}` | `{row.fn_count}` | `{row.candidate_recall:.4f}` | `{row.avg_candidates:.4f}` | `{row.avg_exact_evals:.4f}` | `{backend}` |"
            )
        fastest = min(query_group.rows, key=lambda item: item.total_ms)
        rtstpf = next(item for item in query_group.rows if item.method == "RTSTPFExact")
        no_proposal = next(item for item in query_group.rows if item.method == "NoProposal")
        lines.extend(
            [
                "",
                f"- Fastest: `{fastest.method}` (`{fastest.total_ms:.4f} ms`).",
                f"- RTSTPFExact vs NoProposal wall-time delta: `{rtstpf.total_ms - no_proposal.total_ms:.4f} ms`.",
                "",
            ]
        )
    total_fn = sum(row.trained_stpf.fn_count for row in result.dense_rows) + sum(
        method_row.fn_count for group in result.query_rows for method_row in group.rows
    )
    min_recall = min(method_row.candidate_recall for group in result.query_rows for method_row in group.rows)
    avg_dense_reduction = sum(_dense_reduction(row.trained_stpf, row.no_proposal) for row in result.dense_rows) / max(1, len(result.dense_rows))
    lines.extend(
        [
            "## Conclusion",
            "",
            f"- Correctness: dense + query-level description FN as `{total_fn}`; query-level description recall as `{min_recall:.4f}`. ",
            f"- generalizationdescriptionafter, dense exact-work descriptionreduction `{100.0 * avg_dense_reduction:.2f}%`. ",
            "- currentadvantagedescriptionpositiondescriptionishigh candidate-density / high exact-work scene; ordinary sparse query-level wall time description NoProposal or PureExactCPU description. ",
            "- underdescriptionifdescription wall time descriptionadvantage, descriptionreal primitive-level mesh exact hot path and proposal filtering description, rather thandescriptionin proxy dense workload ondescription exact-work reduction. ",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_json(path: Path, result: GeneralizationPaperBenchmarkResult) -> None:
    payload = {
        "config": asdict(result.config),
        "shard_dir": str(result.shard_dir),
        "train_rows_path": str(result.train_rows_path),
        "validation_rows_path": str(result.validation_rows_path),
        "checkpoint_path": str(result.training_run.artifacts.model_state_path),
        "train_row_count": result.train_row_count,
        "validation_row_count": result.validation_row_count,
        "selected_train_rows_by_source": result.selected_train_rows_by_source,
        "selected_validation_rows_by_source": result.selected_validation_rows_by_source,
        "final_train_loss": result.training_run.final_train_loss,
        "final_validation_loss": result.training_run.final_validation_loss,
        "sources": [
            {
                "name": pack.name,
                "source_name": pack.source_name,
                "train_query_count": len(pack.train_dataset.samples),
                "eval_query_count": len(pack.eval_dataset.samples),
                "dense_train_rows": len(pack.train_workload.rows),
                "dense_eval_rows": len(pack.eval_workload.rows),
                "note": pack.note,
            }
            for pack in result.sources
        ],
        "validation_metrics_by_source": [asdict(metric) for metric in result.validation_metrics_by_source],
        "dense_rows": [
            {
                "example_name": row.example_name,
                "no_proposal": asdict(row.no_proposal),
                "random_stpf": asdict(row.random_stpf),
                "trained_stpf": asdict(row.trained_stpf),
                "reduction_vs_no_proposal": _dense_reduction(row.trained_stpf, row.no_proposal),
            }
            for row in result.dense_rows
        ],
        "query_rows": [
            {
                "example_name": group.example_name,
                "rows": [asdict(row) for row in group.rows],
            }
            for group in result.query_rows
        ],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_generalization_paper_benchmark(
    config: GeneralizationPaperBenchmarkConfig | None = None,
) -> GeneralizationPaperBenchmarkResult:
    cfg = config or GeneralizationPaperBenchmarkConfig()
    sources = _build_sources(cfg)
    shard_dir = Path(cfg.shard_root) / cfg.run_name
    shard_dir.mkdir(parents=True, exist_ok=True)

    selected_train_rows_by_source: dict[str, int] = {}
    selected_validation_rows_by_source: dict[str, int] = {}
    train_rows: list[ProposalFeatureRow] = []
    validation_rows: list[ProposalFeatureRow] = []
    for source_index, pack in enumerate(sources):
        _write_source_shards(shard_dir, pack, cfg.seed + source_index * 100)
        source_train_rows = list(pack.train_dataset.rows) + list(pack.train_workload.rows)
        source_validation_rows = list(pack.eval_dataset.rows) + list(pack.eval_workload.rows)
        selected_train = _sample_rows(source_train_rows, cfg.max_train_rows_per_source, cfg.seed + source_index)
        selected_validation = _sample_rows(source_validation_rows, cfg.max_validation_rows_per_source, cfg.seed + 1000 + source_index)
        selected_train_rows_by_source[pack.name] = len(selected_train)
        selected_validation_rows_by_source[pack.name] = len(selected_validation)
        train_rows.extend(selected_train)
        validation_rows.extend(selected_validation)

    train_rows_path = shard_dir / "combined_train_rows.npz"
    validation_rows_path = shard_dir / "combined_validation_rows.npz"
    _rows_only_npz(
        train_rows_path,
        train_rows,
        metadata={"schema_version": 1, "source": "generalization_mixed_train", "seed": cfg.seed},
    )
    _rows_only_npz(
        validation_rows_path,
        validation_rows,
        metadata={"schema_version": 1, "source": "generalization_mixed_validation", "seed": cfg.seed + 1},
    )

    training_run = run_stpf_training(
        train_rows,
        STPFTrainingRunConfig(
            training=cfg.training,
            output_dir=cfg.training_output_dir,
            run_name=cfg.run_name,
        ),
        validation_rows=validation_rows,
    )
    trained_model = training_run.result.model
    trained_model.eval()
    trained_model.to(cfg.model_device)
    random_model = build_stpf_model(cfg.training.model_preset)
    random_model.eval()
    random_model.to(cfg.model_device)

    validation_metrics = tuple(
        evaluate_stpf_model(
            trained_model,
            _sample_rows(list(pack.eval_dataset.rows) + list(pack.eval_workload.rows), cfg.max_validation_rows_per_source, cfg.seed + 3000 + index),
            cfg.training,
            epoch=cfg.training.epochs,
            split=pack.name,
        )
        for index, pack in enumerate(sources)
    )
    dense_rows = _benchmark_dense_sources(
        sources,
        model=trained_model,
        random_model=random_model,
        device=cfg.model_device,
        batch_size=cfg.proposal_batch_size,
    )
    checkpoint_path = str(training_run.artifacts.model_state_path)
    query_rows = _benchmark_query_sources(cfg, sources, checkpoint_path=checkpoint_path)

    output_root = Path(cfg.benchmark_output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    report_path = output_root / f"{cfg.run_name}.md"
    summary_json_path = output_root / f"{cfg.run_name}.json"
    result = GeneralizationPaperBenchmarkResult(
        config=cfg,
        sources=sources,
        train_row_count=len(train_rows),
        validation_row_count=len(validation_rows),
        selected_train_rows_by_source=selected_train_rows_by_source,
        selected_validation_rows_by_source=selected_validation_rows_by_source,
        training_run=training_run,
        validation_metrics_by_source=validation_metrics,
        dense_rows=dense_rows,
        query_rows=query_rows,
        shard_dir=shard_dir,
        train_rows_path=train_rows_path,
        validation_rows_path=validation_rows_path,
        report_path=report_path,
        summary_json_path=summary_json_path,
    )
    _write_report(report_path, result)
    _write_json(summary_json_path, result)
    return result


__all__ = [
    "GeneralizationPaperBenchmarkConfig",
    "GeneralizationPaperBenchmarkResult",
    "run_generalization_paper_benchmark",
]
