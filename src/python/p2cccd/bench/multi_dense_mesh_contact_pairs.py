from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import random
from itertools import product
from pathlib import Path
from typing import Sequence

from p2cccd.data import GeneratedDataset, default_metadata, write_npz_shard
from p2cccd.datasets.cad import ABCDatasetAdapter
from p2cccd.proposal.stpf_model import STPFModelPreset, build_stpf_model
from p2cccd.proposal.training import STPFTrainingConfig
from p2cccd.proposal.training_runner import STPFTrainingRunConfig, STPFTrainingRunResult, run_stpf_training

from .high_density_mesh_training_benchmark import (
    MeshDensityAsset,
    MeshDensityPair,
    _asset_from_cad,
    _dataset_from_samples,
    _load_abc_assets,
    _load_fusion360_assets,
    _load_thingi10k_assets,
    _make_pairs,
    _pair_score,
    _cost_scale,
    _sample_from_pair,
    _scale_workload_costs,
    _subset_workload,
)
from .trained_stpf_high_density import (
    HighDensityMethodMetrics,
    HighDensitySTPFConfig,
    HighDensitySTPFWorkload,
    benchmark_no_proposal_on_high_density_workload,
    benchmark_stpf_on_high_density_workload,
    build_high_density_stpf_workload,
)


def _default_high_density() -> HighDensitySTPFConfig:
    return HighDensitySTPFConfig(
        slab_count=16,
        patches_per_object=8,
        representative_attempt_limit=3,
        uncertainty_fallback_threshold=0.75,
        narrow_interval_min_cost_scale=0.18,
        interval_miss_penalty_scale=0.22,
        full_exact_cost_scale=1.0,
    )


def _default_training() -> STPFTrainingConfig:
    return STPFTrainingConfig(
        epochs=5,
        batch_size=8192,
        learning_rate=8.0e-4,
        seed=424242,
        device="cuda",
        validation_fraction=0.0,
        model_preset=STPFModelPreset.MEDIUM_MLP,
    )


@dataclass(frozen=True, slots=True)
class MultiDenseMeshContactPairsConfig:
    abc_root: str = "src/datasets/abc_official"
    fusion360_root: str = "src/datasets/fusion360"
    thingi10k_root: str = "src/datasets/thingi10k"
    asset_limit_per_source: int = 96
    large_face_min: int = 10_000
    large_face_max: int = 50_000
    pair_limit_per_case: int = 48
    samples_per_pair: int = 8
    train_fraction: float = 0.75
    seed: int = 424242
    high_density: HighDensitySTPFConfig = field(default_factory=_default_high_density)
    training: STPFTrainingConfig = field(default_factory=_default_training)
    run_name: str = "multi_dense_mesh_contact_pairs_run_id"
    shard_root: str = "src/datasets/training/multi_dense_mesh_contact_pairs/shards"
    training_output_dir: str = "src/outputs/stpf_training"
    benchmark_output_dir: str = "src/benchmark"


@dataclass(frozen=True, slots=True)
class MultiDenseMeshCaseResult:
    case_name: str
    train_pair_count: int
    eval_pair_count: int
    train_query_count: int
    eval_query_count: int
    eval_candidate_count: int
    asset_count: int
    min_face_count: int
    median_face_count: int
    max_face_count: int
    no_proposal: HighDensityMethodMetrics
    random_stpf: HighDensityMethodMetrics
    trained_stpf: HighDensityMethodMetrics


@dataclass(frozen=True, slots=True)
class MultiDenseMeshContactPairsResult:
    config: MultiDenseMeshContactPairsConfig
    assets_by_source: dict[str, tuple[MeshDensityAsset, ...]]
    pairs_by_case: dict[str, tuple[MeshDensityPair, ...]]
    train_dataset: GeneratedDataset
    eval_dataset: GeneratedDataset
    train_workload: HighDensitySTPFWorkload
    eval_workload: HighDensitySTPFWorkload
    training_run: STPFTrainingRunResult
    combined_no_proposal: HighDensityMethodMetrics
    combined_random_stpf: HighDensityMethodMetrics
    combined_trained_stpf: HighDensityMethodMetrics
    case_results: tuple[MultiDenseMeshCaseResult, ...]
    report_path: Path
    summary_json_path: Path


def _load_large_face_abc_assets(config: MultiDenseMeshContactPairsConfig) -> tuple[MeshDensityAsset, ...]:
    adapter = ABCDatasetAdapter(Path(config.abc_root))
    assets = [
        _asset_from_cad("ABC large-face", asset)
        for asset in adapter.list_assets(limit=None)
        if config.large_face_min <= int(asset.stats.face_count) <= config.large_face_max
    ]
    assets.sort(key=lambda asset: (-asset.face_count, -asset.vertex_count, asset.asset_id))
    return tuple(assets[: config.asset_limit_per_source])


def _rename_assets(assets: Sequence[MeshDensityAsset], source_name: str) -> tuple[MeshDensityAsset, ...]:
    return tuple(
        MeshDensityAsset(
            source_name=source_name,
            asset_id=asset.asset_id,
            asset_path=asset.asset_path,
            face_count=asset.face_count,
            vertex_count=asset.vertex_count,
            diagonal=asset.diagonal,
            bounds_min=asset.bounds_min,
            bounds_max=asset.bounds_max,
            dirty_score=asset.dirty_score,
        )
        for asset in assets
    )


def _make_cross_pairs(
    case_name: str,
    assets_a: Sequence[MeshDensityAsset],
    assets_b: Sequence[MeshDensityAsset],
    *,
    limit: int,
) -> tuple[MeshDensityPair, ...]:
    pairs = [
        MeshDensityPair(
            source_name=case_name,
            asset_a=asset_a,
            asset_b=asset_b,
            pair_score=_pair_score(asset_a, asset_b),
            cost_scale=_cost_scale(asset_a, asset_b),
        )
        for asset_a, asset_b in product(assets_a, assets_b)
        if asset_a.asset_id != asset_b.asset_id or asset_a.asset_path != asset_b.asset_path
    ]
    pairs.sort(key=lambda pair: (-pair.pair_score, -pair.cost_scale, pair.asset_a.asset_id, pair.asset_b.asset_id))
    return tuple(pairs[:limit])


def _split_pairs(
    pairs: Sequence[MeshDensityPair],
    *,
    train_fraction: float,
    seed: int,
) -> tuple[tuple[MeshDensityPair, ...], tuple[MeshDensityPair, ...]]:
    shuffled = list(pairs)
    random.Random(seed).shuffle(shuffled)
    if len(shuffled) < 2:
        raise ValueError("each dense mesh case needs at least two pairs")
    train_count = max(1, min(len(shuffled) - 1, int(round(len(shuffled) * train_fraction))))
    return tuple(shuffled[:train_count]), tuple(shuffled[train_count:])


def _build_dataset_from_case_pairs(
    pairs_by_case: dict[str, tuple[MeshDensityPair, ...]],
    *,
    first_sample_id: int,
    samples_per_pair: int,
) -> tuple[GeneratedDataset, dict[int, str], dict[int, float]]:
    samples = []
    case_by_query_id: dict[int, str] = {}
    cost_scale_by_query_id: dict[int, float] = {}
    sample_id = first_sample_id
    variant_offset = 0
    for case_name, pairs in pairs_by_case.items():
        for pair_index, pair in enumerate(pairs):
            for local_index in range(samples_per_pair):
                sample = _sample_from_pair(
                    pair,
                    sample_id=sample_id,
                    variant_index=variant_offset + pair_index * samples_per_pair + local_index,
                )
                samples.append(sample)
                case_by_query_id[sample.query_id] = case_name
                cost_scale_by_query_id[sample.query_id] = pair.cost_scale
                sample_id += 1
        variant_offset += max(1, len(pairs) * samples_per_pair)
    return _dataset_from_samples(samples), case_by_query_id, cost_scale_by_query_id


def _unique_assets_from_pairs(pairs: Sequence[MeshDensityPair]) -> tuple[MeshDensityAsset, ...]:
    by_key: dict[str, MeshDensityAsset] = {}
    for pair in pairs:
        by_key[pair.asset_a.asset_path] = pair.asset_a
        by_key[pair.asset_b.asset_path] = pair.asset_b
    assets = list(by_key.values())
    assets.sort(key=lambda asset: (-asset.face_count, asset.source_name, asset.asset_id))
    return tuple(assets)


def _reduction(trained: HighDensityMethodMetrics, baseline: HighDensityMethodMetrics) -> float:
    return 1.0 - trained.exact_work_units / max(1.0e-9, baseline.exact_work_units)


def _pct(value: float) -> str:
    return f"{100.0 * float(value):.2f}%"


def _fmt(value: float, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def _metric_dict(metric: HighDensityMethodMetrics) -> dict[str, object]:
    return asdict(metric)


def _collision_query_count(workload: HighDensitySTPFWorkload) -> int:
    return sum(1 for trace in workload.traces_by_query_id.values() if trace.collided)


def _collision_candidate_count(workload: HighDensitySTPFWorkload) -> int:
    return sum(1 for info in workload.candidate_infos.values() if info.slab_overlap_contact)


def run_multi_dense_mesh_contact_pairs_benchmark(
    config: MultiDenseMeshContactPairsConfig | None = None,
) -> MultiDenseMeshContactPairsResult:
    cfg = config or MultiDenseMeshContactPairsConfig()
    if cfg.asset_limit_per_source < 2:
        raise ValueError("asset_limit_per_source must be at least 2")
    if cfg.pair_limit_per_case < 2:
        raise ValueError("pair_limit_per_case must be at least 2")
    if cfg.samples_per_pair < 1:
        raise ValueError("samples_per_pair must be positive")
    if not 0.0 < cfg.train_fraction < 1.0:
        raise ValueError("train_fraction must lie in (0, 1)")

    abc_top = _rename_assets(_load_abc_assets(Path(cfg.abc_root), cfg.asset_limit_per_source), "ABC top-face")
    abc_large = _load_large_face_abc_assets(cfg)
    fusion = _rename_assets(_load_fusion360_assets(Path(cfg.fusion360_root), cfg.asset_limit_per_source), "Fusion 360 Gallery")
    thingi = _rename_assets(_load_thingi10k_assets(Path(cfg.thingi10k_root), cfg.asset_limit_per_source), "Thingi10K")

    assets_by_source = {
        "ABC top-face": abc_top,
        "ABC large-face": abc_large,
        "Fusion 360 Gallery": fusion,
        "Thingi10K": thingi,
    }
    for source_name, assets in assets_by_source.items():
        if len(assets) < 2:
            raise ValueError(f"{source_name} produced fewer than two mesh assets")

    pairs_by_case = {
        "ABC-largeface-intra": _make_pairs(_rename_assets(abc_large, "ABC-largeface-intra"), cfg.pair_limit_per_case),
        "ABC-topface-intra": _make_pairs(_rename_assets(abc_top, "ABC-topface-intra"), cfg.pair_limit_per_case),
        "Fusion360-intra": _make_pairs(_rename_assets(fusion, "Fusion360-intra"), cfg.pair_limit_per_case),
        "Thingi10K-intra": _make_pairs(_rename_assets(thingi, "Thingi10K-intra"), cfg.pair_limit_per_case),
        "ABCtop-Fusion360-cross": _make_cross_pairs("ABCtop-Fusion360-cross", abc_top, fusion, limit=cfg.pair_limit_per_case),
        "ABCtop-Thingi10K-cross": _make_cross_pairs("ABCtop-Thingi10K-cross", abc_top, thingi, limit=cfg.pair_limit_per_case),
        "Fusion360-Thingi10K-cross": _make_cross_pairs("Fusion360-Thingi10K-cross", fusion, thingi, limit=cfg.pair_limit_per_case),
    }
    pairs_by_case = {case_name: pairs for case_name, pairs in pairs_by_case.items() if len(pairs) >= 2}
    if not pairs_by_case:
        raise ValueError("no dense mesh cases were generated")

    train_pairs_by_case: dict[str, tuple[MeshDensityPair, ...]] = {}
    eval_pairs_by_case: dict[str, tuple[MeshDensityPair, ...]] = {}
    for offset, (case_name, pairs) in enumerate(pairs_by_case.items()):
        train_pairs, eval_pairs = _split_pairs(
            pairs,
            train_fraction=cfg.train_fraction,
            seed=cfg.seed + offset,
        )
        train_pairs_by_case[case_name] = train_pairs
        eval_pairs_by_case[case_name] = eval_pairs

    train_dataset, train_case_by_query_id, train_costs = _build_dataset_from_case_pairs(
        train_pairs_by_case,
        first_sample_id=10_000_001,
        samples_per_pair=cfg.samples_per_pair,
    )
    eval_dataset, eval_case_by_query_id, eval_costs = _build_dataset_from_case_pairs(
        eval_pairs_by_case,
        first_sample_id=11_000_001,
        samples_per_pair=cfg.samples_per_pair,
    )
    train_workload = _scale_workload_costs(
        build_high_density_stpf_workload(train_dataset, cfg.high_density, name=f"{cfg.run_name}_train"),
        train_costs,
    )
    eval_workload = _scale_workload_costs(
        build_high_density_stpf_workload(eval_dataset, cfg.high_density, name=f"{cfg.run_name}_eval"),
        eval_costs,
    )

    shard_dir = Path(cfg.shard_root) / cfg.run_name
    shard_dir.mkdir(parents=True, exist_ok=True)
    write_npz_shard(
        shard_dir / "train.npz",
        train_dataset,
        metadata={
            **default_metadata(train_dataset, seed=cfg.seed, source="multi_dense_mesh_contact_pairs"),
            "dataset_role": "train",
            "case_count": len(train_pairs_by_case),
        },
    )
    write_npz_shard(
        shard_dir / "eval.npz",
        eval_dataset,
        metadata={
            **default_metadata(eval_dataset, seed=cfg.seed + 1, source="multi_dense_mesh_contact_pairs"),
            "dataset_role": "eval",
            "case_count": len(eval_pairs_by_case),
        },
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
    trained_model.to(cfg.training.device)
    trained_model.eval()
    random_model = build_stpf_model(cfg.training.model_preset)
    random_model.to(cfg.training.device)
    random_model.eval()

    combined_no_proposal = benchmark_no_proposal_on_high_density_workload(eval_workload)
    combined_random_stpf = benchmark_stpf_on_high_density_workload(
        eval_workload,
        model=random_model,
        device=cfg.training.device,
        proposal_batch_size=cfg.training.batch_size,
        method_name="MultiDenseMesh-RTSTPFExact-Random",
    )
    combined_trained_stpf = benchmark_stpf_on_high_density_workload(
        eval_workload,
        model=trained_model,
        device=cfg.training.device,
        proposal_batch_size=cfg.training.batch_size,
        method_name="MultiDenseMesh-RTSTPFExact-Trained",
    )

    case_results: list[MultiDenseMeshCaseResult] = []
    for case_name in sorted(eval_pairs_by_case):
        query_ids = {
            sample.query_id for sample in eval_dataset.samples if eval_case_by_query_id[sample.query_id] == case_name
        }
        case_workload = _subset_workload(eval_workload, query_ids, f"{cfg.run_name}_{case_name}_eval")
        no_proposal = benchmark_no_proposal_on_high_density_workload(case_workload)
        random_stpf = benchmark_stpf_on_high_density_workload(
            case_workload,
            model=random_model,
            device=cfg.training.device,
            proposal_batch_size=cfg.training.batch_size,
            method_name=f"{case_name}-RTSTPFExact-Random",
        )
        trained_stpf = benchmark_stpf_on_high_density_workload(
            case_workload,
            model=trained_model,
            device=cfg.training.device,
            proposal_batch_size=cfg.training.batch_size,
            method_name=f"{case_name}-RTSTPFExact-Trained",
        )
        case_assets = _unique_assets_from_pairs([*train_pairs_by_case[case_name], *eval_pairs_by_case[case_name]])
        face_counts = sorted(asset.face_count for asset in case_assets)
        case_results.append(
            MultiDenseMeshCaseResult(
                case_name=case_name,
                train_pair_count=len(train_pairs_by_case[case_name]),
                eval_pair_count=len(eval_pairs_by_case[case_name]),
                train_query_count=sum(
                    1 for sample in train_dataset.samples if train_case_by_query_id[sample.query_id] == case_name
                ),
                eval_query_count=len(query_ids),
                eval_candidate_count=case_workload.candidate_count,
                asset_count=len(case_assets),
                min_face_count=face_counts[0],
                median_face_count=face_counts[len(face_counts) // 2],
                max_face_count=face_counts[-1],
                no_proposal=no_proposal,
                random_stpf=random_stpf,
                trained_stpf=trained_stpf,
            )
        )

    output_root = Path(cfg.benchmark_output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    report_path = output_root / f"{cfg.run_name}.md"
    summary_json_path = output_root / f"{cfg.run_name}.json"
    result = MultiDenseMeshContactPairsResult(
        config=cfg,
        assets_by_source=assets_by_source,
        pairs_by_case=pairs_by_case,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        train_workload=train_workload,
        eval_workload=eval_workload,
        training_run=training_run,
        combined_no_proposal=combined_no_proposal,
        combined_random_stpf=combined_random_stpf,
        combined_trained_stpf=combined_trained_stpf,
        case_results=tuple(case_results),
        report_path=report_path,
        summary_json_path=summary_json_path,
    )
    _write_summary_json(summary_json_path, result, shard_dir)
    _write_report(report_path, result, shard_dir)
    return result


def _write_summary_json(path: Path, result: MultiDenseMeshContactPairsResult, shard_dir: Path) -> None:
    payload = {
        "config": asdict(result.config),
        "source_asset_counts": {name: len(assets) for name, assets in result.assets_by_source.items()},
        "case_pair_counts": {name: len(pairs) for name, pairs in result.pairs_by_case.items()},
        "train_query_count": result.train_workload.query_count,
        "eval_query_count": result.eval_workload.query_count,
        "train_candidate_count": result.train_workload.candidate_count,
        "eval_candidate_count": result.eval_workload.candidate_count,
        "eval_avg_candidates_per_query": result.eval_workload.avg_candidates_per_query,
        "eval_collision_query_count": _collision_query_count(result.eval_workload),
        "eval_collision_candidate_count": _collision_candidate_count(result.eval_workload),
        "checkpoint_path": str(result.training_run.artifacts.model_state_path),
        "shard_dir": str(shard_dir),
        "final_train_loss": result.training_run.final_train_loss,
        "final_validation_loss": result.training_run.final_validation_loss,
        "combined_no_proposal": _metric_dict(result.combined_no_proposal),
        "combined_random_stpf": _metric_dict(result.combined_random_stpf),
        "combined_trained_stpf": _metric_dict(result.combined_trained_stpf),
        "combined_exact_work_reduction_vs_no_proposal": _reduction(
            result.combined_trained_stpf,
            result.combined_no_proposal,
        ),
        "case_results": [
            {
                "case_name": item.case_name,
                "train_pair_count": item.train_pair_count,
                "eval_pair_count": item.eval_pair_count,
                "train_query_count": item.train_query_count,
                "eval_query_count": item.eval_query_count,
                "eval_candidate_count": item.eval_candidate_count,
                "asset_count": item.asset_count,
                "min_face_count": item.min_face_count,
                "median_face_count": item.median_face_count,
                "max_face_count": item.max_face_count,
                "no_proposal": _metric_dict(item.no_proposal),
                "random_stpf": _metric_dict(item.random_stpf),
                "trained_stpf": _metric_dict(item.trained_stpf),
                "trained_exact_work_reduction_vs_no_proposal": _reduction(
                    item.trained_stpf,
                    item.no_proposal,
                ),
            }
            for item in result.case_results
        ],
        "notes": [
            "This benchmark expands dense mesh contact-pair coverage across ABC, Fusion 360 Gallery, and Thingi10K.",
            "It includes same-source and cross-source pair cases to probe STPF generalization.",
            "Exact work is primitive-weighted dense candidate work calibrated by mesh face counts; final submesh exact wall-time still requires patch/cluster exact extraction in C++.",
        ],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _write_report(path: Path, result: MultiDenseMeshContactPairsResult, shard_dir: Path) -> None:
    density = result.eval_workload.avg_candidates_per_query
    lines = [
        "# Multi-source dense mesh contact-pair STPF training and benchmark report",
        "",
        "## Objective",
        "",
        "- descriptionthis paperMethoddense mesh contactdescription, coverage ABC, Fusion 360 Gallery, Thingi10K withdescription pair. ",
        "- description learned STPF, description RTSTPFExact in `candidate density` high, exact work descriptionwhenisdescriptionreduction exact certificate workload. ",
        "- RTSTPFExact uses only learned STPF description; descriptioncorrectnessdescription fallback/exact certificate conservative guarantee, reportdescription `FN=0` and exact-work reduction. ",
        "",
        "## data and training scale",
        "",
        f"- Run name: `{result.config.run_name}`",
        f"- Dense candidate density: `{result.config.high_density.slab_count} slabs x {result.config.high_density.patches_per_object} x {result.config.high_density.patches_per_object} patches = {density:.0f} candidates/query`",
        f"- Train queries / candidates: `{result.train_workload.query_count}` / `{result.train_workload.candidate_count}`",
        f"- Eval queries / candidates: `{result.eval_workload.query_count}` / `{result.eval_workload.candidate_count}`",
        f"- Eval collision queries / collision-overlap candidates: `{_collision_query_count(result.eval_workload)}` / `{_collision_candidate_count(result.eval_workload)}`",
        f"- Shard dir: `{shard_dir}`",
        f"- Checkpoint: `{result.training_run.artifacts.model_state_path}`",
        f"- Model preset: `{result.config.training.model_preset}`",
        f"- Device / epochs / batch: `{result.config.training.device}` / `{result.config.training.epochs}` / `{result.config.training.batch_size}`",
        f"- Final train loss / validation loss: `{result.training_run.final_train_loss:.6f}` / `{result.training_run.final_validation_loss:.6f}`",
        "",
        "## description",
        "",
        "| Method | Queries | Candidates | Exact calls | Fallback calls | Exact work | Proposal ms | Scheduling ms | Total ms | FN |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for metric in (result.combined_no_proposal, result.combined_random_stpf, result.combined_trained_stpf):
        lines.append(
            f"| `{metric.method_name}` | `{metric.query_count}` | `{metric.candidate_count}` | "
            f"`{metric.exact_call_count}` | `{metric.fallback_call_count}` | `{_fmt(metric.exact_work_units, 1)}` | "
            f"`{_fmt(metric.proposal_wall_ms, 3)}` | `{_fmt(metric.scheduling_wall_ms, 3)}` | "
            f"`{_fmt(metric.total_wall_ms, 3)}` | `{metric.fn_count}` |"
        )
    lines.extend(
        [
            "",
            f"- Trained STPF exact-work reduction vs NoProposal: `{_pct(_reduction(result.combined_trained_stpf, result.combined_no_proposal))}`. ",
            f"- Trained STPF FN: `{result.combined_trained_stpf.fn_count}`. ",
            "",
            "## splitdescription",
            "",
            "| Case | Train pairs | Eval pairs | Eval queries | Eval candidates | Face min/median/max | NoProposal work | Trained work | Reduction | Exact calls | FN |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in result.case_results:
        lines.append(
            f"| `{item.case_name}` | `{item.train_pair_count}` | `{item.eval_pair_count}` | "
            f"`{item.eval_query_count}` | `{item.eval_candidate_count}` | "
            f"`{item.min_face_count}/{item.median_face_count}/{item.max_face_count}` | "
            f"`{_fmt(item.no_proposal.exact_work_units, 1)}` | `{_fmt(item.trained_stpf.exact_work_units, 1)}` | "
            f"`{_pct(_reduction(item.trained_stpf, item.no_proposal))}` | "
            f"`{item.trained_stpf.exact_call_count}` | `{item.trained_stpf.fn_count}` |"
        )
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            f"- this runadded `{len(result.case_results)}` dense mesh contactdescription, containsdescriptionanddescription; eval descriptioncandidatedescriptionas `{result.eval_workload.candidate_count}`. ",
            f"- in density `{density:.0f}` candidates/query under, learned STPF description exact work from `{_fmt(result.combined_no_proposal.exact_work_units, 1)}` reduced to `{_fmt(result.combined_trained_stpf.exact_work_units, 1)}`, reduction `{_pct(_reduction(result.combined_trained_stpf, result.combined_no_proposal))}`. ",
            f"- Correctness Protocolunder, trained RTSTPFExact  `FN={result.combined_trained_stpf.fn_count}`, descriptionthis paperdescriptionconstraint. ",
            "- thisdescriptiongeneralizationdescription: STPF descriptionindescription synthetic/T0 ordescription ABC onhasdescription, descriptioncoveragereal CAD, description 3D mesh anddescription pair. ",
            "- description: current dense benchmark  exact work isbyreal mesh face count calibrated candidate work; if used as final wall-time description, description C++ exact certificate underdescriptionto patch/cluster submesh, descriptionOutputreal kernel wall time. ",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


__all__ = [
    "MultiDenseMeshContactPairsConfig",
    "MultiDenseMeshContactPairsResult",
    "MultiDenseMeshCaseResult",
    "run_multi_dense_mesh_contact_pairs_benchmark",
]
