from __future__ import annotations

from dataclasses import asdict, dataclass, field
import csv
import json
import random
from itertools import combinations
from pathlib import Path
from typing import Sequence

from p2cccd.data import GeneratedDataset, default_metadata, write_npz_shard
from p2cccd.proposal.stpf_model import STPFModelPreset, build_stpf_model
from p2cccd.proposal.training import STPFTrainingConfig
from p2cccd.proposal.training_runner import STPFTrainingRunConfig, run_stpf_training

from .common_modeling_high_density_scenarios import CommonModelingScenarioConfig, _build_samples as _build_physics_samples
from .common_modeling_ort_walltime_benchmark import run_common_modeling_ort_walltime_benchmark
from .fusion360_full_large_training_benchmark import (
    _iter_sequence_summaries,
    _pairs_for_sequence,
    _stratified_select_sequences,
)
from .high_density_mesh_training_benchmark import (
    MeshDensityAsset,
    MeshDensityPair,
    _cost_scale,
    _dataset_from_samples,
    _load_abc_assets,
    _load_thingi10k_assets,
    _pair_score,
    _sample_from_pair,
    _scale_workload_costs,
    _subset_workload,
)
from .large_dense_complex_mesh_cases import _make_heavy_cross_pairs, _make_heavy_intra_pairs
from .multi_dense_mesh_contact_pairs import MultiDenseMeshContactPairsConfig, _load_large_face_abc_assets, _rename_assets
from .trained_stpf_high_density import (
    HighDensityMethodMetrics,
    HighDensitySTPFConfig,
    benchmark_no_proposal_on_high_density_workload,
    benchmark_stpf_on_high_density_workload,
    build_high_density_stpf_workload,
    workload_to_shard_dataset,
)


RUN_NAME = "rtstpf_advantage_cases_v4_large_training_run_id"


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
        epochs=12,
        batch_size=65536,
        learning_rate=7.0e-4,
        seed=424242,
        device="cuda",
        validation_fraction=0.0,
        model_preset=STPFModelPreset.MEDIUM_MLP,
    )


@dataclass(frozen=True, slots=True)
class AdvantageCasesV4Config:
    abc_root: Path = Path("src/datasets/abc_official")
    fusion360_full_root: Path = Path("src/datasets/fusion360_full")
    thingi10k_root: Path = Path("src/datasets/thingi10k")
    run_name: str = RUN_NAME
    asset_limit_per_source: int = 96
    fusion_sequence_limit: int = 512
    fusion_pairs_per_sequence: int = 1
    pair_limit_per_case: int = 96
    samples_per_pair: int = 2
    train_fraction: float = 0.75
    physics_train_per_scenario: int = 256
    physics_eval_per_scenario: int = 96
    seed: int = 424242
    high_density: HighDensitySTPFConfig = field(default_factory=_default_high_density)
    training: STPFTrainingConfig = field(default_factory=_default_training)
    shard_root: Path = Path("src/datasets/training/rtstpf_advantage_cases_v4/shards")
    training_output_dir: Path = Path("src/outputs/stpf_training")
    benchmark_output_dir: Path = Path("src/benchmark")


@dataclass(frozen=True, slots=True)
class AdvantageCaseSummary:
    case_name: str
    role: str
    query_count: int
    candidate_count: int
    positive_queries: int
    exact_work_reduction: float | None = None
    exact_call_reduction: float | None = None
    trained_exact_calls: int | None = None
    trained_exact_work: float | None = None
    fn_count: int | None = None


def _density(cfg: HighDensitySTPFConfig) -> int:
    return int(cfg.slab_count * cfg.patches_per_object * cfg.patches_per_object)


def _split_pairs(
    pairs: Sequence[MeshDensityPair],
    *,
    train_fraction: float,
    seed: int,
) -> tuple[tuple[MeshDensityPair, ...], tuple[MeshDensityPair, ...]]:
    items = list(pairs)
    random.Random(seed).shuffle(items)
    if len(items) < 2:
        raise ValueError("each advantage case needs at least two pairs")
    train_count = max(1, min(len(items) - 1, int(round(len(items) * train_fraction))))
    return tuple(items[:train_count]), tuple(items[train_count:])


def _unique_assets_from_pairs(pairs: Sequence[MeshDensityPair], *, limit: int) -> tuple[MeshDensityAsset, ...]:
    by_path: dict[str, MeshDensityAsset] = {}
    for pair in pairs:
        by_path[pair.asset_a.asset_path] = pair.asset_a
        by_path[pair.asset_b.asset_path] = pair.asset_b
    assets = list(by_path.values())
    assets.sort(key=lambda asset: (-asset.face_count, -asset.vertex_count, asset.asset_id))
    return tuple(assets[:limit])


def _make_mixed_intra_pairs(
    case_name: str,
    assets: Sequence[MeshDensityAsset],
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
        for asset_a, asset_b in combinations(assets, 2)
    ]
    pairs.sort(key=lambda pair: (-pair.cost_scale, -pair.pair_score, pair.asset_a.asset_id, pair.asset_b.asset_id))
    return tuple(pairs[:limit])


def _collect_fusion_full_pairs(cfg: AdvantageCasesV4Config) -> tuple[MeshDensityPair, ...]:
    summaries = _iter_sequence_summaries(cfg.fusion360_full_root)
    selected = _stratified_select_sequences(summaries, limit=cfg.fusion_sequence_limit, seed=cfg.seed)
    pairs: list[MeshDensityPair] = []
    for summary in selected:
        for selected_pair in _pairs_for_sequence(cfg.fusion360_full_root, summary, limit=cfg.fusion_pairs_per_sequence):
            pairs.append(selected_pair.pair)
    pairs.sort(key=lambda pair: (-pair.cost_scale, -pair.pair_score, pair.asset_a.asset_id, pair.asset_b.asset_id))
    return tuple(pairs)


def _build_pair_cases(cfg: AdvantageCasesV4Config) -> dict[str, tuple[MeshDensityPair, ...]]:
    abc_top = _rename_assets(_load_abc_assets(cfg.abc_root, cfg.asset_limit_per_source), "ABC top-face")
    abc_large = _rename_assets(
        _load_large_face_abc_assets(
            MultiDenseMeshContactPairsConfig(
                abc_root=cfg.abc_root.as_posix(),
                asset_limit_per_source=cfg.asset_limit_per_source,
            )
        ),
        "ABC large-face",
    )
    thingi = tuple(
        sorted(
            _rename_assets(_load_thingi10k_assets(cfg.thingi10k_root, cfg.asset_limit_per_source), "Thingi10K dirty"),
            key=lambda asset: (-asset.dirty_score, -asset.face_count, asset.asset_id),
        )
    )
    fusion_pairs = _collect_fusion_full_pairs(cfg)
    fusion_assets = _rename_assets(
        _unique_assets_from_pairs(fusion_pairs, limit=cfg.asset_limit_per_source * 2),
        "Fusion360 full assembly",
    )

    if len(abc_top) < 2 or len(abc_large) < 2 or len(thingi) < 2 or len(fusion_assets) < 2:
        raise RuntimeError("not enough assets for advantage case construction")

    return {
        "A01-ABC-topface-intra": _make_heavy_intra_pairs(
            "A01-ABC-topface-intra",
            abc_top,
            limit=cfg.pair_limit_per_case,
        ),
        "A02-ABC-largeface-intra": _make_heavy_intra_pairs(
            "A02-ABC-largeface-intra",
            abc_large,
            limit=cfg.pair_limit_per_case,
        ),
        "A03-Fusion360Full-assembly-intra": tuple(fusion_pairs[: cfg.pair_limit_per_case]),
        "A04-Fusion360Full-highcomplex-intra": _make_mixed_intra_pairs(
            "A04-Fusion360Full-highcomplex-intra",
            fusion_assets,
            limit=cfg.pair_limit_per_case,
        ),
        "A05-Thingi10K-dirty-intra": _make_heavy_intra_pairs(
            "A05-Thingi10K-dirty-intra",
            thingi,
            limit=cfg.pair_limit_per_case,
        ),
        "A06-ABCtop-Fusion360Full-cross": _make_heavy_cross_pairs(
            "A06-ABCtop-Fusion360Full-cross",
            abc_top,
            fusion_assets,
            limit=cfg.pair_limit_per_case,
        ),
        "A07-ABClarge-Fusion360Full-cross": _make_heavy_cross_pairs(
            "A07-ABClarge-Fusion360Full-cross",
            abc_large,
            fusion_assets,
            limit=cfg.pair_limit_per_case,
        ),
        "A08-ABCtop-Thingi10Kdirty-cross": _make_heavy_cross_pairs(
            "A08-ABCtop-Thingi10Kdirty-cross",
            abc_top,
            thingi,
            limit=cfg.pair_limit_per_case,
        ),
        "A09-Fusion360Full-Thingi10Kdirty-cross": _make_heavy_cross_pairs(
            "A09-Fusion360Full-Thingi10Kdirty-cross",
            fusion_assets,
            thingi,
            limit=cfg.pair_limit_per_case,
        ),
        "A10-ABClarge-ABCtop-cross": _make_heavy_cross_pairs(
            "A10-ABClarge-ABCtop-cross",
            abc_large,
            abc_top,
            limit=cfg.pair_limit_per_case,
        ),
    }


def _build_mesh_samples(
    pairs_by_case: dict[str, tuple[MeshDensityPair, ...]],
    *,
    cfg: AdvantageCasesV4Config,
) -> tuple[list, list, dict[int, str], dict[int, str], dict[int, float], dict[int, float], dict[str, tuple[int, int]]]:
    train_samples = []
    eval_samples = []
    train_case_by_query: dict[int, str] = {}
    eval_case_by_query: dict[int, str] = {}
    train_costs: dict[int, float] = {}
    eval_costs: dict[int, float] = {}
    split_counts: dict[str, tuple[int, int]] = {}
    train_sample_id = 40_000_000
    eval_sample_id = 50_000_000
    for case_index, (case_name, pairs) in enumerate(pairs_by_case.items()):
        train_pairs, eval_pairs = _split_pairs(
            pairs,
            train_fraction=cfg.train_fraction,
            seed=cfg.seed + 97 * case_index,
        )
        split_counts[case_name] = (len(train_pairs), len(eval_pairs))
        for pair_index, pair in enumerate(train_pairs):
            for local_index in range(cfg.samples_per_pair):
                sample = _sample_from_pair(
                    pair,
                    sample_id=train_sample_id,
                    variant_index=case_index * 100_000 + pair_index * cfg.samples_per_pair + local_index,
                )
                train_samples.append(sample)
                train_case_by_query[sample.query_id] = case_name
                train_costs[sample.query_id] = float(pair.cost_scale)
                train_sample_id += 1
        for pair_index, pair in enumerate(eval_pairs):
            for local_index in range(cfg.samples_per_pair):
                sample = _sample_from_pair(
                    pair,
                    sample_id=eval_sample_id,
                    variant_index=case_index * 100_000 + pair_index * cfg.samples_per_pair + local_index,
                )
                eval_samples.append(sample)
                eval_case_by_query[sample.query_id] = case_name
                eval_costs[sample.query_id] = float(pair.cost_scale)
                eval_sample_id += 1
    return train_samples, eval_samples, train_case_by_query, eval_case_by_query, train_costs, eval_costs, split_counts


def _build_physics_datasets(cfg: AdvantageCasesV4Config) -> tuple[GeneratedDataset, GeneratedDataset, dict[int, str], dict[int, str], dict[int, float], dict[int, float]]:
    scenario_cfg = CommonModelingScenarioConfig(
        run_name=f"{cfg.run_name}_physics",
        seed=cfg.seed,
        train_per_scenario=cfg.physics_train_per_scenario,
        eval_per_scenario=cfg.physics_eval_per_scenario,
        slab_count=cfg.high_density.slab_count,
        patches_per_object=cfg.high_density.patches_per_object,
    )
    train_dataset = _dataset_from_samples(_build_physics_samples(cfg=scenario_cfg, role="train"))
    eval_dataset = _dataset_from_samples(_build_physics_samples(cfg=scenario_cfg, role="eval"))
    train_case_by_query = {sample.query_id: f"P-{sample.split}" for sample in train_dataset.samples}
    eval_case_by_query = {sample.query_id: f"P-{sample.split}" for sample in eval_dataset.samples}
    train_costs = {sample.query_id: 1.0 for sample in train_dataset.samples}
    eval_costs = {sample.query_id: 1.0 for sample in eval_dataset.samples}
    return train_dataset, eval_dataset, train_case_by_query, eval_case_by_query, train_costs, eval_costs


def _merge_datasets(mesh_dataset: GeneratedDataset, physics_dataset: GeneratedDataset) -> GeneratedDataset:
    return GeneratedDataset(
        rows=list(mesh_dataset.rows) + list(physics_dataset.rows),
        samples=list(mesh_dataset.samples) + list(physics_dataset.samples),
        traces=list(mesh_dataset.traces) + list(physics_dataset.traces),
        split_names=tuple(dict.fromkeys(list(mesh_dataset.split_names) + list(physics_dataset.split_names)).keys()),
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


def _write_case_summary_csv(path: Path, rows: Sequence[AdvantageCaseSummary]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()) if rows else [])
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _write_report(path: Path, result: dict[str, object]) -> None:
    baseline = result["baseline"]
    random_stpf = result["random_stpf"]
    trained = result["trained_stpf"]
    ort = result["ort_tensorrt_cpp"]
    case_rows = result["case_summaries"]
    assert isinstance(baseline, dict)
    assert isinstance(random_stpf, dict)
    assert isinstance(trained, dict)
    assert isinstance(ort, dict)
    assert isinstance(case_rows, list)
    lines = [
        "# RTSTPFExact advantage cases v4 large-scale training and validation report",
        "",
        "## 1. Objective",
        "",
        "constructdescriptionthis paperadvantagedescription dense/high-cost case: real CAD/mesh high-densitycontactdescription + descriptionphysicsdescription hard case. STPF only performs proposal/scheduling, descriptioncorrectnessdescription exact certificate / conservative fallback description. ",
        "",
        "## 2. data and training scale",
        "",
        f"- run name: `{result['run_name']}`",
        f"- case families: `{result['case_family_count']}`",
        f"- density: `{result['density']}` candidates/query",
        f"- train queries: `{result['train_query_count']}`",
        f"- eval queries: `{result['eval_query_count']}`",
        f"- train dense rows: `{result['train_dense_rows']}`",
        f"- eval dense rows: `{result['eval_dense_rows']}`",
        f"- model: `{result['model_preset']}`",
        f"- epochs: `{result['epochs']}`",
        f"- batch size: `{result['batch_size']}`",
        f"- checkpoint: `{result['checkpoint_path']}`",
        "",
        "## 3. Overall description",
        "",
        "| Method | Exact calls | Exact work | FN | Proposal ms | Schedule ms | Total ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in (baseline, random_stpf, trained):
        lines.append(
            f"| `{row['method_name']}` | `{row['exact_call_count']}` | `{float(row['exact_work_units']):.4f}` | "
            f"`{row['fn_count']}` | `{float(row['proposal_wall_ms']):.3f}` | "
            f"`{float(row['scheduling_wall_ms']):.3f}` | `{float(row['total_wall_ms']):.3f}` |"
        )
    lines.extend(
        [
            "",
            f"- learned STPF vs NoProposal exact-work reduction: `{float(result['trained_reduction_vs_no_proposal']):.4%}`",
            f"- learned STPF vs Random STPF exact-work reduction: `{float(result['trained_reduction_vs_random']):.4%}`",
            f"- learned STPF exact-call reduction vs NoProposal: `{float(result['trained_call_reduction_vs_no_proposal']):.4%}`",
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
            "| Case | Role | Queries | Candidates | Positives | Exact-work reduction | Exact-call reduction | Trained exact calls | FN |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in case_rows:
        work = "" if row["exact_work_reduction"] is None else f"{100.0 * float(row['exact_work_reduction']):.2f}%"
        calls = "" if row["exact_call_reduction"] is None else f"{100.0 * float(row['exact_call_reduction']):.2f}%"
        lines.append(
            f"| `{row['case_name']}` | `{row['role']}` | `{row['query_count']}` | `{row['candidate_count']}` | "
            f"`{row['positive_queries']}` | `{work}` | `{calls}` | `{row.get('trained_exact_calls', '')}` | `{row.get('fn_count', '')}` |"
        )
    lines.extend(
        [
            "",
            "## 6. Conclusion",
            "",
            "- this v4 datasetdescriptioncoverage high candidate-density, large mesh, dirty/OOD mesh, assembly approach, high-speed TOI, flexible wall, rotating sweep, thin-slot insertion. ",
            "- if `FN=0` and exact-work reduction keephighdescription, descriptionasthis paperadvantage case description. ",
            "- thisdescriptionis dense proxy oracle description/descriptionProtocol; final paper correctness descriptionand Tight-Inclusion / exact certificate fallback description. ",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_advantage_cases_v4_large_training(cfg: AdvantageCasesV4Config | None = None) -> dict[str, object]:
    config = cfg or AdvantageCasesV4Config()
    if not 0.0 < config.train_fraction < 1.0:
        raise ValueError("train_fraction must be in (0, 1)")
    shard_dir = config.shard_root / config.run_name
    shard_dir.mkdir(parents=True, exist_ok=True)
    config.benchmark_output_dir.mkdir(parents=True, exist_ok=True)

    pairs_by_case = _build_pair_cases(config)
    (
        mesh_train_samples,
        mesh_eval_samples,
        mesh_train_cases,
        mesh_eval_cases,
        mesh_train_costs,
        mesh_eval_costs,
        pair_split_counts,
    ) = _build_mesh_samples(pairs_by_case, cfg=config)
    mesh_train_dataset = _dataset_from_samples(mesh_train_samples)
    mesh_eval_dataset = _dataset_from_samples(mesh_eval_samples)
    physics_train_dataset, physics_eval_dataset, physics_train_cases, physics_eval_cases, physics_train_costs, physics_eval_costs = _build_physics_datasets(config)

    train_dataset = _merge_datasets(mesh_train_dataset, physics_train_dataset)
    eval_dataset = _merge_datasets(mesh_eval_dataset, physics_eval_dataset)
    train_case_by_query = {**mesh_train_cases, **physics_train_cases}
    eval_case_by_query = {**mesh_eval_cases, **physics_eval_cases}
    train_costs = {**mesh_train_costs, **physics_train_costs}
    eval_costs = {**mesh_eval_costs, **physics_eval_costs}

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
    trained_model = training_run.result.model
    trained_model.to(config.training.device)
    trained_model.eval()
    trained_stpf = benchmark_stpf_on_high_density_workload(
        eval_workload,
        model=trained_model,
        device=config.training.device,
        proposal_batch_size=config.training.batch_size,
        method_name="RTSTPFExact-AdvantageV4-Learned",
    )

    checkpoint_path = training_run.artifacts.model_state_path
    if checkpoint_path is None:
        raise RuntimeError("training did not produce checkpoint")

    case_rows: list[AdvantageCaseSummary] = []
    for case_name in sorted(set(train_case_by_query.values()) | set(eval_case_by_query.values())):
        for role, workload, case_map in (
            ("train", train_workload, train_case_by_query),
            ("eval", eval_workload, eval_case_by_query),
        ):
            query_ids = {query_id for query_id, mapped in case_map.items() if mapped == case_name}
            if not query_ids:
                continue
            if role == "train":
                case_rows.append(
                    AdvantageCaseSummary(
                        case_name=case_name,
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
                method_name=f"RTSTPFExact-AdvantageV4-{case_name}",
            )
            case_rows.append(
                AdvantageCaseSummary(
                    case_name=case_name,
                    role=role,
                    query_count=len(query_ids),
                    candidate_count=len(query_ids) * _density(config.high_density),
                    positive_queries=_positive_query_count(workload, query_ids),
                    exact_work_reduction=_reduction(case_trained, case_baseline),
                    exact_call_reduction=_call_reduction(case_trained, case_baseline),
                    trained_exact_calls=case_trained.exact_call_count,
                    trained_exact_work=case_trained.exact_work_units,
                    fn_count=case_trained.fn_count,
                )
            )

    case_csv_path = config.benchmark_output_dir / f"{config.run_name}_case_summary.csv"
    _write_case_summary_csv(case_csv_path, case_rows)

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
        "density": _density(config.high_density),
        "case_family_count": len(set(eval_case_by_query.values())),
        "pair_split_counts": {key: {"train_pairs": value[0], "eval_pairs": value[1]} for key, value in pair_split_counts.items()},
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
        "trained_stpf": _metric_dict(trained_stpf),
        "trained_reduction_vs_no_proposal": _reduction(trained_stpf, baseline),
        "trained_reduction_vs_random": _reduction(trained_stpf, random_stpf),
        "trained_call_reduction_vs_no_proposal": _call_reduction(trained_stpf, baseline),
        "case_summaries": [asdict(row) for row in case_rows],
        "ort_tensorrt_cpp": ort_result,
        "artifacts": {
            "shard_dir": shard_dir.as_posix(),
            "base_train": (shard_dir / "base_train.npz").as_posix(),
            "base_eval": (shard_dir / "base_eval.npz").as_posix(),
            "dense_train": (shard_dir / "dense_train.npz").as_posix(),
            "dense_eval": (shard_dir / "dense_eval.npz").as_posix(),
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
    result = run_advantage_cases_v4_large_training()
    print(
        json.dumps(
            {
                "report": f"src/benchmark/{RUN_NAME}.md",
                "checkpoint": result["checkpoint_path"],
                "train_dense_rows": result["train_dense_rows"],
                "eval_dense_rows": result["eval_dense_rows"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
