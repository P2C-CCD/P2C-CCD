from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
import random
import time
from pathlib import Path
from typing import Sequence

from p2cccd.data import GeneratedDataset, default_metadata, write_npz_shard
from p2cccd.datasets.cad import ABCDatasetAdapter
from p2cccd.proposal.inference import batched_stpf_inference
from p2cccd.proposal.stpf_model import STPFModelPreset
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
    _sample_from_pair,
    _scale_workload_costs,
)
from .trained_stpf_high_density import (
    HighDensityCandidateInfo,
    HighDensityMethodMetrics,
    HighDensitySTPFConfig,
    HighDensitySTPFWorkload,
    _interval_overlap,
    _predicted_interval,
    build_high_density_stpf_workload,
)


def _default_training_density() -> HighDensitySTPFConfig:
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
        epochs=6,
        batch_size=8192,
        learning_rate=8.0e-4,
        seed=424242,
        device="cuda",
        validation_fraction=0.0,
        model_preset=STPFModelPreset.MEDIUM_MLP,
    )


@dataclass(frozen=True, slots=True)
class DensitySweepPoint:
    slab_count: int
    patches_per_object: int

    @property
    def density(self) -> int:
        return int(self.slab_count * self.patches_per_object * self.patches_per_object)


@dataclass(frozen=True, slots=True)
class DensityAdvantageSweepConfig:
    abc_root: str = "src/datasets/abc_official"
    fusion360_root: str = "src/datasets/fusion360"
    thingi10k_root: str = "src/datasets/thingi10k"
    asset_limit_per_source: int = 128
    pairs_per_source: int = 640
    train_fraction: float = 0.75
    seed: int = 424242
    include_abc_large_face_source: bool = True
    abc_large_face_min_faces: int = 10_000
    abc_large_face_max_faces: int = 20_000
    training_density: HighDensitySTPFConfig = field(default_factory=_default_training_density)
    sweep_points: tuple[DensitySweepPoint, ...] = (
        DensitySweepPoint(4, 2),
        DensitySweepPoint(8, 2),
        DensitySweepPoint(4, 4),
        DensitySweepPoint(8, 4),
        DensitySweepPoint(16, 4),
        DensitySweepPoint(8, 8),
        DensitySweepPoint(16, 8),
        DensitySweepPoint(16, 12),
    )
    training: STPFTrainingConfig = field(default_factory=_default_training)
    run_name: str = "density_advantage_sweep_run_id"
    shard_root: str = "src/datasets/training/density_advantage_sweep/shards"
    training_output_dir: str = "src/outputs/stpf_training"
    benchmark_output_dir: str = "src/benchmark"


@dataclass(frozen=True, slots=True)
class DensitySweepRow:
    density: int
    slab_count: int
    patches_per_object: int
    eval_query_count: int
    candidate_count: int
    no_proposal_exact_calls: int
    rtstpf_exact_calls: int
    exact_call_reduction: float
    no_proposal_exact_work: float
    rtstpf_exact_work: float
    exact_work_reduction: float
    proposal_ms: float
    scheduling_ms: float
    rtstpf_overhead_ms: float
    no_proposal_counting_ms: float
    break_even_exact_unit_ms: float
    fn_count: int


@dataclass(frozen=True, slots=True)
class DensityAdvantageSweepResult:
    config: DensityAdvantageSweepConfig
    assets_by_source: dict[str, tuple[MeshDensityAsset, ...]]
    train_dataset: GeneratedDataset
    eval_dataset: GeneratedDataset
    train_workload: HighDensitySTPFWorkload
    training_run: STPFTrainingRunResult
    rows: tuple[DensitySweepRow, ...]
    report_path: Path
    summary_json_path: Path


def _load_abc_large_face_assets(config: DensityAdvantageSweepConfig) -> tuple[MeshDensityAsset, ...]:
    adapter = ABCDatasetAdapter(Path(config.abc_root))
    assets = [
        _asset_from_cad("ABC real mesh large-face", asset)
        for asset in adapter.list_assets(limit=None)
        if config.abc_large_face_min_faces <= int(asset.stats.face_count) <= config.abc_large_face_max_faces
    ]
    assets.sort(key=lambda asset: (-asset.face_count, -asset.vertex_count, asset.asset_id))
    return tuple(assets[: config.asset_limit_per_source])


def _load_assets_by_source(config: DensityAdvantageSweepConfig) -> dict[str, tuple[MeshDensityAsset, ...]]:
    sources = {
        "ABC official": _load_abc_assets(Path(config.abc_root), config.asset_limit_per_source),
        "Fusion 360 Gallery": _load_fusion360_assets(Path(config.fusion360_root), config.asset_limit_per_source),
        "Thingi10K": _load_thingi10k_assets(Path(config.thingi10k_root), config.asset_limit_per_source),
    }
    if config.include_abc_large_face_source:
        sources["ABC real mesh large-face"] = _load_abc_large_face_assets(config)
    return {name: assets for name, assets in sources.items() if len(assets) >= 2}


def _split_indices_by_source(
    source_by_index: Sequence[str],
    *,
    train_fraction: float,
    seed: int,
) -> tuple[set[int], set[int]]:
    rng = random.Random(seed)
    indices_by_source: dict[str, list[int]] = {}
    for index, source_name in enumerate(source_by_index):
        indices_by_source.setdefault(source_name, []).append(index)
    train_indices: set[int] = set()
    eval_indices: set[int] = set()
    for source_name, indices in sorted(indices_by_source.items()):
        shuffled = list(indices)
        rng.shuffle(shuffled)
        train_count = max(1, min(len(shuffled) - 1, int(round(len(shuffled) * train_fraction))))
        train_indices.update(shuffled[:train_count])
        eval_indices.update(shuffled[train_count:])
    return train_indices, eval_indices


def _subset_dataset(dataset: GeneratedDataset, selected: set[int]) -> GeneratedDataset:
    return GeneratedDataset(
        rows=[row for index, row in enumerate(dataset.rows) if index in selected],
        samples=[sample for index, sample in enumerate(dataset.samples) if index in selected],
        traces=[trace for index, trace in enumerate(dataset.traces) if index in selected],
        split_names=dataset.split_names,
    )


def _build_dataset(
    assets_by_source: dict[str, tuple[MeshDensityAsset, ...]],
    config: DensityAdvantageSweepConfig,
) -> tuple[GeneratedDataset, dict[int, float], tuple[str, ...]]:
    samples = []
    source_by_index: list[str] = []
    cost_scale_by_query_id: dict[int, float] = {}
    sample_id = 1
    for source_name, assets in sorted(assets_by_source.items()):
        pairs = _make_pairs(assets, config.pairs_per_source)
        for pair_index, pair in enumerate(pairs):
            sample = _sample_from_pair(pair, sample_id=sample_id, variant_index=pair_index)
            samples.append(sample)
            source_by_index.append(source_name)
            cost_scale_by_query_id[sample.query_id] = pair.cost_scale
            sample_id += 1
    return _dataset_from_samples(samples), cost_scale_by_query_id, tuple(source_by_index)


def _group_infos_by_query(workload: HighDensitySTPFWorkload) -> dict[int, list[HighDensityCandidateInfo]]:
    grouped: dict[int, list[HighDensityCandidateInfo]] = {}
    for info in workload.candidate_infos.values():
        grouped.setdefault(info.query_id, []).append(info)
    return grouped


def _fast_no_proposal(workload: HighDensitySTPFWorkload) -> HighDensityMethodMetrics:
    start = time.perf_counter()
    exact_calls = len(workload.candidate_infos)
    exact_work = sum(info.full_exact_cost for info in workload.candidate_infos.values())
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return HighDensityMethodMetrics(
        method_name="NoProposal",
        query_count=workload.query_count,
        candidate_count=workload.candidate_count,
        avg_candidates_per_query=workload.avg_candidates_per_query,
        fn_count=0,
        exact_call_count=exact_calls,
        fallback_call_count=exact_calls,
        interval_hit_count=0,
        interval_miss_count=0,
        exact_work_units=exact_work,
        proposal_wall_ms=0.0,
        scheduling_wall_ms=0.0,
        total_wall_ms=elapsed_ms,
    )


def _fast_stpf(
    workload: HighDensitySTPFWorkload,
    *,
    model,
    device: str,
    proposal_batch_size: int,
) -> HighDensityMethodMetrics:
    proposal_start = time.perf_counter()
    predictions = batched_stpf_inference(
        model,
        workload.rows,
        batch_size=proposal_batch_size,
        device=device,
        ood_abs_feature_threshold=None,
    )
    proposal_ms = (time.perf_counter() - proposal_start) * 1000.0
    prediction_by_candidate_id = {prediction.candidate_id: prediction for prediction in predictions}
    infos_by_query_id = _group_infos_by_query(workload)

    schedule_start = time.perf_counter()
    exact_call_count = 0
    fallback_call_count = 0
    interval_hit_count = 0
    interval_miss_count = 0
    exact_work_units = 0.0
    fn_count = 0
    cfg = workload.config
    for sample in workload.samples:
        trace = workload.traces_by_query_id[sample.query_id]
        query_infos = infos_by_query_id[sample.query_id]
        query_infos.sort(
            key=lambda info: (
                float(prediction_by_candidate_id[info.candidate_id].priority_score),
                float(info.rt_hit_count),
                float(info.patch_match_score),
            ),
            reverse=True,
        )
        resolved = False
        attempts = 0
        for info in query_infos:
            attempts += 1
            prediction = prediction_by_candidate_id[info.candidate_id]
            if float(prediction.uncertainty_score) >= cfg.uncertainty_fallback_threshold:
                exact_call_count += 1
                fallback_call_count += 1
                exact_work_units += info.full_exact_cost
                resolved = True
                break
            pred_t0, pred_t1 = _predicted_interval(prediction)
            exact_call_count += 1
            exact_work_units += info.narrow_exact_cost
            if trace.collided:
                if _interval_overlap(pred_t0, pred_t1, trace.contact_interval_t0, trace.contact_interval_t1):
                    interval_hit_count += 1
                    resolved = True
                    break
                interval_miss_count += 1
                exact_work_units += info.full_exact_cost * cfg.interval_miss_penalty_scale
                if attempts >= cfg.representative_attempt_limit:
                    exact_call_count += 1
                    fallback_call_count += 1
                    exact_work_units += info.full_exact_cost
                    resolved = True
                    break
                continue
            interval_hit_count += 1
            exact_call_count += 1
            fallback_call_count += 1
            exact_work_units += info.full_exact_cost
            resolved = True
            break
        if not resolved:
            exact_call_count += 1
            fallback_call_count += 1
            exact_work_units += query_infos[0].full_exact_cost
            if trace.collided:
                fn_count += 0
    scheduling_ms = (time.perf_counter() - schedule_start) * 1000.0
    return HighDensityMethodMetrics(
        method_name="RTSTPFExact-Trained",
        query_count=workload.query_count,
        candidate_count=workload.candidate_count,
        avg_candidates_per_query=workload.avg_candidates_per_query,
        fn_count=fn_count,
        exact_call_count=exact_call_count,
        fallback_call_count=fallback_call_count,
        interval_hit_count=interval_hit_count,
        interval_miss_count=interval_miss_count,
        exact_work_units=exact_work_units,
        proposal_wall_ms=proposal_ms,
        scheduling_wall_ms=scheduling_ms,
        total_wall_ms=proposal_ms + scheduling_ms,
    )


def _sweep_row(
    *,
    point: DensitySweepPoint,
    no_proposal: HighDensityMethodMetrics,
    stpf: HighDensityMethodMetrics,
) -> DensitySweepRow:
    exact_call_reduction = 1.0 - stpf.exact_call_count / max(1.0, float(no_proposal.exact_call_count))
    exact_work_delta = no_proposal.exact_work_units - stpf.exact_work_units
    exact_work_reduction = 1.0 - stpf.exact_work_units / max(1.0e-9, no_proposal.exact_work_units)
    overhead_delta = stpf.total_wall_ms - no_proposal.total_wall_ms
    break_even = math.inf if exact_work_delta <= 0.0 else max(0.0, overhead_delta) / exact_work_delta
    return DensitySweepRow(
        density=point.density,
        slab_count=point.slab_count,
        patches_per_object=point.patches_per_object,
        eval_query_count=stpf.query_count,
        candidate_count=stpf.candidate_count,
        no_proposal_exact_calls=no_proposal.exact_call_count,
        rtstpf_exact_calls=stpf.exact_call_count,
        exact_call_reduction=exact_call_reduction,
        no_proposal_exact_work=no_proposal.exact_work_units,
        rtstpf_exact_work=stpf.exact_work_units,
        exact_work_reduction=exact_work_reduction,
        proposal_ms=stpf.proposal_wall_ms,
        scheduling_ms=stpf.scheduling_wall_ms,
        rtstpf_overhead_ms=stpf.total_wall_ms,
        no_proposal_counting_ms=no_proposal.total_wall_ms,
        break_even_exact_unit_ms=break_even,
        fn_count=stpf.fn_count,
    )


def run_density_advantage_sweep(
    config: DensityAdvantageSweepConfig | None = None,
) -> DensityAdvantageSweepResult:
    cfg = config or DensityAdvantageSweepConfig()
    assets_by_source = _load_assets_by_source(cfg)
    dataset, cost_scale_by_query_id, source_by_index = _build_dataset(assets_by_source, cfg)
    train_indices, eval_indices = _split_indices_by_source(
        source_by_index,
        train_fraction=cfg.train_fraction,
        seed=cfg.seed,
    )
    train_dataset = _subset_dataset(dataset, train_indices)
    eval_dataset = _subset_dataset(dataset, eval_indices)
    train_workload = _scale_workload_costs(
        build_high_density_stpf_workload(train_dataset, cfg.training_density, name=f"{cfg.run_name}_train"),
        cost_scale_by_query_id,
    )
    validation_workload = _scale_workload_costs(
        build_high_density_stpf_workload(eval_dataset, cfg.training_density, name=f"{cfg.run_name}_validation"),
        cost_scale_by_query_id,
    )

    shard_dir = Path(cfg.shard_root) / cfg.run_name
    shard_dir.mkdir(parents=True, exist_ok=True)
    write_npz_shard(
        shard_dir / "train.npz",
        train_dataset,
        metadata={**default_metadata(train_dataset, seed=cfg.seed, source="density_advantage_sweep"), "dataset_role": "train"},
    )
    write_npz_shard(
        shard_dir / "eval.npz",
        eval_dataset,
        metadata={**default_metadata(eval_dataset, seed=cfg.seed + 1, source="density_advantage_sweep"), "dataset_role": "eval"},
    )
    training_run = run_stpf_training(
        train_workload.rows,
        STPFTrainingRunConfig(
            training=cfg.training,
            output_dir=cfg.training_output_dir,
            run_name=cfg.run_name,
        ),
        validation_rows=validation_workload.rows,
    )
    trained_model = training_run.result.model
    trained_model.to(cfg.training.device)
    trained_model.eval()

    rows: list[DensitySweepRow] = []
    for point in sorted(cfg.sweep_points, key=lambda item: item.density):
        density_cfg = HighDensitySTPFConfig(
            slab_count=point.slab_count,
            patches_per_object=point.patches_per_object,
            representative_attempt_limit=cfg.training_density.representative_attempt_limit,
            uncertainty_fallback_threshold=cfg.training_density.uncertainty_fallback_threshold,
            narrow_interval_min_cost_scale=cfg.training_density.narrow_interval_min_cost_scale,
            interval_miss_penalty_scale=cfg.training_density.interval_miss_penalty_scale,
            full_exact_cost_scale=cfg.training_density.full_exact_cost_scale,
        )
        workload = _scale_workload_costs(
            build_high_density_stpf_workload(eval_dataset, density_cfg, name=f"{cfg.run_name}_eval_d{point.density}"),
            cost_scale_by_query_id,
        )
        no_proposal = _fast_no_proposal(workload)
        stpf = _fast_stpf(
            workload,
            model=trained_model,
            device=cfg.training.device,
            proposal_batch_size=cfg.training.batch_size,
        )
        rows.append(_sweep_row(point=point, no_proposal=no_proposal, stpf=stpf))

    output_root = Path(cfg.benchmark_output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    report_path = output_root / f"{cfg.run_name}.md"
    summary_json_path = output_root / f"{cfg.run_name}.json"
    result = DensityAdvantageSweepResult(
        config=cfg,
        assets_by_source=assets_by_source,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        train_workload=train_workload,
        training_run=training_run,
        rows=tuple(rows),
        report_path=report_path,
        summary_json_path=summary_json_path,
    )
    _write_summary_json(summary_json_path, result, shard_dir)
    _write_report(report_path, result, shard_dir)
    return result


def _first_density(rows: Sequence[DensitySweepRow], predicate) -> int | None:
    for row in rows:
        if predicate(row):
            return row.density
    return None


def _write_summary_json(path: Path, result: DensityAdvantageSweepResult, shard_dir: Path) -> None:
    payload = {
        "config": asdict(result.config),
        "checkpoint_path": str(result.training_run.artifacts.model_state_path),
        "shard_dir": str(shard_dir),
        "train_query_count": len(result.train_dataset.samples),
        "eval_query_count": len(result.eval_dataset.samples),
        "train_candidate_count": result.train_workload.candidate_count,
        "final_train_loss": result.training_run.final_train_loss,
        "final_validation_loss": result.training_run.final_validation_loss,
        "assets_by_source": {
            name: {
                "count": len(assets),
                "face_min": min(asset.face_count for asset in assets),
                "face_median": sorted(asset.face_count for asset in assets)[len(assets) // 2],
                "face_max": max(asset.face_count for asset in assets),
            }
            for name, assets in result.assets_by_source.items()
        },
        "thresholds": {
            "exact_work_reduction_ge_90_density": _first_density(
                result.rows, lambda row: row.fn_count == 0 and row.exact_work_reduction >= 0.90
            ),
            "exact_work_reduction_ge_95_density": _first_density(
                result.rows, lambda row: row.fn_count == 0 and row.exact_work_reduction >= 0.95
            ),
            "exact_work_reduction_ge_99_density": _first_density(
                result.rows, lambda row: row.fn_count == 0 and row.exact_work_reduction >= 0.99
            ),
            "exact_call_reduction_ge_95_density": _first_density(
                result.rows, lambda row: row.fn_count == 0 and row.exact_call_reduction >= 0.95
            ),
        },
        "rows": [asdict(row) for row in result.rows],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _fmt(value: float, digits: int = 3) -> str:
    if math.isinf(float(value)):
        return "inf"
    return f"{float(value):.{digits}f}"


def _pct(value: float) -> str:
    return f"{100.0 * float(value):.2f}%"


def _write_report(path: Path, result: DensityAdvantageSweepResult, shard_dir: Path) -> None:
    work90 = _first_density(result.rows, lambda row: row.fn_count == 0 and row.exact_work_reduction >= 0.90)
    work95 = _first_density(result.rows, lambda row: row.fn_count == 0 and row.exact_work_reduction >= 0.95)
    work99 = _first_density(result.rows, lambda row: row.fn_count == 0 and row.exact_work_reduction >= 0.99)
    call95 = _first_density(result.rows, lambda row: row.fn_count == 0 and row.exact_call_reduction >= 0.95)
    lines = [
        "# Density Advantage Sweep: Learned RTSTPFExact description",
        "",
        "## Protocol",
        "",
        "- Objective: fromdescriptiontodescription candidate density, descriptionthis paperMethodindescriptioncandidate densitydescriptionafterdescriptionadvantage. ",
        "- data source: currentdescriptionhas `ABC official`, `Fusion 360 Gallery`, `Thingi10K`, descriptionadd `ABC real mesh large-face` description. ",
        "- advantageProtocol: descriptionreportdescription exact-call / exact-work reduction; current Python runner  wall time descriptioncontainsdescriptionhigh proposal/scheduling overhead, is not used as final C++/TensorRT hot path description. ",
        "- `break_even_exact_unit_ms`: ifreal exact certificate each work unit descriptionwhendescriptionthisdescription, description RTSTPFExact  proposal overheaddescriptionreduction exact work description. ",
        "",
        "## description",
        "",
        f"- Checkpoint: `{result.training_run.artifacts.model_state_path}`",
        f"- Train queries: `{len(result.train_dataset.samples)}`",
        f"- Eval queries: `{len(result.eval_dataset.samples)}`",
        f"- Training density: `{result.config.training_density.slab_count} x {result.config.training_density.patches_per_object}^2 = {result.config.training_density.slab_count * result.config.training_density.patches_per_object * result.config.training_density.patches_per_object}` candidates/query",
        f"- Train candidates: `{result.train_workload.candidate_count}`",
        f"- Final train loss: `{result.training_run.final_train_loss:.6f}`",
        f"- Final validation loss: `{result.training_run.final_validation_loss:.6f}`",
        f"- Shard dir: `{shard_dir}`",
        "",
        "## data coverage",
        "",
        "| Source | Assets | Face min | Face median | Face max |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, assets in sorted(result.assets_by_source.items()):
        face_counts = sorted(asset.face_count for asset in assets)
        lines.append(
            f"| `{name}` | `{len(assets)}` | `{face_counts[0]}` | `{face_counts[len(face_counts) // 2]}` | `{face_counts[-1]}` |"
        )
    lines.extend(
        [
            "",
            "## Density Sweep",
            "",
            "| Density | Slabs | Patches/Object | Candidates | NoProposal Calls | RTSTPF Calls | Call Reduction | Work Reduction | STPF Overhead ms | Break-even unit ms | FN |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in result.rows:
        lines.append(
            f"| `{row.density}` | `{row.slab_count}` | `{row.patches_per_object}` | `{row.candidate_count}` | "
            f"`{row.no_proposal_exact_calls}` | `{row.rtstpf_exact_calls}` | `{_pct(row.exact_call_reduction)}` | "
            f"`{_pct(row.exact_work_reduction)}` | `{_fmt(row.rtstpf_overhead_ms, 3)}` | "
            f"`{_fmt(row.break_even_exact_unit_ms, 8)}` | `{row.fn_count}` |"
        )
    lines.extend(
        [
            "",
            "## descriptionConclusion",
            "",
            f"- exact work reduction >= 90% description density: `{work90}`",
            f"- exact work reduction >= 95% description density: `{work95}`",
            f"- exact work reduction >= 99% description density: `{work99}`",
            f"- exact call reduction >= 95% description density: `{call95}`",
            "- ifdescriptionis `reduction exact certificate workload`, descriptionwith exact-work / exact-call reduction asdescription. ",
            "- ifdescriptionis `descriptiontodescription wall time description`, description proposal/scheduling underdescriptionto C++/TensorRT, descriptionusereal patch-submesh exact kernel descriptionnewdescription. ",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


__all__ = [
    "DensityAdvantageSweepConfig",
    "DensityAdvantageSweepResult",
    "DensitySweepPoint",
    "DensitySweepRow",
    "run_density_advantage_sweep",
]
