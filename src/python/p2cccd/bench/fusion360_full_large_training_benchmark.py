from __future__ import annotations

from dataclasses import asdict, dataclass, field
import csv
import json
import math
import random
from pathlib import Path
from typing import Sequence

from p2cccd.data import GeneratedDataset, default_metadata, write_npz_shard
from p2cccd.proposal.stpf_model import STPFModelPreset, build_stpf_model
from p2cccd.proposal.training import STPFTrainingConfig
from p2cccd.proposal.training_runner import STPFTrainingRunConfig, run_stpf_training

from .common_modeling_ort_walltime_benchmark import run_common_modeling_ort_walltime_benchmark
from .high_density_mesh_training_benchmark import (
    MeshDensityAsset,
    MeshDensityPair,
    _cost_scale,
    _dataset_from_samples,
    _pair_score,
    _sample_from_pair,
    _scale_workload_costs,
)
from .trained_stpf_high_density import (
    HighDensityMethodMetrics,
    HighDensitySTPFConfig,
    benchmark_no_proposal_on_high_density_workload,
    benchmark_stpf_on_high_density_workload,
    build_high_density_stpf_workload,
    workload_to_shard_dataset,
)
from p2cccd.datasets.cad.mesh_io import mesh_stats_from_file, stable_asset_id


RUN_NAME = "fusion360_full_large_training_run_id"


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
        epochs=8,
        batch_size=65536,
        learning_rate=8.0e-4,
        seed=424242,
        device="cuda",
        validation_fraction=0.0,
        model_preset=STPFModelPreset.MEDIUM_MLP,
    )


@dataclass(frozen=True, slots=True)
class Fusion360FullLargeConfig:
    root: Path = Path("src/datasets/fusion360_full")
    run_name: str = RUN_NAME
    selected_sequence_limit: int = 4096
    pairs_per_sequence: int = 1
    train_fraction: float = 0.75
    seed: int = 424242
    high_density: HighDensitySTPFConfig = field(default_factory=_default_high_density)
    training: STPFTrainingConfig = field(default_factory=_default_training)
    shard_root: Path = Path("src/datasets/training/fusion360_full/shards")
    training_output_dir: Path = Path("src/outputs/stpf_training")
    benchmark_output_dir: Path = Path("src/benchmark")


@dataclass(frozen=True, slots=True)
class Fusion360SequenceSummary:
    sequence_name: str
    sequence_path: str
    part_obj_count: int
    total_obj_bytes: int
    max_obj_bytes: int
    has_assembly_json: bool
    complexity_score: float


@dataclass(frozen=True, slots=True)
class Fusion360SelectedPair:
    sequence_name: str
    pair: MeshDensityPair
    selected_rank: int


def _iter_sequence_summaries(root: Path) -> list[Fusion360SequenceSummary]:
    if not root.exists():
        raise FileNotFoundError(root)
    rows: list[Fusion360SequenceSummary] = []
    for sequence_path in sorted(path for path in root.iterdir() if path.is_dir()):
        obj_paths = [
            path
            for path in sequence_path.glob("*.obj")
            if path.is_file() and path.name.lower() != "assembly.obj"
        ]
        if len(obj_paths) < 2:
            continue
        sizes = [int(path.stat().st_size) for path in obj_paths]
        total_bytes = int(sum(sizes))
        max_bytes = int(max(sizes))
        score = math.log1p(len(obj_paths)) + 0.25 * math.log1p(total_bytes) + 0.5 * math.log1p(max_bytes)
        rows.append(
            Fusion360SequenceSummary(
                sequence_name=sequence_path.name,
                sequence_path=sequence_path.as_posix(),
                part_obj_count=len(obj_paths),
                total_obj_bytes=total_bytes,
                max_obj_bytes=max_bytes,
                has_assembly_json=(sequence_path / "assembly.json").exists(),
                complexity_score=float(score),
            )
        )
    rows.sort(key=lambda row: (-row.complexity_score, -row.part_obj_count, row.sequence_name))
    return rows


def _stratified_select_sequences(
    rows: Sequence[Fusion360SequenceSummary],
    *,
    limit: int,
    seed: int,
) -> tuple[Fusion360SequenceSummary, ...]:
    if limit <= 0:
        raise ValueError("selected_sequence_limit must be positive")
    if len(rows) <= limit:
        return tuple(rows)
    rng = random.Random(seed)
    sorted_rows = sorted(rows, key=lambda row: row.complexity_score)
    bin_count = 8
    per_bin = max(1, limit // bin_count)
    selected: list[Fusion360SequenceSummary] = []
    for bin_index in range(bin_count):
        start = bin_index * len(sorted_rows) // bin_count
        end = (bin_index + 1) * len(sorted_rows) // bin_count
        bucket = list(sorted_rows[start:end])
        rng.shuffle(bucket)
        bucket.sort(key=lambda row: (-row.complexity_score, row.sequence_name))
        selected.extend(bucket[:per_bin])
    remaining = [row for row in rows if row.sequence_name not in {item.sequence_name for item in selected}]
    remaining.sort(key=lambda row: (-row.complexity_score, row.sequence_name))
    selected.extend(remaining[: max(0, limit - len(selected))])
    selected = selected[:limit]
    selected.sort(key=lambda row: (-row.complexity_score, row.sequence_name))
    return tuple(selected)


def _asset_from_obj(root: Path, obj_path: Path, *, sequence_name: str) -> MeshDensityAsset:
    stats = mesh_stats_from_file(obj_path)
    return MeshDensityAsset(
        source_name="Fusion360 full assembly",
        asset_id=stable_asset_id(obj_path, root),
        asset_path=str(obj_path),
        face_count=int(stats.face_count),
        vertex_count=int(stats.vertex_count),
        diagonal=float(stats.diagonal),
        bounds_min=stats.bounds_min,
        bounds_max=stats.bounds_max,
        dirty_score=0.0,
    )


def _pairs_for_sequence(
    root: Path,
    summary: Fusion360SequenceSummary,
    *,
    limit: int,
) -> tuple[Fusion360SelectedPair, ...]:
    sequence_path = Path(summary.sequence_path)
    obj_paths = [
        path
        for path in sequence_path.glob("*.obj")
        if path.is_file() and path.name.lower() != "assembly.obj"
    ]
    obj_paths.sort(key=lambda path: (-int(path.stat().st_size), path.name))
    candidate_paths = obj_paths[: min(len(obj_paths), max(4, 2 * limit + 2))]
    assets: list[MeshDensityAsset] = []
    for path in candidate_paths:
        try:
            asset = _asset_from_obj(root, path, sequence_name=summary.sequence_name)
        except Exception:
            continue
        if asset.face_count > 0 and asset.vertex_count > 0:
            assets.append(asset)
    pairs: list[MeshDensityPair] = []
    for index_a, asset_a in enumerate(assets):
        for asset_b in assets[index_a + 1 :]:
            pairs.append(
                MeshDensityPair(
                    source_name="Fusion360 full assembly",
                    asset_a=asset_a,
                    asset_b=asset_b,
                    pair_score=_pair_score(asset_a, asset_b),
                    cost_scale=_cost_scale(asset_a, asset_b),
                )
            )
    pairs.sort(key=lambda pair: (-pair.pair_score, -pair.cost_scale, pair.asset_a.asset_id, pair.asset_b.asset_id))
    return tuple(
        Fusion360SelectedPair(sequence_name=summary.sequence_name, pair=pair, selected_rank=index + 1)
        for index, pair in enumerate(pairs[:limit])
    )


def _split_selected_pairs(
    selected_pairs: Sequence[Fusion360SelectedPair],
    *,
    train_fraction: float,
    seed: int,
) -> tuple[tuple[Fusion360SelectedPair, ...], tuple[Fusion360SelectedPair, ...]]:
    by_sequence: dict[str, list[Fusion360SelectedPair]] = {}
    for item in selected_pairs:
        by_sequence.setdefault(item.sequence_name, []).append(item)
    sequence_names = sorted(by_sequence)
    rng = random.Random(seed)
    rng.shuffle(sequence_names)
    train_count = max(1, min(len(sequence_names) - 1, int(round(len(sequence_names) * train_fraction))))
    train_sequences = set(sequence_names[:train_count])
    train = [item for item in selected_pairs if item.sequence_name in train_sequences]
    eval_items = [item for item in selected_pairs if item.sequence_name not in train_sequences]
    return tuple(train), tuple(eval_items)


def _dataset_from_selected_pairs(
    pairs: Sequence[Fusion360SelectedPair],
    *,
    first_sample_id: int,
) -> tuple[GeneratedDataset, dict[int, float], dict[int, str]]:
    samples = []
    cost_scale_by_query_id: dict[int, float] = {}
    sequence_by_query_id: dict[int, str] = {}
    for offset, item in enumerate(pairs):
        sample = _sample_from_pair(item.pair, sample_id=first_sample_id + offset, variant_index=offset)
        samples.append(sample)
        cost_scale_by_query_id[sample.query_id] = float(item.pair.cost_scale)
        sequence_by_query_id[sample.query_id] = item.sequence_name
    return _dataset_from_samples(samples), cost_scale_by_query_id, sequence_by_query_id


def _metric_dict(metric: HighDensityMethodMetrics) -> dict[str, object]:
    return {
        "method_name": metric.method_name,
        "query_count": metric.query_count,
        "candidate_count": metric.candidate_count,
        "avg_candidates_per_query": metric.avg_candidates_per_query,
        "fn_count": metric.fn_count,
        "exact_call_count": metric.exact_call_count,
        "fallback_call_count": metric.fallback_call_count,
        "interval_hit_count": metric.interval_hit_count,
        "interval_miss_count": metric.interval_miss_count,
        "exact_work_units": metric.exact_work_units,
        "proposal_wall_ms": metric.proposal_wall_ms,
        "scheduling_wall_ms": metric.scheduling_wall_ms,
        "total_wall_ms": metric.total_wall_ms,
    }


def _write_sequence_csv(path: Path, rows: Sequence[Fusion360SequenceSummary]) -> None:
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
    assert isinstance(baseline, dict)
    assert isinstance(random_stpf, dict)
    assert isinstance(trained, dict)
    assert isinstance(ort, dict)
    lines = [
        "# Fusion360 Full Assembly Large STPF training and validation report",
        "",
        "## 1. descriptionposition",
        "",
        f"- full root: `{result['root']}`",
        f"- shard root: `{result['shard_root']}`",
        f"- source sequences scanned: `{result['source_sequence_count']}`",
        f"- source part OBJ files: `{result['source_part_obj_count']}`",
        f"- selected sequences: `{result['selected_sequence_count']}`",
        "",
        "## 2. description",
        "",
        f"- model: `{result['model_preset']}`",
        f"- epochs: `{result['epochs']}`",
        f"- batch size: `{result['batch_size']}`",
        f"- density: `{result['density']}` candidates/query",
        f"- train dense rows: `{result['train_dense_rows']}`",
        f"- eval dense rows: `{result['eval_dense_rows']}`",
        f"- checkpoint: `{result['checkpoint_path']}`",
        "",
        "## 3. description",
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
            f"- learned STPF vs random STPF exact-work reduction: `{float(result['trained_reduction_vs_random']):.4%}`",
            "",
            "## 4. ORT TensorRT + C++ Scheduling",
            "",
            f"- ORT provider: `{ort['ort_provider']}`",
            f"- ORT inference ms: `{float(ort['ort_inference_ms']):.3f}`",
            f"- C++ scheduling ms: `{float(ort['cpp_schedule_ms']):.3f}`",
            f"- proposal total ms: `{float(ort['proposal_total_ms']):.3f}`",
            f"- proposal rows/s: `{float(ort['proposal_rows_per_second']):.1f}`",
            "",
            "## 5. Conclusion",
            "",
            "- thisdatasetdescriptionusecomplete Fusion360 Assembly root perform sequence-level description, description/descriptionby assembly sequence separation, avoid the same assembly leakage. ",
            "- learned STPF in full Fusion360 high-density assembly pair onkeep `FN=0`, description exact work. ",
            "- ORT TensorRT + C++ array scheduling descriptionconnectdescription dense shard, isafterdescription full stack wall-time default proposal Path. ",
            "- current correctness descriptionis analytic proxy oracle Protocol; description SOTA correctness description exact certificate / Tight-Inclusion comparisondescription. ",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_fusion360_full_large_training_benchmark(
    cfg: Fusion360FullLargeConfig | None = None,
) -> dict[str, object]:
    config = cfg or Fusion360FullLargeConfig()
    if not 0.0 < config.train_fraction < 1.0:
        raise ValueError("train_fraction must be in (0, 1)")
    config.benchmark_output_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = config.shard_root / config.run_name
    shard_dir.mkdir(parents=True, exist_ok=True)

    sequence_rows = _iter_sequence_summaries(config.root)
    selected_sequences = _stratified_select_sequences(
        sequence_rows,
        limit=config.selected_sequence_limit,
        seed=config.seed,
    )
    selected_pairs: list[Fusion360SelectedPair] = []
    skipped_sequences: list[str] = []
    for summary in selected_sequences:
        pairs = _pairs_for_sequence(config.root, summary, limit=config.pairs_per_sequence)
        if not pairs:
            skipped_sequences.append(summary.sequence_name)
            continue
        selected_pairs.extend(pairs)
    if len(selected_pairs) < 2:
        raise RuntimeError("Fusion360 full selection produced fewer than two trainable pairs")

    train_pairs, eval_pairs = _split_selected_pairs(
        selected_pairs,
        train_fraction=config.train_fraction,
        seed=config.seed,
    )
    train_dataset, train_costs, _ = _dataset_from_selected_pairs(train_pairs, first_sample_id=31_000_001)
    eval_dataset, eval_costs, _ = _dataset_from_selected_pairs(eval_pairs, first_sample_id=32_000_001)
    train_workload = _scale_workload_costs(
        build_high_density_stpf_workload(train_dataset, config.high_density, name=f"{config.run_name}_train"),
        train_costs,
    )
    eval_workload = _scale_workload_costs(
        build_high_density_stpf_workload(eval_dataset, config.high_density, name=f"{config.run_name}_eval"),
        eval_costs,
    )

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
    dense_train_dataset = workload_to_shard_dataset(train_workload)
    dense_eval_dataset = workload_to_shard_dataset(eval_workload)
    write_npz_shard(
        shard_dir / "dense_train.npz",
        dense_train_dataset,
        metadata={
            **default_metadata(dense_train_dataset, seed=config.seed, source=config.run_name),
            "dataset_role": "dense_train",
            "candidates_per_query": config.high_density.slab_count
            * config.high_density.patches_per_object
            * config.high_density.patches_per_object,
        },
    )
    write_npz_shard(
        shard_dir / "dense_eval.npz",
        dense_eval_dataset,
        metadata={
            **default_metadata(dense_eval_dataset, seed=config.seed + 1, source=config.run_name),
            "dataset_role": "dense_eval",
            "candidates_per_query": config.high_density.slab_count
            * config.high_density.patches_per_object
            * config.high_density.patches_per_object,
        },
    )
    _write_sequence_csv(shard_dir / "selected_sequences.csv", selected_sequences)

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
        method_name="RTSTPFExact-Fusion360Full-Learned",
    )

    checkpoint_path = training_run.artifacts.model_state_path
    if checkpoint_path is None:
        raise RuntimeError("training did not produce a checkpoint")
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

    density = config.high_density.slab_count * config.high_density.patches_per_object * config.high_density.patches_per_object
    result: dict[str, object] = {
        "run_name": config.run_name,
        "root": config.root.as_posix(),
        "shard_root": shard_dir.as_posix(),
        "source_sequence_count": len(sequence_rows),
        "source_part_obj_count": sum(row.part_obj_count for row in sequence_rows),
        "selected_sequence_count": len(selected_sequences),
        "selected_pair_count": len(selected_pairs),
        "skipped_sequence_count": len(skipped_sequences),
        "train_query_count": train_workload.query_count,
        "eval_query_count": eval_workload.query_count,
        "train_dense_rows": train_workload.candidate_count,
        "eval_dense_rows": eval_workload.candidate_count,
        "density": density,
        "model_preset": str(config.training.model_preset),
        "epochs": config.training.epochs,
        "batch_size": config.training.batch_size,
        "checkpoint_path": checkpoint_path.as_posix(),
        "onnx_path": checkpoint_path.with_suffix(".onnx").as_posix(),
        "baseline": _metric_dict(baseline),
        "random_stpf": _metric_dict(random_stpf),
        "trained_stpf": _metric_dict(trained_stpf),
        "trained_reduction_vs_no_proposal": 1.0 - trained_stpf.exact_work_units / max(1.0e-9, baseline.exact_work_units),
        "trained_reduction_vs_random": 1.0 - trained_stpf.exact_work_units / max(1.0e-9, random_stpf.exact_work_units),
        "final_train_loss": training_run.final_train_loss,
        "final_validation_loss": training_run.final_validation_loss,
        "ort_tensorrt_cpp": ort_result,
        "artifacts": {
            "base_train": (shard_dir / "base_train.npz").as_posix(),
            "base_eval": (shard_dir / "base_eval.npz").as_posix(),
            "dense_train": (shard_dir / "dense_train.npz").as_posix(),
            "dense_eval": (shard_dir / "dense_eval.npz").as_posix(),
            "selected_sequences_csv": (shard_dir / "selected_sequences.csv").as_posix(),
            "ort_report": ort_report.as_posix(),
        },
    }
    manifest_path = shard_dir / "manifest.json"
    summary_json = config.benchmark_output_dir / f"{config.run_name}.json"
    report_path = config.benchmark_output_dir / f"{config.run_name}.md"
    manifest_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    summary_json.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    _write_report(report_path, result)
    return result


def main() -> None:
    result = run_fusion360_full_large_training_benchmark()
    print(json.dumps({"report": f"src/benchmark/{RUN_NAME}.md", "checkpoint": result["checkpoint_path"]}, indent=2))


if __name__ == "__main__":
    main()
