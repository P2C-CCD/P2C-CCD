from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from p2cccd.data import DatasetGenerationConfig, GeneratedDataset, generate_exact_oracle_dataset
from p2cccd.data.shards import default_metadata, write_npz_shard
from p2cccd.proposal.stpf_model import STPFModelPreset, build_stpf_model
from p2cccd.proposal.training import STPFEpochMetrics, STPFTrainingConfig, evaluate_stpf_model
from p2cccd.proposal.training_runner import STPFTrainingRunConfig, STPFTrainingRunResult, run_stpf_training

from .trained_stpf_high_density import (
    HighDensityMethodMetrics,
    HighDensitySTPFConfig,
    HighDensitySTPFWorkload,
    benchmark_no_proposal_on_high_density_workload,
    benchmark_stpf_on_high_density_workload,
    build_high_density_stpf_workload,
    workload_to_shard_dataset,
)


@dataclass(frozen=True, slots=True)
class T0SyntheticProxyExperimentConfig:
    train_mesh_count_per_split: int = 256
    train_robot_link_count: int = 128
    eval_mesh_count_per_split: int = 96
    eval_robot_link_count: int = 48
    seed: int = 22
    high_density: HighDensitySTPFConfig = HighDensitySTPFConfig()
    training: STPFTrainingConfig = STPFTrainingConfig(
        epochs=8,
        batch_size=2048,
        learning_rate=1.0e-3,
        seed=22,
        device="cuda",
        validation_fraction=0.0,
        model_preset=STPFModelPreset.LIGHTWEIGHT_MLP,
    )
    shard_root: str = "src/datasets/training/synthetic_proxy/shards"
    training_output_dir: str = "src/outputs/stpf_training"
    run_name: str = "t0_synthetic_proxy_20260422"
    benchmark_device: str = "cuda"


@dataclass(frozen=True, slots=True)
class T0SyntheticProxyArtifacts:
    shard_dir: Path
    manifest_path: Path
    report_path: Path
    summary_json_path: Path


@dataclass(frozen=True, slots=True)
class T0SyntheticProxyExperimentResult:
    config: T0SyntheticProxyExperimentConfig
    base_train_dataset: GeneratedDataset
    base_eval_dataset: GeneratedDataset
    dense_train_workload: HighDensitySTPFWorkload
    dense_eval_workload: HighDensitySTPFWorkload
    training_run: STPFTrainingRunResult
    base_eval_metrics: STPFEpochMetrics
    dense_eval_row_metrics: STPFEpochMetrics
    baseline: HighDensityMethodMetrics
    random_stpf: HighDensityMethodMetrics
    trained_stpf: HighDensityMethodMetrics
    artifacts: T0SyntheticProxyArtifacts

    @property
    def mixed_train_row_count(self) -> int:
        return len(self.base_train_dataset.rows) + len(self.dense_train_workload.rows)

    @property
    def mixed_eval_row_count(self) -> int:
        return len(self.base_eval_dataset.rows) + len(self.dense_eval_workload.rows)

    @property
    def trained_exact_work_reduction_vs_no_proposal(self) -> float:
        baseline = max(1.0e-9, self.baseline.exact_work_units)
        return 1.0 - (self.trained_stpf.exact_work_units / baseline)

    @property
    def trained_exact_work_reduction_vs_random(self) -> float:
        random_units = max(1.0e-9, self.random_stpf.exact_work_units)
        return 1.0 - (self.trained_stpf.exact_work_units / random_units)


def _write_generated_dataset_shard(
    dataset: GeneratedDataset,
    path: Path,
    *,
    seed: int,
    source: str,
    dataset_role: str,
    shard_name: str,
) -> None:
    metadata = default_metadata(dataset, seed=seed, source=source)
    metadata["dataset_role"] = dataset_role
    metadata["shard_name"] = shard_name
    write_npz_shard(path, dataset, metadata=metadata)


def _metrics_to_dict(metrics: STPFEpochMetrics) -> dict[str, float | int | str]:
    return {
        "epoch": metrics.epoch,
        "split": metrics.split,
        "row_count": metrics.row_count,
        "loss": round(metrics.loss, 6),
        "interval_top1_recall": round(metrics.interval_top1_recall, 6),
        "family_top2_recall": round(metrics.family_top2_recall, 6),
        "estimated_exact_work_reduction": round(metrics.estimated_exact_work_reduction, 6),
        "mean_predicted_cost": round(metrics.mean_predicted_cost, 6),
        "mean_target_cost": round(metrics.mean_target_cost, 6),
    }


def _high_density_method_to_dict(metrics: HighDensityMethodMetrics) -> dict[str, float | int | str]:
    return {
        "method_name": metrics.method_name,
        "query_count": metrics.query_count,
        "candidate_count": metrics.candidate_count,
        "avg_candidates_per_query": round(metrics.avg_candidates_per_query, 4),
        "fn_count": metrics.fn_count,
        "exact_call_count": metrics.exact_call_count,
        "fallback_call_count": metrics.fallback_call_count,
        "interval_hit_count": metrics.interval_hit_count,
        "interval_miss_count": metrics.interval_miss_count,
        "exact_work_units": round(metrics.exact_work_units, 4),
        "proposal_wall_ms": round(metrics.proposal_wall_ms, 4),
        "scheduling_wall_ms": round(metrics.scheduling_wall_ms, 4),
        "total_wall_ms": round(metrics.total_wall_ms, 4),
    }


def run_t0_synthetic_proxy_experiment(
    config: T0SyntheticProxyExperimentConfig | None = None,
) -> T0SyntheticProxyExperimentResult:
    cfg = config or T0SyntheticProxyExperimentConfig()
    shard_dir = Path(cfg.shard_root) / cfg.run_name
    shard_dir.mkdir(parents=True, exist_ok=True)

    base_train_dataset = generate_exact_oracle_dataset(
        DatasetGenerationConfig(
            mesh_count_per_split=cfg.train_mesh_count_per_split,
            robot_link_count=cfg.train_robot_link_count,
            seed=cfg.seed,
            include_robot_links=True,
        )
    )
    base_eval_dataset = generate_exact_oracle_dataset(
        DatasetGenerationConfig(
            mesh_count_per_split=cfg.eval_mesh_count_per_split,
            robot_link_count=cfg.eval_robot_link_count,
            seed=cfg.seed + 1,
            include_robot_links=True,
        )
    )

    dense_train_workload = build_high_density_stpf_workload(
        base_train_dataset,
        cfg.high_density,
        name="t0_dense_train",
    )
    dense_eval_workload = build_high_density_stpf_workload(
        base_eval_dataset,
        cfg.high_density,
        name="t0_dense_eval",
    )

    dense_train_dataset = workload_to_shard_dataset(dense_train_workload)
    dense_eval_dataset = workload_to_shard_dataset(dense_eval_workload)

    _write_generated_dataset_shard(
        base_train_dataset,
        shard_dir / "base_train.npz",
        seed=cfg.seed,
        source="t0_synthetic_proxy_base_train",
        dataset_role="training",
        shard_name="base_train",
    )
    _write_generated_dataset_shard(
        base_eval_dataset,
        shard_dir / "base_eval.npz",
        seed=cfg.seed + 1,
        source="t0_synthetic_proxy_base_eval",
        dataset_role="training_eval",
        shard_name="base_eval",
    )
    _write_generated_dataset_shard(
        dense_train_dataset,
        shard_dir / "dense_train.npz",
        seed=cfg.seed,
        source="t0_synthetic_proxy_dense_train",
        dataset_role="training",
        shard_name="dense_train",
    )
    _write_generated_dataset_shard(
        dense_eval_dataset,
        shard_dir / "dense_eval.npz",
        seed=cfg.seed + 1,
        source="t0_synthetic_proxy_dense_eval",
        dataset_role="training_eval",
        shard_name="dense_eval",
    )

    train_rows = list(base_train_dataset.rows) + list(dense_train_workload.rows)
    eval_rows = list(base_eval_dataset.rows) + list(dense_eval_workload.rows)
    training_run = run_stpf_training(
        train_rows,
        STPFTrainingRunConfig(
            training=cfg.training,
            output_dir=cfg.training_output_dir,
            run_name=cfg.run_name,
        ),
        validation_rows=eval_rows,
    )

    trained_model = training_run.result.model
    trained_model.eval()
    trained_model.to(cfg.benchmark_device)

    base_eval_metrics = evaluate_stpf_model(
        trained_model,
        base_eval_dataset.rows,
        cfg.training,
        epoch=cfg.training.epochs,
        split="base_eval",
    )
    dense_eval_row_metrics = evaluate_stpf_model(
        trained_model,
        dense_eval_workload.rows,
        cfg.training,
        epoch=cfg.training.epochs,
        split="dense_eval_rows",
    )

    baseline = benchmark_no_proposal_on_high_density_workload(dense_eval_workload)
    random_model = build_stpf_model(cfg.training.model_preset)
    random_model.eval()
    random_model.to(cfg.benchmark_device)
    random_stpf = benchmark_stpf_on_high_density_workload(
        dense_eval_workload,
        model=random_model,
        device=cfg.benchmark_device,
        proposal_batch_size=cfg.training.batch_size,
        method_name="RTSTPFExact-Random",
    )
    trained_stpf = benchmark_stpf_on_high_density_workload(
        dense_eval_workload,
        model=trained_model,
        device=cfg.benchmark_device,
        proposal_batch_size=cfg.training.batch_size,
        method_name="RTSTPFExact-Trained-T0",
    )

    manifest = {
        "run_name": cfg.run_name,
        "dataset_role": "training",
        "base_train_rows": len(base_train_dataset.rows),
        "base_eval_rows": len(base_eval_dataset.rows),
        "dense_train_rows": len(dense_train_workload.rows),
        "dense_eval_rows": len(dense_eval_workload.rows),
        "mixed_train_rows": len(train_rows),
        "mixed_eval_rows": len(eval_rows),
        "train_query_count": len(base_train_dataset.samples),
        "eval_query_count": len(base_eval_dataset.samples),
        "dense_avg_candidates_per_query": dense_eval_workload.avg_candidates_per_query,
        "seed": cfg.seed,
    }
    manifest_path = shard_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    artifacts = T0SyntheticProxyArtifacts(
        shard_dir=shard_dir,
        manifest_path=manifest_path,
        report_path=Path("src/benchmark") / f"{cfg.run_name}_report.md",
        summary_json_path=Path("src/benchmark") / f"{cfg.run_name}_summary.json",
    )
    return T0SyntheticProxyExperimentResult(
        config=cfg,
        base_train_dataset=base_train_dataset,
        base_eval_dataset=base_eval_dataset,
        dense_train_workload=dense_train_workload,
        dense_eval_workload=dense_eval_workload,
        training_run=training_run,
        base_eval_metrics=base_eval_metrics,
        dense_eval_row_metrics=dense_eval_row_metrics,
        baseline=baseline,
        random_stpf=random_stpf,
        trained_stpf=trained_stpf,
        artifacts=artifacts,
    )


def write_t0_synthetic_proxy_report(
    path: str | Path,
    result: T0SyntheticProxyExperimentResult,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "base_eval_metrics": _metrics_to_dict(result.base_eval_metrics),
        "dense_eval_row_metrics": _metrics_to_dict(result.dense_eval_row_metrics),
        "baseline_no_proposal": _high_density_method_to_dict(result.baseline),
        "random_stpf": _high_density_method_to_dict(result.random_stpf),
        "trained_stpf": _high_density_method_to_dict(result.trained_stpf),
        "trained_exact_work_reduction_vs_no_proposal": round(
            result.trained_exact_work_reduction_vs_no_proposal, 6
        ),
        "trained_exact_work_reduction_vs_random": round(
            result.trained_exact_work_reduction_vs_random, 6
        ),
    }
    history = result.training_run.result.history
    final_validation = next((metric for metric in reversed(history) if metric.split == "validation"), None)

    lines = [
        "# T0 synthetic proxy training and evaluation report",
        "",
        "## dataset",
        "",
        f"- run name: `{result.config.run_name}`",
        f"- shard dir: `{result.artifacts.shard_dir}`",
        f"- base train queries: `{len(result.base_train_dataset.samples)}`",
        f"- base eval queries: `{len(result.base_eval_dataset.samples)}`",
        f"- dense train rows: `{len(result.dense_train_workload.rows)}`",
        f"- dense eval rows: `{len(result.dense_eval_workload.rows)}`",
        f"- mixed train rows: `{result.mixed_train_row_count}`",
        f"- mixed eval rows: `{result.mixed_eval_row_count}`",
        f"- dense avg candidates/query: `{result.dense_eval_workload.avg_candidates_per_query:.3f}`",
        "",
        "## description",
        "",
        f"- training output dir: `{result.training_run.artifacts.output_dir}`",
        f"- model state: `{result.training_run.artifacts.model_state_path}`",
        f"- final train loss: `{result.training_run.final_train_loss:.6f}`",
        f"- final validation loss: `{result.training_run.final_validation_loss:.6f}`",
    ]
    if final_validation is not None:
        lines.extend(
            [
                f"- validation interval top1 recall: `{final_validation.interval_top1_recall:.4f}`",
                f"- validation family top2 recall: `{final_validation.family_top2_recall:.4f}`",
                f"- validation estimated exact work reduction: `{final_validation.estimated_exact_work_reduction:.4f}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Dense Eval Benchmark",
            "",
            f"- NoProposal exact work: `{result.baseline.exact_work_units:.4f}`",
            f"- Random STPF exact work: `{result.random_stpf.exact_work_units:.4f}`",
            f"- Trained STPF exact work: `{result.trained_stpf.exact_work_units:.4f}`",
            f"- Trained vs NoProposal reduction: `{result.trained_exact_work_reduction_vs_no_proposal:.4%}`",
            f"- Trained vs Random reduction: `{result.trained_exact_work_reduction_vs_random:.4%}`",
            f"- Trained fn_count: `{result.trained_stpf.fn_count}`",
            "",
            "## Metrics JSON",
            "",
            "```json",
            json.dumps(payload, indent=2, sort_keys=True),
            "```",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def write_t0_synthetic_proxy_summary_json(
    path: str | Path,
    result: T0SyntheticProxyExperimentResult,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_name": result.config.run_name,
        "shard_dir": str(result.artifacts.shard_dir),
        "training_output_dir": str(result.training_run.artifacts.output_dir),
        "model_state_path": None
        if result.training_run.artifacts.model_state_path is None
        else str(result.training_run.artifacts.model_state_path),
        "base_eval_metrics": _metrics_to_dict(result.base_eval_metrics),
        "dense_eval_row_metrics": _metrics_to_dict(result.dense_eval_row_metrics),
        "baseline_no_proposal": _high_density_method_to_dict(result.baseline),
        "random_stpf": _high_density_method_to_dict(result.random_stpf),
        "trained_stpf": _high_density_method_to_dict(result.trained_stpf),
        "trained_exact_work_reduction_vs_no_proposal": result.trained_exact_work_reduction_vs_no_proposal,
        "trained_exact_work_reduction_vs_random": result.trained_exact_work_reduction_vs_random,
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


__all__ = [
    "T0SyntheticProxyArtifacts",
    "T0SyntheticProxyExperimentConfig",
    "T0SyntheticProxyExperimentResult",
    "run_t0_synthetic_proxy_experiment",
    "write_t0_synthetic_proxy_report",
    "write_t0_synthetic_proxy_summary_json",
]
