from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import torch

from p2cccd.data import default_metadata, write_npz_shard
from p2cccd.datasets.objects.thingi10k_training import (
    Thingi10KProxyDatasetBundle,
    Thingi10KProxyDatasetConfig,
    generate_thingi10k_proxy_datasets,
)
from p2cccd.proposal.stpf_model import STPFModelPreset, build_stpf_model
from p2cccd.proposal.training import STPFEpochMetrics, STPFTrainingConfig, evaluate_stpf_model
from p2cccd.proposal.training_runner import STPFTrainingRunConfig, STPFTrainingRunResult, run_stpf_training

from .t0_synthetic_proxy import _high_density_method_to_dict, _metrics_to_dict
from .trained_stpf_high_density import (
    HighDensityMethodMetrics,
    HighDensitySTPFConfig,
    HighDensitySTPFWorkload,
    benchmark_no_proposal_on_high_density_workload,
    benchmark_stpf_on_high_density_workload,
    build_high_density_stpf_workload,
    workload_to_shard_dataset,
)


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass(frozen=True, slots=True)
class Thingi10KTrainingExperimentConfig:
    dataset: Thingi10KProxyDatasetConfig = Thingi10KProxyDatasetConfig()
    high_density: HighDensitySTPFConfig = HighDensitySTPFConfig(
        slab_count=12,
        patches_per_object=6,
        representative_attempt_limit=2,
        uncertainty_fallback_threshold=0.75,
    )
    training: STPFTrainingConfig = STPFTrainingConfig(
        epochs=8,
        batch_size=2048,
        learning_rate=1.0e-3,
        seed=424242,
        device=_default_device(),
        validation_fraction=0.0,
        model_preset=STPFModelPreset.LIGHTWEIGHT_MLP,
    )
    shard_root: str = "src/datasets/training/ood_train/thingi10k/shards"
    training_output_dir: str = "src/outputs/stpf_training"
    run_name: str = "thingi10k_training_run_id"
    benchmark_device: str = _default_device()


@dataclass(frozen=True, slots=True)
class Thingi10KTrainingArtifacts:
    shard_dir: Path
    manifest_path: Path
    report_path: Path
    summary_json_path: Path


@dataclass(frozen=True, slots=True)
class Thingi10KTrainingExperimentResult:
    config: Thingi10KTrainingExperimentConfig
    bundle: Thingi10KProxyDatasetBundle
    dense_train_workload: HighDensitySTPFWorkload
    dense_eval_workload: HighDensitySTPFWorkload
    training_run: STPFTrainingRunResult
    base_eval_metrics: STPFEpochMetrics
    dense_eval_row_metrics: STPFEpochMetrics
    baseline: HighDensityMethodMetrics
    random_stpf: HighDensityMethodMetrics
    trained_stpf: HighDensityMethodMetrics
    artifacts: Thingi10KTrainingArtifacts

    @property
    def mixed_train_row_count(self) -> int:
        return len(self.bundle.train_dataset.rows) + len(self.dense_train_workload.rows)

    @property
    def mixed_eval_row_count(self) -> int:
        return len(self.bundle.eval_dataset.rows) + len(self.dense_eval_workload.rows)

    @property
    def trained_exact_work_reduction_vs_no_proposal(self) -> float:
        baseline = max(1.0e-9, self.baseline.exact_work_units)
        return 1.0 - (self.trained_stpf.exact_work_units / baseline)

    @property
    def trained_exact_work_reduction_vs_random(self) -> float:
        random_units = max(1.0e-9, self.random_stpf.exact_work_units)
        return 1.0 - (self.trained_stpf.exact_work_units / random_units)


def _write_dataset_shard(
    *,
    output_path: Path,
    source: str,
    dataset_role: str,
    shard_name: str,
    dataset,
    seed: int,
) -> None:
    metadata = default_metadata(dataset, seed=seed, source=source)
    metadata["dataset_role"] = dataset_role
    metadata["shard_name"] = shard_name
    metadata["source_dataset"] = "Thingi10K"
    write_npz_shard(output_path, dataset, metadata=metadata)


def run_thingi10k_training_experiment(
    config: Thingi10KTrainingExperimentConfig | None = None,
) -> Thingi10KTrainingExperimentResult:
    cfg = config or Thingi10KTrainingExperimentConfig()
    shard_dir = Path(cfg.shard_root) / cfg.run_name
    shard_dir.mkdir(parents=True, exist_ok=True)

    bundle = generate_thingi10k_proxy_datasets(cfg.dataset)
    dense_train_workload = build_high_density_stpf_workload(
        bundle.train_dataset,
        cfg.high_density,
        name="thingi10k_dense_train",
    )
    dense_eval_workload = build_high_density_stpf_workload(
        bundle.eval_dataset,
        cfg.high_density,
        name="thingi10k_dense_eval",
    )
    dense_train_dataset = workload_to_shard_dataset(dense_train_workload)
    dense_eval_dataset = workload_to_shard_dataset(dense_eval_workload)

    _write_dataset_shard(
        output_path=shard_dir / "base_train.npz",
        source="thingi10k_proxy_train",
        dataset_role="training",
        shard_name="base_train",
        dataset=bundle.train_dataset,
        seed=cfg.training.seed,
    )
    _write_dataset_shard(
        output_path=shard_dir / "base_eval.npz",
        source="thingi10k_proxy_eval",
        dataset_role="training_eval",
        shard_name="base_eval",
        dataset=bundle.eval_dataset,
        seed=cfg.training.seed + 1,
    )
    _write_dataset_shard(
        output_path=shard_dir / "dense_train.npz",
        source="thingi10k_dense_train",
        dataset_role="training",
        shard_name="dense_train",
        dataset=dense_train_dataset,
        seed=cfg.training.seed,
    )
    _write_dataset_shard(
        output_path=shard_dir / "dense_eval.npz",
        source="thingi10k_dense_eval",
        dataset_role="training_eval",
        shard_name="dense_eval",
        dataset=dense_eval_dataset,
        seed=cfg.training.seed + 1,
    )

    train_rows = list(bundle.train_dataset.rows) + list(dense_train_workload.rows)
    eval_rows = list(bundle.eval_dataset.rows) + list(dense_eval_workload.rows)
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
        bundle.eval_dataset.rows,
        cfg.training,
        epoch=cfg.training.epochs,
        split="thingi10k_base_eval",
    )
    dense_eval_row_metrics = evaluate_stpf_model(
        trained_model,
        dense_eval_workload.rows,
        cfg.training,
        epoch=cfg.training.epochs,
        split="thingi10k_dense_eval_rows",
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
        method_name="Thingi10K-RTSTPFExact-Random",
    )
    trained_stpf = benchmark_stpf_on_high_density_workload(
        dense_eval_workload,
        model=trained_model,
        device=cfg.benchmark_device,
        proposal_batch_size=cfg.training.batch_size,
        method_name="Thingi10K-RTSTPFExact-Trained",
    )

    manifest = {
        "run_name": cfg.run_name,
        "source_root": str(bundle.source_root),
        "asset_count": len(bundle.assets),
        "train_pair_count": len(bundle.train_pairs),
        "eval_pair_count": len(bundle.eval_pairs),
        "base_train_rows": len(bundle.train_dataset.rows),
        "base_eval_rows": len(bundle.eval_dataset.rows),
        "dense_train_rows": len(dense_train_workload.rows),
        "dense_eval_rows": len(dense_eval_workload.rows),
        "mixed_train_rows": len(train_rows),
        "mixed_eval_rows": len(eval_rows),
        "train_query_count": len(bundle.train_dataset.samples),
        "eval_query_count": len(bundle.eval_dataset.samples),
        "dense_avg_candidates_per_query": dense_eval_workload.avg_candidates_per_query,
        "seed": cfg.training.seed,
    }
    manifest_path = shard_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    artifacts = Thingi10KTrainingArtifacts(
        shard_dir=shard_dir,
        manifest_path=manifest_path,
        report_path=Path("src/benchmark") / f"{cfg.run_name}.md",
        summary_json_path=Path("src/benchmark") / f"{cfg.run_name}.json",
    )
    return Thingi10KTrainingExperimentResult(
        config=cfg,
        bundle=bundle,
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


def write_thingi10k_training_report(
    path: str | Path,
    result: Thingi10KTrainingExperimentResult,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_validation = next(
        (metric for metric in reversed(result.training_run.result.history) if metric.split == "validation"),
        None,
    )
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
    lines = [
        "# Thingi10K training and evaluation report",
        "",
        "## data source",
        "",
        f"- source root: `{result.bundle.source_root}`",
        f"- asset count: `{len(result.bundle.assets)}`",
        f"- train pair count: `{len(result.bundle.train_pairs)}`",
        f"- eval pair count: `{len(result.bundle.eval_pairs)}`",
        "",
        "## dataset",
        "",
        f"- shard dir: `{result.artifacts.shard_dir}`",
        f"- base train queries: `{len(result.bundle.train_dataset.samples)}`",
        f"- base eval queries: `{len(result.bundle.eval_dataset.samples)}`",
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


def write_thingi10k_training_summary_json(
    path: str | Path,
    result: Thingi10KTrainingExperimentResult,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_name": result.config.run_name,
        "source_root": str(result.bundle.source_root),
        "asset_count": len(result.bundle.assets),
        "train_pair_count": len(result.bundle.train_pairs),
        "eval_pair_count": len(result.bundle.eval_pairs),
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
    "Thingi10KTrainingArtifacts",
    "Thingi10KTrainingExperimentConfig",
    "Thingi10KTrainingExperimentResult",
    "run_thingi10k_training_experiment",
    "write_thingi10k_training_report",
    "write_thingi10k_training_summary_json",
]
