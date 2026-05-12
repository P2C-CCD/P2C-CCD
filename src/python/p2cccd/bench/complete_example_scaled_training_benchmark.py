from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

from p2cccd.data import DatasetGenerationConfig, GeneratedDataset, generate_exact_oracle_dataset
from p2cccd.datasets.cad import ABCProxyDatasetConfig, default_abc_official_root
from p2cccd.datasets.objects.thingi10k_training import (
    Thingi10KOfficialSubsetConfig,
    Thingi10KProxyDatasetConfig,
)
from p2cccd.proposal.stpf_model import STPFModelPreset
from p2cccd.proposal.training import STPFTrainingConfig

from .abc_training import ABCTrainingExperimentConfig, ABCTrainingExperimentResult, run_abc_training_experiment
from .bvh_exact import BVHExactConfig, BVHExactResult, run_bvh_exact_on_generated_dataset
from .no_proposal import NoProposalConfig, NoProposalResult, run_no_proposal_on_generated_dataset
from .pure_exact_cpu import PureExactCPUConfig, PureExactCPUResult, run_pure_exact_cpu_on_generated_dataset
from .rt_exact import RTExactConfig, RTExactResult, run_rt_exact_on_generated_dataset
from .rt_stpf_exact import RTSTPFExactConfig, RTSTPFExactResult, run_rt_stpf_exact_on_generated_dataset
from .t0_synthetic_proxy import T0SyntheticProxyExperimentConfig, T0SyntheticProxyExperimentResult, run_t0_synthetic_proxy_experiment
from .thingi10k_training import (
    Thingi10KTrainingExperimentConfig,
    Thingi10KTrainingExperimentResult,
    run_thingi10k_training_experiment,
)
from .trained_stpf_high_density import (
    HighDensitySTPFConfig,
    TrainedSTPFHighDensityExperimentResult,
    run_trained_stpf_high_density_experiment,
)


def _benchmark_root() -> Path:
    return Path("src/benchmark")


def _default_exact() -> PureExactCPUConfig:
    return PureExactCPUConfig(
        eps_time=1.0e-5,
        eps_space=1.0e-8,
        max_subdivision_depth=24,
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
class _MethodRow:
    method: str
    query_count: int
    total_ms: float
    rt_ms: float
    proposal_ms: float
    exact_ms: float
    qps: float
    fn_count: int
    candidate_recall: float
    avg_candidates: float
    avg_exact_evals: float
    exact_backend_name: str
    resolved_execution_profile_name: str = "n/a"
    inference_backend_name: str = "n/a"
    inference_provider_name: str = "n/a"


@dataclass(frozen=True, slots=True)
class _ExampleBenchmarkResult:
    example_name: str
    source_name: str
    scene_name: str
    train_query_count: int
    eval_query_count: int
    checkpoint_path: str
    checkpoint_note: str
    training_note: str
    final_validation_interval_top1_recall: float
    rows: tuple[_MethodRow, ...]


@dataclass(frozen=True, slots=True)
class CompleteExampleScaledTrainingBenchmarkConfig:
    exact: PureExactCPUConfig = _default_exact()
    training: STPFTrainingConfig = _default_training()
    rt_backend_name: str = "optix_rt"
    model_device: str = "cuda"
    proposal_batch_size: int = 4096
    training_output_dir: str = "src/outputs/stpf_training"
    benchmark_output_dir: str = "src/benchmark"
    run_name: str = "complete_example_scaled_training_benchmark_run_id"


@dataclass(frozen=True, slots=True)
class CompleteExampleScaledTrainingBenchmarkResult:
    config: CompleteExampleScaledTrainingBenchmarkConfig
    examples: tuple[_ExampleBenchmarkResult, ...]
    report_path: Path
    summary_json_path: Path


def _rtstpf_config(
    cfg: CompleteExampleScaledTrainingBenchmarkConfig,
    *,
    checkpoint_path: str,
) -> RTSTPFExactConfig:
    return RTSTPFExactConfig(
        exact=cfg.exact,
        rt_backend_name=cfg.rt_backend_name,
        enable_cuda_exact=True,
        execution_profile="fastest_learned",
        allow_default_model=False,
        model_checkpoint_path=checkpoint_path,
        model_device=cfg.model_device,
        model_preset=STPFModelPreset.MEDIUM_MLP,
        inference_backend="ort",
        ort_prefer_tensorrt=True,
        ort_allow_cuda_fallback=False,
        ort_allow_cpu_fallback=False,
        proposal_batch_size=cfg.proposal_batch_size,
        cpu_inference_row_threshold=0,
    )


def _row_from_pure(result: PureExactCPUResult) -> _MethodRow:
    row = result.benchmark
    return _MethodRow(
        method="PureExactCPU",
        query_count=row.query_count,
        total_ms=row.total_ms,
        rt_ms=row.rt_ms,
        proposal_ms=row.proposal_ms,
        exact_ms=row.exact_ms,
        qps=row.qps,
        fn_count=row.fn_count,
        candidate_recall=row.candidate_recall,
        avg_candidates=row.avg_candidates,
        avg_exact_evals=row.avg_exact_evals,
        exact_backend_name="swept_sphere_oracle_cpu",
    )


def _row_from_bvh(result: BVHExactResult) -> _MethodRow:
    row = result.benchmark
    return _MethodRow(
        method="BVHExact",
        query_count=row.query_count,
        total_ms=row.total_ms,
        rt_ms=row.rt_ms,
        proposal_ms=row.proposal_ms,
        exact_ms=row.exact_ms,
        qps=row.qps,
        fn_count=row.fn_count,
        candidate_recall=row.candidate_recall,
        avg_candidates=row.avg_candidates,
        avg_exact_evals=row.avg_exact_evals,
        exact_backend_name=f"{result.broad_phase_stats.backend_name}; swept_sphere_oracle_cpu",
    )


def _row_from_rt_exact(result: RTExactResult) -> _MethodRow:
    row = result.benchmark
    return _MethodRow(
        method="RTExact",
        query_count=row.query_count,
        total_ms=row.total_ms,
        rt_ms=row.rt_ms,
        proposal_ms=row.proposal_ms,
        exact_ms=row.exact_ms,
        qps=row.qps,
        fn_count=row.fn_count,
        candidate_recall=row.candidate_recall,
        avg_candidates=row.avg_candidates,
        avg_exact_evals=row.avg_exact_evals,
        exact_backend_name=result.exact_backend_name,
    )


def _row_from_no_proposal(result: NoProposalResult) -> _MethodRow:
    row = result.benchmark
    return _MethodRow(
        method="NoProposal",
        query_count=row.query_count,
        total_ms=row.total_ms,
        rt_ms=row.rt_ms,
        proposal_ms=row.proposal_ms,
        exact_ms=row.exact_ms,
        qps=row.qps,
        fn_count=row.fn_count,
        candidate_recall=row.candidate_recall,
        avg_candidates=row.avg_candidates,
        avg_exact_evals=row.avg_exact_evals,
        exact_backend_name=result.exact_backend_name,
    )


def _row_from_rtstpf(result: RTSTPFExactResult) -> _MethodRow:
    row = result.benchmark
    return _MethodRow(
        method="RTSTPFExact",
        query_count=row.query_count,
        total_ms=row.total_ms,
        rt_ms=row.rt_ms,
        proposal_ms=row.proposal_ms,
        exact_ms=row.exact_ms,
        qps=row.qps,
        fn_count=row.fn_count,
        candidate_recall=row.candidate_recall,
        avg_candidates=row.avg_candidates,
        avg_exact_evals=row.avg_exact_evals,
        exact_backend_name=result.exact_backend_name,
        resolved_execution_profile_name=result.resolved_execution_profile_name,
        inference_backend_name=result.inference_backend_name,
        inference_provider_name=result.inference_provider_name,
    )


def _benchmark_dataset(
    cfg: CompleteExampleScaledTrainingBenchmarkConfig,
    *,
    dataset: GeneratedDataset,
    checkpoint_path: str,
) -> tuple[_MethodRow, ...]:
    pure = run_pure_exact_cpu_on_generated_dataset(dataset)
    bvh = run_bvh_exact_on_generated_dataset(dataset, BVHExactConfig(exact=cfg.exact))
    rt_exact = run_rt_exact_on_generated_dataset(
        dataset,
        RTExactConfig(
            exact=cfg.exact,
            backend_name=cfg.rt_backend_name,
            enable_cuda_exact=True,
        ),
    )
    rtstpf = run_rt_stpf_exact_on_generated_dataset(
        dataset,
        _rtstpf_config(cfg, checkpoint_path=checkpoint_path),
        device=cfg.model_device,
    )
    no_proposal = run_no_proposal_on_generated_dataset(
        dataset,
        NoProposalConfig(
            exact=cfg.exact,
            rt_backend_name=cfg.rt_backend_name,
            enable_cuda_exact=True,
        ),
    )
    return (
        _row_from_pure(pure),
        _row_from_bvh(bvh),
        _row_from_rt_exact(rt_exact),
        _row_from_rtstpf(rtstpf),
        _row_from_no_proposal(no_proposal),
    )


def _run_t0_scaled(
    cfg: CompleteExampleScaledTrainingBenchmarkConfig,
) -> _ExampleBenchmarkResult:
    experiment = run_t0_synthetic_proxy_experiment(
        T0SyntheticProxyExperimentConfig(
            train_mesh_count_per_split=900,
            train_robot_link_count=500,
            eval_mesh_count_per_split=1800,
            eval_robot_link_count=1000,
            seed=2401,
            training=cfg.training,
            training_output_dir=cfg.training_output_dir,
            run_name=f"{cfg.run_name}_t0_q10000",
            benchmark_device=cfg.model_device,
        )
    )
    checkpoint_path = str(experiment.training_run.artifacts.model_state_path)
    rows = _benchmark_dataset(cfg, dataset=experiment.base_eval_dataset, checkpoint_path=checkpoint_path)
    return _ExampleBenchmarkResult(
        example_name="T0 synthetic_proxy",
        source_name="analytic_swept_sphere_proxy",
        scene_name="scaled_q10000",
        train_query_count=len(experiment.base_train_dataset.samples),
        eval_query_count=len(experiment.base_eval_dataset.samples),
        checkpoint_path=checkpoint_path,
        checkpoint_note="Freshly trained scaled T0 checkpoint.",
        training_note="Train queries=5000, eval queries=10000, medium_mlp + ort_tensorrt benchmark path.",
        final_validation_interval_top1_recall=experiment.base_eval_metrics.interval_top1_recall,
        rows=rows,
    )


def _run_high_density_scaled(
    cfg: CompleteExampleScaledTrainingBenchmarkConfig,
) -> _ExampleBenchmarkResult:
    train_dataset = generate_exact_oracle_dataset(
        DatasetGenerationConfig(
            mesh_count_per_split=800,
            robot_link_count=0,
            seed=3401,
            include_robot_links=False,
        )
    )
    eval_dataset = generate_exact_oracle_dataset(
        DatasetGenerationConfig(
            mesh_count_per_split=200,
            robot_link_count=0,
            seed=3402,
            include_robot_links=False,
        )
    )
    experiment = run_trained_stpf_high_density_experiment(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        training_output_dir=cfg.training_output_dir,
        run_name=f"{cfg.run_name}_high_density_q1000",
        training_device=cfg.training.device,
        benchmark_device=cfg.model_device,
        model_preset=cfg.training.model_preset,
        epochs=cfg.training.epochs,
        batch_size=cfg.training.batch_size,
        learning_rate=cfg.training.learning_rate,
        seed=4401,
    )
    checkpoint_path = str(experiment.training_run.artifacts.model_state_path)
    rows = _benchmark_dataset(cfg, dataset=eval_dataset, checkpoint_path=checkpoint_path)
    validation_metrics = next(
        metric for metric in reversed(experiment.training_run.result.history) if metric.split == "validation"
    )
    return _ExampleBenchmarkResult(
        example_name="trained_stpf_high_density",
        source_name="analytic_swept_sphere_proxy",
        scene_name="scaled_q1000",
        train_query_count=len(train_dataset.samples),
        eval_query_count=len(eval_dataset.samples),
        checkpoint_path=checkpoint_path,
        checkpoint_note="Freshly trained scaled high-density checkpoint.",
        training_note="Train queries=4000, eval queries=1000, benchmark runs on query-level eval set.",
        final_validation_interval_top1_recall=validation_metrics.interval_top1_recall,
        rows=rows,
    )


def _run_abc_scaled(
    cfg: CompleteExampleScaledTrainingBenchmarkConfig,
) -> _ExampleBenchmarkResult:
    experiment = run_abc_training_experiment(
        ABCTrainingExperimentConfig(
            dataset=ABCProxyDatasetConfig(
                root=default_abc_official_root(),
                allow_demo_bootstrap=False,
                asset_limit=96,
                pair_limit=2000,
                train_fraction=0.5,
                seed=5401,
            ),
            training=cfg.training,
            training_output_dir=cfg.training_output_dir,
            run_name=f"{cfg.run_name}_abc_q4000",
            benchmark_device=cfg.model_device,
        )
    )
    checkpoint_path = str(experiment.training_run.artifacts.model_state_path)
    rows = _benchmark_dataset(cfg, dataset=experiment.bundle.eval_dataset, checkpoint_path=checkpoint_path)
    return _ExampleBenchmarkResult(
        example_name="ABC CAD",
        source_name="ABC official CAD",
        scene_name="scaled_q4000",
        train_query_count=len(experiment.bundle.train_dataset.samples),
        eval_query_count=len(experiment.bundle.eval_dataset.samples),
        checkpoint_path=checkpoint_path,
        checkpoint_note="Freshly trained scaled ABC checkpoint.",
        training_note="Official ABC root, 2000 pairs total, 50/50 split -> 4000 eval queries.",
        final_validation_interval_top1_recall=experiment.base_eval_metrics.interval_top1_recall,
        rows=rows,
    )


def _run_thingi_scaled(
    cfg: CompleteExampleScaledTrainingBenchmarkConfig,
) -> _ExampleBenchmarkResult:
    experiment = run_thingi10k_training_experiment(
        Thingi10KTrainingExperimentConfig(
            dataset=Thingi10KProxyDatasetConfig(
                subset=Thingi10KOfficialSubsetConfig(asset_limit=96),
                train_fraction=0.5,
                train_pair_limit=1000,
                eval_pair_limit=1000,
                seed=6401,
            ),
            training=cfg.training,
            training_output_dir=cfg.training_output_dir,
            run_name=f"{cfg.run_name}_thingi_q4000",
            benchmark_device=cfg.model_device,
        )
    )
    checkpoint_path = str(experiment.training_run.artifacts.model_state_path)
    rows = _benchmark_dataset(cfg, dataset=experiment.bundle.eval_dataset, checkpoint_path=checkpoint_path)
    return _ExampleBenchmarkResult(
        example_name="Thingi10K",
        source_name="Thingi10K official subset",
        scene_name="scaled_q4000",
        train_query_count=len(experiment.bundle.train_dataset.samples),
        eval_query_count=len(experiment.bundle.eval_dataset.samples),
        checkpoint_path=checkpoint_path,
        checkpoint_note="Freshly trained scaled Thingi10K checkpoint.",
        training_note="Official subset asset_limit=96, 50/50 asset split, eval_pair_limit=1000 -> 4000 eval queries.",
        final_validation_interval_top1_recall=experiment.base_eval_metrics.interval_top1_recall,
        rows=rows,
    )


def write_complete_example_scaled_training_benchmark_report(
    path: str | Path,
    result: CompleteExampleScaledTrainingBenchmarkResult,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Complete example scaled training and benchmark",
        "",
        "## Protocol",
        "",
        "- 4 descriptioncomplete exampledescriptionnewdescription, descriptionusedescription checkpoint. ",
        "- query descriptioncoverage `1,000` to `10,000`. ",
        "- `RTSTPFExact` fixedas learned-only: `medium_mlp + ORT(TensorRT EP) + optix_rt + cuda_exact`. ",
        "",
    ]
    for example in result.examples:
        lines.extend(
            [
                f"## {example.example_name}",
                "",
                f"- Source: `{example.source_name}`",
                f"- Scene: `{example.scene_name}`",
                f"- Train queries: `{example.train_query_count}`",
                f"- Eval queries: `{example.eval_query_count}`",
                f"- Checkpoint: `{example.checkpoint_path}`",
                f"- Validation interval top1 recall: `{example.final_validation_interval_top1_recall:.4f}`",
                f"- Note: {example.training_note}",
                "",
                "| Method | Total(ms) | RT(ms) | Proposal(ms) | Exact(ms) | QPS | FN | Recall | AvgCandidates | AvgExactEvals | Backend |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for row in example.rows:
            backend = row.exact_backend_name
            if row.method == "RTSTPFExact":
                backend = (
                    f"{backend}; {row.inference_backend_name}/{row.inference_provider_name}; "
                    f"{row.resolved_execution_profile_name}"
                )
            lines.append(
                f"| `{row.method}` | `{row.total_ms:.4f}` | `{row.rt_ms:.4f}` | `{row.proposal_ms:.4f}` | "
                f"`{row.exact_ms:.4f}` | `{row.qps:.2f}` | `{row.fn_count}` | `{row.candidate_recall:.4f}` | "
                f"`{row.avg_candidates:.4f}` | `{row.avg_exact_evals:.4f}` | `{backend}` |"
            )
        fastest = min(example.rows, key=lambda item: item.total_ms)
        lines.extend(
            [
                "",
                f"- Fastest method: `{fastest.method}` with `{fastest.total_ms:.4f} ms`.",
                "",
            ]
        )
    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return output_path


def write_complete_example_scaled_training_benchmark_json(
    path: str | Path,
    result: CompleteExampleScaledTrainingBenchmarkResult,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": asdict(result.config),
        "examples": [
            {
                "example_name": example.example_name,
                "source_name": example.source_name,
                "scene_name": example.scene_name,
                "train_query_count": example.train_query_count,
                "eval_query_count": example.eval_query_count,
                "checkpoint_path": example.checkpoint_path,
                "checkpoint_note": example.checkpoint_note,
                "training_note": example.training_note,
                "final_validation_interval_top1_recall": example.final_validation_interval_top1_recall,
                "rows": [asdict(row) for row in example.rows],
            }
            for example in result.examples
        ],
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def run_complete_example_scaled_training_benchmark(
    config: CompleteExampleScaledTrainingBenchmarkConfig | None = None,
) -> CompleteExampleScaledTrainingBenchmarkResult:
    cfg = config or CompleteExampleScaledTrainingBenchmarkConfig()
    output_root = Path(cfg.benchmark_output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    examples = (
        _run_t0_scaled(cfg),
        _run_high_density_scaled(cfg),
        _run_abc_scaled(cfg),
        _run_thingi_scaled(cfg),
    )
    report_path = output_root / f"{cfg.run_name}.md"
    summary_json_path = output_root / f"{cfg.run_name}.json"
    result = CompleteExampleScaledTrainingBenchmarkResult(
        config=cfg,
        examples=examples,
        report_path=report_path,
        summary_json_path=summary_json_path,
    )
    write_complete_example_scaled_training_benchmark_report(report_path, result)
    write_complete_example_scaled_training_benchmark_json(summary_json_path, result)
    return result


__all__ = [
    "CompleteExampleScaledTrainingBenchmarkConfig",
    "CompleteExampleScaledTrainingBenchmarkResult",
    "run_complete_example_scaled_training_benchmark",
    "write_complete_example_scaled_training_benchmark_json",
    "write_complete_example_scaled_training_benchmark_report",
]
