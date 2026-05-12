from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Callable

from p2cccd.data import DatasetGenerationConfig, GeneratedDataset, generate_exact_oracle_dataset

from .abc_paper_benchmark import ABCPaperBenchmarkConfig, build_abc_paper_benchmark_dataset
from .bvh_exact import BVHExactConfig, BVHExactResult, run_bvh_exact_on_generated_dataset
from .no_proposal import NoProposalConfig, NoProposalResult, run_no_proposal_on_generated_dataset
from .pure_exact_cpu import PureExactCPUConfig, PureExactCPUResult, run_pure_exact_cpu_on_generated_dataset
from .rt_exact import RTExactConfig, RTExactResult, run_rt_exact_on_generated_dataset
from .rt_stpf_exact import RTSTPFExactConfig, RTSTPFExactResult, run_rt_stpf_exact_on_generated_dataset
from .t0_synthetic_proxy import T0SyntheticProxyExperimentConfig
from .thingi10k_paper_benchmark import Thingi10KPaperBenchmarkConfig, build_thingi10k_paper_benchmark_dataset


def _benchmark_root() -> Path:
    return Path("src/benchmark")


def _default_exact() -> PureExactCPUConfig:
    return PureExactCPUConfig(
        eps_time=1.0e-5,
        eps_space=1.0e-8,
        max_subdivision_depth=24,
    )


def _t0_checkpoint() -> str:
    return str(
        Path("src/outputs/stpf_training")
        / "t0_synthetic_proxy_20260422_main"
        / "model_state.pt"
    )


def _trained_high_density_checkpoint() -> str:
    return str(
        Path("src/outputs/stpf_training")
        / "trained_stpf_high_density_20260421_final"
        / "model_state.pt"
    )


def _abc_checkpoint() -> str:
    return str(
        Path("src/outputs/stpf_training")
        / "abc_training_20260422_demo_main"
        / "model_state.pt"
    )


def _thingi10k_checkpoint() -> str:
    return str(
        Path("src/outputs/stpf_training")
        / "thingi10k_training_run_id"
        / "model_state.pt"
    )


@dataclass(frozen=True, slots=True)
class _ExampleSpec:
    example_name: str
    source_name: str
    scene_name: str
    dataset_note: str
    checkpoint_path: str
    checkpoint_note: str
    dataset_builder: Callable[[PureExactCPUConfig], GeneratedDataset]


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
class _ExampleResult:
    example_name: str
    source_name: str
    scene_name: str
    dataset_note: str
    checkpoint_path: str
    checkpoint_note: str
    query_count: int
    rows: tuple[_MethodRow, ...]


@dataclass(frozen=True, slots=True)
class CompleteExampleInferenceSweepConfig:
    exact: PureExactCPUConfig = _default_exact()
    rt_backend_name: str = "optix_rt"
    model_device: str = "cuda"
    proposal_batch_size: int = 4096
    benchmark_output_dir: str = "src/benchmark"
    run_name: str = "complete_example_inference_sweep_run_id"


@dataclass(frozen=True, slots=True)
class CompleteExampleInferenceSweepResult:
    config: CompleteExampleInferenceSweepConfig
    examples: tuple[_ExampleResult, ...]
    report_path: Path
    summary_json_path: Path


def _t0_base_eval_dataset(_: PureExactCPUConfig) -> GeneratedDataset:
    cfg = T0SyntheticProxyExperimentConfig(
        train_mesh_count_per_split=384,
        train_robot_link_count=192,
        eval_mesh_count_per_split=128,
        eval_robot_link_count=64,
        seed=22,
        run_name="t0_synthetic_proxy_20260422_main",
    )
    return generate_exact_oracle_dataset(
        DatasetGenerationConfig(
            mesh_count_per_split=cfg.eval_mesh_count_per_split,
            robot_link_count=cfg.eval_robot_link_count,
            seed=cfg.seed + 1,
            include_robot_links=True,
        )
    )


def _trained_high_density_eval_dataset(_: PureExactCPUConfig) -> GeneratedDataset:
    # This workbench expands each query into 128 candidates. The generic five-method
    # benchmark consumes query-level motion samples, so we benchmark the documented
    # held-out eval query set rather than the candidate-expanded workbench rows.
    return generate_exact_oracle_dataset(
        DatasetGenerationConfig(
            mesh_count_per_split=20,
            robot_link_count=0,
            seed=402,
            include_robot_links=False,
        )
    )


def _abc_demo_benchmark_dataset(exact: PureExactCPUConfig) -> GeneratedDataset:
    return build_abc_paper_benchmark_dataset(
        ABCPaperBenchmarkConfig(
            exact=exact,
            rt_backend_name="optix_rt",
            model_checkpoint_path=_abc_checkpoint(),
            model_device="cuda",
            include_random_stpf=False,
            hard_case_enabled=False,
            run_name="abc_cad_paper_benchmark_run_id",
        )
    ).generated_dataset


def _thingi10k_benchmark_dataset(exact: PureExactCPUConfig) -> GeneratedDataset:
    bundle, dataset = build_thingi10k_paper_benchmark_dataset(
        Thingi10KPaperBenchmarkConfig(
            exact=exact,
            rt_backend_name="optix_rt",
            model_checkpoint_path=_thingi10k_checkpoint(),
            model_device="cuda",
            include_random_stpf=False,
            hard_case_enabled=False,
            run_name="thingi10k_paper_benchmark_run_id",
        )
    )
    del bundle
    return dataset.generated_dataset


def _example_specs() -> tuple[_ExampleSpec, ...]:
    return (
        _ExampleSpec(
            example_name="T0 synthetic_proxy",
            source_name="analytic_swept_sphere_proxy",
            scene_name="t0_base_eval_queries",
            dataset_note="Use the documented T0 base-eval query set (704 queries), not the 128x candidate-expanded dense workbench rows.",
            checkpoint_path=_t0_checkpoint(),
            checkpoint_note="Checkpoint from t0_synthetic_proxy_20260422_main.",
            dataset_builder=_t0_base_eval_dataset,
        ),
        _ExampleSpec(
            example_name="trained_stpf_high_density",
            source_name="analytic_swept_sphere_proxy",
            scene_name="high_density_eval_queries",
            dataset_note="Use the documented high-density held-out eval query set (100 queries); the candidate-expanded workload is not directly comparable to the query-level five-method pipeline.",
            checkpoint_path=_trained_high_density_checkpoint(),
            checkpoint_note="Checkpoint from trained_stpf_high_density_20260421_final.",
            dataset_builder=_trained_high_density_eval_dataset,
        ),
        _ExampleSpec(
            example_name="ABC CAD",
            source_name="ABC Dataset",
            scene_name="demo_heldout_cad",
            dataset_note="Use the complete-example demo held-out CAD benchmark slice (640 queries).",
            checkpoint_path=_abc_checkpoint(),
            checkpoint_note="Checkpoint from abc_training_20260422_demo_main.",
            dataset_builder=_abc_demo_benchmark_dataset,
        ),
        _ExampleSpec(
            example_name="Thingi10K",
            source_name="Thingi10K",
            scene_name="heldout_eval",
            dataset_note="Use the held-out Thingi10K benchmark slice (512 queries).",
            checkpoint_path=_thingi10k_checkpoint(),
            checkpoint_note="Checkpoint from thingi10k_training_run_id.",
            dataset_builder=_thingi10k_benchmark_dataset,
        ),
    )


def _rtstpf_config(
    cfg: CompleteExampleInferenceSweepConfig,
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


def _run_one_example(
    cfg: CompleteExampleInferenceSweepConfig,
    spec: _ExampleSpec,
) -> _ExampleResult:
    checkpoint_path = Path(spec.checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"missing checkpoint for {spec.example_name}: {checkpoint_path}")

    dataset = spec.dataset_builder(cfg.exact)
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
        _rtstpf_config(cfg, checkpoint_path=spec.checkpoint_path),
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

    rows = (
        _row_from_pure(pure),
        _row_from_bvh(bvh),
        _row_from_rt_exact(rt_exact),
        _row_from_rtstpf(rtstpf),
        _row_from_no_proposal(no_proposal),
    )
    return _ExampleResult(
        example_name=spec.example_name,
        source_name=spec.source_name,
        scene_name=spec.scene_name,
        dataset_note=spec.dataset_note,
        checkpoint_path=spec.checkpoint_path,
        checkpoint_note=spec.checkpoint_note,
        query_count=len(dataset.samples),
        rows=rows,
    )


def write_complete_example_inference_sweep_report(
    path: str | Path,
    result: CompleteExampleInferenceSweepResult,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Complete example five-method inference benchmark summary",
        "",
        "## Protocol",
        "",
        "- only performsdescription, description checkpoint. ",
        "- `RTSTPFExact` fixedas learned-only description: `fastest_learned + ORT(TensorRT EP description) + optix_rt + cuda_exact`. ",
        "- `T0 synthetic_proxy` and `trained_stpf_high_density` descriptionuse query-level eval description; description candidate-expanded dense workbench descriptionconnectdescriptionMethod query-level pipeline. ",
        "",
    ]
    for example in result.examples:
        lines.extend(
            [
                f"## {example.example_name}",
                "",
                f"- Source: `{example.source_name}`",
                f"- Scene: `{example.scene_name}`",
                f"- Queries: `{example.query_count}`",
                f"- Checkpoint: `{example.checkpoint_path}`",
                f"- Checkpoint note: {example.checkpoint_note}",
                f"- Dataset note: {example.dataset_note}",
                "",
                "| Method | Total(ms) | RT(ms) | Proposal(ms) | Exact(ms) | QPS | FN | Recall | AvgCandidates | AvgExactEvals | Backend |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for row in sorted(example.rows, key=lambda item: item.total_ms):
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
        best = min(example.rows, key=lambda item: item.total_ms)
        lines.extend(
            [
                "",
                f"- Fastest method: `{best.method}` with `{best.total_ms:.4f} ms`.",
                "",
            ]
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def write_complete_example_inference_sweep_json(
    path: str | Path,
    result: CompleteExampleInferenceSweepResult,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": asdict(result.config),
        "examples": [asdict(example) for example in result.examples],
        "report_path": str(result.report_path),
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def run_complete_example_inference_sweep(
    config: CompleteExampleInferenceSweepConfig | None = None,
) -> CompleteExampleInferenceSweepResult:
    cfg = config or CompleteExampleInferenceSweepConfig()
    examples = tuple(_run_one_example(cfg, spec) for spec in _example_specs())
    output_root = Path(cfg.benchmark_output_dir)
    report_path = write_complete_example_inference_sweep_report(
        output_root / f"{cfg.run_name}.md",
        CompleteExampleInferenceSweepResult(
            config=cfg,
            examples=examples,
            report_path=output_root / f"{cfg.run_name}.md",
            summary_json_path=output_root / f"{cfg.run_name}.json",
        ),
    )
    summary_json_path = write_complete_example_inference_sweep_json(
        output_root / f"{cfg.run_name}.json",
        CompleteExampleInferenceSweepResult(
            config=cfg,
            examples=examples,
            report_path=report_path,
            summary_json_path=output_root / f"{cfg.run_name}.json",
        ),
    )
    return CompleteExampleInferenceSweepResult(
        config=cfg,
        examples=examples,
        report_path=report_path,
        summary_json_path=summary_json_path,
    )


__all__ = [
    "CompleteExampleInferenceSweepConfig",
    "CompleteExampleInferenceSweepResult",
    "run_complete_example_inference_sweep",
    "write_complete_example_inference_sweep_json",
    "write_complete_example_inference_sweep_report",
]
