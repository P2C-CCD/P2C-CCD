from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

from p2cccd.cuda_bindings import get_cuda_binding_status
from p2cccd.datasets.ccd import ScalableCCDSampleAdapter

from .no_proposal import NoProposalConfig, NoProposalResult, run_no_proposal_on_external_batch
from .pure_exact_cpu import PureExactCPUConfig
from .rt_exact import RTExactConfig, RTExactResult, run_rt_exact_on_external_batch
from .rt_stpf_exact import RTSTPFExactConfig, RTSTPFExactResult, run_rt_stpf_exact_on_external_batch


def _default_checkpoint() -> str:
    return str(Path("src/outputs/stpf_training") / "thingi10k_training_run_id" / "model_state.pt")


@dataclass(frozen=True, slots=True)
class RTSTPFFastestStackBenchmarkConfig:
    scene_name: str = "armadillo-rollers"
    family: str = "vf"
    step: int = 326
    limit: int = 4096
    exact: PureExactCPUConfig = PureExactCPUConfig(eps_time=1.0e-5, eps_space=1.0e-8, max_subdivision_depth=24)
    model_checkpoint_path: str = _default_checkpoint()
    model_preset: str = "lightweight_mlp"
    model_device: str = "cuda"
    proposal_batch_size: int = 4096
    ort_warmup_passes: int = 1
    run_name: str = "rtstpf_fastest_stack_external_run_id"
    benchmark_output_dir: str = "src/benchmark"


@dataclass(frozen=True, slots=True)
class _MethodSummary:
    method: str
    query_count: int
    fn_count: int
    fp_count: int
    candidate_recall: float
    avg_candidates: float
    avg_exact_evals: float
    fallback_ratio: float
    rt_ms: float
    proposal_ms: float
    exact_ms: float
    total_ms: float
    qps: float
    rt_build_ms: float
    rt_update_ms: float
    rt_trace_ms: float
    inference_backend_name: str
    inference_provider_name: str
    exact_backend_name: str


@dataclass(frozen=True, slots=True)
class RTSTPFFastestStackBenchmarkResult:
    config: RTSTPFFastestStackBenchmarkConfig
    dataset_source: str
    batch_scene_name: str
    batch_id: str
    rt_exact: RTExactResult
    no_proposal: NoProposalResult
    rtstpf_torch: RTSTPFExactResult
    rtstpf_ort: RTSTPFExactResult
    report_path: Path
    summary_json_path: Path


def _summary_from_rt_exact(method: str, result: RTExactResult) -> _MethodSummary:
    row = result.benchmark
    return _MethodSummary(
        method=method,
        query_count=row.query_count,
        fn_count=row.fn_count,
        fp_count=row.fp_count,
        candidate_recall=row.candidate_recall,
        avg_candidates=row.avg_candidates,
        avg_exact_evals=row.avg_exact_evals,
        fallback_ratio=row.fallback_ratio,
        rt_ms=row.rt_ms,
        proposal_ms=row.proposal_ms,
        exact_ms=row.exact_ms,
        total_ms=row.total_ms,
        qps=row.qps,
        rt_build_ms=result.candidate_stats.timing.build_ms,
        rt_update_ms=result.candidate_stats.timing.update_ms,
        rt_trace_ms=result.candidate_stats.timing.trace_ms,
        inference_backend_name="none",
        inference_provider_name="none",
        exact_backend_name=result.exact_backend_name,
    )


def _summary_from_no_proposal(method: str, result: NoProposalResult) -> _MethodSummary:
    row = result.benchmark
    return _MethodSummary(
        method=method,
        query_count=row.query_count,
        fn_count=row.fn_count,
        fp_count=row.fp_count,
        candidate_recall=row.candidate_recall,
        avg_candidates=row.avg_candidates,
        avg_exact_evals=row.avg_exact_evals,
        fallback_ratio=row.fallback_ratio,
        rt_ms=row.rt_ms,
        proposal_ms=row.proposal_ms,
        exact_ms=row.exact_ms,
        total_ms=row.total_ms,
        qps=row.qps,
        rt_build_ms=result.candidate_stats.timing.build_ms,
        rt_update_ms=result.candidate_stats.timing.update_ms,
        rt_trace_ms=result.candidate_stats.timing.trace_ms,
        inference_backend_name="none",
        inference_provider_name="none",
        exact_backend_name=result.exact_backend_name,
    )


def _summary_from_rtstpf(method: str, result: RTSTPFExactResult) -> _MethodSummary:
    row = result.benchmark
    return _MethodSummary(
        method=method,
        query_count=row.query_count,
        fn_count=row.fn_count,
        fp_count=row.fp_count,
        candidate_recall=row.candidate_recall,
        avg_candidates=row.avg_candidates,
        avg_exact_evals=row.avg_exact_evals,
        fallback_ratio=row.fallback_ratio,
        rt_ms=row.rt_ms,
        proposal_ms=row.proposal_ms,
        exact_ms=row.exact_ms,
        total_ms=row.total_ms,
        qps=row.qps,
        rt_build_ms=result.candidate_stats.timing.build_ms,
        rt_update_ms=result.candidate_stats.timing.update_ms,
        rt_trace_ms=result.candidate_stats.timing.trace_ms,
        inference_backend_name=result.inference_backend_name,
        inference_provider_name=result.inference_provider_name,
        exact_backend_name=result.exact_backend_name,
    )


def _report_markdown(result: RTSTPFFastestStackBenchmarkResult) -> str:
    rows = (
        _summary_from_rt_exact("RTExact", result.rt_exact),
        _summary_from_no_proposal("NoProposal", result.no_proposal),
        _summary_from_rtstpf("RTSTPFExact-Torch", result.rtstpf_torch),
        _summary_from_rtstpf("RTSTPFExact-ORT", result.rtstpf_ort),
    )
    cuda_status = get_cuda_binding_status()
    lines = [
        "# RTSTPF Fastest-Stack External Benchmark",
        "",
        "## Setup",
        "",
        f"- source: `{result.dataset_source}`",
        f"- scene: `{result.batch_scene_name}`",
        f"- batch: `{result.batch_id}`",
        f"- query_count: `{result.rt_exact.benchmark.query_count}`",
        f"- target stack: `optix_rt + C++ scheduling + ORT(TensorRT EP preferred) + CUDA exact`",
        f"- ort_warmup_passes: `{result.config.ort_warmup_passes}`",
        f"- cuda_exact_ready: `{cuda_status.ready_for_cuda_execution}`",
        f"- cuda_backend_name: `{cuda_status.backend_name}`",
        "",
        "## Results",
        "",
        "| Method | FN | Recall | Total ms | RT ms | Proposal ms | Exact ms | QPS | RT build | RT update | RT trace | Inference | Provider | Exact backend |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row.method} | {row.fn_count} | {row.candidate_recall:.4f} | "
            f"{row.total_ms:.4f} | {row.rt_ms:.4f} | {row.proposal_ms:.4f} | {row.exact_ms:.4f} | "
            f"{row.qps:.2f} | {row.rt_build_ms:.4f} | {row.rt_update_ms:.4f} | {row.rt_trace_ms:.4f} | "
            f"{row.inference_backend_name} | {row.inference_provider_name} | {row.exact_backend_name} |"
        )
    ort_delta = result.rtstpf_torch.benchmark.total_ms - result.rtstpf_ort.benchmark.total_ms
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            f"- `RTSTPFExact-ORT` total latency delta vs `RTSTPFExact-Torch`: `{ort_delta:.4f} ms`.",
            f"- ORT provider actually used: `{result.rtstpf_ort.inference_provider_name}`.",
            f"- CUDA exact backend used by `RTSTPFExact-ORT`: `{result.rtstpf_ort.exact_backend_name}`.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_rtstpf_fastest_stack_benchmark_json(
    path: str | Path,
    result: RTSTPFFastestStackBenchmarkResult,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": asdict(result.config),
        "dataset_source": result.dataset_source,
        "batch_scene_name": result.batch_scene_name,
        "batch_id": result.batch_id,
        "cuda_binding_status": asdict(get_cuda_binding_status()),
        "methods": [
            asdict(_summary_from_rt_exact("RTExact", result.rt_exact)),
            asdict(_summary_from_no_proposal("NoProposal", result.no_proposal)),
            asdict(_summary_from_rtstpf("RTSTPFExact-Torch", result.rtstpf_torch)),
            asdict(_summary_from_rtstpf("RTSTPFExact-ORT", result.rtstpf_ort)),
        ],
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def run_rtstpf_fastest_stack_external_benchmark(
    config: RTSTPFFastestStackBenchmarkConfig | None = None,
) -> RTSTPFFastestStackBenchmarkResult:
    cfg = config or RTSTPFFastestStackBenchmarkConfig()
    adapter = ScalableCCDSampleAdapter()
    batch = adapter.load_query_batch(
        cfg.scene_name,
        family=cfg.family,
        step=cfg.step,
        limit=cfg.limit,
    )
    output_root = Path(cfg.benchmark_output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    ort_model_path = output_root / f"{cfg.run_name}.onnx"

    rt_exact = run_rt_exact_on_external_batch(
        batch,
        RTExactConfig(
            exact=cfg.exact,
            backend_name="optix_rt",
            enable_cuda_exact=True,
        ),
    )
    no_proposal = run_no_proposal_on_external_batch(
        batch,
        NoProposalConfig(
            exact=cfg.exact,
            rt_backend_name="optix_rt",
            enable_cuda_exact=True,
        ),
    )
    common_stpf_config = dict(
        exact=cfg.exact,
        rt_backend_name="optix_rt",
        enable_cuda_exact=True,
        use_dummy_policy=False,
        allow_default_model=False,
        model_preset=cfg.model_preset,
        model_checkpoint_path=cfg.model_checkpoint_path,
        model_device=cfg.model_device,
        proposal_batch_size=cfg.proposal_batch_size,
        cpu_inference_row_threshold=0,
    )
    rtstpf_torch = run_rt_stpf_exact_on_external_batch(
        batch,
        RTSTPFExactConfig(
            **common_stpf_config,
            inference_backend="torch",
        ),
        device=cfg.model_device,
    )
    rtstpf_ort = run_rt_stpf_exact_on_external_batch(
        batch,
        RTSTPFExactConfig(
            **common_stpf_config,
            inference_backend="ort",
            ort_model_path=str(ort_model_path),
            ort_prefer_tensorrt=True,
            ort_allow_cuda_fallback=True,
            ort_allow_cpu_fallback=True,
            ort_warmup_passes=cfg.ort_warmup_passes,
        ),
        device=cfg.model_device,
    )
    report_path = output_root / f"{cfg.run_name}.md"
    summary_json_path = output_root / f"{cfg.run_name}.json"
    result = RTSTPFFastestStackBenchmarkResult(
        config=cfg,
        dataset_source=batch.source_name,
        batch_scene_name=batch.scene_name,
        batch_id=batch.batch_id,
        rt_exact=rt_exact,
        no_proposal=no_proposal,
        rtstpf_torch=rtstpf_torch,
        rtstpf_ort=rtstpf_ort,
        report_path=report_path,
        summary_json_path=summary_json_path,
    )
    report_path.write_text(_report_markdown(result), encoding="utf-8")
    write_rtstpf_fastest_stack_benchmark_json(summary_json_path, result)
    return result


__all__ = [
    "RTSTPFFastestStackBenchmarkConfig",
    "RTSTPFFastestStackBenchmarkResult",
    "run_rtstpf_fastest_stack_external_benchmark",
    "write_rtstpf_fastest_stack_benchmark_json",
]
