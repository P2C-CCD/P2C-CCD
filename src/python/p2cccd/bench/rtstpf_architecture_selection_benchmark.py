from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Callable

from p2cccd.datasets.ccd import DatasetQueryBatch, ScalableCCDSampleAdapter

from .abc_paper_benchmark import ABCPaperBenchmarkConfig, build_abc_paper_benchmark_dataset
from .no_proposal import (
    NoProposalConfig,
    NoProposalResult,
    run_no_proposal_on_external_batch,
    run_no_proposal_on_generated_dataset,
)
from .pure_exact_cpu import PureExactCPUConfig
from .rt_stpf_exact import (
    RTSTPFExactConfig,
    RTSTPFExactResult,
    run_rt_stpf_exact_on_external_batch,
    run_rt_stpf_exact_on_generated_dataset,
)
from .thingi10k_paper_benchmark import (
    Thingi10KPaperBenchmarkConfig,
    build_thingi10k_paper_benchmark_dataset,
)


def _default_abc_checkpoint() -> str:
    return str(
        Path("src/outputs/stpf_training")
        / "abc_training_20260422_demo_main"
        / "model_state.pt"
    )


def _default_thingi_checkpoint() -> str:
    return str(
        Path("src/outputs/stpf_training")
        / "thingi10k_training_run_id"
        / "model_state.pt"
    )


@dataclass(frozen=True, slots=True)
class RTSTPFArchitectureSelectionBenchmarkConfig:
    exact: PureExactCPUConfig = PureExactCPUConfig(
        eps_time=1.0e-5,
        eps_space=1.0e-8,
        max_subdivision_depth=24,
    )
    abc_checkpoint_path: str = _default_abc_checkpoint()
    thingi10k_checkpoint_path: str = _default_thingi_checkpoint()
    model_device: str = "cuda"
    proposal_batch_size: int = 4096
    benchmark_output_dir: str = "src/benchmark"
    run_name: str = "rtstpf_architecture_selection_run_id"


@dataclass(frozen=True, slots=True)
class _ScenarioInput:
    scenario: str
    source_name: str
    scene_name: str
    query_count: int
    checkpoint_path: str
    run_no_proposal: Callable[[], NoProposalResult]
    run_rtstpf: Callable[[RTSTPFExactConfig], RTSTPFExactResult]


@dataclass(frozen=True, slots=True)
class _VariantSummary:
    scenario: str
    source_name: str
    scene_name: str
    query_count: int
    variant: str
    total_ms: float
    rt_ms: float
    proposal_ms: float
    exact_ms: float
    qps: float
    fn_count: int
    candidate_recall: float
    resolved_execution_profile_name: str
    inference_backend_name: str
    inference_provider_name: str
    exact_backend_name: str


@dataclass(frozen=True, slots=True)
class RTSTPFArchitectureSelectionBenchmarkResult:
    config: RTSTPFArchitectureSelectionBenchmarkConfig
    summaries: tuple[_VariantSummary, ...]
    report_path: Path
    summary_json_path: Path


def _make_rtstpf_config(
    cfg: RTSTPFArchitectureSelectionBenchmarkConfig,
    *,
    checkpoint_path: str,
    execution_profile: str,
) -> RTSTPFExactConfig:
    return RTSTPFExactConfig(
        exact=cfg.exact,
        rt_backend_name="optix_rt",
        enable_cuda_exact=True,
        execution_profile=execution_profile,
        allow_default_model=False,
        model_checkpoint_path=checkpoint_path,
        model_device=cfg.model_device,
        inference_backend="ort",
        proposal_batch_size=cfg.proposal_batch_size,
        cpu_inference_row_threshold=0,
    )


def _build_scenarios(
    cfg: RTSTPFArchitectureSelectionBenchmarkConfig,
) -> tuple[_ScenarioInput, ...]:
    external_batch = ScalableCCDSampleAdapter().load_query_batch(
        "armadillo-rollers",
        family="vf",
        step=326,
        limit=4096,
    )

    abc_dataset = build_abc_paper_benchmark_dataset(
        ABCPaperBenchmarkConfig(
            run_name="abc_cad_paper_benchmark_official_large_run_id",
            use_official_root=True,
            allow_official_download=False,
            official_asset_limit=256,
            benchmark_asset_offset=128,
            benchmark_asset_count=64,
            pair_limit=512,
            exact=cfg.exact,
            model_checkpoint_path=cfg.abc_checkpoint_path,
            model_device=cfg.model_device,
        )
    ).generated_dataset
    thingi_bundle, thingi_dataset = build_thingi10k_paper_benchmark_dataset(
        Thingi10KPaperBenchmarkConfig(
            exact=cfg.exact,
            rt_backend_name="optix_rt",
            model_checkpoint_path=cfg.thingi10k_checkpoint_path,
            model_device=cfg.model_device,
        )
    )
    del thingi_bundle

    return (
        _ScenarioInput(
            scenario="external_sparse",
            source_name=external_batch.source_name,
            scene_name=external_batch.scene_name,
            query_count=len(external_batch.queries),
            checkpoint_path=cfg.thingi10k_checkpoint_path,
            run_no_proposal=lambda: run_no_proposal_on_external_batch(
                external_batch,
                NoProposalConfig(
                    exact=cfg.exact,
                    rt_backend_name="optix_rt",
                    enable_cuda_exact=True,
                ),
            ),
            run_rtstpf=lambda stpf_cfg: run_rt_stpf_exact_on_external_batch(
                external_batch,
                stpf_cfg,
                device=cfg.model_device,
            ),
        ),
        _ScenarioInput(
            scenario="abc_official_large",
            source_name="ABC Dataset",
            scene_name="official_large_heldout_cad",
            query_count=len(abc_dataset.samples),
            checkpoint_path=cfg.abc_checkpoint_path,
            run_no_proposal=lambda: run_no_proposal_on_generated_dataset(
                abc_dataset,
                NoProposalConfig(
                    exact=cfg.exact,
                    rt_backend_name="optix_rt",
                    enable_cuda_exact=True,
                ),
            ),
            run_rtstpf=lambda stpf_cfg: run_rt_stpf_exact_on_generated_dataset(
                abc_dataset,
                stpf_cfg,
                device=cfg.model_device,
            ),
        ),
        _ScenarioInput(
            scenario="thingi10k_heldout",
            source_name="Thingi10K",
            scene_name="heldout_eval",
            query_count=len(thingi_dataset.generated_dataset.samples),
            checkpoint_path=cfg.thingi10k_checkpoint_path,
            run_no_proposal=lambda: run_no_proposal_on_generated_dataset(
                thingi_dataset.generated_dataset,
                NoProposalConfig(
                    exact=cfg.exact,
                    rt_backend_name="optix_rt",
                    enable_cuda_exact=True,
                ),
            ),
            run_rtstpf=lambda stpf_cfg: run_rt_stpf_exact_on_generated_dataset(
                thingi_dataset.generated_dataset,
                stpf_cfg,
                device=cfg.model_device,
            ),
        ),
    )


def _summary_from_result(
    *,
    scenario: _ScenarioInput,
    variant: str,
    result: RTSTPFExactResult,
) -> _VariantSummary:
    row = result.benchmark
    return _VariantSummary(
        scenario=scenario.scenario,
        source_name=scenario.source_name,
        scene_name=scenario.scene_name,
        query_count=scenario.query_count,
        variant=variant,
        total_ms=row.total_ms,
        rt_ms=row.rt_ms,
        proposal_ms=row.proposal_ms,
        exact_ms=row.exact_ms,
        qps=row.qps,
        fn_count=row.fn_count,
        candidate_recall=row.candidate_recall,
        resolved_execution_profile_name=result.resolved_execution_profile_name,
        inference_backend_name=result.inference_backend_name,
        inference_provider_name=result.inference_provider_name,
        exact_backend_name=result.exact_backend_name,
    )


def _summary_from_no_proposal(
    *,
    scenario: _ScenarioInput,
    result: NoProposalResult,
) -> _VariantSummary:
    row = result.benchmark
    return _VariantSummary(
        scenario=scenario.scenario,
        source_name=scenario.source_name,
        scene_name=scenario.scene_name,
        query_count=scenario.query_count,
        variant="reference:no_proposal",
        total_ms=row.total_ms,
        rt_ms=row.rt_ms,
        proposal_ms=row.proposal_ms,
        exact_ms=row.exact_ms,
        qps=row.qps,
        fn_count=row.fn_count,
        candidate_recall=row.candidate_recall,
        resolved_execution_profile_name="reference",
        inference_backend_name="none",
        inference_provider_name="none",
        exact_backend_name=result.exact_backend_name,
    )


def _report_markdown(result: RTSTPFArchitectureSelectionBenchmarkResult) -> str:
    lines = [
        "# RTSTPFExact Architecture Selection Benchmark",
        "",
        "## Results",
        "",
        "| Scenario | Variant | Queries | Total ms | RT ms | Proposal ms | Exact ms | QPS | FN | Recall | Resolved profile | Inference | Provider | Exact backend |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |",
    ]
    by_scenario: dict[str, list[_VariantSummary]] = {}
    for row in result.summaries:
        lines.append(
            f"| {row.scenario} | {row.variant} | {row.query_count} | {row.total_ms:.4f} | {row.rt_ms:.4f} | "
            f"{row.proposal_ms:.4f} | {row.exact_ms:.4f} | {row.qps:.2f} | {row.fn_count} | "
            f"{row.candidate_recall:.4f} | {row.resolved_execution_profile_name} | {row.inference_backend_name} | "
            f"{row.inference_provider_name} | {row.exact_backend_name} |"
        )
        by_scenario.setdefault(row.scenario, []).append(row)
    lines.extend(["", "## Best Per Scenario", ""])
    for scenario, rows in by_scenario.items():
        candidates = [item for item in rows if not item.variant.startswith("reference:")]
        best = min(candidates, key=lambda item: item.total_ms)
        lines.append(
            f"- `{scenario}` fastest variant: `{best.variant}` with `{best.total_ms:.4f} ms` "
            f"({best.resolved_execution_profile_name}, provider `{best.inference_provider_name}`)."
        )
    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            "- `RTSTPFExact` is now learned-only; dummy proposal execution is no longer supported.",
            "- `execution_profile=fastest_learned` maps to `optix_rt + ORT(TensorRT EP preferred) + CUDA exact`.",
            "- `execution_profile=auto_fastest` currently resolves to the same learned ORT path.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_rtstpf_architecture_selection_benchmark(
    config: RTSTPFArchitectureSelectionBenchmarkConfig | None = None,
) -> RTSTPFArchitectureSelectionBenchmarkResult:
    cfg = config or RTSTPFArchitectureSelectionBenchmarkConfig()
    output_root = Path(cfg.benchmark_output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    summaries: list[_VariantSummary] = []
    for scenario in _build_scenarios(cfg):
        summaries.append(
            _summary_from_no_proposal(
                scenario=scenario,
                result=scenario.run_no_proposal(),
            )
        )
        for variant in ("fastest_learned", "auto_fastest"):
            stpf_result = scenario.run_rtstpf(
                _make_rtstpf_config(
                    cfg,
                    checkpoint_path=scenario.checkpoint_path,
                    execution_profile=variant,
                )
            )
            summaries.append(
                _summary_from_result(
                    scenario=scenario,
                    variant=variant,
                    result=stpf_result,
                )
            )

    report_path = output_root / f"{cfg.run_name}.md"
    summary_json_path = output_root / f"{cfg.run_name}.json"
    result = RTSTPFArchitectureSelectionBenchmarkResult(
        config=cfg,
        summaries=tuple(summaries),
        report_path=report_path,
        summary_json_path=summary_json_path,
    )
    report_path.write_text(_report_markdown(result), encoding="utf-8")
    summary_json_path.write_text(
        json.dumps(
            {
                "config": asdict(cfg),
                "summaries": [asdict(item) for item in result.summaries],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return result


__all__ = [
    "RTSTPFArchitectureSelectionBenchmarkConfig",
    "RTSTPFArchitectureSelectionBenchmarkResult",
    "run_rtstpf_architecture_selection_benchmark",
]
