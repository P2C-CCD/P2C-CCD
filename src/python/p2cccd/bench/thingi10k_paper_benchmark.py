from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path

import torch

from p2cccd.bench.bvh_exact import BVHExactConfig, BVHExactResult, run_bvh_exact_on_generated_dataset
from p2cccd.bench.no_proposal import NoProposalConfig, NoProposalResult, run_no_proposal_on_generated_dataset
from p2cccd.bench.pure_exact_cpu import PureExactCPUConfig, PureExactCPUResult, run_pure_exact_cpu_on_generated_dataset
from p2cccd.bench.rt_exact import RTExactConfig, RTExactResult, run_rt_exact_on_generated_dataset
from p2cccd.bench.rt_stpf_exact import RTSTPFExactConfig, RTSTPFExactResult, run_rt_stpf_exact_on_generated_dataset
from p2cccd.bench.trained_stpf_high_density import (
    HighDensityMethodMetrics,
    HighDensitySTPFConfig,
    benchmark_no_proposal_on_high_density_workload,
    benchmark_stpf_on_high_density_workload,
    build_high_density_stpf_workload,
)
from p2cccd.contracts import BenchmarkRow
from p2cccd.data import default_metadata, write_npz_shard
from p2cccd.data.dataset import GeneratedDataset
from p2cccd.datasets.objects.thingi10k_training import (
    Thingi10KProxyDatasetBundle,
    Thingi10KProxyDatasetConfig,
    generate_thingi10k_proxy_datasets,
)
from p2cccd.proposal.stpf_model import build_stpf_model, build_stpf_model_from_checkpoint_payload


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _default_trained_checkpoint() -> str:
    return str(Path("src/outputs/stpf_training") / "thingi10k_training_run_id" / "model_state.pt")


@dataclass(frozen=True, slots=True)
class Thingi10KPaperBenchmarkConfig:
    dataset: Thingi10KProxyDatasetConfig = Thingi10KProxyDatasetConfig()
    exact: PureExactCPUConfig = field(default_factory=PureExactCPUConfig)
    rt_backend_name: str = "cpu_reference_rt"
    model_checkpoint_path: str = field(default_factory=_default_trained_checkpoint)
    model_device: str = _default_device()
    include_random_stpf: bool = True
    hard_case_enabled: bool = True
    hard_case: HighDensitySTPFConfig = HighDensitySTPFConfig(
        slab_count=12,
        patches_per_object=6,
        representative_attempt_limit=2,
        uncertainty_fallback_threshold=0.75,
    )
    benchmark_output_dir: str = "src/benchmark"
    benchmark_dataset_dir: str = "src/datasets/benchmark/ood_stress/thingi10k"
    run_name: str = "thingi10k_paper_benchmark_run_id"


@dataclass(frozen=True, slots=True)
class Thingi10KPaperBenchmarkDataset:
    source_root: Path
    asset_count: int
    pair_count: int
    pair_ids: tuple[str, ...]
    asset_paths: tuple[str, ...]
    generated_dataset: GeneratedDataset
    dataset_npz_path: Path
    dataset_manifest_path: Path


@dataclass(frozen=True, slots=True)
class Thingi10KPaperBenchmarkArtifacts:
    report_path: Path
    summary_json_path: Path


@dataclass(frozen=True, slots=True)
class Thingi10KPaperBenchmarkResult:
    config: Thingi10KPaperBenchmarkConfig
    bundle: Thingi10KProxyDatasetBundle
    dataset: Thingi10KPaperBenchmarkDataset
    pure_exact_cpu: PureExactCPUResult
    bvh_exact: BVHExactResult
    rt_exact: RTExactResult
    no_proposal: NoProposalResult
    rtstpf_random: RTSTPFExactResult | None
    rtstpf_trained: RTSTPFExactResult
    hard_case_no_proposal: HighDensityMethodMetrics | None
    hard_case_random: HighDensityMethodMetrics | None
    hard_case_trained: HighDensityMethodMetrics | None
    artifacts: Thingi10KPaperBenchmarkArtifacts


@dataclass(frozen=True, slots=True)
class _MethodSummary:
    method: str
    query_count: int
    fn_count: int
    fp_count: int
    candidate_recall: float
    avg_candidates: float
    avg_exact_evals: float
    avg_subdivision_depth: float
    fallback_ratio: float
    rt_ms: float
    proposal_ms: float
    exact_ms: float
    total_ms: float
    qps: float
    rt_build_ms: float
    rt_update_ms: float
    rt_trace_ms: float


def build_thingi10k_paper_benchmark_dataset(
    config: Thingi10KPaperBenchmarkConfig | None = None,
) -> tuple[Thingi10KProxyDatasetBundle, Thingi10KPaperBenchmarkDataset]:
    cfg = config or Thingi10KPaperBenchmarkConfig()
    bundle = generate_thingi10k_proxy_datasets(cfg.dataset)
    output_root = Path(cfg.benchmark_dataset_dir) / cfg.run_name
    output_root.mkdir(parents=True, exist_ok=True)
    dataset_npz_path = output_root / "benchmark_dataset.npz"
    metadata = default_metadata(bundle.eval_dataset, seed=cfg.dataset.seed, source="thingi10k_heldout_benchmark")
    metadata["dataset_role"] = "benchmark"
    metadata["source_dataset"] = "Thingi10K"
    metadata["pair_count"] = len(bundle.eval_pairs)
    write_npz_shard(dataset_npz_path, bundle.eval_dataset, metadata=metadata)
    dataset_manifest = {
        "run_name": cfg.run_name,
        "source_root": str(bundle.source_root),
        "asset_count": len(bundle.assets),
        "pair_count": len(bundle.eval_pairs),
        "query_count": len(bundle.eval_dataset.samples),
        "dataset_npz_path": str(dataset_npz_path),
        "pair_ids": [pair.pair_id for pair in bundle.eval_pairs],
        "asset_paths": [str(asset.metadata.get("source_relative_path", asset.mesh_path)) for asset in bundle.assets],
        "training_separation_note": "This benchmark uses the held-out Thingi10K eval slice and should not be mixed with training shards.",
    }
    dataset_manifest_path = output_root / "dataset_manifest.json"
    dataset_manifest_path.write_text(json.dumps(dataset_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    dataset = Thingi10KPaperBenchmarkDataset(
        source_root=bundle.source_root,
        asset_count=len(bundle.assets),
        pair_count=len(bundle.eval_pairs),
        pair_ids=tuple(pair.pair_id for pair in bundle.eval_pairs),
        asset_paths=tuple(str(asset.metadata.get("source_relative_path", asset.mesh_path)) for asset in bundle.assets),
        generated_dataset=bundle.eval_dataset,
        dataset_npz_path=dataset_npz_path,
        dataset_manifest_path=dataset_manifest_path,
    )
    return bundle, dataset


def _summary_from_row(
    method: str,
    row: BenchmarkRow,
    *,
    rt_build_ms: float = 0.0,
    rt_update_ms: float = 0.0,
    rt_trace_ms: float = 0.0,
) -> _MethodSummary:
    return _MethodSummary(
        method=method,
        query_count=row.query_count,
        fn_count=row.fn_count,
        fp_count=row.fp_count,
        candidate_recall=row.candidate_recall,
        avg_candidates=row.avg_candidates,
        avg_exact_evals=row.avg_exact_evals,
        avg_subdivision_depth=row.avg_subdivision_depth,
        fallback_ratio=row.fallback_ratio,
        rt_ms=row.rt_ms,
        proposal_ms=row.proposal_ms,
        exact_ms=row.exact_ms,
        total_ms=row.total_ms,
        qps=row.qps,
        rt_build_ms=rt_build_ms,
        rt_update_ms=rt_update_ms,
        rt_trace_ms=rt_trace_ms,
    )


def _result_summaries(result: Thingi10KPaperBenchmarkResult) -> tuple[_MethodSummary, ...]:
    rows = [
        _summary_from_row("PureExactCPU", result.pure_exact_cpu.benchmark),
        _summary_from_row("BVHExact", result.bvh_exact.benchmark),
        _summary_from_row(
            "RTExact",
            result.rt_exact.benchmark,
            rt_build_ms=result.rt_exact.candidate_stats.timing.build_ms,
            rt_update_ms=result.rt_exact.candidate_stats.timing.update_ms,
            rt_trace_ms=result.rt_exact.candidate_stats.timing.trace_ms,
        ),
        _summary_from_row(
            "NoProposal",
            result.no_proposal.benchmark,
            rt_build_ms=result.no_proposal.candidate_stats.timing.build_ms,
            rt_update_ms=result.no_proposal.candidate_stats.timing.update_ms,
            rt_trace_ms=result.no_proposal.candidate_stats.timing.trace_ms,
        ),
    ]
    if result.rtstpf_random is not None:
        rows.append(
            _summary_from_row(
                "RTSTPFExact-Random",
                result.rtstpf_random.benchmark,
                rt_build_ms=result.rtstpf_random.candidate_stats.timing.build_ms,
                rt_update_ms=result.rtstpf_random.candidate_stats.timing.update_ms,
                rt_trace_ms=result.rtstpf_random.candidate_stats.timing.trace_ms,
            )
        )
    rows.append(
        _summary_from_row(
            "RTSTPFExact-Trained",
            result.rtstpf_trained.benchmark,
            rt_build_ms=result.rtstpf_trained.candidate_stats.timing.build_ms,
            rt_update_ms=result.rtstpf_trained.candidate_stats.timing.update_ms,
            rt_trace_ms=result.rtstpf_trained.candidate_stats.timing.trace_ms,
        )
    )
    return tuple(rows)


def run_thingi10k_paper_benchmark(
    config: Thingi10KPaperBenchmarkConfig | None = None,
) -> Thingi10KPaperBenchmarkResult:
    cfg = config or Thingi10KPaperBenchmarkConfig()
    bundle, dataset = build_thingi10k_paper_benchmark_dataset(cfg)
    pure_exact_cpu = run_pure_exact_cpu_on_generated_dataset(dataset.generated_dataset)
    bvh_exact = run_bvh_exact_on_generated_dataset(dataset.generated_dataset, BVHExactConfig(exact=cfg.exact))
    rt_exact = run_rt_exact_on_generated_dataset(
        dataset.generated_dataset,
        RTExactConfig(exact=cfg.exact, backend_name=cfg.rt_backend_name),
    )
    no_proposal = run_no_proposal_on_generated_dataset(
        dataset.generated_dataset,
        NoProposalConfig(exact=cfg.exact, rt_backend_name=cfg.rt_backend_name),
    )
    rtstpf_random = None
    if cfg.include_random_stpf:
        rtstpf_random = run_rt_stpf_exact_on_generated_dataset(
            dataset.generated_dataset,
            RTSTPFExactConfig(
                exact=cfg.exact,
                rt_backend_name=cfg.rt_backend_name,
                use_dummy_policy=False,
                allow_default_model=True,
                model_checkpoint_path=None,
                model_device=cfg.model_device,
            ),
        )
    rtstpf_trained = run_rt_stpf_exact_on_generated_dataset(
        dataset.generated_dataset,
        RTSTPFExactConfig(
            exact=cfg.exact,
            rt_backend_name=cfg.rt_backend_name,
            use_dummy_policy=False,
            allow_default_model=False,
            model_checkpoint_path=cfg.model_checkpoint_path,
            model_device=cfg.model_device,
        ),
    )
    hard_case_no_proposal = None
    hard_case_random = None
    hard_case_trained = None
    if cfg.hard_case_enabled:
        hard_case_workload = build_high_density_stpf_workload(
            dataset.generated_dataset,
            cfg.hard_case,
            name=f"{cfg.run_name}_hard_case",
        )
        hard_case_no_proposal = benchmark_no_proposal_on_high_density_workload(hard_case_workload)
        random_model = build_stpf_model()
        random_model.to(cfg.model_device)
        random_model.eval()
        hard_case_random = benchmark_stpf_on_high_density_workload(
            hard_case_workload,
            model=random_model,
            device=cfg.model_device,
            method_name="Thingi10KHard-RTSTPFExact-Random",
        )
        checkpoint = torch.load(cfg.model_checkpoint_path, map_location=cfg.model_device)
        trained_model, state_dict = build_stpf_model_from_checkpoint_payload(checkpoint)
        trained_model.load_state_dict(state_dict)
        trained_model.to(cfg.model_device)
        trained_model.eval()
        hard_case_trained = benchmark_stpf_on_high_density_workload(
            hard_case_workload,
            model=trained_model,
            device=cfg.model_device,
            method_name="Thingi10KHard-RTSTPFExact-Trained",
        )
    artifacts = Thingi10KPaperBenchmarkArtifacts(
        report_path=Path(cfg.benchmark_output_dir) / f"{cfg.run_name}.md",
        summary_json_path=Path(cfg.benchmark_output_dir) / f"{cfg.run_name}.json",
    )
    return Thingi10KPaperBenchmarkResult(
        config=cfg,
        bundle=bundle,
        dataset=dataset,
        pure_exact_cpu=pure_exact_cpu,
        bvh_exact=bvh_exact,
        rt_exact=rt_exact,
        no_proposal=no_proposal,
        rtstpf_random=rtstpf_random,
        rtstpf_trained=rtstpf_trained,
        hard_case_no_proposal=hard_case_no_proposal,
        hard_case_random=hard_case_random,
        hard_case_trained=hard_case_trained,
        artifacts=artifacts,
    )


def write_thingi10k_paper_benchmark_summary_json(
    path: str | Path,
    result: Thingi10KPaperBenchmarkResult,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_name": result.config.run_name,
        "dataset": {
            "source_root": str(result.dataset.source_root),
            "asset_count": result.dataset.asset_count,
            "pair_count": result.dataset.pair_count,
            "query_count": len(result.dataset.generated_dataset.samples),
            "dataset_npz_path": str(result.dataset.dataset_npz_path),
            "dataset_manifest_path": str(result.dataset.dataset_manifest_path),
        },
        "config": {
            "rt_backend_name": result.config.rt_backend_name,
            "model_checkpoint_path": result.config.model_checkpoint_path,
            "model_device": result.config.model_device,
            "seed": result.config.dataset.seed,
            "hard_case_enabled": result.config.hard_case_enabled,
        },
        "methods": [asdict(summary) for summary in _result_summaries(result)],
    }
    if result.hard_case_no_proposal is not None and result.hard_case_trained is not None:
        payload["hard_case"] = {
            "avg_candidates_per_query": result.hard_case_no_proposal.avg_candidates_per_query,
            "config": asdict(result.config.hard_case),
            "methods": [
                asdict(item)
                for item in (
                    result.hard_case_no_proposal,
                    result.hard_case_random,
                    result.hard_case_trained,
                )
                if item is not None
            ],
        }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def write_thingi10k_paper_benchmark_report(
    path: str | Path,
    result: Thingi10KPaperBenchmarkResult,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summaries = _result_summaries(result)
    header = (
        "| Method | Queries | FN | Recall | Avg Cand | Avg Exact | RT(ms) | Proposal(ms) | Exact(ms) | Total(ms) | QPS | RT Build | RT Update | RT Trace |\n"
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n"
    )
    body = "".join(
        (
            f"| {item.method} | {item.query_count} | {item.fn_count} | {item.candidate_recall:.4f} "
            f"| {item.avg_candidates:.4f} | {item.avg_exact_evals:.4f} | {item.rt_ms:.4f} "
            f"| {item.proposal_ms:.4f} | {item.exact_ms:.4f} | {item.total_ms:.4f} | {item.qps:.4f} "
            f"| {item.rt_build_ms:.4f} | {item.rt_update_ms:.4f} | {item.rt_trace_ms:.4f} |\n"
        )
        for item in summaries
    )
    trained = next(item for item in summaries if item.method == "RTSTPFExact-Trained")
    no_proposal = next(item for item in summaries if item.method == "NoProposal")
    random_row = next((item for item in summaries if item.method == "RTSTPFExact-Random"), None)
    lines = [
        "# Thingi10K paper-track benchmark",
        "",
        "## dataset",
        "",
        f"- Source root: `{result.dataset.source_root}`",
        f"- Asset count: `{result.dataset.asset_count}`",
        f"- Pair count: `{result.dataset.pair_count}`",
        f"- Query count: `{len(result.dataset.generated_dataset.samples)}`",
        f"- Dataset npz: `{result.dataset.dataset_npz_path}`",
        f"- Dataset manifest: `{result.dataset.dataset_manifest_path}`",
        "",
        "## description/Benchmark separation",
        "",
        "- description checkpoint descriptionusedescription Thingi10K training shard description. ",
        "- this run benchmark descriptionuse held-out eval slice, descriptiontodescription shard. ",
        "",
        "## method comparison",
        "",
        header + body,
        "## Conclusion",
        "",
        f"- `RTSTPFExact-Trained` final FN: `{trained.fn_count}`, candidate recall: `{trained.candidate_recall:.4f}`. ",
        f"- description `NoProposal`, trained STPF total-time difference: `{trained.total_ms - no_proposal.total_ms:.4f} ms`. ",
    ]
    if random_row is not None:
        lines.append(
            f"- description `RTSTPFExact-Random`, trained STPF total-time difference: `{trained.total_ms - random_row.total_ms:.4f} ms`. "
        )
    if result.hard_case_no_proposal is not None and result.hard_case_trained is not None:
        trained_hard = result.hard_case_trained
        baseline_hard = result.hard_case_no_proposal
        hard_reduction_vs_baseline = 1.0 - (
            trained_hard.exact_work_units / max(1.0e-9, baseline_hard.exact_work_units)
        )
        lines.extend(
            [
                "",
                "## Thingi10K Hard Cases high-density Benchmark",
                "",
                f"- avg candidates/query: `{result.hard_case_no_proposal.avg_candidates_per_query:.4f}`",
                f"- slab count: `{result.config.hard_case.slab_count}`",
                f"- patches/object: `{result.config.hard_case.patches_per_object}`",
                "",
                "| Method | Exact Calls | Fallback Calls | Exact Work Units | Proposal Wall(ms) | Scheduling Wall(ms) | Total Wall(ms) | FN |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for name, metrics in (
            ("NoProposal", result.hard_case_no_proposal),
            ("RTSTPFExact-Random", result.hard_case_random),
            ("RTSTPFExact-Trained", result.hard_case_trained),
        ):
            if metrics is None:
                continue
            lines.append(
                f"| {name} | {metrics.exact_call_count} | {metrics.fallback_call_count} "
                f"| {metrics.exact_work_units:.4f} | {metrics.proposal_wall_ms:.4f} "
                f"| {metrics.scheduling_wall_ms:.4f} | {metrics.total_wall_ms:.4f} | {metrics.fn_count} |"
            )
        lines.extend(
            [
                "",
                f"- hard-case trained STPF description `NoProposal`  exact work reduction: `{100.0 * hard_reduction_vs_baseline:.4f}%`. ",
            ]
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


__all__ = [
    "Thingi10KPaperBenchmarkArtifacts",
    "Thingi10KPaperBenchmarkConfig",
    "Thingi10KPaperBenchmarkDataset",
    "Thingi10KPaperBenchmarkResult",
    "build_thingi10k_paper_benchmark_dataset",
    "run_thingi10k_paper_benchmark",
    "write_thingi10k_paper_benchmark_report",
    "write_thingi10k_paper_benchmark_summary_json",
]
