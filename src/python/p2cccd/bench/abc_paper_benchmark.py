from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import random
from pathlib import Path

from p2cccd.bench.trained_stpf_high_density import (
    HighDensityMethodMetrics,
    HighDensitySTPFConfig,
    benchmark_no_proposal_on_high_density_workload,
    benchmark_stpf_on_high_density_workload,
    build_high_density_stpf_workload,
)
from p2cccd.bench.bvh_exact import BVHExactConfig, BVHExactResult, run_bvh_exact_on_generated_dataset
from p2cccd.bench.no_proposal import NoProposalConfig, NoProposalResult, run_no_proposal_on_generated_dataset
from p2cccd.bench.pure_exact_cpu import PureExactCPUConfig, PureExactCPUResult, run_pure_exact_cpu_on_generated_dataset
from p2cccd.bench.rt_exact import RTExactConfig, RTExactResult, run_rt_exact_on_generated_dataset
from p2cccd.bench.rt_stpf_exact import RTSTPFExactConfig, RTSTPFExactResult, run_rt_stpf_exact_on_generated_dataset
from p2cccd.contracts import BenchmarkRow
from p2cccd.data import default_metadata, write_npz_shard
from p2cccd.data.dataset import GeneratedDataset
from p2cccd.datasets.cad.abc_adapter import ABCDatasetAdapter, default_abc_root
from p2cccd.datasets.cad.abc_official import default_abc_official_root, prepare_official_abc_minimal_root
from p2cccd.datasets.cad.abc_training import (
    ABC_DEMO_SUBSET_DIRNAME,
    _dataset_from_samples,
    _samples_from_pairs,
    bootstrap_abc_demo_subset,
)
from p2cccd.proposal.stpf_model import build_stpf_model, build_stpf_model_from_checkpoint_payload


def _default_trained_checkpoint() -> str:
    return str(
        Path("src/outputs/stpf_training")
        / "abc_training_20260422_demo_main"
        / "model_state.pt"
    )


@dataclass(frozen=True, slots=True)
class ABCPaperBenchmarkConfig:
    root: Path | None = None
    use_official_root: bool = False
    allow_official_download: bool = False
    official_asset_limit: int = 64
    official_mesh_variant: str = "stl2"
    official_chunk_name: str | None = None
    allow_demo_bootstrap: bool = True
    ensure_demo_asset_count: int = 48
    benchmark_asset_offset: int = 24
    benchmark_asset_count: int = 24
    pair_limit: int = 160
    seed: int = 424242
    exact: PureExactCPUConfig = field(default_factory=PureExactCPUConfig)
    rt_backend_name: str = "optix_rt"
    model_checkpoint_path: str = field(default_factory=_default_trained_checkpoint)
    model_device: str = "cuda"
    include_random_stpf: bool = True
    hard_case_enabled: bool = True
    hard_case: HighDensitySTPFConfig = HighDensitySTPFConfig(
        slab_count=16,
        patches_per_object=6,
        representative_attempt_limit=2,
        uncertainty_fallback_threshold=0.75,
    )
    benchmark_output_dir: str = "src/benchmark"
    benchmark_dataset_dir: str = "src/datasets/benchmark/cad_motion_bench"
    run_name: str = "abc_cad_paper_benchmark_run_id"


@dataclass(frozen=True, slots=True)
class ABCPaperBenchmarkDataset:
    source_root: Path
    used_demo_subset: bool
    asset_offset: int
    asset_count: int
    pair_count: int
    pair_ids: tuple[str, ...]
    asset_paths: tuple[str, ...]
    generated_dataset: GeneratedDataset
    dataset_npz_path: Path
    dataset_manifest_path: Path


@dataclass(frozen=True, slots=True)
class ABCPaperBenchmarkArtifacts:
    report_path: Path
    summary_json_path: Path


@dataclass(frozen=True, slots=True)
class ABCPaperBenchmarkResult:
    config: ABCPaperBenchmarkConfig
    dataset: ABCPaperBenchmarkDataset
    pure_exact_cpu: PureExactCPUResult
    bvh_exact: BVHExactResult
    rt_exact: RTExactResult
    no_proposal: NoProposalResult
    rtstpf_random: RTSTPFExactResult | None
    rtstpf_trained: RTSTPFExactResult
    hard_case_no_proposal: HighDensityMethodMetrics | None
    hard_case_random: HighDensityMethodMetrics | None
    hard_case_trained: HighDensityMethodMetrics | None
    artifacts: ABCPaperBenchmarkArtifacts


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


def _resolve_root(config: ABCPaperBenchmarkConfig) -> tuple[Path, bool]:
    if config.use_official_root:
        source_root = config.root if config.root is not None else default_abc_official_root()
        if config.allow_official_download:
            prepared = prepare_official_abc_minimal_root(
                source_root,
                asset_limit=max(config.official_asset_limit, config.benchmark_asset_offset + config.benchmark_asset_count),
                mesh_variant=config.official_mesh_variant,
                chunk_name=config.official_chunk_name,
            )
            return prepared, False
        adapter = ABCDatasetAdapter(source_root)
        asset_count = len(adapter.list_mesh_paths(limit=1_000_000)) if source_root.exists() else 0
        if asset_count >= config.benchmark_asset_offset + config.benchmark_asset_count:
            return source_root, False
        raise FileNotFoundError(
            f"official ABC root {source_root} does not provide enough assets; enable allow_official_download or prepare the root manually"
        )

    source_root = config.root if config.root is not None else default_abc_root()
    adapter = ABCDatasetAdapter(source_root)
    asset_count = len(adapter.list_mesh_paths(limit=1_000_000)) if source_root.exists() else 0
    used_demo_subset = (source_root / ABC_DEMO_SUBSET_DIRNAME).exists()
    if asset_count >= config.benchmark_asset_offset + config.benchmark_asset_count:
        return source_root, used_demo_subset

    if not config.allow_demo_bootstrap:
        raise FileNotFoundError(
            f"ABC root {source_root} does not provide enough assets for benchmark selection"
        )
    if asset_count > 0 and not used_demo_subset:
        raise ValueError(
            "ABC root has real assets but not enough for the requested held-out benchmark; "
            "either lower the benchmark_asset_offset/count or provide a larger root"
        )

    bootstrap_abc_demo_subset(source_root, asset_count=config.ensure_demo_asset_count)
    return source_root, True


def build_abc_paper_benchmark_dataset(
    config: ABCPaperBenchmarkConfig | None = None,
) -> ABCPaperBenchmarkDataset:
    cfg = config or ABCPaperBenchmarkConfig()
    if cfg.benchmark_asset_offset < 0:
        raise ValueError("benchmark_asset_offset must be non-negative")
    if cfg.benchmark_asset_count <= 0:
        raise ValueError("benchmark_asset_count must be positive")
    if cfg.pair_limit <= 0:
        raise ValueError("pair_limit must be positive")

    source_root, used_demo_subset = _resolve_root(cfg)
    adapter = ABCDatasetAdapter(source_root)
    assets = tuple(
        sorted(
            adapter.list_assets(limit=None),
            key=lambda asset: (str(asset.metadata.get("source_relative_path", "")), asset.asset_id),
        )
    )
    start = cfg.benchmark_asset_offset
    end = start + cfg.benchmark_asset_count
    if end > len(assets):
        raise ValueError(
            f"requested benchmark asset slice [{start}:{end}) exceeds available asset count {len(assets)}"
        )
    benchmark_assets = assets[start:end]
    all_pairs = list(adapter.generate_mesh_pairs(assets=benchmark_assets, limit=None))
    if len(all_pairs) < cfg.pair_limit:
        raise ValueError(
            f"benchmark asset subset only yields {len(all_pairs)} pairs, below pair_limit={cfg.pair_limit}"
        )
    rng = random.Random(cfg.seed)
    rng.shuffle(all_pairs)
    selected_pairs = tuple(sorted(all_pairs[: cfg.pair_limit], key=lambda pair: pair.pair_id))
    generated_dataset = _dataset_from_samples(_samples_from_pairs(selected_pairs, first_sample_id=4_000_001))

    output_root = Path(cfg.benchmark_dataset_dir) / cfg.run_name
    output_root.mkdir(parents=True, exist_ok=True)
    dataset_npz_path = output_root / "benchmark_dataset.npz"
    metadata = default_metadata(generated_dataset, seed=cfg.seed, source="abc_heldout_cad_benchmark")
    metadata["dataset_role"] = "benchmark"
    metadata["source_dataset"] = "ABC Dataset"
    metadata["used_demo_subset"] = used_demo_subset
    metadata["benchmark_asset_offset"] = cfg.benchmark_asset_offset
    metadata["benchmark_asset_count"] = cfg.benchmark_asset_count
    metadata["pair_limit"] = cfg.pair_limit
    write_npz_shard(dataset_npz_path, generated_dataset, metadata=metadata)

    dataset_manifest = {
        "run_name": cfg.run_name,
        "source_root": str(source_root),
        "used_demo_subset": used_demo_subset,
        "benchmark_asset_offset": cfg.benchmark_asset_offset,
        "benchmark_asset_count": cfg.benchmark_asset_count,
        "pair_limit": cfg.pair_limit,
        "seed": cfg.seed,
        "query_count": len(generated_dataset.samples),
        "row_count": len(generated_dataset.rows),
        "asset_paths": [
            str(asset.metadata.get("source_relative_path", asset.asset_path))
            for asset in benchmark_assets
        ],
        "pair_ids": [pair.pair_id for pair in selected_pairs],
        "dataset_npz_path": str(dataset_npz_path),
        "training_separation_note": (
            "This benchmark uses a held-out CAD asset slice and should not be mixed with ABC training shards."
        ),
    }
    dataset_manifest_path = output_root / "dataset_manifest.json"
    dataset_manifest_path.write_text(
        json.dumps(dataset_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return ABCPaperBenchmarkDataset(
        source_root=source_root,
        used_demo_subset=used_demo_subset,
        asset_offset=cfg.benchmark_asset_offset,
        asset_count=cfg.benchmark_asset_count,
        pair_count=len(selected_pairs),
        pair_ids=tuple(pair.pair_id for pair in selected_pairs),
        asset_paths=tuple(
            str(asset.metadata.get("source_relative_path", asset.asset_path))
            for asset in benchmark_assets
        ),
        generated_dataset=generated_dataset,
        dataset_npz_path=dataset_npz_path,
        dataset_manifest_path=dataset_manifest_path,
    )


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


def _result_summaries(result: ABCPaperBenchmarkResult) -> tuple[_MethodSummary, ...]:
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


def _load_trained_model_for_dense_benchmark(checkpoint_path: str, *, device: str):
    import torch

    resolved_device = str(device)
    checkpoint_payload = torch.load(checkpoint_path, map_location=resolved_device)
    model, state_dict = build_stpf_model_from_checkpoint_payload(checkpoint_payload)
    model.load_state_dict(state_dict)
    model.to(resolved_device)
    model.eval()
    return model


def run_abc_paper_benchmark(
    config: ABCPaperBenchmarkConfig | None = None,
) -> ABCPaperBenchmarkResult:
    cfg = config or ABCPaperBenchmarkConfig()
    dataset = build_abc_paper_benchmark_dataset(cfg)
    pure_exact_cpu = run_pure_exact_cpu_on_generated_dataset(dataset.generated_dataset)
    bvh_exact = run_bvh_exact_on_generated_dataset(
        dataset.generated_dataset,
        BVHExactConfig(exact=cfg.exact),
    )
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
            name=f"{cfg.run_name}_cad_hard_case",
        )
        hard_case_no_proposal = benchmark_no_proposal_on_high_density_workload(hard_case_workload)
        random_model = build_stpf_model()
        random_model.to(cfg.model_device)
        random_model.eval()
        hard_case_random = benchmark_stpf_on_high_density_workload(
            hard_case_workload,
            model=random_model,
            device=cfg.model_device,
            method_name="CADHard-RTSTPFExact-Random",
        )
        trained_model = _load_trained_model_for_dense_benchmark(
            cfg.model_checkpoint_path,
            device=cfg.model_device,
        )
        hard_case_trained = benchmark_stpf_on_high_density_workload(
            hard_case_workload,
            model=trained_model,
            device=cfg.model_device,
            method_name="CADHard-RTSTPFExact-Trained",
        )
    artifacts = ABCPaperBenchmarkArtifacts(
        report_path=Path(cfg.benchmark_output_dir) / f"{cfg.run_name}.md",
        summary_json_path=Path(cfg.benchmark_output_dir) / f"{cfg.run_name}.json",
    )
    result = ABCPaperBenchmarkResult(
        config=cfg,
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
    write_abc_paper_benchmark_report(artifacts.report_path, result)
    write_abc_paper_benchmark_summary_json(artifacts.summary_json_path, result)
    return result


def write_abc_paper_benchmark_summary_json(
    path: str | Path,
    result: ABCPaperBenchmarkResult,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_name": result.config.run_name,
        "dataset": {
            "source_root": str(result.dataset.source_root),
            "used_demo_subset": result.dataset.used_demo_subset,
            "asset_offset": result.dataset.asset_offset,
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
            "seed": result.config.seed,
            "pair_limit": result.config.pair_limit,
            "use_official_root": result.config.use_official_root,
            "hard_case_enabled": result.config.hard_case_enabled,
            "official_mesh_variant": result.config.official_mesh_variant,
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


def write_abc_paper_benchmark_report(
    path: str | Path,
    result: ABCPaperBenchmarkResult,
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
        "# ABC CAD paper-track benchmark",
        "",
        "## dataset",
        "",
        f"- Source root: `{result.dataset.source_root}`",
        f"- Used demo subset: `{result.dataset.used_demo_subset}`",
        f"- Held-out asset slice: `{result.dataset.asset_offset}:{result.dataset.asset_offset + result.dataset.asset_count}`",
        f"- Pair count: `{result.dataset.pair_count}`",
        f"- Query count: `{len(result.dataset.generated_dataset.samples)}`",
        f"- Dataset npz: `{result.dataset.dataset_npz_path}`",
        f"- Dataset manifest: `{result.dataset.dataset_manifest_path}`",
        "",
        "## description/Benchmark separation",
        "",
        "- description checkpoint descriptionusedescription ABC demo description. ",
        (
            "- this run benchmark descriptionusecurrent root in held-out CAD asset slice. "
            if not result.dataset.used_demo_subset
            else "- this run benchmark defaultdescription ABC demo subset afterdescription CAD assets description held-out benchmark slice. "
        ),
        "- thisdescriptionused foravoiddescription shard  pair/query descriptionconnectdescriptiontodescription benchmark. ",
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
        hard_rows = [
            ("NoProposal", result.hard_case_no_proposal),
            ("RTSTPFExact-Random", result.hard_case_random),
            ("RTSTPFExact-Trained", result.hard_case_trained),
        ]
        lines.extend(
            [
                "",
                "## CAD Hard Cases high-density Benchmark",
                "",
                f"- avg candidates/query: `{result.hard_case_no_proposal.avg_candidates_per_query:.4f}`",
                f"- slab count: `{result.config.hard_case.slab_count}`",
                f"- patches/object: `{result.config.hard_case.patches_per_object}`",
                "",
                "| Method | Exact Calls | Fallback Calls | Exact Work Units | Proposal Wall(ms) | Scheduling Wall(ms) | Total Wall(ms) | FN |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for name, metrics in hard_rows:
            if metrics is None:
                continue
            lines.append(
                f"| {name} | {metrics.exact_call_count} | {metrics.fallback_call_count} "
                f"| {metrics.exact_work_units:.4f} | {metrics.proposal_wall_ms:.4f} "
                f"| {metrics.scheduling_wall_ms:.4f} | {metrics.total_wall_ms:.4f} | {metrics.fn_count} |"
            )
        trained_hard = result.hard_case_trained
        baseline_hard = result.hard_case_no_proposal
        hard_reduction_vs_baseline = 1.0 - (
            trained_hard.exact_work_units / max(1.0e-9, baseline_hard.exact_work_units)
        )
        lines.extend(
            [
                "",
                f"- hard-case trained STPF description `NoProposal`  exact work reduction: `{100.0 * hard_reduction_vs_baseline:.4f}%`. ",
            ]
        )
    if result.dataset.used_demo_subset:
        lines.extend(
            [
                "",
                "## Protocol notes",
                "",
                "- currentis `ABC-compatible local demo subset`  CAD benchmark, is notofficial full-scale ABC description. ",
                "- description, description/benchmark separation, RT/STPF/exact descriptionMethoddescriptionbydescriptionPathrealdescription. ",
            ]
        )
    elif result.config.use_official_root:
        lines.extend(
            [
                "",
                "## official ABC Root Notes",
                "",
                f"- this run root fromofficial ABC `{result.config.official_mesh_variant}` mesh chunk descriptionofficialdescription. ",
                "- this root is not demo subset, sourceand chunk description root under `official_subset_manifest.json`. ",
            ]
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


__all__ = [
    "ABCPaperBenchmarkArtifacts",
    "ABCPaperBenchmarkConfig",
    "ABCPaperBenchmarkDataset",
    "ABCPaperBenchmarkResult",
    "build_abc_paper_benchmark_dataset",
    "run_abc_paper_benchmark",
    "write_abc_paper_benchmark_report",
    "write_abc_paper_benchmark_summary_json",
]
