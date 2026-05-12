from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Callable, Sequence

from p2cccd.proposal.stpf_model import STPFModelPreset, build_stpf_model, build_stpf_model_from_checkpoint_payload
from p2cccd.proposal.training import STPFTrainingConfig

from .complete_example_scaled_training_benchmark import _MethodRow
from .generalization_paper_benchmark import (
    GeneralizationDenseRow,
    GeneralizationPaperBenchmarkConfig,
    GeneralizationQueryRow,
    GeneralizationSourcePack,
    _benchmark_dense_sources,
    _benchmark_query_sources,
    _build_abc_pack,
    _build_fusion360_pack,
    _build_high_density_mesh_pack,
    _build_synthetic_high_density_pack,
    _build_t0_pack,
    _build_thingi10k_pack,
    _dense_reduction,
)


def _default_training() -> STPFTrainingConfig:
    return STPFTrainingConfig(
        epochs=4,
        batch_size=8192,
        learning_rate=8.0e-4,
        seed=424242,
        device="cuda",
        validation_fraction=0.0,
        model_preset=STPFModelPreset.MEDIUM_MLP,
    )


@dataclass(frozen=True, slots=True)
class CurrentPaperBenchmarkSuiteConfig:
    run_name: str = "current_paper_benchmark_suite_run_id"
    checkpoint_path: str = "src/outputs/stpf_training/generalization_paper_benchmark_full_run_id/model_state.pt"
    benchmark_output_dir: str = "src/benchmark"
    training_output_dir: str = "src/outputs/stpf_training"
    shard_root: str = "src/datasets/training/generalization/shards"
    model_device: str = "cuda"
    rt_backend_name: str = "optix_rt"
    proposal_batch_size: int = 8192
    training: STPFTrainingConfig = _default_training()
    include_random_stpf_dense: bool = True


@dataclass(frozen=True, slots=True)
class SourceBuildFailure:
    builder_name: str
    error: str


@dataclass(frozen=True, slots=True)
class CurrentPaperBenchmarkSuiteResult:
    config: CurrentPaperBenchmarkSuiteConfig
    source_config: GeneralizationPaperBenchmarkConfig
    sources: tuple[GeneralizationSourcePack, ...]
    source_build_failures: tuple[SourceBuildFailure, ...]
    dense_rows: tuple[GeneralizationDenseRow, ...]
    query_rows: tuple[GeneralizationQueryRow, ...]
    elapsed_wall_s: float
    report_path: Path
    summary_json_path: Path


def _source_config(cfg: CurrentPaperBenchmarkSuiteConfig) -> GeneralizationPaperBenchmarkConfig:
    return GeneralizationPaperBenchmarkConfig(
        run_name=cfg.run_name,
        training=cfg.training,
        benchmark_output_dir=cfg.benchmark_output_dir,
        training_output_dir=cfg.training_output_dir,
        shard_root=cfg.shard_root,
        model_device=cfg.model_device,
        rt_backend_name=cfg.rt_backend_name,
        proposal_batch_size=cfg.proposal_batch_size,
    )


def _build_available_sources(
    cfg: GeneralizationPaperBenchmarkConfig,
) -> tuple[tuple[GeneralizationSourcePack, ...], tuple[SourceBuildFailure, ...]]:
    builders: tuple[Callable[[GeneralizationPaperBenchmarkConfig], GeneralizationSourcePack], ...] = (
        _build_t0_pack,
        _build_synthetic_high_density_pack,
        _build_abc_pack,
        _build_thingi10k_pack,
        _build_fusion360_pack,
        _build_high_density_mesh_pack,
    )
    sources: list[GeneralizationSourcePack] = []
    failures: list[SourceBuildFailure] = []
    for builder in builders:
        try:
            sources.append(builder(cfg))
        except Exception as exc:  # pragma: no cover - used to keep partial suites runnable.
            failures.append(SourceBuildFailure(builder.__name__, repr(exc)))
    return tuple(sources), tuple(failures)


def _load_trained_model(checkpoint_path: Path, *, device: str):
    import torch

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"STPF checkpoint not found: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location=device)
    model, state_dict = build_stpf_model_from_checkpoint_payload(
        payload,
        fallback_preset=STPFModelPreset.MEDIUM_MLP,
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def _fmt_ms(value: float) -> str:
    return f"{float(value):.3f}"


def _fmt_pct(value: float) -> str:
    return f"{100.0 * float(value):.2f}%"


def _fmt_ratio(value: float) -> str:
    return f"{float(value):.2f}x"


def _row_by_method(rows: Sequence[_MethodRow], method: str) -> _MethodRow | None:
    return next((row for row in rows if row.method == method), None)


def _fastest_method(rows: Sequence[_MethodRow]) -> _MethodRow:
    return min(rows, key=lambda row: row.total_ms)


def _write_report(path: Path, result: CurrentPaperBenchmarkSuiteResult) -> None:
    dense_reductions = [_dense_reduction(row.trained_stpf, row.no_proposal) for row in result.dense_rows]
    avg_dense_reduction = sum(dense_reductions) / max(1, len(dense_reductions))
    dense_total_fn = sum(row.trained_stpf.fn_count for row in result.dense_rows)
    query_total_fn = sum(method.fn_count for group in result.query_rows for method in group.rows)
    query_count = sum(group.rows[0].query_count for group in result.query_rows if group.rows)
    provider_names = sorted(
        {
            row.inference_provider_name
            for group in result.query_rows
            for row in group.rows
            if row.method == "RTSTPFExact"
        }
    )

    lines = [
        "# Current runnable benchmark full replay and performance analysis",
        "",
        "## Protocol",
        "",
        f"- Run name: `{result.config.run_name}`",
        f"- Checkpoint: `{result.config.checkpoint_path}`",
        f"- descriptionwhen: `{result.elapsed_wall_s:.2f} s`",
        "- descriptionnewdescription, descriptionconnectdescriptionusedescriptiongeneralization STPF checkpoint; descriptionPath, descriptionPathand exact work currentdescription. ",
        "- RTSTPFExact constraint: description learned STPF description; descriptionlayer `use_dummy_policy=True` descriptionconnectdescription. ",
        "- defaultthis paperPath: `medium_mlp + ORT(TensorRT EP description) + optix_rt + CUDA exact`. ",
        "- Dense exact-work description proposal isdescriptionreduction candidate/primitive level exact work; Query-level descriptioncurrentdescriptiontodescription runner  wall time. ",
        "",
        "## description",
        "",
        f"- coveragedescription: `{len(result.sources)}` , query-level description: `{query_count}`. ",
        f"- Dense learned STPF description exact-work reduction: `{_fmt_pct(avg_dense_reduction)}`. ",
        f"- Dense RTSTPFExact FN: `{dense_total_fn}`; descriptionPath query-level description FN: `{query_total_fn}`. ",
        f"- description RTSTPFExact ORT provider: `{', '.join(provider_names) if provider_names else 'n/a'}`. ",
    ]
    if result.source_build_failures:
        lines.extend(
            [
                "- hasdata sourcedescriptionconstruct, description; description `descriptiondata source`. ",
            ]
        )
    lines.extend(
        [
            "",
            "currentConclusion: this paperMethoddescriptioninhigh candidate-density / high exact-work scene, description exact work descriptionto NoProposal description, descriptionkeep FN=0. ordinary query-level wall time on, `PureExactCPU` or `NoProposal` description, descriptionascurrent runner  exact oracle description, STPF descriptionanddescriptionoverheaddescription. ",
            "",
            "## data coverage",
            "",
            "| Example | Source | Train queries | Eval queries | Dense eval rows | Note |",
            "| --- | --- | ---: | ---: | ---: | --- |",
        ]
    )
    for pack in result.sources:
        lines.append(
            f"| `{pack.name}` | `{pack.source_name}` | `{len(pack.train_dataset.samples)}` | "
            f"`{len(pack.eval_dataset.samples)}` | `{len(pack.eval_workload.rows)}` | {pack.note} |"
        )
    if result.source_build_failures:
        lines.extend(
            [
                "",
                "## descriptiondata source",
                "",
                "| Builder | Error |",
                "| --- | --- |",
            ]
        )
        for failure in result.source_build_failures:
            lines.append(f"| `{failure.builder_name}` | `{failure.error}` |")
    lines.extend(
        [
            "",
            "## Dense Exact-Work description",
            "",
            "| Example | Avg candidates/query | NoProposal exact calls | RTSTPFExact exact calls | Exact-work reduction | Proposal ms | Total ms | FN |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in result.dense_rows:
        reduction = _dense_reduction(row.trained_stpf, row.no_proposal)
        lines.append(
            f"| `{row.example_name}` | `{row.trained_stpf.avg_candidates_per_query:.1f}` | "
            f"`{row.no_proposal.exact_call_count}` | `{row.trained_stpf.exact_call_count}` | "
            f"`{_fmt_pct(reduction)}` | `{_fmt_ms(row.trained_stpf.proposal_wall_ms)}` | "
            f"`{_fmt_ms(row.trained_stpf.total_wall_ms)}` | `{row.trained_stpf.fn_count}` |"
        )
    lines.extend(
        [
            "",
            "## Query-Level descriptionPathdescription",
            "",
            "| Example | Method | Queries | Total ms | RT ms | Proposal ms | Exact ms | QPS | Avg candidates | Avg exact evals | FN | Recall | Provider |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for group in result.query_rows:
        for row in group.rows:
            provider = row.inference_provider_name if row.method == "RTSTPFExact" else row.exact_backend_name
            lines.append(
                f"| `{group.example_name}` | `{row.method}` | `{row.query_count}` | `{_fmt_ms(row.total_ms)}` | "
                f"`{_fmt_ms(row.rt_ms)}` | `{_fmt_ms(row.proposal_ms)}` | `{_fmt_ms(row.exact_ms)}` | "
                f"`{row.qps:.1f}` | `{row.avg_candidates:.2f}` | `{row.avg_exact_evals:.2f}` | "
                f"`{row.fn_count}` | `{row.candidate_recall:.4f}` | `{provider}` |"
            )
    lines.extend(
        [
            "",
            "## Wall-Time descriptionConclusion",
            "",
            "| Example | Fastest | Fastest total ms | RTSTPFExact total ms | NoProposal total ms | RTSTPF/NoProposal | RTSTPF/PureCPU |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for group in result.query_rows:
        fastest = _fastest_method(group.rows)
        rtstpf = _row_by_method(group.rows, "RTSTPFExact")
        no_proposal = _row_by_method(group.rows, "NoProposal")
        pure = _row_by_method(group.rows, "PureExactCPU")
        rt_vs_no = (
            _fmt_ratio(rtstpf.total_ms / max(1.0e-9, no_proposal.total_ms))
            if rtstpf is not None and no_proposal is not None
            else "n/a"
        )
        rt_vs_pure = (
            _fmt_ratio(rtstpf.total_ms / max(1.0e-9, pure.total_ms))
            if rtstpf is not None and pure is not None
            else "n/a"
        )
        lines.append(
            f"| `{group.example_name}` | `{fastest.method}` | `{_fmt_ms(fastest.total_ms)}` | "
            f"`{_fmt_ms(rtstpf.total_ms) if rtstpf else 'n/a'}` | "
            f"`{_fmt_ms(no_proposal.total_ms) if no_proposal else 'n/a'}` | `{rt_vs_no}` | `{rt_vs_pure}` |"
        )
    lines.extend(
        [
            "",
            "## descriptionsplitdescription",
            "",
            "- correctness: description dense and query-level description `FN=0`. ifafterdescription, descriptionkeepdescription exact certificate description. ",
            "- this paperadvantage: RTSTPFExact description dense candidate queue hasdescription, descriptionisuse learned proposal descriptionsplitdescriptioncandidatefrom exact work indescription, description exact certificate descriptioncorrectness. ",
            "- currentdescription: query-level wall time is notdescription, descriptioniscurrent benchmark  primitive/exact oracle description, proposal description, descriptionconstruct, descriptionand ORT runtime fixedoverheaddescription exact-work description. ",
            "- description: descriptioninhigh candidate-density, highdescription, high exact-cost  mesh-mesh CCD; description query scenedescriptionasdescriptionNotes, must not treat RTSTPFExact descriptionasallscenedescription. ",
            "- underdescription: real mesh primitive exact descriptionwhendescriptionsame hot path, description ABC/Thingi10K/Fusion real triangle-surface contact benchmark, description exact-work reduction descriptionasdescriptiontodescription wall-time advantage. ",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_json(path: Path, result: CurrentPaperBenchmarkSuiteResult) -> None:
    payload = {
        "config": asdict(result.config),
        "source_config": asdict(result.source_config),
        "elapsed_wall_s": result.elapsed_wall_s,
        "sources": [
            {
                "name": pack.name,
                "source_name": pack.source_name,
                "train_query_count": len(pack.train_dataset.samples),
                "eval_query_count": len(pack.eval_dataset.samples),
                "dense_train_rows": len(pack.train_workload.rows),
                "dense_eval_rows": len(pack.eval_workload.rows),
                "note": pack.note,
            }
            for pack in result.sources
        ],
        "source_build_failures": [asdict(failure) for failure in result.source_build_failures],
        "dense_rows": [
            {
                "example_name": row.example_name,
                "no_proposal": asdict(row.no_proposal),
                "random_stpf": asdict(row.random_stpf),
                "trained_stpf": asdict(row.trained_stpf),
                "reduction_vs_no_proposal": _dense_reduction(row.trained_stpf, row.no_proposal),
            }
            for row in result.dense_rows
        ],
        "query_rows": [
            {
                "example_name": group.example_name,
                "rows": [asdict(row) for row in group.rows],
            }
            for group in result.query_rows
        ],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def run_current_paper_benchmark_suite(
    config: CurrentPaperBenchmarkSuiteConfig | None = None,
) -> CurrentPaperBenchmarkSuiteResult:
    cfg = config or CurrentPaperBenchmarkSuiteConfig()
    start = time.perf_counter()
    source_cfg = _source_config(cfg)
    sources, source_failures = _build_available_sources(source_cfg)
    if not sources:
        raise RuntimeError("no benchmark sources could be constructed")

    checkpoint_path = Path(cfg.checkpoint_path)
    trained_model = _load_trained_model(checkpoint_path, device=cfg.model_device)
    random_model = build_stpf_model(cfg.training.model_preset)
    random_model.eval()
    random_model.to(cfg.model_device)

    dense_rows = _benchmark_dense_sources(
        sources,
        model=trained_model,
        random_model=random_model,
        device=cfg.model_device,
        batch_size=cfg.proposal_batch_size,
    )
    query_rows = _benchmark_query_sources(source_cfg, sources, checkpoint_path=str(checkpoint_path))

    output_root = Path(cfg.benchmark_output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    report_path = output_root / f"{cfg.run_name}.md"
    summary_json_path = output_root / f"{cfg.run_name}.json"
    result = CurrentPaperBenchmarkSuiteResult(
        config=cfg,
        source_config=source_cfg,
        sources=sources,
        source_build_failures=source_failures,
        dense_rows=dense_rows,
        query_rows=query_rows,
        elapsed_wall_s=time.perf_counter() - start,
        report_path=report_path,
        summary_json_path=summary_json_path,
    )
    _write_report(report_path, result)
    _write_json(summary_json_path, result)
    return result


__all__ = [
    "CurrentPaperBenchmarkSuiteConfig",
    "CurrentPaperBenchmarkSuiteResult",
    "SourceBuildFailure",
    "run_current_paper_benchmark_suite",
]
