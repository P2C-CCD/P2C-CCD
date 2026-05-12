from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import torch

from p2cccd.datasets.cad.abc_official import default_abc_official_root
from p2cccd.datasets.cad.abc_training import ABCProxyDatasetConfig, generate_abc_proxy_datasets
from p2cccd.datasets.objects.thingi10k_training import Thingi10KProxyDatasetConfig, generate_thingi10k_proxy_datasets
from p2cccd.proposal import (
    STPFModelPreset,
    STPFTrainingConfig,
    STPFTrainingRunConfig,
    build_stpf_model_from_checkpoint_payload,
    run_stpf_training,
)

from .abc_paper_benchmark import ABCPaperBenchmarkConfig, build_abc_paper_benchmark_dataset
from .pure_exact_cpu import PureExactCPUConfig
from .rt_stpf_exact import RTSTPFExactConfig, RTSTPFExactResult, run_rt_stpf_exact_on_generated_dataset
from .thingi10k_paper_benchmark import Thingi10KPaperBenchmarkConfig, build_thingi10k_paper_benchmark_dataset
from .trained_stpf_high_density import (
    HighDensitySTPFConfig,
    benchmark_no_proposal_on_high_density_workload,
    benchmark_stpf_on_high_density_workload,
    build_high_density_stpf_workload,
)


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass(frozen=True, slots=True)
class RTSTPFModelArchitectureSweepConfig:
    presets: tuple[str, ...] = (
        "micro_mlp",
        "tiny_mlp",
        "lightweight_mlp",
        "medium_mlp",
    )
    exact: PureExactCPUConfig = PureExactCPUConfig(
        eps_time=1.0e-5,
        eps_space=1.0e-8,
        max_subdivision_depth=24,
    )
    training_epochs: int = 4
    training_batch_size: int = 2048
    learning_rate: float = 1.0e-3
    model_device: str = _default_device()
    benchmark_output_dir: str = "src/benchmark"
    training_output_dir: str = "src/outputs/stpf_training"
    run_name: str = "rtstpf_model_architecture_sweep_run_id"
    abc_training_asset_limit: int = 96
    abc_training_pair_limit: int = 256
    abc_training_seed: int = 424242
    thingi_training_seed: int = 424242
    abc_hard_case: HighDensitySTPFConfig = HighDensitySTPFConfig(
        slab_count=16,
        patches_per_object=6,
        representative_attempt_limit=2,
        uncertainty_fallback_threshold=0.75,
    )
    thingi_hard_case: HighDensitySTPFConfig = HighDensitySTPFConfig(
        slab_count=12,
        patches_per_object=6,
        representative_attempt_limit=2,
        uncertainty_fallback_threshold=0.75,
    )


@dataclass(frozen=True, slots=True)
class _PresetDomainSummary:
    domain: str
    preset: str
    parameter_count: int
    checkpoint_path: str
    validation_interval_top1_recall: float
    benchmark_total_ms: float
    benchmark_rt_ms: float
    benchmark_proposal_ms: float
    benchmark_exact_ms: float
    benchmark_qps: float
    benchmark_fn_count: int
    benchmark_candidate_recall: float
    hard_case_exact_work_reduction: float


@dataclass(frozen=True, slots=True)
class RTSTPFModelArchitectureSweepResult:
    config: RTSTPFModelArchitectureSweepConfig
    summaries: tuple[_PresetDomainSummary, ...]
    recommended_preset: str
    report_path: Path
    summary_json_path: Path


def _parameter_count(model: torch.nn.Module) -> int:
    return sum(int(parameter.numel()) for parameter in model.parameters())


def _training_config(
    cfg: RTSTPFModelArchitectureSweepConfig,
    *,
    preset: str,
    seed: int,
) -> STPFTrainingConfig:
    return STPFTrainingConfig(
        epochs=cfg.training_epochs,
        batch_size=cfg.training_batch_size,
        learning_rate=cfg.learning_rate,
        seed=seed,
        device=cfg.model_device,
        validation_fraction=0.0,
        model_preset=STPFModelPreset(preset),
    )


def _load_model_from_checkpoint(path: str | Path, *, device: str) -> torch.nn.Module:
    payload = torch.load(path, map_location=device)
    model, state_dict = build_stpf_model_from_checkpoint_payload(payload)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def _train_on_thingi(
    cfg: RTSTPFModelArchitectureSweepConfig,
    *,
    preset: str,
) -> tuple[_PresetDomainSummary, torch.nn.Module]:
    bundle = generate_thingi10k_proxy_datasets(
        Thingi10KProxyDatasetConfig(seed=cfg.thingi_training_seed)
    )
    train_rows = list(bundle.train_dataset.rows)
    eval_rows = list(bundle.eval_dataset.rows)
    run = run_stpf_training(
        train_rows,
        STPFTrainingRunConfig(
            training=_training_config(cfg, preset=preset, seed=cfg.thingi_training_seed),
            output_dir=cfg.training_output_dir,
            run_name=f"{cfg.run_name}_thingi_{preset}",
        ),
        validation_rows=eval_rows,
    )
    dataset = build_thingi10k_paper_benchmark_dataset(
        Thingi10KPaperBenchmarkConfig(
            exact=cfg.exact,
            rt_backend_name="optix_rt",
            model_device=cfg.model_device,
            include_random_stpf=False,
            hard_case_enabled=False,
        )
    )[1]
    result = run_rt_stpf_exact_on_generated_dataset(
        dataset.generated_dataset,
        RTSTPFExactConfig(
            exact=cfg.exact,
            rt_backend_name="optix_rt",
            execution_profile="fastest_learned",
            allow_default_model=False,
            model_checkpoint_path=str(run.artifacts.model_state_path),
            model_device=cfg.model_device,
            model_preset=STPFModelPreset(preset),
        ),
    )
    model = _load_model_from_checkpoint(run.artifacts.model_state_path, device=cfg.model_device)
    hard_case_workload = build_high_density_stpf_workload(
        dataset.generated_dataset,
        cfg.thingi_hard_case,
        name=f"{cfg.run_name}_thingi_hard_{preset}",
    )
    hard_case_baseline = benchmark_no_proposal_on_high_density_workload(hard_case_workload)
    hard_case_trained = benchmark_stpf_on_high_density_workload(
        hard_case_workload,
        model=model,
        device=cfg.model_device,
        method_name=f"Thingi10K-{preset}-Trained",
    )
    summary = _PresetDomainSummary(
        domain="thingi10k_heldout",
        preset=preset,
        parameter_count=_parameter_count(model),
        checkpoint_path=str(run.artifacts.model_state_path),
        validation_interval_top1_recall=run.result.history[-1].interval_top1_recall,
        benchmark_total_ms=result.benchmark.total_ms,
        benchmark_rt_ms=result.benchmark.rt_ms,
        benchmark_proposal_ms=result.benchmark.proposal_ms,
        benchmark_exact_ms=result.benchmark.exact_ms,
        benchmark_qps=result.benchmark.qps,
        benchmark_fn_count=result.benchmark.fn_count,
        benchmark_candidate_recall=result.benchmark.candidate_recall,
        hard_case_exact_work_reduction=1.0
        - (hard_case_trained.exact_work_units / max(1.0e-9, hard_case_baseline.exact_work_units)),
    )
    return summary, model


def _train_on_abc_official(
    cfg: RTSTPFModelArchitectureSweepConfig,
    *,
    preset: str,
) -> _PresetDomainSummary:
    bundle = generate_abc_proxy_datasets(
        ABCProxyDatasetConfig(
            root=default_abc_official_root(),
            allow_demo_bootstrap=False,
            asset_limit=cfg.abc_training_asset_limit,
            pair_limit=cfg.abc_training_pair_limit,
            train_fraction=0.75,
            seed=cfg.abc_training_seed,
        )
    )
    train_rows = list(bundle.train_dataset.rows)
    eval_rows = list(bundle.eval_dataset.rows)
    run = run_stpf_training(
        train_rows,
        STPFTrainingRunConfig(
            training=_training_config(cfg, preset=preset, seed=cfg.abc_training_seed),
            output_dir=cfg.training_output_dir,
            run_name=f"{cfg.run_name}_abc_{preset}",
        ),
        validation_rows=eval_rows,
    )
    dataset = build_abc_paper_benchmark_dataset(
        ABCPaperBenchmarkConfig(
            use_official_root=True,
            allow_official_download=False,
            official_asset_limit=256,
            benchmark_asset_offset=128,
            benchmark_asset_count=64,
            pair_limit=512,
            exact=cfg.exact,
            rt_backend_name="optix_rt",
            model_device=cfg.model_device,
            include_random_stpf=False,
            hard_case_enabled=False,
            run_name=f"{cfg.run_name}_abc_official_large_benchmark",
        )
    )
    result = run_rt_stpf_exact_on_generated_dataset(
        dataset.generated_dataset,
        RTSTPFExactConfig(
            exact=cfg.exact,
            rt_backend_name="optix_rt",
            execution_profile="fastest_learned",
            allow_default_model=False,
            model_checkpoint_path=str(run.artifacts.model_state_path),
            model_device=cfg.model_device,
            model_preset=STPFModelPreset(preset),
        ),
    )
    model = _load_model_from_checkpoint(run.artifacts.model_state_path, device=cfg.model_device)
    hard_case_workload = build_high_density_stpf_workload(
        dataset.generated_dataset,
        cfg.abc_hard_case,
        name=f"{cfg.run_name}_abc_hard_{preset}",
    )
    hard_case_baseline = benchmark_no_proposal_on_high_density_workload(hard_case_workload)
    hard_case_trained = benchmark_stpf_on_high_density_workload(
        hard_case_workload,
        model=model,
        device=cfg.model_device,
        method_name=f"ABCOfficial-{preset}-Trained",
    )
    return _PresetDomainSummary(
        domain="abc_official_large",
        preset=preset,
        parameter_count=_parameter_count(model),
        checkpoint_path=str(run.artifacts.model_state_path),
        validation_interval_top1_recall=run.result.history[-1].interval_top1_recall,
        benchmark_total_ms=result.benchmark.total_ms,
        benchmark_rt_ms=result.benchmark.rt_ms,
        benchmark_proposal_ms=result.benchmark.proposal_ms,
        benchmark_exact_ms=result.benchmark.exact_ms,
        benchmark_qps=result.benchmark.qps,
        benchmark_fn_count=result.benchmark.fn_count,
        benchmark_candidate_recall=result.benchmark.candidate_recall,
        hard_case_exact_work_reduction=1.0
        - (hard_case_trained.exact_work_units / max(1.0e-9, hard_case_baseline.exact_work_units)),
    )


def _pick_recommended_preset(summaries: tuple[_PresetDomainSummary, ...]) -> str:
    by_preset: dict[str, list[float]] = {}
    for row in summaries:
        by_preset.setdefault(row.preset, []).append(row.benchmark_total_ms)
    return min(by_preset.items(), key=lambda item: sum(item[1]) / len(item[1]))[0]


def _report_markdown(result: RTSTPFModelArchitectureSweepResult) -> str:
    lines = [
        "# RTSTPFExact Model Architecture Sweep",
        "",
        "## Results",
        "",
        "| Domain | Preset | Params | Val Interval Recall | Total ms | RT ms | Proposal ms | Exact ms | QPS | FN | Recall | Hard-case exact-work reduction |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in result.summaries:
        lines.append(
            f"| {row.domain} | {row.preset} | {row.parameter_count} | {row.validation_interval_top1_recall:.4f} | "
            f"{row.benchmark_total_ms:.4f} | {row.benchmark_rt_ms:.4f} | {row.benchmark_proposal_ms:.4f} | "
            f"{row.benchmark_exact_ms:.4f} | {row.benchmark_qps:.2f} | {row.benchmark_fn_count} | "
            f"{row.benchmark_candidate_recall:.4f} | {100.0 * row.hard_case_exact_work_reduction:.4f}% |"
        )
    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            f"- Recommended learned preset: `{result.recommended_preset}`",
            "- Selection rule: lowest mean end-to-end `benchmark_total_ms` across `thingi10k_heldout` and `abc_official_large`.",
            "- Correctness gate: all kept presets must satisfy `FN = 0` and `Recall = 1.0`.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_rtstpf_model_architecture_sweep(
    config: RTSTPFModelArchitectureSweepConfig | None = None,
) -> RTSTPFModelArchitectureSweepResult:
    cfg = config or RTSTPFModelArchitectureSweepConfig()
    output_root = Path(cfg.benchmark_output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    summaries: list[_PresetDomainSummary] = []
    for preset in cfg.presets:
        thingi_summary, _ = _train_on_thingi(cfg, preset=preset)
        summaries.append(thingi_summary)
        summaries.append(_train_on_abc_official(cfg, preset=preset))

    result = RTSTPFModelArchitectureSweepResult(
        config=cfg,
        summaries=tuple(summaries),
        recommended_preset=_pick_recommended_preset(tuple(summaries)),
        report_path=output_root / f"{cfg.run_name}.md",
        summary_json_path=output_root / f"{cfg.run_name}.json",
    )
    result.report_path.write_text(_report_markdown(result), encoding="utf-8")
    result.summary_json_path.write_text(
        json.dumps(
            {
                "config": asdict(cfg),
                "recommended_preset": result.recommended_preset,
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
    "RTSTPFModelArchitectureSweepConfig",
    "RTSTPFModelArchitectureSweepResult",
    "run_rtstpf_model_architecture_sweep",
]
