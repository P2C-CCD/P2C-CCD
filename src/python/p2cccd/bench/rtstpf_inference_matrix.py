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
from .rt_stpf_exact import RTSTPFExactConfig, run_rt_stpf_exact_on_generated_dataset
from .thingi10k_paper_benchmark import Thingi10KPaperBenchmarkConfig, build_thingi10k_paper_benchmark_dataset


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass(frozen=True, slots=True)
class RTSTPFInferenceMatrixConfig:
    presets: tuple[str, ...] = (
        "micro_mlp",
        "tiny_mlp",
        "lightweight_mlp",
        "medium_mlp",
        "high_capacity_mlp",
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
    run_name: str = "rtstpf_inference_matrix_run_id"
    abc_training_asset_limit: int = 96
    abc_training_pair_limit: int = 256
    abc_training_seed: int = 424242
    thingi_training_seed: int = 424242


@dataclass(frozen=True, slots=True)
class _DomainAssets:
    domain: str
    benchmark_dataset: object
    train_rows: list
    validation_rows: list


@dataclass(frozen=True, slots=True)
class _ModelArtifacts:
    checkpoint_path: str
    summary_path: str
    parameter_count: int
    validation_interval_top1_recall: float


@dataclass(frozen=True, slots=True)
class _RouteSpec:
    route_name: str
    inference_backend: str
    model_device: str
    ort_prefer_tensorrt: bool
    ort_allow_cuda_fallback: bool
    ort_allow_cpu_fallback: bool


@dataclass(frozen=True, slots=True)
class RTSTPFInferenceMatrixRow:
    domain: str
    preset: str
    route_name: str
    parameter_count: int
    validation_interval_top1_recall: float
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
class RTSTPFInferenceMatrixResult:
    config: RTSTPFInferenceMatrixConfig
    rows: tuple[RTSTPFInferenceMatrixRow, ...]
    report_path: Path
    summary_json_path: Path


def _training_config(
    cfg: RTSTPFInferenceMatrixConfig,
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


def _architecture_sweep_training_dir(domain: str, preset: str) -> Path:
    suffix = "thingi" if domain == "thingi10k_heldout" else "abc"
    return Path("src/outputs/stpf_training") / f"rtstpf_model_architecture_sweep_run_id_{suffix}_{preset}"


def _load_model_artifacts(checkpoint_path: Path, summary_path: Path) -> _ModelArtifacts:
    payload = torch.load(checkpoint_path, map_location="cpu")
    model, _ = build_stpf_model_from_checkpoint_payload(payload)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return _ModelArtifacts(
        checkpoint_path=str(checkpoint_path),
        summary_path=str(summary_path),
        parameter_count=sum(int(parameter.numel()) for parameter in model.parameters()),
        validation_interval_top1_recall=float(summary["final_validation_interval_top1_recall"]),
    )


def _ensure_checkpoint(
    cfg: RTSTPFInferenceMatrixConfig,
    *,
    domain_assets: _DomainAssets,
    preset: str,
) -> _ModelArtifacts:
    existing_dir = _architecture_sweep_training_dir(domain_assets.domain, preset)
    existing_checkpoint = existing_dir / "model_state.pt"
    existing_summary = existing_dir / "summary.json"
    if existing_checkpoint.exists() and existing_summary.exists():
        return _load_model_artifacts(existing_checkpoint, existing_summary)

    output_dir = (
        Path(cfg.training_output_dir)
        / f"{cfg.run_name}_{domain_assets.domain}_{preset}"
    )
    checkpoint_path = output_dir / "model_state.pt"
    summary_path = output_dir / "summary.json"
    if checkpoint_path.exists() and summary_path.exists():
        return _load_model_artifacts(checkpoint_path, summary_path)

    seed = cfg.thingi_training_seed if domain_assets.domain == "thingi10k_heldout" else cfg.abc_training_seed
    run = run_stpf_training(
        domain_assets.train_rows,
        STPFTrainingRunConfig(
            training=_training_config(cfg, preset=preset, seed=seed),
            output_dir=cfg.training_output_dir,
            run_name=f"{cfg.run_name}_{domain_assets.domain}_{preset}",
        ),
        validation_rows=domain_assets.validation_rows,
    )
    return _load_model_artifacts(run.artifacts.model_state_path, run.artifacts.summary_json)


def _build_domain_assets(
    cfg: RTSTPFInferenceMatrixConfig,
) -> tuple[_DomainAssets, ...]:
    thingi_bundle, thingi_dataset = build_thingi10k_paper_benchmark_dataset(
        Thingi10KPaperBenchmarkConfig(
            exact=cfg.exact,
            rt_backend_name="optix_rt",
            model_device=cfg.model_device,
            include_random_stpf=False,
            hard_case_enabled=False,
        )
    )
    abc_bundle = generate_abc_proxy_datasets(
        ABCProxyDatasetConfig(
            root=default_abc_official_root(),
            allow_demo_bootstrap=False,
            asset_limit=cfg.abc_training_asset_limit,
            pair_limit=cfg.abc_training_pair_limit,
            train_fraction=0.75,
            seed=cfg.abc_training_seed,
        )
    )
    abc_dataset = build_abc_paper_benchmark_dataset(
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
    return (
        _DomainAssets(
            domain="thingi10k_heldout",
            benchmark_dataset=thingi_dataset.generated_dataset,
            train_rows=list(thingi_bundle.train_dataset.rows),
            validation_rows=list(thingi_bundle.eval_dataset.rows),
        ),
        _DomainAssets(
            domain="abc_official_large",
            benchmark_dataset=abc_dataset.generated_dataset,
            train_rows=list(abc_bundle.train_dataset.rows),
            validation_rows=list(abc_bundle.eval_dataset.rows),
        ),
    )


def _route_specs() -> tuple[_RouteSpec, ...]:
    return (
        _RouteSpec(
            route_name="torch",
            inference_backend="torch",
            model_device="cuda",
            ort_prefer_tensorrt=False,
            ort_allow_cuda_fallback=False,
            ort_allow_cpu_fallback=False,
        ),
        _RouteSpec(
            route_name="ort_tensorrt",
            inference_backend="ort",
            model_device="cuda",
            ort_prefer_tensorrt=True,
            ort_allow_cuda_fallback=False,
            ort_allow_cpu_fallback=False,
        ),
        _RouteSpec(
            route_name="ort_cudaep",
            inference_backend="ort",
            model_device="cuda",
            ort_prefer_tensorrt=False,
            ort_allow_cuda_fallback=True,
            ort_allow_cpu_fallback=False,
        ),
        _RouteSpec(
            route_name="ort_cpu",
            inference_backend="ort",
            model_device="cpu",
            ort_prefer_tensorrt=False,
            ort_allow_cuda_fallback=False,
            ort_allow_cpu_fallback=True,
        ),
    )


def _run_one(
    cfg: RTSTPFInferenceMatrixConfig,
    *,
    domain_assets: _DomainAssets,
    preset: str,
    model_artifacts: _ModelArtifacts,
    route: _RouteSpec,
) -> RTSTPFInferenceMatrixRow:
    result = run_rt_stpf_exact_on_generated_dataset(
        domain_assets.benchmark_dataset,
        RTSTPFExactConfig(
            exact=cfg.exact,
            rt_backend_name="optix_rt",
            execution_profile="manual",
            allow_default_model=False,
            model_checkpoint_path=model_artifacts.checkpoint_path,
            model_preset=STPFModelPreset(preset),
            model_device=route.model_device,
            inference_backend=route.inference_backend,
            ort_prefer_tensorrt=route.ort_prefer_tensorrt,
            ort_allow_cuda_fallback=route.ort_allow_cuda_fallback,
            ort_allow_cpu_fallback=route.ort_allow_cpu_fallback,
            cpu_inference_row_threshold=0,
            enable_cuda_exact=True,
        ),
        device=route.model_device,
    )
    row = result.benchmark
    return RTSTPFInferenceMatrixRow(
        domain=domain_assets.domain,
        preset=preset,
        route_name=route.route_name,
        parameter_count=model_artifacts.parameter_count,
        validation_interval_top1_recall=model_artifacts.validation_interval_top1_recall,
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


def _report_markdown(result: RTSTPFInferenceMatrixResult) -> str:
    lines = [
        "# RTSTPFExact Inference Matrix",
        "",
        "- Meaningful combinations from the requested module set: `5 presets x 4 runtime routes = 20`.",
        "- Routes in this matrix:",
        "  - `torch`",
        "  - `ort_tensorrt`",
        "  - `ort_cudaep`",
        "  - `ort_cpu`",
        "",
    ]
    for domain in ("thingi10k_heldout", "abc_official_large"):
        lines.extend(
            [
                f"## {domain}",
                "",
                "| Preset | Route | Params | Val Interval Recall | Total ms | RT ms | Proposal ms | Exact ms | QPS | FN | Recall | Provider |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        rows = [row for row in result.rows if row.domain == domain]
        for row in sorted(rows, key=lambda item: (item.preset, item.total_ms)):
            lines.append(
                f"| {row.preset} | {row.route_name} | {row.parameter_count} | {row.validation_interval_top1_recall:.4f} | "
                f"{row.total_ms:.4f} | {row.rt_ms:.4f} | {row.proposal_ms:.4f} | {row.exact_ms:.4f} | "
                f"{row.qps:.2f} | {row.fn_count} | {row.candidate_recall:.4f} | {row.inference_provider_name} |"
            )
        best = min(rows, key=lambda item: item.total_ms)
        lines.extend(
            [
                "",
                f"- Fastest on `{domain}`: `{best.preset} + {best.route_name}` with `{best.total_ms:.4f} ms`.",
                "",
            ]
        )
    lines.extend(
        [
            "## Notes",
            "",
            "- `FN` and `Recall` are end-to-end safety metrics from the RTSTPFExact benchmark result.",
            "- `Val Interval Recall` is the final validation interval top-1 recall from training output for the corresponding checkpoint.",
            "- Generated-dataset exact backends remain dataset-dependent (`swept_sphere_oracle_cpu` on these two domains).",
        ]
    )
    return "\n".join(lines) + "\n"


def run_rtstpf_inference_matrix(
    config: RTSTPFInferenceMatrixConfig | None = None,
) -> RTSTPFInferenceMatrixResult:
    cfg = config or RTSTPFInferenceMatrixConfig()
    output_root = Path(cfg.benchmark_output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    rows: list[RTSTPFInferenceMatrixRow] = []
    for domain_assets in _build_domain_assets(cfg):
        for preset in cfg.presets:
            artifacts = _ensure_checkpoint(cfg, domain_assets=domain_assets, preset=preset)
            for route in _route_specs():
                rows.append(
                    _run_one(
                        cfg,
                        domain_assets=domain_assets,
                        preset=preset,
                        model_artifacts=artifacts,
                        route=route,
                    )
                )

    result = RTSTPFInferenceMatrixResult(
        config=cfg,
        rows=tuple(rows),
        report_path=output_root / f"{cfg.run_name}.md",
        summary_json_path=output_root / f"{cfg.run_name}.json",
    )
    result.report_path.write_text(_report_markdown(result), encoding="utf-8")
    result.summary_json_path.write_text(
        json.dumps(
            {
                "config": asdict(cfg),
                "rows": [asdict(row) for row in result.rows],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return result


__all__ = [
    "RTSTPFInferenceMatrixConfig",
    "RTSTPFInferenceMatrixResult",
    "run_rtstpf_inference_matrix",
]
