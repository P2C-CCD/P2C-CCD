from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

from p2cccd.contracts import CertificateStatus
from p2cccd.data import DatasetGenerationConfig, generate_exact_oracle_dataset
from p2cccd.proposal import STPFModelPreset, build_stpf_model

from .rt_exact import RTExactConfig, RTExactResult, run_rt_exact_on_generated_dataset
from .rt_stpf_exact import RTSTPFExactConfig, RTSTPFExactResult, run_rt_stpf_exact_on_generated_dataset


NEAR_TERM_EXECUTION_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class NearTermExecutionConfig:
    dataset_seed: int = 127
    mesh_count_per_split: int = 2
    robot_link_count: int = 2
    include_robot_links: bool = True
    stpf_seed: int = 128
    stpf_hidden_dim: int = 16
    stpf_num_layers: int = 1
    proposal_batch_size: int = 4


@dataclass(frozen=True, slots=True)
class NearTermPipelineSummary:
    method_name: str
    query_count: int
    candidate_count: int
    work_item_count: int
    certificate_count: int
    collision_certificate_count: int
    separation_certificate_count: int
    undecided_certificate_count: int
    candidate_recall: float
    final_fn_zero: bool
    queue_conserved: bool
    monotonic_safe: bool
    fallback_ratio: float
    rt_ms: float
    proposal_ms: float
    exact_ms: float
    total_ms: float


@dataclass(frozen=True, slots=True)
class NearTermExecutionGateResult:
    schema_version: int
    dataset_seed: int
    dataset_query_count: int
    rt_exact_passed: bool
    stpf_v1_ran: bool
    stpf_v1_passed: bool
    rt_exact: NearTermPipelineSummary
    stpf_v1: NearTermPipelineSummary | None


def _certificate_status_count(result: RTExactResult | RTSTPFExactResult, status: CertificateStatus) -> int:
    return sum(1 for certificate in result.certificates if certificate.status is status)


def _summary(method_name: str, result: RTExactResult | RTSTPFExactResult) -> NearTermPipelineSummary:
    schedule_stats = getattr(result, "schedule_stats", None)
    monotonic_safe = bool(getattr(schedule_stats, "monotonic_safe", True))
    return NearTermPipelineSummary(
        method_name=method_name,
        query_count=result.benchmark.query_count,
        candidate_count=len(result.candidates),
        work_item_count=len(result.work_items),
        certificate_count=len(result.certificates),
        collision_certificate_count=_certificate_status_count(result, CertificateStatus.COLLISION),
        separation_certificate_count=_certificate_status_count(result, CertificateStatus.SEPARATION),
        undecided_certificate_count=_certificate_status_count(result, CertificateStatus.UNDECIDED),
        candidate_recall=result.benchmark.candidate_recall,
        final_fn_zero=result.final_fn_zero,
        queue_conserved=result.queue_conserved,
        monotonic_safe=monotonic_safe,
        fallback_ratio=result.benchmark.fallback_ratio,
        rt_ms=result.benchmark.rt_ms,
        proposal_ms=result.benchmark.proposal_ms,
        exact_ms=result.benchmark.exact_ms,
        total_ms=result.benchmark.total_ms,
    )


def _pipeline_passed(summary: NearTermPipelineSummary) -> bool:
    return (
        summary.final_fn_zero
        and summary.candidate_recall == 1.0
        and summary.queue_conserved
        and summary.monotonic_safe
        and summary.candidate_count == summary.work_item_count == summary.certificate_count
    )


def _build_lightweight_stpf_model(config: NearTermExecutionConfig):
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("STPF v1 gate requires PyTorch to instantiate the lightweight MLP") from exc
    torch.manual_seed(config.stpf_seed)
    model = build_stpf_model(
        STPFModelPreset.LIGHTWEIGHT_MLP,
        hidden_dim=config.stpf_hidden_dim,
        num_layers=config.stpf_num_layers,
    )
    model.eval()
    return model


def run_near_term_execution_gate(
    config: NearTermExecutionConfig | None = None,
) -> NearTermExecutionGateResult:
    cfg = config or NearTermExecutionConfig()
    dataset = generate_exact_oracle_dataset(
        DatasetGenerationConfig(
            mesh_count_per_split=cfg.mesh_count_per_split,
            robot_link_count=cfg.robot_link_count,
            seed=cfg.dataset_seed,
            include_robot_links=cfg.include_robot_links,
        )
    )

    rt_result = run_rt_exact_on_generated_dataset(dataset, RTExactConfig())
    rt_summary = _summary("RTExact", rt_result)
    rt_passed = _pipeline_passed(rt_summary)
    if not rt_passed:
        return NearTermExecutionGateResult(
            schema_version=NEAR_TERM_EXECUTION_SCHEMA_VERSION,
            dataset_seed=cfg.dataset_seed,
            dataset_query_count=len(dataset.rows),
            rt_exact_passed=False,
            stpf_v1_ran=False,
            stpf_v1_passed=False,
            rt_exact=rt_summary,
            stpf_v1=None,
        )

    model = _build_lightweight_stpf_model(cfg)
    stpf_result = run_rt_stpf_exact_on_generated_dataset(
        dataset,
        RTSTPFExactConfig(
            use_dummy_policy=False,
            proposal_batch_size=cfg.proposal_batch_size,
        ),
        model=model,
        device="cpu",
    )
    stpf_summary = _summary("RTSTPFExact:STPFv1", stpf_result)
    stpf_passed = _pipeline_passed(stpf_summary)
    return NearTermExecutionGateResult(
        schema_version=NEAR_TERM_EXECUTION_SCHEMA_VERSION,
        dataset_seed=cfg.dataset_seed,
        dataset_query_count=len(dataset.rows),
        rt_exact_passed=True,
        stpf_v1_ran=True,
        stpf_v1_passed=stpf_passed,
        rt_exact=rt_summary,
        stpf_v1=stpf_summary,
    )


def write_near_term_execution_gate_json(
    path: str | Path,
    result: NearTermExecutionGateResult,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(asdict(result), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output
