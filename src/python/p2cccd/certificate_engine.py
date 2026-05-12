from __future__ import annotations

from dataclasses import dataclass, field
import importlib
from typing import Any, Sequence

from p2cccd.contracts import AuditLogRow, CandidateRecord, CertificateResult, ExactWorkItem
from p2cccd.data.dataset import GeneratedDataset
from p2cccd.data.samplers import MotionDiscPairSample
from p2cccd.datasets.ccd import DatasetQueryBatch
from p2cccd.validators import (
    validate_audit_log_row,
    validate_certificate_result,
    validate_exact_work_item,
)

from .bench.bvh_exact import BroadPhaseBackend
from .bench.pure_exact_cpu import PureExactCPUConfig, PureExactQueryResult
from .bench.rt_exact import (
    RTExactConfig,
    _process_external_exact_work_queue,
    _process_internal_exact_work_queue,
    schedule_exact_work_items_without_stpf,
    validate_rt_exact_coverage,
)
from .candidate_generation import (
    CandidateGenerationWrapperConfig,
    CandidateGenerationWrapperResult,
    generate_candidates_for_external_batch,
    generate_candidates_for_generated_dataset,
    generate_candidates_for_internal_samples,
)


CPP_CERTIFICATE_ENTRYPOINTS = (
    "execute_certificate_engine",
    "process_exact_work_queue",
    "evaluate_certificate_query",
)


@dataclass(frozen=True, slots=True)
class CertificateEngineWrapperConfig:
    exact: PureExactCPUConfig = field(default_factory=PureExactCPUConfig)
    first_work_item_id: int = 1
    first_event_id: int = 1
    first_timestamp_us: int = 1
    prefer_cpp_backend: bool = False
    allow_python_fallback: bool = True


@dataclass(frozen=True, slots=True)
class CertificateEngineWrapperResult:
    query_results: tuple[PureExactQueryResult, ...]
    candidates: tuple[CandidateRecord, ...]
    work_items: tuple[ExactWorkItem, ...]
    certificates: tuple[CertificateResult, ...]
    audit_log: tuple[AuditLogRow, ...]
    exact_ms: float
    source_name: str
    scene_name: str
    batch_id: str
    used_cpp_backend: bool
    fallback_reason: str

    @property
    def final_fn_zero(self) -> bool:
        return all(
            result.ground_truth_collision is not True or result.predicted_collision
            for result in self.query_results
        )

    @property
    def queue_conserved(self) -> bool:
        return len(self.candidates) == len(self.work_items) == len(self.certificates)


def _load_cpp_module() -> Any | None:
    try:
        return importlib.import_module("p2cccd_cpp")
    except ImportError:
        return None


def is_cpp_certificate_engine_available() -> bool:
    module = _load_cpp_module()
    if module is None:
        return False
    return any(callable(getattr(module, name, None)) for name in CPP_CERTIFICATE_ENTRYPOINTS)


def _validate_config(config: CertificateEngineWrapperConfig) -> CertificateEngineWrapperConfig:
    if config.first_work_item_id <= 0:
        raise ValueError("CertificateEngineWrapperConfig.first_work_item_id must be positive")
    if config.first_event_id <= 0:
        raise ValueError("CertificateEngineWrapperConfig.first_event_id must be positive")
    if config.first_timestamp_us <= 0:
        raise ValueError("CertificateEngineWrapperConfig.first_timestamp_us must be positive")
    if config.prefer_cpp_backend and not config.allow_python_fallback and not is_cpp_certificate_engine_available():
        raise RuntimeError("C++ certificate engine binding is unavailable and Python fallback is disabled")
    return config


def _fallback_reason(config: CertificateEngineWrapperConfig) -> str:
    if config.prefer_cpp_backend:
        return "p2cccd_cpp certificate engine entrypoint is unavailable; used Python CPU fallback"
    return "Python CPU certificate engine fallback"


def _rt_config(config: CertificateEngineWrapperConfig) -> RTExactConfig:
    return RTExactConfig(
        exact=config.exact,
        first_work_item_id=config.first_work_item_id,
        first_event_id=config.first_event_id,
        first_timestamp_us=config.first_timestamp_us,
    )


def _work_items(
    candidate_result: CandidateGenerationWrapperResult,
    config: CertificateEngineWrapperConfig,
) -> tuple[ExactWorkItem, ...]:
    return tuple(
        validate_exact_work_item(item)
        for item in schedule_exact_work_items_without_stpf(
            candidate_result.candidates,
            family_by_runtime_query_id=candidate_result.family_by_runtime_query_id,
            first_work_item_id=config.first_work_item_id,
        )
    )


def _checked_result(
    *,
    query_results: tuple[PureExactQueryResult, ...],
    candidate_result: CandidateGenerationWrapperResult,
    work_items: tuple[ExactWorkItem, ...],
    certificates: tuple[CertificateResult, ...],
    audit_log: tuple[AuditLogRow, ...],
    exact_ms: float,
    config: CertificateEngineWrapperConfig,
) -> CertificateEngineWrapperResult:
    checked_certificates = tuple(validate_certificate_result(certificate) for certificate in certificates)
    checked_audit = tuple(validate_audit_log_row(row) for row in audit_log)
    validate_rt_exact_coverage(candidate_result.candidates, work_items, checked_certificates)
    return CertificateEngineWrapperResult(
        query_results=query_results,
        candidates=candidate_result.candidates,
        work_items=work_items,
        certificates=checked_certificates,
        audit_log=checked_audit,
        exact_ms=exact_ms,
        source_name=candidate_result.source_name,
        scene_name=candidate_result.scene_name,
        batch_id=candidate_result.batch_id,
        used_cpp_backend=False,
        fallback_reason=_fallback_reason(config),
    )


def execute_certificate_engine_for_internal_samples(
    samples: Sequence[MotionDiscPairSample],
    candidate_result: CandidateGenerationWrapperResult | None = None,
    config: CertificateEngineWrapperConfig | None = None,
    *,
    candidate_config: CandidateGenerationWrapperConfig | None = None,
    backend: BroadPhaseBackend | None = None,
) -> CertificateEngineWrapperResult:
    if not samples:
        raise ValueError("certificate engine wrapper requires at least one internal sample")
    cfg = _validate_config(config or CertificateEngineWrapperConfig())
    candidates = candidate_result or generate_candidates_for_internal_samples(
        samples,
        candidate_config,
        backend=backend,
    )
    work_items = _work_items(candidates, cfg)
    query_results, certificates, audit_log, exact_ms = _process_internal_exact_work_queue(
        samples,
        candidates.candidates,
        work_items,
        candidates.runtime_query_ids,
        _rt_config(cfg),
    )
    return _checked_result(
        query_results=query_results,
        candidate_result=candidates,
        work_items=work_items,
        certificates=certificates,
        audit_log=audit_log,
        exact_ms=exact_ms,
        config=cfg,
    )


def execute_certificate_engine_for_generated_dataset(
    dataset: GeneratedDataset,
    candidate_result: CandidateGenerationWrapperResult | None = None,
    config: CertificateEngineWrapperConfig | None = None,
    *,
    candidate_config: CandidateGenerationWrapperConfig | None = None,
    backend: BroadPhaseBackend | None = None,
) -> CertificateEngineWrapperResult:
    candidates = candidate_result or generate_candidates_for_generated_dataset(
        dataset,
        candidate_config,
        backend=backend,
    )
    return execute_certificate_engine_for_internal_samples(
        dataset.samples,
        candidates,
        config,
        candidate_config=candidate_config,
        backend=backend,
    )


def execute_certificate_engine_for_external_batch(
    batch: DatasetQueryBatch,
    candidate_result: CandidateGenerationWrapperResult | None = None,
    config: CertificateEngineWrapperConfig | None = None,
    *,
    candidate_config: CandidateGenerationWrapperConfig | None = None,
    backend: BroadPhaseBackend | None = None,
) -> CertificateEngineWrapperResult:
    if not batch.queries:
        raise ValueError("certificate engine wrapper requires at least one external query")
    cfg = _validate_config(config or CertificateEngineWrapperConfig())
    candidates = candidate_result or generate_candidates_for_external_batch(
        batch,
        candidate_config,
        backend=backend,
    )
    work_items = _work_items(candidates, cfg)
    query_results, certificates, audit_log, exact_ms = _process_external_exact_work_queue(
        batch,
        candidates.candidates,
        work_items,
        candidates.runtime_query_ids,
        _rt_config(cfg),
    )
    return _checked_result(
        query_results=query_results,
        candidate_result=candidates,
        work_items=work_items,
        certificates=certificates,
        audit_log=audit_log,
        exact_ms=exact_ms,
        config=cfg,
    )
