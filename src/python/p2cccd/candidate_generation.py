from __future__ import annotations

from dataclasses import dataclass
import importlib
from typing import Any, Sequence

from p2cccd.contracts import CandidateRecord
from p2cccd.data.dataset import GeneratedDataset
from p2cccd.data.samplers import MotionDiscPairSample
from p2cccd.datasets.ccd import DatasetQueryBatch
from p2cccd.validators import validate_candidate_record

from .bench.bvh_exact import BroadPhaseBackend
from .bench.rt_exact import (
    FEATURE_FAMILY_CONSERVATIVE,
    RTExactConfig,
    RtCandidateStats,
    _family_mask_for_external,
    _make_external_candidates,
    _make_internal_candidates,
)


CPP_CANDIDATE_ENTRYPOINTS = (
    "generate_candidates_for_internal_samples",
    "generate_candidates_for_external_batch",
    "generate_candidates",
)


@dataclass(frozen=True, slots=True)
class CandidateGenerationWrapperConfig:
    backend_name: str = "cpu_reference_rt"
    same_query_only: bool = True
    prefer_cpp_backend: bool = False
    allow_python_fallback: bool = True


@dataclass(frozen=True, slots=True)
class CandidateGenerationWrapperResult:
    candidates: tuple[CandidateRecord, ...]
    candidate_stats: RtCandidateStats
    runtime_query_ids: dict[int, int]
    family_by_runtime_query_id: dict[int, int]
    source_name: str
    scene_name: str
    batch_id: str
    used_cpp_backend: bool
    fallback_reason: str

    @property
    def candidate_count(self) -> int:
        return len(self.candidates)


def _validate_config(config: CandidateGenerationWrapperConfig) -> CandidateGenerationWrapperConfig:
    if config.backend_name not in {"cpu_reference_rt", "optix_compatible", "optix_rt"}:
        raise ValueError("CandidateGenerationWrapperConfig.backend_name is unsupported")
    if config.prefer_cpp_backend and not config.allow_python_fallback and not is_cpp_candidate_generation_available():
        raise RuntimeError("C++ candidate generation binding is unavailable and Python fallback is disabled")
    return config


def _load_cpp_module() -> Any | None:
    try:
        return importlib.import_module("p2cccd_cpp")
    except ImportError:
        return None


def is_cpp_candidate_generation_available() -> bool:
    module = _load_cpp_module()
    if module is None:
        return False
    return any(callable(getattr(module, name, None)) for name in CPP_CANDIDATE_ENTRYPOINTS)


def _fallback_reason(config: CandidateGenerationWrapperConfig) -> str:
    if config.prefer_cpp_backend:
        return "p2cccd_cpp candidate generation entrypoint is unavailable; used Python CPU fallback"
    return "Python CPU candidate generation backend"


def _rt_config(config: CandidateGenerationWrapperConfig) -> RTExactConfig:
    return RTExactConfig(
        backend_name=config.backend_name,
        same_query_only=config.same_query_only,
    )


def generate_candidates_for_internal_samples(
    samples: Sequence[MotionDiscPairSample],
    config: CandidateGenerationWrapperConfig | None = None,
    *,
    backend: BroadPhaseBackend | None = None,
) -> CandidateGenerationWrapperResult:
    if not samples:
        raise ValueError("candidate generation requires at least one internal sample")
    cfg = _validate_config(config or CandidateGenerationWrapperConfig())
    candidates, stats, runtime_ids = _make_internal_candidates(samples, _rt_config(cfg), backend=backend)
    checked_candidates = tuple(validate_candidate_record(candidate) for candidate in candidates)
    family_by_runtime_query_id = {
        runtime_ids[sample.query_id]: FEATURE_FAMILY_CONSERVATIVE
        for sample in samples
    }
    return CandidateGenerationWrapperResult(
        candidates=checked_candidates,
        candidate_stats=stats,
        runtime_query_ids=dict(runtime_ids),
        family_by_runtime_query_id=family_by_runtime_query_id,
        source_name="internal_analytic_oracle",
        scene_name="programmatic_motion_disc_pairs",
        batch_id="internal_samples",
        used_cpp_backend=False,
        fallback_reason=_fallback_reason(cfg),
    )


def generate_candidates_for_generated_dataset(
    dataset: GeneratedDataset,
    config: CandidateGenerationWrapperConfig | None = None,
    *,
    backend: BroadPhaseBackend | None = None,
) -> CandidateGenerationWrapperResult:
    return generate_candidates_for_internal_samples(dataset.samples, config, backend=backend)


def generate_candidates_for_external_batch(
    batch: DatasetQueryBatch,
    config: CandidateGenerationWrapperConfig | None = None,
    *,
    backend: BroadPhaseBackend | None = None,
) -> CandidateGenerationWrapperResult:
    if not batch.queries:
        raise ValueError("candidate generation requires at least one external query")
    cfg = _validate_config(config or CandidateGenerationWrapperConfig())
    candidates, stats, runtime_ids = _make_external_candidates(batch, _rt_config(cfg), backend=backend)
    checked_candidates = tuple(validate_candidate_record(candidate) for candidate in candidates)
    family_by_runtime_query_id = {
        runtime_ids[query.query_id]: _family_mask_for_external(query)
        for query in batch.queries
    }
    return CandidateGenerationWrapperResult(
        candidates=checked_candidates,
        candidate_stats=stats,
        runtime_query_ids=dict(runtime_ids),
        family_by_runtime_query_id=family_by_runtime_query_id,
        source_name=batch.source_name,
        scene_name=batch.scene_name,
        batch_id=batch.batch_id,
        used_cpp_backend=False,
        fallback_reason=_fallback_reason(cfg),
    )
