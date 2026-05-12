from __future__ import annotations

from dataclasses import dataclass
import math
import time

from p2cccd.contracts import CandidateRecord, ProxyType
from p2cccd.proposal.inference import (
    DEFAULT_OOD_ABS_FEATURE_THRESHOLD,
    batched_stpf_inference,
    dummy_proposal_policy,
)

from .rt_exact import (
    FEATURE_FAMILY_CONSERVATIVE,
    RtCandidateStats,
    RtCandidateTiming,
)
from .rt_stpf_exact import proposal_feature_rows_from_rt_candidates


CASE_PURE_CANDIDATE_WRITES = "pure_candidate_writes"
CASE_INLINE_TINY_LOGIC = "inline_tiny_logic"
CASE_INLINE_SURROGATE_SCORING = "inline_surrogate_scoring"
CASE_QUEUE_DECOUPLED_BATCH_PROPOSAL = "queue_decoupled_batch_proposal"

_CANDIDATE_WRITE_BYTES = 64
_TINY_LOGIC_BYTES = 72
_SURROGATE_SCORING_BYTES = 96
_QUEUE_PROPOSAL_BYTES = 256


@dataclass(frozen=True, slots=True)
class NoQueueDecoupleConfig:
    candidate_count: int = 4096
    repeat_count: int = 5
    warmup_count: int = 1
    batch_size: int = 1024
    seed: int = 17
    use_stpf_model: bool = False
    ood_abs_feature_threshold: float = DEFAULT_OOD_ABS_FEATURE_THRESHOLD


@dataclass(frozen=True, slots=True)
class NoQueueDecoupleCaseResult:
    case_name: str
    candidate_count: int
    repeat_count: int
    elapsed_ms: float
    trace_ms: float
    proposal_enqueue_dequeue_ms: float
    total_tail_latency_ms: float
    candidates_per_sec: float
    approx_bytes_written: int
    approx_bandwidth_mb_s: float
    candidate_write_count: int
    proposal_row_count: int
    proposal_output_count: int
    checksum: int


@dataclass(frozen=True, slots=True)
class NoQueueDecoupleResult:
    config: NoQueueDecoupleConfig
    case_results: tuple[NoQueueDecoupleCaseResult, ...]

    @property
    def case_names(self) -> tuple[str, ...]:
        return tuple(result.case_name for result in self.case_results)

    def result_for(self, case_name: str) -> NoQueueDecoupleCaseResult:
        for result in self.case_results:
            if result.case_name == case_name:
                return result
        raise KeyError(f"unknown NoQueueDecouple case: {case_name}")


def validate_no_queue_decouple_config(config: NoQueueDecoupleConfig) -> NoQueueDecoupleConfig:
    if config.candidate_count <= 0:
        raise ValueError("NoQueueDecoupleConfig.candidate_count must be positive")
    if config.repeat_count <= 0:
        raise ValueError("NoQueueDecoupleConfig.repeat_count must be positive")
    if config.warmup_count < 0:
        raise ValueError("NoQueueDecoupleConfig.warmup_count must be non-negative")
    if config.batch_size <= 0:
        raise ValueError("NoQueueDecoupleConfig.batch_size must be positive")
    if (
        not math.isfinite(config.ood_abs_feature_threshold)
        or config.ood_abs_feature_threshold <= 0.0
    ):
        raise ValueError("NoQueueDecoupleConfig.ood_abs_feature_threshold must be finite and positive")
    return config


def _candidate_scalar(index: int, seed: int) -> int:
    return (index * 1_103_515_245 + seed * 12_345 + 97) & 0x7FFF_FFFF


def _candidate_tuple(index: int, seed: int) -> tuple[int, int, int, int, int, int, int, int, float, float, float, float]:
    value = _candidate_scalar(index, seed)
    candidate_id = index + 1
    query_id = index + 1
    slab_id = value & 3
    object_a_id = 1 + (value & 15)
    object_b_id = 1 + ((value >> 4) & 15)
    patch_a_id = (value >> 8) & 255
    patch_b_id = (value >> 16) & 255
    rt_hit_count = 1 + ((value >> 5) & 7)
    motion0 = float(value & 31) / 31.0
    motion1 = float((value >> 5) & 31) / 31.0
    motion2 = float((value >> 10) & 31) / 31.0
    motion3 = float((value >> 15) & 31) / 31.0
    return (
        candidate_id,
        query_id,
        slab_id,
        object_a_id,
        patch_a_id,
        object_b_id,
        patch_b_id,
        rt_hit_count,
        motion0,
        motion1,
        motion2,
        motion3,
    )


def _candidate_record(index: int, seed: int) -> CandidateRecord:
    (
        candidate_id,
        query_id,
        slab_id,
        object_a_id,
        patch_a_id,
        object_b_id,
        patch_b_id,
        rt_hit_count,
        motion0,
        motion1,
        motion2,
        motion3,
    ) = _candidate_tuple(index, seed)
    return CandidateRecord(
        candidate_id=candidate_id,
        query_id=query_id,
        slab_id=slab_id,
        object_a_id=object_a_id,
        patch_a_id=patch_a_id,
        object_b_id=object_b_id,
        patch_b_id=patch_b_id,
        proxy_type_a=ProxyType.SWEPT_AABB,
        proxy_type_b=ProxyType.SWEPT_AABB,
        rt_hit_count=rt_hit_count,
        motion_bound=[motion0, motion1, motion2, motion3],
        flags=1,
    )


def _case_result(
    *,
    case_name: str,
    config: NoQueueDecoupleConfig,
    elapsed_ms: float,
    approx_bytes_per_candidate: int,
    candidate_write_count: int,
    proposal_row_count: int,
    proposal_output_count: int,
    checksum: int,
    trace_ms: float | None = None,
    proposal_enqueue_dequeue_ms: float = 0.0,
    total_tail_latency_ms: float | None = None,
) -> NoQueueDecoupleCaseResult:
    total_candidates = config.candidate_count * config.repeat_count
    elapsed_seconds = elapsed_ms / 1000.0
    approx_bytes = approx_bytes_per_candidate * total_candidates
    candidates_per_sec = 0.0 if elapsed_seconds <= 0.0 else total_candidates / elapsed_seconds
    bandwidth = 0.0 if elapsed_seconds <= 0.0 else (approx_bytes / (1024.0 * 1024.0)) / elapsed_seconds
    return NoQueueDecoupleCaseResult(
        case_name=case_name,
        candidate_count=config.candidate_count,
        repeat_count=config.repeat_count,
        elapsed_ms=elapsed_ms,
        trace_ms=elapsed_ms if trace_ms is None else float(trace_ms),
        proposal_enqueue_dequeue_ms=float(proposal_enqueue_dequeue_ms),
        total_tail_latency_ms=elapsed_ms if total_tail_latency_ms is None else float(total_tail_latency_ms),
        candidates_per_sec=candidates_per_sec,
        approx_bytes_written=approx_bytes,
        approx_bandwidth_mb_s=bandwidth,
        candidate_write_count=candidate_write_count,
        proposal_row_count=proposal_row_count,
        proposal_output_count=proposal_output_count,
        checksum=checksum & 0x7FFF_FFFF_FFFF_FFFF,
    )


def _run_timed(config: NoQueueDecoupleConfig, fn) -> tuple[float, tuple[int, int, int, int]]:
    for _ in range(config.warmup_count):
        fn(config)
    start = time.perf_counter()
    counters = fn(config)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return elapsed_ms, counters


def _pure_candidate_writes(config: NoQueueDecoupleConfig) -> tuple[int, int, int, int]:
    candidate_write_count = 0
    checksum = 0
    for repeat in range(config.repeat_count):
        buffer: list[tuple[int, int, int, int, int, int, int, int, float, float, float, float]] = []
        for index in range(config.candidate_count):
            candidate = _candidate_tuple(index, config.seed + repeat)
            buffer.append(candidate)
            candidate_write_count += 1
            checksum += candidate[0] * 31 + candidate[7] * 17 + int(candidate[8] * 1000.0)
        checksum += len(buffer)
    return candidate_write_count, 0, 0, checksum


def _inline_tiny_logic(config: NoQueueDecoupleConfig) -> tuple[int, int, int, int]:
    candidate_write_count = 0
    checksum = 0
    accepted = 0
    for repeat in range(config.repeat_count):
        buffer: list[tuple[int, int, bool, int]] = []
        for index in range(config.candidate_count):
            candidate = _candidate_tuple(index, config.seed + repeat)
            tiny_keep = (candidate[7] >= 4) or ((candidate[2] & 1) == 0)
            tiny_bucket = 1 if tiny_keep else 0
            buffer.append((candidate[0], candidate[1], tiny_keep, tiny_bucket))
            candidate_write_count += 1
            accepted += 1 if tiny_keep else 0
            checksum += candidate[0] * 31 + tiny_bucket * 97 + accepted
        checksum += len(buffer)
    return candidate_write_count, 0, 0, checksum


def _inline_surrogate_scoring(config: NoQueueDecoupleConfig) -> tuple[int, int, int, int]:
    candidate_write_count = 0
    checksum = 0
    for repeat in range(config.repeat_count):
        buffer: list[tuple[int, int, float, float, int]] = []
        for index in range(config.candidate_count):
            candidate = _candidate_tuple(index, config.seed + repeat)
            motion_sum = candidate[8] + candidate[9] + candidate[10] + candidate[11]
            priority = 0.125 * float(candidate[7]) + 0.25 * motion_sum
            cost = 1.0 + float(candidate[7]) * (1.0 + candidate[8])
            family_mask = 1 if candidate[7] % 2 else 2
            buffer.append((candidate[0], candidate[1], priority, cost, family_mask))
            candidate_write_count += 1
            checksum += candidate[0] * 31 + int(priority * 10_000.0) + int(cost * 1000.0) + family_mask
        checksum += len(buffer)
    return candidate_write_count, 0, 0, checksum


def _queue_decoupled_batch_proposal(
    config: NoQueueDecoupleConfig,
    *,
    model=None,
    device: str | None = None,
) -> tuple[int, int, int, int]:
    if config.use_stpf_model and model is None:
        raise ValueError("NoQueueDecouple requires a model when use_stpf_model is True")

    candidate_write_count = 0
    proposal_row_count = 0
    proposal_output_count = 0
    checksum = 0
    for repeat in range(config.repeat_count):
        candidates = tuple(
            _candidate_record(index, config.seed + repeat)
            for index in range(config.candidate_count)
        )
        candidate_write_count += len(candidates)
        family_by_query_id = {
            candidate.query_id: FEATURE_FAMILY_CONSERVATIVE
            for candidate in candidates
        }
        stats = RtCandidateStats(
            backend_name="no_queue_decouple_microbench",
            primitive_count=2 * len(candidates),
            raw_hit_count=len(candidates),
            compact_candidate_count=len(candidates),
            candidate_recall=1.0,
            timing=RtCandidateTiming(),
        )
        rows = proposal_feature_rows_from_rt_candidates(
            candidates,
            family_by_runtime_query_id=family_by_query_id,
            candidate_stats=stats,
        )
        proposal_row_count += len(rows)
        predictions = []
        for start in range(0, len(rows), config.batch_size):
            batch_rows = rows[start : start + config.batch_size]
            if config.use_stpf_model:
                predictions.extend(
                    batched_stpf_inference(
                        model,
                        batch_rows,
                        batch_size=config.batch_size,
                        device=device,
                        ood_abs_feature_threshold=config.ood_abs_feature_threshold,
                    )
                )
            else:
                predictions.extend(
                    dummy_proposal_policy(
                        batch_rows,
                        ood_abs_feature_threshold=config.ood_abs_feature_threshold,
                    )
                )
        proposal_output_count += len(predictions)
        for prediction in predictions:
            checksum += (
                prediction.candidate_id * 31
                + int(prediction.priority_score * 10_000.0)
                + int(prediction.cost_score * 1000.0)
                + int(prediction.uncertainty_score * 1000.0)
            )
        checksum += len(candidates) + len(rows) + len(predictions)
    return candidate_write_count, proposal_row_count, proposal_output_count, checksum


def run_no_queue_decouple_microbenchmark(
    config: NoQueueDecoupleConfig | None = None,
    *,
    model=None,
    device: str | None = None,
) -> NoQueueDecoupleResult:
    cfg = validate_no_queue_decouple_config(config or NoQueueDecoupleConfig())

    elapsed_ms, counters = _run_timed(cfg, _pure_candidate_writes)
    pure = _case_result(
        case_name=CASE_PURE_CANDIDATE_WRITES,
        config=cfg,
        elapsed_ms=elapsed_ms,
        approx_bytes_per_candidate=_CANDIDATE_WRITE_BYTES,
        candidate_write_count=counters[0],
        proposal_row_count=counters[1],
        proposal_output_count=counters[2],
        checksum=counters[3],
    )

    elapsed_ms, counters = _run_timed(cfg, _inline_tiny_logic)
    tiny = _case_result(
        case_name=CASE_INLINE_TINY_LOGIC,
        config=cfg,
        elapsed_ms=elapsed_ms,
        approx_bytes_per_candidate=_TINY_LOGIC_BYTES,
        candidate_write_count=counters[0],
        proposal_row_count=counters[1],
        proposal_output_count=counters[2],
        checksum=counters[3],
    )

    elapsed_ms, counters = _run_timed(cfg, _inline_surrogate_scoring)
    surrogate = _case_result(
        case_name=CASE_INLINE_SURROGATE_SCORING,
        config=cfg,
        elapsed_ms=elapsed_ms,
        approx_bytes_per_candidate=_SURROGATE_SCORING_BYTES,
        candidate_write_count=counters[0],
        proposal_row_count=counters[1],
        proposal_output_count=counters[2],
        checksum=counters[3],
    )

    def queue_case(case_config: NoQueueDecoupleConfig) -> tuple[int, int, int, int]:
        return _queue_decoupled_batch_proposal(case_config, model=model, device=device)

    elapsed_ms, counters = _run_timed(cfg, queue_case)
    queue = _case_result(
        case_name=CASE_QUEUE_DECOUPLED_BATCH_PROPOSAL,
        config=cfg,
        elapsed_ms=elapsed_ms,
        approx_bytes_per_candidate=_QUEUE_PROPOSAL_BYTES,
        candidate_write_count=counters[0],
        proposal_row_count=counters[1],
        proposal_output_count=counters[2],
        checksum=counters[3],
        trace_ms=0.0,
        proposal_enqueue_dequeue_ms=elapsed_ms,
    )

    return NoQueueDecoupleResult(
        config=cfg,
        case_results=(pure, tiny, surrogate, queue),
    )
