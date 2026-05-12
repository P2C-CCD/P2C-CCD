from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from p2cccd.bench import (  # noqa: E402
    CASE_INLINE_SURROGATE_SCORING,
    CASE_INLINE_TINY_LOGIC,
    CASE_PURE_CANDIDATE_WRITES,
    CASE_QUEUE_DECOUPLED_BATCH_PROPOSAL,
    NoQueueDecoupleConfig,
    run_no_queue_decouple_microbenchmark,
    validate_no_queue_decouple_config,
)
from p2cccd.proposal import PROPOSAL_FEATURE_DIM, STPFConfig, STPFModel  # noqa: E402


EXPECTED_CASES = (
    CASE_PURE_CANDIDATE_WRITES,
    CASE_INLINE_TINY_LOGIC,
    CASE_INLINE_SURROGATE_SCORING,
    CASE_QUEUE_DECOUPLED_BATCH_PROPOSAL,
)


def test_no_queue_decouple_runs_all_microbenchmark_cases() -> None:
    config = NoQueueDecoupleConfig(
        candidate_count=64,
        repeat_count=2,
        warmup_count=0,
        batch_size=16,
        seed=7,
    )

    result = run_no_queue_decouple_microbenchmark(config)

    assert result.config == config
    assert result.case_names == EXPECTED_CASES
    for case in result.case_results:
        assert case.candidate_count == 64
        assert case.repeat_count == 2
        assert case.elapsed_ms >= 0.0
        assert case.trace_ms >= 0.0
        assert case.proposal_enqueue_dequeue_ms >= 0.0
        assert case.total_tail_latency_ms >= 0.0
        assert case.candidates_per_sec >= 0.0
        assert case.approx_bytes_written > 0
        assert case.approx_bandwidth_mb_s >= 0.0
        assert case.candidate_write_count == 128
        assert case.checksum > 0

    queue = result.result_for(CASE_QUEUE_DECOUPLED_BATCH_PROPOSAL)
    assert queue.trace_ms == 0.0
    assert queue.proposal_enqueue_dequeue_ms >= 0.0
    assert queue.proposal_row_count == 128
    assert queue.proposal_output_count == 128
    assert result.result_for(CASE_PURE_CANDIDATE_WRITES).proposal_output_count == 0


def test_no_queue_decouple_checksums_are_deterministic_for_same_config() -> None:
    config = NoQueueDecoupleConfig(
        candidate_count=32,
        repeat_count=2,
        warmup_count=0,
        batch_size=8,
        seed=11,
    )

    lhs = run_no_queue_decouple_microbenchmark(config)
    rhs = run_no_queue_decouple_microbenchmark(config)

    assert [case.checksum for case in lhs.case_results] == [
        case.checksum for case in rhs.case_results
    ]


def test_no_queue_decouple_queue_case_can_use_real_stpf_model() -> None:
    config = NoQueueDecoupleConfig(
        candidate_count=8,
        repeat_count=1,
        warmup_count=0,
        batch_size=4,
        seed=13,
        use_stpf_model=True,
    )
    model = STPFModel(STPFConfig(feature_dim=PROPOSAL_FEATURE_DIM, hidden_dim=8, num_layers=1))

    result = run_no_queue_decouple_microbenchmark(config, model=model, device="cpu")

    queue = result.result_for(CASE_QUEUE_DECOUPLED_BATCH_PROPOSAL)
    assert queue.candidate_write_count == 8
    assert queue.proposal_row_count == 8
    assert queue.proposal_output_count == 8
    assert queue.checksum > 0


def test_no_queue_decouple_requires_model_when_stpf_enabled() -> None:
    config = NoQueueDecoupleConfig(
        candidate_count=4,
        repeat_count=1,
        warmup_count=0,
        use_stpf_model=True,
    )

    try:
        run_no_queue_decouple_microbenchmark(config)
    except ValueError as exc:
        assert "model" in str(exc)
    else:
        raise AssertionError("expected missing model validation error")


def test_no_queue_decouple_config_validation_rejects_invalid_values() -> None:
    invalid_configs = (
        NoQueueDecoupleConfig(candidate_count=0),
        NoQueueDecoupleConfig(repeat_count=0),
        NoQueueDecoupleConfig(warmup_count=-1),
        NoQueueDecoupleConfig(batch_size=0),
        NoQueueDecoupleConfig(ood_abs_feature_threshold=0.0),
    )

    for config in invalid_configs:
        try:
            validate_no_queue_decouple_config(config)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected validation error for {config}")
