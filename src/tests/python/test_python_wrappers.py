from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from p2cccd import (  # noqa: E402
    CandidateGenerationWrapperConfig,
    CertificateEngineWrapperConfig,
    execute_certificate_engine_for_generated_dataset,
    generate_candidates_for_generated_dataset,
    is_cpp_candidate_generation_available,
    is_cpp_certificate_engine_available,
)
from p2cccd.contracts import AuditStage  # noqa: E402
from p2cccd.data import DatasetGenerationConfig, generate_exact_oracle_dataset  # noqa: E402


def _dataset():
    return generate_exact_oracle_dataset(
        DatasetGenerationConfig(mesh_count_per_split=2, robot_link_count=1, seed=909)
    )


def test_candidate_generation_wrapper_runs_with_python_fallback() -> None:
    result = generate_candidates_for_generated_dataset(_dataset())

    assert result.candidate_count > 0
    assert result.candidate_stats.candidate_recall == 1.0
    assert result.source_name == "internal_analytic_oracle"
    assert not result.used_cpp_backend
    assert "Python" in result.fallback_reason
    assert set(result.family_by_runtime_query_id) == set(result.runtime_query_ids.values())


def test_candidate_generation_wrapper_rejects_required_unavailable_cpp_backend() -> None:
    assert not is_cpp_candidate_generation_available()

    with pytest.raises(RuntimeError):
        generate_candidates_for_generated_dataset(
            _dataset(),
            CandidateGenerationWrapperConfig(
                prefer_cpp_backend=True,
                allow_python_fallback=False,
            ),
        )


def test_certificate_engine_wrapper_runs_exact_queue_with_python_fallback() -> None:
    dataset = _dataset()
    candidates = generate_candidates_for_generated_dataset(dataset)

    result = execute_certificate_engine_for_generated_dataset(dataset, candidates)

    assert result.queue_conserved
    assert result.final_fn_zero
    assert len(result.certificates) == len(candidates.candidates)
    assert len(result.work_items) == len(candidates.candidates)
    assert result.exact_ms >= 0.0
    assert not result.used_cpp_backend
    assert any(row.stage is AuditStage.EXACT for row in result.audit_log)


def test_certificate_engine_wrapper_rejects_required_unavailable_cpp_backend() -> None:
    assert not is_cpp_certificate_engine_available()

    with pytest.raises(RuntimeError):
        execute_certificate_engine_for_generated_dataset(
            _dataset(),
            config=CertificateEngineWrapperConfig(
                prefer_cpp_backend=True,
                allow_python_fallback=False,
            ),
        )
