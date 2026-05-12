from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from p2cccd.bench import load_benchmark_suite_config, validate_benchmark_suite_config  # noqa: E402


QUALITY_GATES_PATH = PROJECT_ROOT / "tests" / "quality_gates.json"
CPP_CMAKELISTS_PATH = PROJECT_ROOT / "cpp" / "CMakeLists.txt"


def _load_quality_gates() -> dict[str, Any]:
    payload = json.loads(QUALITY_GATES_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _split_python_test_ref(ref: str) -> tuple[Path, str | None]:
    path_text, _, test_name = ref.partition("::")
    return PROJECT_ROOT / path_text, test_name or None


def test_quality_gate_manifest_covers_testing_todo_items() -> None:
    payload = _load_quality_gates()
    gates = payload["gates"]

    assert payload["schema_version"] == 1
    assert [gate["todo_id"] for gate in gates] == list(range(107, 117))
    assert {gate["name"] for gate in gates} == {
        "tests_readme",
        "contract_validation",
        "proxy_coverage",
        "candidate_recall",
        "proposal_monotonicity",
        "cpu_exact_certificates",
        "cpu_cuda_exact_consistency",
        "final_fn_zero",
        "performance_timing_export",
        "ci_minimal_cpu",
    }
    assert all(gate["status"] == "done" for gate in gates)
    for gate in gates:
        assert gate["purpose"]
        artifact_count = (
            len(gate["docs"])
            + len(gate["python_tests"])
            + len(gate["cpp_tests"])
            + len(gate["benchmark_suites"])
        )
        assert artifact_count > 0, gate["name"]


def test_quality_gate_manifest_artifacts_exist_and_are_valid() -> None:
    payload = _load_quality_gates()
    cmake_text = CPP_CMAKELISTS_PATH.read_text(encoding="utf-8")

    for gate in payload["gates"]:
        for doc_ref in gate["docs"]:
            assert (PROJECT_ROOT / doc_ref).exists(), f"{gate['name']} missing doc {doc_ref}"

        for test_ref in gate["python_tests"]:
            test_path, test_name = _split_python_test_ref(test_ref)
            assert test_path.exists(), f"{gate['name']} missing Python test {test_ref}"
            if test_name is not None:
                test_text = test_path.read_text(encoding="utf-8")
                assert f"def {test_name}(" in test_text, f"{gate['name']} missing test function {test_ref}"

        for cpp_test_name in gate["cpp_tests"]:
            assert f"add_test(NAME {cpp_test_name} " in cmake_text, f"{gate['name']} missing CTest {cpp_test_name}"

        for suite_ref in gate["benchmark_suites"]:
            suite_path = PROJECT_ROOT / suite_ref
            assert suite_path.exists(), f"{gate['name']} missing suite {suite_ref}"
            suite = load_benchmark_suite_config(suite_path)
            assert validate_benchmark_suite_config(suite) is suite


def test_quality_gate_manifest_reports_completed_release_status() -> None:
    payload = _load_quality_gates()

    for gate in payload["gates"]:
        assert gate["status"] == "done", f"{gate['name']} is not release-ready"


def test_ci_minimal_gate_has_no_accelerator_runtime_dependency() -> None:
    payload = _load_quality_gates()
    ci_gate = next(gate for gate in payload["gates"] if gate["name"] == "ci_minimal_cpu")

    assert ci_gate["requires_cuda_runtime"] is False
    assert ci_gate["requires_optix_runtime"] is False

    suite = load_benchmark_suite_config(PROJECT_ROOT / ci_gate["benchmark_suites"][0])
    serialized_cases = json.dumps([case.config for case in suite.cases], sort_keys=True).lower()
    assert "cuda" not in serialized_cases
    assert "optix" not in serialized_cases
