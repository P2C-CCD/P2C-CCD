from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from p2cccd.bench import (  # noqa: E402
    NEAR_TERM_EXECUTION_SCHEMA_VERSION,
    NearTermExecutionConfig,
    run_near_term_execution_gate,
    write_near_term_execution_gate_json,
)


def test_near_term_execution_runs_rtexact_then_stpf_v1_gate(tmp_path: Path) -> None:
    result = run_near_term_execution_gate(
        NearTermExecutionConfig(
            dataset_seed=127,
            mesh_count_per_split=1,
            robot_link_count=1,
            stpf_hidden_dim=16,
            stpf_num_layers=1,
            proposal_batch_size=2,
        )
    )
    output_path = write_near_term_execution_gate_json(tmp_path / "near_term_execution_gate.json", result)
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert result.schema_version == NEAR_TERM_EXECUTION_SCHEMA_VERSION
    assert result.dataset_query_count > 0
    assert result.rt_exact_passed
    assert result.stpf_v1_ran
    assert result.stpf_v1_passed
    assert result.rt_exact.method_name == "RTExact"
    assert result.rt_exact.final_fn_zero
    assert result.rt_exact.candidate_recall == 1.0
    assert result.rt_exact.queue_conserved
    assert result.rt_exact.proposal_ms == 0.0
    assert result.rt_exact.candidate_count == result.rt_exact.work_item_count == result.rt_exact.certificate_count
    assert result.stpf_v1 is not None
    assert result.stpf_v1.method_name == "RTSTPFExact:STPFv1"
    assert result.stpf_v1.final_fn_zero
    assert result.stpf_v1.candidate_recall == 1.0
    assert result.stpf_v1.queue_conserved
    assert result.stpf_v1.monotonic_safe
    assert result.stpf_v1.proposal_ms >= 0.0
    assert result.stpf_v1.candidate_count == result.stpf_v1.work_item_count == result.stpf_v1.certificate_count
    assert payload["rt_exact_passed"] is True
    assert payload["stpf_v1_passed"] is True


def test_near_term_execution_config_can_disable_robot_links() -> None:
    result = run_near_term_execution_gate(
        NearTermExecutionConfig(
            dataset_seed=129,
            mesh_count_per_split=1,
            robot_link_count=0,
            include_robot_links=False,
            stpf_hidden_dim=8,
            stpf_num_layers=1,
            proposal_batch_size=1,
        )
    )

    assert result.dataset_query_count == 5
    assert result.rt_exact_passed
    assert result.stpf_v1_passed
