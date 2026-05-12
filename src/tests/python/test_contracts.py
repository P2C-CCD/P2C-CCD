from __future__ import annotations

import re
import sys
from dataclasses import replace
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from p2cccd import (  # noqa: E402
    AuditLogRow,
    AuditStage,
    BenchmarkRow,
    CandidateRecord,
    CertificateRefinementMode,
    CertificateResult,
    CertificateStatus,
    CONTRACT_SCHEMAS,
    ExactWorkItem,
    ProposalOutput,
    ProposalSource,
    ProxyType,
    ValidationError,
    from_dict,
    load_runtime_config,
    read_jsonl,
    schema_field_names,
    to_dict,
    validate_audit_log_row,
    validate_benchmark_row,
    validate_candidate_record,
    validate_certificate_result,
    validate_dict_schema,
    validate_exact_work_item,
    validate_proposal_output,
    validate_runtime_config,
    write_csv,
    write_jsonl,
)


def _cpp_struct_fields(struct_name: str) -> tuple[str, ...]:
    header = PROJECT_ROOT / "cpp" / "common" / "runtime_contracts.h"
    text = header.read_text(encoding="utf-8")
    match = re.search(rf"struct\s+{struct_name}\s*\{{(?P<body>.*?)\n\}};", text, re.DOTALL)
    assert match is not None, f"{struct_name} not found in C++ runtime_contracts.h"
    fields: list[str] = []
    for raw_line in match.group("body").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("//"):
            continue
        declaration = line.rstrip(";").split("=", 1)[0].split("{", 1)[0].strip()
        if declaration:
            fields.append(declaration.split()[-1])
    return tuple(fields)


def _cpp_struct_schema(struct_name: str) -> tuple[tuple[str, str], ...]:
    type_map = {
        "std::uint32_t": "uint32",
        "std::uint64_t": "uint64",
        "std::uint16_t": "uint16",
        "std::uint8_t": "uint8",
        "std::int64_t": "int64",
        "double": "float64",
        "float": "float32",
        "std::array<float, 4>": "float32[4]",
        "std::array<float, 8>": "float32[8]",
        "ProxyType": "ProxyType:uint8",
        "ProposalSource": "ProposalSource:uint8",
        "CertificateStatus": "CertificateStatus:uint8",
        "CertificateRefinementMode": "CertificateRefinementMode:uint8",
        "AuditStage": "AuditStage:uint8",
    }
    header = PROJECT_ROOT / "cpp" / "common" / "runtime_contracts.h"
    text = header.read_text(encoding="utf-8")
    match = re.search(rf"struct\s+{struct_name}\s*\{{(?P<body>.*?)\n\}};", text, re.DOTALL)
    assert match is not None, f"{struct_name} not found in C++ runtime_contracts.h"
    schema: list[tuple[str, str]] = []
    for raw_line in match.group("body").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("//"):
            continue
        declaration = line.rstrip(";").split("=", 1)[0].split("{", 1)[0].strip()
        match_decl = re.match(r"(?P<type>.+?)\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)$", declaration)
        assert match_decl is not None, declaration
        cpp_type = re.sub(r"\s+", " ", match_decl.group("type"))
        assert cpp_type in type_map, f"unmapped C++ type {cpp_type!r}"
        schema.append((match_decl.group("name"), type_map[cpp_type]))
    return tuple(schema)


@pytest.mark.parametrize(
    "contract_name",
    [
        "CandidateRecord",
        "ProposalOutput",
        "ExactWorkItem",
        "CertificateResult",
        "AuditLogRow",
        "BenchmarkRow",
    ],
)
def test_python_contract_fields_match_cpp_contract_order(contract_name: str) -> None:
    assert schema_field_names(contract_name) == _cpp_struct_fields(contract_name)
    assert CONTRACT_SCHEMAS[contract_name] == _cpp_struct_schema(contract_name)


def test_candidate_schema_and_required_fields() -> None:
    candidate = CandidateRecord(
        candidate_id=1,
        query_id=10,
        proxy_type_a=ProxyType.SWEPT_AABB,
        proxy_type_b=ProxyType.CAPSULE,
        rt_hit_count=1,
        motion_bound=[0.1, 0.2, 0.3, 0.4],
    )
    assert validate_candidate_record(candidate) is candidate

    payload = to_dict(candidate)
    validate_dict_schema(payload, CandidateRecord)

    missing_query = dict(payload)
    missing_query.pop("query_id")
    with pytest.raises(ValidationError):
        validate_dict_schema(missing_query, CandidateRecord)

    with pytest.raises(ValidationError):
        validate_dict_schema(payload, "NotAContract")

    with pytest.raises(ValidationError):
        validate_candidate_record(
            replace(candidate, proxy_type_a=ProxyType.UNKNOWN),
        )

    with pytest.raises(ValidationError):
        validate_candidate_record(
            replace(candidate, proxy_type_a=True),
        )

    with pytest.raises(ValidationError):
        validate_candidate_record(
            replace(candidate, rt_hit_count=0),
        )

    bool_enum_payload = dict(payload)
    bool_enum_payload["proxy_type_a"] = True
    with pytest.raises(ValidationError):
        from_dict(CandidateRecord, bool_enum_payload)

    with pytest.raises(ValidationError):
        validate_candidate_record(
            replace(candidate, candidate_id=2**64),
        )

    with pytest.raises(ValidationError):
        validate_candidate_record(replace(candidate, slab_id=2**32))


def test_proposal_exact_and_certificate_validators() -> None:
    proposal = ProposalOutput(
        candidate_id=1,
        interval_scores=[0.1] * 8,
        family_scores=[0.2] * 8,
        priority_score=0.5,
        cost_score=1.0,
        uncertainty_score=0.05,
    )
    assert validate_proposal_output(proposal) is proposal

    work_item = ExactWorkItem(
        work_item_id=2,
        parent_candidate_id=1,
        query_id=10,
        interval_t0=0.25,
        interval_t1=0.5,
        feature_family_mask=1,
        priority_score=0.4,
        source=ProposalSource.RAW,
    )
    assert validate_exact_work_item(work_item) is work_item

    certificate = CertificateResult(
        work_item_id=2,
        query_id=10,
        status=CertificateStatus.SEPARATION,
        interval_t0=0.25,
        interval_t1=0.5,
        toi_upper=0.5,
        safe_margin_lb=1.0e-3,
        covered_feature_mask=1,
        eps_time=1.0e-4,
        eps_space=1.0e-6,
        next_refinement_mode=CertificateRefinementMode.NONE,
    )
    assert validate_certificate_result(certificate) is certificate

    with pytest.raises(ValidationError):
        validate_exact_work_item(replace(work_item, interval_t0=0.75, interval_t1=0.5))

    with pytest.raises(ValidationError):
        validate_exact_work_item(replace(work_item, source=True))

    with pytest.raises(ValidationError):
        validate_exact_work_item(replace(work_item, depth=2**16))

    with pytest.raises(ValidationError):
        validate_certificate_result(replace(certificate, witness_id_a=-(2**63) - 1))

    with pytest.raises(ValidationError):
        validate_certificate_result(
            replace(
                certificate,
                status=CertificateStatus.UNDECIDED,
                reason_code=0,
                next_refinement_mode=CertificateRefinementMode.BISECT_INTERVAL,
            ),
        )

    with pytest.raises(ValidationError):
        validate_certificate_result(replace(certificate, witness_family=2**8))

    with pytest.raises(ValidationError):
        validate_certificate_result(
            replace(
                certificate,
                status=CertificateStatus.UNDECIDED,
                reason_code=1,
                next_refinement_mode=CertificateRefinementMode.NONE,
            ),
        )


def test_audit_and_benchmark_serialization_are_stable(tmp_path: Path) -> None:
    audit = AuditLogRow(
        event_id=1,
        query_id=10,
        candidate_id=1,
        work_item_id=2,
        stage=AuditStage.RT,
        interval_t0=0.0,
        interval_t1=1.0,
        timestamp_us=123,
    )
    assert validate_audit_log_row(audit) is audit

    with pytest.raises(ValidationError):
        validate_audit_log_row(replace(audit, action=2**16))

    jsonl_path = tmp_path / "nested" / "audit.jsonl"
    write_jsonl(jsonl_path, [audit])
    loaded = list(read_jsonl(jsonl_path, AuditLogRow))
    assert loaded == [audit]
    assert jsonl_path.read_text(encoding="utf-8").startswith('{"event_id":1,')

    benchmark = BenchmarkRow(
        query_count=100,
        fn_count=0,
        fp_count=2,
        candidate_recall=1.0,
        avg_candidates=4.0,
        avg_exact_evals=2.0,
        avg_subdivision_depth=1.0,
        fallback_ratio=0.1,
        rt_ms=1.0,
        proposal_ms=0.5,
        exact_ms=2.0,
        total_ms=3.5,
        qps=1000.0,
    )
    assert validate_benchmark_row(benchmark) is benchmark

    csv_path = tmp_path / "nested" / "benchmark.csv"
    write_csv(csv_path, [benchmark])
    header = csv_path.read_text(encoding="utf-8").splitlines()[0]
    assert header == ",".join(schema_field_names(BenchmarkRow))

    nested_jsonl_path = tmp_path / "nested" / "logs" / "audit.jsonl"
    nested_csv_path = tmp_path / "nested" / "bench" / "benchmark.csv"
    write_jsonl(nested_jsonl_path, [audit])
    write_csv(nested_csv_path, [benchmark])
    assert nested_jsonl_path.exists()
    assert nested_csv_path.exists()


def test_default_runtime_config_validates() -> None:
    config = load_runtime_config(PROJECT_ROOT / "configs" / "default_runtime.json")
    assert config.epsilon.eps_time > 0.0
    assert config.runtime.max_interval_bins == 8
    assert validate_runtime_config(config) == config

    bad = replace(config, epsilon=replace(config.epsilon, eps_time=0.0))
    with pytest.raises(ValueError):
        validate_runtime_config(bad)
