from __future__ import annotations

import csv
import json
from dataclasses import is_dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any, Iterable, Iterator, TypeVar

from .contracts import (
    AuditLogRow,
    AuditStage,
    BenchmarkRow,
    BenchmarkRowV2,
    BenchmarkRunMeta,
    CandidateRecord,
    CertificateRefinementMode,
    CertificateResult,
    CertificateStatus,
    ExactWorkItem,
    ProposalOutput,
    ProposalSource,
    ProxyType,
    schema_field_names,
)
from .validators import ValidationError, validate_contract, validate_dict_schema

T = TypeVar("T")

_ENUM_FIELDS: dict[type, dict[str, type[IntEnum]]] = {
    CandidateRecord: {
        "proxy_type_a": ProxyType,
        "proxy_type_b": ProxyType,
    },
    ExactWorkItem: {
        "source": ProposalSource,
    },
    CertificateResult: {
        "status": CertificateStatus,
        "next_refinement_mode": CertificateRefinementMode,
    },
    AuditLogRow: {
        "stage": AuditStage,
    },
}


def _plain_value(value: Any) -> Any:
    if isinstance(value, IntEnum):
        return int(value)
    if isinstance(value, list):
        return [_plain_value(item) for item in value]
    return value


def to_dict(contract: object, *, validate: bool = True) -> dict[str, Any]:
    if not is_dataclass(contract):
        raise TypeError("contract must be a dataclass instance")
    if validate:
        validate_contract(contract)
    return {name: _plain_value(getattr(contract, name)) for name in schema_field_names(contract)}


def from_dict(contract_type: type[T], data: dict[str, Any], *, validate: bool = True) -> T:
    validate_dict_schema(data, contract_type)
    enum_fields = _ENUM_FIELDS.get(contract_type, {})
    kwargs = dict(data)
    for field_name, enum_type in enum_fields.items():
        if isinstance(kwargs[field_name], bool):
            raise ValidationError(f"{field_name} must be an integer enum value, not bool")
        kwargs[field_name] = enum_type(kwargs[field_name])
    contract = contract_type(**kwargs)
    if validate:
        validate_contract(contract)
    return contract


def to_json(contract: object, *, validate: bool = True) -> str:
    return json.dumps(to_dict(contract, validate=validate), sort_keys=False, separators=(",", ":"))


def from_json(contract_type: type[T], payload: str, *, validate: bool = True) -> T:
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise TypeError("contract JSON payload must contain an object")
    return from_dict(contract_type, data, validate=validate)


def write_jsonl(path: str | Path, rows: Iterable[object], *, validate: bool = True) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(to_json(row, validate=validate))
            handle.write("\n")


def read_jsonl(path: str | Path, contract_type: type[T], *, validate: bool = True) -> Iterator[T]:
    input_path = Path(path)
    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield from_json(contract_type, stripped, validate=validate)
            except Exception as exc:
                raise ValueError(f"failed to parse {input_path}:{line_number}") from exc


def write_csv(path: str | Path, rows: Iterable[object], *, validate: bool = True) -> None:
    output_path = Path(path)
    iterator = iter(rows)
    try:
        first = next(iterator)
    except StopIteration:
        raise ValueError("cannot infer CSV schema from empty rows") from None

    field_names = schema_field_names(first)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=field_names, lineterminator="\n")
        writer.writeheader()
        writer.writerow(to_dict(first, validate=validate))
        for row in iterator:
            if type(row) is not type(first):
                raise TypeError("all CSV rows must have the same contract type")
            writer.writerow(to_dict(row, validate=validate))


def write_audit_jsonl(path: str | Path, rows: Iterable[AuditLogRow]) -> None:
    write_jsonl(path, rows, validate=True)


def write_benchmark_csv(path: str | Path, rows: Iterable[BenchmarkRow]) -> None:
    write_csv(path, rows, validate=True)


def write_benchmark_v2_csv(path: str | Path, rows: Iterable[BenchmarkRowV2]) -> None:
    write_csv(path, rows, validate=True)


def write_benchmark_v2_jsonl(path: str | Path, rows: Iterable[BenchmarkRowV2]) -> None:
    write_jsonl(path, rows, validate=True)


def write_benchmark_run_meta_json(path: str | Path, meta: BenchmarkRunMeta) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(to_json(meta, validate=True) + "\n", encoding="utf-8")
