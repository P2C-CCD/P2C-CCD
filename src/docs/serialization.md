# Stable Runtime Serialization

P2CCCD uses the C++ structs in `cpp/common/runtime_contracts.h` as the source of truth for runtime record layout.

## Contract Version

- `schema_version` is currently `1`.
- `CandidateRecord.schema_version` is mandatory and must match the runtime contract version.
- Any future incompatible field change must increment the schema version and add a migration path.

## JSONL

- Audit logs are written as UTF-8 JSON Lines.
- Each line is one complete `AuditLogRow` object.
- Field order follows `CONTRACT_SCHEMAS["AuditLogRow"]` in `python/p2cccd/contracts.py`.
- Enum values are serialized as unsigned integer values matching the C++ enum values.
- Floating-point values are serialized as JSON numbers and must be finite.

## CSV

- Benchmark rows are written as UTF-8 CSV with LF line endings.
- The header order follows `CONTRACT_SCHEMAS["BenchmarkRow"]`.
- Enum fields are not used in `BenchmarkRow`.
- Every numeric value must pass the runtime validator before it is written.

## Compatibility Rule

The Python regression test parses `cpp/common/runtime_contracts.h` and compares C++ struct field order against `CONTRACT_SCHEMAS`. This is the guardrail that prevents Python logging, training data, and benchmark export code from drifting away from the C++ runtime ABI contract.
