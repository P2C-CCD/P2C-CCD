# Exact Local Certificate Engine

The current CPU baseline implements the first exact-certificate path for
linearly moving local features.

## Implemented Scope

- Point-triangle interval oracle.
- Edge-edge interval oracle.
- Recursive interval subdivision on CPU.
- Collision certificate output with witness family, witness ids, and
  conservative `toi_upper`.
- Separation certificate output with `covered_feature_mask` and
  `safe_margin_lb`.
- Undecided output with reason codes and explicit `next_refinement_mode`.
- CPU exact work queue processing with audit log emission.
- Queue coverage guard: every queued work item must emit exactly one
  certificate.
- Conservative refinement work item generation for TOI tightening and interval
  bisection.
- Optional CUDA batched point-triangle and edge-edge kernels behind
  `P2CCCD_ENABLE_CUDA`.
- CPU-vs-CUDA cross-check under caller-provided `eps_cert`.

The engine consumes an `ExactCertificateQuery`, which combines the existing
runtime `ExactWorkItem` with explicit local feature trajectories. The old
`Evaluate(ExactWorkItem)` overload is kept for compatibility and still returns
an undecided result when no geometry is provided.

## Conservative Interval Logic

For each interval, the CPU oracle:

1. Samples the start, midpoint, and end times. If any sampled feature distance is
   within `eps_space`, it emits a collision certificate.
2. Computes a conservative separation lower bound from midpoint distance minus
   the maximum feature motion radius over the interval.
3. Emits a separation certificate when that lower bound is greater than
   `eps_space`.
4. Otherwise recursively subdivides until a child certifies collision or all
   children certify separation.
5. Emits undecided when `max_subdivision_depth` or `eps_time` is reached without
   a proof.

This is a correctness-oriented CPU baseline. It is not yet the final high-order
root isolation kernel that would be expected for a performance paper, but it
provides a concrete oracle path for queue integration and later CUDA cross-checks.

## Work Queue And Audit

`ProcessExactWorkQueueCpu` consumes a vector of `ExactCertificateQuery` rows and
returns:

- one `CertificateResult` per queued work item,
- dequeue and certify `AuditLogRow` events,
- `processed_count` for coverage validation.

`ValidateExactWorkQueueCoverage` rejects missing certificates, duplicate
certificates, and certificates that do not correspond to queued work items. This
is the illegal-termination guard for the current CPU exact stage.

Undecided certificates use:

- `reason_code = kCertificateReasonMissingGeometry` with
  `next_refinement_mode = kRequestGeometry`,
- `reason_code = kCertificateReasonMaxSubdivisionDepth` with
  `next_refinement_mode = kBisectInterval`,
- `reason_code = kCertificateReasonInvalidInput` with
  `next_refinement_mode = kRequestGeometry`.

`GenerateConservativeRefinementWorkItems` provides two baseline heuristics:

- collision certificates can generate a narrower `[t0, toi_upper]` work item for
  conservative TOI tightening,
- undecided certificates with `kBisectInterval` generate two child work items
  split at the interval midpoint.

## Witness Encoding

- `witness_family = kFeatureFamilyPointTriangle` for point-triangle collision.
- `witness_family = kFeatureFamilyEdgeEdge` for edge-edge collision.
- `witness_id_a` stores the point id or first edge id.
- `witness_id_b` stores the triangle id or second edge id.

For separation, `covered_feature_mask` records the certified feature families.

## CUDA Baseline

`certificate_cuda.h` exposes optional batched CUDA entry points:

- `EvaluatePointTriangleBatchCuda`,
- `EvaluateEdgeEdgeBatchCuda`,
- `CrossCheckCpuCudaExact`.

The default build uses a stub and does not require CUDA. Configure with:

```powershell
cmake -S src -B src\build_cuda -DP2CCCD_ENABLE_CUDA=ON
```

The CUDA kernels mirror the CPU baseline's sampled collision test and
midpoint-distance motion-bound subdivision, then transfer compact
`PrimitiveIntervalResult` rows back to host memory.

## Tests

`certificate_engine_test` covers:

- point-triangle collision found by subdivision,
- point-triangle separation,
- edge-edge collision found by subdivision,
- edge-edge separation,
- engine-level collision certificate validation,
- engine-level separation certificate validation,
- undecided result for missing geometry,
- CPU exact queue audit and coverage guards,
- conservative refinement generation,
- compatibility of the legacy placeholder overload,
- CUDA stub behavior in CPU-only builds,
- CPU-vs-CUDA cross-checks when `P2CCCD_ENABLE_CUDA=ON`.
