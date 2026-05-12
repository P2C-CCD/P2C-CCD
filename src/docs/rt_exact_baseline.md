# RTExact Baseline

`RTExact` is the first no-STPF end-to-end correctness runner:

```text
conservative RT candidates
  -> direct exact work queue
  -> exact certificates
  -> BenchmarkRow
```

Runtime module:

```text
src/python/p2cccd/bench/rt_exact.py
```

## Current Scope

The default implementation is a Python workbench baseline using conservative swept-AABB overlap logic and `cpu_reference_rt`.

When `p2cccd_cpp` is built with `P2CCCD_ENABLE_OPTIX=ON`, `RTExactConfig(backend_name="optix_rt")` routes candidate generation through the C++ OptiX RT Core backend. That path builds a GAS over conservative proxy AABBs, launches the OptiX candidate device program, emits `RawCandidateHit` records on device, compacts them into `CandidateRecord`, and then reuses the same exact certificate path.

It does not run the STPF proposal model. Every compact candidate is scheduled directly into one `ExactWorkItem` with `ProposalSource.RAW`.

## Inputs

Supported inputs:

- Internal generated `MotionDiscPairSample` / `GeneratedDataset` samples.
- External CCD `DatasetQueryBatch` rows from adapters.
- Scalable CCD sample `vf` and `ee` batches.

External feature-family mapping:

- `vf` -> `FEATURE_FAMILY_POINT_TRIANGLE`
- `ee` -> `FEATURE_FAMILY_EDGE_EDGE`

Internal swept-sphere proxy samples use a conservative point-triangle plus edge-edge family mask until mesh-feature replay is exposed.

## Runtime Contracts

The runner emits validated runtime contracts:

- `CandidateRecord`
- `ExactWorkItem`
- `CertificateResult`
- `AuditLogRow`
- `BenchmarkRow`

Runtime contract query ids are normalized to positive ids because external benchmark query ids can start at zero. Source query ids remain in `PureExactQueryResult`.

## No-STPF Queue Policy

`schedule_exact_work_items_without_stpf()` enforces the policy:

- one exact work item per candidate,
- full interval `[0, 1]`,
- `ProposalSource.RAW`,
- no proposal scores,
- conservative feature-family mask from the candidate/query family.

`validate_rt_exact_coverage()` checks that no candidate disappears before a certificate is emitted.

## Correctness Policy

Candidate generation is conservative: a query is culled only when the two swept feature proxies do not overlap. Culled queries are reported as `rt_candidate_separation`.

Surviving candidates are evaluated by `PureExactCPU`. The default exact config keeps `conservative_undecided_as_collision=True`, so undecided exact intervals are safe for `final FN = 0` but can increase false positives.

## Benchmark Fields

`RTExactResult.benchmark` contains:

- `candidate_recall`: recall of labeled collisions after RT candidate generation,
- `avg_candidates`: compact candidates per input query,
- `rt_ms`: candidate build, trace, and compaction time,
- `proposal_ms = 0`,
- `exact_ms`: exact evaluation and certificate emission time,
- `fn_count` / `fp_count`: final predictions after exact evaluation or conservative undecided handling.

`RTExactResult.queue_conserved` is true when:

```text
candidate_count == work_item_count == certificate_count
```

## Limitations

- `optix_rt` is the real OptiX RT Core smoke backend.
- `optix_compatible` remains the compatibility/fallback name for CPU-only runs.
- The benchmark runner exact stage defaults to `PureExactCPU` because internal
  generated samples use analytic swept-proxy labels and external adapter rows
  use imported primitive query contracts. Low-level C++ CPU exact APIs are
  available through pybind for direct certificate-engine wrappers and regression
  replay.
- External batches are adapter-level primitive CCD queries, not full mesh-scene RT traversal.
