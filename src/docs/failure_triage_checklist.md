# Failure Triage Checklist

Use this checklist before changing algorithms. Most failures should first be classified as recall, proposal scheduling, certificate, or export failures.

## Candidate Recall Failure

- Confirm proxy inflation uses the intended `eps_proxy`.
- Check patch granularity and slab count against the failing scene.
- Compare CPU oracle candidates against compacted candidate IDs.
- Verify query IDs survive raw hit emission, compaction, and work-item scheduling.
- Re-run `candidate_recall_test` and `slab_refinement_recall_test` before changing STPF code.

## Proposal Monotonicity Failure

- Check that every raw `CandidateRecord.candidate_id` maps to exactly one `ExactWorkItem.parent_candidate_id`.
- Treat missing, low-confidence, or OOD proposals as fallback work rather than dropped work.
- Verify duplicate work items are rejected unless they are explicit refinement children.
- Inspect `fallback_ratio`, `exact_queue_occupancy`, and proposal audit rows.
- Re-run `proposal_monotonicity_test`, `queue_conservation_test`, and OOD fallback tests.

## Certificate Mismatch

- Check point-triangle and edge-edge family masks before blaming interval subdivision.
- Confirm `toi_upper` lies within the certified interval for collision certificates.
- Confirm separation certificates have nonzero covered feature masks and non-negative safe margins.
- For undecided outputs, verify reason code and next refinement mode are nonzero.
- Re-run CPU exact regression tests and CPU-vs-CUDA exact cross-checks when CUDA is enabled.

## Benchmark Export Failure

- Validate `BenchmarkRunMeta` and every `BenchmarkRowV2` before writing files.
- Check family-wise exact calls sum to `exact_calls_total`.
- Check latency percentile monotonicity: min <= p50 <= p90 <= p95 <= p99 <= max.
- Check CSV header equals `schema_field_names(BenchmarkRowV2)`.
- Re-run `test_benchmark_export_v2.py` and `test_correctness_and_performance_gates.py`.

## Escalation Rule

Do not tune STPF, patch granularity, or proxy families until the same failing case has been reduced to a minimal deterministic query batch and the baseline `PureExactCPU` result is understood.
