# NoQueueDecouple Microbenchmark

`NoQueueDecouple` is a microbenchmark for measuring why P2CCCD keeps proposal
work queue-decoupled from the RT candidate write path.

## Compared Cases

The runner compares four paths over the same deterministic candidate stream:

- `pure_candidate_writes`: writes candidate-like records only.
- `inline_tiny_logic`: writes candidates plus a tiny branch and bucket decision.
- `inline_surrogate_scoring`: writes candidates plus arithmetic priority, cost,
  and family surrogate scores inline.
- `queue_decoupled_batch_proposal`: writes compact candidates, then performs a
  separate `CandidateRecord -> ProposalFeatureRow -> ProposalPrediction` batch
  proposal pass.

This is not a paper-grade GPU timing replacement. It is a CI-friendly
workbench microbenchmark that makes the queue-decoupling comparison executable
before the CUDA/OptiX path is fully wired.

## Python Entry Point

```python
from p2cccd.bench import (
    NoQueueDecoupleConfig,
    run_no_queue_decouple_microbenchmark,
)

result = run_no_queue_decouple_microbenchmark(
    NoQueueDecoupleConfig(candidate_count=4096, repeat_count=5, batch_size=1024)
)
```

Each `NoQueueDecoupleCaseResult` reports:

- `elapsed_ms`
- `candidates_per_sec`
- `approx_bytes_written`
- `approx_bandwidth_mb_s`
- `candidate_write_count`
- `proposal_row_count`
- `proposal_output_count`
- `checksum`

The checksum is deterministic for a fixed config and catches accidental
dead-code removal or output disappearance in tests.

## STPF Option

By default the queue-decoupled path uses the deterministic dummy proposal
policy. Setting `use_stpf_model=True` requires passing a `STPFModel` instance:

```python
result = run_no_queue_decouple_microbenchmark(config, model=model, device="cpu")
```

This exercises the same batched inference wrapper used by the STPF pipeline.

## Interpretation

The intended comparison is qualitative at this stage:

- pure candidate writes show the lower-bound write path,
- inline tiny logic approximates adding cheap decision code directly in the
  candidate production loop,
- inline surrogate scoring approximates doing proposal-like arithmetic inline,
- queue-decoupled batch proposal measures the cost of keeping proposal work out
  of the candidate writer and processing it as a batch.

The final GPU version should replace the Python candidate loops with CUDA or
OptiX-emitted buffers and report rays/sec or queries/sec, candidate-buffer
bandwidth, trace time, proposal enqueue/dequeue time, total latency, and tail
latency under `BenchmarkRowV2`.

## Validation

`test_no_queue_decouple_microbenchmark.py` covers:

- all four cases are executed and produce positive counters,
- queue-decoupled proposal emits one feature row and one proposal output per
  candidate,
- checksums are deterministic for a fixed config,
- optional STPF model inference works,
- invalid config values are rejected.
