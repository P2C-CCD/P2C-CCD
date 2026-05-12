# Benchmark Result Template

Use this template when copying `BenchmarkRowV2` exports into paper-style tables or experiment notes.

## Run Metadata

| field | value |
| --- | --- |
| run_id |  |
| dataset_name |  |
| scene_name |  |
| method_name |  |
| config_hash |  |
| seed |  |
| git_commit |  |
| gpu_name |  |
| driver_version |  |
| cuda_version |  |
| optix_version |  |

## Correctness Table

| method | dataset | scene | queries | candidate recall | FN | FP | final FN = 0 |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
|  |  |  |  |  |  |  |  |

## Performance Table

| method | dataset | scene | qps | total ms | rt build ms | rt update ms | rt trace ms | proposal ms | exact ms | p50 ms | p95 ms | p99 ms |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
|  |  |  |  |  |  |  |  |  |  |  |  |  |

## Workload Table

| method | avg candidates | avg exact evals | candidate inflation | exact queue occupancy | fallback ratio | PT calls | EE calls | conservative calls | unknown calls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
|  |  |  |  |  |  |  |  |  |  |

## NoQueueDecouple Table

| case | queries/sec | candidate buffer bandwidth MB/s | trace ms | proposal enqueue/dequeue ms | total tail latency ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| pure_candidate_writes |  |  |  |  |  |
| inline_tiny_logic |  |  |  |  |  |
| inline_surrogate_scoring |  |  |  |  |  |
| queue_decoupled_batch_proposal |  |  |  |  |  |

## Required Gate Notes

- Correctness rows must report `fn_count = 0`.
- Candidate-generation rows must report `candidate_recall = 1.0`.
- STPF rows must preserve queue conservation and proposal monotonicity.
- Any OOD fallback is allowed to be slower, but cannot introduce false negatives.
