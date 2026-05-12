# Tight-Inclusion / NYU 100GB Full-query Every-candidate Wall-time Benchmark

Run identifier: `run_id`

This report aggregates the native C++ Tight-Inclusion full-query shards. Every query in the manifest was sent to the exact Tight-Inclusion kernel; there is no proposal skipping in this baseline.

## Inputs

- Manifest: `src/datasets/manifests/tight_inclusion_nyu_full_manifest_run_id.json`
- Shard output dir: `src/benchmark/ti_full_query_every_candidate_walltime_run_id`
- Exact parameters: `ms=0.0, tolerance=1e-6, t_max=1.0, max_itr=1000000, no_zero_toi=false, root=BREADTH_FIRST_SEARCH`
- Native executable: `src/build_tools/tight_inclusion_full_query_benchmark.exe`

## Overall

| Files | Queries | Exact calls | TP | TN | FP | FN | Recall | Wall ms | Exact ms | QPS | Avg wall us/query |
| --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| `5203` | `59387883` | `59387883` | `26982` | `59360589` | `312` | `0` | `1` | `1181351.147` | `48249.376` | `50271.152` | `19.892` |

Correctness result: `FN=0` across the aggregated full-query manifest rows. `FP` is reported because Tight-Inclusion is conservative.

## By Split

| Split | Files | Queries | Positives | Exact calls | TP | TN | FP | FN | Wall ms | QPS | Avg us/query |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |
| `heldout_test` | `1038` | `11873539` | `5055` | `11873539` | `5055` | `11868424` | `60` | `0` | `237077.393` | `50082.966` | `19.967` |
| `train` | `3638` | `41583240` | `18816` | `41583240` | `18816` | `41564224` | `200` | `0` | `824273.218` | `50448.370` | `19.822` |
| `unit_smoke` | `8` | `690` | `338` | `690` | `338` | `352` | `0` | `0` | `2729.620` | `252.782` | `3955.971` |
| `validation` | `519` | `5930414` | `2773` | `5930414` | `2773` | `5927589` | `52` | `0` | `117270.916` | `50570.203` | `19.774` |

## Heldout Test Per Case / Kind

| Case | Kind | Queries | Positives | TP | TN | FP | FN | Wall ms | QPS | p50 us | p90 us | p99 us |
| --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `chain` | `edge-edge` | `4312500` | `557` | `557` | `4311940` | `3` | `0` | `80894.700` | `53310.044` | `16.9` | `18.7` | `29.7` |
| `chain` | `vertex-face` | `1075000` | `9` | `9` | `1074991` | `0` | `0` | `20399.200` | `52698.145` | `17.1` | `19` | `30.7` |
| `cow-heads` | `edge-edge` | `2312500` | `376` | `376` | `2312124` | `0` | `0` | `43831.700` | `52758.620` | `17.2` | `18.7` | `29.3` |
| `cow-heads` | `vertex-face` | `637500` | `44` | `44` | `637456` | `0` | `0` | `12167.200` | `52394.964` | `17.3` | `18.9` | `30.2` |
| `erleben-cube-cliff-edges` | `edge-edge` | `250` | `18` | `18` | `190` | `42` | `0` | `3162.870` | `79.042` | `18.1` | `64.46` | `218140` |
| `erleben-cube-cliff-edges` | `vertex-face` | `250` | `0` | `0` | `250` | `0` | `0` | `4.973` | `50269.444` | `17.1` | `19.3` | `24.751` |
| `erleben-cube-internal-edges` | `edge-edge` | `600` | `18` | `18` | `574` | `8` | `0` | `12.727` | `47142.015` | `17` | `21.12` | `49.927` |
| `erleben-cube-internal-edges` | `vertex-face` | `375` | `16` | `16` | `354` | `5` | `0` | `149.480` | `2508.697` | `17.7` | `23.46` | `108.318` |
| `erleben-sliding-spike` | `edge-edge` | `125` | `0` | `0` | `125` | `0` | `0` | `2.683` | `46589.638` | `17.9` | `19.6` | `45.172` |
| `erleben-sliding-spike` | `vertex-face` | `125` | `7` | `7` | `118` | `0` | `0` | `2.700` | `46291.153` | `16.9` | `20.98` | `55.612` |
| `erleben-sliding-wedge` | `edge-edge` | `125` | `0` | `0` | `125` | `0` | `0` | `2.701` | `46282.583` | `17.5` | `21.72` | `42.272` |
| `erleben-sliding-wedge` | `vertex-face` | `375` | `0` | `0` | `375` | `0` | `0` | `7.234` | `51842.123` | `16.4` | `17.66` | `26.126` |
| `erleben-spike-crack` | `edge-edge` | `125` | `0` | `0` | `125` | `0` | `0` | `2.839` | `44024.936` | `18.1` | `19.76` | `40.956` |
| `erleben-spike-crack` | `vertex-face` | `125` | `2` | `2` | `123` | `0` | `0` | `142.917` | `874.634` | `17.8` | `573.84` | `29778.7` |
| `erleben-spike-hole` | `edge-edge` | `4125` | `0` | `0` | `4125` | `0` | `0` | `85.180` | `48426.861` | `17.7` | `20` | `25.1` |
| `erleben-spike-hole` | `vertex-face` | `1875` | `1` | `1` | `1874` | `0` | `0` | `37.793` | `49612.362` | `17.5` | `18.9` | `24.426` |
| `erleben-spike-wedge` | `edge-edge` | `375` | `19` | `19` | `354` | `2` | `0` | `84.839` | `4420.147` | `20.5` | `314.22` | `4437.13` |
| `erleben-spike-wedge` | `vertex-face` | `125` | `5` | `5` | `120` | `0` | `0` | `84.073` | `1486.812` | `20.8` | `446.78` | `23922.7` |
| `erleben-spikes` | `edge-edge` | `250` | `4` | `4` | `246` | `0` | `0` | `14.897` | `16781.790` | `20.1` | `48.69` | `1233.64` |
| `erleben-spikes` | `vertex-face` | `125` | `0` | `0` | `125` | `0` | `0` | `2.959` | `42242.574` | `19.7` | `24.44` | `60.836` |
| `erleben-wedge-crack` | `edge-edge` | `484` | `0` | `0` | `484` | `0` | `0` | `10.355` | `46741.156` | `18` | `19.8` | `28.67` |
| `erleben-wedge-crack` | `vertex-face` | `330` | `5` | `5` | `325` | `0` | `0` | `8.418` | `39201.245` | `18` | `23.11` | `158.404` |
| `erleben-wedges` | `edge-edge` | `500` | `7` | `7` | `493` | `0` | `0` | `30.758` | `16255.828` | `19` | `34.24` | `591.523` |
| `erleben-wedges` | `vertex-face` | `375` | `11` | `11` | `364` | `0` | `0` | `9.597` | `39076.340` | `18.8` | `24.8` | `112.224` |
| `golf-ball` | `edge-edge` | `812500` | `328` | `328` | `812172` | `0` | `0` | `17973.100` | `45206.447` | `20.8` | `21.3` | `33.4` |
| `golf-ball` | `vertex-face` | `1287500` | `3417` | `3417` | `1284083` | `0` | `0` | `28836.800` | `44647.811` | `20.7` | `21.4` | `39.2` |
| `mat-twist` | `edge-edge` | `812500` | `174` | `174` | `812326` | `0` | `0` | `16803.000` | `48354.461` | `18.8` | `20.6` | `34.3` |
| `mat-twist` | `vertex-face` | `612500` | `37` | `37` | `612463` | `0` | `0` | `12311.700` | `49749.425` | `18.5` | `20` | `29.3` |

## Validation / Train Summary

- Validation shards: `26` case/kind rows.
- Train shards: `28` case/kind rows.
- Full per-shard JSONL/Markdown files are kept in the shard directory for replay and audit.

## Reproduce / Resume

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File src/tools/run_tight_inclusion_full_query_shards.ps1 -OutputDir src/benchmark/ti_full_query_every_candidate_walltime_run_id -Split heldout_test
powershell -NoProfile -ExecutionPolicy Bypass -File src/tools/run_tight_inclusion_full_query_shards.ps1 -OutputDir src/benchmark/ti_full_query_every_candidate_walltime_run_id -Split validation
powershell -NoProfile -ExecutionPolicy Bypass -File src/tools/run_tight_inclusion_full_query_shards.ps1 -OutputDir src/benchmark/ti_full_query_every_candidate_walltime_run_id -Split train
powershell -NoProfile -ExecutionPolicy Bypass -File src/tools/summarize_tight_inclusion_full_query_walltime.py -OutputDir src/benchmark/ti_full_query_every_candidate_walltime_run_id
```

Resume rule: the shard wrapper skips existing `jsonl/md` pairs unless `-Force` is passed.

## Notes

- Top-level aggregate p50/p90/p99 are not recomputed from raw per-query latencies because raw latencies are not persisted for the 100GB run. Per-shard exact percentiles are in each shard JSONL/MD.
- This is the SOTA primitive exact wall-time baseline. It should be compared against RTSTPFExact only when RTSTPFExact uses the same exact certificate/fallback policy.
