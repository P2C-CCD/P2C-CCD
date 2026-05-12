# Large Selected-real Tight-Inclusion Dense Group Benchmark

## Scope

- Exact payload: native `ticcd::vertexFaceCCD` / `ticcd::edgeEdgeCCD` with the same parameters as the Tight-Inclusion baseline.
- Compared methods: `NoProposal+TI`, `Random+TI`, and `RTSTPFExact+TI`.
- STPF only changes the order of exact work items. It does not delete candidates or output collision truth.
- Negative or uncertain groups are evaluated to exhaustion; positive groups stop only after a certified TI hit.
- Shared preload reads the selected CSV query blocks once and is reported separately so the table measures native scheduling and exact certification.

## Inputs

- Dataset root: `src/benchmark/native_ti_heldout_dense_group_run_id/ti_csv_dataset`
- Learned schedule: `src/benchmark/native_ti_heldout_dense_group_run_id/schedules/group_heldout/HeuristicFeatureEnergy.csv`
- Random schedule: `src/benchmark/native_ti_heldout_dense_group_run_id/schedules/group_heldout/random.csv`
- Unique query blocks preloaded: `65536`
- Shared preload time: `1923.69 ms`
- TI parameters: `ms=0, tolerance=1e-06, t_max=1, max_itr=1000000`

## Results

| Method | Groups | Candidates | Positive groups | Exact calls | Skipped calls | Call reduction | TP | TN | FP | FN | Recall | Precision | First positive rank | Exact ms | Wall ms | Wall + shared preload ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| NoProposal+TI | 512 | 65536 | 269 | 65536 | 0 | 0% | 269 | 240 | 3 | 0 | 1 | 0.988971 | 66.1004 | 1405.93 | 1432.07 | 3355.77 |
| Random+TI | 512 | 65536 | 269 | 48062 | 17474 | 26.6632% | 269 | 240 | 3 | 0 | 1 | 0.988971 | 64.4684 | 1359.08 | 1384.29 | 3307.99 |
| RTSTPFExact+TI | 512 | 65536 | 269 | 48445 | 17091 | 26.0788% | 269 | 240 | 3 | 0 | 1 | 0.988971 | 65.8922 | 1158.95 | 1184.08 | 3107.77 |

## Interpretation

- `FN=0` is the required correctness condition. Any nonzero FP is conservative and is reported.
- `NoProposal+TI` is the all-candidate exact baseline.
- `Random+TI` tests whether early-stop alone explains the gain.
- `RTSTPFExact+TI` is the learned route: the same candidates are ordered by the learned STPF policy before TI certification.

