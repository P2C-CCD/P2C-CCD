# Large Selected-real Tight-Inclusion Dense Group Benchmark

## Scope

- Exact payload: native `ticcd::vertexFaceCCD` / `ticcd::edgeEdgeCCD` with the same parameters as the Tight-Inclusion baseline.
- Compared methods: `NoProposal+TI`, `Random+TI`, and `RTSTPFExact+TI`.
- STPF only changes the order of exact work items. It does not delete candidates or output collision truth.
- Negative or uncertain groups are evaluated to exhaustion; positive groups stop only after a certified TI hit.
- Shared preload reads the selected CSV query blocks once and is reported separately so the table measures native scheduling and exact certification.

## Inputs

- Dataset root: `src/benchmark/native_ti_heldout_dense_group_run_id/ti_csv_dataset`
- Learned schedule: `src/benchmark/native_ti_heldout_dense_group_run_id/schedules/source_heldout_shapenetcore/HeuristicFeatureEnergy.csv`
- Random schedule: `src/benchmark/native_ti_heldout_dense_group_run_id/schedules/source_heldout_shapenetcore/random.csv`
- Unique query blocks preloaded: `32768`
- Shared preload time: `898.615 ms`
- TI parameters: `ms=0, tolerance=1e-06, t_max=1, max_itr=1000000`

## Results

| Method | Groups | Candidates | Positive groups | Exact calls | Skipped calls | Call reduction | TP | TN | FP | FN | Recall | Precision | First positive rank | Exact ms | Wall ms | Wall + shared preload ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| NoProposal+TI | 256 | 32768 | 113 | 32768 | 0 | 0% | 113 | 138 | 5 | 0 | 1 | 0.957627 | 72.8673 | 1932.98 | 1946.51 | 2845.12 |
| Random+TI | 256 | 32768 | 113 | 25142 | 7626 | 23.2727% | 113 | 138 | 5 | 0 | 1 | 0.957627 | 66.177 | 1504.32 | 1516.91 | 2415.52 |
| RTSTPFExact+TI | 256 | 32768 | 113 | 25101 | 7667 | 23.3978% | 113 | 138 | 5 | 0 | 1 | 0.957627 | 65.8142 | 1700.72 | 1713.08 | 2611.7 |

## Interpretation

- `FN=0` is the required correctness condition. Any nonzero FP is conservative and is reported.
- `NoProposal+TI` is the all-candidate exact baseline.
- `Random+TI` tests whether early-stop alone explains the gain.
- `RTSTPFExact+TI` is the learned route: the same candidates are ordered by the learned STPF policy before TI certification.

