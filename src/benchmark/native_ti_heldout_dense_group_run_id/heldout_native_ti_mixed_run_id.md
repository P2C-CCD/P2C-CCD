# Public held-out native TI mixed benchmark

This table reorganizes the completed frozen-checkpoint native TI held-out
benchmark so the deployable fixed `MotionHigh` rule is visible rather than
hidden only inside the retrospective heuristic-oracle row.  All rows use
native Tight-Inclusion certificates and the same mixed positive/negative
hard-near-miss groups; FP entries are native TI positives in nominal
near-miss groups, not learned classifier false positives.

| Split | Method | Pos./neg. groups | Exact calls | Call red. | TP/TN/FP/FN | First-hit rank | Wall ms |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: |
| group-heldout | all-exact | 269/243 | 65,536 | 0.000% | 269/240/3/0 | 66.100 | 1445.422 |
| group-heldout | random | 269/243 | 48,062 | 26.663% | 269/240/3/0 | 64.468 | 1402.583 |
| group-heldout | frozen STPF | 269/243 | 31,445 | 52.019% | 269/240/3/0 | 2.695 | 739.876 |
| group-heldout | fixed proximity | 269/243 | 60,157 | 8.208% | 269/240/3/0 | 109.431 | 1487.197 |
| group-heldout | predeclared motion-high | 269/243 | 34,047 | 48.048% | 269/240/3/0 | 12.368 | 760.076 |
| group-heldout | best fixed heuristic oracle | 269/243 | 34,047 | 48.048% | 269/240/3/0 | 12.368 | 760.076 |
| source-heldout shapenetcore | all-exact | 113/143 | 32,768 | 0.000% | 113/138/5/0 | 72.867 | 1948.634 |
| source-heldout shapenetcore | random | 113/143 | 25,142 | 23.273% | 113/138/5/0 | 66.177 | 1528.715 |
| source-heldout shapenetcore | frozen STPF | 113/143 | 18,118 | 44.708% | 113/138/5/0 | 4.018 | 1081.133 |
| source-heldout shapenetcore | fixed proximity | 113/143 | 31,406 | 4.156% | 113/138/5/0 | 121.611 | 1925.187 |
| source-heldout shapenetcore | predeclared motion-high | 113/143 | 18,110 | 44.733% | 113/138/5/0 | 3.947 | 1081.516 |
| source-heldout shapenetcore | best fixed heuristic oracle | 113/143 | 18,110 | 44.733% | 113/138/5/0 | 3.947 | 1081.516 |

## Main reading

- On group-heldout mixed groups, frozen STPF reduces exact calls from 65,536 to 31,445, beating random, fixed proximity, and the predeclared motion-high rule while preserving FN=0.
- On source-heldout ShapeNetCore, frozen STPF reduces exact calls from 32,768 to 18,118 and is effectively tied with predeclared motion-high / best fixed heuristic oracle; this supports cross-source stability, not universal dominance over every hand rule.
- The table is therefore the main learned-vs-fixed-rule mixed evidence, while scene/object envelope rows remain a certificate/fallback audit.
