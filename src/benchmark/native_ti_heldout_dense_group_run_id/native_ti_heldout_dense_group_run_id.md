# Frozen-checkpoint Held-out Native TI Dense-group Benchmark (native_ti_heldout_dense_group_run_id)

## Scope

- Exact payload: native Tight-Inclusion `vertexFaceCCD` / `edgeEdgeCCD`.
- STPF is trained once per split, checkpointed, then evaluated only on held-out groups/source.
- Test rows report all-exact, random, frozen learned, one predeclared fixed heuristic, and a retrospective best fixed heuristic oracle.
- Negative candidates are near-miss hard negatives with clearance above TI tolerance; final TP/TN/FP/FN is still measured by native TI.
- The best heuristic oracle is diagnostic only; it is selected after test evaluation and is not a deployable baseline.

## Sources

| Source | Dataset | Mesh | Bytes |
| --- | --- | --- | ---: |
| `abc` | ABC official | `src/datasets/abc_official/official_obj_subset/00140115/00140115_5608d6d981742143e5e69179_trimesh_001.stl` | 11900884 |
| `fusion360` | Fusion 360 Gallery Assembly | `src/datasets/fusion360_full/137878_9c5480b1/assembly.obj` | 89927098 |
| `thingi10k` | Thingi10K | `src/datasets/thingi10k/official_subset/None/50309_a46b0d84.obj` | 46732 |
| `shapenetcore` | ShapeNetCore | `src/datasets/shapenet_core_v2/selected_ood_dense_run_id/04090263/8f5da7a2501f1018ae1a1b4c30d8ff9b/models/model_normalized.obj` | 98266468 |

## Generation Parameters

- Groups per source: `256`
- Group size: `128`
- Negative group ratio: `0.5`
- Epochs: `8`
- Seed: `fixed_seed`

## Split: `group_heldout`

- Frozen checkpoint: `src/outputs/stpf_training/native_ti_heldout_dense_group_run_id/group_heldout/model_state.pt`
- Train candidates: `65536`
- Test groups/candidates: `512` / `65536`
- Test positive/negative groups: `269` / `243`
- Validation recall@0.5 / precision@0.5: `0.800000` / `0.109589`

| Method | Groups | Candidates | Positive groups | Exact calls | Call reduction | TP/TN/FP/FN | First positive rank | Exact ms | Wall ms |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| AllExact+TI | 512 | 65536 | 269 | 65536 | 0.000% | 269/240/3/0 | 66.100 | 1418.130 | 1445.422 |
| Random+TI | 512 | 65536 | 269 | 48062 | 26.663% | 269/240/3/0 | 64.468 | 1376.558 | 1402.583 |
| FrozenLearned+TI | 512 | 65536 | 269 | 31445 | 52.019% | 269/240/3/0 | 2.695 | 716.778 | 739.876 |
| SingleHeuristicProximity+TI | 512 | 65536 | 269 | 60157 | 8.208% | 269/240/3/0 | 109.431 | 1424.975 | 1453.683 |
| BestFixedHeuristicOracle+TI (HeuristicMotionHigh) | 512 | 65536 | 269 | 34047 | 48.048% | 269/240/3/0 | 12.368 | 738.079 | 760.076 |

All fixed heuristic candidates:

| Heuristic | Exact calls | Call reduction | TP/TN/FP/FN | First positive rank | Wall ms |
| --- | ---: | ---: | --- | ---: | ---: |
| SingleHeuristicProximity+TI | 60157 | 8.208% | 269/240/3/0 | 109.431 | 1487.197 |
| HeuristicSmallGap+TI | 60157 | 8.208% | 269/240/3/0 | 109.431 | 1429.406 |
| HeuristicMotionHigh+TI | 34047 | 48.048% | 269/240/3/0 | 12.368 | 760.076 |
| HeuristicFeatureEnergy+TI | 48445 | 26.079% | 269/240/3/0 | 65.892 | 1184.079 |
| HeuristicExtentLow+TI | 51683 | 21.138% | 269/240/3/0 | 77.929 | 986.936 |

## Split: `source_heldout_shapenetcore`

- Frozen checkpoint: `src/outputs/stpf_training/native_ti_heldout_dense_group_run_id/source_heldout_shapenetcore/model_state.pt`
- Train candidates: `98304`
- Test groups/candidates: `256` / `32768`
- Test positive/negative groups: `113` / `143`
- Validation recall@0.5 / precision@0.5: `0.855263` / `0.081761`

| Method | Groups | Candidates | Positive groups | Exact calls | Call reduction | TP/TN/FP/FN | First positive rank | Exact ms | Wall ms |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| AllExact+TI | 256 | 32768 | 113 | 32768 | 0.000% | 113/138/5/0 | 72.867 | 1935.225 | 1948.634 |
| Random+TI | 256 | 32768 | 113 | 25142 | 23.273% | 113/138/5/0 | 66.177 | 1515.616 | 1528.715 |
| FrozenLearned+TI | 256 | 32768 | 113 | 18118 | 44.708% | 113/138/5/0 | 4.018 | 1069.940 | 1081.133 |
| SingleHeuristicProximity+TI | 256 | 32768 | 113 | 31406 | 4.156% | 113/138/5/0 | 121.611 | 1917.233 | 1930.109 |
| BestFixedHeuristicOracle+TI (HeuristicMotionHigh) | 256 | 32768 | 113 | 18110 | 44.733% | 113/138/5/0 | 3.947 | 1071.001 | 1081.516 |

All fixed heuristic candidates:

| Heuristic | Exact calls | Call reduction | TP/TN/FP/FN | First positive rank | Wall ms |
| --- | ---: | ---: | --- | ---: | ---: |
| SingleHeuristicProximity+TI | 31406 | 4.156% | 113/138/5/0 | 121.611 | 1925.187 |
| HeuristicSmallGap+TI | 31406 | 4.156% | 113/138/5/0 | 121.611 | 1948.276 |
| HeuristicMotionHigh+TI | 18110 | 44.733% | 113/138/5/0 | 3.947 | 1081.516 |
| HeuristicFeatureEnergy+TI | 25101 | 23.398% | 113/138/5/0 | 65.814 | 1713.084 |
| HeuristicExtentLow+TI | 27644 | 15.637% | 113/138/5/0 | 88.319 | 1325.346 |

## Reproduction

```powershell
conda activate cudadev
python src/tools/native_ti_heldout_dense_group_benchmark.py --run-name native_ti_heldout_dense_group_run_id --groups-per-source 256 --group-size 128 --negative-group-ratio 0.5 --epochs 8 --seed fixed_seed --heldout-source shapenetcore
```
