# P1 Ablation Suite Summary

## Artifacts

- P1-1 component ablation: `src\benchmark\p1_ablation_suite_run_id\p1_component_learned_vs_random_run_id.md`
- P1-1 native dense wall-time: `src\benchmark\p1_ablation_suite_run_id\p1_component_native_dense_walltime_run_id.md`
- P1-2 fallback threshold sweep: `src\benchmark\p1_ablation_suite_run_id\p1_fallback_threshold_sweep_run_id.md`
- P1-3 native density wall-time: `src\benchmark\p1_ablation_suite_run_id\p1_native_density_walltime_run_id.md`

## P1-1 Component Highlights

| Dataset | Method | Exact calls | Work reduction | FN |
| --- | --- | ---: | ---: | ---: |
| `common_modeling_large` | `NoProposalAllExact` | `111104` | `0.00%` | `0` |
| `common_modeling_large` | `ValidationSelectedFullSTPF` | `13024` | `90.54%` | `0` |
| `common_modeling_large` | `LearnedPriorityOnly` | `13024` | `90.54%` | `0` |
| `common_modeling_large` | `LearnedCostAware` | `13070` | `90.50%` | `0` |
| `common_modeling_large` | `IntervalOnly` | `32944` | `61.07%` | `0` |
| `common_modeling_large` | `RankingOnly` | `32042` | `61.86%` | `0` |
| `common_modeling_large` | `HeuristicCostLow` | `90969` | `25.69%` | `0` |
| `common_modeling_large` | `HeuristicCostHigh` | `3649` | `95.15%` | `0` |
| `common_modeling_large` | `RandomUniform(mean over seeds)` | `22728` | `79.53%` | `0` |
| `fusion360_full_assembly` | `NoProposalAllExact` | `262144` | `0.00%` | `0` |
| `fusion360_full_assembly` | `ValidationSelectedFullSTPF` | `1694` | `98.92%` | `0` |
| `fusion360_full_assembly` | `LearnedPriorityOnly` | `1934` | `99.59%` | `0` |
| `fusion360_full_assembly` | `LearnedCostAware` | `1694` | `98.92%` | `0` |
| `fusion360_full_assembly` | `IntervalOnly` | `229531` | `4.71%` | `0` |
| `fusion360_full_assembly` | `RankingOnly` | `185472` | `17.08%` | `0` |
| `fusion360_full_assembly` | `HeuristicCostLow` | `195752` | `73.29%` | `0` |
| `fusion360_full_assembly` | `HeuristicCostHigh` | `23885` | `37.46%` | `0` |
| `fusion360_full_assembly` | `RandomUniform(mean over seeds)` | `52791` | `79.84%` | `0` |
| `rtstpf_advantage_v4` | `NoProposalAllExact` | `262144` | `0.00%` | `0` |
| `rtstpf_advantage_v4` | `ValidationSelectedFullSTPF` | `9885` | `96.69%` | `0` |
| `rtstpf_advantage_v4` | `LearnedPriorityOnly` | `9885` | `96.69%` | `0` |
| `rtstpf_advantage_v4` | `LearnedCostAware` | `7124` | `97.58%` | `0` |
| `rtstpf_advantage_v4` | `IntervalOnly` | `127491` | `32.85%` | `0` |
| `rtstpf_advantage_v4` | `RankingOnly` | `93205` | `54.78%` | `0` |
| `rtstpf_advantage_v4` | `HeuristicCostLow` | `171559` | `58.28%` | `0` |
| `rtstpf_advantage_v4` | `HeuristicCostHigh` | `33620` | `68.65%` | `0` |
| `rtstpf_advantage_v4` | `RandomUniform(mean over seeds)` | `52082` | `80.11%` | `0` |
| `shapenet_ood_dense` | `NoProposalAllExact` | `262144` | `0.00%` | `0` |
| `shapenet_ood_dense` | `ValidationSelectedFullSTPF` | `16796` | `93.51%` | `0` |
| `shapenet_ood_dense` | `LearnedPriorityOnly` | `10340` | `95.98%` | `0` |
| `shapenet_ood_dense` | `LearnedCostAware` | `16796` | `93.51%` | `0` |
| `shapenet_ood_dense` | `IntervalOnly` | `167429` | `36.09%` | `0` |
| `shapenet_ood_dense` | `RankingOnly` | `167772` | `35.97%` | `0` |
| `shapenet_ood_dense` | `HeuristicCostLow` | `260608` | `0.90%` | `0` |
| `shapenet_ood_dense` | `HeuristicCostHigh` | `512` | `99.69%` | `0` |
| `shapenet_ood_dense` | `RandomUniform(mean over seeds)` | `52702` | `79.88%` | `0` |

## P1-2 Default Fallback Operating Point

| Dataset | Threshold | Attempts | Calls | Work reduction | E2E ms | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `common_modeling_large` | `0.75` | `3` | `110893` | `92.42%` | `401.045` | `0` |
| `fusion360_full_assembly` | `0.75` | `3` | `263974` | `72.67%` | `739.334` | `0` |
| `rtstpf_advantage_v4` | `0.75` | `3` | `404847` | `84.81%` | `1381.113` | `0` |
| `shapenet_ood_dense` | `0.75` | `3` | `622944` | `69.63%` | `1173.037` | `0` |

## P1-3 Native Density Wall-Time

| Density | Candidates | Calls | Work reduction | E2E ms | FN |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `16` | `8448` | `2448` | `93.02%` | `14.612` | `0` |
| `32` | `16896` | `4429` | `95.34%` | `31.812` | `0` |
| `64` | `33792` | `8370` | `94.61%` | `97.124` | `0` |
| `128` | `67584` | `16297` | `96.00%` | `264.441` | `0` |
| `256` | `135168` | `32166` | `95.97%` | `291.557` | `0` |
| `512` | `270336` | `63907` | `87.58%` | `427.094` | `0` |
| `1024` | `540672` | `127386` | `84.34%` | `463.910` | `0` |
| `2304` | `1216512` | `286218` | `83.54%` | `1023.163` | `0` |

## Interpretation

- P1-1 now separates validation-selected full STPF, individual heads, random, heuristics, and all-exact/no-proposal budgets under the same balanced hard-negative ranking protocol.
- P1-2 confirms the native dense fallback operating point preserves FN=0 under the swept uncertainty/attempt settings.
- P1-3 converts the existing density sweep from exact-work only into native ORT + C++ dense replay wall-time for the density rows.
