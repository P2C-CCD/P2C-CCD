# P1-2 Conservative Fallback Threshold Sweep

## Scope

- Native dense-group early-stop path over the four main dense sources.
- Swept native-exposed safety knobs: `uncertainty_fallback_threshold` and `representative_attempt_limit`.
- `family_score_threshold` and OOD thresholds are not exposed by the native dense oracle kernel, so they remain Python-contract checks rather than this native sweep.

## Default Operating Point

| Dataset | Head | Threshold | Attempts | Exact calls | Call reduction | Work reduction | Fallback calls | E2E ms | FN |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `common_modeling_large` | `priority_only` | `0.75` | `3` | `110893` | `83.29%` | `92.42%` | `48264` | `401.045` | `0` |
| `fusion360_full_assembly` | `cost_aware` | `0.75` | `3` | `263974` | `74.83%` | `72.67%` | `259070` | `739.334` | `0` |
| `rtstpf_advantage_v4` | `priority_only` | `0.75` | `3` | `404847` | `79.66%` | `84.81%` | `382091` | `1381.113` | `0` |
| `shapenet_ood_dense` | `cost_aware` | `0.75` | `3` | `622944` | `58.28%` | `69.63%` | `583362` | `1173.037` | `0` |

## Full Sweep

| Dataset | Threshold | Attempts | Exact calls | Work reduction | Fallback calls | Interval miss | E2E ms | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `common_modeling_large` | `0.5` | `1` | `110752` | `88.03%` | `110752` | `0` | `413.915` | `0` |
| `common_modeling_large` | `0.5` | `3` | `110752` | `88.03%` | `110752` | `0` | `408.762` | `0` |
| `common_modeling_large` | `0.5` | `5` | `110752` | `88.03%` | `110752` | `0` | `408.856` | `0` |
| `common_modeling_large` | `0.75` | `1` | `110799` | `92.42%` | `48264` | `47` | `416.819` | `0` |
| `common_modeling_large` | `0.75` | `3` | `110893` | `92.42%` | `48264` | `141` | `401.045` | `0` |
| `common_modeling_large` | `0.75` | `5` | `110987` | `92.41%` | `48264` | `235` | `379.990` | `0` |
| `common_modeling_large` | `0.95` | `1` | `110799` | `97.84%` | `47` | `47` | `414.838` | `0` |
| `common_modeling_large` | `0.95` | `3` | `110893` | `97.83%` | `47` | `141` | `377.138` | `0` |
| `common_modeling_large` | `0.95` | `5` | `110987` | `97.83%` | `47` | `235` | `407.659` | `0` |
| `common_modeling_large` | `1.1` | `1` | `110799` | `97.84%` | `47` | `47` | `438.476` | `0` |
| `common_modeling_large` | `1.1` | `3` | `110893` | `97.83%` | `47` | `141` | `394.105` | `0` |
| `common_modeling_large` | `1.1` | `5` | `110987` | `97.83%` | `47` | `235` | `441.636` | `0` |
| `fusion360_full_assembly` | `0.5` | `1` | `263268` | `72.68%` | `259594` | `356` | `762.434` | `0` |
| `fusion360_full_assembly` | `0.5` | `3` | `263968` | `72.66%` | `259588` | `1056` | `771.997` | `0` |
| `fusion360_full_assembly` | `0.5` | `5` | `264665` | `72.64%` | `259586` | `1753` | `772.031` | `0` |
| `fusion360_full_assembly` | `0.75` | `1` | `263270` | `72.69%` | `259076` | `358` | `798.353` | `0` |
| `fusion360_full_assembly` | `0.75` | `3` | `263974` | `72.67%` | `259070` | `1062` | `739.334` | `0` |
| `fusion360_full_assembly` | `0.75` | `5` | `264675` | `72.66%` | `259068` | `1763` | `746.450` | `0` |
| `fusion360_full_assembly` | `0.95` | `1` | `263270` | `72.72%` | `258074` | `358` | `841.045` | `0` |
| `fusion360_full_assembly` | `0.95` | `3` | `263974` | `72.70%` | `258068` | `1062` | `763.513` | `0` |
| `fusion360_full_assembly` | `0.95` | `5` | `264675` | `72.69%` | `258066` | `1763` | `710.757` | `0` |
| `fusion360_full_assembly` | `1.1` | `1` | `263270` | `95.04%` | `358` | `358` | `728.724` | `0` |
| `fusion360_full_assembly` | `1.1` | `3` | `263974` | `95.02%` | `352` | `1062` | `772.819` | `0` |
| `fusion360_full_assembly` | `1.1` | `5` | `264675` | `95.01%` | `350` | `1763` | `750.282` | `0` |
| `rtstpf_advantage_v4` | `0.5` | `1` | `404226` | `84.75%` | `384219` | `337` | `1357.005` | `0` |
| `rtstpf_advantage_v4` | `0.5` | `3` | `404847` | `84.74%` | `384192` | `958` | `1319.871` | `0` |
| `rtstpf_advantage_v4` | `0.5` | `5` | `405432` | `84.73%` | `384160` | `1543` | `1347.944` | `0` |
| `rtstpf_advantage_v4` | `0.75` | `1` | `404226` | `84.82%` | `382118` | `337` | `1339.295` | `0` |
| `rtstpf_advantage_v4` | `0.75` | `3` | `404847` | `84.81%` | `382091` | `958` | `1381.113` | `0` |
| `rtstpf_advantage_v4` | `0.75` | `5` | `405432` | `84.80%` | `382059` | `1543` | `1345.100` | `0` |
| `rtstpf_advantage_v4` | `0.95` | `1` | `404226` | `85.17%` | `374299` | `337` | `1342.383` | `0` |
| `rtstpf_advantage_v4` | `0.95` | `3` | `404847` | `85.16%` | `374272` | `958` | `1331.237` | `0` |
| `rtstpf_advantage_v4` | `0.95` | `5` | `405432` | `85.15%` | `374240` | `1543` | `1345.352` | `0` |
| `rtstpf_advantage_v4` | `1.1` | `1` | `404226` | `97.16%` | `337` | `337` | `1329.906` | `0` |
| `rtstpf_advantage_v4` | `1.1` | `3` | `404847` | `97.15%` | `310` | `958` | `1358.069` | `0` |
| `rtstpf_advantage_v4` | `1.1` | `5` | `405432` | `97.14%` | `278` | `1543` | `1346.703` | `0` |
| `shapenet_ood_dense` | `0.5` | `1` | `622620` | `69.64%` | `583362` | `162` | `1138.420` | `0` |
| `shapenet_ood_dense` | `0.5` | `3` | `622944` | `69.63%` | `583362` | `486` | `1158.120` | `0` |
| `shapenet_ood_dense` | `0.5` | `5` | `623268` | `69.62%` | `583362` | `810` | `1169.556` | `0` |
| `shapenet_ood_dense` | `0.75` | `1` | `622620` | `69.64%` | `583362` | `162` | `1154.254` | `0` |
| `shapenet_ood_dense` | `0.75` | `3` | `622944` | `69.63%` | `583362` | `486` | `1173.037` | `0` |
| `shapenet_ood_dense` | `0.75` | `5` | `623268` | `69.62%` | `583362` | `810` | `1155.411` | `0` |
| `shapenet_ood_dense` | `0.95` | `1` | `622620` | `69.64%` | `583362` | `162` | `1159.418` | `0` |
| `shapenet_ood_dense` | `0.95` | `3` | `622944` | `69.63%` | `583362` | `486` | `1161.794` | `0` |
| `shapenet_ood_dense` | `0.95` | `5` | `623268` | `69.62%` | `583362` | `810` | `1145.989` | `0` |
| `shapenet_ood_dense` | `1.1` | `1` | `622620` | `94.22%` | `162` | `162` | `1156.452` | `0` |
| `shapenet_ood_dense` | `1.1` | `3` | `622944` | `94.21%` | `162` | `486` | `1168.064` | `0` |
| `shapenet_ood_dense` | `1.1` | `5` | `623268` | `94.20%` | `162` | `810` | `1185.892` | `0` |

## Conclusion

- All swept conservative settings reported FN=0 on these dense oracle groups.
- Higher uncertainty thresholds can reduce fallback exact calls only when the learned intervals resolve early; if not, exact work can rise through interval-miss recovery.
- The default threshold 0.75 / attempts 3 remains a conservative operating point rather than the fastest diagnostic setting.
