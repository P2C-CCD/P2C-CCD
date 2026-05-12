# P2-1 Training-source Ablation

## Scope

P2-1 sampled retraining over four current dense sources; evaluation uses balanced hard-negative rank challenges.

The run retrains lightweight STPF variants from sampled rows because this workstation exposes a CPU-only PyTorch build; ORT/C++ paths are still evaluated in the other P2 parts.

## Variant Summary

| variant | train_row_count | left_out | drop_hard_negatives | mean_exact_work_reduction | mean_speedup_vs_random_work | mean_first_positive_rank | mean_win_rate_vs_random | max_fn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full_mixed_25pct | 2048 |  | False | 0.5276 | 0.5418 | 151.5227 | 0.2000 | 0 |
| full_mixed_50pct | 4096 |  | False | 0.8013 | 1.0089 | 101.8125 | 0.6083 | 0 |
| full_mixed_100pct | 8192 |  | False | 0.6639 | 0.7361 | 145.9529 | 0.4333 | 0 |
| no_hard_negative_100pct | 8192 |  | True | 0.8324 | 2.1107 | 109.1630 | 0.7500 | 0 |
| leave_out_common_modeling_large | 8193 | common_modeling_large | False | 0.8378 | 1.7364 | 115.6604 | 0.7000 | 0 |
| leave_out_fusion360_full_assembly | 8193 | fusion360_full_assembly | False | 0.9495 | 6.5527 | 32.3177 | 1.0000 | 0 |
| leave_out_rtstpf_advantage_v4 | 8193 | rtstpf_advantage_v4 | False | 0.8361 | 1.9513 | 37.3624 | 0.7500 | 0 |
| leave_out_shapenet_ood_dense | 8193 | shapenet_ood_dense | False | 0.7064 | 3.6021 | 118.0481 | 0.2500 | 0 |

## Per-source Rows

| variant | eval_source | scheduled_exact_calls | scheduled_exact_work | exact_work_reduction | first_positive_rank_mean | speedup_vs_random_work | win_rate_vs_random | fn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full_mixed_25pct | common_modeling_large | 33108 | 554012.9 | 0.6099 | 152.5714 | 0.5090 | 0.0000 | 0 |
| full_mixed_25pct | fusion360_full_assembly | 86557 | 5970763.5 | 0.2619 | 169.0566 | 0.2711 | 0.0000 | 0 |
| full_mixed_25pct | rtstpf_advantage_v4 | 94480 | 2475797.1 | 0.4342 | 184.5312 | 0.3543 | 0.0000 | 0 |
| full_mixed_25pct | shapenet_ood_dense | 51165 | 598000.4 | 0.8045 | 99.9316 | 1.0329 | 0.8000 | 0 |
| full_mixed_50pct | common_modeling_large | 23091 | 295411.5 | 0.7920 | 106.4101 | 0.9545 | 0.2667 | 0 |
| full_mixed_50pct | fusion360_full_assembly | 51357 | 1565172.4 | 0.8065 | 100.3066 | 1.0341 | 0.7000 | 0 |
| full_mixed_50pct | rtstpf_advantage_v4 | 51508 | 865162.9 | 0.8023 | 100.6016 | 1.0140 | 0.6667 | 0 |
| full_mixed_50pct | shapenet_ood_dense | 51165 | 598000.4 | 0.8045 | 99.9316 | 1.0329 | 0.8000 | 0 |
| full_mixed_100pct | common_modeling_large | 42247 | 637118.4 | 0.5514 | 194.6866 | 0.4426 | 0.0000 | 0 |
| full_mixed_100pct | fusion360_full_assembly | 65588 | 1498931.6 | 0.8147 | 128.1016 | 1.0797 | 0.9333 | 0 |
| full_mixed_100pct | rtstpf_advantage_v4 | 82479 | 2253964.9 | 0.4849 | 161.0918 | 0.3892 | 0.0000 | 0 |
| full_mixed_100pct | shapenet_ood_dense | 51165 | 598000.4 | 0.8045 | 99.9316 | 1.0329 | 0.8000 | 0 |
| no_hard_negative_100pct | common_modeling_large | 14344 | 167155.0 | 0.8823 | 66.1014 | 1.6869 | 1.0000 | 0 |
| no_hard_negative_100pct | fusion360_full_assembly | 42911 | 409976.4 | 0.9493 | 83.8105 | 3.9477 | 1.0000 | 0 |
| no_hard_negative_100pct | rtstpf_advantage_v4 | 37930 | 377721.7 | 0.9137 | 74.0820 | 2.3225 | 1.0000 | 0 |
| no_hard_negative_100pct | shapenet_ood_dense | 108881 | 1272006.1 | 0.5842 | 212.6582 | 0.4856 | 0.0000 | 0 |
| leave_out_common_modeling_large | common_modeling_large | 13525 | 140543.9 | 0.9010 | 62.3272 | 2.0063 | 1.0000 | 0 |
| leave_out_common_modeling_large | fusion360_full_assembly | 125076 | 2364197.5 | 0.7078 | 244.2891 | 0.6846 | 0.0000 | 0 |
| leave_out_common_modeling_large | rtstpf_advantage_v4 | 28720 | 272271.5 | 0.9378 | 56.0938 | 3.2220 | 1.0000 | 0 |
| leave_out_common_modeling_large | shapenet_ood_dense | 51165 | 598000.4 | 0.8045 | 99.9316 | 1.0329 | 0.8000 | 0 |
| leave_out_fusion360_full_assembly | common_modeling_large | 11514 | 132851.6 | 0.9065 | 53.0599 | 2.1224 | 1.0000 | 0 |
| leave_out_fusion360_full_assembly | fusion360_full_assembly | 10097 | 111263.1 | 0.9862 | 19.7207 | 14.5463 | 1.0000 | 0 |
| leave_out_fusion360_full_assembly | rtstpf_advantage_v4 | 12592 | 138137.2 | 0.9684 | 24.5938 | 6.3507 | 1.0000 | 0 |
| leave_out_fusion360_full_assembly | shapenet_ood_dense | 16331 | 193557.2 | 0.9367 | 31.8965 | 3.1911 | 1.0000 | 0 |
| leave_out_rtstpf_advantage_v4 | common_modeling_large | 13343 | 203215.1 | 0.8569 | 61.4885 | 1.3875 | 1.0000 | 0 |
| leave_out_rtstpf_advantage_v4 | fusion360_full_assembly | 4418 | 2490320.2 | 0.6922 | 8.6289 | 0.6499 | 0.0000 | 0 |
| leave_out_rtstpf_advantage_v4 | rtstpf_advantage_v4 | 29153 | 699088.6 | 0.8402 | 56.9395 | 1.2549 | 1.0000 | 0 |
| leave_out_rtstpf_advantage_v4 | shapenet_ood_dense | 11465 | 136867.9 | 0.9553 | 22.3926 | 4.5129 | 1.0000 | 0 |
| leave_out_shapenet_ood_dense | common_modeling_large | 32813 | 549279.5 | 0.6132 | 151.2120 | 0.5133 | 0.0000 | 0 |
| leave_out_shapenet_ood_dense | fusion360_full_assembly | 80251 | 2146218.0 | 0.7347 | 156.7402 | 0.7541 | 0.0000 | 0 |
| leave_out_shapenet_ood_dense | rtstpf_advantage_v4 | 80147 | 2216025.1 | 0.4935 | 156.5371 | 0.3959 | 0.0000 | 0 |
| leave_out_shapenet_ood_dense | shapenet_ood_dense | 3944 | 48462.4 | 0.9842 | 7.7031 | 12.7453 | 1.0000 | 0 |
