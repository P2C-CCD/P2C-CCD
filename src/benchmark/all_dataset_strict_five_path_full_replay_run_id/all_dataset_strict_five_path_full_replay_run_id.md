# All-dataset Strict Five-path Full Replay

Run identifier: `run_id`

This is the P0-A strict replay pass over all currently adapter-ready P2CCCD candidate-row datasets. Each listed dataset is evaluated by the same adapter and emits the same five method rows: `PureExactCPU`, `BVHExact`, `RTExact`, `RTSTPFExact`, and `NoProposal`.

Scope caveat: this is a strict same-runner replay over the unified candidate-row representation. It does not replace native full-scene simulators or the native Tight-Inclusion full-query exact table; those remain separate baseline tables. The all-exact rows report adapter scan cost over candidate rows, not native primitive exact kernel wall time.

## Overall Summary

| Metric | Value |
| --- | ---: |
| datasets | `12` |
| candidate rows | `8204800` |
| query groups | `25819` |
| RTSTPFExact exact calls | `2188010` |
| RTSTPFExact call reduction | `73.3326%` |
| RTSTPFExact work reduction | `46.4189%` |
| RTSTPFExact FN | `0` |
| RTSTPFExact wall ms sum | `3772.729` |
| ORT providers | `TensorrtExecutionProvider` |

## Dataset Coverage

| Dataset | Source | Candidate rows | Query groups | RTSTPF exact calls | RTSTPF call reduction | RTSTPF work reduction | FN | Provider |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `T0 synthetic_proxy` | `generalization_full` | `430080` | `3360` | `79560` | `81.5011%` | `36.3898%` | `0` | `TensorrtExecutionProvider` |
| `trained_stpf_high_density` | `generalization_full` | `448000` | `3500` | `92400` | `79.3750%` | `34.6769%` | `0` | `TensorrtExecutionProvider` |
| `ABC CAD` | `generalization_full` | `921600` | `7200` | `235800` | `74.4141%` | `49.5259%` | `0` | `TensorrtExecutionProvider` |
| `Thingi10K` | `generalization_full` | `577536` | `4512` | `147768` | `74.4141%` | `54.8694%` | `0` | `TensorrtExecutionProvider` |
| `Fusion 360 Gallery Assembly` | `generalization_full` | `230400` | `1800` | `126133` | `45.2548%` | `19.7577%` | `0` | `TensorrtExecutionProvider` |
| `high_density_mesh_multi_source` | `generalization_full` | `345600` | `2700` | `89314` | `74.1568%` | `56.3263%` | `0` | `TensorrtExecutionProvider` |
| `Fusion360 Gallery Assembly Full` | `fusion360_full` | `1048576` | `1024` | `262912` | `74.9268%` | `33.1845%` | `0` | `TensorrtExecutionProvider` |
| `common_modeling_high_density_scenarios_large` | `common_modeling` | `663552` | `192` | `110752` | `83.3092%` | `43.7862%` | `0` | `TensorrtExecutionProvider` |
| `rtstpf_advantage_cases_v4_large_training` | `advantage_v4` | `1990656` | `864` | `403889` | `79.7108%` | `53.8655%` | `0` | `TensorrtExecutionProvider` |
| `ShapeNetCore OOD dense/high-speed/thin-feature` | `shapenet_ood` | `1492992` | `648` | `622458` | `58.3080%` | `25.4147%` | `0` | `TensorrtExecutionProvider` |
| `ShapeNet car-wall dense wall patch` | `car_wall_impact` | `49152` | `6` | `12277` | `75.0224%` | `75.3239%` | `0` | `TensorrtExecutionProvider` |
| `common_daily_physics_collision_cases_run_id` | `aris_daily_physics` | `6656` | `13` | `4747` | `28.6809%` | `31.2454%` | `0` | `TensorrtExecutionProvider` |

## Five-path Rows

| Dataset | Method | Queries | Candidates | Exact calls | Call reduction | Work reduction | Wall ms | Proposal ms | Scheduling ms | FN | Recall | Scope |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `T0 synthetic_proxy` | `PureExactCPU` | `3360` | `430080` | `430080` | `0.0000%` | `0.0000%` | `20.874` | `0.000` | `20.874` | `0` | `1.000000` | `strict_candidate_row_replay_all_exact` |
| `T0 synthetic_proxy` | `BVHExact` | `3360` | `430080` | `430080` | `0.0000%` | `0.0000%` | `24.807` | `0.000` | `24.807` | `0` | `1.000000` | `strict_candidate_row_replay_all_exact` |
| `T0 synthetic_proxy` | `RTExact` | `3360` | `430080` | `430080` | `0.0000%` | `0.0000%` | `35.753` | `0.000` | `35.753` | `0` | `1.000000` | `strict_candidate_row_replay_all_exact` |
| `T0 synthetic_proxy` | `RTSTPFExact` | `3360` | `430080` | `79560` | `81.5011%` | `36.3898%` | `255.837` | `162.089` | `93.748` | `0` | `1.000000` | `strict_candidate_row_replay_ort_stpf_group_early_stop` |
| `T0 synthetic_proxy` | `NoProposal` | `3360` | `430080` | `430080` | `0.0000%` | `0.0000%` | `15.932` | `0.000` | `15.932` | `0` | `1.000000` | `strict_candidate_row_replay_all_exact` |
| `trained_stpf_high_density` | `PureExactCPU` | `3500` | `448000` | `448000` | `0.0000%` | `0.0000%` | `24.347` | `0.000` | `24.347` | `0` | `1.000000` | `strict_candidate_row_replay_all_exact` |
| `trained_stpf_high_density` | `BVHExact` | `3500` | `448000` | `448000` | `0.0000%` | `0.0000%` | `19.676` | `0.000` | `19.676` | `0` | `1.000000` | `strict_candidate_row_replay_all_exact` |
| `trained_stpf_high_density` | `RTExact` | `3500` | `448000` | `448000` | `0.0000%` | `0.0000%` | `31.128` | `0.000` | `31.128` | `0` | `1.000000` | `strict_candidate_row_replay_all_exact` |
| `trained_stpf_high_density` | `RTSTPFExact` | `3500` | `448000` | `92400` | `79.3750%` | `34.6769%` | `290.716` | `154.507` | `136.210` | `0` | `1.000000` | `strict_candidate_row_replay_ort_stpf_group_early_stop` |
| `trained_stpf_high_density` | `NoProposal` | `3500` | `448000` | `448000` | `0.0000%` | `0.0000%` | `14.319` | `0.000` | `14.319` | `0` | `1.000000` | `strict_candidate_row_replay_all_exact` |
| `ABC CAD` | `PureExactCPU` | `7200` | `921600` | `921600` | `0.0000%` | `0.0000%` | `38.529` | `0.000` | `38.529` | `0` | `1.000000` | `strict_candidate_row_replay_all_exact` |
| `ABC CAD` | `BVHExact` | `7200` | `921600` | `921600` | `0.0000%` | `0.0000%` | `33.586` | `0.000` | `33.586` | `0` | `1.000000` | `strict_candidate_row_replay_all_exact` |
| `ABC CAD` | `RTExact` | `7200` | `921600` | `921600` | `0.0000%` | `0.0000%` | `34.220` | `0.000` | `34.220` | `0` | `1.000000` | `strict_candidate_row_replay_all_exact` |
| `ABC CAD` | `RTSTPFExact` | `7200` | `921600` | `235800` | `74.4141%` | `49.5259%` | `414.763` | `221.690` | `193.073` | `0` | `1.000000` | `strict_candidate_row_replay_ort_stpf_group_early_stop` |
| `ABC CAD` | `NoProposal` | `7200` | `921600` | `921600` | `0.0000%` | `0.0000%` | `28.864` | `0.000` | `28.864` | `0` | `1.000000` | `strict_candidate_row_replay_all_exact` |
| `Thingi10K` | `PureExactCPU` | `4512` | `577536` | `577536` | `0.0000%` | `0.0000%` | `34.134` | `0.000` | `34.134` | `0` | `1.000000` | `strict_candidate_row_replay_all_exact` |
| `Thingi10K` | `BVHExact` | `4512` | `577536` | `577536` | `0.0000%` | `0.0000%` | `34.922` | `0.000` | `34.922` | `0` | `1.000000` | `strict_candidate_row_replay_all_exact` |
| `Thingi10K` | `RTExact` | `4512` | `577536` | `577536` | `0.0000%` | `0.0000%` | `23.058` | `0.000` | `23.058` | `0` | `1.000000` | `strict_candidate_row_replay_all_exact` |
| `Thingi10K` | `RTSTPFExact` | `4512` | `577536` | `147768` | `74.4141%` | `54.8694%` | `299.057` | `187.850` | `111.208` | `0` | `1.000000` | `strict_candidate_row_replay_ort_stpf_group_early_stop` |
| `Thingi10K` | `NoProposal` | `4512` | `577536` | `577536` | `0.0000%` | `0.0000%` | `17.213` | `0.000` | `17.213` | `0` | `1.000000` | `strict_candidate_row_replay_all_exact` |
| `Fusion 360 Gallery Assembly` | `PureExactCPU` | `1800` | `230400` | `230400` | `0.0000%` | `0.0000%` | `8.177` | `0.000` | `8.177` | `0` | `1.000000` | `strict_candidate_row_replay_all_exact` |
| `Fusion 360 Gallery Assembly` | `BVHExact` | `1800` | `230400` | `230400` | `0.0000%` | `0.0000%` | `11.569` | `0.000` | `11.569` | `0` | `1.000000` | `strict_candidate_row_replay_all_exact` |
| `Fusion 360 Gallery Assembly` | `RTExact` | `1800` | `230400` | `230400` | `0.0000%` | `0.0000%` | `8.531` | `0.000` | `8.531` | `0` | `1.000000` | `strict_candidate_row_replay_all_exact` |
| `Fusion 360 Gallery Assembly` | `RTSTPFExact` | `1800` | `230400` | `126133` | `45.2548%` | `19.7577%` | `128.606` | `75.165` | `53.442` | `0` | `1.000000` | `strict_candidate_row_replay_ort_stpf_group_early_stop` |
| `Fusion 360 Gallery Assembly` | `NoProposal` | `1800` | `230400` | `230400` | `0.0000%` | `0.0000%` | `7.768` | `0.000` | `7.768` | `0` | `1.000000` | `strict_candidate_row_replay_all_exact` |
| `high_density_mesh_multi_source` | `PureExactCPU` | `2700` | `345600` | `345600` | `0.0000%` | `0.0000%` | `27.847` | `0.000` | `27.847` | `0` | `1.000000` | `strict_candidate_row_replay_all_exact` |
| `high_density_mesh_multi_source` | `BVHExact` | `2700` | `345600` | `345600` | `0.0000%` | `0.0000%` | `13.821` | `0.000` | `13.821` | `0` | `1.000000` | `strict_candidate_row_replay_all_exact` |
| `high_density_mesh_multi_source` | `RTExact` | `2700` | `345600` | `345600` | `0.0000%` | `0.0000%` | `15.333` | `0.000` | `15.333` | `0` | `1.000000` | `strict_candidate_row_replay_all_exact` |
| `high_density_mesh_multi_source` | `RTSTPFExact` | `2700` | `345600` | `89314` | `74.1568%` | `56.3263%` | `192.743` | `113.948` | `78.796` | `0` | `1.000000` | `strict_candidate_row_replay_ort_stpf_group_early_stop` |
| `high_density_mesh_multi_source` | `NoProposal` | `2700` | `345600` | `345600` | `0.0000%` | `0.0000%` | `16.022` | `0.000` | `16.022` | `0` | `1.000000` | `strict_candidate_row_replay_all_exact` |
| `Fusion360 Gallery Assembly Full` | `PureExactCPU` | `1024` | `1048576` | `1048576` | `0.0000%` | `0.0000%` | `41.953` | `0.000` | `41.953` | `0` | `1.000000` | `strict_candidate_row_replay_all_exact` |
| `Fusion360 Gallery Assembly Full` | `BVHExact` | `1024` | `1048576` | `1048576` | `0.0000%` | `0.0000%` | `30.273` | `0.000` | `30.273` | `0` | `1.000000` | `strict_candidate_row_replay_all_exact` |
| `Fusion360 Gallery Assembly Full` | `RTExact` | `1024` | `1048576` | `1048576` | `0.0000%` | `0.0000%` | `38.496` | `0.000` | `38.496` | `0` | `1.000000` | `strict_candidate_row_replay_all_exact` |
| `Fusion360 Gallery Assembly Full` | `RTSTPFExact` | `1024` | `1048576` | `262912` | `74.9268%` | `33.1845%` | `417.715` | `262.863` | `154.853` | `0` | `1.000000` | `strict_candidate_row_replay_ort_stpf_group_early_stop` |
| `Fusion360 Gallery Assembly Full` | `NoProposal` | `1024` | `1048576` | `1048576` | `0.0000%` | `0.0000%` | `25.228` | `0.000` | `25.228` | `0` | `1.000000` | `strict_candidate_row_replay_all_exact` |
| `common_modeling_high_density_scenarios_large` | `PureExactCPU` | `192` | `663552` | `663552` | `0.0000%` | `0.0000%` | `17.481` | `0.000` | `17.481` | `0` | `1.000000` | `strict_candidate_row_replay_all_exact` |
| `common_modeling_high_density_scenarios_large` | `BVHExact` | `192` | `663552` | `663552` | `0.0000%` | `0.0000%` | `16.039` | `0.000` | `16.039` | `0` | `1.000000` | `strict_candidate_row_replay_all_exact` |
| `common_modeling_high_density_scenarios_large` | `RTExact` | `192` | `663552` | `663552` | `0.0000%` | `0.0000%` | `19.641` | `0.000` | `19.641` | `0` | `1.000000` | `strict_candidate_row_replay_all_exact` |
| `common_modeling_high_density_scenarios_large` | `RTSTPFExact` | `192` | `663552` | `110752` | `83.3092%` | `43.7862%` | `272.556` | `183.277` | `89.279` | `0` | `1.000000` | `strict_candidate_row_replay_ort_stpf_group_early_stop` |
| `common_modeling_high_density_scenarios_large` | `NoProposal` | `192` | `663552` | `663552` | `0.0000%` | `0.0000%` | `22.541` | `0.000` | `22.541` | `0` | `1.000000` | `strict_candidate_row_replay_all_exact` |
| `rtstpf_advantage_cases_v4_large_training` | `PureExactCPU` | `864` | `1990656` | `1990656` | `0.0000%` | `0.0000%` | `78.218` | `0.000` | `78.218` | `0` | `1.000000` | `strict_candidate_row_replay_all_exact` |
| `rtstpf_advantage_cases_v4_large_training` | `BVHExact` | `864` | `1990656` | `1990656` | `0.0000%` | `0.0000%` | `88.574` | `0.000` | `88.574` | `0` | `1.000000` | `strict_candidate_row_replay_all_exact` |
| `rtstpf_advantage_cases_v4_large_training` | `RTExact` | `864` | `1990656` | `1990656` | `0.0000%` | `0.0000%` | `67.805` | `0.000` | `67.805` | `0` | `1.000000` | `strict_candidate_row_replay_all_exact` |
| `rtstpf_advantage_cases_v4_large_training` | `RTSTPFExact` | `864` | `1990656` | `403889` | `79.7108%` | `53.8655%` | `665.913` | `416.210` | `249.702` | `0` | `1.000000` | `strict_candidate_row_replay_ort_stpf_group_early_stop` |
| `rtstpf_advantage_cases_v4_large_training` | `NoProposal` | `864` | `1990656` | `1990656` | `0.0000%` | `0.0000%` | `45.862` | `0.000` | `45.862` | `0` | `1.000000` | `strict_candidate_row_replay_all_exact` |
| `ShapeNetCore OOD dense/high-speed/thin-feature` | `PureExactCPU` | `648` | `1492992` | `1492992` | `0.0000%` | `0.0000%` | `50.477` | `0.000` | `50.477` | `0` | `1.000000` | `strict_candidate_row_replay_all_exact` |
| `ShapeNetCore OOD dense/high-speed/thin-feature` | `BVHExact` | `648` | `1492992` | `1492992` | `0.0000%` | `0.0000%` | `38.000` | `0.000` | `38.000` | `0` | `1.000000` | `strict_candidate_row_replay_all_exact` |
| `ShapeNetCore OOD dense/high-speed/thin-feature` | `RTExact` | `648` | `1492992` | `1492992` | `0.0000%` | `0.0000%` | `38.674` | `0.000` | `38.674` | `0` | `1.000000` | `strict_candidate_row_replay_all_exact` |
| `ShapeNetCore OOD dense/high-speed/thin-feature` | `RTSTPFExact` | `648` | `1492992` | `622458` | `58.3080%` | `25.4147%` | `740.174` | `561.983` | `178.191` | `0` | `1.000000` | `strict_candidate_row_replay_ort_stpf_group_early_stop` |
| `ShapeNetCore OOD dense/high-speed/thin-feature` | `NoProposal` | `648` | `1492992` | `1492992` | `0.0000%` | `0.0000%` | `40.250` | `0.000` | `40.250` | `0` | `1.000000` | `strict_candidate_row_replay_all_exact` |
| `ShapeNet car-wall dense wall patch` | `PureExactCPU` | `6` | `49152` | `49152` | `0.0000%` | `0.0000%` | `1.355` | `0.000` | `1.355` | `0` | `1.000000` | `strict_candidate_row_replay_all_exact` |
| `ShapeNet car-wall dense wall patch` | `BVHExact` | `6` | `49152` | `49152` | `0.0000%` | `0.0000%` | `1.065` | `0.000` | `1.065` | `0` | `1.000000` | `strict_candidate_row_replay_all_exact` |
| `ShapeNet car-wall dense wall patch` | `RTExact` | `6` | `49152` | `49152` | `0.0000%` | `0.0000%` | `1.064` | `0.000` | `1.064` | `0` | `1.000000` | `strict_candidate_row_replay_all_exact` |
| `ShapeNet car-wall dense wall patch` | `RTSTPFExact` | `6` | `49152` | `12277` | `75.0224%` | `75.3239%` | `63.343` | `52.063` | `11.280` | `0` | `1.000000` | `strict_candidate_row_replay_ort_stpf_group_early_stop` |
| `ShapeNet car-wall dense wall patch` | `NoProposal` | `6` | `49152` | `49152` | `0.0000%` | `0.0000%` | `1.024` | `0.000` | `1.024` | `0` | `1.000000` | `strict_candidate_row_replay_all_exact` |
| `common_daily_physics_collision_cases_run_id` | `PureExactCPU` | `13` | `6656` | `6656` | `0.0000%` | `0.0000%` | `0.217` | `0.000` | `0.217` | `0` | `1.000000` | `strict_candidate_row_replay_all_exact` |
| `common_daily_physics_collision_cases_run_id` | `BVHExact` | `13` | `6656` | `6656` | `0.0000%` | `0.0000%` | `0.110` | `0.000` | `0.110` | `0` | `1.000000` | `strict_candidate_row_replay_all_exact` |
| `common_daily_physics_collision_cases_run_id` | `RTExact` | `13` | `6656` | `6656` | `0.0000%` | `0.0000%` | `0.105` | `0.000` | `0.105` | `0` | `1.000000` | `strict_candidate_row_replay_all_exact` |
| `common_daily_physics_collision_cases_run_id` | `RTSTPFExact` | `13` | `6656` | `4747` | `28.6809%` | `31.2454%` | `31.305` | `29.634` | `1.671` | `0` | `1.000000` | `strict_candidate_row_replay_ort_stpf_group_early_stop` |
| `common_daily_physics_collision_cases_run_id` | `NoProposal` | `13` | `6656` | `6656` | `0.0000%` | `0.0000%` | `0.239` | `0.000` | `0.239` | `0` | `1.000000` | `strict_candidate_row_replay_all_exact` |

## Reproduce / Resume

```powershell
python src/tools/run_all_dataset_strict_five_path_replay.py --output-dir src/benchmark/all_dataset_strict_five_path_full_replay_run_id
```

Resume rule: one JSON file is written per dataset. Existing dataset JSON files are skipped unless `--force` is passed.
