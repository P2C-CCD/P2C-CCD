# Tight-Inclusion / NYU Learned STPF Streaming training report

- Run name: `standard_graphics_models_run_id`
- Shards: `src/datasets/training/standard_graphics_models_run_id/shards`
- Model: `medium_mlp`
- Device: `cuda`
- Batch size: `8192`
- Epochs: `6`
- Optimizer: `AdamW`
- Loss: `multitask + cost-aware`
- Model state: `src/outputs/stpf_training/standard_graphics_models_run_id/standard_graphics_models_run_id/model_state.pt`
- ONNX: `src\outputs\stpf_training\standard_graphics_models_run_id\standard_graphics_models_run_id\model.onnx`
- ONNX export error: `None`
- Train split: `train`
- Validation split: `validation`

## Dataset

| Split | Rows | Eval cap | Positive ratio | Interval top1 | Family top2 | Estimated exact-work reduction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `train` | `16384` | `n/a` | `0.1243896484375` | `0.8756103515625` | `0.25` | `0.6205157702590542` |
| `validation` | `8192` | `n/a` | `0.0675048828125` | `0.9324951171875` | `0.25` | `0.6223769105419645` |

## Threshold Calibration

- Calibrated zero-FN threshold: `0.4731125012040138`

| Threshold | TP | TN | FP | FN | Recall | Exact-call reduction | Exact-work reduction |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0.0` | `553` | `7639` | `0` | `0` | `1.0` | `0.0` | `0.0` |
| `0.25` | `553` | `7639` | `0` | `0` | `1.0` | `0.0` | `0.0` |
| `0.4731125012040138` | `553` | `7639` | `0` | `0` | `1.0` | `0.0147705078125` | `0.016829186239856075` |
| `0.5` | `0` | `7639` | `0` | `553` | `0.0` | `1.0` | `1.0` |
| `0.75` | `0` | `7639` | `0` | `553` | `0.0` | `1.0` | `1.0` |
| `0.9` | `0` | `7639` | `0` | `553` | `0.0` | `1.0` | `1.0` |

## Per-Kind Recall

| Kind | Rows | Positives | TP | FN | Recall | Exact-call rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `object-ground` | `7168` | `423` | `423` | `0` | `1.0` | `0.9831194196428571` |
| `scene-container` | `1024` | `130` | `130` | `0` | `1.0` | `1.0` |

## Per-Case Recall

| Case | Rows | Positives | TP | FN | Recall | Exact-call rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `armadillo_frictional_floor_drop` | `1024` | `64` | `64` | `0` | `1.0` | `1.0` |
| `bunny_frictional_floor_drop` | `1024` | `64` | `64` | `0` | `1.0` | `1.0` |
| `cornell_box_multimodel_scene_drop` | `1024` | `130` | `130` | `0` | `1.0` | `1.0` |
| `dragon_frictional_floor_drop` | `1024` | `64` | `64` | `0` | `1.0` | `0.8818359375` |
| `fandisk_frictional_floor_drop` | `1024` | `64` | `64` | `0` | `1.0` | `1.0` |
| `spot_frictional_floor_drop` | `1024` | `64` | `64` | `0` | `1.0` | `1.0` |
| `suzanne_frictional_floor_drop` | `1024` | `39` | `39` | `0` | `1.0` | `1.0` |
| `teapot_frictional_floor_drop` | `1024` | `64` | `64` | `0` | `1.0` | `1.0` |

## FN Risk Top-K

| Score | Case | Kind | CSV | Query |
| ---: | --- | --- | --- | ---: |
| `0.4731125012040138` | `dragon_frictional_floor_drop` | `object-ground` | `validation/dragon_frictional_floor_drop.csv` | `fixed_seed` |
| `0.4731292948126793` | `dragon_frictional_floor_drop` | `object-ground` | `validation/dragon_frictional_floor_drop.csv` | `fixed_seed` |
| `0.47323621809482574` | `dragon_frictional_floor_drop` | `object-ground` | `validation/dragon_frictional_floor_drop.csv` | `fixed_seed` |
| `0.47335805743932724` | `dragon_frictional_floor_drop` | `object-ground` | `validation/dragon_frictional_floor_drop.csv` | `fixed_seed` |
| `0.473381944000721` | `dragon_frictional_floor_drop` | `object-ground` | `validation/dragon_frictional_floor_drop.csv` | `fixed_seed` |
| `0.4733966588973999` | `dragon_frictional_floor_drop` | `object-ground` | `validation/dragon_frictional_floor_drop.csv` | `fixed_seed` |
| `0.4734080135822296` | `dragon_frictional_floor_drop` | `object-ground` | `validation/dragon_frictional_floor_drop.csv` | `fixed_seed` |
| `0.47345370054244995` | `dragon_frictional_floor_drop` | `object-ground` | `validation/dragon_frictional_floor_drop.csv` | `fixed_seed` |
| `0.47346191108226776` | `dragon_frictional_floor_drop` | `object-ground` | `validation/dragon_frictional_floor_drop.csv` | `fixed_seed` |
| `0.4735364094376564` | `dragon_frictional_floor_drop` | `object-ground` | `validation/dragon_frictional_floor_drop.csv` | `fixed_seed` |
| `0.47356001287698746` | `dragon_frictional_floor_drop` | `object-ground` | `validation/dragon_frictional_floor_drop.csv` | `fixed_seed` |
| `0.473581962287426` | `dragon_frictional_floor_drop` | `object-ground` | `validation/dragon_frictional_floor_drop.csv` | `fixed_seed` |
| `0.4736037030816078` | `dragon_frictional_floor_drop` | `object-ground` | `validation/dragon_frictional_floor_drop.csv` | `fixed_seed` |
| `0.47363457828760147` | `dragon_frictional_floor_drop` | `object-ground` | `validation/dragon_frictional_floor_drop.csv` | `fixed_seed` |
| `0.4736439436674118` | `dragon_frictional_floor_drop` | `object-ground` | `validation/dragon_frictional_floor_drop.csv` | `fixed_seed` |
| `0.47364672273397446` | `dragon_frictional_floor_drop` | `object-ground` | `validation/dragon_frictional_floor_drop.csv` | `fixed_seed` |
| `0.47364939749240875` | `dragon_frictional_floor_drop` | `object-ground` | `validation/dragon_frictional_floor_drop.csv` | `fixed_seed` |
| `0.4736786112189293` | `dragon_frictional_floor_drop` | `object-ground` | `validation/dragon_frictional_floor_drop.csv` | `fixed_seed` |
| `0.47368670254945755` | `dragon_frictional_floor_drop` | `object-ground` | `validation/dragon_frictional_floor_drop.csv` | `fixed_seed` |
| `0.4737110435962677` | `dragon_frictional_floor_drop` | `object-ground` | `validation/dragon_frictional_floor_drop.csv` | `fixed_seed` |
