# Tight-Inclusion / NYU Learned STPF Streaming training report

- Run name: `scalable_ccd_scene_supplementary_training_run_id`
- Shards: `src/datasets/training/scalable_ccd_scene_groups/shards/scalable_ccd_scene_supplementary_training_run_id`
- Model: `medium_mlp`
- Device: `cuda`
- Batch size: `4096`
- Epochs: `8`
- Optimizer: `AdamW`
- Loss: `multitask + cost-aware`
- Model state: `src/outputs/stpf_training/scalable_ccd_scene_supplementary_training_run_id/model_state.pt`
- ONNX: `src\outputs\stpf_training\scalable_ccd_scene_supplementary_training_run_id\model.onnx`
- ONNX export error: `None`
- Train split: `train`
- Validation split: `validation`

## Dataset

| Split | Rows | Eval cap | Positive ratio | Interval top1 | Family top2 | Estimated exact-work reduction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `train` | `266326` | `n/a` | `0.9950173847089657` | `0.995160066985574` | `1.0` | `0.48537825248558975` |
| `validation` | `355` | `n/a` | `0.37746478873239436` | `0.011267605633802818` | `1.0` | `0.4904466909704993` |

## Threshold Calibration

- Calibrated zero-FN threshold: `1.0164346005767584`

| Threshold | TP | TN | FP | FN | Recall | Exact-call reduction | Exact-work reduction |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0.0` | `134` | `221` | `0` | `0` | `1.0` | `0.0` | `0.0` |
| `0.25` | `134` | `221` | `0` | `0` | `1.0` | `0.0` | `0.0` |
| `0.5` | `134` | `221` | `0` | `0` | `1.0` | `0.0` | `0.0` |
| `0.75` | `134` | `221` | `0` | `0` | `1.0` | `0.0` | `0.0` |
| `0.9` | `134` | `221` | `0` | `0` | `1.0` | `0.0` | `0.0` |
| `1.0164346005767584` | `134` | `221` | `0` | `0` | `1.0` | `0.0` | `0.0` |

## Per-Kind Recall

| Kind | Rows | Positives | TP | FN | Recall | Exact-call rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ee` | `263` | `107` | `107` | `0` | `1.0` | `1.0` |
| `vf` | `92` | `27` | `27` | `0` | `1.0` | `1.0` |

## Per-Case Recall

| Case | Rows | Positives | TP | FN | Recall | Exact-call rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `cloth-funnel` | `355` | `134` | `134` | `0` | `1.0` | `1.0` |

## FN Risk Top-K

| Score | Case | Kind | CSV | Query |
| ---: | --- | --- | --- | ---: |
| `1.0164346005767584` | `cloth-funnel` | `ee` | `src/baseline/Sample-Scalable-CCD-Data/cloth-funnel/queries/227ee.csv` | `4000216` |
| `1.0167772565037012` | `cloth-funnel` | `ee` | `src/baseline/Sample-Scalable-CCD-Data/cloth-funnel/queries/227ee.csv` | `4000022` |
| `1.0173101294785738` | `cloth-funnel` | `vf` | `src/baseline/Sample-Scalable-CCD-Data/cloth-funnel/queries/227vf.csv` | `5000025` |
| `1.0180027559399605` | `cloth-funnel` | `vf` | `src/baseline/Sample-Scalable-CCD-Data/cloth-funnel/queries/227vf.csv` | `5000074` |
| `1.0180276278406382` | `cloth-funnel` | `vf` | `src/baseline/Sample-Scalable-CCD-Data/cloth-funnel/queries/227vf.csv` | `5000070` |
| `1.0180377885699272` | `cloth-funnel` | `vf` | `src/baseline/Sample-Scalable-CCD-Data/cloth-funnel/queries/227vf.csv` | `5000083` |
| `1.0182874742895365` | `cloth-funnel` | `vf` | `src/baseline/Sample-Scalable-CCD-Data/cloth-funnel/queries/227vf.csv` | `5000043` |
| `1.0183659791946411` | `cloth-funnel` | `vf` | `src/baseline/Sample-Scalable-CCD-Data/cloth-funnel/queries/227vf.csv` | `5000042` |
| `1.01844940520823` | `cloth-funnel` | `vf` | `src/baseline/Sample-Scalable-CCD-Data/cloth-funnel/queries/227vf.csv` | `5000050` |
| `1.0184508971869946` | `cloth-funnel` | `vf` | `src/baseline/Sample-Scalable-CCD-Data/cloth-funnel/queries/227vf.csv` | `5000044` |
| `1.0186199694871902` | `cloth-funnel` | `vf` | `src/baseline/Sample-Scalable-CCD-Data/cloth-funnel/queries/227vf.csv` | `5000020` |
| `1.0186869911849499` | `cloth-funnel` | `vf` | `src/baseline/Sample-Scalable-CCD-Data/cloth-funnel/queries/227vf.csv` | `5000013` |
| `1.0187024883925915` | `cloth-funnel` | `vf` | `src/baseline/Sample-Scalable-CCD-Data/cloth-funnel/queries/227vf.csv` | `5000009` |
| `1.0187593791633844` | `cloth-funnel` | `vf` | `src/baseline/Sample-Scalable-CCD-Data/cloth-funnel/queries/227vf.csv` | `5000004` |
| `1.0188069362193346` | `cloth-funnel` | `vf` | `src/baseline/Sample-Scalable-CCD-Data/cloth-funnel/queries/227vf.csv` | `5000005` |
| `1.0191582888364792` | `cloth-funnel` | `ee` | `src/baseline/Sample-Scalable-CCD-Data/cloth-funnel/queries/227ee.csv` | `4000258` |
| `1.0191758126020432` | `cloth-funnel` | `vf` | `src/baseline/Sample-Scalable-CCD-Data/cloth-funnel/queries/227vf.csv` | `5000065` |
| `1.0191897060722113` | `cloth-funnel` | `ee` | `src/baseline/Sample-Scalable-CCD-Data/cloth-funnel/queries/227ee.csv` | `4000257` |
| `1.0192201863974333` | `cloth-funnel` | `ee` | `src/baseline/Sample-Scalable-CCD-Data/cloth-funnel/queries/227ee.csv` | `4000236` |
| `1.0192229747772217` | `cloth-funnel` | `vf` | `src/baseline/Sample-Scalable-CCD-Data/cloth-funnel/queries/227vf.csv` | `5000078` |
