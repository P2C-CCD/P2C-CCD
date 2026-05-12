# Scalable-CCD Scene-level Supplementary Training Benchmark

Run identifier: `run_id 18:31:01 UTC`

## Scope

- Benchmark Inputisdescription Sample-Scalable-CCD-Data convertdescriptionto P2C candidate groups.
- descriptionand Scalable-CCD native kernel / simulator wall time direct comparison.
- description sample scenes descriptionsplit candidate descriptionis positive, group early-stop descriptionconnectdescription scene-level data source, descriptionas learned scheduling description SOTA description.

## Paths

- Split shards: `src/datasets/training/scalable_ccd_scene_groups/shards/scalable_ccd_scene_supplementary_training_run_id`
- Training report: `src/benchmark/scalable_ccd_scene_supplementary_training_run_id/scalable_ccd_scene_supplementary_training_run_id_training.md`
- Model state: `src/outputs/stpf_training/scalable_ccd_scene_supplementary_training_run_id/model_state.pt`
- Visualization: `src/MyDemo/scalable_ccd_scene_supplementary_training_run_id`

## Results

| Split | Method | Groups | Candidates | Positives | Exact calls | Call reduction | Work reduction | First-positive rank | FN |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | NoProposalAllExact | 8 | 266326 | 264999 | 266326 | 0.000% | 0.000% | 0.000 | 0 |
| train | TrainedSTPFGroupEarlyStop | 8 | 266326 | 264999 | 8 | 99.997% | 99.997% | 1.000 | 0 |
| train | HeuristicProximityGroupEarlyStop | 8 | 266326 | 264999 | 10 | 99.996% | 99.997% | 1.250 | 0 |
| train | RandomUniformExpectedOneSeed | 8 | 266326 | 264999 | 8 | 99.997% | 99.997% | 1.000 | 0 |
| train | OraclePositiveFirst | 8 | 266326 | 264999 | 8 | 99.997% | 99.997% | 1.000 | 0 |
| validation | NoProposalAllExact | 2 | 355 | 134 | 355 | 0.000% | 0.000% | 0.000 | 0 |
| validation | TrainedSTPFGroupEarlyStop | 2 | 355 | 134 | 2 | 99.437% | 99.465% | 1.000 | 0 |
| validation | HeuristicProximityGroupEarlyStop | 2 | 355 | 134 | 2 | 99.437% | 99.465% | 1.000 | 0 |
| validation | RandomUniformExpectedOneSeed | 2 | 355 | 134 | 7 | 98.028% | 98.039% | 3.500 | 0 |
| validation | OraclePositiveFirst | 2 | 355 | 134 | 2 | 99.437% | 99.465% | 1.000 | 0 |
| heldout_test | NoProposalAllExact | 2 | 5171 | 4944 | 5171 | 0.000% | 0.000% | 0.000 | 0 |
| heldout_test | TrainedSTPFGroupEarlyStop | 2 | 5171 | 4944 | 2 | 99.961% | 99.964% | 1.000 | 0 |
| heldout_test | HeuristicProximityGroupEarlyStop | 2 | 5171 | 4944 | 2 | 99.961% | 99.964% | 1.000 | 0 |
| heldout_test | RandomUniformExpectedOneSeed | 2 | 5171 | 4944 | 2 | 99.961% | 99.964% | 1.000 | 0 |
| heldout_test | OraclePositiveFirst | 2 | 5171 | 4944 | 2 | 99.961% | 99.964% | 1.000 | 0 |
| all_scene | NoProposalAllExact | 12 | 271852 | 270077 | 271852 | 0.000% | 0.000% | 0.000 | 0 |
| all_scene | TrainedSTPFGroupEarlyStop | 12 | 271852 | 270077 | 12 | 99.996% | 99.996% | 1.000 | 0 |
| all_scene | HeuristicProximityGroupEarlyStop | 12 | 271852 | 270077 | 14 | 99.995% | 99.995% | 1.167 | 0 |
| all_scene | RandomUniformExpectedOneSeed | 12 | 271852 | 270077 | 15 | 99.994% | 99.995% | 1.250 | 0 |
| all_scene | OraclePositiveFirst | 12 | 271852 | 270077 | 12 | 99.996% | 99.996% | 1.000 | 0 |

## Scope Note

This supplementary should be cited as a scene-data conversion and P2C replay compatibility result. It should not be worded as P2C-CCD beating Scalable-CCD native runtime.
