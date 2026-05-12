# Scalable-CCD sample converted to P2C scene-level candidate groups

Run identifier: `run_id 04:12:17 UTC`

## Scope

- descriptionreport `Sample-Scalable-CCD-Data`  full-scene time-step query files convertas P2C candidate-row groups.
- Group granularity is `scene + timestep + primitive kind`, descriptionscenewhendescriptionall EE or VF primitive queries asdescription P2C candidate group.
- descriptioncompare Scalable-CCD native kernel time, description scene simulator runtime and P2C candidate-row replay runtime descriptioninsamedescription.
- `mma_bool` as ground truth label; `boxes` asoriginal broad-phase intersecting box ids; query CSV  rational coordinates generate 32 description STPF-compatible feature rows.

## Outputs

- Combined shard: `src/datasets/training/scalable_ccd_scene_groups/shards/scalable_ccd_sample_scene_candidate_groups_run_id/scene_eval.npz`
- Per-group shards: `src/datasets/training/scalable_ccd_scene_groups/shards/scalable_ccd_sample_scene_candidate_groups_run_id/groups`
- Manifest: `src/datasets/training/scalable_ccd_scene_groups/shards/scalable_ccd_sample_scene_candidate_groups_run_id/manifest.json`
- Summary CSV: `src/benchmark/scalable_ccd_sample_scene_candidate_groups_run_id/scalable_ccd_sample_scene_candidate_groups_run_id.csv`
- Summary JSON: `src/benchmark/scalable_ccd_sample_scene_candidate_groups_run_id/scalable_ccd_sample_scene_candidate_groups_run_id.json`

## Overall

| Metric | Value |
| --- | --- |
| source scenes | 6 |
| scene-step-kind groups | 12 |
| candidate rows | 271852 |
| positive rows | 270077 |
| negative rows | 1775 |
| positive groups | 12 |
| heuristic schedule exact calls | 14 |
| heuristic call reduction | 99.995% |
| random expected call reduction | 99.994% |
| oracle lower-bound call reduction | 99.996% |

## Per Scene-step-kind Group

| Scene | Kind | Step | Candidates | Positive | Positive % | Heuristic first rank | Heuristic call red. | Random expected red. | Oracle red. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| armadillo-rollers | EE | 326 | 19313 | 19308 | 99.974% | 2 | 99.990% | 99.995% | 99.995% |
| armadillo-rollers | VF | 326 | 4652 | 4650 | 99.957% | 1 | 99.979% | 99.978% | 99.979% |
| cloth-ball | EE | 92 | 94825 | 94822 | 99.997% | 1 | 99.999% | 99.999% | 99.999% |
| cloth-ball | VF | 92 | 19034 | 19032 | 99.989% | 2 | 99.989% | 99.995% | 99.995% |
| cloth-funnel | EE | 227 | 263 | 107 | 40.684% | 1 | 99.620% | 99.071% | 99.620% |
| cloth-funnel | VF | 227 | 92 | 27 | 29.348% | 1 | 98.913% | 96.390% | 98.913% |
| n-body-simulation | EE | 18 | 41036 | 41031 | 99.988% | 1 | 99.998% | 99.998% | 99.998% |
| n-body-simulation | VF | 18 | 9460 | 9459 | 99.989% | 1 | 99.989% | 99.989% | 99.989% |
| puffer-ball | EE | 20 | 61844 | 60894 | 98.464% | 1 | 99.998% | 99.998% | 99.998% |
| puffer-ball | VF | 20 | 16162 | 15803 | 97.779% | 1 | 99.994% | 99.994% | 99.994% |
| rod-twist | EE | 3036 | 4370 | 4179 | 95.629% | 1 | 99.977% | 99.976% | 99.977% |
| rod-twist | VF | 3036 | 801 | 765 | 95.506% | 1 | 99.875% | 99.869% | 99.875% |

## Interpretation

- this supplementary descriptionInputis converted candidate groups, rather than Scalable-CCD original pipeline time.
- this sample description scene-step-kind group  positive fraction descriptionhigh, therefore early-stop reduction descriptionanddescription; descriptionas learned scheduling better than SOTA description.
- `heuristic` descriptionis motion/proximity feature  sanity diagnostic, usedescription converted scene groups has scheduling pressure; descriptionis notthis paperdescription SOTA Methoddescription.
- description TOG writedescription, description supplementary: descriptionthis paperdescriptionconnectdescription Scalable-CCD full-scene data source, descriptionby scene time-step description P2C candidate groups.
- ifdescriptionenterdescription, underdescriptionindescription converted groups onfixed STPF checkpoint / validation-selected schedule, descriptionusesame exact certificate policy perform replay.

## Reproduce

```powershell
python src\tools\convert_scalable_ccd_sample_to_p2c_groups.py
```
