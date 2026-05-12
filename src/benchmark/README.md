# Curated Benchmark Evidence

This repository snapshot includes a curated subset of benchmark outputs.
They are meant to demonstrate the bundled evidence path without shipping the
full local benchmark cache.

## Included Directories

- `all_dataset_strict_five_path_full_replay_run_id`
- `ti_full_query_every_candidate_walltime_run_id`
- `p1_ablation_suite_run_id`
- `p2_ablation_suite_run_id`
- `native_ti_heldout_dense_group_run_id`
- `scene_object_envelope_strong_native_run_id`

## Omitted On Purpose

- temporary probe runs
- feature-export scratch directories
- large visual media and local convenience caches
- the full 3.8 GB benchmark tree from the author workspace

The bundled reports are enough to inspect result schemas, public summary
tables, and the exact report files referenced by the release README. For larger
reproduction, rerun the corresponding scripts under `../tools/` with the data
and baseline roots documented in `../datasets/README.md` and
`../baseline/README.md`.
