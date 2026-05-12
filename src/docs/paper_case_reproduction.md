# Paper Case Reproduction From This Release

This document maps paper-level visual cases and result figures to paths bundled
inside the public GitHub repository tree. The mapping is self-contained within
the public release.

## One-command bundle check

Run from the release repository root:

```powershell
conda activate cudadev
python scripts\verify_release_cases.py
```

The script checks `artifacts/release_case_manifest.json` and verifies that every
listed figure, report, evidence file, directory, and rerun entry point is
release-local and non-empty.

## Case matrix

| Paper item | Release-local figures | Release-local evidence / runners |
| --- | --- | --- |
| Teaser and method overview | `assets/figures/main/First_fig.jpg`, `assets/figures/main/STPF.jpg` | `src/docs/architecture.md`, `src/docs/reproducibility_quickstart.md` |
| Primary scope results | `assets/figures/results/result_primary_scope_combined.png` | `src/benchmark/all_dataset_strict_five_path_full_replay_run_id/`, `src/benchmark/ti_full_query_every_candidate_walltime_run_id/`, `src/tools/run_all_dataset_strict_five_path_replay.py` |
| Native end-to-end and ablations | `assets/figures/results/result_native_e2e_ablation_combined_column.png` | `src/benchmark/native_end_to_end_dense_run_id/`, `src/benchmark/p1_ablation_suite_run_id/`, `src/benchmark/p2_ablation_suite_run_id/` |
| Held-out native TI | `assets/figures/results/result_heldout_native_ti.png` | `src/benchmark/native_ti_heldout_dense_group_run_id/`, `src/tools/native_ti_heldout_dense_group_benchmark.py` |
| Motion-trap scene-object envelope | `assets/figures/results/result_motion_trap_native_envelope.png` | `src/benchmark/scene_object_envelope_*`, `src/tools/run_scene_object_envelope_native_ti_walltime.py` |
| Standard mesh-drop models | `assets/figures/cases/base_model_analysis.jpg` | `src/benchmark/standard_graphics_models_run_id/standard_graphics_model_suite_summary.md`, `src/tools/standard_graphics_model_collision_suite.py` |
| Twisting towel | `assets/figures/cases/05_twisting_towel.png` | `src/benchmark/standard_graphics_models_run_id/twisting_towel_wringer_ccd_metrics.json`, `src/tools/twisting_towel_ccd_case.py` |
| Particle sphere bowl | `assets/figures/cases/traj_and_reduction.jpg` | `src/benchmark/particle_sphere_bowl_explosion_run_id.json`, `src/tools/particle_sphere_bowl_explosion_case.py` |
| Car-wall impact | `assets/figures/cases/Car_wall.jpg` | `src/benchmark/car_wall_impact_dense_wall_patch_rtstpf_training_benchmark_run_id.json`, `src/tools/generate_car_wall_analysis_curves.py` |
| Ball-cloth and Scalable-CCD gallery | `assets/figures/cases/ball-cloth.jpg`, `assets/figures/cases/scalable_ccd_six_case_gallery_tight.jpg` | `src/benchmark/scalable_ccd_sample_scene_candidate_groups_run_id/`, `src/benchmark/scalable_ccd_scene_supplementary_training_run_id/`, `src/tools/run_scalable_ccd_scene_supplementary.py` |
| ABC real mesh contact | `assets/figures/cases/04_abc_real_mesh_contact.png` | `src/benchmark/object_object_dense_mesh_contact_run_id/`, `src/tools/object_object_dense_mesh_contact_case.py` |
| Repeated sphere-funnel contact | `assets/figures/cases/15_repeated_sphere_funnel_drop.png` | `src/benchmark/section3_full_rerun_run_id/`, `src/tools/repeated_contact_case_suite_run_id.py` |

## Rerun boundary

The release bundles the paper-facing images, summary evidence, manifests, and
runnable entry points needed to inspect the reported cases from this tree. Small
CPU smoke tests and manifest checks run directly from the release. Large GPU
sweeps, full external-baseline reruns, and raw third-party datasets remain
optional reruns; their mount points and license constraints are documented in
`src/datasets/README.md`, `src/baseline/README.md`, and
`src/docs/third_party_manifest.md`.
