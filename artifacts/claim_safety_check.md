# Claim Safety Check

## Safe Claims

| Claim | Status | Evidence path | Notes |
| --- | --- | --- | --- |
| P2C-CCD is a proposal-to-certificate pipeline: conservative candidates, learned scheduling, exact certificate/fallback. | SAFE_STRONG | `src/docs/architecture.md`, `src/README.md` | Neural scheduling does not replace the final exact certificate/fallback. |
| Dense/high-cost candidate groups show strong exact-work reduction with `FN=0`. | SAFE_STRONG | `src/benchmark/complete_benchmark_vs_baselines_run_id.md`, `multi_dense_mesh_contact_pairs_run_id.md`, `large_dense_complex_mesh_cases_run_id.md` | Main performance claim. |
| All adapter-ready datasets have strict five-path candidate-row replay with `FN=0`. | SAFE_STRONG | `src/benchmark/all_dataset_strict_five_path_full_replay_run_id/all_dataset_strict_five_path_full_replay_run_id.md` | Must state candidate-row replay timing scope. |
| Tight-Inclusion/NYU 100GB every-candidate native wall-time baseline is complete. | SAFE_STRONG | `src/benchmark/ti_full_query_every_candidate_walltime_run_id/ti_full_query_every_candidate_walltime_run_id.md` | Baseline table, not RTSTPF speedup claim. |
| Native dense group scheduling reduces proposal overhead relative to Python/PyTorch. | SAFE_CONDITIONAL | `src/benchmark/common_modeling_high_density_scenarios_large_run_id_compiled_ort_tensorrt_summary.md`, `native_dense_group_walltime_head_selected_run_id.md` | Be explicit about exact-driver scope. |
| Learned scheduling can beat random in selected hard-negative groups. | SAFE_CONDITIONAL | `src/benchmark/learned_vs_random_ablation_head_selected_run_id.md`, `tight_inclusion_dense_group_real_exact_run_id.md` | Dataset-dependent; not universal. |
| Generalization across ABC/Fusion360/Thingi10K/ShapeNet/common modeling is supported. | SAFE_CONDITIONAL | `src/benchmark/aris_p0e_generalization_selected_real_matrix_run_id.md`, `shapenet_ood_dense_cases_run_id.md` | Some matrix cells are consolidation, not exhaustive source-pair real runs. |
| Visualizations are paper-grade case studies. | SAFE_CONDITIONAL | `assets/figures/cases/`, `src/benchmark/` | Visualizations do not prove correctness; cite benchmark reports for correctness. |

## Claims to Avoid

| Prohibited wording | Status | Reason |
| --- | --- | --- |
| "RTSTPFExact is faster on all CCD queries." | DO_NOT_CLAIM | Sparse primitive TI benchmark shows fallback and overhead can dominate. |
| "The neural network certifies collision or separation." | DO_NOT_CLAIM | Final correctness comes from exact certificate/fallback. |
| "RT cores perform exact CCD certificate." | DO_NOT_CLAIM | RT/OptiX only generate or traverse conservative candidates. |
| "All source-pair generalization matrix entries are exhaustive real full-runs." | DO_NOT_CLAIM | Matrix includes selected-real and consolidation entries. |
| "All historical benchmark candidates have full every-candidate certificate replay audit." | DO_NOT_CLAIM | Main dependencies are audited; full historical replay remains extensible. |
| "Learned STPF always beats random scheduling." | DO_NOT_CLAIM | Learned-vs-random advantage is dataset-dependent. |
| "All-exact wall time in strict replay is native primitive exact kernel time." | DO_NOT_CLAIM | It is candidate-row replay scan overhead. |

## Release Notes

- The public release keeps claim boundaries aligned with the bundled benchmark reports.
- Sparse primitive Tight-Inclusion is still a boundary case, not a universal speedup claim.
- Local absolute paths were removed from the public release materials.
- Release-local figures and benchmark evidence are indexed by `artifacts/release_case_manifest.json`.
