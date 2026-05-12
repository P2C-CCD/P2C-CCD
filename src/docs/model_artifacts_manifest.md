# Model And Data Artifact Manifest

Generated: release artifact bundle.

This manifest documents bundled compact checkpoints and evaluation shards used
by the paper-critical strict five-path replay and dense replay evidence:

```text
src/tools/run_all_dataset_strict_five_path_replay.py
```

These compact artifacts are included in the anonymous code-and-data package so
reviewers can run the release-local checks and inspect the model inputs used by
the reported replay evidence. Full raw datasets and full-scale regenerated
training products remain external; a complete rerun should either verify the
hashes below or rerun the corresponding data-generation/training pipeline.

Commands below assume the repository root as the current directory and:

```powershell
conda activate cudadev
```

## Provenance / Regeneration

| Artifact group | generated_by | regen_command | download_or_regen | expected_cost |
| --- | --- | --- | --- | --- |
| `generalization_paper_benchmark_full_run_id` | `p2cccd.bench.generalization_paper_benchmark` | `python -c "from p2cccd.bench.generalization_paper_benchmark import GeneralizationPaperBenchmarkConfig, run_generalization_paper_benchmark; run_generalization_paper_benchmark(GeneralizationPaperBenchmarkConfig(run_name='generalization_paper_benchmark_full_run_id'))"` | Restore exact hashes from an artifact bundle if available; otherwise regenerate from local source datasets. | GPU training plus multi-source dense/query replay; hours on a CUDA workstation. |
| `fusion360_full_large_training_run_id` | `p2cccd.bench.fusion360_full_large_training_benchmark` | `python -m p2cccd.bench.fusion360_full_large_training_benchmark` | Requires restored Fusion360 Gallery extraction or equivalent local dataset root. | GPU training plus dense evaluation; hours depending on dataset storage speed. |
| `common_modeling_high_density_scenarios_large_run_id` | `p2cccd.bench.common_modeling_high_density_scenarios` + `p2cccd.bench.tight_inclusion_stpf_training` | `python -m p2cccd.bench.common_modeling_high_density_scenarios; python -m p2cccd.bench.tight_inclusion_stpf_training --shards src\datasets\training\common_modeling_high_density\shards\common_modeling_high_density_scenarios_large_run_id --run-name common_modeling_high_density_scenarios_large_run_id_medium_mlp_10epoch --report-name common_modeling_high_density_scenarios_large_run_id_medium_mlp_10epoch --epochs 10 --batch-size 65536 --device cuda` | Prefer restoring exact hashes; regeneration may require matching the large high-density scenario config. | GPU training; minutes to hours. |
| `rtstpf_advantage_cases_v4_large_training_run_id` | `p2cccd.bench.rtstpf_advantage_cases_v4_large_training` | `python -m p2cccd.bench.rtstpf_advantage_cases_v4_large_training` | Requires the Fusion360/full and generated advantage-case inputs present locally. | GPU training plus ORT/TensorRT probe when configured; hours. |
| `shapenet_ood_dense_cases_run_id` | `p2cccd.bench.shapenet_ood_dense_cases` | `python -m p2cccd.bench.shapenet_ood_dense_cases` | Requires restored ShapeNetCore-derived local assets and the upstream advantage-case checkpoint. | GPU training/evaluation plus local mesh preprocessing; hours. |
| `car_wall_impact_dense_wall_patch_rtstpf_training_benchmark_run_id` | `p2cccd.bench.car_wall_impact_training_benchmark` | `python -m p2cccd.bench.car_wall_impact_training_benchmark --train-rows 196608 --validation-rows 49152 --heldout-rows 49152 --epochs 10 --batch-size 16384` | Can be regenerated from the analytic car-wall case and local generated visualization inputs. | GPU training; typically shorter than full multi-dataset runs. |
| `common_daily_physics_collision_cases_run_id` | `internal daily-physics generator omitted from public release` | `restore matching shard by hash or regenerate from an equivalent local daily-physics generator` | Restore exact shard hashes when available; otherwise regenerate from a local daily-physics source with the same schema. | CPU/GPU smoke-to-full cost depends on the local generator and hardware. |

## Checkpoints

| Path | Bytes | SHA-256 |
| --- | ---: | --- |
| `src/outputs/stpf_training/generalization_paper_benchmark_full_run_id/model_state.pt` | 849,869 | `dce75e7887c17da104aa3fc1d6c09503f691a86fc38d410997cb7cd4037ef713` |
| `src/outputs/stpf_training/fusion360_full_large_training_run_id/model_state.pt` | 849,933 | `47234558402eb9bcec70abc4e2758812000542122a06a5fab81bc99775016fa1` |
| `src/outputs/stpf_training/common_modeling_high_density_scenarios_large_run_id_medium_mlp_10epoch/model_state.pt` | 849,869 | `f12c7f3d96f4a2be014728f0e53f0eb2303f0990aff48a295b377eb677697eb4` |
| `src/outputs/stpf_training/rtstpf_advantage_cases_v4_large_training_run_id/model_state.pt` | 849,933 | `0a2b20e537d8b243cda776d9891ce800d2a2f8ac44a8f5cbc3c0a0a1ba2887da` |
| `src/outputs/stpf_training/shapenet_ood_dense_cases_run_id/model_state.pt` | 849,933 | `e0917266340571960442ad2ab7223dda8fd7e59d788f44858c1b13e0ae1126e7` |
| `src/outputs/stpf_training/car_wall_impact_dense_wall_patch_rtstpf_training_benchmark_run_id/model_state.pt` | 849,197 | `58aa49d0a681fde727d54a05ce7aa571ac810f55ae50a7b042a30c88c8e27615` |

## Training / Evaluation Shards

| Path | Bytes | SHA-256 |
| --- | ---: | --- |
| `src/datasets/training/generalization/shards/generalization_paper_benchmark_full_run_id/t0_synthetic_proxy/dense_eval.npz` | 4,831,647 | `a727ea7bdad23d36802e7c3fde987bca9a8598af66479ff6e70a8d0ae0e2d12f` |
| `src/datasets/training/generalization/shards/generalization_paper_benchmark_full_run_id/trained_stpf_high_density/dense_eval.npz` | 5,025,319 | `4b3bda7dbd89ecf96e7c6411718241a7dd17285c5fbe6338ece573ea8e964d84` |
| `src/datasets/training/generalization/shards/generalization_paper_benchmark_full_run_id/abc_cad/dense_eval.npz` | 10,335,111 | `090a70995e1e83d767e60078995c9e7dbb8ffd2edb52d73cfa99a72c62098e15` |
| `src/datasets/training/generalization/shards/generalization_paper_benchmark_full_run_id/thingi10k/dense_eval.npz` | 6,490,112 | `d48a57587ad6ddcbbb3c994d65b73104c96803655482256232a4b5fb65e15c65` |
| `src/datasets/training/generalization/shards/generalization_paper_benchmark_full_run_id/fusion_360_gallery_assembly/dense_eval.npz` | 2,600,150 | `f70df250cc65eed0104a72d657651861ef96be9c3029c2f06c2ccb9d2c8eaf7e` |
| `src/datasets/training/generalization/shards/generalization_paper_benchmark_full_run_id/high_density_mesh_multi_source/dense_eval.npz` | 3,964,357 | `d220bc4a2b1a7d3a2fd28a1320fee32053f88a76c97f1b62ab3f6faa4564ee1e` |
| `src/datasets/training/fusion360_full/shards/fusion360_full_large_training_run_id/dense_eval.npz` | 10,181,038 | `d173c52531806cffd683f23ed75255f16e33cd182d87d74cc6dd2ab7a3307960` |
| `src/datasets/training/common_modeling_high_density/shards/common_modeling_high_density_scenarios_large_run_id/dense_eval.npz` | 6,175,609 | `783f694e54a6880c75fcd2476f6b2510e8d7b1122ce4782c5515c8cd46bca539` |
| `src/datasets/training/rtstpf_advantage_cases_v4/shards/rtstpf_advantage_cases_v4_large_training_run_id/dense_eval.npz` | 18,553,927 | `b02f3ad6d3e4973f7362e8239f49c77dbfa3f66193895e14cd918d37918a24d5` |
| `src/datasets/training/shapenet_ood_dense_cases/shards/shapenet_ood_dense_cases_run_id/dense_eval.npz` | 14,019,798 | `b35c04d81ed57a5bb869c3885eb88013886a60b7dc99af1195301ad1f1d68620` |
| `src/datasets/training/car_wall_impact_rtstpf/shards/car_wall_impact_dense_wall_patch_rtstpf_training_benchmark_run_id/heldout.npz` | 4,782,088 | `61d5ecd8bd6fa27fe9beacb66abcce88df2783f073cc8e14ef8c9cc0d4052e30` |
| `src/datasets/training/aris_ccf_a_expansion_run_id/common_daily_physics_collision_cases/shards/common_daily_physics_collision_cases_run_id/dense_eval_full.npz` | 604,925 | `8e02f8686c7127e966a78e3d3375a14a48aecbb482d3a06bcc8f504af9bac421` |

## Verification Command

```powershell
Get-FileHash -Algorithm SHA256 -LiteralPath <path>
```

## Clean-checkout Policy

Do not silently substitute a different checkpoint or shard for paper-critical
checks. Verify matching hashes, rerun the documented training/data generation
pipeline, or mark the result as not reproduced.
