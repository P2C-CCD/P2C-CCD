# P2-2 Patch / Slab / Proxy Ablation

## Scope

P2-2 production proxy-path ablation. Candidate generation uses the C++/OptiX proxy-scene wrapper when available and CPU fallback only if the binding cannot run; correctness is checked against the analytic swept-sphere oracle.

## Selected Safe Rows

| mode | scene | option_name | candidate_recall | compact_candidate_count | raw_hit_count | proxy_count | total_ms | fn_count | fp_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| patch_granularity | car_wall_local_refinement | patches1_conservative | 1.0000 | 72 | 72 | 192 | 214.4129 | 0 | 0 |
| slab_count | car_wall_local_refinement | slab1_aabb | 1.0000 | 72 | 72 | 192 | 5.7980 | 0 | 0 |
| proxy_family | car_wall_local_refinement | slab4_aabb_aabb | 1.0000 | 72 | 72 | 768 | 22.3085 | 0 | 0 |
| patch_granularity | standard_graphics_dense_contact | patches1_conservative | 1.0000 | 80 | 80 | 240 | 6.5759 | 0 | 0 |
| slab_count | standard_graphics_dense_contact | slab1_aabb | 1.0000 | 80 | 80 | 240 | 7.0615 | 0 | 0 |
| proxy_family | standard_graphics_dense_contact | slab4_aabb_aabb | 1.0000 | 80 | 80 | 960 | 26.2363 | 0 | 0 |
| patch_granularity | real_mesh_contact_proxy | patches1_conservative | 1.0000 | 124 | 124 | 256 | 6.9454 | 0 | 0 |
| slab_count | real_mesh_contact_proxy | slab1_aabb | 1.0000 | 124 | 124 | 256 | 7.7538 | 0 | 0 |
| proxy_family | real_mesh_contact_proxy | slab4_aabb_aabb | 1.0000 | 122 | 122 | 1024 | 103.8828 | 0 | 0 |

## All Rows

| mode | scene | option_name | selected | feasible | candidate_recall | compact_candidate_count | raw_hit_count | proxy_count | total_ms | fn_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| patch_granularity | car_wall_local_refinement | patches1_conservative | True | True | 1.0000 | 72 | 72 | 192 | 214.4129 | 0 |
| patch_granularity | car_wall_local_refinement | patches2_local | False | True | 1.0000 | 72 | 288 | 384 | 11.0948 | 0 |
| patch_granularity | car_wall_local_refinement | patches4_local | False | True | 1.0000 | 72 | 1152 | 768 | 23.4563 | 0 |
| patch_granularity | car_wall_local_refinement | patches8_fine | False | True | 1.0000 | 79 | 4404 | 1536 | 54.7300 | 0 |
| slab_count | car_wall_local_refinement | slab1_aabb | True | True | 1.0000 | 72 | 72 | 192 | 5.7980 | 0 |
| slab_count | car_wall_local_refinement | slab2_aabb | False | True | 1.0000 | 72 | 72 | 384 | 10.5525 | 0 |
| slab_count | car_wall_local_refinement | slab4_aabb | False | True | 1.0000 | 72 | 72 | 768 | 20.6525 | 0 |
| slab_count | car_wall_local_refinement | slab8_aabb | False | True | 1.0000 | 72 | 72 | 1536 | 42.3837 | 0 |
| proxy_family | car_wall_local_refinement | slab4_aabb_aabb | True | True | 1.0000 | 72 | 72 | 768 | 22.3085 | 0 |
| proxy_family | car_wall_local_refinement | slab4_capsule_capsule | False | True | 1.0000 | 72 | 72 | 768 | 22.0294 | 0 |
| proxy_family | car_wall_local_refinement | slab4_aabb_capsule | False | True | 1.0000 | 72 | 72 | 768 | 21.9364 | 0 |
| patch_granularity | standard_graphics_dense_contact | patches1_conservative | True | True | 1.0000 | 80 | 80 | 240 | 6.5759 | 0 |
| patch_granularity | standard_graphics_dense_contact | patches2_local | False | True | 1.0000 | 80 | 310 | 480 | 13.1717 | 0 |
| patch_granularity | standard_graphics_dense_contact | patches4_local | False | True | 1.0000 | 80 | 1261 | 960 | 28.4585 | 0 |
| patch_granularity | standard_graphics_dense_contact | patches8_fine | False | True | 1.0000 | 120 | 4369 | 1920 | 64.8863 | 0 |
| slab_count | standard_graphics_dense_contact | slab1_aabb | True | True | 1.0000 | 80 | 80 | 240 | 7.0615 | 0 |
| slab_count | standard_graphics_dense_contact | slab2_aabb | False | True | 1.0000 | 80 | 80 | 480 | 13.2811 | 0 |
| slab_count | standard_graphics_dense_contact | slab4_aabb | False | True | 1.0000 | 80 | 80 | 960 | 25.7821 | 0 |
| slab_count | standard_graphics_dense_contact | slab8_aabb | False | True | 1.0000 | 80 | 80 | 1920 | 52.7615 | 0 |
| proxy_family | standard_graphics_dense_contact | slab4_aabb_aabb | True | True | 1.0000 | 80 | 80 | 960 | 26.2363 | 0 |
| proxy_family | standard_graphics_dense_contact | slab4_capsule_capsule | False | True | 1.0000 | 80 | 80 | 960 | 27.4188 | 0 |
| proxy_family | standard_graphics_dense_contact | slab4_aabb_capsule | False | True | 1.0000 | 80 | 80 | 960 | 26.9151 | 0 |
| patch_granularity | real_mesh_contact_proxy | patches1_conservative | True | True | 1.0000 | 124 | 124 | 256 | 6.9454 | 0 |
| patch_granularity | real_mesh_contact_proxy | patches2_local | False | True | 1.0000 | 124 | 496 | 512 | 14.3201 | 0 |
| patch_granularity | real_mesh_contact_proxy | patches4_local | False | True | 1.0000 | 125 | 1981 | 1024 | 32.2685 | 0 |
| patch_granularity | real_mesh_contact_proxy | patches8_fine | False | True | 1.0000 | 127 | 7669 | 2048 | 82.4298 | 0 |
| slab_count | real_mesh_contact_proxy | slab1_aabb | True | True | 1.0000 | 124 | 124 | 256 | 7.7538 | 0 |
| slab_count | real_mesh_contact_proxy | slab2_aabb | False | True | 1.0000 | 122 | 122 | 512 | 14.1926 | 0 |
| slab_count | real_mesh_contact_proxy | slab4_aabb | False | True | 1.0000 | 122 | 122 | 1024 | 28.1604 | 0 |
| slab_count | real_mesh_contact_proxy | slab8_aabb | False | True | 1.0000 | 187 | 187 | 2048 | 56.2873 | 0 |
| proxy_family | real_mesh_contact_proxy | slab4_aabb_aabb | True | True | 1.0000 | 122 | 122 | 1024 | 103.8828 | 0 |
| proxy_family | real_mesh_contact_proxy | slab4_capsule_capsule | False | True | 1.0000 | 122 | 122 | 1024 | 29.2358 | 0 |
| proxy_family | real_mesh_contact_proxy | slab4_aabb_capsule | False | True | 1.0000 | 122 | 122 | 1024 | 28.8185 | 0 |
