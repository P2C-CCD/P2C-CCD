# Patch Granularity Tuning

`PatchGranularityTuning` is the current CPU-baseline implementation for RT
candidate generation task 40.

The tuner sweeps a list of BVH leaf-cluster patch sizes. For each
`max_triangles_per_leaf` option it:

1. Builds object patches with `BuildPatchesFromBvhLeafClusters`.
2. Builds a slab-local `ProxyScene`.
3. Runs the configured `CandidateGenerator` backend.
4. Recomputes CPU raw/compact candidates as the oracle for the same proxy scene.
5. Rejects options below `min_oracle_recall`.
6. Scores feasible options by compact candidate count, raw hit count, proxy
   count, and optional average patch radius.

The default contract is conservative: the selected option must keep oracle
recall at `1.0`. This is not a replacement for the future exact CCD oracle; it
only verifies that the chosen candidate backend agrees with the CPU broad-phase
oracle for the proxy scene being tuned.

Run the developer smoke tool from `src`:

```powershell
.\build\cpp\Release\p2cccd_patch_granularity_tuner.exe outputs\patch_granularity_tuning.csv
```

The output CSV has one row per evaluated option and includes the selected flag,
patch/proxy/candidate counts, oracle recall, score, density ratios, and backend
timing fields.

When the real OptiX device program is implemented, the same tuner can run with
`CandidateBackend::kOptix` and use CPU oracle recall as a backend regression
gate before performance-driven patch granularity ablations.
