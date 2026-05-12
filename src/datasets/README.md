# Dataset Release Notes

This public repository snapshot ships dataset manifests, not the full local
dataset roots.

## Bundled

- `manifests/`
- `README_upstream_adapters.md`

## Not Bundled

- raw or extracted CAD/mesh datasets
- generated training shards
- large benchmark-side feature exports

## Expected Local Layout For Full Reproduction

Place externally obtained data under these paths when reproducing the larger
benchmark suites:

```text
src/datasets/abc/
src/datasets/abc_official/
src/datasets/fusion360/
src/datasets/fusion360_full/
src/datasets/shapenet_core_v2/
src/datasets/thingi10k/
src/datasets/training/
```

The authoritative source and licensing notes are:

- `manifests/datasets_manifest.md`
- `manifests/licensing_manifest.md`
- `../docs/model_artifacts_manifest.md`

Do not treat missing local dataset roots as a silent fallback condition. Either
mount the expected source data, regenerate the missing shards, or report that a
full reproduction was not attempted.
