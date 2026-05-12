# External Baseline Release Notes

The original author workspace keeps large third-party CCD baselines under
`src/baseline/`. Those repositories are not bundled into this public release.

## Expected Directory Names

```text
src/baseline/Tight-Inclusion/
src/baseline/CCD-Wrapper/
src/baseline/Scalable-CCD/
src/baseline/Sample-Scalable-CCD-Data/
src/baseline/Exact-Root-Parity-CCD/
src/baseline/rigid-ipc/
```

## Bundled Here

- `download_ccd_datasets.ps1`
- `download_remaining_ccd_datasets.sh`

## Reference Files

- `../docs/third_party_manifest.md`
- `../datasets/manifests/licensing_manifest.md`

Use adapter contracts and manifests to reconnect the external baselines. This
repository snapshot is intentionally source-light so the public package stays
small and license-auditable.
