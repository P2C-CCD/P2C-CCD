# Evidence Manifest

This public release ships a redacted evidence manifest.

The primary release-local case map is:

- `artifacts/release_case_manifest.json`
- `src/docs/paper_case_reproduction.md`

This manifest is limited to release-local paper evidence and runnable entry
points required for bundled evidence verification.

## Verification Command

```powershell
conda activate cudadev
python scripts\verify_release_cases.py
```

## Scope

This public manifest is intentionally limited to release-local assets and
bundled evidence.
