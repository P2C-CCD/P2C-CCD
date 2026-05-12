# Evidence Artifacts

This directory holds paper-facing manifests used to check the release-local
figures, benchmark evidence, claim boundaries, and rerun entry points:

- `release_case_manifest.json`
- `evidence_manifest.md`
- `evidence_manifest.json`
- `claim_safety_check.md`

Use `release_case_manifest.json` as the primary release-local entry point for
paper case verification.

`evidence_manifest.*` records the release-local evidence scope and primary
verification entry points.

The goal is to keep the release repository self-contained for evidence
inspection and smoke reproducibility checks.
