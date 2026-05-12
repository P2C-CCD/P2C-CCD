# P2CCCD Architecture Notes

P2CCCD implements a proposal-to-certificate CCD pipeline. The architecture is
designed to keep learned and hardware-accelerated stages outside the final
correctness decision.

## Core Data Flow

```text
geometry + motion
  -> conservative proxy / swept-volume candidate generation
  -> CandidateRecord feature rows
  -> learned STPF group scheduling
  -> ExactWorkItem queue
  -> exact certificate or conservative fallback
  -> audit rows and benchmark reports
```

## Stage Responsibilities

- Conservative candidate generation builds swept AABBs, patch proxies, CPU/BVH
  candidates, or optional RT/OptiX candidate traces. This stage must preserve
  recall; it is not allowed to drop a possible collision.
- Candidate records encode motion, primitive family, proxy distance,
  degeneracy, source, and cost-prior features for scheduling.
- STPF inference ranks candidates inside a group. The neural model is a
  proposal/scheduling policy only.
- Exact certification evaluates local primitive CCD queries. Positive groups may
  stop after a certified hit; negative or uncertain groups use all-exact or
  conservative fallback coverage.
- Audit and benchmark export record exact calls, exact-work units, wall time,
  false negatives, false positives, recall, and evidence scope.

## CPU / GPU / RT / NN Boundary

- CPU owns orchestration, dataset streaming, fallback control, and audit export.
- C++ owns hot-path arrays, exact work queues, candidate compaction, and native
  scheduling.
- CUDA/OptiX are optional acceleration paths for candidate generation and exact
  batches when enabled.
- ONNX Runtime / TensorRT executes the STPF scheduler when available.
- Neither RT traversal nor neural inference is the final CCD certificate.

## Evidence Scopes

The repository intentionally separates timing scopes:

- `candidate-row replay`: uniform adapter rows and exact-work accounting.
- `native dense group wall-time`: C++/ORT/TensorRT scheduling hot path, sometimes
  with proxy or cost-trace exact drivers.
- `selected-real TI replay`: real Tight-Inclusion exact payload on selected
  dense groups.
- `full TI/NYU primitive wall-time`: native Tight-Inclusion every-candidate
  primitive baseline.

Do not mix these scopes in claims. The release claim safety file is the
current authority:

```text
../../artifacts/claim_safety_check.md
```

## Quality Gates

The minimum CPU-friendly checks are:

```powershell
conda activate cudadev
python -m pytest src\tests\python\test_contracts.py src\tests\python\test_correctness_and_performance_gates.py src\tests\python\test_quality_gate_inventory.py -q
ctest --test-dir src\build -C Release --output-on-failure
```

The machine-readable gate inventory is:

```text
src/tests/quality_gates.json
```
