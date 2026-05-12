# P2CCCD Dataset Adapter Layer

This directory documents repository-level adapter targets. Runtime Python adapters live in:

```text
src/python/p2cccd/datasets/
```

The first implemented adapter layer targets the correctness / CCD benchmark sources from `P2CCCD_revise_run_id.md` section 2.1:

```text
adapter -> DatasetScene / DatasetQueryBatch -> internal mesh/motion/query -> CandidateOracle -> ExactOracle -> BenchmarkRowV2
```

The current implementation intentionally does not vendor external baseline code into the P2CCCD build. The downloaded repositories under `src/baseline/` are treated as reference stacks and external data sources.
