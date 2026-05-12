# Data Generation Baseline

This document covers the non-training part of TODO 69-73 and 77.

The current implementation is the executable Python dataset layer under
`python/p2cccd/data/`. It defines deterministic synthetic motion samples,
oracle traces, stable shard schemas, offline metrics, and optional Warp-ready
arrays. STPF training lives in `python/p2cccd/proposal/`.

## Scope

Implemented:

- programmatic mesh-pair motion samples for easy negatives, near-contact hard
  negatives, grazing contacts, multiple-contact intervals, and OOD meshes,
- robot link-pair motion samples represented as moving capsule-style proxy
  pairs,
- analytic swept-sphere proxy oracle traces for deterministic labels,
- conversion to `ProposalFeatureRow`,
- interval, family, priority, exact-cost, and uncertainty targets,
- stable compressed `npz` shard export with metadata,
- offline metric helpers for interval recall, top-k family recall, label
  summaries, and estimated exact work reduction,
- optional Warp-aware array preparation with CPU fallback when Warp is absent.

Implemented in adjacent modules:

- multi-task STPF training loop and cost-aware objective in
  `python/p2cccd/proposal/`,
- Python wrappers for C++ candidate generation and CPU exact certificate
  execution in `python/p2cccd/candidate_generation.py` and
  `python/p2cccd/certificate_engine.py`,
- pybind11 exposure of low-level C++ candidate, exact, audit, CUDA status, and
  host-batch CUDA exact APIs.

## Why Analytic Proxy Oracle

The generated sampler uses an analytic CCD oracle for two linearly moving
enclosing spheres/discs. For these generated samples this is the intended
reference because the synthetic mesh and robot link pairs are represented by
their proxy radii and linear center trajectories.

C++ candidate and exact APIs are exposed through pybind for runtime wrappers.
Future full-mesh dataset generation can add a second feature-level replay backend
using the same `oracle_trace` and row-target schema without changing the
existing analytic proxy shards.

## NPZ Schema

Each shard has:

- `ids`: row schema, query id, candidate id, slab/object/patch ids, target mask,
- `features`: `[N, 32]` float32 STPF inputs,
- `interval_targets`: `[N, 8]` float32 one-hot interval labels,
- `family_targets`: `[N, 8]` float32 multi-label feature-family targets,
- `scalar_targets`: `[N, 3]` float32 priority, cost, uncertainty targets,
- `split_ids`: `[N]` int32 split ids,
- `oracle_trace`: `[N, 8]` float64 oracle outputs,
- `sample_metadata`: `[N, 8]` float64 compact sampler metadata,
- `metadata_json`: schema version, split names, row count, source, and seed.

Use `write_npz_shard` and `read_npz_shard` rather than writing raw arrays in
training scripts.
