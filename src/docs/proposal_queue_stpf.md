# Proposal Queue And STPF

This document covers the first implementation slices for TODO 57-67.

## Data Flow

The C++ proposal module defines a conservative no-drop path:

1. `RawCandidateQueue` stores compact `CandidateRecord` rows and density stats
   from RT candidate generation.
2. `ProposalFeatureRow` stores fixed-width STPF input features plus optional
   row targets.
3. `BuildExactWorkQueuePassthrough` converts every raw candidate into one
   `ExactWorkItem`.
4. `ValidateProposalDataFlow` checks that candidate, feature-row, and exact-work
   counts match and that each exact item references its parent candidate.

The current path intentionally does not drop candidates. Learned proposal
ranking can reorder or prioritize work, but the scheduler must still conserve
the candidate set before exact certification.

## Feature Rows

`ExtractProposalFeatureRows` maps each compact candidate back to its slab-local
proxy primitives and emits 32 numeric features:

- slab time range,
- RT hit count,
- proxy types,
- motion bounds,
- AABB extents, volumes, surface areas, overlap ratio,
- center distance,
- conservative radii and motion bounds,
- candidate-density context.

`WriteProposalFeatureRowsCsv` exports a stable CSV for dataset smoke tests and
early STPF experiments.

## STPF v1

The Python model `p2cccd.proposal.STPFModel` is a lightweight MLP with five
heads:

- interval logits,
- feature-family logits,
- priority score,
- cost score,
- uncertainty score.

`stpf_multitask_loss` provides a basic multi-task loss for unit tests and first
training smoke runs.

## Dummy Policy And Scheduling

`BuildDummyProposalOutputs` copies `ProposalFeatureRow` targets into
`ProposalOutput` rows. This gives the end-to-end queue a deterministic policy
before a trained STPF checkpoint exists.

`ScheduleExactWorkItemsFromProposals` consumes raw candidates, feature rows, and
proposal outputs. It may sort the exact queue by proposal priority, but it always
emits exactly one `ExactWorkItem` per raw candidate. `ValidateProposalScheduleConservation`
checks the no-drop invariant by comparing parent candidate ids against the raw
candidate queue.

The scheduler is conservative by default:

- exact intervals stay at the full proxy slab interval,
- point-triangle and edge-edge feature families remain enabled,
- unknown exact feature-family bits are rejected at scheduling config validation,
- missing proposal rows, invalid predictions, OOD features, and high uncertainty
  all route to `ProposalSource::kFallback`,
- fallback is allowed to slow the system down but not to create false negatives.

Python exposes the same first-stage interface through `dummy_proposal_policy` and
`batched_stpf_inference`. The batching runner converts feature rows to tensors,
executes `STPFModel` in chunks, and returns validated `ProposalPrediction` rows
in the same candidate order. OOD rows are not sent through the model by default;
they are converted to dummy fallback predictions with high uncertainty. The
runner restores the model training state after inference errors and treats a
missing prediction for any input row as a hard runtime error rather than silently
dropping the row.

## Runtime Fast Path

The benchmark path no longer performs proposal scheduling in Python candidate
loops by default when `p2cccd_cpp` is available.

- `build_runtime_proposal_feature_rows(...)` parses Python contract rows directly
  inside pybind and emits `ProposalFeatureRow` in C++.
- `run_dummy_runtime_proposal_schedule(...)` executes the deterministic dummy
  policy and exact-queue scheduling in C++.
- `schedule_runtime_exact_work_items(...)` consumes Python-side learned STPF
  predictions but performs fallback checks, family-mask expansion, priority
  ordering, and monotonic queue construction in C++.

This keeps Python responsible for orchestration and model inference, while the
runtime scheduling hot path is moved into the compiled backend. The current
implementation is CPU/C++ first; a dedicated CUDA proposal kernel is still a
follow-up optimization rather than the default path.
