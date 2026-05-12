# ABC Training Pipeline

descriptionrecordcurrent `ABC Dataset` descriptiongenerateand STPF description.

## currentdescription

currentdescription:

1. `ABC` mesh asset ingestion
2. patch sidecar / mesh statistics / hard-negative pair generation
3. `ABC mesh pair -> proxy motion sample -> ProposalFeatureRow shard`
4. `ABC base + dense workload -> STPF training -> benchmark`

Code entry point:

- adapter:
  [abc_adapter.py](../python/p2cccd/datasets/cad/abc_adapter.py)
- dataset generation:
  [abc_training.py](../python/p2cccd/datasets/cad/abc_training.py)
- training / benchmark runner:
  [abc_training.py](../python/p2cccd/bench/abc_training.py)

## descriptionofficial ABC description

official `ABC Dataset` description. with `abc_0000_obj_v00.7z` asdescription, descriptionconnectdescription `8 GB`.

thereforecurrentdescriptionusedescriptionlayerdescription:

1. if `src/datasets/abc` underdescriptionhasusedescriptionreal `ABC` mesh root, descriptionconnectdescriptionuserealdescription.
2. ifdescriptionhasreal `ABC` root, descriptiongeneratedescription **ABC-compatible local demo subset**, used for ingestion / shard / training / benchmark description.

this demo subset descriptionuseis:

- description
- descriptionanddescriptiontest

description**is not**official ABC description benchmark replacedescription.

## current demo subset

defaultdescription:

```text
src/datasets/abc
```

ifdescriptionhasrealdescription, descriptionwritedescription:

```text
src/datasets/abc/demo_subset_generated/
```

descriptionisdescriptiongenerate `obj + patch.json + metadata.json` description CAD-like assets.

## descriptiongeneratedescription

current `ABC` descriptiongenerateperformis:

```text
ABC assets
  -> deterministic hard-negative mesh pairs
  -> proxy radii from mesh diagonal
  -> 4 linear motion variants per pair
     - easy_negative
     - near_contact_hard_negative
     - grazing_contact
     - multiple_contact_interval / strong approach
  -> analytic swept-sphere oracle
  -> ProposalFeatureRow
  -> base train/eval shards
  -> dense slab x patch x patch expansion
  -> dense train/eval shards
```

this benchmarkdescriptionObjectiveisdescription:

- CAD-derived hard-negative distribution
- patch metadata
- high candidate-density proposal learning

connectdescriptionhas STPF description.

## Output directory

currentdescriptiondefaultwritedescription:

- report: `src/benchmark/abc_training_run_id.md`
- summary: `src/benchmark/abc_training_run_id.json`
- training shards: `src/datasets/training/cad_train/abc/shards/abc_training_20260422_demo_main`
- training output: `src/outputs/stpf_training/abc_training_20260422_demo_main`

description public release descriptionleveldescription; descriptionondescription
`abc_training.py` Entry pointafterdescriptionnewgenerate.

## currentdescription currentdescriptionuseis `demo subset`:

- asset count: `24`
- train pair count: `120`
- eval pair count: `40`
- base train queries: `480`
- base eval queries: `160`
- dense eval rows: `20480`
- avg candidates/query: `128`

descriptionafterdescription:

- validation interval top1 recall: `0.8519`
- `NoProposal exact work`: `302706.7316`
- `Random STPF exact work`: `2866.4087`
- `Trained STPF exact work`: `1977.4800`
- trained vs `NoProposal`: `99.3467%`
- trained vs `Random STPF`: `31.0119%`
- `fn_count = 0`

## Conclusion

current `ABC` descriptionisdescription:

- descriptionon: from CAD asset ingestion to shard, description, benchmark description
- descriptionon: descriptionafter STPF in `ABC-compatible` CAD-like high candidate densitydistributiononbetter than random STPF

butdescriptionwritedescription `ABC Dataset` description, description:

1. usedescriptionrealofficial `ABC` description
2. recordreal source version / license / commit / chunk identity
3. inreal `ABC` descriptionondescriptioncurrentdescriptionand benchmark
