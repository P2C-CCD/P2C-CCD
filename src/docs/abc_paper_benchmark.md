# ABC CAD Paper Benchmark

## Purpose

this benchmark benchmark used for CAD sourcedescriptionconnecttopaper-trackMethodon, descriptioncompare:

- `PureExactCPU`
- `BVHExact`
- `RTExact`
- `NoProposal`
- `RTSTPFExact-Random`
- `RTSTPFExact-Trained`

among them `RTSTPFExact-Trained` defaultloaddescriptionhas ABC STPF checkpoint.

description, runner descriptionindescriptionsupportdescription `CAD hard-case high-density benchmark`:

- based onsame batch held-out CAD pair
- throughdescriptionhigh `slab_count` and `patches_per_object` constructdescriptionhigh `candidate inflation`
- used fordescription"in CAD hard cases under, trained STPF whether it really reduces exact work"

## description/Benchmark separation

defaultdescriptionunder, runner description `ABC demo subset` descriptiontodescription `48`  asset, descriptionuse:

- description asset: descriptionhas ABC demo description checkpoint sourcedistribution
- afterdescription asset: description benchmark  held-out CAD benchmark slice

description shard in pair/query descriptionconnectdescription benchmark.

ifdescriptionuse `use_official_root=True`, runner descriptionto `src/datasets/abc_official`, descriptionthroughofficial `obj_v00` chunk descriptionofficialdescription root.

## Entry point

- Python module: `p2cccd.bench.abc_paper_benchmark`
- Main API:
  - `build_abc_paper_benchmark_dataset(...)`
  - `run_abc_paper_benchmark(...)`
  - `write_abc_paper_benchmark_report(...)`
  - `write_abc_paper_benchmark_summary_json(...)`

## Output

Benchmark Outputdescriptionwriteto:

- `src/benchmark/<run_name>.md`
- `src/benchmark/<run_name>.json`

Held-out CAD benchmark dataset descriptionwriteto:

- `src/datasets/benchmark/cad_motion_bench/<run_name>/benchmark_dataset.npz`
- `src/datasets/benchmark/cad_motion_bench/<run_name>/dataset_manifest.json`

official root description:

- `p2cccd.datasets.cad.abc_official`

description:

1. descriptionofficial `size.yml` and `obj_v00.txt`
2. selectdescriptionofficial `obj` chunk
3. underdescription `.7z`
4. description `.obj` to `abc_official` root

## Protocol notes

currentdefaultdescriptionis `ABC-compatible local demo subset`  CAD benchmark, is notofficial full-scale ABC description.

descriptionusedescriptionis:

1. descriptionpaper-track `RT + STPF + Exact Certificate` in CAD-like descriptionondescription.
2. descriptionafter STPF in held-out CAD slice onisdescriptionkeep `final FN = 0`.
3. descriptionafterdescriptionconnectdescriptionofficial ABC root afterdescription benchmark descriptionsamedescription runner.
