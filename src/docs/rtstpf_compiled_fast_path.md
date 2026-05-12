# RTSTPFExact Compiled ORT Fast Path

## Purpose

 learned `RTSTPFExact` descriptionPathfrom

```text
C++ candidate -> Python ProposalFeatureRow dataclass
-> ORT/TensorRT inference
-> Python ProposalPrediction dataclass
-> C++ scheduling
```

description

```text
C++ candidate -> C++/pybind feature arrays
-> ORT/TensorRT array inference
-> C++/pybind scheduling from arrays
-> C++/pybind ExactWorkItem work queue
-> Python artifact/materialization (descriptionwhendescription)
```

Objectiveisreduction Python/C++ description,  `proposal_ms` descriptiontodescriptionconnectdescriptionreal learned proposal runtime.

## descriptionconnectdescription

### C++ / pybind

- `build_runtime_proposal_feature_arrays(...)`
- `schedule_runtime_exact_work_items_from_arrays(...)`
- `schedule_runtime_exact_work_items_from_proposal_arrays(...)`

position:
- [py_module.cpp](../cpp/bindings/py_module.cpp)

### Python ORT

- `batched_stpf_inference_ort_arrays(...)`

position:
- [ort_inference.py](../python/p2cccd/proposal/ort_inference.py)

### RTSTPFExact descriptionPath

- ORT learned Pathdescription compiled arrays fast path
- proposal hot path descriptionconnectdescription C++/pybind `ExactWorkItem`
- exact descriptionconnectdescription pybind work queue
- `ProposalFeatureRow` / `ProposalPrediction` descriptionindescription materialize
- Python `ExactWorkItem` dataclass descriptionin coverage descriptionanddescription materialize

position:
- [rt_stpf_exact.py](../python/p2cccd/bench/rt_stpf_exact.py)

## timing protocol

new `proposal_ms` contains:

- runtime feature array build
- ORT/TensorRT inference
- compiled scheduling
- pybind C++ work queue description

new `proposal_ms` descriptioncontains:

- benchmark description Python dataclass materialization
- description ORT runtime descriptionand warmup

## currentdescription defaultwritedescription
`src/benchmark/rtstpf_compiled_fastpath_optimization_run_id.md`. thisdescriptionPathdescriptionreport
description public release descriptionleveldescription; descriptionanddescriptionProtocol.

descriptionConclusion:

- `external_sparse` fastest_learned:
  - total `207.3041 -> 132.2682 ms`
  - proposal `146.9134 -> 62.5367 ms`
- `abc_official_large` fastest_learned:
  - total `127.0296 -> 78.4694 ms`
  - proposal `42.1486 -> 23.1931 ms`
- `thingi10k_heldout` fastest_learned:
  - total `21.5539 -> 15.9404 ms`
  - proposal `10.9459 -> 5.2391 ms`

description hot path descriptionafter, `schedule_runtime_exact_work_items_from_proposal_arrays(...)`
avoiddescription candidate-object reparse, and proposal description C++ work queue description
Python `ExactWorkItem` dataclass. description steady-state descriptiondefaultwritedescription
`src/benchmark/rtstpf_hotpath_optimization_run_id.md`.

description:

- `external_sparse` auto_fastest:
  - total `93.3727 -> 61.5118 ms`
  - proposal `40.4244 -> 8.9693 ms`
- `abc_official_large` auto_fastest:
  - total `66.3456 -> 54.8951 ms`
  - proposal `11.9292 -> 4.4688 ms`
- `thingi10k_heldout` auto_fastest:
  - total `17.9617 -> 14.9499 ms`
  - proposal `4.4404 -> 1.2106 ms`

## description

- currentdescriptioncoverage learned `ORT` Path.
- `torch` descriptionPathdescription Python object path.
- descriptionhasdescription correctness contract; `FN = 0` / `Recall = 1.0` constraintkeepdescription.
- description synthetic/oracle exact benchmark, `PureExactCPU` description; here exact isdescription/descriptionPath, RT and NN fixedoverheaddescription amortize.
- description learned STPF description, descriptionis TensorRT inference, insteaddescriptionlayerdescriptionisdescriptionreduction exact work: currentdescriptionisdescription candidate description exact work item, anddefault interval iscomplete `[0, 1]`.
