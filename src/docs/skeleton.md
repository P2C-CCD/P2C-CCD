# P2CCCD Skeleton

this paperdescriptionrecordcurrent `src` descriptionleveldescription. Objectiveisafterdescription `RT Core + STPF/neural proposal + exact CCD certificate` descriptionfixedunderdescription, avoiddescriptionindescription.

## currentdescription

`P2CCCD` is Proposal-to-Certificate Continuous Collision Detection description.

currentdescriptionis notcomplete CCD description, insteaddescriptiondeploymentdescriptionwithunderdescription:

- C++ hot path descriptionand CMake target.
- C++/Python description runtime contract.
- candidate, proposal, exact work item, certificate, audit log, benchmark row description.
- Python orchestration package descriptionEntry point.
- runtime config descriptionloadanddescription.
- audit/benchmark description.
- C++/Python contract descriptiontest.

## description

```text
src/
  CMakeLists.txt                 # descriptionlevel CMake Entry point
  pyproject.toml                 # Python editable package Entry point
  configs/
    default_runtime.json         # default epsilon, runtime limit, benchmark Outputdescription
  cpp/
    CMakeLists.txt               # p2cccd_core target and C++ tests
    common/
      runtime_contracts.h        # C++ runtime contract description
      status.h                   # description Status
      validators.{h,cpp}         # C++ contract validators
    geometry/                    # Mesh / Patch / MotionSegment placeholder interface
    rt_candidate/                # CandidateGenerator placeholder interface
    certificate/                 # CertificateEngine placeholder interface
    bindings/                    # pybind11 description
  python/p2cccd/
    contracts.py                 # Python dataclass mirror + CONTRACT_SCHEMAS
    validators.py                # Python contract validators
    config.py                    # RuntimeConfig dataclass + validation
    serialization.py             # JSONL/CSV stable serialization
    bench/ data/ proposal/ viz/  # afterdescription orchestration descriptionEntry point
  docs/
    architecture.md              # highlayerdescription
    serialization.md             # audit/benchmark description
    skeleton.md                  # currentdescriptionNotes
  tests/
    cpp/test_validators.cpp      # C++ validators smoke/regression
    python/test_contracts.py     # Python validators + C++ description
```

## descriptiondeployment Contract

current C++ descriptionis `cpp/common/runtime_contracts.h`.

description runtime record:

- `CandidateRecord`: RT candidate record, description query/slab/object/patch/proxy/motion bound description.
- `ProposalOutput`: proposal descriptionordescriptionOutput, description interval/family score, priority, cost, uncertainty.
- `ExactWorkItem`: exact checker Inputdescription, description candidate, whendescription, feature family, descriptionsource.
- `CertificateResult`: exact checker Outputdescription, description collision/separation/undecided, TOI, margin, witness, epsilon.
- `AuditLogRow`: descriptionwhendescription, description stage/action/depth/interval/timestamp/aux value.
- `BenchmarkRow`: benchmark description, description recall, FN/FP, descriptioncandidate, exact eval, timing, QPS.

Python description `contracts.py` through `CONTRACT_SCHEMAS` fixeddescriptionanddescription. `tests/python/test_contracts.py` description C++ header  struct description, description Python schema description.

## description

C++:

- `cpp/common/validators.h`
- `cpp/common/validators.cpp`

Python:

- `python/p2cccd/validators.py`

currentdescription:

- schema version description.
- required id descriptionas 0.
- enum descriptionisdescription, candidate proxy descriptionis `UNKNOWN`.
- whendescription `0 <= interval_t0 <= interval_t1 <= 1`.
- score, margin, timing, epsilon descriptionishasdescription.
- epsilon descriptionasdescription.
- separation certificate descriptionhas `covered_feature_mask`.
- undecided certificate descriptionhas `reason_code`.

## descriptionFile

defaultdescription:

```text
configs/default_runtime.json
```

Python API:

- `load_runtime_config(path) -> RuntimeConfig`
- `load_runtime_config_dict(path) -> dict`
- `validate_runtime_config(config) -> RuntimeConfig`

currentdescription contract ondescription:

- `runtime.max_interval_bins <= 8`
- `runtime.max_family_scores <= 8`
- `epsilon.* > 0`

## description File:

```text
python/p2cccd/serialization.py
```

currentdescription:

- audit log use JSONL, description `AuditLogRow`.
- benchmark row use CSV, header descriptionfrom `CONTRACT_SCHEMAS["BenchmarkRow"]`.
- enum by C++ enum description.
- writedescriptiondefaultdescription validator.

descriptionNotesdescription:

```text
docs/serialization.md
```

## descriptionandtest

fromdescription:

```powershell
cmake -S src -B src\build
cmake --build src\build --config Release
ctest --test-dir src\build -C Release --output-on-failure
```

Python testdescriptionuse `cudadev`:

```powershell
python -m pytest src\tests\python -q
```

## Debug visualization

currentdescription C++ debug tool, usedescriptionhas geometry/motion/proxy descriptiongeneratedescriptioncontains HTML:

```powershell
src\build\cpp\Release\p2cccd_geometry_proxy_viewer.exe src\outputs\geometry_proxy_debug.html
```

Output files:

```text
src/outputs/geometry_proxy_debug.html
```

this HTML description:

- patch triangle mesh descriptionanddescription.
- patch center descriptiontrajectory.
- each time slab  endpoint swept AABB.
- each time slab  capsule proxy and `eps_proxy` description.

this is debug description, is notdescription. descriptionusedescriptionisdescription proxy isdescription, slab isdescription, description bound isdescriptioncoverage sampled patch center.

current capsule proxy is broad-phase conservative proxy: descriptionuse slab description patch center  chord descriptioncoveragedescriptionPath, descriptionPath tight swept-sphere capsule. afterdescriptionused for no-FN CCD when, description proxy coverage descriptiontest, descriptionin OptiX candidate generator connectdescriptionafteruse CPU oracle perform candidate recall description.

current `ProxyScene` descriptionall object descriptionusesamedescription slab partition: description `slab_id` description `[t0, t1]`, description `slab_id` whendescription. thisdescriptionavoid `CandidateRecord` descriptionhasdescription `slab_id` whendescription slab description.

## underdescriptionconnectdescription bywithunderdescription:

- description CPU exact oracle: point-triangle, edge-edge interval checks.
- description conservative proxy generation: swept AABB, capsule, proxy inflation.
- current RT candidate description proxy scene, raw candidate buffer, CPU reference generator, descriptionwhendescriptionsplit, candidate density export and compaction; underdescriptionconnectdescription CPU reference candidate recall oracle.
- OptiX afterdescriptionhasdescription CMake gate and tracer Entry point; descriptionOutput `RawCandidateHit` descriptionafterdescription device program, SBT descriptionand launch description.
- description dummy/no-proposal policy, description `candidate -> exact work item -> certificate`.
- descriptionafterconnect STPF Modeldescription, proposal descriptionreductiondescription/description, description certificate coverage.

## Git description

shoulddescription git description:

- `src/cpp/`
- `src/python/`
- `src/configs/`
- `src/docs/`
- `src/tests/`
- `src/CMakeLists.txt`
- `src/pyproject.toml`
- `src/README.md`

descriptionshoulddescription git description:

- `src/build/`
- `src/lib/`
- `src/.pytest_cache/`
- `__pycache__/`
- `*.pyc`

`lib/` descriptionisdescriptionandthird-partydescription, descriptionunderdescription, descriptionordescription, avoiddescriptionconnectdescription.
