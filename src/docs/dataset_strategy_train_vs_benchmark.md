# Dataset Strategy: Training Vs Benchmark

description P2CCCD description, Objectiveisdescriptionwhendescription:

1. descriptionwithdescriptiongenerate, description, coveragedescriptionanddescriptiondistribution.
2. benchmark description, avoiddescription, descriptionanddescription.
3. descriptionand benchmark descriptionindescription, description, description, sourceandusedescriptionondescriptionsplitdescription.

descriptionis not"descriptiondataset"description, insteaddescription.

## 1. description

P2CCCD description:

- `training corpora`
- `benchmark corpora`

descriptionisdescriptionsplitdescription, descriptionunderdescription 4 description:

1. physicsdescriptionsplitdescription.
2. manifest splitdescription.
3. schema description, butdescriptionhasdescriptionlayerdescription.
4. benchmark descriptiontodescriptionanddescription.

description:

**descriptionisdescription, benchmark descriptionisdescription. **

## 2. description

### 2.1 Training Corpora

description:

- description STPF interval / family / priority / cost / uncertainty description;
- perform OOD, fallback, hard-negative and patch/slab generalization;
- description proposal hasdescription, rather thanperformfinal paper correctness claim.

description:

- description;
- description;
- patch/slab augmentation;
- description;
- curriculum;
- description.

description:

- descriptionanddescription benchmark description;
- descriptionuseofficial benchmark original query asdescriptionInput.

### 2.2 Benchmark Corpora

benchmark description:

- main-text table correctness;
- main-text table performance;
- ablation;
- OOD / stress;
- downstream robot scene description.

benchmark description:

- data sourcefixed;
- descriptionfixed;
- descriptionfixed;
- query descriptionfixed;
- seed fixed;
- description.

benchmark description:

- descriptiongeneratedescription query descriptionandofficial benchmark descriptionuse;
- description benchmark query asdescriptionordescription;
- description benchmark mesh performdescriptionwhendescriptionafterdescriptionto benchmark.

## 3. recommenddescription :

```text
src/datasets/
  training/
    synthetic_proxy/
    cad_train/
    robot_train/
    ood_train/
    manifests/
  benchmark/
    correctness/
      scalable_ccd/
      tight_inclusion/
      root_parity/
      rigid_ipc/
    performance/
      scalable_ccd/
      rigid_ipc/
      cad_motion_bench/
      robot_motion_bench/
    ablation/
      internal_high_density/
    manifests/
```

descriptionasdescription:

```text
raw/
processed/
shards/
manifest.json
license.json
README.md
```

description shard and benchmark query batch descriptionindescriptionOutput directorydescription.

## 4. description

### 4.1 T0: Synthetic Proxy Training

this isNo.description.

source:

- `python/p2cccd/data/`
- current analytic swept-sphere oracle
- current high-density STPF workbench

usedescription:

- descriptionNo.description STPF checkpoint;
- perform interval / uncertainty / priority description;
- descriptiontodescriptionlevel row.

descriptiongeneratedescription:

1. `base_proxy_train`
   - current mesh-pair / robot-link analytic proxy description
   - easy negative / near contact / grazing / multiple contact / OOD
2. `dense_candidate_train`
   - current `trained_stpf_high_density.py` description slab + patch augmentationdescription
   - usedescription high candidate-density distribution
3. `mixed_proxy_train`
   -  base and dense description
   - descriptionin manifest descriptionrecorddescriptionsourcedescription

### 4.2 T1: CAD Train Corpus

recommended data source:

- ABC
- Fusion 360 Gallery
- Better STEP

usedescription:

- descriptionhigh patch descriptionanddescriptiondistributioncoverage;
- generatedescriptionreal hard-negative;
- description, description, description, connectdescription motion.

generatedescription:

```text
STEP/B-rep/CAD source
  -> preprocess metadata
  -> tessellation / mesh normalization
  -> patch metadata extraction
  -> motion sampler
  -> proxy builder
  -> candidate cache
  -> exact labeler
  -> training shard
```

descriptionis:

- descriptionwithfrom CAD assetsgeneratedescription motion query;
- butdescriptionconnectafterdescription benchmark scene  query description.

### 4.3 T2: Robot / Manipulation Train Corpus

recommended data source:

- MoveIt resources
- PartNet-Mobility
- YCB
- Google Scanned Objects

usedescription:

- description link-pair sceneunder family / uncertainty / OOD fallback;
- description STPF descriptionto capsule-like / articulated / manipulation object distribution.

### 4.4 T3: OOD / Dirty Geometry Train Corpus

recommended data source:

- Thingi10K
- ShapeNet subset
- Objaverse subset

usedescription:

- is notasdescription;
- insteaddescription uncertainty / fallback / monotonic safety.

descriptionObjectiveis:

- slowdown descriptionwith;
- false negative descriptionwith.

## 5. Benchmark description

### 5.1 B0: Correctness Benchmark

main text correctness descriptionuse:

- `Sample-Scalable-CCD-Data`
- `Scalable CCD`
- `Tight Inclusion`
- `Rigid-IPC`

descriptionwith"official query ordescription adapter query"enter benchmark runner.

benchmark Pathshouldis:

```text
official dataset / official query batch
  -> adapter
  -> P2CCCD runner
  -> BenchmarkRowV2
```

is not:

```text
official dataset
  -> description query
  -> descriptionnewdescriptionofficial benchmark
```

### 5.2 B1: Performance Benchmark

main text performance descriptioninin:

- `Scalable CCD full`
- `Rigid-IPC selected scenes`
- future addition `cad_motion_bench`
- future addition `robot_motion_bench`

description benchmark descriptionis:

- query description;
- candidate density descriptionsplitlayer;
- description;
- data sourcedescriptionfixed.

### 5.3 B2: Ablation Benchmark

description benchmark description external benchmark.

descriptionwithdescription:

- current `high-density synthetic workbench`
- patch granularity ablation
- slab/proxy ablation
- interval-only / ranking-only

butdescription:

**this is ablation/workbench, is not external correctness benchmark. **

## 6. descriptionavoiddescription benchmark

this isdescription.

### 6.1 descriptionleveldescription and benchmark descriptionusedescription:

```text
datasets/training/*
datasets/benchmark/*
```

anydescriptiondefaultdescription `training/`.
any benchmark descriptiondefaultdescription `benchmark/`.

### 6.2 Manifest leveldescription

each shard / batch / scene manifest description:

- `dataset_role`: `training` / `benchmark`
- `source_dataset`
- `source_version`
- `source_scene_or_sequence`
- `source_query_id_range`
- `generator_version`
- `license_id`
- `contains_official_queries`

if `dataset_role == benchmark`, description loader descriptionconnectdescriptionload.

### 6.3 Source ID description write shard when, descriptionwhendescription:

- `source_dataset`
- `source_scene`
- `source_object_ids`
- `source_query_ids`

benchmark descriptionloadwhen, description source id descriptionadd lockbox.

ifdescriptionand benchmark description:

- sameofficial query id;
- same benchmark scene id;
- orsamedescription;

descriptionconnectdescription.

### 6.4 description layer:

- `train`
- `validation`
- `benchmark_lockbox`

among them:

- `validation` descriptionwithdescription;
- `benchmark_lockbox` descriptionwhendescription.

description `Scalable CCD` performance description, descriptionafterdescription test.

## 7. description shard / batch description

### 7.1 Training Shard

description shard recommenddescriptionusecurrent:

- `npz`

descriptionuse.

indescription:

- `parquet`

description:

- descriptionsplitdescription;
- descriptionstatisticsanddescription;
- descriptionsplitdescription.

description shard descriptioncontains:

- `ProposalFeatureRow`
- oracle trace
- split id
- source metadata
- difficulty tags
- density tags

description:

- `candidate_density_bucket`
- `patch_granularity_bucket`
- `slab_count_bucket`
- `ood_tag`
- `source_role=train`

### 7.2 Benchmark Batch

benchmark description shard description.

descriptionuse:

- `query batch json/jsonl`
- officialoriginaldescription
- description `BenchmarkRunMeta`

benchmark batch description:

- official query identity
- official source path
- family
- scene
- dataset version

rather thandescriptionnewdescription"description".

## 8. descriptiongeneratedescription

### 8.1 Phase 1: descriptionondescription perform:

- synthetic base proxy: `1e5` rows
- dense candidate train: `1e6` rows
- mixed proxy train: `2e6+` rows

descriptioncurrent analytic + high-density workbench description.

### 8.2 Phase 2: connect CAD realdescription connect:

- ABC subset
- Fusion 360 subset

description.

descriptionperform:

- `1e4` shapes
- `1e5` pair motions
- `1e6+` candidate rows

### 8.3 Phase 3: connect robot / OOD

add:

- MoveIt
- PartNet-Mobility
- YCB/GSO
- Thingi10K dirty subset

descriptionObjectiveis notdescription, insteaddescription:

- uncertainty quality
- fallback description
- downstream robustness

## 9. descriptionshoulddescription indescriptionsplitdescription:

### 9.1 Training Data

write:

- synthetic proxy train
- CAD-derived train
- robot/OOD train

Notes:

- used for STPF description
- descriptionused fordescription benchmark

### 9.2 External Benchmark

write:

- Scalable CCD
- Tight Inclusion
- Rigid-IPC

Notes:

- version pin
- official queries or fixed adapter queries
- never used for training

### 9.3 Workbench / Ablation

write:

- high-density synthetic workbench
- patch/slab ablation

Notes:

- used fordescription trained STPF whether it really reduces exact work
- is not final external correctness benchmark

## 10. descriptioncurrentdescriptionconnectdeploymentdescription

underdescriptionconnectdescriptionunderdescriptionandFile:

```text
src/datasets/training/manifests/
src/datasets/benchmark/manifests/
src/benchmark/dataset_lockbox.json
src/docs/dataset_strategy_train_vs_benchmark.md
```

description:

1. `build_training_shards.py`
   - only writes `datasets/training/`
2. `build_benchmark_batches.py`
   - only writes `datasets/benchmark/`
3. `validate_dataset_separation.py`
   - descriptionand benchmark isdescriptionhas source id leakage

## 11. description :

**descriptionperformdescription"description", benchmark descriptionperformdescription"description". **

description P2CCCD description, descriptionis:

1. use synthetic + high-density workbench  STPF description;
2. use ABC / Fusion360 / Better STEP descriptiondistribution;
3. use Scalable CCD / Tight Inclusion / Rigid-IPC performdescription benchmark;
4. description benchmark query descriptiontodescription.
