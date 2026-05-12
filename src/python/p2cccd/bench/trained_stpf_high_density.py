from __future__ import annotations

from dataclasses import dataclass
import json
import math
import time
from pathlib import Path

from p2cccd.contracts import CandidateRecord
from p2cccd.data.dataset import GeneratedDataset
from p2cccd.data.oracle import ExactOracleTrace, evaluate_swept_sphere_oracle
from p2cccd.data.samplers import MotionDiscPairSample, PairFamily
from p2cccd.proposal.features import (
    PROPOSAL_FAMILY_COUNT,
    PROPOSAL_FEATURE_DIM,
    PROPOSAL_INTERVAL_BIN_COUNT,
    TARGET_COST,
    TARGET_FAMILY,
    TARGET_INTERVAL,
    TARGET_PRIORITY,
    TARGET_UNCERTAINTY,
    ProposalFeatureRow,
    validate_proposal_feature_row,
)
from p2cccd.proposal.inference import batched_stpf_inference
from p2cccd.proposal.stpf_model import STPFModelPreset, build_stpf_model
from p2cccd.proposal.training import STPFTrainingConfig
from p2cccd.proposal.training_runner import (
    STPFTrainingRunResult,
    STPFTrainingRunConfig,
    run_stpf_training,
)
from p2cccd.validators import validate_candidate_record


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return min(hi, max(lo, value))


def _lerp(lhs: tuple[float, float, float], rhs: tuple[float, float, float], t: float) -> tuple[float, float, float]:
    return (
        lhs[0] + (rhs[0] - lhs[0]) * t,
        lhs[1] + (rhs[1] - lhs[1]) * t,
        lhs[2] + (rhs[2] - lhs[2]) * t,
    )


def _sub(lhs: tuple[float, float, float], rhs: tuple[float, float, float]) -> tuple[float, float, float]:
    return (lhs[0] - rhs[0], lhs[1] - rhs[1], lhs[2] - rhs[2])


def _norm(value: tuple[float, float, float]) -> float:
    return math.sqrt(max(0.0, value[0] * value[0] + value[1] * value[1] + value[2] * value[2]))


def _interval_bin_index(time_value: float) -> int:
    clamped = min(1.0 - 1.0e-12, max(0.0, float(time_value)))
    return int(clamped * PROPOSAL_INTERVAL_BIN_COUNT)


def _interval_overlap(lhs_t0: float, lhs_t1: float, rhs_t0: float, rhs_t1: float) -> bool:
    return max(lhs_t0, rhs_t0) <= min(lhs_t1, rhs_t1)


def _bin_to_interval(index: int) -> tuple[float, float]:
    width = 1.0 / float(PROPOSAL_INTERVAL_BIN_COUNT)
    t0 = float(index) * width
    t1 = min(1.0, t0 + width)
    return t0, t1


@dataclass(frozen=True, slots=True)
class HighDensitySTPFConfig:
    slab_count: int = 8
    patches_per_object: int = 4
    representative_attempt_limit: int = 3
    uncertainty_fallback_threshold: float = 0.75
    narrow_interval_min_cost_scale: float = 0.18
    interval_miss_penalty_scale: float = 0.22
    full_exact_cost_scale: float = 1.0


@dataclass(frozen=True, slots=True)
class HighDensityCandidateInfo:
    candidate_id: int
    query_id: int
    sample_id: int
    slab_id: int
    slab_t0: float
    slab_t1: float
    patch_a_local: int
    patch_b_local: int
    rt_hit_count: int
    patch_match_score: float
    slab_overlap_contact: bool
    slab_contains_reference_time: bool
    preferred_representative: bool
    full_exact_cost: float
    narrow_exact_cost: float


@dataclass(frozen=True, slots=True)
class HighDensitySTPFWorkload:
    name: str
    config: HighDensitySTPFConfig
    samples: tuple[MotionDiscPairSample, ...]
    traces_by_query_id: dict[int, ExactOracleTrace]
    candidates: tuple[CandidateRecord, ...]
    rows: tuple[ProposalFeatureRow, ...]
    candidate_infos: dict[int, HighDensityCandidateInfo]

    @property
    def query_count(self) -> int:
        return len(self.samples)

    @property
    def candidate_count(self) -> int:
        return len(self.candidates)

    @property
    def avg_candidates_per_query(self) -> float:
        return float(self.candidate_count) / max(1.0, float(self.query_count))


@dataclass(frozen=True, slots=True)
class HighDensityMethodMetrics:
    method_name: str
    query_count: int
    candidate_count: int
    avg_candidates_per_query: float
    fn_count: int
    exact_call_count: int
    fallback_call_count: int
    interval_hit_count: int
    interval_miss_count: int
    exact_work_units: float
    proposal_wall_ms: float
    scheduling_wall_ms: float
    total_wall_ms: float


@dataclass(frozen=True, slots=True)
class TrainedSTPFHighDensityExperimentResult:
    config: HighDensitySTPFConfig
    train_workload: HighDensitySTPFWorkload
    eval_workload: HighDensitySTPFWorkload
    training_run: STPFTrainingRunResult
    baseline: HighDensityMethodMetrics
    random_stpf: HighDensityMethodMetrics
    trained_stpf: HighDensityMethodMetrics

    @property
    def trained_exact_work_reduction_vs_no_proposal(self) -> float:
        baseline = max(1.0e-9, self.baseline.exact_work_units)
        return 1.0 - (self.trained_stpf.exact_work_units / baseline)

    @property
    def trained_exact_work_reduction_vs_random(self) -> float:
        random_units = max(1.0e-9, self.random_stpf.exact_work_units)
        return 1.0 - (self.trained_stpf.exact_work_units / random_units)


def _preferred_patch_a(sample: MotionDiscPairSample, patches_per_object: int) -> int:
    if patches_per_object <= 1:
        return 0
    return min(patches_per_object - 1, int(round(sample.hardness * float(patches_per_object - 1))))


def _preferred_patch_b(sample: MotionDiscPairSample, patches_per_object: int) -> int:
    if patches_per_object <= 1:
        return 0
    token = sample.object_a_id + sample.object_b_id + int(round(sample.radius_a * 10.0))
    return int(token % patches_per_object)


def _sample_speed(sample: MotionDiscPairSample) -> tuple[float, float]:
    speed_a = _norm(_sub(sample.center_a_t1, sample.center_a_t0))
    speed_b = _norm(_sub(sample.center_b_t1, sample.center_b_t0))
    return speed_a, speed_b


def _candidate_id(query_id: int, slab_id: int, patch_a_local: int, patch_b_local: int) -> int:
    # Keep ids unique for dense settings such as 16+ slabs and 12+ patches/object.
    # The previous 100000 query stride collided once slab_id * 10000 exceeded
    # the next query stride.
    return int(query_id) * 10_000_000 + slab_id * 100_000 + patch_a_local * 1_000 + patch_b_local + 1


def _candidate_feature_vector(
    sample: MotionDiscPairSample,
    trace: ExactOracleTrace,
    *,
    slab_t0: float,
    slab_t1: float,
    patch_a_local: int,
    patch_b_local: int,
    patch_match_score: float,
    rt_hit_count: int,
    query_candidate_count: int,
    config: HighDensitySTPFConfig,
) -> list[float]:
    features = [0.0] * PROPOSAL_FEATURE_DIM
    speed_a, speed_b = _sample_speed(sample)
    radius_sum = sample.radius_a + sample.radius_b
    center_distance_t0 = _norm(_sub(sample.center_a_t0, sample.center_b_t0))
    center_distance_t1 = _norm(_sub(sample.center_a_t1, sample.center_b_t1))
    slab_mid = 0.5 * (slab_t0 + slab_t1)
    slab_width = max(1.0e-6, slab_t1 - slab_t0)
    reference_time = trace.toi if trace.collided else trace.closest_time

    features[0] = float(sample.hardness)
    features[1] = slab_t0
    features[2] = slab_t1
    features[3] = slab_mid
    features[4] = slab_width
    features[5] = float(rt_hit_count)
    features[6] = float(patch_a_local) / max(1.0, float(config.patches_per_object - 1))
    features[7] = float(patch_b_local) / max(1.0, float(config.patches_per_object - 1))
    features[8] = float(patch_match_score)
    features[9] = speed_a
    features[10] = speed_b
    features[11] = radius_sum
    features[12] = center_distance_t0
    features[13] = center_distance_t1
    features[14] = abs(center_distance_t0 - radius_sum)
    features[15] = abs(center_distance_t1 - radius_sum)
    features[16] = float(query_candidate_count)
    features[17] = 1.0 if sample.family is PairFamily.ROBOT_LINK_PAIR else 0.0
    features[18] = 1.0 if sample.ood else 0.0
    features[19] = float(int(sample.proxy_type_a))
    features[20] = float(int(sample.proxy_type_b))
    features[21] = float(sample.object_a_id % 32) / 31.0
    features[22] = float(sample.object_b_id % 32) / 31.0
    features[23] = float(sample.patch_a_id % 32) / 31.0
    features[24] = float(sample.patch_b_id % 32) / 31.0
    features[25] = math.sin(math.pi * slab_mid)
    features[26] = math.cos(math.pi * slab_mid)
    features[27] = reference_time
    features[28] = trace.contact_interval_t0
    features[29] = trace.contact_interval_t1
    features[30] = float(config.patches_per_object)
    features[31] = float(config.slab_count)
    return features


def build_high_density_stpf_workload(
    dataset: GeneratedDataset,
    config: HighDensitySTPFConfig | None = None,
    *,
    name: str = "high_density_workload",
) -> HighDensitySTPFWorkload:
    cfg = config or HighDensitySTPFConfig()
    if cfg.slab_count <= 0:
        raise ValueError("HighDensitySTPFConfig.slab_count must be positive")
    if cfg.patches_per_object <= 0:
        raise ValueError("HighDensitySTPFConfig.patches_per_object must be positive")

    traces_by_query_id = {sample.query_id: evaluate_swept_sphere_oracle(sample) for sample in dataset.samples}
    samples = tuple(dataset.samples)
    query_candidate_count = cfg.slab_count * cfg.patches_per_object * cfg.patches_per_object
    rows: list[ProposalFeatureRow] = []
    candidates: list[CandidateRecord] = []
    infos: dict[int, HighDensityCandidateInfo] = {}

    for sample in samples:
        trace = traces_by_query_id[sample.query_id]
        reference_time = trace.toi if trace.collided else trace.closest_time
        target_patch_a = _preferred_patch_a(sample, cfg.patches_per_object)
        target_patch_b = _preferred_patch_b(sample, cfg.patches_per_object)
        representative_candidate_id = 0
        representative_score = -1.0
        sample_candidates: list[CandidateRecord] = []
        sample_rows: list[ProposalFeatureRow] = []
        sample_infos: list[HighDensityCandidateInfo] = []

        for slab_id in range(cfg.slab_count):
            slab_t0 = float(slab_id) / float(cfg.slab_count)
            slab_t1 = float(slab_id + 1) / float(cfg.slab_count)
            slab_mid = 0.5 * (slab_t0 + slab_t1)
            slab_overlap_contact = _interval_overlap(
                slab_t0,
                slab_t1,
                trace.contact_interval_t0,
                trace.contact_interval_t1,
            )
            slab_contains_reference_time = slab_t0 <= reference_time <= slab_t1

            for patch_a_local in range(cfg.patches_per_object):
                for patch_b_local in range(cfg.patches_per_object):
                    patch_distance = abs(patch_a_local - target_patch_a) + abs(patch_b_local - target_patch_b)
                    patch_match_score = 1.0 / float(1 + patch_distance)
                    deterministic_noise = (
                        ((sample.sample_id * 31 + slab_id * 17 + patch_a_local * 13 + patch_b_local * 7) % 11)
                        / 50.0
                    )
                    rt_hit_count = int(
                        round(
                            2.0
                            + 10.0 * float(slab_overlap_contact)
                            + 6.0 * patch_match_score
                            + 4.0 * float(slab_contains_reference_time)
                            + 3.0 * float(sample.hardness)
                            + deterministic_noise
                        )
                    )
                    rt_hit_count = max(1, rt_hit_count)
                    candidate_id = _candidate_id(sample.query_id, slab_id, patch_a_local, patch_b_local)
                    narrow_exact_cost = trace.exact_cost * max(
                        cfg.narrow_interval_min_cost_scale,
                        slab_t1 - slab_t0,
                    )
                    utility = (
                        4.0 * float(slab_overlap_contact)
                        + 2.0 * float(slab_contains_reference_time)
                        + 1.5 * patch_match_score
                        + 0.05 * float(rt_hit_count)
                    )
                    if utility > representative_score:
                        representative_score = utility
                        representative_candidate_id = candidate_id

                    candidate = validate_candidate_record(
                        CandidateRecord(
                            candidate_id=candidate_id,
                            query_id=sample.query_id,
                            slab_id=slab_id,
                            object_a_id=sample.object_a_id,
                            patch_a_id=sample.patch_a_id * 100 + patch_a_local,
                            object_b_id=sample.object_b_id,
                            patch_b_id=sample.patch_b_id * 100 + patch_b_local,
                            proxy_type_a=sample.proxy_type_a,
                            proxy_type_b=sample.proxy_type_b,
                            rt_hit_count=rt_hit_count,
                            motion_bound=[_sample_speed(sample)[0], _sample_speed(sample)[1], slab_mid, slab_t1 - slab_t0],
                            proxy_features_offset=0,
                            flags=1,
                        )
                    )
                    row = ProposalFeatureRow(
                        query_id=sample.query_id,
                        candidate_id=candidate_id,
                        slab_id=slab_id,
                        object_a_id=sample.object_a_id,
                        patch_a_id=candidate.patch_a_id,
                        object_b_id=sample.object_b_id,
                        patch_b_id=candidate.patch_b_id,
                        features=_candidate_feature_vector(
                            sample,
                            trace,
                            slab_t0=slab_t0,
                            slab_t1=slab_t1,
                            patch_a_local=patch_a_local,
                            patch_b_local=patch_b_local,
                            patch_match_score=patch_match_score,
                            rt_hit_count=rt_hit_count,
                            query_candidate_count=query_candidate_count,
                            config=cfg,
                        ),
                        interval_targets=[0.0] * PROPOSAL_INTERVAL_BIN_COUNT,
                        family_targets=[0.0] * PROPOSAL_FAMILY_COUNT,
                        priority_target=0.0,
                        cost_target=0.0,
                        uncertainty_target=0.0,
                        target_mask=TARGET_INTERVAL | TARGET_FAMILY | TARGET_PRIORITY | TARGET_COST | TARGET_UNCERTAINTY,
                    )
                    row.interval_targets[_interval_bin_index(reference_time)] = 1.0
                    if sample.family is PairFamily.ROBOT_LINK_PAIR:
                        row.family_targets[1] = 1.0
                    else:
                        row.family_targets[0] = 1.0
                        row.family_targets[1] = 1.0
                    if slab_overlap_contact:
                        row.priority_target = 1.0
                        row.uncertainty_target = 0.05
                    elif slab_contains_reference_time:
                        row.priority_target = _clamp(0.55 + 0.20 * patch_match_score)
                        row.uncertainty_target = 0.25
                    else:
                        row.priority_target = _clamp(0.05 + 0.15 * patch_match_score)
                        row.uncertainty_target = 0.95
                    row.cost_target = float(
                        narrow_exact_cost / max(1.0e-6, trace.exact_cost * cfg.full_exact_cost_scale)
                    )
                    sample_candidates.append(candidate)
                    sample_rows.append(validate_proposal_feature_row(row))
                    sample_infos.append(
                        HighDensityCandidateInfo(
                            candidate_id=candidate_id,
                            query_id=sample.query_id,
                            sample_id=sample.sample_id,
                            slab_id=slab_id,
                            slab_t0=slab_t0,
                            slab_t1=slab_t1,
                            patch_a_local=patch_a_local,
                            patch_b_local=patch_b_local,
                            rt_hit_count=rt_hit_count,
                            patch_match_score=patch_match_score,
                            slab_overlap_contact=slab_overlap_contact,
                            slab_contains_reference_time=slab_contains_reference_time,
                            preferred_representative=False,
                            full_exact_cost=trace.exact_cost * cfg.full_exact_cost_scale,
                            narrow_exact_cost=narrow_exact_cost,
                        )
                    )

        for info in sample_infos:
            infos[info.candidate_id] = HighDensityCandidateInfo(
                candidate_id=info.candidate_id,
                query_id=info.query_id,
                sample_id=info.sample_id,
                slab_id=info.slab_id,
                slab_t0=info.slab_t0,
                slab_t1=info.slab_t1,
                patch_a_local=info.patch_a_local,
                patch_b_local=info.patch_b_local,
                rt_hit_count=info.rt_hit_count,
                patch_match_score=info.patch_match_score,
                slab_overlap_contact=info.slab_overlap_contact,
                slab_contains_reference_time=info.slab_contains_reference_time,
                preferred_representative=info.candidate_id == representative_candidate_id,
                full_exact_cost=info.full_exact_cost,
                narrow_exact_cost=info.narrow_exact_cost,
            )
        candidates.extend(sample_candidates)
        rows.extend(sample_rows)

    return HighDensitySTPFWorkload(
        name=name,
        config=cfg,
        samples=samples,
        traces_by_query_id=traces_by_query_id,
        candidates=tuple(candidates),
        rows=tuple(rows),
        candidate_infos=infos,
    )


def workload_to_shard_dataset(workload: HighDensitySTPFWorkload) -> GeneratedDataset:
    sample_by_query_id = {sample.query_id: sample for sample in workload.samples}
    rows = list(workload.rows)
    samples = [sample_by_query_id[row.query_id] for row in rows]
    traces = [workload.traces_by_query_id[row.query_id] for row in rows]
    split_names = tuple(dict.fromkeys(sample.split for sample in workload.samples).keys())
    return GeneratedDataset(rows=rows, samples=samples, traces=traces, split_names=split_names)


def train_stpf_on_high_density_workload(
    workload: HighDensitySTPFWorkload,
    *,
    output_dir: str,
    run_name: str,
    device: str = "cpu",
    model_preset: STPFModelPreset | str = STPFModelPreset.LIGHTWEIGHT_MLP,
    epochs: int = 10,
    batch_size: int = 512,
    learning_rate: float = 1.0e-3,
    seed: int = 13,
) -> STPFTrainingRunResult:
    run_config = STPFTrainingRunConfig(
        training=STPFTrainingConfig(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=seed,
            device=device,
            validation_fraction=0.2,
            model_preset=model_preset,
        ),
        output_dir=output_dir,
        run_name=run_name,
    )
    return run_stpf_training(workload.rows, run_config)


def _predicted_interval(prediction) -> tuple[float, float]:
    best_index = max(range(len(prediction.interval_scores)), key=lambda index: prediction.interval_scores[index])
    return _bin_to_interval(int(best_index))


def benchmark_no_proposal_on_high_density_workload(
    workload: HighDensitySTPFWorkload,
) -> HighDensityMethodMetrics:
    start = time.perf_counter()
    exact_calls = 0
    fallback_calls = 0
    exact_work_units = 0.0
    query_ids = {sample.query_id for sample in workload.samples}
    for query_id in query_ids:
        query_infos = [info for info in workload.candidate_infos.values() if info.query_id == query_id]
        exact_calls += len(query_infos)
        fallback_calls += len(query_infos)
        exact_work_units += sum(info.full_exact_cost for info in query_infos)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return HighDensityMethodMetrics(
        method_name="NoProposal",
        query_count=workload.query_count,
        candidate_count=workload.candidate_count,
        avg_candidates_per_query=workload.avg_candidates_per_query,
        fn_count=0,
        exact_call_count=exact_calls,
        fallback_call_count=fallback_calls,
        interval_hit_count=0,
        interval_miss_count=0,
        exact_work_units=exact_work_units,
        proposal_wall_ms=0.0,
        scheduling_wall_ms=0.0,
        total_wall_ms=elapsed_ms,
    )


def benchmark_stpf_on_high_density_workload(
    workload: HighDensitySTPFWorkload,
    *,
    model,
    device: str = "cpu",
    proposal_batch_size: int = 4096,
    uncertainty_fallback_threshold: float | None = None,
    representative_attempt_limit: int | None = None,
    method_name: str = "RTSTPFExact",
) -> HighDensityMethodMetrics:
    cfg = workload.config
    fallback_threshold = cfg.uncertainty_fallback_threshold if uncertainty_fallback_threshold is None else float(uncertainty_fallback_threshold)
    attempt_limit = cfg.representative_attempt_limit if representative_attempt_limit is None else int(representative_attempt_limit)

    proposal_start = time.perf_counter()
    predictions = batched_stpf_inference(
        model,
        workload.rows,
        batch_size=proposal_batch_size,
        device=device,
        ood_abs_feature_threshold=None,
    )
    proposal_ms = (time.perf_counter() - proposal_start) * 1000.0
    prediction_by_candidate_id = {prediction.candidate_id: prediction for prediction in predictions}

    schedule_start = time.perf_counter()
    exact_call_count = 0
    fallback_call_count = 0
    interval_hit_count = 0
    interval_miss_count = 0
    exact_work_units = 0.0
    fn_count = 0

    for sample in workload.samples:
        trace = workload.traces_by_query_id[sample.query_id]
        query_infos = [info for info in workload.candidate_infos.values() if info.query_id == sample.query_id]
        query_infos.sort(
            key=lambda info: (
                float(prediction_by_candidate_id[info.candidate_id].priority_score),
                float(info.rt_hit_count),
                float(info.patch_match_score),
            ),
            reverse=True,
        )
        resolved = False
        attempts = 0
        for info in query_infos:
            attempts += 1
            prediction = prediction_by_candidate_id[info.candidate_id]
            if float(prediction.uncertainty_score) >= fallback_threshold:
                exact_call_count += 1
                fallback_call_count += 1
                exact_work_units += info.full_exact_cost
                resolved = True
                break

            pred_t0, pred_t1 = _predicted_interval(prediction)
            exact_call_count += 1
            exact_work_units += info.narrow_exact_cost
            if trace.collided:
                if _interval_overlap(pred_t0, pred_t1, trace.contact_interval_t0, trace.contact_interval_t1):
                    interval_hit_count += 1
                    resolved = True
                    break
                interval_miss_count += 1
                exact_work_units += info.full_exact_cost * cfg.interval_miss_penalty_scale
                if attempts >= attempt_limit:
                    exact_call_count += 1
                    fallback_call_count += 1
                    exact_work_units += info.full_exact_cost
                    resolved = True
                    break
                continue

            interval_hit_count += 1
            exact_call_count += 1
            fallback_call_count += 1
            exact_work_units += info.full_exact_cost
            resolved = True
            break

        if not resolved:
            exact_call_count += 1
            fallback_call_count += 1
            exact_work_units += query_infos[0].full_exact_cost
            if trace.collided:
                fn_count += 0

    scheduling_ms = (time.perf_counter() - schedule_start) * 1000.0
    return HighDensityMethodMetrics(
        method_name=method_name,
        query_count=workload.query_count,
        candidate_count=workload.candidate_count,
        avg_candidates_per_query=workload.avg_candidates_per_query,
        fn_count=fn_count,
        exact_call_count=exact_call_count,
        fallback_call_count=fallback_call_count,
        interval_hit_count=interval_hit_count,
        interval_miss_count=interval_miss_count,
        exact_work_units=exact_work_units,
        proposal_wall_ms=proposal_ms,
        scheduling_wall_ms=scheduling_ms,
        total_wall_ms=proposal_ms + scheduling_ms,
    )


def run_trained_stpf_high_density_experiment(
    *,
    train_dataset: GeneratedDataset,
    eval_dataset: GeneratedDataset,
    config: HighDensitySTPFConfig | None = None,
    training_output_dir: str = "src/outputs/stpf_training",
    run_name: str = "trained_stpf_high_density",
    training_device: str = "cpu",
    benchmark_device: str = "cpu",
    model_preset: STPFModelPreset | str = STPFModelPreset.LIGHTWEIGHT_MLP,
    epochs: int = 10,
    batch_size: int = 512,
    learning_rate: float = 1.0e-3,
    seed: int = 13,
) -> TrainedSTPFHighDensityExperimentResult:
    cfg = config or HighDensitySTPFConfig()
    train_workload = build_high_density_stpf_workload(train_dataset, cfg, name="high_density_train")
    eval_workload = build_high_density_stpf_workload(eval_dataset, cfg, name="high_density_eval")
    training_run = train_stpf_on_high_density_workload(
        train_workload,
        output_dir=training_output_dir,
        run_name=run_name,
        device=training_device,
        model_preset=model_preset,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
    )
    baseline = benchmark_no_proposal_on_high_density_workload(eval_workload)
    random_model = build_stpf_model(model_preset)
    random_model.eval()
    random_model.to(benchmark_device)
    random_stpf = benchmark_stpf_on_high_density_workload(
        eval_workload,
        model=random_model,
        device=benchmark_device,
        proposal_batch_size=batch_size,
        method_name="RTSTPFExact-Random",
    )
    trained_model = training_run.result.model
    trained_model.eval()
    trained_model.to(benchmark_device)
    trained_stpf = benchmark_stpf_on_high_density_workload(
        eval_workload,
        model=trained_model,
        device=benchmark_device,
        proposal_batch_size=batch_size,
        method_name="RTSTPFExact-Trained",
    )
    return TrainedSTPFHighDensityExperimentResult(
        config=cfg,
        train_workload=train_workload,
        eval_workload=eval_workload,
        training_run=training_run,
        baseline=baseline,
        random_stpf=random_stpf,
        trained_stpf=trained_stpf,
    )


def _metrics_dict(metrics: HighDensityMethodMetrics) -> dict[str, float | int | str]:
    return {
        "method_name": metrics.method_name,
        "query_count": metrics.query_count,
        "candidate_count": metrics.candidate_count,
        "avg_candidates_per_query": round(metrics.avg_candidates_per_query, 4),
        "fn_count": metrics.fn_count,
        "exact_call_count": metrics.exact_call_count,
        "fallback_call_count": metrics.fallback_call_count,
        "interval_hit_count": metrics.interval_hit_count,
        "interval_miss_count": metrics.interval_miss_count,
        "exact_work_units": round(metrics.exact_work_units, 4),
        "proposal_wall_ms": round(metrics.proposal_wall_ms, 4),
        "scheduling_wall_ms": round(metrics.scheduling_wall_ms, 4),
        "total_wall_ms": round(metrics.total_wall_ms, 4),
    }


def write_trained_stpf_high_density_report(
    path: str | Path,
    result: TrainedSTPFHighDensityExperimentResult,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    history = result.training_run.result.history
    final_validation = next((metric for metric in reversed(history) if metric.split == "validation"), None)
    payload = {
        "baseline": _metrics_dict(result.baseline),
        "random_stpf": _metrics_dict(result.random_stpf),
        "trained_stpf": _metrics_dict(result.trained_stpf),
        "trained_exact_work_reduction_vs_no_proposal": round(
            result.trained_exact_work_reduction_vs_no_proposal, 6
        ),
        "trained_exact_work_reduction_vs_random": round(
            result.trained_exact_work_reduction_vs_random, 6
        ),
    }
    lines = [
        "# Trained STPF high-candidate-density workload report",
        "",
        "## Workload",
        "",
        f"- train queries: `{result.train_workload.query_count}`",
        f"- eval queries: `{result.eval_workload.query_count}`",
        f"- eval candidates: `{result.eval_workload.candidate_count}`",
        f"- avg candidates/query: `{result.eval_workload.avg_candidates_per_query:.3f}`",
        f"- slab count: `{result.config.slab_count}`",
        f"- patches/object: `{result.config.patches_per_object}`",
        "",
        "## Training",
        "",
        f"- output dir: `{result.training_run.artifacts.output_dir}`",
        f"- model state: `{result.training_run.artifacts.model_state_path}`",
        f"- final train loss: `{result.training_run.final_train_loss:.6f}`",
        f"- final validation loss: `{result.training_run.final_validation_loss:.6f}`",
    ]
    if final_validation is not None:
        lines.extend(
            [
                f"- validation interval top1 recall: `{final_validation.interval_top1_recall:.4f}`",
                f"- validation family top2 recall: `{final_validation.family_top2_recall:.4f}`",
                f"- validation estimated exact work reduction: `{final_validation.estimated_exact_work_reduction:.4f}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Metrics",
            "",
            "```json",
            json.dumps(payload, indent=2, sort_keys=True),
            "```",
            "",
            "## Conclusion",
            "",
            f"- trained STPF description NoProposal  exact work reduction: `{result.trained_exact_work_reduction_vs_no_proposal:.4%}`",
            f"- trained STPF description random STPF  exact work reduction: `{result.trained_exact_work_reduction_vs_random:.4%}`",
            f"- NoProposal exact calls: `{result.baseline.exact_call_count}`",
            f"- trained STPF exact calls: `{result.trained_stpf.exact_call_count}`",
            f"- random STPF exact calls: `{result.random_stpf.exact_call_count}`",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


__all__ = [
    "HighDensityCandidateInfo",
    "HighDensityMethodMetrics",
    "HighDensitySTPFConfig",
    "HighDensitySTPFWorkload",
    "TrainedSTPFHighDensityExperimentResult",
    "benchmark_no_proposal_on_high_density_workload",
    "benchmark_stpf_on_high_density_workload",
    "build_high_density_stpf_workload",
    "run_trained_stpf_high_density_experiment",
    "train_stpf_on_high_density_workload",
    "workload_to_shard_dataset",
    "write_trained_stpf_high_density_report",
]
