from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Sequence

import numpy as np

from p2cccd.proposal.ort_inference import (
    batched_stpf_inference_ort_arrays,
    create_ort_inference_session,
    ensure_stpf_model_onnx,
)
from p2cccd.proposal.policy_head_selection import (
    RTSTPFPolicyHead,
    select_rtstpf_policy_head,
    score_rtstpf_candidates,
)

from .common_modeling_ort_walltime_benchmark import _feature_arrays_from_npz, _load_metadata
from .native_dense_group_benchmark import DEFAULT_CASES, NativeDenseGroupCaseSpec, _load_model


RUN_NAME = "learned_vs_random_ablation_run_id"


@dataclass(frozen=True, slots=True)
class OriginalGroupDiagnosis:
    group_count: int
    candidate_count: int
    positive_group_count: int
    negative_group_count: int
    mixed_group_count: int
    pure_positive_group_count: int
    positive_candidate_count: int
    mean_candidates_per_group: float
    mean_positive_fraction_in_positive_groups: float
    rank_ablation_is_informative: bool


@dataclass(frozen=True, slots=True)
class RankChallengeSpec:
    group_size: int = 512
    positives_per_group: int = 4
    max_groups: int = 512
    random_seed: int = 424242
    hard_negative_pool_multiplier: int = 4


@dataclass(frozen=True, slots=True)
class MethodMetrics:
    method: str
    group_count: int
    candidate_count: int
    positive_candidates_per_group: int
    no_proposal_exact_calls: int
    scheduled_exact_calls: int
    no_proposal_exact_work: float
    scheduled_exact_work: float
    exact_call_reduction: float
    exact_work_reduction: float
    fn: int
    first_positive_rank_mean: float
    first_positive_rank_p50: float
    first_positive_rank_p90: float
    first_positive_rank_p99: float
    cost_weighted_first_positive_mean: float
    cost_weighted_first_positive_p50: float
    cost_weighted_first_positive_p90: float
    hard_negative_rejection_mean: float


def _safe_zscore(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return np.zeros_like(arr, dtype=np.float64)
    cleaned = np.where(finite, arr, float(np.nanmean(arr[finite])))
    sigma = float(np.std(cleaned))
    if sigma <= 1.0e-12:
        return np.zeros_like(cleaned, dtype=np.float64)
    return (cleaned - float(np.mean(cleaned))) / sigma


def _quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.quantile(np.asarray(values, dtype=np.float64), q))


def _diagnose_original_groups(feature_arrays: dict[str, np.ndarray | int]) -> OriginalGroupDiagnosis:
    query_ids = np.asarray(feature_arrays["query_id"], dtype=np.uint64)
    positive = np.asarray(feature_arrays["oracle_trace"], dtype=np.float64)[:, 0] > 0.5
    unique_ids, inverse = np.unique(query_ids, return_inverse=True)
    group_sizes = np.bincount(inverse)
    positive_counts = np.bincount(inverse, weights=positive.astype(np.int64))
    positive_groups = positive_counts > 0
    pure_positive = positive_counts == group_sizes
    mixed = (positive_counts > 0) & (positive_counts < group_sizes)
    positive_fractions = np.divide(
        positive_counts[positive_groups],
        np.maximum(1, group_sizes[positive_groups]),
    )
    mean_positive_fraction = float(np.mean(positive_fractions)) if positive_fractions.size else 0.0
    return OriginalGroupDiagnosis(
        group_count=int(unique_ids.shape[0]),
        candidate_count=int(query_ids.shape[0]),
        positive_group_count=int(np.count_nonzero(positive_groups)),
        negative_group_count=int(np.count_nonzero(positive_counts == 0)),
        mixed_group_count=int(np.count_nonzero(mixed)),
        pure_positive_group_count=int(np.count_nonzero(pure_positive & positive_groups)),
        positive_candidate_count=int(np.count_nonzero(positive)),
        mean_candidates_per_group=float(np.mean(group_sizes)) if group_sizes.size else 0.0,
        mean_positive_fraction_in_positive_groups=mean_positive_fraction,
        rank_ablation_is_informative=bool(np.count_nonzero(mixed) > 0),
    )


def _make_rank_challenge_indices(
    feature_arrays: dict[str, np.ndarray | int],
    spec: RankChallengeSpec,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(spec.random_seed))
    trace = np.asarray(feature_arrays["oracle_trace"], dtype=np.float64)
    positive = trace[:, 0] > 0.5
    full_cost = np.asarray(trace[:, 5], dtype=np.float64)
    positive_indices = np.flatnonzero(positive)
    negative_indices = np.flatnonzero(~positive)
    if positive_indices.size < spec.positives_per_group:
        raise ValueError("rank challenge requires positive candidate rows")
    negatives_per_group = int(spec.group_size) - int(spec.positives_per_group)
    if negatives_per_group <= 0:
        raise ValueError("group_size must be larger than positives_per_group")
    if negative_indices.size < negatives_per_group:
        raise ValueError("rank challenge requires enough negative candidate rows")

    max_by_pos = int(positive_indices.size // int(spec.positives_per_group))
    max_by_neg = int(negative_indices.size // negatives_per_group)
    group_count = min(int(spec.max_groups), max_by_pos, max_by_neg)
    if group_count <= 0:
        raise ValueError("rank challenge produced zero groups")

    rng.shuffle(positive_indices)
    hard_pool_size = min(
        negative_indices.size,
        max(group_count * negatives_per_group, group_count * negatives_per_group * spec.hard_negative_pool_multiplier),
    )
    hard_order = negative_indices[np.argsort(full_cost[negative_indices])[::-1][:hard_pool_size]]
    rng.shuffle(hard_order)

    selected = np.empty((group_count, int(spec.group_size)), dtype=np.int64)
    for group_index in range(group_count):
        pos_start = group_index * int(spec.positives_per_group)
        neg_start = group_index * negatives_per_group
        group_rows = np.concatenate(
            [
                positive_indices[pos_start : pos_start + int(spec.positives_per_group)],
                hard_order[neg_start : neg_start + negatives_per_group],
            ]
        )
        rng.shuffle(group_rows)
        selected[group_index, :] = group_rows
    remixed_group_ids = np.repeat(np.arange(1, group_count + 1, dtype=np.uint64), int(spec.group_size))
    return selected.reshape(-1), remixed_group_ids


def _prediction_subset(predictions: dict[str, np.ndarray], indices: np.ndarray) -> dict[str, np.ndarray]:
    return {
        key: np.ascontiguousarray(value[indices])
        for key, value in predictions.items()
        if isinstance(value, np.ndarray) and value.shape[0] > int(np.max(indices))
    }


def _feature_subset(
    feature_arrays: dict[str, np.ndarray | int],
    indices: np.ndarray,
) -> dict[str, np.ndarray | int]:
    out: dict[str, np.ndarray | int] = {}
    max_index = int(np.max(indices))
    for key, value in feature_arrays.items():
        if isinstance(value, np.ndarray) and value.shape[0] > max_index:
            out[key] = np.ascontiguousarray(value[indices])
        else:
            out[key] = value
    return out


def _score_methods(
    *,
    source_name: str,
    predictions: dict[str, np.ndarray],
    feature_arrays: dict[str, np.ndarray | int],
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    trace = np.asarray(feature_arrays["oracle_trace"], dtype=np.float64)
    full_cost = np.asarray(trace[:, 5], dtype=np.float64)
    positive = trace[:, 0] > 0.5
    priority = np.asarray(predictions["priority_score"], dtype=np.float64)
    pred_cost = np.asarray(predictions["cost_score"], dtype=np.float64)
    uncertainty = np.asarray(predictions["uncertainty_score"], dtype=np.float64)
    interval_scores = np.asarray(predictions["interval_scores"], dtype=np.float64)
    family_scores = np.asarray(predictions["family_scores"], dtype=np.float64)
    features = np.asarray(feature_arrays["features"], dtype=np.float32)
    feature_energy = np.linalg.norm(features[:, : min(16, features.shape[1])], axis=1).astype(np.float64)
    selected_head = select_rtstpf_policy_head(
        source_name,
        candidate_density=float(priority.shape[0]) / max(1.0, float(np.unique(feature_arrays["query_id"]).shape[0])),
        hard_negative_group=False,
    )

    methods = {
        "ValidationSelectedFullSTPF": score_rtstpf_candidates(
            predictions,
            feature_arrays,
            head=selected_head.head,
        ),
        "LearnedPriorityOnly": priority,
        "LearnedCostAware": _safe_zscore(priority) - 0.20 * _safe_zscore(pred_cost) - 0.10 * _safe_zscore(uncertainty),
        "LearnedRiskProximityHybrid": score_rtstpf_candidates(
            predictions,
            feature_arrays,
            head=RTSTPFPolicyHead.RISK_PROXIMITY_HYBRID,
        ),
        "LearnedCalibrated": _safe_zscore(priority) - 0.25 * _safe_zscore(uncertainty) - 0.10 * _safe_zscore(full_cost),
        "IntervalOnly": np.max(interval_scores, axis=1),
        "RankingOnly": np.max(family_scores, axis=1),
        "UncertaintyOnly": -uncertainty,
        "HeuristicCostLow": -full_cost,
        "HeuristicCostHigh": full_cost,
        "HeuristicFeatureEnergy": feature_energy,
        "OracleUpperBound": positive.astype(np.float64) * 1.0e9 - _safe_zscore(full_cost),
    }
    methods["RandomUniform"] = rng.random(priority.shape[0], dtype=np.float64)
    return {key: np.ascontiguousarray(value, dtype=np.float64) for key, value in methods.items()}


def _evaluate_scores(
    *,
    method: str,
    scores: np.ndarray,
    positive: np.ndarray,
    full_cost: np.ndarray,
    group_size: int,
    positives_per_group: int,
) -> MethodMetrics:
    row_count = int(scores.shape[0])
    if row_count % group_size != 0:
        raise ValueError("rank challenge rows must be divisible by group_size")
    group_count = row_count // group_size
    no_proposal_calls = row_count
    no_proposal_work = float(np.sum(full_cost, dtype=np.float64))
    scheduled_calls = 0
    scheduled_work = 0.0
    ranks: list[float] = []
    cost_weighted: list[float] = []
    rejection: list[float] = []

    for group_index in range(group_count):
        begin = group_index * group_size
        end = begin + group_size
        local_scores = scores[begin:end]
        local_positive = positive[begin:end]
        local_cost = full_cost[begin:end]
        order = np.argsort(local_scores, kind="stable")[::-1]
        positive_positions = np.flatnonzero(local_positive[order])
        if positive_positions.size == 0:
            rank = group_size
            selected = order
        else:
            rank = int(positive_positions[0]) + 1
            selected = order[:rank]
        work = float(np.sum(local_cost[selected], dtype=np.float64))
        scheduled_calls += int(selected.shape[0])
        scheduled_work += work
        ranks.append(float(rank))
        cost_weighted.append(work)
        hard_negative_count = max(1, group_size - positives_per_group)
        rejection.append(1.0 - max(0.0, float(rank - 1)) / float(hard_negative_count))

    return MethodMetrics(
        method=method,
        group_count=group_count,
        candidate_count=row_count,
        positive_candidates_per_group=int(positives_per_group),
        no_proposal_exact_calls=no_proposal_calls,
        scheduled_exact_calls=int(scheduled_calls),
        no_proposal_exact_work=no_proposal_work,
        scheduled_exact_work=float(scheduled_work),
        exact_call_reduction=1.0 - float(scheduled_calls) / max(1, no_proposal_calls),
        exact_work_reduction=1.0 - float(scheduled_work) / max(1.0e-12, no_proposal_work),
        fn=0,
        first_positive_rank_mean=float(mean(ranks)) if ranks else 0.0,
        first_positive_rank_p50=_quantile(ranks, 0.50),
        first_positive_rank_p90=_quantile(ranks, 0.90),
        first_positive_rank_p99=_quantile(ranks, 0.99),
        cost_weighted_first_positive_mean=float(mean(cost_weighted)) if cost_weighted else 0.0,
        cost_weighted_first_positive_p50=_quantile(cost_weighted, 0.50),
        cost_weighted_first_positive_p90=_quantile(cost_weighted, 0.90),
        hard_negative_rejection_mean=float(mean(rejection)) if rejection else 0.0,
    )


def _evaluate_no_proposal_all_exact(
    *,
    positive: np.ndarray,
    full_cost: np.ndarray,
    group_size: int,
    positives_per_group: int,
) -> MethodMetrics:
    row_count = int(positive.shape[0])
    if row_count % group_size != 0:
        raise ValueError("rank challenge rows must be divisible by group_size")
    group_count = row_count // int(group_size)
    total_work = float(np.sum(full_cost, dtype=np.float64))
    group_work = [
        float(np.sum(full_cost[index * group_size : (index + 1) * group_size], dtype=np.float64))
        for index in range(group_count)
    ]
    return MethodMetrics(
        method="NoProposalAllExact",
        group_count=group_count,
        candidate_count=row_count,
        positive_candidates_per_group=int(positives_per_group),
        no_proposal_exact_calls=row_count,
        scheduled_exact_calls=row_count,
        no_proposal_exact_work=total_work,
        scheduled_exact_work=total_work,
        exact_call_reduction=0.0,
        exact_work_reduction=0.0,
        fn=0,
        first_positive_rank_mean=float(group_size),
        first_positive_rank_p50=float(group_size),
        first_positive_rank_p90=float(group_size),
        first_positive_rank_p99=float(group_size),
        cost_weighted_first_positive_mean=float(mean(group_work)) if group_work else 0.0,
        cost_weighted_first_positive_p50=_quantile(group_work, 0.50),
        cost_weighted_first_positive_p90=_quantile(group_work, 0.90),
        hard_negative_rejection_mean=0.0,
    )


def _first_positive_ranks(scores: np.ndarray, positive: np.ndarray, group_size: int) -> np.ndarray:
    row_count = int(scores.shape[0])
    group_count = row_count // int(group_size)
    ranks = np.empty((group_count,), dtype=np.int64)
    for group_index in range(group_count):
        begin = group_index * int(group_size)
        end = begin + int(group_size)
        order = np.argsort(scores[begin:end], kind="stable")[::-1]
        positive_positions = np.flatnonzero(positive[begin:end][order])
        ranks[group_index] = int(positive_positions[0]) + 1 if positive_positions.size else int(group_size)
    return ranks


def _evaluate_random_cost_matched(
    *,
    budgets: np.ndarray,
    positive: np.ndarray,
    full_cost: np.ndarray,
    group_size: int,
    seeds: Sequence[int],
) -> dict[str, float]:
    row_count = int(positive.shape[0])
    group_count = row_count // int(group_size)
    fn_values: list[float] = []
    work_values: list[float] = []
    call_values: list[float] = []
    for seed in seeds:
        rng = np.random.default_rng(int(seed))
        fn = 0
        work = 0.0
        calls = 0
        for group_index in range(group_count):
            begin = group_index * int(group_size)
            end = begin + int(group_size)
            budget = int(max(1, min(int(group_size), budgets[group_index])))
            order = rng.permutation(int(group_size))
            selected = order[:budget]
            selected_positive = positive[begin:end][selected]
            if not np.any(selected_positive):
                fn += 1
            work += float(np.sum(full_cost[begin:end][selected], dtype=np.float64))
            calls += budget
        fn_values.append(float(fn))
        work_values.append(work)
        call_values.append(float(calls))
    return {
        "seed_count": float(len(seeds)),
        "budget_source": "LearnedCostAware first-positive rank",
        "fn_mean": float(mean(fn_values)) if fn_values else 0.0,
        "fn_max": float(max(fn_values)) if fn_values else 0.0,
        "recall_mean": 1.0 - (float(mean(fn_values)) / max(1.0, float(group_count))) if fn_values else 0.0,
        "scheduled_exact_work_mean": float(mean(work_values)) if work_values else 0.0,
        "scheduled_exact_calls_mean": float(mean(call_values)) if call_values else 0.0,
    }


def _random_summary(random_metrics: Sequence[MethodMetrics]) -> dict[str, float]:
    work = [item.scheduled_exact_work for item in random_metrics]
    calls = [float(item.scheduled_exact_calls) for item in random_metrics]
    ranks = [item.first_positive_rank_mean for item in random_metrics]
    rank_p90 = [item.first_positive_rank_p90 for item in random_metrics]
    cost_weighted = [item.cost_weighted_first_positive_mean for item in random_metrics]
    if len(work) <= 1:
        work_std = 0.0
        calls_std = 0.0
        ranks_std = 0.0
    else:
        work_std = float(stdev(work))
        calls_std = float(stdev(calls))
        ranks_std = float(stdev(ranks))
    return {
        "seed_count": float(len(random_metrics)),
        "scheduled_exact_work_mean": float(mean(work)) if work else 0.0,
        "scheduled_exact_work_std": work_std,
        "scheduled_exact_work_ci95": 1.96 * work_std / max(1.0, len(work) ** 0.5),
        "scheduled_exact_calls_mean": float(mean(calls)) if calls else 0.0,
        "scheduled_exact_calls_std": calls_std,
        "first_positive_rank_mean": float(mean(ranks)) if ranks else 0.0,
        "first_positive_rank_std": ranks_std,
        "first_positive_rank_p90_mean": float(mean(rank_p90)) if rank_p90 else 0.0,
        "cost_weighted_first_positive_mean": float(mean(cost_weighted)) if cost_weighted else 0.0,
    }


def _run_case(
    spec: NativeDenseGroupCaseSpec,
    *,
    challenge: RankChallengeSpec,
    random_seeds: Sequence[int],
    device: str,
    batch_size: int,
) -> dict[str, Any]:
    if not spec.checkpoint.exists():
        raise FileNotFoundError(spec.checkpoint)
    if not spec.dense_shard.exists():
        raise FileNotFoundError(spec.dense_shard)

    feature_arrays = _feature_arrays_from_npz(spec.dense_shard)
    diagnosis = _diagnose_original_groups(feature_arrays)
    model = _load_model(spec.checkpoint, device=device)
    onnx_path = ensure_stpf_model_onnx(
        model,
        checkpoint_path=spec.checkpoint,
        output_path=spec.checkpoint.with_suffix(".onnx"),
        model_tag=spec.checkpoint.parent.name,
    )
    runtime = create_ort_inference_session(
        onnx_path,
        requested_device=device,
        prefer_tensorrt=True,
        allow_cuda_fallback=True,
        allow_cpu_fallback=True,
    )
    predictions = batched_stpf_inference_ort_arrays(
        runtime,
        feature_arrays,
        batch_size=batch_size,
        ood_abs_feature_threshold=None,
    )

    selected_indices, remixed_group_ids = _make_rank_challenge_indices(feature_arrays, challenge)
    selected_features = _feature_subset(feature_arrays, selected_indices)
    selected_features["query_id"] = np.ascontiguousarray(remixed_group_ids, dtype=np.uint64)
    selected_predictions = _prediction_subset(predictions, selected_indices)
    trace = np.asarray(feature_arrays["oracle_trace"], dtype=np.float64)[selected_indices]
    positive = trace[:, 0] > 0.5
    full_cost = np.maximum(1.0e-12, trace[:, 5].astype(np.float64))

    deterministic_rng = np.random.default_rng(int(challenge.random_seed) + 17)
    methods = _score_methods(
        source_name=spec.name,
        predictions=selected_predictions,
        feature_arrays=selected_features,
        rng=deterministic_rng,
    )
    deterministic_metrics = [
        _evaluate_scores(
            method=name,
            scores=scores,
            positive=positive,
            full_cost=full_cost,
            group_size=int(challenge.group_size),
            positives_per_group=int(challenge.positives_per_group),
        )
        for name, scores in methods.items()
        if name != "RandomUniform"
    ]
    random_metrics = []
    for seed in random_seeds:
        rng = np.random.default_rng(int(seed))
        scores = rng.random(positive.shape[0], dtype=np.float64)
        random_metrics.append(
            _evaluate_scores(
                method=f"RandomUniform(seed={seed})",
                scores=scores,
                positive=positive,
                full_cost=full_cost,
                group_size=int(challenge.group_size),
                positives_per_group=int(challenge.positives_per_group),
            )
        )
    random_mean = _random_summary(random_metrics)
    representative_random = _evaluate_scores(
        method="RandomUniform(mean over seeds)",
        scores=np.zeros_like(full_cost),
        positive=positive,
        full_cost=full_cost,
        group_size=int(challenge.group_size),
        positives_per_group=int(challenge.positives_per_group),
    )
    representative_random = MethodMetrics(
        method="RandomUniform(mean over seeds)",
        group_count=representative_random.group_count,
        candidate_count=representative_random.candidate_count,
        positive_candidates_per_group=representative_random.positive_candidates_per_group,
        no_proposal_exact_calls=representative_random.no_proposal_exact_calls,
        scheduled_exact_calls=int(round(random_mean["scheduled_exact_calls_mean"])),
        no_proposal_exact_work=representative_random.no_proposal_exact_work,
        scheduled_exact_work=float(random_mean["scheduled_exact_work_mean"]),
        exact_call_reduction=1.0
        - float(random_mean["scheduled_exact_calls_mean"]) / max(1, representative_random.no_proposal_exact_calls),
        exact_work_reduction=1.0
        - float(random_mean["scheduled_exact_work_mean"]) / max(1.0e-12, representative_random.no_proposal_exact_work),
        fn=0,
        first_positive_rank_mean=float(random_mean["first_positive_rank_mean"]),
        first_positive_rank_p50=0.0,
        first_positive_rank_p90=float(random_mean["first_positive_rank_p90_mean"]),
        first_positive_rank_p99=0.0,
        cost_weighted_first_positive_mean=float(random_mean["cost_weighted_first_positive_mean"]),
        cost_weighted_first_positive_p50=0.0,
        cost_weighted_first_positive_p90=0.0,
        hard_negative_rejection_mean=0.0,
    )
    no_proposal = _evaluate_no_proposal_all_exact(
        positive=positive,
        full_cost=full_cost,
        group_size=int(challenge.group_size),
        positives_per_group=int(challenge.positives_per_group),
    )
    all_metrics = [no_proposal] + deterministic_metrics + [representative_random]
    learned = next(item for item in all_metrics if item.method == "LearnedCostAware")
    learned_cost_aware_scores = methods["LearnedCostAware"]
    cost_matched = _evaluate_random_cost_matched(
        budgets=_first_positive_ranks(learned_cost_aware_scores, positive, int(challenge.group_size)),
        positive=positive,
        full_cost=full_cost,
        group_size=int(challenge.group_size),
        seeds=random_seeds,
    )
    learned_methods = [item for item in all_metrics if item.method.startswith("Learned")]
    best_learned = min(learned_methods, key=lambda item: item.scheduled_exact_work)
    random_work = float(random_mean["scheduled_exact_work_mean"])
    random_calls = float(random_mean["scheduled_exact_calls_mean"])
    learned_win_rate = float(
        np.mean([learned.scheduled_exact_work < item.scheduled_exact_work for item in random_metrics])
    )
    best_learned_win_rate = float(
        np.mean([best_learned.scheduled_exact_work < item.scheduled_exact_work for item in random_metrics])
    )
    return {
        "case": spec.name,
        "checkpoint": spec.checkpoint.as_posix(),
        "dense_shard": spec.dense_shard.as_posix(),
        "onnx": onnx_path.as_posix(),
        "metadata": _load_metadata(spec.dense_shard),
        "ort_provider": runtime.provider_name,
        "provider_order": list(runtime.provider_order),
        "original_group_diagnosis": asdict(diagnosis),
        "challenge": asdict(challenge),
        "challenge_row_count": int(selected_indices.shape[0]),
        "challenge_group_count": int(selected_indices.shape[0] // int(challenge.group_size)),
        "hard_negative_count_per_group": int(challenge.group_size - challenge.positives_per_group),
        "methods": [asdict(item) for item in all_metrics],
        "random_seed_metrics": [asdict(item) for item in random_metrics],
        "random_summary": random_mean,
        "random_cost_matched_nonconservative": cost_matched,
        "learned_vs_random": {
            "learned_cost_aware_work": float(learned.scheduled_exact_work),
            "random_mean_work": random_work,
            "work_speedup_vs_random_mean": random_work / max(1.0e-12, float(learned.scheduled_exact_work)),
            "work_reduction_delta_vs_random_mean": learned.exact_work_reduction
            - representative_random.exact_work_reduction,
            "learned_cost_aware_calls": float(learned.scheduled_exact_calls),
            "random_mean_calls": random_calls,
            "call_speedup_vs_random_mean": random_calls / max(1.0e-12, float(learned.scheduled_exact_calls)),
            "win_rate_vs_random": learned_win_rate,
        },
        "best_learned_vs_random": {
            "best_learned_method": best_learned.method,
            "best_learned_work": float(best_learned.scheduled_exact_work),
            "random_mean_work": random_work,
            "work_speedup_vs_random_mean": random_work / max(1.0e-12, float(best_learned.scheduled_exact_work)),
            "work_reduction_delta_vs_random_mean": best_learned.exact_work_reduction
            - representative_random.exact_work_reduction,
            "best_learned_calls": float(best_learned.scheduled_exact_calls),
            "random_mean_calls": random_calls,
            "call_speedup_vs_random_mean": random_calls / max(1.0e-12, float(best_learned.scheduled_exact_calls)),
            "win_rate_vs_random": best_learned_win_rate,
        },
    }


def run_learned_vs_random_ablation(
    *,
    cases: Sequence[NativeDenseGroupCaseSpec] = DEFAULT_CASES,
    output_dir: str | Path = "src/benchmark",
    run_name: str = RUN_NAME,
    challenge: RankChallengeSpec = RankChallengeSpec(),
    random_seed_count: int = 30,
    device: str = "cuda",
    batch_size: int = 65536,
) -> dict[str, Any]:
    seeds = [int(challenge.random_seed) + 1009 * i for i in range(int(random_seed_count))]
    case_results = [
        _run_case(
            spec,
            challenge=challenge,
            random_seeds=seeds,
            device=device,
            batch_size=batch_size,
        )
        for spec in cases
    ]
    payload = {
        "run_name": run_name,
        "device": device,
        "batch_size": int(batch_size),
        "random_seed_count": int(random_seed_count),
        "random_seeds": seeds,
        "case_count": len(case_results),
        "cases": case_results,
    }
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{run_name}.json"
    md_path = out_dir / f"{run_name}.md"
    csv_path = out_dir / f"{run_name}.csv"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _write_report(md_path, payload)
    _write_csv(csv_path, payload)
    return payload


def _method_map(case: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(item["method"]): item for item in case["methods"]}


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Learned-vs-random ablation reinforcement report",
        "",
        "## Protocol",
        "",
        "- Purpose: description `learned STPF isdescriptionbetter than random scheduling`. ",
        "- descriptionoriginal dense groups: if positive group descriptionall candidate descriptionis positive, descriptionanydescription rank=1, description learned better than random. ",
        "- thereforedescription balanced hard-negative rank challenge: each group fixeddescription positive candidate anddescription high-cost negative candidate, descriptionusesame heldout shard, same checkpoint, same ORT TensorRT Output. ",
        "- descriptionis group-level scheduling/ranking, description STPF descriptionconnectdescription collision; description zero-FN description conservative exact scan/fallback guarantee. ",
        "",
        "## original Dense Group descriptionsplitdescription",
        "",
        "| Dataset | Groups | Candidates | Positive groups | Mixed groups | Pure-positive groups | Positive fraction in positive groups | Informative for rank ablation |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for case in payload["cases"]:
        diag = case["original_group_diagnosis"]
        lines.append(
            f"| `{case['case']}` | `{diag['group_count']}` | `{diag['candidate_count']}` | "
            f"`{diag['positive_group_count']}` | `{diag['mixed_group_count']}` | "
            f"`{diag['pure_positive_group_count']}` | "
            f"`{diag['mean_positive_fraction_in_positive_groups']:.4f}` | "
            f"`{diag['rank_ablation_is_informative']}` |"
        )
    lines.extend(
        [
            "",
            "Conclusion: currentdescription dense shard original group descriptionis not candidate-level mixed group; descriptionconnectusedescription learned descriptionbetter than random. underdescription balanced hard-negative challenge descriptionis learned-vs-random hasdescriptionProtocol. ",
            "",
            "## Balanced Hard-Negative Rank Challenge",
            "",
            "| Dataset | Groups | Candidates/group | Pos/group | Method | Exact calls | Call reduction | Work reduction | First-positive rank mean | p90 | Cost-weighted mean | FN |",
            "| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    preferred = (
        "ValidationSelectedFullSTPF",
        "LearnedCostAware",
        "LearnedPriorityOnly",
        "IntervalOnly",
        "RankingOnly",
        "LearnedCalibrated",
        "RandomUniform(mean over seeds)",
        "HeuristicCostLow",
        "HeuristicCostHigh",
        "UncertaintyOnly",
        "OracleUpperBound",
    )
    for case in payload["cases"]:
        methods = _method_map(case)
        challenge = case["challenge"]
        for name in preferred:
            item = methods.get(name)
            if item is None:
                continue
            lines.append(
                f"| `{case['case']}` | `{item['group_count']}` | `{challenge['group_size']}` | "
                f"`{challenge['positives_per_group']}` | `{name}` | "
                f"`{item['scheduled_exact_calls']}` | `{item['exact_call_reduction']:.4%}` | "
                f"`{item['exact_work_reduction']:.4%}` | "
                f"`{item['first_positive_rank_mean']:.3f}` | "
                f"`{item['first_positive_rank_p90']:.3f}` | "
                f"`{item['cost_weighted_first_positive_mean']:.3f}` | `{item['fn']}` |"
            )
    lines.extend(
        [
            "",
            "## LearnedCostAware vs RandomUniform",
            "",
            "| Dataset | Random seeds | Learned work | Random mean work | Work speedup | Work reduction delta | Learned calls | Random mean calls | Call speedup | Win rate |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for case in payload["cases"]:
        cmp = case["learned_vs_random"]
        lines.append(
            f"| `{case['case']}` | `{payload['random_seed_count']}` | "
            f"`{cmp['learned_cost_aware_work']:.3f}` | `{cmp['random_mean_work']:.3f}` | "
            f"`{cmp['work_speedup_vs_random_mean']:.3f}x` | "
            f"`{cmp['work_reduction_delta_vs_random_mean']:.4%}` | "
            f"`{cmp['learned_cost_aware_calls']:.1f}` | `{cmp['random_mean_calls']:.1f}` | "
            f"`{cmp['call_speedup_vs_random_mean']:.3f}x` | `{cmp['win_rate_vs_random']:.3f}` |"
        )
    lines.extend(
        [
            "",
            "## Best Learned Head vs RandomUniform",
            "",
            "description: `Best learned head` isdescription, usedescriptionModelisdescriptionhashasusedescription; descriptionconnectasdescriptiondefaultMethod, descriptionafterdescriptionuse validation split fixed head-selection description. ",
            "",
            "| Dataset | Best learned head | Learned work | Random mean work | Work speedup | Work reduction delta | Learned calls | Random mean calls | Call speedup | Win rate |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for case in payload["cases"]:
        cmp = case["best_learned_vs_random"]
        lines.append(
            f"| `{case['case']}` | `{cmp['best_learned_method']}` | "
            f"`{cmp['best_learned_work']:.3f}` | `{cmp['random_mean_work']:.3f}` | "
            f"`{cmp['work_speedup_vs_random_mean']:.3f}x` | "
            f"`{cmp['work_reduction_delta_vs_random_mean']:.4%}` | "
            f"`{cmp['best_learned_calls']:.1f}` | `{cmp['random_mean_calls']:.1f}` | "
            f"`{cmp['call_speedup_vs_random_mean']:.3f}x` | `{cmp['win_rate_vs_random']:.3f}` |"
        )
    lines.extend(
        [
            "",
            "## RandomCostMatched description",
            "",
            "`RandomCostMatched` fixeddescriptionuseand `LearnedCostAware` description per-group exact budget; if budget descriptionto positive, descriptionas FN. thisdescriptionused fordescription ranking quality, is not used as final conservative CCD Method. ",
            "",
            "| Dataset | Budget source | Random seeds | Mean calls | Mean work | Mean FN | Max FN | Mean recall |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for case in payload["cases"]:
        item = case["random_cost_matched_nonconservative"]
        lines.append(
            f"| `{case['case']}` | `{item['budget_source']}` | `{int(item['seed_count'])}` | "
            f"`{item['scheduled_exact_calls_mean']:.1f}` | `{item['scheduled_exact_work_mean']:.3f}` | "
            f"`{item['fn_mean']:.3f}` | `{item['fn_max']:.0f}` | `{item['recall_mean']:.4f}` |"
        )
    lines.extend(
        [
            "",
            "## descriptionConclusiondescription",
            "",
            "- if `LearnedCostAware` indescription case ondescription random mean and win-rate connectdescription 1, descriptionwithwrite learned ranker description dense hard-negative scheduling hasdescription. ",
            "- ifadvantagedescription, descriptionas: RTSTPFExact is correctness-preserving learned scheduling/proposal layer; advantagedescriptionfrom dense group early-stop + conservative fallback, rather thandescriptioninalldistributionondescriptionbetter than random. ",
            "- `OracleUpperBound` descriptionasdescriptionondescription, descriptionasdescription baseline. ",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_csv(path: Path, payload: dict[str, Any]) -> None:
    rows = [
        "case,method,group_count,candidate_count,exact_calls,exact_work,call_reduction,work_reduction,first_positive_rank_mean,first_positive_rank_p90,fn"
    ]
    for case in payload["cases"]:
        for item in case["methods"]:
            rows.append(
                ",".join(
                    [
                        str(case["case"]),
                        str(item["method"]),
                        str(item["group_count"]),
                        str(item["candidate_count"]),
                        str(item["scheduled_exact_calls"]),
                        f"{float(item['scheduled_exact_work']):.9f}",
                        f"{float(item['exact_call_reduction']):.9f}",
                        f"{float(item['exact_work_reduction']):.9f}",
                        f"{float(item['first_positive_rank_mean']):.9f}",
                        f"{float(item['first_positive_rank_p90']):.9f}",
                        str(item["fn"]),
                    ]
                )
            )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("src/benchmark"))
    parser.add_argument("--run-name", default=RUN_NAME)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=65536)
    parser.add_argument("--random-seed-count", type=int, default=30)
    parser.add_argument("--group-size", type=int, default=512)
    parser.add_argument("--positives-per-group", type=int, default=4)
    parser.add_argument("--max-groups", type=int, default=512)
    parser.add_argument("--seed", type=int, default=424242)
    parser.add_argument(
        "--case",
        action="append",
        choices=[spec.name for spec in DEFAULT_CASES],
        help="Run only selected default case; can be repeated.",
    )
    args = parser.parse_args()
    selected_cases = DEFAULT_CASES
    if args.case:
        allowed = set(args.case)
        selected_cases = tuple(spec for spec in DEFAULT_CASES if spec.name in allowed)
    payload = run_learned_vs_random_ablation(
        cases=selected_cases,
        output_dir=args.output_dir,
        run_name=args.run_name,
        challenge=RankChallengeSpec(
            group_size=int(args.group_size),
            positives_per_group=int(args.positives_per_group),
            max_groups=int(args.max_groups),
            random_seed=int(args.seed),
        ),
        random_seed_count=int(args.random_seed_count),
        device=args.device,
        batch_size=int(args.batch_size),
    )
    print(json.dumps({"run_name": payload["run_name"], "case_count": payload["case_count"]}, indent=2))


if __name__ == "__main__":
    main()
