from __future__ import annotations

from dataclasses import asdict, dataclass, fields, is_dataclass, replace
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from p2cccd.contracts import BenchmarkRow, BenchmarkRowV2, ProxyType
from p2cccd.data import (
    DatasetGenerationConfig,
    GeneratedDataset,
    generate_exact_oracle_dataset,
)
from p2cccd.datasets.ccd import (
    CCDQueryFamily,
    DatasetQueryBatch,
    RigidIPCSceneAdapter,
    ScalableCCDSampleAdapter,
    TightInclusionAdapter,
)
from p2cccd.validators import validate_benchmark_row_v2

from .bvh_exact import BVHExactConfig, run_bvh_exact_on_external_batch, run_bvh_exact_on_generated_dataset
from .curobo_downstream import (
    CuRoboDownstreamConfig,
    run_curobo_downstream_on_generated_dataset,
)
from .learned_style_comparison import (
    CabiNetStyleConfig,
    NeuralSVCDStyleConfig,
    run_cabinet_style_on_generated_dataset,
    run_neural_svcd_style_on_generated_dataset,
)
from .no_proposal import NoProposalConfig, run_no_proposal_on_external_batch, run_no_proposal_on_generated_dataset
from .no_queue_decouple import (
    NoQueueDecoupleConfig,
    NoQueueDecoupleCaseResult,
    run_no_queue_decouple_microbenchmark,
)
from .patch_granularity_ablation import (
    PatchGranularityAblationConfig,
    PatchGranularityAblationOption,
    PatchGranularityAblationRow,
    run_patch_granularity_ablation_on_generated_dataset,
)
from .pure_exact_cpu import run_pure_exact_cpu_on_external_batch, run_pure_exact_cpu_on_generated_dataset
from .rt_exact import RTExactConfig, RtCandidateStats, run_rt_exact_on_external_batch, run_rt_exact_on_generated_dataset
from .rt_stpf_exact import RTSTPFExactConfig, run_rt_stpf_exact_on_external_batch, run_rt_stpf_exact_on_generated_dataset
from .rt_style_reproduction import (
    RTCCDStyleConfig,
    RTDCDStyleConfig,
    run_rt_ccd_style_on_generated_dataset,
    run_rt_dcd_style_on_generated_dataset,
)
from .slab_proxy_ablation import (
    SlabProxyAblationConfig,
    SlabProxyAblationOption,
    SlabProxyAblationRow,
    run_slab_proxy_ablation_on_generated_dataset,
)
from .sort_broad_phase_exact import (
    SortBroadPhaseConfig,
    run_sort_broad_phase_exact_on_external_batch,
    run_sort_broad_phase_exact_on_generated_dataset,
)
from .stpf_head_ablations import (
    IntervalOnlyConfig,
    RankingOnlyConfig,
    run_interval_only_on_generated_dataset,
    run_ranking_only_on_generated_dataset,
)
from .summary import (
    BenchmarkExportPaths,
    benchmark_row_v2_from_legacy,
    create_benchmark_run_meta,
    export_benchmark_run,
)


SUITE_CONFIG_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class BenchmarkSuiteDatasetConfig:
    kind: str = "internal_generated"
    mesh_count_per_split: int = 2
    robot_link_count: int = 1
    include_robot_links: bool = True
    seed: int | None = None
    splits: tuple[str, ...] = ()
    scene_name: str | None = None
    family: str = "vf"
    step: int | None = None
    limit: int | None = None
    dimension: str | None = None
    include_fixed_fixed_pairs: bool = False


@dataclass(frozen=True, slots=True)
class BenchmarkSuiteCaseConfig:
    name: str
    method: str
    dataset: BenchmarkSuiteDatasetConfig = BenchmarkSuiteDatasetConfig()
    config: dict[str, Any] | None = None
    seed: int | None = None
    repeat: int = 1


@dataclass(frozen=True, slots=True)
class BenchmarkSuiteConfig:
    schema_version: int
    suite_name: str
    suite_type: str
    seed: int
    output_root: str
    cases: tuple[BenchmarkSuiteCaseConfig, ...]
    notes: str = ""


@dataclass(frozen=True, slots=True)
class BenchmarkSuiteCaseResult:
    case_name: str
    method: str
    row_count: int
    final_fn_zero: bool


@dataclass(frozen=True, slots=True)
class BenchmarkSuiteRunResult:
    suite: BenchmarkSuiteConfig
    meta_run_id: str
    rows: tuple[BenchmarkRowV2, ...]
    case_results: tuple[BenchmarkSuiteCaseResult, ...]
    export_paths: BenchmarkExportPaths | None


def _tuple_of_strings(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise ValueError("splits must be a list of strings")
    return tuple(str(item) for item in value)


def _dataset_config_from_dict(data: Mapping[str, Any] | None) -> BenchmarkSuiteDatasetConfig:
    if data is None:
        return BenchmarkSuiteDatasetConfig()
    return BenchmarkSuiteDatasetConfig(
        kind=str(data.get("kind", "internal_generated")),
        mesh_count_per_split=int(data.get("mesh_count_per_split", 2)),
        robot_link_count=int(data.get("robot_link_count", 1)),
        include_robot_links=bool(data.get("include_robot_links", True)),
        seed=None if data.get("seed") is None else int(data["seed"]),
        splits=_tuple_of_strings(data.get("splits", ())),
        scene_name=None if data.get("scene_name") is None else str(data["scene_name"]),
        family=str(data.get("family", "vf")),
        step=None if data.get("step") is None else int(data["step"]),
        limit=None if data.get("limit") is None else int(data["limit"]),
        dimension=None if data.get("dimension") is None else str(data["dimension"]),
        include_fixed_fixed_pairs=bool(data.get("include_fixed_fixed_pairs", False)),
    )


def _case_config_from_dict(data: Mapping[str, Any]) -> BenchmarkSuiteCaseConfig:
    if "name" not in data or "method" not in data:
        raise ValueError("suite case requires name and method")
    return BenchmarkSuiteCaseConfig(
        name=str(data["name"]),
        method=str(data["method"]),
        dataset=_dataset_config_from_dict(data.get("dataset")),
        config=dict(data.get("config", {})),
        seed=None if data.get("seed") is None else int(data["seed"]),
        repeat=int(data.get("repeat", 1)),
    )


def benchmark_suite_config_from_dict(data: Mapping[str, Any]) -> BenchmarkSuiteConfig:
    if int(data.get("schema_version", 0)) != SUITE_CONFIG_SCHEMA_VERSION:
        raise ValueError("unsupported benchmark suite schema_version")
    cases = tuple(_case_config_from_dict(case) for case in data.get("cases", ()))
    if not cases:
        raise ValueError("benchmark suite requires at least one case")
    suite = BenchmarkSuiteConfig(
        schema_version=SUITE_CONFIG_SCHEMA_VERSION,
        suite_name=str(data.get("suite_name", "")),
        suite_type=str(data.get("suite_type", "")),
        seed=int(data.get("seed", 0)),
        output_root=str(data.get("output_root", "outputs/benchmark_suites")),
        cases=cases,
        notes=str(data.get("notes", "")),
    )
    validate_benchmark_suite_config(suite)
    return suite


def load_benchmark_suite_config(path: str | Path) -> BenchmarkSuiteConfig:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("benchmark suite config must be a JSON object")
    return benchmark_suite_config_from_dict(data)


def validate_benchmark_suite_config(config: BenchmarkSuiteConfig) -> BenchmarkSuiteConfig:
    if config.schema_version != SUITE_CONFIG_SCHEMA_VERSION:
        raise ValueError("unsupported benchmark suite schema_version")
    if not config.suite_name:
        raise ValueError("suite_name is required")
    if config.suite_type not in {"correctness", "performance", "ablation", "ood_stress"}:
        raise ValueError("suite_type must be correctness, performance, ablation, or ood_stress")
    if not config.cases:
        raise ValueError("benchmark suite requires at least one case")
    names: set[str] = set()
    for case in config.cases:
        if not case.name:
            raise ValueError("case.name is required")
        if case.name in names:
            raise ValueError(f"duplicate benchmark suite case name: {case.name}")
        names.add(case.name)
        if case.repeat <= 0:
            raise ValueError("case.repeat must be positive")
        if case.dataset.kind not in _SUPPORTED_DATASET_KINDS:
            raise ValueError(f"unsupported benchmark suite dataset kind: {case.dataset.kind}")
        if case.dataset.mesh_count_per_split < 0:
            raise ValueError("mesh_count_per_split must be non-negative")
        if case.dataset.robot_link_count < 0:
            raise ValueError("robot_link_count must be non-negative")
        if case.dataset.limit is not None and case.dataset.limit <= 0:
            raise ValueError("dataset.limit must be positive when provided")
        if case.method not in _SUPPORTED_METHODS:
            raise ValueError(f"unsupported benchmark suite method: {case.method}")
    return config


def _dataclass_from_dict(dataclass_type: type, data: Mapping[str, Any] | None):
    payload = dict(data or {})
    kwargs: dict[str, Any] = {}
    for field in fields(dataclass_type):
        if field.name not in payload:
            continue
        value = payload[field.name]
        default_value = getattr(dataclass_type(), field.name) if callable(dataclass_type) else None
        if is_dataclass(default_value) and isinstance(value, Mapping):
            kwargs[field.name] = _dataclass_from_dict(type(default_value), value)
        else:
            kwargs[field.name] = value
    return dataclass_type(**kwargs)


def _proxy_type(value: Any) -> ProxyType:
    if isinstance(value, ProxyType):
        return value
    if isinstance(value, str):
        normalized = value.upper()
        if normalized in {"AABB", "SWEPT_AABB"}:
            return ProxyType.SWEPT_AABB
        if normalized == "CAPSULE":
            return ProxyType.CAPSULE
    return ProxyType(value)


def _patch_config_from_dict(data: Mapping[str, Any] | None) -> PatchGranularityAblationConfig:
    payload = dict(data or {})
    option_payloads = payload.pop("options", None)
    if option_payloads is not None:
        payload["options"] = tuple(PatchGranularityAblationOption(**dict(option)) for option in option_payloads)
    return PatchGranularityAblationConfig(**payload)


def _slab_proxy_config_from_dict(data: Mapping[str, Any] | None) -> SlabProxyAblationConfig:
    payload = dict(data or {})
    option_payloads = payload.pop("options", None)
    if option_payloads is not None:
        options: list[SlabProxyAblationOption] = []
        for option in option_payloads:
            option_data = dict(option)
            option_data["proxy_type_a"] = _proxy_type(option_data.get("proxy_type_a", "SWEPT_AABB"))
            option_data["proxy_type_b"] = _proxy_type(option_data.get("proxy_type_b", "SWEPT_AABB"))
            options.append(SlabProxyAblationOption(**option_data))
        payload["options"] = tuple(options)
    return SlabProxyAblationConfig(**payload)


def _rt_ccd_style_config_from_dict(data: Mapping[str, Any] | None) -> RTCCDStyleConfig:
    payload = dict(data or {})
    if "proxy_type_a" in payload:
        payload["proxy_type_a"] = _proxy_type(payload["proxy_type_a"])
    if "proxy_type_b" in payload:
        payload["proxy_type_b"] = _proxy_type(payload["proxy_type_b"])
    return RTCCDStyleConfig(**payload)


def _dataset_for_case(case: BenchmarkSuiteCaseConfig, suite_seed: int) -> GeneratedDataset:
    dataset_seed = case.dataset.seed if case.dataset.seed is not None else (case.seed if case.seed is not None else suite_seed)
    dataset = generate_exact_oracle_dataset(
        DatasetGenerationConfig(
            mesh_count_per_split=case.dataset.mesh_count_per_split,
            robot_link_count=case.dataset.robot_link_count,
            seed=dataset_seed,
            include_robot_links=case.dataset.include_robot_links,
        )
    )
    if not case.dataset.splits:
        return dataset
    allowed = set(case.dataset.splits)
    filtered = [
        (row, sample, trace)
        for row, sample, trace in zip(dataset.rows, dataset.samples, dataset.traces)
        if sample.split in allowed
    ]
    if not filtered:
        raise ValueError(f"case {case.name} dataset split filter produced no samples")
    rows, samples, traces = zip(*filtered)
    split_names = tuple(split for split in dataset.split_names if split in allowed)
    return GeneratedDataset(rows=list(rows), samples=list(samples), traces=list(traces), split_names=split_names)


def _external_batch_for_case(case: BenchmarkSuiteCaseConfig) -> DatasetQueryBatch:
    dataset = case.dataset
    query_family = CCDQueryFamily(dataset.family)
    if dataset.kind == "scalable_ccd_sample":
        adapter = ScalableCCDSampleAdapter()
        if dataset.scene_name is None or dataset.step is None:
            batches = adapter.list_query_batches(dataset.scene_name)
            matching = [batch for batch in batches if batch.family == query_family]
            if not matching:
                raise ValueError("Scalable CCD sample suite case has no matching query batch")
            selected = matching[0]
            return adapter.load_query_batch(
                selected.scene_name,
                family=selected.family,
                step=selected.step,
                limit=dataset.limit,
            )
        return adapter.load_query_batch(
            dataset.scene_name,
            family=query_family,
            step=dataset.step,
            limit=dataset.limit,
        )
    if dataset.kind == "rigid_ipc":
        adapter = RigidIPCSceneAdapter()
        scene_name = dataset.scene_name
        if scene_name is None:
            infos = adapter.list_fixture_infos(dimension=dataset.dimension, limit=1)
            if not infos:
                raise ValueError("Rigid-IPC suite case has no matching fixture")
            scene_name = infos[0].scene_name
        return adapter.load_body_pair_query_batch(
            scene_name,
            family=query_family,
            limit=dataset.limit,
            include_fixed_fixed_pairs=dataset.include_fixed_fixed_pairs,
        )
    raise ValueError(f"dataset kind is not an external query batch: {dataset.kind}")


def _reference_availability_row(
    case: BenchmarkSuiteCaseConfig,
    *,
    meta,
) -> tuple[BenchmarkRowV2, ...]:
    if case.dataset.kind == "tight_inclusion_reference":
        TightInclusionAdapter().reference_entry_points()
        dataset_name = "Tight Inclusion"
    else:
        raise ValueError(f"unsupported ReferenceAvailability dataset kind: {case.dataset.kind}")
    benchmark = BenchmarkRow(
        query_count=1,
        fn_count=0,
        fp_count=0,
        candidate_recall=1.0,
        avg_candidates=0.0,
        avg_exact_evals=0.0,
        avg_subdivision_depth=0.0,
        fallback_ratio=0.0,
        rt_ms=0.0,
        proposal_ms=0.0,
        exact_ms=0.0,
        total_ms=0.0,
        qps=0.0,
    )
    return (
        validate_benchmark_row_v2(
            replace(
                benchmark_row_v2_from_legacy(
                    benchmark,
                    meta,
                    candidate_inflation_ratio=0.0,
                    exact_queue_occupancy=0.0,
                ),
                dataset_name=dataset_name,
                scene_name=f"reference_availability:{case.name}",
                method_name=case.method,
                seed=case.seed if case.seed is not None else meta.seed,
            )
        ),
    )


def _family_exact_calls_from_result(result: object) -> dict[str, int]:
    family_calls = {"point_triangle": 0, "edge_edge": 0, "conservative": 0, "unknown": 0}
    for query_result in getattr(result, "query_results", ()):
        exact_evals = max(0, int(getattr(query_result, "exact_evals", 0)))
        family = str(getattr(query_result, "family", "")).lower()
        if family in {"point_triangle", "vertex_face", "vf"}:
            family_calls["point_triangle"] += exact_evals
        elif family in {"edge_edge", "ee"}:
            family_calls["edge_edge"] += exact_evals
        elif family in {"swept_sphere_proxy", "conservative"}:
            family_calls["conservative"] += exact_evals
        else:
            family_calls["unknown"] += exact_evals
    return family_calls


def _rt_timing_from_result(result: object) -> tuple[float, float, float]:
    stats = getattr(result, "candidate_stats", None)
    if isinstance(stats, RtCandidateStats):
        return stats.timing.build_ms, stats.timing.update_ms, stats.timing.trace_ms
    return 0.0, 0.0, getattr(getattr(result, "benchmark", None), "rt_ms", 0.0)


def _result_context(result: object, case: BenchmarkSuiteCaseConfig) -> tuple[str, str]:
    dataset_name = str(getattr(result, "source_name", "internal_analytic_oracle"))
    scene_name = str(getattr(result, "scene_name", "generated_exact_oracle_dataset"))
    batch_id = str(getattr(result, "batch_id", "generated_dataset"))
    return dataset_name, f"{scene_name}:{batch_id}:{case.name}"


def _legacy_result_to_v2_rows(
    result: object,
    *,
    meta,
    case: BenchmarkSuiteCaseConfig,
    method_name: str,
) -> tuple[BenchmarkRowV2, ...]:
    benchmark = getattr(result, "benchmark")
    rt_build_ms, rt_update_ms, rt_trace_ms = _rt_timing_from_result(result)
    dataset_name, scene_name = _result_context(result, case)
    row = benchmark_row_v2_from_legacy(
        benchmark,
        meta,
        family_exact_calls=_family_exact_calls_from_result(result),
        rt_build_ms=rt_build_ms,
        rt_update_ms=rt_update_ms,
        rt_trace_ms=rt_trace_ms,
    )
    return (
        validate_benchmark_row_v2(
            replace(
                row,
                dataset_name=dataset_name,
                scene_name=scene_name,
                method_name=method_name,
                seed=case.seed if case.seed is not None else meta.seed,
            )
        ),
    )


def _ablation_row_to_legacy(row: PatchGranularityAblationRow | SlabProxyAblationRow) -> BenchmarkRow:
    total_ms = max(0.0, row.total_ms)
    query_count = max(1, row.query_count)
    avg_candidates = row.avg_candidates_per_query
    return BenchmarkRow(
        query_count=row.query_count,
        fn_count=row.fn_count,
        fp_count=row.fp_count,
        candidate_recall=row.candidate_recall,
        avg_candidates=avg_candidates,
        avg_exact_evals=row.compact_candidate_count / query_count,
        avg_subdivision_depth=0.0,
        fallback_ratio=0.0,
        rt_ms=row.broad_phase_ms,
        proposal_ms=0.0,
        exact_ms=row.exact_ms,
        total_ms=total_ms,
        qps=0.0 if total_ms <= 0.0 else 1000.0 * row.query_count / total_ms,
    )


def _patch_ablation_to_v2_rows(result, *, meta, case: BenchmarkSuiteCaseConfig) -> tuple[BenchmarkRowV2, ...]:
    rows: list[BenchmarkRowV2] = []
    for option_row in result.rows:
        row_v2 = benchmark_row_v2_from_legacy(
            _ablation_row_to_legacy(option_row),
            meta,
            rt_trace_ms=option_row.broad_phase_ms,
        )
        rows.append(
            validate_benchmark_row_v2(
                replace(
                    row_v2,
                    dataset_name=result.source_name,
                    scene_name=f"{result.scene_name}:{result.batch_id}:{case.name}",
                    method_name=f"{case.method}:{option_row.option_name}",
                    seed=case.seed if case.seed is not None else meta.seed,
                )
            )
        )
    return tuple(rows)


def _slab_proxy_ablation_to_v2_rows(result, *, meta, case: BenchmarkSuiteCaseConfig) -> tuple[BenchmarkRowV2, ...]:
    rows: list[BenchmarkRowV2] = []
    for option_row in result.rows:
        row_v2 = benchmark_row_v2_from_legacy(
            _ablation_row_to_legacy(option_row),
            meta,
            rt_trace_ms=option_row.broad_phase_ms,
        )
        rows.append(
            validate_benchmark_row_v2(
                replace(
                    row_v2,
                    dataset_name=result.source_name,
                    scene_name=f"{result.scene_name}:{result.batch_id}:{case.name}",
                    method_name=f"{case.method}:{option_row.option_name}",
                    seed=case.seed if case.seed is not None else meta.seed,
                )
            )
        )
    return tuple(rows)


def _no_queue_case_to_v2(
    result: NoQueueDecoupleCaseResult,
    *,
    meta,
    case: BenchmarkSuiteCaseConfig,
) -> BenchmarkRowV2:
    benchmark = BenchmarkRow(
        query_count=result.candidate_count,
        fn_count=0,
        fp_count=0,
        candidate_recall=1.0,
        avg_candidates=1.0,
        avg_exact_evals=0.0,
        avg_subdivision_depth=0.0,
        fallback_ratio=0.0,
        rt_ms=result.elapsed_ms,
        proposal_ms=0.0,
        exact_ms=0.0,
        total_ms=result.elapsed_ms,
        qps=result.candidates_per_sec,
    )
    row_v2 = benchmark_row_v2_from_legacy(benchmark, meta, rt_trace_ms=result.elapsed_ms)
    return validate_benchmark_row_v2(
        replace(
            row_v2,
            dataset_name="synthetic_candidates",
            scene_name=f"no_queue_decouple:{case.name}",
            method_name=f"{case.method}:{result.case_name}",
            seed=case.seed if case.seed is not None else meta.seed,
            candidate_inflation_ratio=1.0,
            exact_queue_occupancy=0.0,
            rt_trace_ms=result.trace_ms,
            rt_ms=result.trace_ms,
            proposal_ms=result.proposal_enqueue_dequeue_ms,
            total_ms=result.elapsed_ms,
            candidate_buffer_bandwidth_mb_s=result.approx_bandwidth_mb_s,
            proposal_enqueue_dequeue_ms=result.proposal_enqueue_dequeue_ms,
            total_tail_latency_ms=result.total_tail_latency_ms,
        )
    )


def _run_external_case(
    case: BenchmarkSuiteCaseConfig,
    meta,
) -> tuple[tuple[BenchmarkRowV2, ...], bool]:
    if case.method == "ReferenceAvailability":
        return _reference_availability_row(case, meta=meta), True

    batch = _external_batch_for_case(case)
    if case.method == "PureExactCPU":
        result = run_pure_exact_cpu_on_external_batch(batch)
        return _legacy_result_to_v2_rows(result, meta=meta, case=case, method_name=case.method), result.final_fn_zero
    if case.method == "BVHExact":
        result = run_bvh_exact_on_external_batch(batch, _dataclass_from_dict(BVHExactConfig, case.config))
        return _legacy_result_to_v2_rows(result, meta=meta, case=case, method_name=case.method), result.final_fn_zero
    if case.method == "SortBroadPhaseExact":
        result = run_sort_broad_phase_exact_on_external_batch(
            batch,
            _dataclass_from_dict(SortBroadPhaseConfig, case.config),
        )
        return _legacy_result_to_v2_rows(result, meta=meta, case=case, method_name=case.method), result.final_fn_zero
    if case.method == "RTExact":
        result = run_rt_exact_on_external_batch(batch, _dataclass_from_dict(RTExactConfig, case.config))
        return _legacy_result_to_v2_rows(result, meta=meta, case=case, method_name=case.method), result.final_fn_zero
    if case.method == "RTSTPFExact":
        result = run_rt_stpf_exact_on_external_batch(batch, _dataclass_from_dict(RTSTPFExactConfig, case.config))
        return _legacy_result_to_v2_rows(result, meta=meta, case=case, method_name=case.method), result.final_fn_zero
    if case.method == "NoProposal":
        result = run_no_proposal_on_external_batch(batch, _dataclass_from_dict(NoProposalConfig, case.config))
        return _legacy_result_to_v2_rows(result, meta=meta, case=case, method_name=case.method), result.final_fn_zero
    raise ValueError(f"method {case.method} does not support external suite dataset kind {case.dataset.kind}")


def _run_case(case: BenchmarkSuiteCaseConfig, suite_seed: int, meta) -> tuple[tuple[BenchmarkRowV2, ...], bool]:
    if case.method == "NoQueueDecouple":
        result = run_no_queue_decouple_microbenchmark(
            _dataclass_from_dict(NoQueueDecoupleConfig, case.config)
        )
        rows = tuple(_no_queue_case_to_v2(case_result, meta=meta, case=case) for case_result in result.case_results)
        return rows, True
    if case.dataset.kind != "internal_generated":
        return _run_external_case(case, meta)

    dataset = _dataset_for_case(case, suite_seed)
    if case.method == "PureExactCPU":
        result = run_pure_exact_cpu_on_generated_dataset(dataset)
        return _legacy_result_to_v2_rows(result, meta=meta, case=case, method_name=case.method), result.final_fn_zero
    if case.method == "BVHExact":
        result = run_bvh_exact_on_generated_dataset(dataset, _dataclass_from_dict(BVHExactConfig, case.config))
        return _legacy_result_to_v2_rows(result, meta=meta, case=case, method_name=case.method), result.final_fn_zero
    if case.method == "SortBroadPhaseExact":
        result = run_sort_broad_phase_exact_on_generated_dataset(
            dataset,
            _dataclass_from_dict(SortBroadPhaseConfig, case.config),
        )
        return _legacy_result_to_v2_rows(result, meta=meta, case=case, method_name=case.method), result.final_fn_zero
    if case.method == "RTExact":
        result = run_rt_exact_on_generated_dataset(dataset, _dataclass_from_dict(RTExactConfig, case.config))
        return _legacy_result_to_v2_rows(result, meta=meta, case=case, method_name=case.method), result.final_fn_zero
    if case.method == "RTSTPFExact":
        result = run_rt_stpf_exact_on_generated_dataset(dataset, _dataclass_from_dict(RTSTPFExactConfig, case.config))
        return _legacy_result_to_v2_rows(result, meta=meta, case=case, method_name=case.method), result.final_fn_zero
    if case.method == "RTDCDStyle":
        result = run_rt_dcd_style_on_generated_dataset(dataset, _dataclass_from_dict(RTDCDStyleConfig, case.config))
        return _legacy_result_to_v2_rows(result, meta=meta, case=case, method_name=case.method), result.final_fn_zero
    if case.method == "RTCCDStyle":
        result = run_rt_ccd_style_on_generated_dataset(dataset, _rt_ccd_style_config_from_dict(case.config))
        return _legacy_result_to_v2_rows(result, meta=meta, case=case, method_name=case.method), result.final_fn_zero
    if case.method == "NeuralSVCDStyle":
        result = run_neural_svcd_style_on_generated_dataset(
            dataset,
            _dataclass_from_dict(NeuralSVCDStyleConfig, case.config),
        )
        return _legacy_result_to_v2_rows(result, meta=meta, case=case, method_name=case.method), result.final_fn_zero
    if case.method == "CabiNetStyle":
        result = run_cabinet_style_on_generated_dataset(
            dataset,
            _dataclass_from_dict(CabiNetStyleConfig, case.config),
        )
        return _legacy_result_to_v2_rows(result, meta=meta, case=case, method_name=case.method), result.final_fn_zero
    if case.method == "CuRoboDownstream":
        result = run_curobo_downstream_on_generated_dataset(
            dataset,
            _dataclass_from_dict(CuRoboDownstreamConfig, case.config),
        )
        return _legacy_result_to_v2_rows(result, meta=meta, case=case, method_name=case.method), result.final_fn_zero
    if case.method == "NoProposal":
        result = run_no_proposal_on_generated_dataset(dataset, _dataclass_from_dict(NoProposalConfig, case.config))
        return _legacy_result_to_v2_rows(result, meta=meta, case=case, method_name=case.method), result.final_fn_zero
    if case.method == "IntervalOnly":
        result = run_interval_only_on_generated_dataset(dataset, _dataclass_from_dict(IntervalOnlyConfig, case.config))
        return _legacy_result_to_v2_rows(result, meta=meta, case=case, method_name=case.method), result.final_fn_zero
    if case.method == "RankingOnly":
        result = run_ranking_only_on_generated_dataset(dataset, _dataclass_from_dict(RankingOnlyConfig, case.config))
        return _legacy_result_to_v2_rows(result, meta=meta, case=case, method_name=case.method), result.final_fn_zero
    if case.method == "PatchGranularityAblation":
        result = run_patch_granularity_ablation_on_generated_dataset(dataset, _patch_config_from_dict(case.config))
        rows = _patch_ablation_to_v2_rows(result, meta=meta, case=case)
        return rows, all(row.fn_count == 0 for row in rows)
    if case.method == "SlabProxyAblation":
        result = run_slab_proxy_ablation_on_generated_dataset(dataset, _slab_proxy_config_from_dict(case.config))
        rows = _slab_proxy_ablation_to_v2_rows(result, meta=meta, case=case)
        return rows, all(row.fn_count == 0 for row in rows)
    raise ValueError(f"unsupported benchmark suite method: {case.method}")


_SUPPORTED_METHODS = {
    "ReferenceAvailability",
    "PureExactCPU",
    "BVHExact",
    "SortBroadPhaseExact",
    "RTExact",
    "RTSTPFExact",
    "RTDCDStyle",
    "RTCCDStyle",
    "NeuralSVCDStyle",
    "CabiNetStyle",
    "CuRoboDownstream",
    "NoProposal",
    "IntervalOnly",
    "RankingOnly",
    "PatchGranularityAblation",
    "SlabProxyAblation",
    "NoQueueDecouple",
}


_SUPPORTED_DATASET_KINDS = {
    "internal_generated",
    "scalable_ccd_sample",
    "rigid_ipc",
    "tight_inclusion_reference",
}


def run_benchmark_suite(
    config: BenchmarkSuiteConfig,
    *,
    output_root: str | Path | None = None,
    environment: Mapping[str, object] | None = None,
    run_id: str | None = None,
    export: bool = True,
) -> BenchmarkSuiteRunResult:
    validate_benchmark_suite_config(config)
    suite_payload = asdict(config)
    meta = create_benchmark_run_meta(
        dataset_name=f"suite:{config.suite_type}",
        scene_name=config.suite_name,
        method_name="benchmark_suite",
        config=suite_payload,
        seed=config.seed,
        run_id=run_id,
        environment=environment,
        notes=config.notes,
    )
    all_rows: list[BenchmarkRowV2] = []
    case_results: list[BenchmarkSuiteCaseResult] = []
    for case in config.cases:
        case_rows: list[BenchmarkRowV2] = []
        final_fn_zero = True
        for repeat_index in range(case.repeat):
            repeated_case = (
                case
                if case.repeat == 1
                else replace(
                    case,
                    name=f"{case.name}_repeat{repeat_index}",
                    seed=(case.seed if case.seed is not None else config.seed) + repeat_index,
                )
            )
            rows, case_fn_zero = _run_case(repeated_case, config.seed, meta)
            case_rows.extend(rows)
            final_fn_zero = final_fn_zero and case_fn_zero
        all_rows.extend(case_rows)
        case_results.append(
            BenchmarkSuiteCaseResult(
                case_name=case.name,
                method=case.method,
                row_count=len(case_rows),
                final_fn_zero=final_fn_zero,
            )
        )
    if not all_rows:
        raise ValueError("benchmark suite produced no rows")
    paths = None
    if export:
        root = Path(output_root or config.output_root)
        paths = export_benchmark_run(root / config.suite_name / meta.run_id, meta, tuple(all_rows))
    return BenchmarkSuiteRunResult(
        suite=config,
        meta_run_id=meta.run_id,
        rows=tuple(all_rows),
        case_results=tuple(case_results),
        export_paths=paths,
    )


def run_benchmark_suite_from_config_path(
    path: str | Path,
    *,
    output_root: str | Path | None = None,
    environment: Mapping[str, object] | None = None,
    run_id: str | None = None,
    export: bool = True,
) -> BenchmarkSuiteRunResult:
    return run_benchmark_suite(
        load_benchmark_suite_config(path),
        output_root=output_root,
        environment=environment,
        run_id=run_id,
        export=export,
    )
