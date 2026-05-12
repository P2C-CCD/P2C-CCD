from __future__ import annotations

import time

from p2cccd.datasets.ccd import CCDQueryFamily, DatasetQueryBatch, RigidIPCScene, RigidIPCSceneAdapter

from .bvh_exact import (
    Aabb,
    BroadPhaseBackend,
    BroadPhasePrimitive,
    CpuAabbBroadPhaseBackend,
    _aabb_from_points,
)
from .no_proposal import (
    NoProposalConfig,
    NoProposalResult,
    _make_benchmark_row as _make_no_proposal_benchmark_row,
    _rt_config_from_no_proposal,
    _validate_config as _validate_no_proposal_config,
    schedule_exact_work_items_no_proposal,
)
from .rt_exact import (
    RTExactConfig,
    RTExactResult,
    RtCandidateStats,
    _candidate_recall,
    _default_rt_broad_phase,
    _external_candidate_from_pair,
    _family_mask_for_external,
    _make_benchmark_row as _make_rt_benchmark_row,
    _make_timing,
    _process_external_exact_work_queue,
    _runtime_query_ids,
    _validate_config as _validate_rt_exact_config,
    schedule_exact_work_items_without_stpf,
    validate_rt_exact_coverage,
)
from .rt_stpf_exact import (
    RTSTPFExactConfig,
    RTSTPFExactResult,
    _make_benchmark_row as _make_rt_stpf_benchmark_row,
    _rt_config_from_stpf,
    _resolve_stpf_runtime,
    _run_dummy_stpf_fast_path,
    _run_stpf_predictions,
    _runtime_backend_name,
    _runtime_device_for_backend,
    _runtime_provider_name,
    _validate_config as _validate_rt_stpf_config,
    proposal_feature_rows_from_rt_candidates,
    schedule_exact_work_items_with_stpf,
)


def _scene_aabb(body_position_t0: tuple[float, float, float], body_position_t1: tuple[float, float, float], radius: float) -> Aabb:
    return _aabb_from_points((body_position_t0, body_position_t1), inflation=max(radius, 1.0e-6))


def _scene_body_primitives(
    scene: RigidIPCScene,
    family: CCDQueryFamily,
) -> tuple[tuple[BroadPhasePrimitive, ...], dict[int, int]]:
    family_name = family.p2cccd_witness_family
    primitives: list[BroadPhasePrimitive] = []
    body_id_by_primitive_id: dict[int, int] = {}
    for body in scene.bodies:
        center_t0 = body.position
        center_t1 = (
            body.position[0] + body.linear_velocity[0] * scene.timestep,
            body.position[1] + body.linear_velocity[1] * scene.timestep,
            body.position[2] + body.linear_velocity[2] * scene.timestep,
        )
        body_aabb = _scene_aabb(center_t0, center_t1, body.radius)
        primitive_a_id = body.body_id * 2
        primitive_b_id = primitive_a_id + 1
        primitives.append(
            BroadPhasePrimitive(
                primitive_id=primitive_a_id,
                query_id=1,
                role="a",
                aabb=body_aabb,
                family=family_name,
                metadata={"body_id": body.body_id},
            )
        )
        primitives.append(
            BroadPhasePrimitive(
                primitive_id=primitive_b_id,
                query_id=1,
                role="b",
                aabb=body_aabb,
                family=family_name,
                metadata={"body_id": body.body_id},
            )
        )
        body_id_by_primitive_id[primitive_a_id] = body.body_id
        body_id_by_primitive_id[primitive_b_id] = body.body_id
    return tuple(primitives), body_id_by_primitive_id


def _scene_query_batch(
    adapter: RigidIPCSceneAdapter,
    scene: RigidIPCScene,
    *,
    family: CCDQueryFamily,
    limit: int | None,
    include_fixed_fixed_pairs: bool,
) -> DatasetQueryBatch:
    return adapter.load_body_pair_query_batch(
        scene.scene_name,
        family=family,
        limit=limit,
        include_fixed_fixed_pairs=include_fixed_fixed_pairs,
    )


def _scene_candidate_bundle(
    scene: RigidIPCScene,
    *,
    family: CCDQueryFamily,
    config: RTExactConfig,
    adapter: RigidIPCSceneAdapter,
    backend: BroadPhaseBackend | None,
    limit: int | None,
    include_fixed_fixed_pairs: bool,
) -> tuple[DatasetQueryBatch, tuple, RtCandidateStats, dict[int, int]]:
    build_start = time.perf_counter()
    batch = _scene_query_batch(
        adapter,
        scene,
        family=family,
        limit=limit,
        include_fixed_fixed_pairs=include_fixed_fixed_pairs,
    )
    primitives, body_id_by_primitive_id = _scene_body_primitives(scene, family)
    build_ms = (time.perf_counter() - build_start) * 1000.0

    broad_phase = backend or _default_rt_broad_phase(config)
    raw_pairs, broad_phase_stats = broad_phase.find_pairs(
        primitives,
        same_query_only=True,
    )
    query_by_body_pair = {
        (min(query.box_pair[0], query.box_pair[1]), max(query.box_pair[0], query.box_pair[1])): query
        for query in batch.queries
        if query.box_pair is not None
    }
    runtime_ids = _runtime_query_ids([query.query_id for query in batch.queries])
    index_by_query_id = {query.query_id: index for index, query in enumerate(batch.queries)}

    compact_start = time.perf_counter()
    seen_body_pairs: set[tuple[int, int]] = set()
    active_query_ids: list[int] = []
    candidates = []
    for pair in raw_pairs:
        body_a_id = body_id_by_primitive_id.get(pair.primitive_a_id)
        body_b_id = body_id_by_primitive_id.get(pair.primitive_b_id)
        if body_a_id is None or body_b_id is None or body_a_id == body_b_id:
            continue
        body_pair = (min(body_a_id, body_b_id), max(body_a_id, body_b_id))
        if body_pair in seen_body_pairs:
            continue
        query = query_by_body_pair.get(body_pair)
        if query is None:
            continue
        seen_body_pairs.add(body_pair)
        active_query_ids.append(query.query_id)
        candidates.append(
            _external_candidate_from_pair(
                query,
                runtime_ids[query.query_id],
                pair,
                ordinal=len(candidates),
            )
        )
    compact_ms = (time.perf_counter() - compact_start) * 1000.0
    active_indices = {
        index_by_query_id[query_id]
        for query_id in active_query_ids
        if query_id in index_by_query_id
    }
    stats = RtCandidateStats(
        backend_name=broad_phase_stats.backend_name,
        primitive_count=len(primitives),
        raw_hit_count=len(raw_pairs),
        compact_candidate_count=len(candidates),
        candidate_recall=_candidate_recall(
            [query.ground_truth_collides for query in batch.queries],
            active_indices,
        ),
        timing=_make_timing(
            build_ms=build_ms,
            broad_phase_stats=broad_phase_stats,
            compact_ms=compact_ms,
        ),
    )
    return batch, tuple(candidates), stats, runtime_ids


def _default_scene_backend(config: RTExactConfig) -> BroadPhaseBackend:
    backend = _default_rt_broad_phase(config)
    if backend.name == "cpu_reference_rt":
        return CpuAabbBroadPhaseBackend(name="cpu_reference_rt_scene")
    return backend


def run_rt_exact_on_rigid_ipc_scene(
    scene: str | RigidIPCScene,
    *,
    family: CCDQueryFamily | str = CCDQueryFamily.VERTEX_FACE,
    config: RTExactConfig | None = None,
    adapter: RigidIPCSceneAdapter | None = None,
    backend: BroadPhaseBackend | None = None,
    limit: int | None = None,
    include_fixed_fixed_pairs: bool = False,
) -> RTExactResult:
    cfg = _validate_rt_exact_config(config or RTExactConfig())
    query_family = family if isinstance(family, CCDQueryFamily) else CCDQueryFamily(str(family))
    scene_adapter = adapter or RigidIPCSceneAdapter()
    loaded_scene = scene if isinstance(scene, RigidIPCScene) else scene_adapter.load_scene(scene)
    batch, candidates, candidate_stats, runtime_ids = _scene_candidate_bundle(
        loaded_scene,
        family=query_family,
        config=cfg,
        adapter=scene_adapter,
        backend=backend or _default_scene_backend(cfg),
        limit=limit,
        include_fixed_fixed_pairs=include_fixed_fixed_pairs,
    )
    family_by_runtime_query_id = {
        runtime_ids[query.query_id]: _family_mask_for_external(query)
        for query in batch.queries
    }
    work_items = schedule_exact_work_items_without_stpf(
        candidates,
        family_by_runtime_query_id=family_by_runtime_query_id,
        first_work_item_id=cfg.first_work_item_id,
    )
    query_results, certificates, audit_log, exact_elapsed_ms, exact_backend_name = _process_external_exact_work_queue(
        batch,
        candidates,
        work_items,
        runtime_ids,
        cfg,
    )
    validate_rt_exact_coverage(candidates, work_items, certificates)
    return RTExactResult(
        benchmark=_make_rt_benchmark_row(
            query_results,
            candidate_stats=candidate_stats,
            exact_elapsed_ms=exact_elapsed_ms,
        ),
        query_results=query_results,
        candidates=candidates,
        work_items=work_items,
        certificates=certificates,
        audit_log=audit_log,
        candidate_stats=candidate_stats,
        exact_backend_name=exact_backend_name,
        source_name=batch.source_name,
        scene_name=f"{batch.scene_name}:full_scene_rt",
        batch_id=batch.batch_id,
    )


def run_no_proposal_on_rigid_ipc_scene(
    scene: str | RigidIPCScene,
    *,
    family: CCDQueryFamily | str = CCDQueryFamily.VERTEX_FACE,
    config: NoProposalConfig | None = None,
    adapter: RigidIPCSceneAdapter | None = None,
    backend: BroadPhaseBackend | None = None,
    limit: int | None = None,
    include_fixed_fixed_pairs: bool = False,
) -> NoProposalResult:
    cfg = _validate_no_proposal_config(config or NoProposalConfig())
    rt_config = _rt_config_from_no_proposal(cfg)
    query_family = family if isinstance(family, CCDQueryFamily) else CCDQueryFamily(str(family))
    scene_adapter = adapter or RigidIPCSceneAdapter()
    loaded_scene = scene if isinstance(scene, RigidIPCScene) else scene_adapter.load_scene(scene)
    batch, candidates, candidate_stats, runtime_ids = _scene_candidate_bundle(
        loaded_scene,
        family=query_family,
        config=rt_config,
        adapter=scene_adapter,
        backend=backend or _default_scene_backend(rt_config),
        limit=limit,
        include_fixed_fixed_pairs=include_fixed_fixed_pairs,
    )
    family_by_runtime_query_id = {
        runtime_ids[query.query_id]: _family_mask_for_external(query)
        for query in batch.queries
    }
    work_items, no_proposal_stats = schedule_exact_work_items_no_proposal(
        candidates,
        family_by_runtime_query_id=family_by_runtime_query_id,
        config=cfg,
    )
    query_results, certificates, audit_log, exact_elapsed_ms, exact_backend_name = _process_external_exact_work_queue(
        batch,
        candidates,
        work_items,
        runtime_ids,
        rt_config,
    )
    validate_rt_exact_coverage(candidates, work_items, certificates)
    return NoProposalResult(
        benchmark=_make_no_proposal_benchmark_row(
            query_results,
            candidate_stats=candidate_stats,
            no_proposal_stats=no_proposal_stats,
            exact_elapsed_ms=exact_elapsed_ms,
        ),
        query_results=query_results,
        candidates=candidates,
        work_items=work_items,
        certificates=certificates,
        audit_log=audit_log,
        candidate_stats=candidate_stats,
        no_proposal_stats=no_proposal_stats,
        exact_backend_name=exact_backend_name,
        source_name=batch.source_name,
        scene_name=f"{batch.scene_name}:full_scene_rt",
        batch_id=batch.batch_id,
    )


def run_rt_stpf_exact_on_rigid_ipc_scene(
    scene: str | RigidIPCScene,
    *,
    family: CCDQueryFamily | str = CCDQueryFamily.VERTEX_FACE,
    config: RTSTPFExactConfig | None = None,
    adapter: RigidIPCSceneAdapter | None = None,
    backend: BroadPhaseBackend | None = None,
    model=None,
    device: str | None = None,
    limit: int | None = None,
    include_fixed_fixed_pairs: bool = False,
) -> RTSTPFExactResult:
    cfg = _validate_rt_stpf_config(config or RTSTPFExactConfig())
    rt_config = _rt_config_from_stpf(cfg)
    query_family = family if isinstance(family, CCDQueryFamily) else CCDQueryFamily(str(family))
    scene_adapter = adapter or RigidIPCSceneAdapter()
    loaded_scene = scene if isinstance(scene, RigidIPCScene) else scene_adapter.load_scene(scene)
    batch, candidates, candidate_stats, runtime_ids = _scene_candidate_bundle(
        loaded_scene,
        family=query_family,
        config=rt_config,
        adapter=scene_adapter,
        backend=backend or _default_scene_backend(rt_config),
        limit=limit,
        include_fixed_fixed_pairs=include_fixed_fixed_pairs,
    )
    family_by_runtime_query_id = {
        runtime_ids[query.query_id]: _family_mask_for_external(query)
        for query in batch.queries
    }

    resolved_runtime = None
    inference_backend_name = _runtime_backend_name(cfg)
    inference_provider_name = "dummy"
    inference_device = device
    proposal_start = time.perf_counter()
    if cfg.use_dummy_policy:
        feature_rows, proposal_predictions, work_items, schedule_stats = _run_dummy_stpf_fast_path(
            candidates,
            family_by_runtime_query_id=family_by_runtime_query_id,
            candidate_stats=candidate_stats,
            config=cfg,
            materialize_artifacts=False,
        )
    else:
        feature_rows = proposal_feature_rows_from_rt_candidates(
            candidates,
            family_by_runtime_query_id=family_by_runtime_query_id,
            candidate_stats=candidate_stats,
        )
        inference_device = _runtime_device_for_backend(
            cfg,
            row_count=len(feature_rows),
            requested_device=device,
        )
        if feature_rows:
            resolved_runtime = _resolve_stpf_runtime(
                cfg,
                model=model,
                device=inference_device,
            )
            inference_provider_name = _runtime_provider_name(resolved_runtime)
        proposal_predictions = _run_stpf_predictions(
            feature_rows,
            cfg,
            runtime=resolved_runtime,
            device=inference_device,
        )
        work_items, schedule_stats = schedule_exact_work_items_with_stpf(
            candidates,
            feature_rows,
            proposal_predictions,
            family_by_runtime_query_id=family_by_runtime_query_id,
            config=cfg,
        )
    proposal_elapsed_ms = (time.perf_counter() - proposal_start) * 1000.0

    query_results, certificates, audit_log, exact_elapsed_ms, exact_backend_name = _process_external_exact_work_queue(
        batch,
        candidates,
        work_items,
        runtime_ids,
        rt_config,
    )
    validate_rt_exact_coverage(candidates, work_items, certificates)
    if cfg.use_dummy_policy:
        feature_rows, proposal_predictions, _, _ = _run_dummy_stpf_fast_path(
            candidates,
            family_by_runtime_query_id=family_by_runtime_query_id,
            candidate_stats=candidate_stats,
            config=cfg,
            materialize_artifacts=True,
        )
    return RTSTPFExactResult(
        benchmark=_make_rt_stpf_benchmark_row(
            query_results,
            candidate_stats=candidate_stats,
            schedule_stats=schedule_stats,
            proposal_elapsed_ms=proposal_elapsed_ms,
            exact_elapsed_ms=exact_elapsed_ms,
        ),
        query_results=query_results,
        candidates=candidates,
        feature_rows=feature_rows,
        proposal_predictions=proposal_predictions,
        work_items=work_items,
        certificates=certificates,
        audit_log=audit_log,
        candidate_stats=candidate_stats,
        schedule_stats=schedule_stats,
        inference_backend_name=inference_backend_name,
        inference_provider_name=inference_provider_name,
        exact_backend_name=exact_backend_name,
        source_name=batch.source_name,
        scene_name=f"{batch.scene_name}:full_scene_rt",
        batch_id=batch.batch_id,
    )
