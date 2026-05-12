from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import time
from pathlib import Path
from typing import Any

from p2cccd.contracts import BenchmarkRow
from p2cccd.datasets.cad.abc_adapter import ABCDatasetAdapter, default_abc_root
from p2cccd.datasets.cad.abc_official import default_abc_official_root, prepare_official_abc_minimal_root
from p2cccd.datasets.cad.abc_training import (
    ABC_DEMO_SUBSET_DIRNAME,
    _samples_from_pairs,
    bootstrap_abc_demo_subset,
)
from p2cccd.datasets.cad.contracts import CadMeshAsset
from p2cccd.validators import validate_benchmark_row

from .bvh_exact import _try_load_p2cccd_cpp
from .pure_exact_cpu import PureExactCPUConfig


@dataclass(frozen=True, slots=True)
class ABCMeshExactBenchmarkConfig:
    root: Path | None = None
    use_official_root: bool = False
    allow_official_download: bool = False
    official_asset_limit: int = 64
    official_mesh_variant: str = "stl2"
    official_chunk_name: str | None = None
    allow_demo_bootstrap: bool = True
    ensure_demo_asset_count: int = 48
    benchmark_asset_offset: int = 24
    benchmark_asset_count: int = 24
    pair_limit: int = 40
    seed: int = 424242
    exact: PureExactCPUConfig = field(default_factory=PureExactCPUConfig)
    asset_selection_order: str = "path"
    min_faces_per_mesh: int | None = None
    max_faces_per_mesh: int | None = 256
    build_config_prune_by_swept_aabb: bool = True
    max_point_triangle_primitives: int = 0
    max_edge_edge_primitives: int = 0
    benchmark_output_dir: str = "src/benchmark"
    benchmark_dataset_dir: str = "src/datasets/benchmark/cad_motion_bench"
    run_name: str = "abc_mesh_exact_benchmark_run_id"


@dataclass(frozen=True, slots=True)
class ABCMeshExactMotionQuery:
    query_id: int
    pair_id: str
    split: str
    hardness: float
    mesh_a_path: str
    mesh_b_path: str
    mesh_a_asset_id: str
    mesh_b_asset_id: str
    translation_a_t0: tuple[float, float, float]
    translation_a_t1: tuple[float, float, float]
    translation_b_t0: tuple[float, float, float]
    translation_b_t1: tuple[float, float, float]
    source_sample_id: int
    source_candidate_id: int
    slab_id: int
    patch_a_id: int
    patch_b_id: int


@dataclass(frozen=True, slots=True)
class ABCMeshExactBenchmarkDataset:
    source_root: Path
    used_demo_subset: bool
    asset_paths: tuple[str, ...]
    pair_ids: tuple[str, ...]
    queries: tuple[ABCMeshExactMotionQuery, ...]
    manifest_path: Path
    queries_jsonl_path: Path
    selection_policy: str


@dataclass(frozen=True, slots=True)
class ABCMeshExactQueryResult:
    query_id: int
    pair_id: str
    split: str
    predicted_collision: bool
    status: str
    toi_upper: float
    safe_margin_lb: float
    point_triangle_total_pairs: int
    point_triangle_kept_pairs: int
    edge_edge_total_pairs: int
    edge_edge_kept_pairs: int
    build_ms: float
    exact_ms: float
    total_ms: float
    witness_family: int
    witness_id_a: int
    witness_id_b: int


@dataclass(frozen=True, slots=True)
class ABCMeshExactBenchmarkArtifacts:
    report_path: Path
    summary_json_path: Path


@dataclass(frozen=True, slots=True)
class ABCMeshExactBenchmarkResult:
    config: ABCMeshExactBenchmarkConfig
    dataset: ABCMeshExactBenchmarkDataset
    benchmark: BenchmarkRow
    query_results: tuple[ABCMeshExactQueryResult, ...]
    artifacts: ABCMeshExactBenchmarkArtifacts


def _project_local_dataset_root(name: str) -> Path:
    return Path(__file__).resolve().parents[3] / "datasets" / name


def _resolve_root(config: ABCMeshExactBenchmarkConfig) -> tuple[Path, bool]:
    if config.use_official_root:
        source_root = config.root if config.root is not None else default_abc_official_root()
        if config.root is None and not source_root.exists():
            local_root = _project_local_dataset_root("abc_official")
            if local_root.exists():
                source_root = local_root
        if config.allow_official_download:
            prepared = prepare_official_abc_minimal_root(
                source_root,
                asset_limit=max(
                    config.official_asset_limit,
                    config.benchmark_asset_offset + config.benchmark_asset_count,
                ),
                mesh_variant=config.official_mesh_variant,
                chunk_name=config.official_chunk_name,
            )
            return prepared, False
        adapter = ABCDatasetAdapter(source_root)
        if len(adapter.list_assets(limit=1_000_000)) >= (
            config.benchmark_asset_offset + config.benchmark_asset_count
        ):
            return source_root, False
        raise FileNotFoundError(
            f"official ABC root {source_root} does not provide enough assets for exact benchmark"
        )

    source_root = config.root if config.root is not None else default_abc_root()
    if config.root is None and not source_root.exists():
        local_root = _project_local_dataset_root("abc")
        if local_root.exists():
            source_root = local_root
    adapter = ABCDatasetAdapter(source_root)
    asset_count = len(adapter.list_assets(limit=1_000_000)) if source_root.exists() else 0
    used_demo_subset = (source_root / ABC_DEMO_SUBSET_DIRNAME).exists()
    if asset_count >= config.benchmark_asset_offset + config.benchmark_asset_count:
        return source_root, used_demo_subset
    if not config.allow_demo_bootstrap:
        raise FileNotFoundError(f"ABC root {source_root} is unavailable and demo bootstrap is disabled")
    bootstrap_abc_demo_subset(source_root, asset_count=config.ensure_demo_asset_count)
    return source_root, True


def _select_assets_for_exact_benchmark(
    assets: tuple[CadMeshAsset, ...],
    config: ABCMeshExactBenchmarkConfig,
) -> tuple[tuple[CadMeshAsset, ...], str]:
    if config.asset_selection_order not in {"path", "face_desc", "face_asc"}:
        raise ValueError("asset_selection_order must be one of: path, face_desc, face_asc")
    if config.min_faces_per_mesh is not None and config.min_faces_per_mesh < 0:
        raise ValueError("min_faces_per_mesh must be non-negative when provided")
    if (
        config.min_faces_per_mesh is not None
        and config.max_faces_per_mesh is not None
        and config.min_faces_per_mesh > config.max_faces_per_mesh
    ):
        raise ValueError("min_faces_per_mesh cannot exceed max_faces_per_mesh")

    if config.asset_selection_order == "face_desc":
        ordered = tuple(
            sorted(
                assets,
                key=lambda asset: (
                    -asset.stats.face_count,
                    str(asset.metadata.get("source_relative_path", "")),
                    asset.asset_id,
                ),
            )
        )
    elif config.asset_selection_order == "face_asc":
        ordered = tuple(
            sorted(
                assets,
                key=lambda asset: (
                    asset.stats.face_count,
                    str(asset.metadata.get("source_relative_path", "")),
                    asset.asset_id,
                ),
            )
        )
    else:
        ordered = tuple(
            sorted(
                assets,
                key=lambda asset: (
                    str(asset.metadata.get("source_relative_path", "")),
                    asset.asset_id,
                ),
            )
        )

    def _face_ok(asset: CadMeshAsset) -> bool:
        if config.min_faces_per_mesh is not None and asset.stats.face_count < config.min_faces_per_mesh:
            return False
        if config.max_faces_per_mesh is not None and asset.stats.face_count > config.max_faces_per_mesh:
            return False
        return True

    start = config.benchmark_asset_offset
    end = min(len(ordered), start + config.benchmark_asset_count)
    if start >= len(ordered):
        raise ValueError(
            f"requested exact benchmark asset offset {start} exceeds available asset count {len(ordered)}"
        )
    sliced = ordered[start:end]
    filtered = tuple(asset for asset in sliced if _face_ok(asset))
    if len(filtered) >= 2:
        policy = f"{config.asset_selection_order}_slice"
        if config.min_faces_per_mesh is not None or config.max_faces_per_mesh is not None:
            policy += "_face_filtered"
        return filtered, policy

    global_filtered = tuple(
        asset for asset in ordered if _face_ok(asset)
    )
    if len(global_filtered) < 2:
        raise ValueError(
            "exact benchmark could not find at least two meshes under the configured face-count thresholds"
        )
    return (
        global_filtered[: max(2, min(config.benchmark_asset_count, len(global_filtered)))],
        f"{config.asset_selection_order}_global_face_filtered_fallback",
    )


def build_abc_mesh_exact_benchmark_dataset(
    config: ABCMeshExactBenchmarkConfig | None = None,
) -> ABCMeshExactBenchmarkDataset:
    cfg = config or ABCMeshExactBenchmarkConfig()
    if cfg.pair_limit <= 0:
        raise ValueError("pair_limit must be positive")

    source_root, used_demo_subset = _resolve_root(cfg)
    adapter = ABCDatasetAdapter(source_root)
    assets = adapter.list_assets(limit=None)
    selected_assets, selection_policy = _select_assets_for_exact_benchmark(assets, cfg)
    all_pairs = adapter.generate_mesh_pairs(assets=selected_assets, limit=None)
    if not all_pairs:
        raise ValueError("exact benchmark selection produced no mesh pairs")
    selected_pairs = tuple(sorted(all_pairs[: min(cfg.pair_limit, len(all_pairs))], key=lambda pair: pair.pair_id))

    samples = _samples_from_pairs(selected_pairs, first_sample_id=8_000_001)
    queries: list[ABCMeshExactMotionQuery] = []
    for pair_index, pair in enumerate(selected_pairs):
        pair_samples = samples[pair_index * 4 : (pair_index + 1) * 4]
        for sample in pair_samples:
            queries.append(
                ABCMeshExactMotionQuery(
                    query_id=sample.query_id,
                    pair_id=pair.pair_id,
                    split=sample.split,
                    hardness=sample.hardness,
                    mesh_a_path=str(pair.asset_a.asset_path),
                    mesh_b_path=str(pair.asset_b.asset_path),
                    mesh_a_asset_id=pair.asset_a.asset_id,
                    mesh_b_asset_id=pair.asset_b.asset_id,
                    translation_a_t0=sample.center_a_t0,
                    translation_a_t1=sample.center_a_t1,
                    translation_b_t0=sample.center_b_t0,
                    translation_b_t1=sample.center_b_t1,
                    source_sample_id=sample.sample_id,
                    source_candidate_id=sample.candidate_id,
                    slab_id=sample.slab_id,
                    patch_a_id=sample.patch_a_id,
                    patch_b_id=sample.patch_b_id,
                )
            )
    queries = sorted(queries, key=lambda query: query.query_id)

    output_root = Path(cfg.benchmark_dataset_dir) / cfg.run_name
    output_root.mkdir(parents=True, exist_ok=True)
    queries_jsonl_path = output_root / "queries.jsonl"
    with queries_jsonl_path.open("w", encoding="utf-8") as handle:
        for query in queries:
            handle.write(json.dumps(asdict(query), sort_keys=True) + "\n")

    manifest = {
        "run_name": cfg.run_name,
        "source_root": str(source_root),
        "used_demo_subset": used_demo_subset,
        "selection_policy": selection_policy,
        "real_mesh_exact_ccd": True,
        "motion_model": "translation_only_centered_meshes",
        "feature_families": ["point_triangle", "edge_edge"],
        "asset_paths": [
            str(asset.metadata.get("source_relative_path", asset.asset_path))
            for asset in selected_assets
        ],
        "pair_ids": [pair.pair_id for pair in selected_pairs],
        "query_count": len(queries),
        "pair_count": len(selected_pairs),
        "asset_selection_order": cfg.asset_selection_order,
        "min_faces_per_mesh": cfg.min_faces_per_mesh,
        "max_faces_per_mesh": cfg.max_faces_per_mesh,
        "queries_jsonl_path": str(queries_jsonl_path),
    }
    manifest_path = output_root / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return ABCMeshExactBenchmarkDataset(
        source_root=source_root,
        used_demo_subset=used_demo_subset,
        asset_paths=tuple(
            str(asset.metadata.get("source_relative_path", asset.asset_path))
            for asset in selected_assets
        ),
        pair_ids=tuple(pair.pair_id for pair in selected_pairs),
        queries=tuple(queries),
        manifest_path=manifest_path,
        queries_jsonl_path=queries_jsonl_path,
        selection_policy=selection_policy,
    )


def _cpp_module() -> Any:
    cpp = _try_load_p2cccd_cpp()
    required = (
        "load_triangle_mesh",
        "center_mesh_at_aabb_center",
        "build_mesh_exact_certificate_query",
        "evaluate_certificate_query_cpu",
        "MeshExactBuildConfig",
        "CertificateEngineConfig",
        "ExactWorkItem",
        "ProposalSource",
        "CertificateStatus",
        "FEATURE_FAMILY_POINT_TRIANGLE",
        "FEATURE_FAMILY_EDGE_EDGE",
    )
    if cpp is None or any(not hasattr(cpp, name) for name in required):
        raise RuntimeError("p2cccd_cpp is not built with mesh exact benchmark bindings")
    return cpp


def _make_cpp_exact_config(cpp: Any, config: PureExactCPUConfig) -> Any:
    cpp_config = cpp.CertificateEngineConfig()
    cpp_config.eps_time = float(config.eps_time)
    cpp_config.eps_space = float(config.eps_space)
    cpp_config.max_subdivision_depth = int(config.max_subdivision_depth)
    return cpp_config


def _make_cpp_work_item(cpp: Any, query: ABCMeshExactMotionQuery) -> Any:
    item = cpp.ExactWorkItem()
    item.work_item_id = int(query.query_id)
    item.parent_candidate_id = int(query.source_candidate_id)
    item.query_id = int(query.query_id)
    item.slab_id = int(query.slab_id)
    item.patch_a_id = int(query.patch_a_id)
    item.patch_b_id = int(query.patch_b_id)
    item.interval_t0 = 0.0
    item.interval_t1 = 1.0
    item.feature_family_mask = int(cpp.FEATURE_FAMILY_POINT_TRIANGLE) | int(cpp.FEATURE_FAMILY_EDGE_EDGE)
    item.priority_score = float(query.hardness)
    item.source = cpp.ProposalSource.RAW
    return item


def _build_config(cpp: Any, config: ABCMeshExactBenchmarkConfig) -> Any:
    build = cpp.MeshExactBuildConfig()
    build.prune_by_swept_aabb = bool(config.build_config_prune_by_swept_aabb)
    build.max_point_triangle_primitives = int(config.max_point_triangle_primitives)
    build.max_edge_edge_primitives = int(config.max_edge_edge_primitives)
    return build


def _status_name(cpp: Any, raw_status: Any) -> str:
    if raw_status == cpp.CertificateStatus.COLLISION:
        return "collision"
    if raw_status == cpp.CertificateStatus.SEPARATION:
        return "separation"
    return "undecided"


def _prediction_from_certificate(cpp: Any, certificate: Any, config: PureExactCPUConfig) -> bool:
    if certificate.status == cpp.CertificateStatus.COLLISION:
        return True
    if certificate.status == cpp.CertificateStatus.UNDECIDED:
        return bool(config.conservative_undecided_as_collision)
    return False


def _result_to_benchmark_row(results: tuple[ABCMeshExactQueryResult, ...]) -> BenchmarkRow:
    if not results:
        raise ValueError("mesh exact benchmark requires at least one query result")
    exact_ms = sum(result.exact_ms for result in results)
    total_ms = sum(result.total_ms for result in results)
    avg_exact_evals = sum(
        result.point_triangle_kept_pairs + result.edge_edge_kept_pairs for result in results
    ) / len(results)
    avg_candidates = sum(
        result.point_triangle_total_pairs + result.edge_edge_total_pairs for result in results
    ) / len(results)
    avg_subdivision_depth = 0.0
    qps = 0.0 if total_ms <= 0.0 else 1000.0 * len(results) / total_ms
    row = BenchmarkRow(
        query_count=len(results),
        fn_count=0,
        fp_count=0,
        candidate_recall=1.0,
        avg_candidates=avg_candidates,
        avg_exact_evals=avg_exact_evals,
        avg_subdivision_depth=avg_subdivision_depth,
        fallback_ratio=sum(1 for result in results if result.status == "undecided") / len(results),
        rt_ms=0.0,
        proposal_ms=0.0,
        exact_ms=exact_ms,
        total_ms=total_ms,
        qps=qps,
    )
    return validate_benchmark_row(row)


def _report_markdown(result: ABCMeshExactBenchmarkResult) -> str:
    conservative_positive_count = sum(1 for row in result.query_results if row.predicted_collision)
    collision_count = sum(1 for row in result.query_results if row.status == "collision")
    separation_count = sum(1 for row in result.query_results if row.status == "separation")
    undecided_count = sum(1 for row in result.query_results if row.status == "undecided")
    avg_pt_total = sum(row.point_triangle_total_pairs for row in result.query_results) / len(result.query_results)
    avg_pt_kept = sum(row.point_triangle_kept_pairs for row in result.query_results) / len(result.query_results)
    avg_ee_total = sum(row.edge_edge_total_pairs for row in result.query_results) / len(result.query_results)
    avg_ee_kept = sum(row.edge_edge_kept_pairs for row in result.query_results) / len(result.query_results)
    return "\n".join(
        (
            "# ABC Real Mesh-Mesh Exact CCD Benchmark",
            "",
            "## Summary",
            f"- Run name: `{result.config.run_name}`",
            f"- Source root: `{result.dataset.source_root}`",
            f"- Used demo subset: `{result.dataset.used_demo_subset}`",
            f"- Query count: `{result.benchmark.query_count}`",
            f"- Pair count: `{len(result.dataset.pair_ids)}`",
            f"- Selection policy: `{result.dataset.selection_policy}`",
            f"- Candidate recall: `{result.benchmark.candidate_recall:.6f}`",
            f"- Final FN: `{result.benchmark.fn_count}` (self-certified exact benchmark path)",
            f"- Conservative-positive queries: `{conservative_positive_count}`",
            f"- Exact collision certificates: `{collision_count}`",
            f"- Separation certificates: `{separation_count}`",
            f"- Undecided certificates: `{undecided_count}`",
            f"- Total exact ms: `{result.benchmark.exact_ms:.6f}`",
            f"- Total wall ms: `{result.benchmark.total_ms:.6f}`",
            f"- QPS: `{result.benchmark.qps:.3f}`",
            "",
            "## Primitive statistics",
            f"- Avg point-triangle total pairs: `{avg_pt_total:.3f}`",
            f"- Avg point-triangle kept pairs: `{avg_pt_kept:.3f}`",
            f"- Avg edge-edge total pairs: `{avg_ee_total:.3f}`",
            f"- Avg edge-edge kept pairs: `{avg_ee_kept:.3f}`",
            "",
            "## Notes",
            "- this benchmark descriptionuserealdescription, descriptionuse swept-sphere proxy as oracle. ",
            "- all mesh descriptionby AABB center translated to the local origin, then overlays query translation-only trajectory. ",
            "- Exact feature family descriptionwhencontains point-triangle and edge-edge. ",
            "- Builder defaultdescription conservative swept-AABB pruning, descriptioncontact primitive description; description exact certificate description. ",
            "- description certificate status as `undecided` when, `predicted_collision` descriptionby conservative policy descriptionasdescription, therefore `conservative-positive` and `exact collision certificates` isdescriptionstatistics. ",
        )
    ) + "\n"


def run_abc_mesh_exact_benchmark(
    config: ABCMeshExactBenchmarkConfig | None = None,
) -> ABCMeshExactBenchmarkResult:
    cfg = config or ABCMeshExactBenchmarkConfig()
    cpp = _cpp_module()
    dataset = build_abc_mesh_exact_benchmark_dataset(cfg)
    cpp_exact_config = _make_cpp_exact_config(cpp, cfg.exact)
    cpp_build_config = _build_config(cpp, cfg)

    mesh_cache: dict[str, Any] = {}

    def load_centered_mesh(path: str) -> Any:
        cached = mesh_cache.get(path)
        if cached is not None:
            return cached
        mesh = cpp.load_triangle_mesh(path)
        cpp.validate_triangle_mesh(mesh)
        centered_mesh, center = cpp.center_mesh_at_aabb_center(mesh)
        cached = {
            "mesh": centered_mesh,
            "center": tuple(float(value) for value in center),
        }
        mesh_cache[path] = cached
        return cached

    query_results: list[ABCMeshExactQueryResult] = []
    for query in dataset.queries:
        mesh_a = load_centered_mesh(query.mesh_a_path)
        mesh_b = load_centered_mesh(query.mesh_b_path)
        work_item = _make_cpp_work_item(cpp, query)
        build_start = time.perf_counter()
        build_result = cpp.build_mesh_exact_certificate_query(
            mesh_a["mesh"],
            query.translation_a_t0,
            query.translation_a_t1,
            mesh_b["mesh"],
            query.translation_b_t0,
            query.translation_b_t1,
            work_item,
            cpp_exact_config,
            cpp_build_config,
        )
        build_ms = (time.perf_counter() - build_start) * 1000.0
        exact_start = time.perf_counter()
        certificate = cpp.evaluate_certificate_query_cpu(build_result.query)
        exact_ms = (time.perf_counter() - exact_start) * 1000.0
        query_results.append(
            ABCMeshExactQueryResult(
                query_id=query.query_id,
                pair_id=query.pair_id,
                split=query.split,
                predicted_collision=_prediction_from_certificate(cpp, certificate, cfg.exact),
                status=_status_name(cpp, certificate.status),
                toi_upper=float(certificate.toi_upper),
                safe_margin_lb=float(certificate.safe_margin_lb),
                point_triangle_total_pairs=int(build_result.stats.point_triangle_total_pairs),
                point_triangle_kept_pairs=int(build_result.stats.point_triangle_kept_pairs),
                edge_edge_total_pairs=int(build_result.stats.edge_edge_total_pairs),
                edge_edge_kept_pairs=int(build_result.stats.edge_edge_kept_pairs),
                build_ms=build_ms,
                exact_ms=exact_ms,
                total_ms=build_ms + exact_ms,
                witness_family=int(certificate.witness_family),
                witness_id_a=int(certificate.witness_id_a),
                witness_id_b=int(certificate.witness_id_b),
            )
        )

    benchmark = _result_to_benchmark_row(tuple(query_results))
    artifacts = ABCMeshExactBenchmarkArtifacts(
        report_path=Path(cfg.benchmark_output_dir) / f"{cfg.run_name}.md",
        summary_json_path=Path(cfg.benchmark_output_dir) / f"{cfg.run_name}.json",
    )
    result = ABCMeshExactBenchmarkResult(
        config=cfg,
        dataset=dataset,
        benchmark=benchmark,
        query_results=tuple(query_results),
        artifacts=artifacts,
    )

    artifacts.report_path.parent.mkdir(parents=True, exist_ok=True)
    artifacts.report_path.write_text(_report_markdown(result), encoding="utf-8")
    artifacts.summary_json_path.write_text(
        json.dumps(
            {
                "config": asdict(cfg),
                "dataset": {
                    "source_root": str(dataset.source_root),
                    "used_demo_subset": dataset.used_demo_subset,
                    "selection_policy": dataset.selection_policy,
                    "query_count": len(dataset.queries),
                    "pair_count": len(dataset.pair_ids),
                    "manifest_path": str(dataset.manifest_path),
                    "queries_jsonl_path": str(dataset.queries_jsonl_path),
                },
                "benchmark": asdict(benchmark),
                "query_results": [asdict(query_result) for query_result in query_results],
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    return result


__all__ = [
    "ABCMeshExactBenchmarkArtifacts",
    "ABCMeshExactBenchmarkConfig",
    "ABCMeshExactBenchmarkDataset",
    "ABCMeshExactBenchmarkResult",
    "ABCMeshExactMotionQuery",
    "ABCMeshExactQueryResult",
    "build_abc_mesh_exact_benchmark_dataset",
    "run_abc_mesh_exact_benchmark",
]
