from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from p2cccd.datasets.ccd import (  # noqa: E402
    CCDQueryFamily,
    CCDWrapperAdapter,
    RootParityAdapter,
    ScalableCCDSampleAdapter,
    TightInclusionAdapter,
    build_first_layer_manifest,
    discover_first_layer_sources,
    validate_query_batch,
)


BASELINE_ROOT = PROJECT_ROOT / "baseline"


def test_first_layer_downloaded_repositories_are_discoverable() -> None:
    manifest = build_first_layer_manifest(BASELINE_ROOT)
    statuses = {status.name: status for status in discover_first_layer_sources(BASELINE_ROOT)}

    assert manifest["schema_version"] == 1
    assert manifest["required_available"] is True
    assert statuses["Tight Inclusion"].available
    assert statuses["CCD-Wrapper"].available
    assert statuses["Scalable CCD"].available
    assert statuses["Sample Scalable CCD Data"].available
    assert statuses["Exact Root Parity CCD"].available
    assert statuses["Rigid IPC scenes"].available
    assert statuses["Rigid IPC scenes"].path.name == "rigid-ipc"


def test_reference_adapters_report_entry_points_and_licenses() -> None:
    tight = TightInclusionAdapter(BASELINE_ROOT / "Tight-Inclusion")
    wrapper = CCDWrapperAdapter(BASELINE_ROOT / "CCD-Wrapper")
    root_parity = RootParityAdapter(BASELINE_ROOT / "Exact-Root-Parity-CCD")

    assert tight.license().available()
    assert "double_ccd_header" in tight.reference_entry_points()
    assert wrapper.license().available()
    assert "benchmark" in wrapper.benchmark_entry_points()
    assert "tight_inclusion" in wrapper.method_recipe_names()
    assert root_parity.license().available()
    assert "ccd_header" in root_parity.reference_entry_points()


def test_scalable_ccd_sample_adapter_indexes_scenes_and_batches() -> None:
    adapter = ScalableCCDSampleAdapter(BASELINE_ROOT / "Sample-Scalable-CCD-Data")

    scenes = adapter.list_scenes()
    scene_names = {scene.scene_name for scene in scenes}
    assert "cloth-funnel" in scene_names
    assert "rod-twist" in scene_names
    assert all(scene.schema_version == 1 for scene in scenes)

    batches = adapter.list_query_batches("cloth-funnel")
    by_id = {batch.batch_id: batch for batch in batches}
    assert "cloth-funnel:227vf" in by_id
    assert by_id["cloth-funnel:227vf"].query_count > 0
    assert by_id["cloth-funnel:227vf"].collision_count is not None


def test_scalable_ccd_sample_query_batch_loads_rational_queries_and_labels() -> None:
    adapter = ScalableCCDSampleAdapter(BASELINE_ROOT / "Sample-Scalable-CCD-Data")
    batch = adapter.load_query_batch(
        "cloth-funnel",
        family=CCDQueryFamily.VERTEX_FACE,
        step=227,
        limit=5,
    )

    assert validate_query_batch(batch) is batch
    assert batch.source_name == "Sample-Scalable-CCD-Data"
    assert batch.scene_name == "cloth-funnel"
    assert batch.family == CCDQueryFamily.VERTEX_FACE
    assert batch.query_count == 5
    assert batch.known_label_count == 5
    assert batch.collision_count == 2
    assert [query.ground_truth_collides for query in batch.queries] == [
        False,
        False,
        False,
        True,
        True,
    ]
    assert all(len(query.vertices_t0) == 4 and len(query.vertices_t1) == 4 for query in batch.queries)
    assert batch.queries[0].metadata["witness_family"] == "point_triangle"


def test_scalable_ccd_sample_edge_edge_query_batch_uses_edge_family() -> None:
    adapter = ScalableCCDSampleAdapter(BASELINE_ROOT / "Sample-Scalable-CCD-Data")
    batch = adapter.load_query_batch("cloth-funnel", family="ee", step=227, limit=3)

    assert batch.family == CCDQueryFamily.EDGE_EDGE
    assert batch.query_count == 3
    assert batch.queries[0].metadata["witness_family"] == "edge_edge"
