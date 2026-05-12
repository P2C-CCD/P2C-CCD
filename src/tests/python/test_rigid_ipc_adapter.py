from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from p2cccd.datasets.ccd import (  # noqa: E402
    CCDQueryFamily,
    RigidIPCSceneAdapter,
    validate_query_batch,
)
from p2cccd.viz import (  # noqa: E402
    render_rigid_ipc_scene_svg,
    summarize_rigid_ipc_visualization,
    write_rigid_ipc_scene_debug_html,
)


BASELINE_ROOT = PROJECT_ROOT / "baseline"


def test_rigid_ipc_adapter_discovers_fixtures_meshes_and_license() -> None:
    adapter = RigidIPCSceneAdapter(BASELINE_ROOT / "rigid-ipc")

    adapter.require_available()
    assert adapter.license().available()
    entry_points = adapter.fixture_entry_points()
    assert Path(entry_points["fixtures"]).exists()
    assert Path(entry_points["meshes"]).exists()

    infos = adapter.list_fixture_infos(dimension="3D", limit=3)
    assert infos
    assert all(info.dimension == "3D" for info in infos)
    assert all(info.body_count > 0 for info in infos)

    scenes = adapter.list_scenes(dimension="3D", limit=2)
    assert len(scenes) == 2
    assert scenes[0].source_name == "Rigid IPC scenes"
    assert scenes[0].metadata["body_count"] > 0


def test_rigid_ipc_adapter_loads_complex_rigid_body_scene() -> None:
    adapter = RigidIPCSceneAdapter(BASELINE_ROOT / "rigid-ipc")
    scene = adapter.load_scene("3D/chain/3-links")

    assert scene.scene_name == "3D/chain/3-links"
    assert scene.body_count == 3
    assert scene.timestep > 0.0
    assert scene.bodies[0].mesh == "torus.obj"
    assert scene.bodies[0].mesh_path is not None
    assert scene.bodies[0].mesh_path.exists()
    assert scene.bodies[0].radius > 0.0
    assert scene.moving_body_count >= 1


def test_rigid_ipc_adapter_generates_valid_body_pair_query_batch() -> None:
    adapter = RigidIPCSceneAdapter(BASELINE_ROOT / "rigid-ipc")
    batch = adapter.load_body_pair_query_batch(
        "3D/chain/3-links",
        family=CCDQueryFamily.VERTEX_FACE,
        limit=2,
    )

    assert validate_query_batch(batch) is batch
    assert batch.source_name == "Rigid IPC scenes"
    assert batch.scene_name == "3D/chain/3-links"
    assert batch.family == CCDQueryFamily.VERTEX_FACE
    assert batch.query_count == 2
    assert batch.known_label_count == 0
    assert batch.metadata["ground_truth_labels"] == "unknown"
    assert all(query.box_pair is not None for query in batch.queries)
    assert all(query.metadata["adapter_query_type"] == "rigid_ipc_body_pair_proxy" for query in batch.queries)


def test_rigid_ipc_visualization_renders_scene_and_body_pair_queries(tmp_path: Path) -> None:
    adapter = RigidIPCSceneAdapter(BASELINE_ROOT / "rigid-ipc")
    scene = adapter.load_scene("3D/chain/3-links")
    batch = adapter.load_body_pair_query_batch(
        "3D/chain/3-links",
        family=CCDQueryFamily.VERTEX_FACE,
        limit=3,
    )

    summary = summarize_rigid_ipc_visualization(scene, batch)
    svg = render_rigid_ipc_scene_svg(scene, batch)
    html_path = write_rigid_ipc_scene_debug_html(
        tmp_path / "rigid_ipc_scene.html",
        scene=scene,
        batch=batch,
    )

    assert summary.scene_name == "3D/chain/3-links"
    assert summary.body_count == 3
    assert summary.query_count == 3
    assert "<svg" in svg
    assert "generated body-pair query" in svg
    assert html_path.exists()
    html = html_path.read_text(encoding="utf-8")
    assert "Rigid-IPC Scene Debug View" in html
    assert "3D/chain/3-links" in html
