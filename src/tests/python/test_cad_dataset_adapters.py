from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from p2cccd.datasets.cad import (  # noqa: E402
    ABCDatasetAdapter,
    Fusion360GalleryAdapter,
    mesh_stats_from_file,
)


OBJ_A = """\
v 0 0 0
v 1 0 0
v 0 1 0
v 0 0 1
f 1 2 3
f 1 3 4
"""

OBJ_B = """\
v 0 0 0
v 1.1 0 0
v 0 1.1 0
v 0 0 1.1
f 1 2 3
f 1 3 4
"""


def _write_obj(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_binary_stl(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = bytes(80)
    face_count = 1
    normal = (0.0, 0.0, 1.0)
    v0 = (0.0, 0.0, 0.0)
    v1 = (1.0, 0.0, 0.0)
    v2 = (0.0, 1.0, 0.0)
    payload = struct.pack("<12fH", *(normal + v0 + v1 + v2 + (0,)))
    with path.open("wb") as handle:
        handle.write(header)
        handle.write(struct.pack("<I", face_count))
        handle.write(payload)


def test_mesh_stats_from_obj_counts_vertices_faces_and_bounds(tmp_path: Path) -> None:
    mesh_path = tmp_path / "part.obj"
    _write_obj(mesh_path, OBJ_A)

    stats = mesh_stats_from_file(mesh_path)

    assert stats.vertex_count == 4
    assert stats.face_count == 2
    assert stats.bounds_min == (0.0, 0.0, 0.0)
    assert stats.bounds_max == (1.0, 1.0, 1.0)
    assert stats.diagonal > 1.7


def test_mesh_stats_from_binary_stl_counts_vertices_faces_and_bounds(tmp_path: Path) -> None:
    mesh_path = tmp_path / "part.stl"
    _write_binary_stl(mesh_path)

    stats = mesh_stats_from_file(mesh_path)

    assert stats.vertex_count == 3
    assert stats.face_count == 1
    assert stats.bounds_min == (0.0, 0.0, 0.0)
    assert stats.bounds_max == (1.0, 1.0, 0.0)
    assert stats.diagonal > 1.4


def test_abc_adapter_ingests_assets_patch_metadata_and_hard_negative_pairs(tmp_path: Path) -> None:
    root = tmp_path / "abc"
    root.mkdir()
    (root / "LICENSE").write_text("test license", encoding="utf-8")
    _write_obj(root / "gear.obj", OBJ_A)
    _write_obj(root / "bracket.obj", OBJ_B)
    (root / "gear.patch.json").write_text(
        json.dumps({"patches": [{"patch_id": 1}, {"patch_id": 2}]}),
        encoding="utf-8",
    )

    adapter = ABCDatasetAdapter(root)
    assets = adapter.list_assets()
    pairs = adapter.industrial_hard_negative_pairs(limit=1)

    assert adapter.license().available()
    assert len(assets) == 2
    assert {asset.mesh_format for asset in assets} == {"obj"}
    assert any(asset.has_patch_metadata for asset in assets)
    assert len(pairs) == 1
    assert pairs[0].source_name == "ABC Dataset"
    assert pairs[0].hardness_score > 0.5
    assert pairs[0].patch_pair_count >= 2
    assert pairs[0].metadata["sampling_role"] == "industrial_hard_negative"


def test_fusion360_adapter_ingests_sequences_and_assembly_motion_samples(tmp_path: Path) -> None:
    root = tmp_path / "fusion360"
    assembly = root / "assembly_001"
    _write_obj(assembly / "base.obj", OBJ_A)
    _write_obj(assembly / "slider.obj", OBJ_B)
    (root / "LICENSE").write_text("test license", encoding="utf-8")
    (assembly / "assembly.json").write_text(
        json.dumps({"assembly_id": "assembly_001", "joint_count": 1}),
        encoding="utf-8",
    )

    adapter = Fusion360GalleryAdapter(root)
    sequences = adapter.list_sequences()
    samples = adapter.generate_assembly_motion_samples(sequence_name="assembly_001", limit=1)

    assert adapter.license().available()
    assert len(sequences) == 1
    assert sequences[0].sequence_name == "assembly_001"
    assert sequences[0].metadata["asset_count"] == 2
    assert sequences[0].metadata["joint_count"] == 1
    assert len(samples) == 1
    assert samples[0].source_name == "Fusion 360 Gallery"
    assert samples[0].motion_type == "linear_assembly_approach"
    assert samples[0].center_a_t0 != samples[0].center_a_t1
    assert samples[0].center_b_t0 == samples[0].center_b_t1
