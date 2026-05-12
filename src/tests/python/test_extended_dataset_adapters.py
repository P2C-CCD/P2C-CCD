from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from p2cccd.datasets.cad import BetterSTEPAdapter  # noqa: E402
from p2cccd.datasets.objects import (  # noqa: E402
    GoogleScannedObjectsAdapter,
    ObjaverseXLAdapter,
    PartNetAdapter,
    PartNetMobilityAdapter,
    ShapeNetAdapter,
    Thingi10KAdapter,
    YCBObjectSetAdapter,
)
from p2cccd.datasets.robot import MoveItResourcesAdapter  # noqa: E402


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
v 1.2 0 0
v 0 1.2 0
v 0 0 1.2
f 1 2 3
f 1 3 4
"""

STEP_SAMPLE = """\
ISO-10303-21;
HEADER;
FILE_DESCRIPTION(('fixture'),'2;1');
FILE_NAME('demo.step','run_idT00:00:00',('p2cccd'),('p2cccd'),'','','');
FILE_SCHEMA(('AP242_MANAGED_MODEL_BASED_3D_ENGINEERING_MIM_LF'));
ENDSEC;
DATA;
#1=PRODUCT('gear','gear','',(#2));
#2=PRODUCT_CONTEXT('',#3,'mechanical');
#3=APPLICATION_CONTEXT('configuration controlled 3d designs');
#4=SI_UNIT(.MILLI.,.METRE.);
ENDSEC;
END-ISO-10303-21;
"""


def _write_obj(path: Path, text: str = OBJ_A) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_license(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "LICENSE").write_text("fixture license", encoding="utf-8")


def test_better_step_adapter_discovers_native_step_and_preprocess_records(tmp_path: Path) -> None:
    root = tmp_path / "better_step"
    _write_license(root)
    step_path = root / "assembly" / "gear.step"
    step_path.parent.mkdir()
    step_path.write_text(STEP_SAMPLE, encoding="utf-8")
    step_path.with_suffix(".json").write_text(json.dumps({"output_stem": "gear_mesh"}), encoding="utf-8")

    adapter = BetterSTEPAdapter(root)
    assets = adapter.list_assets()
    records = adapter.generate_preprocess_records(assets=assets, target_mesh_format="obj")

    assert adapter.license().available()
    assert len(assets) == 1
    assert assets[0].entity_count == 4
    assert assets[0].product_names == ("gear",)
    assert assets[0].schema_names == ("AP242_MANAGED_MODEL_BASED_3D_ENGINEERING_MIM_LF",)
    assert records[0].output_stem == "gear_mesh"
    assert records[0].metadata["requires_meshing_backend"] is True


def test_ycb_and_google_scanned_objects_generate_robot_validation_samples(tmp_path: Path) -> None:
    ycb_root = tmp_path / "ycb"
    gso_root = tmp_path / "gso"
    _write_license(ycb_root)
    _write_license(gso_root)
    _write_obj(ycb_root / "003_cracker_box" / "textured.obj", OBJ_A)
    _write_obj(ycb_root / "004_sugar_box" / "textured.obj", OBJ_B)
    _write_obj(gso_root / "mug" / "mesh.obj", OBJ_A)
    _write_obj(gso_root / "bowl" / "mesh.obj", OBJ_B)

    ycb = YCBObjectSetAdapter(ycb_root)
    gso = GoogleScannedObjectsAdapter(gso_root)
    ycb_samples = ycb.generate_robot_validation_samples(limit=1)
    gso_samples = gso.generate_robot_validation_samples(limit=1)

    assert len(ycb.list_assets()) == 2
    assert len(gso.list_assets()) == 2
    assert ycb_samples[0].motion_type == "robot_gripper_object_approach"
    assert gso_samples[0].metadata["allow_slowdown_not_false_negative"] is True
    assert ycb_samples[0].center_a_t0 != ycb_samples[0].center_a_t1


def test_moveit_resources_adapter_discovers_planning_scene_and_motion_queries(tmp_path: Path) -> None:
    root = tmp_path / "moveit_resources"
    _write_license(root)
    scene = root / "panda" / "scene.json"
    scene.parent.mkdir()
    scene.write_text(
        json.dumps({"robot_name": "panda", "links": ["panda_link0", "panda_link1", "panda_link2"]}),
        encoding="utf-8",
    )
    _write_obj(root / "panda" / "meshes" / "link0.obj", OBJ_A)

    adapter = MoveItResourcesAdapter(root)
    scenes = adapter.list_scene_assets()
    queries = adapter.generate_motion_queries(limit=2)

    assert len(scenes) == 1
    assert scenes[0].robot_name == "panda"
    assert scenes[0].metadata["mesh_count"] == 1
    assert len(queries) == 2
    assert queries[0].link_a == "panda_link0"
    assert queries[0].metadata["benchmark_resource"] is True


def test_thingi10k_adapter_generates_ood_stress_samples_with_dirty_metadata(tmp_path: Path) -> None:
    root = tmp_path / "thingi10k"
    _write_license(root)
    _write_obj(root / "dirty_nonmanifold.obj", OBJ_A)
    (root / "dirty_nonmanifold.json").write_text(
        json.dumps({"dirty": True, "non_manifold": True}),
        encoding="utf-8",
    )
    _write_obj(root / "clean.obj", OBJ_B)

    adapter = Thingi10KAdapter(root)
    assets = adapter.list_assets()
    samples = adapter.generate_ood_stress_samples(assets=assets, limit=1)

    assert assets[0].dirty_score > 0.0
    assert samples[0].metadata["fallback_expected"] is True
    assert samples[0].metadata["proxy_inflation_scale"] == 1.5


def test_partnet_and_partnet_mobility_adapters_cover_part_aware_and_articulated_scenes(tmp_path: Path) -> None:
    partnet_root = tmp_path / "partnet"
    mobility_root = tmp_path / "partnet_mobility"
    _write_license(partnet_root)
    _write_license(mobility_root)
    _write_obj(partnet_root / "chair" / "seat.obj", OBJ_A)
    (partnet_root / "chair" / "seat.json").write_text(
        json.dumps({"part_count": 3, "category": "chair"}),
        encoding="utf-8",
    )
    scene_root = mobility_root / "drawer_001"
    _write_obj(scene_root / "base.obj", OBJ_A)
    _write_obj(scene_root / "drawer.obj", OBJ_B)
    (scene_root / "mobility.json").write_text(
        json.dumps({"category": "drawer", "joints": [{"type": "prismatic"}]}),
        encoding="utf-8",
    )

    partnet = PartNetAdapter(partnet_root)
    mobility = PartNetMobilityAdapter(mobility_root)
    assets = partnet.list_assets()
    scenes = mobility.list_scenes()
    samples = mobility.generate_articulated_motion_samples(limit=1)

    assert assets[0].part_count == 3
    assert assets[0].metadata["part_aware"] is True
    assert scenes[0].joint_count == 1
    assert len(scenes[0].assets) == 2
    assert samples[0].motion_type == "articulated_part_motion"
    assert samples[0].metadata["part_aware"] is True


def test_shapenet_and_objaverse_xl_adapters_generate_large_scale_ood_subset_samples(tmp_path: Path) -> None:
    shapenet_root = tmp_path / "shapenet"
    objaverse_root = tmp_path / "objaverse_xl"
    _write_license(shapenet_root)
    _write_license(objaverse_root)
    _write_obj(shapenet_root / "chair" / "model_a" / "model.obj", OBJ_A)
    _write_obj(shapenet_root / "table" / "model_b" / "model.obj", OBJ_B)
    _write_obj(objaverse_root / "uid_a" / "mesh.obj", OBJ_A)
    _write_obj(objaverse_root / "uid_b" / "mesh.obj", OBJ_B)
    (objaverse_root / "uid_a" / "mesh.json").write_text(
        json.dumps({"name": "uid_a", "category": "household", "dirty_score": 0.25}),
        encoding="utf-8",
    )

    shapenet = ShapeNetAdapter(shapenet_root)
    objaverse = ObjaverseXLAdapter(objaverse_root)
    shapenet_samples = shapenet.generate_ood_subset_samples(limit=1)
    objaverse_assets = objaverse.list_assets()
    objaverse_samples = objaverse.generate_ood_subset_samples(assets=objaverse_assets, limit=1)

    assert len(shapenet.list_assets()) == 2
    assert shapenet_samples[0].metadata["ood_subset"] is True
    assert len(objaverse_assets) == 2
    assert objaverse_assets[0].source_name == "Objaverse-XL"
    assert any(asset.dirty_score > 0.0 for asset in objaverse_assets)
    assert objaverse_samples[0].motion_type == "large_scale_ood_object_approach"
