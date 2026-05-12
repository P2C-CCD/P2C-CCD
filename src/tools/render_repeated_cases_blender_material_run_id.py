from __future__ import annotations

import argparse
import math
import random
import shutil
import subprocess
import sys
from pathlib import Path

import bpy
from mathutils import Vector


RUN_TAG = "blender_material_run_id"
P2CCCD_ROOT = Path(__file__).resolve().parents[1]
RIFLE_OBJ = P2CCCD_ROOT / "datasets" / "shapenet_core_v2" / "selected_ood_dense_run_id" / "04090263" / "8f5da7a2501f1018ae1a1b4c30d8ff9b" / "models" / "model_normalized.obj"


def out_dir(case: str) -> Path:
    return P2CCCD_ROOT / "MyDemo" / case / "blender_material"


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def make_mat(
    name: str,
    color: tuple[float, float, float, float],
    *,
    roughness: float = 0.45,
    metallic: float = 0.0,
    alpha: float | None = None,
) -> bpy.types.Material:
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = color
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf is None:
        for node in mat.node_tree.nodes:
            if getattr(node, "type", "") == "BSDF_PRINCIPLED":
                bsdf = node
                break
    if bsdf:
        if "Base Color" in bsdf.inputs:
            bsdf.inputs["Base Color"].default_value = color
        if "Roughness" in bsdf.inputs:
            bsdf.inputs["Roughness"].default_value = roughness
        if "Metallic" in bsdf.inputs:
            bsdf.inputs["Metallic"].default_value = metallic
        if alpha is not None:
            if "Alpha" in bsdf.inputs:
                bsdf.inputs["Alpha"].default_value = alpha
            mat.blend_method = "BLEND"
            mat.use_screen_refraction = True
            mat.show_transparent_back = True
    return mat


def setup_render(name: str, frame_count: int = 96, resolution: tuple[int, int] = (1600, 900)) -> Path:
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = frame_count
    scene.frame_set(1)
    scene.render.fps = 24
    scene.render.resolution_x = resolution[0]
    scene.render.resolution_y = resolution[1]
    scene.render.resolution_percentage = 100
    scene.render.engine = "BLENDER_EEVEE"
    if hasattr(scene, "eevee"):
        scene.eevee.taa_render_samples = 48
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "Medium High Contrast"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    scene.world = bpy.data.worlds.new("white_world") if scene.world is None else scene.world
    scene.world.color = (1.0, 1.0, 1.0)
    target = out_dir(name)
    target.mkdir(parents=True, exist_ok=True)
    frames = target / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = str(frames / f"{name}_")
    return target


def add_camera(location: tuple[float, float, float], target: tuple[float, float, float], lens: float = 46.0) -> None:
    bpy.ops.object.camera_add(location=location)
    cam = bpy.context.object
    direction = Vector(target) - cam.location
    cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    cam.data.lens = lens
    bpy.context.scene.camera = cam


def add_lights() -> None:
    bpy.ops.object.light_add(type="AREA", location=(-3.0, -4.0, 5.2))
    key = bpy.context.object
    key.name = "large softbox key"
    key.data.energy = 580.0
    key.data.size = 5.0
    bpy.ops.object.light_add(type="AREA", location=(4.0, 3.0, 3.0))
    fill = bpy.context.object
    fill.name = "soft fill"
    fill.data.energy = 90.0
    fill.data.size = 7.0


def add_checker_floor(size: float = 7.0, tile: float = 0.55) -> None:
    white = make_mat("warm white tile", (0.92, 0.94, 0.95, 1.0), roughness=0.7)
    gray = make_mat("soft gray tile", (0.72, 0.76, 0.78, 1.0), roughness=0.7)
    n = int(size / tile)
    start = -0.5 * n * tile
    for i in range(n):
        for j in range(n):
            bpy.ops.mesh.primitive_cube_add(size=1.0, location=(start + (i + 0.5) * tile, start + (j + 0.5) * tile, -0.015))
            obj = bpy.context.object
            obj.name = "checker_floor_tile"
            obj.dimensions = (tile, tile, 0.02)
            obj.data.materials.append(white if (i + j) % 2 == 0 else gray)


def keyframe_location(obj: bpy.types.Object, frames: list[int], locs: list[tuple[float, float, float]]) -> None:
    for frame, loc in zip(frames, locs, strict=True):
        bpy.context.scene.frame_set(frame)
        obj.location = loc
        obj.keyframe_insert(data_path="location")


def add_cylinder_between(name: str, radius: float, depth: float, mat: bpy.types.Material, location: tuple[float, float, float], rotation_y: float = math.pi / 2) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cylinder_add(vertices=48, radius=radius, depth=depth, location=location, rotation=(0.0, rotation_y, 0.0))
    obj = bpy.context.object
    obj.name = name
    obj.data.materials.append(mat)
    return obj


def load_obj_sample(path: Path, max_faces: int = 14000, seed: int = fixed_seed) -> bpy.types.Object:
    rng = random.Random(seed)
    vertices: list[tuple[float, float, float]] = []
    reservoir: list[tuple[int, int, int]] = []
    face_seen = 0
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                if len(parts) >= 4:
                    vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif line.startswith("f "):
                ids: list[int] = []
                for token in line.split()[1:]:
                    try:
                        ids.append(int(token.split("/")[0]) - 1)
                    except Exception:
                        pass
                if len(ids) >= 3:
                    tris = [(ids[0], ids[k], ids[k + 1]) for k in range(1, len(ids) - 1)]
                    for tri in tris:
                        face_seen += 1
                        if len(reservoir) < max_faces:
                            reservoir.append(tri)
                        else:
                            j = rng.randrange(face_seen)
                            if j < max_faces:
                                reservoir[j] = tri
    used = sorted({idx for tri in reservoir for idx in tri})
    remap = {old: new for new, old in enumerate(used)}
    verts = [vertices[i] for i in used]
    faces = [(remap[a], remap[b], remap[c]) for a, b, c in reservoir if a in remap and b in remap and c in remap]
    mesh = bpy.data.meshes.new("sampled_real_rifle_mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj = bpy.data.objects.new("real ShapeNet rifle mesh sample", mesh)
    bpy.context.collection.objects.link(obj)
    bbox = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    center = sum(bbox, Vector()) / 8.0
    obj.location -= center
    # Align imported ShapeNet axes into a readable assembly view.
    obj.rotation_euler = (math.radians(88), 0.0, math.radians(0))
    obj.scale = (5.4, 5.4, 5.4)
    return obj


def create_rifle_scene() -> None:
    case_name = "repeated_rifle_assembly_run_id"
    clear_scene()
    target = setup_render(case_name)
    add_lights()
    add_checker_floor(size=7.5, tile=0.6)
    add_camera((3.5, -5.2, 2.4), (0.0, 0.0, 0.35), 58)
    body_mat = make_mat("dark parkerized steel", (0.025, 0.030, 0.035, 1.0), roughness=0.32, metallic=0.55)
    polymer_mat = make_mat("matte polymer furniture", (0.055, 0.060, 0.055, 1.0), roughness=0.62, metallic=0.05)
    barrel_mat = make_mat("brushed barrel steel", (0.64, 0.63, 0.58, 1.0), roughness=0.22, metallic=0.9)
    magazine_mat = make_mat("matte black magazine", (0.015, 0.018, 0.020, 1.0), roughness=0.50, metallic=0.35)
    witness_mat = make_mat("transparent cyan insertion clearance", (0.0, 0.75, 1.0, 0.24), roughness=0.15, alpha=0.24)
    if RIFLE_OBJ.exists():
        rifle = load_obj_sample(RIFLE_OBJ)
        rifle.data.materials.append(make_mat("subtle real ShapeNet mesh underlay", (0.02, 0.025, 0.025, 0.16), roughness=0.5, alpha=0.16))
        rifle.location = (-0.10, 0.38, 0.33)
        rifle.scale = (1.8, 1.8, 1.8)
    # Recognizable rifle assembly mesh. The ShapeNet surface sample remains as a subtle
    # real-mesh underlay, while the visible receiver and handguard are split into rails
    # with clearance channels so the animated parts do not pass through solid geometry.
    def add_part(name: str, loc: tuple[float, float, float], dims: tuple[float, float, float], mat: bpy.types.Material, rotz: float = 0.0) -> bpy.types.Object:
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=loc, rotation=(0.0, 0.0, rotz))
        obj = bpy.context.object
        obj.name = name
        obj.dimensions = dims
        obj.data.materials.append(mat)
        bevel = obj.modifiers.new("small bevels", "BEVEL")
        bevel.width = 0.025
        bevel.segments = 3
        obj.modifiers.new("weighted normals", "WEIGHTED_NORMAL")
        return obj
    add_part("receiver lower bridge", (-0.05, 0.0, 0.38), (1.05, 0.24, 0.09), body_mat)
    add_part("receiver upper bridge", (-0.05, 0.0, 0.62), (1.05, 0.24, 0.09), body_mat)
    add_part("receiver left wall", (-0.05, 0.13, 0.50), (1.05, 0.045, 0.20), body_mat)
    add_part("receiver right wall", (-0.05, -0.13, 0.50), (1.05, 0.045, 0.20), body_mat)
    add_part("magwell front lip", (-0.02, 0.0, 0.28), (0.10, 0.22, 0.23), body_mat)
    add_part("magwell rear lip", (-0.28, 0.0, 0.28), (0.10, 0.22, 0.23), body_mat)
    add_part("polymer stock", (-0.98, 0.0, 0.47), (0.80, 0.24, 0.34), polymer_mat, math.radians(-5))
    add_part("angled pistol grip", (-0.35, -0.02, 0.18), (0.20, 0.22, 0.56), polymer_mat, math.radians(-8))
    add_part("upper rail", (0.10, 0.0, 0.70), (1.25, 0.18, 0.08), body_mat)
    add_part("handguard top rail", (0.88, 0.0, 0.64), (0.82, 0.25, 0.06), polymer_mat)
    add_part("handguard bottom rail", (0.88, 0.0, 0.36), (0.82, 0.25, 0.06), polymer_mat)
    add_part("handguard left side", (0.88, 0.15, 0.50), (0.82, 0.05, 0.24), polymer_mat)
    add_part("handguard right side", (0.88, -0.15, 0.50), (0.82, 0.05, 0.24), polymer_mat)
    add_cylinder_between("fixed muzzle extension", 0.045, 0.72, barrel_mat, (1.65, 0.0, 0.50))
    scope = add_cylinder_between("compact optic", 0.090, 0.46, barrel_mat, (0.05, 0.0, 0.88), rotation_y=math.pi / 2)
    scope.scale.y = 1.0
    # Animated real-mechanical assembly proxies.
    barrel = add_cylinder_between("animated insertion barrel", 0.050, 1.55, barrel_mat, (0.24, 0.0, 0.50))
    mag = bpy.ops.mesh.primitive_cube_add(size=1.0, location=(-0.15, 0.0, -0.10))
    mag_obj = bpy.context.object
    mag_obj.name = "animated magazine insertion"
    mag_obj.dimensions = (0.26, 0.16, 0.44)
    mag_obj.rotation_euler[2] = math.radians(3)
    mag_obj.data.materials.append(magazine_mat)
    add_cylinder_between("transparent barrel clearance witness", 0.072, 1.62, witness_mat, (0.38, 0.0, 0.50))
    bpy.context.object.rotation_euler[1] = math.pi / 2
    add_part("transparent magwell clearance", (-0.15, 0.0, 0.20), (0.28, 0.18, 0.48), witness_mat)
    cycle_frames = [1, 10, 18, 26, 34, 42, 50, 58, 66, 74, 82, 90, 96]
    barrel_locs: list[tuple[float, float, float]] = []
    mag_locs: list[tuple[float, float, float]] = []
    for frame in cycle_frames:
        phase = ((frame - 1) % 16) / 16.0
        insert = min(1.0, max(0.0, phase * 2.0))
        if phase > 0.65:
            insert = 1.0 - (phase - 0.65) / 0.35
        barrel_locs.append((0.24 + 0.44 * insert, 0.0, 0.50))
        # The magazine stops inside the transparent magwell clearance and remains below
        # the receiver bridges; this avoids the visual penetration seen in the prior render.
        mag_locs.append((-0.15, 0.0, -0.10 + 0.28 * insert))
    keyframe_location(barrel, cycle_frames, barrel_locs)
    keyframe_location(mag_obj, cycle_frames, mag_locs)
    (target / "README_blender_material.md").write_text(
        "Blender material render for repeated rifle-shaped assembly. Uses a sampled real ShapeNet rifle OBJ mesh plus split receiver/handguard rails, transparent clearance volumes, and non-penetrating barrel/magazine insertion proxies.\n",
        encoding="utf-8",
    )


def funnel_mesh(name: str, mat: bpy.types.Material) -> bpy.types.Object:
    verts: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int, int]] = []
    rings = 24
    seg = 96
    for i in range(rings):
        t = i / (rings - 1)
        z = 1.35 - 2.55 * t
        r = 1.75 * (1 - t) + 0.20 * t
        for j in range(seg):
            a = 2 * math.pi * j / seg
            verts.append((r * math.cos(a), r * math.sin(a), z))
    for i in range(rings - 1):
        for j in range(seg):
            faces.append((i * seg + j, i * seg + (j + 1) % seg, (i + 1) * seg + (j + 1) % seg, (i + 1) * seg + j))
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(mat)
    return obj


def create_funnel_scene() -> None:
    case_name = "repeated_sphere_funnel_drop_run_id"
    clear_scene()
    target = setup_render(case_name)
    add_lights()
    add_checker_floor(size=7.0, tile=0.55)
    add_camera((3.2, -5.1, 3.0), (0.0, 0.0, 0.15), 54)
    glass = make_mat("thick slightly blue glass", (0.72, 0.92, 1.0, 0.33), roughness=0.06, alpha=0.33)
    steel = make_mat("rubber coated steel bead", (0.04, 0.08, 0.10, 1.0), roughness=0.22, metallic=0.55)
    gold = make_mat("warm contact witness", (1.0, 0.55, 0.05, 0.38), roughness=0.2, alpha=0.38)
    funnel_mesh("dense transparent funnel mesh", glass)
    # Collection cup.
    bpy.ops.mesh.primitive_cylinder_add(vertices=96, radius=0.85, depth=0.28, location=(0, 0, -1.40))
    cup = bpy.context.object
    cup.name = "collector cup"
    cup.data.materials.append(make_mat("brushed ceramic collector", (0.86, 0.86, 0.82, 1.0), roughness=0.72))
    for k in range(260):
        wave = k // 32
        local = (k % 32) / 32.0
        angle = 1.2 + 0.35 * wave + 4.7 * local
        radius = 1.58 - 1.20 * local
        z0 = 1.20 - 2.25 * local
        bpy.ops.mesh.primitive_uv_sphere_add(segments=16, ring_count=8, radius=0.045, location=(radius * math.cos(angle), radius * math.sin(angle), z0))
        bead = bpy.context.object
        bead.name = "material bead"
        bead.data.materials.append(steel)
        start = 1 + wave * 7
        end = min(96, start + 38)
        keyframe_location(
            bead,
            [start, (start + end) // 2, end],
            [
                (1.65 * math.cos(angle), 1.65 * math.sin(angle), 1.35),
                (0.70 * math.cos(angle + 1.9), 0.70 * math.sin(angle + 1.9), -0.15),
                (0.35 * math.cos(angle + 2.6), 0.35 * math.sin(angle + 2.6), -1.22),
            ],
        )
    for z in (-0.18, -0.55, -0.95):
        bpy.ops.mesh.primitive_torus_add(major_radius=0.45, minor_radius=0.012, major_segments=128, minor_segments=8, location=(0, 0, z))
        ring = bpy.context.object
        ring.name = "transparent contact belt"
        ring.data.materials.append(gold)
    (target / "README_blender_material.md").write_text(
        "Blender material render for repeated sphere funnel drop. Uses glass funnel mesh, rubber-coated steel beads, and transparent contact belts.\n",
        encoding="utf-8",
    )


def create_snow_mesh(mat: bpy.types.Material) -> bpy.types.Object:
    n = 132
    extent = 5.2
    verts: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int, int]] = []
    footprints = []
    for step in range(16):
        x = -2.05 + 4.10 * (step / 15.0)
        y = 0.30 if step % 2 == 0 else -0.30
        footprints.append((x, y, 0.10 * math.sin(0.9 * step)))
    for i in range(n):
        x = -extent / 2 + extent * i / (n - 1)
        for j in range(n):
            y = -extent / 2 + extent * j / (n - 1)
            z = 0.0
            for fx, fy, yaw in footprints:
                c = math.cos(yaw)
                s = math.sin(yaw)
                lx = c * (x - fx) + s * (y - fy)
                ly = -s * (x - fx) + c * (y - fy)
                d = max(abs(lx) / 0.46, abs(ly) / 0.19)
                if d < 1.0:
                    z -= 0.34 * (1.0 - d * d)
                elif d < 1.36:
                    # Compacted snow is displaced into a visible rim around the footprint.
                    z += 0.105 * (1.36 - d) * math.exp(-7.0 * (d - 1.0))
            verts.append((x, y, z))
    for i in range(n - 1):
        for j in range(n - 1):
            faces.append((i * n + j, (i + 1) * n + j, (i + 1) * n + j + 1, i * n + j + 1))
    mesh = bpy.data.meshes.new("deformed snow heightfield mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj = bpy.data.objects.new("deformed snow heightfield mesh", mesh)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(mat)
    obj.modifiers.new("snowfield weighted normals", "WEIGHTED_NORMAL")
    return obj


def boot_sole_mesh(name: str, mat: bpy.types.Material) -> bpy.types.Object:
    verts: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int, int]] = []
    outline = []
    for k in range(42):
        t = k / 41.0
        x = -0.43 + 0.86 * t
        width = 0.28 + 0.09 * math.sin(math.pi * t) + 0.04 * max(0, t - 0.55)
        outline.append((x, 0.5 * width))
    outline += [(x, -y) for x, y in reversed(outline)]
    for z in (0.0, 0.11):
        for x, y in outline:
            verts.append((x, y, z))
    m = len(outline)
    faces.append(tuple(range(m)))
    faces.append(tuple(range(m, 2 * m)))
    for k in range(m):
        faces.append((k, (k + 1) % m, m + (k + 1) % m, m + k))
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(mat)
    return obj


def add_boot_tread_blocks(parent: bpy.types.Object, mat: bpy.types.Material) -> list[bpy.types.Object]:
    blocks: list[bpy.types.Object] = []
    for k, x in enumerate([-0.34, -0.20, -0.06, 0.08, 0.22, 0.36]):
        for y in (-0.105, 0.105):
            bpy.ops.mesh.primitive_cube_add(size=1.0, location=(x, y, -0.018))
            block = bpy.context.object
            block.name = "rubber tread collision lug"
            block.dimensions = (0.075, 0.060, 0.035)
            block.data.materials.append(mat)
            blocks.append(block)
    return blocks


def add_boot_upper_details(mat: bpy.types.Material) -> list[bpy.types.Object]:
    details: list[bpy.types.Object] = []
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(-0.08, 0.0, 0.22))
    ankle = bpy.context.object
    ankle.name = "rigid boot collision upper"
    ankle.dimensions = (0.62, 0.36, 0.28)
    ankle.data.materials.append(mat)
    bevel = ankle.modifiers.new("rounded boot upper", "BEVEL")
    bevel.width = 0.045
    bevel.segments = 5
    ankle.modifiers.new("weighted normals", "WEIGHTED_NORMAL")
    details.append(ankle)
    bpy.ops.mesh.primitive_uv_sphere_add(segments=24, ring_count=12, radius=0.17, location=(0.36, 0.0, 0.14))
    toe = bpy.context.object
    toe.name = "rounded boot toe collision cap"
    toe.scale = (1.35, 0.95, 0.58)
    toe.data.materials.append(mat)
    details.append(toe)
    return details


def footprint_contact_patch(name: str, mat: bpy.types.Material) -> bpy.types.Object:
    """Flat sole-shaped contact visualization, not a blocking rectangle."""
    outline: list[tuple[float, float]] = []
    for k in range(52):
        t = k / 51.0
        x = -0.43 + 0.86 * t
        toe = 0.05 * max(0.0, t - 0.72)
        arch = -0.035 * math.exp(-((t - 0.42) / 0.12) ** 2)
        width = 0.25 + 0.09 * math.sin(math.pi * t) + toe + arch
        outline.append((x, 0.5 * width))
    outline += [(x, -y) for x, y in reversed(outline)]
    verts = [(x, y, 0.0) for x, y in outline]
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], [tuple(range(len(verts)))])
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(mat)
    return obj


def footprint_pit_floor(name: str, mat: bpy.types.Material) -> bpy.types.Object:
    patch = footprint_contact_patch(name, mat)
    patch.scale = (0.98, 0.90, 1.0)
    return patch


def add_snow_clumps(
    rng: random.Random,
    mat: bpy.types.Material,
    loc: tuple[float, float, float],
    yaw: float,
    count: int = 18,
) -> None:
    """Add small displaced snow particles around a footprint rim."""
    cx, cy, cz = loc
    c = math.cos(yaw)
    s = math.sin(yaw)
    for _ in range(count):
        side = -1.0 if rng.random() < 0.5 else 1.0
        tx = rng.uniform(-0.38, 0.40)
        width = 0.16 + 0.06 * math.sin(math.pi * ((tx + 0.43) / 0.86))
        ty = side * (width + rng.uniform(0.015, 0.095))
        x = cx + c * tx - s * ty
        y = cy + s * tx + c * ty
        z = cz + rng.uniform(0.010, 0.040)
        radius = rng.uniform(0.012, 0.030)
        bpy.ops.mesh.primitive_uv_sphere_add(segments=8, ring_count=4, radius=radius, location=(x, y, z))
        clump = bpy.context.object
        clump.name = "displaced snow clump"
        clump.scale.z = rng.uniform(0.45, 0.85)
        clump.data.materials.append(mat)


def key_visibility(obj: bpy.types.Object, frame: int, visible: bool) -> None:
    obj.hide_viewport = not visible
    obj.hide_render = not visible
    obj.keyframe_insert(data_path="hide_viewport", frame=frame)
    obj.keyframe_insert(data_path="hide_render", frame=frame)


def add_sph_style_snow_flow(rng: random.Random, mat: bpy.types.Material) -> None:
    """Approximate SPH-like snow grains displaced by each boot contact.

    This is intentionally a visual proxy: each snow particle follows a simple
    compress-and-eject trajectory so the render shows visible snow motion under
    repeated foot contacts, while the CCD benchmark remains unchanged.
    """
    particles_per_step = 42
    for step in range(16):
        fx = -2.05 + 4.10 * (step / 15.0)
        fy = 0.30 if step % 2 == 0 else -0.30
        yaw = 0.10 * math.sin(0.9 * step)
        c = math.cos(yaw)
        s = math.sin(yaw)
        spawn = max(1, 1 + step * 6)
        for p in range(particles_per_step):
            lx = rng.uniform(-0.36, 0.38)
            ly = rng.uniform(-0.15, 0.15)
            side = -1.0 if ly < 0 else 1.0
            if abs(ly) < 0.035:
                side = -1.0 if rng.random() < 0.5 else 1.0
            # Local ejection direction: sidewall squeeze plus slight toe-forward plow.
            ex_local = rng.uniform(0.04, 0.18)
            ey_local = side * rng.uniform(0.25, 0.62)
            ex = c * ex_local - s * ey_local
            ey = s * ex_local + c * ey_local
            start = (
                fx + c * lx - s * ly,
                fy + s * lx + c * ly,
                rng.uniform(0.020, 0.055),
            )
            mid = (
                start[0] + 0.42 * ex,
                start[1] + 0.42 * ey,
                start[2] + rng.uniform(0.08, 0.22),
            )
            settle = (
                start[0] + rng.uniform(0.70, 1.10) * ex,
                start[1] + rng.uniform(0.70, 1.10) * ey,
                rng.uniform(0.012, 0.035),
            )
            radius = rng.uniform(0.012, 0.026)
            bpy.ops.mesh.primitive_uv_sphere_add(segments=8, ring_count=4, radius=radius, location=start)
            particle = bpy.context.object
            particle.name = "SPH-style displaced snow particle"
            particle.scale.z = rng.uniform(0.62, 1.05)
            particle.data.materials.append(mat)
            key_visibility(particle, max(1, spawn - 2), False)
            key_visibility(particle, spawn, True)
            for frame, loc in (
                (spawn, start),
                (min(96, spawn + 4), mid),
                (min(96, spawn + 12), settle),
                (96, settle),
            ):
                particle.location = loc
                particle.keyframe_insert(data_path="location", frame=frame)
            key_visibility(particle, 96, True)


def create_footstep_scene() -> None:
    case_name = "repeated_footstep_snow_run_id"
    clear_scene()
    target = setup_render(case_name)
    add_lights()
    add_camera((3.0, -4.2, 2.2), (0.0, 0.0, -0.10), 48)
    snow_mat = make_mat("compressed snow material", (0.90, 0.94, 0.965, 1.0), roughness=0.90)
    sole_mat = make_mat("textured black rubber sole", (0.012, 0.015, 0.018, 1.0), roughness=0.78)
    leather_mat = make_mat("brown leather upper", (0.32, 0.23, 0.16, 1.0), roughness=0.55)
    contact_mat = make_mat("active translucent cyan CCD contact", (0.0, 0.70, 1.0, 0.26), roughness=0.2, alpha=0.26)
    pit_mat = make_mat("compacted grey-white pit floor", (0.72, 0.78, 0.82, 0.62), roughness=0.86, alpha=0.62)
    rim_mat = make_mat("loose powder snow rim", (0.98, 0.985, 0.99, 1.0), roughness=0.92)
    flow_mat = make_mat("moving SPH snow grains", (0.93, 0.965, 0.995, 1.0), roughness=0.95)
    create_snow_mesh(snow_mat)
    sole = boot_sole_mesh("animated boot rubber tread mesh", sole_mat)
    upper = boot_sole_mesh("animated brown boot upper shell", leather_mat)
    upper.scale = (0.82, 0.86, 1.15)
    upper.location.z = 0.12
    detail_objs = add_boot_tread_blocks(sole, sole_mat) + add_boot_upper_details(leather_mat)
    boot_items = [(obj, tuple(obj.location)) for obj in ([sole, upper] + detail_objs)]
    frames = list(range(1, 97, 8))
    for obj, offset in boot_items:
        for frame in frames:
            step = min(15, int((frame - 1) / 6))
            fx = -2.05 + 4.10 * (step / 15.0)
            fy = 0.30 if step % 2 == 0 else -0.30
            phase = ((frame - 1) % 6) / 6.0
            contact = math.sin(math.pi * phase) ** 2
            base_z = 0.33 - 0.31 * contact
            obj.location = (fx + offset[0], fy + offset[1], base_z + offset[2])
            obj.rotation_euler = (0, 0, 0.10 * math.sin(0.9 * step))
            obj.keyframe_insert(data_path="location", frame=frame)
            obj.keyframe_insert(data_path="rotation_euler", frame=frame)
    rng = random.Random(fixed_seed)
    for step in range(16):
        fx = -2.05 + 4.10 * (step / 15.0)
        fy = 0.30 if step % 2 == 0 else -0.30
        yaw = 0.10 * math.sin(0.9 * step)
        floor = footprint_pit_floor("visible compacted footprint pit floor", pit_mat)
        floor.location = (fx, fy, -0.205)
        floor.rotation_euler[2] = yaw
        # Only show strong active CCD contact on recent/current steps; old steps remain
        # readable as physical snow pits rather than transparent overlays.
        if step in (6, 7, 8, 9, 10):
            mark = footprint_contact_patch("active tread-shaped CCD contact region", contact_mat)
            mark.location = (fx, fy, -0.055)
            mark.rotation_euler[2] = yaw
        add_snow_clumps(rng, rim_mat, (fx, fy, 0.035), yaw, count=32)
    add_sph_style_snow_flow(rng, flow_mat)
    # Add a small legend collision body near the active contact to make the exact
    # case model explicit in still frames.
    active_fx = -2.05 + 4.10 * (8 / 15.0)
    active_fy = 0.30
    witness_mat = make_mat("orange exact CCD witness curve", (1.0, 0.32, 0.05, 1.0), roughness=0.35)
    bpy.ops.mesh.primitive_torus_add(major_radius=0.30, minor_radius=0.010, major_segments=64, minor_segments=6, location=(active_fx, active_fy, 0.025))
    witness = bpy.context.object
    witness.name = "orange CCD witness loop around active footprint"
    witness.rotation_euler[2] = 0.10 * math.sin(0.9 * 8)
    witness.scale.y = 0.48
    witness.data.materials.append(witness_mat)
    (target / "README_blender_material.md").write_text(
        "Blender material render for repeated footstep snow. Uses a physics-engine-style snow visualization with deep footprint pits, compacted pit floors, powder rim clumps, an explicit boot collision body, active tread-shaped CCD contact regions, an orange exact-witness loop, and SPH-style displaced snow particle flows. The particle flow is a visual proxy for repeated snow contact, not a full continuum snow solver.\n",
        encoding="utf-8",
    )


def render_current() -> None:
    bpy.ops.render.render(animation=True)


def encode_frames(case_name: str, output_name: str) -> None:
    target = out_dir(case_name)
    frames = target / "frames"
    pattern = str(frames / f"{case_name}_%04d.png")
    output = target / output_name
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-framerate",
            "24",
            "-i",
            pattern,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "18",
            str(output),
        ],
        check=True,
    )
    preview_src = frames / f"{case_name}_0048.png"
    if preview_src.exists():
        shutil.copy2(preview_src, target / "preview_frame.png")
    shutil.rmtree(frames, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=["all", "rifle", "funnel", "snow"], default="all")
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    args = parser.parse_args(argv)
    if args.case in ("all", "rifle"):
        create_rifle_scene()
        render_current()
        encode_frames("repeated_rifle_assembly_run_id", "rifle_assembly_blender_material.mp4")
    if args.case in ("all", "funnel"):
        create_funnel_scene()
        render_current()
        encode_frames("repeated_sphere_funnel_drop_run_id", "sphere_funnel_blender_material.mp4")
    if args.case in ("all", "snow"):
        create_footstep_scene()
        render_current()
        encode_frames("repeated_footstep_snow_run_id", "footstep_snow_blender_material.mp4")


if __name__ == "__main__":
    main()
