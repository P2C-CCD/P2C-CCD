from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEMO_DIR = ROOT / "src" / "MyDemo" / "selected_real_ti_dense_group_large_run_id"
SCENE_FRAMES = ROOT / "src" / "baseline" / "datasets" / "continuous-collision-detection" / "cloth-ball" / "frames"


def find_tool(env_var: str, executable: str) -> str:
    configured = os.environ.get(env_var)
    if configured:
        return configured
    discovered = shutil.which(executable)
    return discovered or executable


BLENDER = find_tool("P2CCCD_BLENDER", "blender")
FFMPEG = find_tool("P2CCCD_FFMPEG", "ffmpeg")


BLENDER_SCRIPT = r'''
import math
import struct
import sys
from pathlib import Path

import bpy
from mathutils import Vector


scene_dir = Path(r"__SCENE_DIR__")
demo_dir = Path(r"__DEMO_DIR__")
frame_dir = demo_dir / "physical_scene_frames"
frame_dir.mkdir(parents=True, exist_ok=True)


def frame_index(path):
    stem = path.stem
    digits = "".join(ch for ch in stem if ch.isdigit())
    return int(digits) if digits else -1


def read_ply(path, with_faces=False):
    with path.open("rb") as handle:
        header = []
        while True:
            line = handle.readline()
            if not line:
                raise RuntimeError(f"Unexpected EOF in {path}")
            decoded = line.decode("ascii").strip()
            header.append(decoded)
            if decoded == "end_header":
                break
        fmt = next(line for line in header if line.startswith("format "))
        if "binary_big_endian" not in fmt:
            raise RuntimeError(f"Only binary_big_endian PLY is supported: {path}")
        vertex_count = int(next(line for line in header if line.startswith("element vertex")).split()[2])
        face_count = int(next(line for line in header if line.startswith("element face")).split()[2])
        vertex_blob = handle.read(vertex_count * 12)
        raw_vertices = [struct.unpack_from(">fff", vertex_blob, i * 12) for i in range(vertex_count)]
        # Dataset coordinates are y-up. Blender is z-up, so convert to x-z-y.
        vertices = [(x, z, y) for (x, y, z) in raw_vertices]
        faces = None
        if with_faces:
            faces = []
            for _ in range(face_count):
                count = struct.unpack(">B", handle.read(1))[0]
                if count != 3:
                    indices = struct.unpack(">" + "i" * count, handle.read(4 * count))
                    if len(indices) >= 3:
                        faces.append(tuple(indices[:3]))
                else:
                    faces.append(struct.unpack(">iii", handle.read(12)))
        return vertices, faces


all_frames = [
    path
    for path in scene_dir.glob("*.ply")
    if not path.name.startswith("._") and path.name != ".DS_Store"
]
all_frames = sorted(all_frames, key=frame_index)
if len(all_frames) < 2:
    raise RuntimeError(f"No full-scene frames found in {scene_dir}")

target_frames = 96
if len(all_frames) > target_frames:
    indices = [round(i * (len(all_frames) - 1) / (target_frames - 1)) for i in range(target_frames)]
    all_frames = [all_frames[i] for i in indices]

vertices0, faces = read_ply(all_frames[0], with_faces=True)
mesh = bpy.data.meshes.new("cloth_ball_full_scene_mesh")
mesh.from_pydata(vertices0, [], faces)
mesh.update()
obj = bpy.data.objects.new("ScalableCCD_cloth_ball_full_scene", mesh)
bpy.context.collection.objects.link(obj)

def make_emission_material(name, color, strength=0.72, alpha=1.0):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = color
    mat.use_nodes = True
    mat.blend_method = "OPAQUE" if alpha >= 0.999 else "BLEND"
    mat.use_screen_refraction = alpha < 0.999
    mat.show_transparent_back = alpha < 0.999
    mat.diffuse_color = (color[0], color[1], color[2], alpha)
    nodes = mat.node_tree.nodes
    for node in list(nodes):
        nodes.remove(node)
    output = nodes.new(type="ShaderNodeOutputMaterial")
    emission = nodes.new(type="ShaderNodeEmission")
    emission.inputs["Color"].default_value = (color[0], color[1], color[2], alpha)
    emission.inputs["Strength"].default_value = strength
    if alpha >= 0.999:
        mat.node_tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
        return mat
    transparent = nodes.new(type="ShaderNodeBsdfTransparent")
    mix = nodes.new(type="ShaderNodeMixShader")
    mix.inputs["Fac"].default_value = alpha
    mat.node_tree.links.new(transparent.outputs["BSDF"], mix.inputs[1])
    mat.node_tree.links.new(emission.outputs["Emission"], mix.inputs[2])
    mat.node_tree.links.new(mix.outputs["Shader"], output.inputs["Surface"])
    return mat


def make_checker_material(name):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    for node in list(nodes):
        nodes.remove(node)
    output = nodes.new(type="ShaderNodeOutputMaterial")
    emission = nodes.new(type="ShaderNodeEmission")
    checker = nodes.new(type="ShaderNodeTexChecker")
    checker.inputs["Color1"].default_value = (0.99, 0.99, 0.99, 1.0)
    checker.inputs["Color2"].default_value = (0.80, 0.82, 0.85, 1.0)
    checker.inputs["Scale"].default_value = 8.0
    emission.inputs["Strength"].default_value = 0.72
    mat.node_tree.links.new(checker.outputs["Color"], emission.inputs["Color"])
    mat.node_tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return mat


cloth_mat = make_emission_material("opaque deforming cloth surface", (0.04, 0.34, 0.95, 1.0), 0.74, 1.0)
ball_mat = make_emission_material("impacting ball surface", (1.0, 0.22, 0.12, 1.0), 0.82, 1.0)
wire_mat = make_emission_material("visible triangle mesh lines", (0.015, 0.025, 0.035, 1.0), 0.95, 1.0)
obj.data.materials.append(cloth_mat)
obj.data.materials.append(ball_mat)
obj.data.materials.append(wire_mat)
obj.data.materials.append(wire_mat)

# The first frame contains two disconnected components: the deforming cloth and
# the rigid ball. Color the smaller component as the ball, then play the source
# mesh sequence directly without visual-only constraints or geometry projection.
parents = list(range(len(vertices0)))


def find_component(index):
    while parents[index] != index:
        parents[index] = parents[parents[index]]
        index = parents[index]
    return index


def union_component(a, b):
    root_a = find_component(a)
    root_b = find_component(b)
    if root_a != root_b:
        parents[root_b] = root_a


for face in faces:
    for face_index in range(1, len(face)):
        union_component(face[0], face[face_index])

component_sizes = {}
for vertex_index in range(len(vertices0)):
    root = find_component(vertex_index)
    component_sizes[root] = component_sizes.get(root, 0) + 1
ball_component = min(component_sizes, key=component_sizes.get)
ball_vertex_index_set = {
    vertex_index
    for vertex_index in range(len(vertices0))
    if find_component(vertex_index) == ball_component
}

for poly in obj.data.polygons:
    poly.material_index = 1 if all(index in ball_vertex_index_set for index in poly.vertices) else 0

wire = obj.modifiers.new("paper-visible triangle mesh overlay", "WIREFRAME")
wire.thickness = 0.018
wire.use_even_offset = True
wire.use_replace = False
wire.material_offset = 2

for item in list(bpy.context.scene.objects):
    if item.name != obj.name:
        bpy.data.objects.remove(item, do_unlink=True)

world = bpy.context.scene.world or bpy.data.worlds.new("World")
bpy.context.scene.world = world
world.color = (1.0, 1.0, 1.0)

floor_mat = make_checker_material("gray-white checker floor")
studio_white_mat = make_emission_material("white studio background", (1.0, 1.0, 1.0, 1.0), 0.86, 1.0)
bpy.ops.mesh.primitive_plane_add(size=240, location=(0, 0, -0.20))
studio_floor = bpy.context.object
studio_floor.name = "white studio background plane"
studio_floor.data.materials.append(studio_white_mat)
bpy.ops.mesh.primitive_plane_add(size=58, location=(0, 0, -0.12))
floor = bpy.context.object
floor.name = "reference floor"
floor.data.materials.append(floor_mat)

bpy.ops.object.light_add(type="AREA", location=(0, -20, 34))
light = bpy.context.object
light.name = "large softbox"
light.data.energy = 950
light.data.size = 14
bpy.ops.object.light_add(type="AREA", location=(-26, 18, 16))
fill_light = bpy.context.object
fill_light.name = "side fill softbox"
fill_light.data.energy = 230
fill_light.data.size = 18
bpy.ops.object.camera_add(location=(44, -52, 35))
camera = bpy.context.object
bpy.context.scene.camera = camera
target = Vector((0.0, 0.0, 5.0))
direction = target - camera.location
camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
camera.data.type = "ORTHO"
camera.data.ortho_scale = 82
camera.data.clip_end = 300
camera.data.dof.use_dof = False

try:
    bpy.context.scene.render.engine = "BLENDER_EEVEE_NEXT"
except TypeError:
    bpy.context.scene.render.engine = "BLENDER_EEVEE"
if hasattr(bpy.context.scene, "eevee"):
    bpy.context.scene.eevee.taa_render_samples = 64
bpy.context.scene.render.resolution_x = 1920
bpy.context.scene.render.resolution_y = 1080
bpy.context.scene.render.film_transparent = False
bpy.context.scene.view_settings.view_transform = "Standard"
bpy.context.scene.view_settings.look = "Medium High Contrast"
bpy.context.scene.view_settings.exposure = 0.15
bpy.context.scene.view_settings.gamma = 1.0

mesh_vertices = obj.data.vertices
for frame_id, ply_path in enumerate(all_frames):
    vertices, _ = read_ply(ply_path, with_faces=False)
    if len(vertices) != len(mesh_vertices):
        raise RuntimeError(f"Vertex count changed in {ply_path}")
    for vertex, co in zip(mesh_vertices, vertices):
        vertex.co = co
    obj.data.update()
    bpy.context.scene.frame_set(frame_id)
    bpy.context.scene.render.filepath = str(frame_dir / f"frame_{frame_id:04d}.png")
    bpy.ops.render.render(write_still=True)

print(f"rendered_frames={len(all_frames)}")
'''


def run_checked(args: list[str]) -> None:
    completed = subprocess.run(args, check=False, capture_output=True)
    if completed.returncode != 0:
        stdout = completed.stdout.decode("utf-8", errors="replace")
        stderr = completed.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"Command failed: {' '.join(args)}\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}")


def run_blender(args: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        completed = subprocess.run(args, check=False, stdout=log, stderr=subprocess.STDOUT)
    if completed.returncode != 0:
        raise RuntimeError(f"Blender failed. See log: {log_path}")


def encode_mp4(frame_dir: Path, output_mp4: Path) -> None:
    temp = output_mp4.with_suffix(".tmp.h264.mp4")
    if temp.exists():
        temp.unlink()
    run_checked(
        [
            FFMPEG,
            "-y",
            "-framerate",
            "24",
            "-i",
            str(frame_dir / "frame_%04d.png"),
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(temp),
        ]
    )
    temp.replace(output_mp4)


def main() -> None:
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    original = DEMO_DIR / "global.mp4"
    primitive_backup = DEMO_DIR / "primitive_query_global.mp4"
    if original.exists() and not primitive_backup.exists():
        shutil.copy2(original, primitive_backup)

    script_path = DEMO_DIR / "_render_physical_scene_blender.py"
    script_path.write_text(
        BLENDER_SCRIPT.replace("__SCENE_DIR__", SCENE_FRAMES.as_posix()).replace("__DEMO_DIR__", DEMO_DIR.as_posix()),
        encoding="utf-8",
    )
    blender_log = DEMO_DIR / "physical_scene_blender_render.log"
    run_blender([BLENDER, "--background", "--python", str(script_path)], blender_log)

    frame_dir = DEMO_DIR / "physical_scene_frames"
    global_mp4 = DEMO_DIR / "global.mp4"
    encode_mp4(frame_dir, global_mp4)

    frames = sorted(frame_dir.glob("frame_*.png"))
    from PIL import Image

    sheet = Image.new("RGB", (3840, 2160), (255, 255, 255))
    for k, idx in enumerate([0, len(frames) // 3, 2 * len(frames) // 3, len(frames) - 1]):
        img = Image.open(frames[idx]).convert("RGB")
        sheet.paste(img, ((k % 2) * 1920, (k // 2) * 1080))
    sheet_path = DEMO_DIR / "contact_sheet.png"
    sheet.save(sheet_path)

    report = DEMO_DIR / "case_report.md"
    report.write_text(
        "\n".join(
            [
                "# Selected-real TI Dense Group Physical Collision Scene",
                "",
                "This visualization now shows a complete full-scene mesh sequence, not only primitive-query geometry.",
                "",
                "Source scene:",
                "",
                f"- Full-scene frames: `{SCENE_FRAMES.as_posix()}`",
                "- Dataset: Continuous-Collision-Detection / Scalable-CCD ground-truth scene `cloth-ball`.",
                "- Geometry: binary PLY mesh sequence with complete triangle faces.",
                "- Visualization mode: raw source mesh sequence playback; no ball locking, no cloth projection, and no artificial contact correction.",
                "- Rendering: Blender 5.0 background render, full surface materials, H.264/yuv420p MP4.",
                "",
                "Relation to selected-real TI benchmark:",
                "",
                "- The selected-real TI dense group benchmark itself stores primitive queries and performance evidence.",
                "- This companion visualization is a true full-scene physical collision display from the same CCD dataset family.",
                "- Exact correctness and wall-time metrics remain documented in the benchmark report; this video is for paper-level scene illustration.",
                "",
                "Outputs:",
                "",
                f"- `global.mp4`: `{global_mp4.as_posix()}`",
                f"- `contact_sheet.png`: `{sheet_path.as_posix()}`",
                f"- Previous primitive visualization backup: `{primitive_backup.as_posix()}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    payload = {
        "global_mp4": global_mp4.as_posix(),
        "contact_sheet": sheet_path.as_posix(),
        "primitive_query_backup": primitive_backup.as_posix(),
        "frames": len(frames),
        "source_scene": SCENE_FRAMES.as_posix(),
        "blender_log": blender_log.as_posix(),
        "rigid_ball_translation_locked": False,
        "rigid_ball_motion": "raw source mesh sequence playback",
        "cloth_projection_clearance": 0.0,
    }
    (DEMO_DIR / "physical_scene_render_manifest.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
