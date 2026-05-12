#include "geometry/mesh_io.h"
#include "geometry/patch_builder.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <vector>

namespace {

int g_failures = 0;

void Expect(bool condition, const char* label) {
  if (!condition) {
    std::cerr << "FAIL " << label << '\n';
    ++g_failures;
  }
}

void ExpectOk(const p2cccd::Status& status, const char* label) {
  if (!status.ok) {
    std::cerr << "FAIL " << label << ": " << status.message << '\n';
    ++g_failures;
  }
}

void ExpectError(const p2cccd::Status& status, const char* label) {
  if (status.ok) {
    std::cerr << "FAIL " << label << ": expected error\n";
    ++g_failures;
  }
}

bool Near(double lhs, double rhs, double eps = 1.0e-12) {
  return std::abs(lhs - rhs) <= eps;
}

p2cccd::Mesh MakeFourTriangleMesh() {
  p2cccd::Mesh mesh;
  mesh.vertices_ref = {
      {0.0, 0.0, 0.0},
      {1.0, 0.0, 0.0},
      {0.0, 1.0, 0.0},
      {1.0, 1.0, 0.0},
      {10.0, 0.0, 0.0},
      {11.0, 0.0, 0.0},
      {10.0, 1.0, 0.0},
      {11.0, 1.0, 0.0},
  };
  mesh.triangles = {
      {0, 1, 2},
      {1, 3, 2},
      {4, 5, 6},
      {5, 7, 6},
  };
  mesh.patch_ids = {2, 2, 7, 7};
  return mesh;
}

p2cccd::Mesh MakeSameCentroidMesh() {
  p2cccd::Mesh mesh;
  mesh.vertices_ref = {
      {1.0, 0.0, 0.0},
      {0.0, 1.0, 0.0},
      {-1.0, -1.0, 0.0},
      {-1.0, 0.0, 0.0},
      {0.0, -1.0, 0.0},
      {1.0, 1.0, 0.0},
      {0.0, 0.0, 1.0},
      {1.0, 0.0, -1.0},
      {-1.0, 0.0, 0.0},
      {0.0, 0.0, -1.0},
      {-1.0, 0.0, 1.0},
      {1.0, 0.0, 0.0},
  };
  mesh.triangles = {
      {0, 1, 2},
      {3, 4, 5},
      {6, 7, 8},
      {9, 10, 11},
  };
  return mesh;
}

}  // namespace

int main() {
  const std::filesystem::path obj_path =
      std::filesystem::temp_directory_path() / "p2cccd_geometry_loader_test.obj";
  {
    std::ofstream obj(obj_path);
    obj << "v 0 0 0\n";
    obj << "v 1 0 0\n";
    obj << "v 1 1 0\n";
    obj << "v 0 1 0\n";
    obj << "f 1 2 3 4\n";
    obj << "f -4 -3 -2\n";
  }

  p2cccd::Mesh loaded;
  ExpectOk(p2cccd::LoadTriangleMeshObj(obj_path, &loaded), "load OBJ");
  Expect(loaded.vertices_ref.size() == 4, "OBJ vertex count");
  Expect(loaded.triangles.size() == 3, "OBJ triangulated face count");
  Expect(loaded.patch_ids.size() == loaded.triangles.size(), "OBJ patch id count");

  p2cccd::Mesh mesh = MakeFourTriangleMesh();
  ExpectOk(p2cccd::ValidateTriangleMesh(mesh), "validate in-memory mesh");

  std::vector<p2cccd::Patch> rigid_patches;
  ExpectOk(p2cccd::BuildPatchesFromMeshPatchIds(mesh, &rigid_patches), "rigid part patches");
  Expect(rigid_patches.size() == 2, "rigid patch count");
  Expect(rigid_patches[0].patch_id == 2, "rigid patch id order 0");
  Expect(rigid_patches[1].patch_id == 7, "rigid patch id order 1");
  Expect(rigid_patches[0].triangle_count == 2, "rigid patch triangle count");
  Expect(Near(rigid_patches[0].area, 1.0), "rigid patch area");
  Expect(rigid_patches[0].radius > 0.0, "rigid patch radius");

  p2cccd::Patch single;
  single.patch_id = 99;
  single.triangle_ids = {0};
  ExpectOk(p2cccd::ComputePatchStatistics(mesh, &single), "single patch statistics");
  Expect(single.triangle_count == 1, "single patch triangle count");
  Expect(Near(single.area, 0.5), "single patch area");

  std::vector<p2cccd::Patch> bvh_patches;
  p2cccd::BvhPatchBuildOptions options;
  options.max_triangles_per_leaf = 2;
  options.max_depth = 8;
  ExpectOk(p2cccd::BuildPatchesFromBvhLeafClusters(mesh, options, &bvh_patches),
           "BVH leaf patches");
  Expect(bvh_patches.size() == 2, "BVH patch count");

  std::set<std::uint32_t> covered_triangles;
  std::uint32_t covered_count = 0;
  for (const p2cccd::Patch& patch : bvh_patches) {
    Expect(patch.triangle_count <= options.max_triangles_per_leaf, "BVH leaf size");
    covered_count += patch.triangle_count;
    covered_triangles.insert(patch.triangle_ids.begin(), patch.triangle_ids.end());
  }
  Expect(covered_count == mesh.triangles.size(), "BVH covered triangle count");
  Expect(covered_triangles.size() == mesh.triangles.size(), "BVH unique coverage");

  std::filesystem::remove(obj_path);

  const std::filesystem::path degenerate_obj_path =
      std::filesystem::temp_directory_path() / "p2cccd_degenerate_loader_test.obj";
  {
    std::ofstream obj(degenerate_obj_path);
    obj << "v 0 0 0\n";
    obj << "v 1 0 0\n";
    obj << "f 1 1 2\n";
  }
  p2cccd::Mesh bad_loaded;
  ExpectError(p2cccd::LoadTriangleMeshObj(degenerate_obj_path, &bad_loaded),
              "reject degenerate OBJ triangle");
  std::filesystem::remove(degenerate_obj_path);

  const std::filesystem::path zero_area_obj_path =
      std::filesystem::temp_directory_path() / "p2cccd_zero_area_loader_test.obj";
  {
    std::ofstream obj(zero_area_obj_path);
    obj << "v 0 0 0\n";
    obj << "v 1 0 0\n";
    obj << "v 2 0 0\n";
    obj << "f 1 2 3\n";
  }
  ExpectError(p2cccd::LoadTriangleMeshObj(zero_area_obj_path, &bad_loaded),
              "reject zero-area OBJ triangle");
  std::filesystem::remove(zero_area_obj_path);

  p2cccd::Mesh bad_mesh = mesh;
  bad_mesh.triangles[0] = {0, 0, 1};
  ExpectError(p2cccd::ValidateTriangleMesh(bad_mesh), "reject duplicate triangle indices");

  bad_mesh = mesh;
  bad_mesh.vertices_ref[2] = {2.0, 0.0, 0.0};
  ExpectError(p2cccd::ValidateTriangleMesh(bad_mesh), "reject zero-area triangle");

  p2cccd::Mesh bad_patch_ids = mesh;
  bad_patch_ids.patch_ids.pop_back();
  ExpectError(p2cccd::ValidateTriangleMesh(bad_patch_ids), "reject mismatched patch ids");

  p2cccd::Patch duplicate_triangle_patch;
  duplicate_triangle_patch.patch_id = 88;
  duplicate_triangle_patch.triangle_ids = {0, 0};
  ExpectError(p2cccd::ComputePatchStatistics(mesh, &duplicate_triangle_patch),
              "reject duplicate patch triangle ids");

  std::vector<p2cccd::Patch> same_centroid_patches;
  p2cccd::BvhPatchBuildOptions same_centroid_options;
  same_centroid_options.max_triangles_per_leaf = 1;
  same_centroid_options.max_depth = 8;
  ExpectOk(p2cccd::BuildPatchesFromBvhLeafClusters(MakeSameCentroidMesh(),
                                                    same_centroid_options,
                                                    &same_centroid_patches),
           "BVH splits same-centroid triangles");
  Expect(same_centroid_patches.size() == 4, "same-centroid BVH leaf count");
  for (const p2cccd::Patch& patch : same_centroid_patches) {
    Expect(patch.triangle_count == 1, "same-centroid leaf size");
  }

  return g_failures == 0 ? 0 : 1;
}
