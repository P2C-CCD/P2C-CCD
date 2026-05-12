#include "certificate/mesh_exact_query.h"
#include "certificate/certificate_engine.h"
#include "geometry/mesh_io.h"

#include <filesystem>
#include <fstream>
#include <iostream>
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

p2cccd::Mesh SingleTriangleMesh() {
  p2cccd::Mesh mesh;
  mesh.vertices_ref = {
      {-0.5, -0.5, 0.0},
      {0.5, -0.5, 0.0},
      {0.0, 0.5, 0.0},
  };
  mesh.triangles = {{0, 1, 2}};
  return mesh;
}

p2cccd::Mesh GridMesh(std::uint32_t resolution) {
  p2cccd::Mesh mesh;
  mesh.vertices_ref.reserve(static_cast<std::size_t>((resolution + 1) * (resolution + 1)));
  mesh.triangles.reserve(static_cast<std::size_t>(resolution * resolution * 2));
  for (std::uint32_t y = 0; y <= resolution; ++y) {
    for (std::uint32_t x = 0; x <= resolution; ++x) {
      mesh.vertices_ref.push_back({static_cast<double>(x),
                                   static_cast<double>(y),
                                   0.0});
    }
  }
  const auto index = [resolution](std::uint32_t x, std::uint32_t y) {
    return y * (resolution + 1) + x;
  };
  for (std::uint32_t y = 0; y < resolution; ++y) {
    for (std::uint32_t x = 0; x < resolution; ++x) {
      mesh.triangles.push_back({index(x, y), index(x + 1, y), index(x, y + 1)});
      mesh.triangles.push_back({index(x + 1, y), index(x + 1, y + 1), index(x, y + 1)});
    }
  }
  return mesh;
}

p2cccd::ExactWorkItem WorkItem() {
  p2cccd::ExactWorkItem item;
  item.work_item_id = 1;
  item.parent_candidate_id = 1;
  item.query_id = 11;
  item.patch_a_id = 1;
  item.patch_b_id = 2;
  item.interval_t0 = 0.0;
  item.interval_t1 = 1.0;
  item.feature_family_mask =
      p2cccd::kFeatureFamilyPointTriangle | p2cccd::kFeatureFamilyEdgeEdge;
  item.priority_score = 1.0F;
  item.source = p2cccd::ProposalSource::kRaw;
  return item;
}

p2cccd::CertificateEngineConfig Config() {
  p2cccd::CertificateEngineConfig config;
  config.eps_time = 1.0e-5;
  config.eps_space = 1.0e-6;
  config.max_subdivision_depth = 32;
  return config;
}

void TestBuildQueryDetectsCollision() {
  const p2cccd::Mesh mesh_a = SingleTriangleMesh();
  const p2cccd::Mesh mesh_b = SingleTriangleMesh();
  p2cccd::MeshExactBuildResult build;
  ExpectOk(p2cccd::BuildMeshExactCertificateQuery(mesh_a,
                                                  {0.0, 0.0, 0.0},
                                                  {0.0, 0.0, 0.0},
                                                  mesh_b,
                                                  {0.0, 0.0, 1.0},
                                                  {0.0, 0.0, -1.0},
                                                  WorkItem(),
                                                  Config(),
                                                  {},
                                                  &build),
           "build mesh exact collision query");
  Expect(build.stats.point_triangle_kept_pairs > 0, "collision PT primitives kept");
  Expect(build.stats.edge_edge_kept_pairs > 0, "collision EE primitives kept");

  p2cccd::CertificateEngine engine;
  p2cccd::CertificateResult certificate;
  ExpectOk(engine.Evaluate(build.query, &certificate), "evaluate mesh exact collision query");
  Expect(certificate.status == p2cccd::CertificateStatus::kCollision,
         "mesh exact collision status");
}

void TestBuildQueryDetectsSeparation() {
  const p2cccd::Mesh mesh_a = SingleTriangleMesh();
  const p2cccd::Mesh mesh_b = SingleTriangleMesh();
  p2cccd::MeshExactBuildResult build;
  ExpectOk(p2cccd::BuildMeshExactCertificateQuery(mesh_a,
                                                  {0.0, 0.0, 0.0},
                                                  {0.0, 0.0, 0.0},
                                                  mesh_b,
                                                  {0.0, 0.0, 2.0},
                                                  {0.0, 0.0, 2.0},
                                                  WorkItem(),
                                                  Config(),
                                                  {},
                                                  &build),
           "build mesh exact separation query");
  Expect(build.stats.point_triangle_kept_pairs == 0, "separation PT pruned");
  Expect(build.stats.edge_edge_kept_pairs == 0, "separation EE pruned");

  p2cccd::CertificateEngine engine;
  p2cccd::CertificateResult certificate;
  ExpectOk(engine.Evaluate(build.query, &certificate), "evaluate mesh exact separation query");
  Expect(certificate.status == p2cccd::CertificateStatus::kUndecided,
         "mesh exact pruned query stays undecided without geometry");
}

void TestLoadTriangleMeshBinaryStlAndCenter() {
  const std::filesystem::path temp_dir =
      std::filesystem::temp_directory_path() / "p2cccd_mesh_exact_query_test";
  std::filesystem::create_directories(temp_dir);
  const std::filesystem::path stl_path = temp_dir / "triangle.stl";

  std::ofstream output(stl_path, std::ios::binary);
  std::array<char, 80> header{};
  output.write(header.data(), static_cast<std::streamsize>(header.size()));
  const std::uint32_t triangle_count = 1;
  output.write(reinterpret_cast<const char*>(&triangle_count), sizeof(triangle_count));
  const float triangle_data[12] = {
      0.0F, 0.0F, 1.0F,
      0.0F, 0.0F, 0.0F,
      1.0F, 0.0F, 0.0F,
      0.0F, 1.0F, 0.0F,
  };
  output.write(reinterpret_cast<const char*>(triangle_data), sizeof(triangle_data));
  const std::uint16_t attribute_bytes = 0;
  output.write(reinterpret_cast<const char*>(&attribute_bytes), sizeof(attribute_bytes));
  output.close();

  p2cccd::Mesh mesh;
  ExpectOk(p2cccd::LoadTriangleMesh(stl_path, &mesh), "load binary STL");
  Expect(mesh.vertices_ref.size() == 3, "binary STL unique vertex count");
  Expect(mesh.triangles.size() == 1, "binary STL triangle count");

  p2cccd::Mesh centered;
  std::array<double, 3> center{};
  ExpectOk(p2cccd::CenterMeshAtAabbCenter(mesh, &centered, &center),
           "center mesh at AABB center");
  Expect(center[0] == 0.5 && center[1] == 0.5 && center[2] == 0.0,
         "binary STL center coordinates");
  Expect(centered.vertices_ref[0][0] == -0.5, "centered STL vertex x");
}

void TestPrimitiveBudgetExceededReturnsError() {
  const p2cccd::Mesh mesh_a = GridMesh(8);
  const p2cccd::Mesh mesh_b = GridMesh(8);
  p2cccd::MeshExactBuildResult build;
  p2cccd::MeshExactBuildConfig build_config;
  build_config.prune_by_swept_aabb = false;
  build_config.max_point_triangle_primitives = 64;
  const p2cccd::Status status = p2cccd::BuildMeshExactCertificateQuery(mesh_a,
                                                                        {0.0, 0.0, 0.0},
                                                                        {0.0, 0.0, 0.0},
                                                                        mesh_b,
                                                                        {0.0, 0.0, 0.1},
                                                                        {0.0, 0.0, 0.1},
                                                                        WorkItem(),
                                                                        Config(),
                                                                        build_config,
                                                                        &build);
  Expect(!status.ok, "primitive budget exceeded status");
  Expect(status.message.find("point-triangle primitive budget exceeded") != std::string::npos,
         "primitive budget exceeded message");
}

}  // namespace

int main() {
  TestBuildQueryDetectsCollision();
  TestBuildQueryDetectsSeparation();
  TestLoadTriangleMeshBinaryStlAndCenter();
  TestPrimitiveBudgetExceededReturnsError();
  if (g_failures != 0) {
    std::cerr << g_failures << " test(s) failed\n";
  }
  return g_failures == 0 ? 0 : 1;
}
