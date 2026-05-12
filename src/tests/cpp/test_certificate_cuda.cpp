#include "certificate/certificate_cuda.h"

#include <array>
#include <cstdint>
#include <iostream>
#include <vector>

namespace {

int g_failures = 0;

using Vec3 = std::array<double, 3>;

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

p2cccd::LinearVertexTrajectory Vertex(std::int64_t id, Vec3 p0, Vec3 p1) {
  p2cccd::LinearVertexTrajectory trajectory;
  trajectory.feature_id = id;
  trajectory.position_t0 = p0;
  trajectory.position_t1 = p1;
  return trajectory;
}

p2cccd::PointTriangleIntervalPrimitive PointTriangle(double z0,
                                                     double z1,
                                                     std::int64_t point_id,
                                                     std::int64_t triangle_id) {
  p2cccd::PointTriangleIntervalPrimitive primitive;
  primitive.point_id = point_id;
  primitive.triangle_id = triangle_id;
  primitive.point = Vertex(point_id, {0.25, 0.25, z0}, {0.25, 0.25, z1});
  primitive.triangle_v0 = Vertex(101, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0});
  primitive.triangle_v1 = Vertex(102, {1.0, 0.0, 0.0}, {1.0, 0.0, 0.0});
  primitive.triangle_v2 = Vertex(103, {0.0, 1.0, 0.0}, {0.0, 1.0, 0.0});
  return primitive;
}

p2cccd::EdgeEdgeIntervalPrimitive EdgeEdge(double z0,
                                           double z1,
                                           std::int64_t edge_a_id,
                                           std::int64_t edge_b_id) {
  p2cccd::EdgeEdgeIntervalPrimitive primitive;
  primitive.edge_a_id = edge_a_id;
  primitive.edge_b_id = edge_b_id;
  primitive.edge_a0 = Vertex(201, {-1.0, 0.0, 0.0}, {-1.0, 0.0, 0.0});
  primitive.edge_a1 = Vertex(202, {1.0, 0.0, 0.0}, {1.0, 0.0, 0.0});
  primitive.edge_b0 = Vertex(301, {0.0, -1.0, z0}, {0.0, -1.0, z1});
  primitive.edge_b1 = Vertex(302, {0.0, 1.0, z0}, {0.0, 1.0, z1});
  return primitive;
}

p2cccd::CertificateEngineConfig Config() {
  p2cccd::CertificateEngineConfig config;
  config.eps_time = 1.0e-5;
  config.eps_space = 1.0e-6;
  config.max_subdivision_depth = 32;
  return config;
}

}  // namespace

int main() {
#if P2CCCD_HAS_CUDA
  Expect(p2cccd::IsCudaExactBuilt(), "CUDA exact backend is built");
  const std::vector<p2cccd::PointTriangleIntervalPrimitive> point_triangles{
      PointTriangle(1.0, -3.0, 10, 20),
      PointTriangle(2.0, 2.0, 11, 21),
  };
  const std::vector<p2cccd::EdgeEdgeIntervalPrimitive> edge_edges{
      EdgeEdge(1.0, -3.0, 30, 40),
      EdgeEdge(2.0, 2.0, 31, 41),
  };
  ExpectOk(p2cccd::CrossCheckCpuCudaExact(point_triangles,
                                          edge_edges,
                                          0.0,
                                          1.0,
                                          Config(),
                                          1.0e-9),
           "CPU CUDA exact cross-check");
#else
  Expect(!p2cccd::IsCudaExactBuilt(), "CUDA exact backend is not built by default");
  std::vector<p2cccd::PrimitiveIntervalResult> results;
  Expect(!p2cccd::EvaluatePointTriangleBatchCuda({}, 0.0, 1.0, Config(), &results).ok,
         "CUDA point-triangle stub rejects calls");
  Expect(!p2cccd::CrossCheckCpuCudaExact({}, {}, 0.0, 1.0, Config(), 1.0e-9).ok,
         "CUDA cross-check stub rejects calls");
#endif
  return g_failures == 0 ? 0 : 1;
}
