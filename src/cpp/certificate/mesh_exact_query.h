#pragma once

#include "certificate/certificate_engine.h"
#include "common/status.h"
#include "geometry/mesh.h"

#include <array>
#include <cstdint>
#include <vector>

namespace p2cccd {

struct MeshExactBuildConfig {
  bool prune_by_swept_aabb = true;
  std::uint64_t max_point_triangle_primitives = 0;
  std::uint64_t max_edge_edge_primitives = 0;
};

struct MeshExactBuildStats {
  std::uint64_t vertex_count_a = 0;
  std::uint64_t vertex_count_b = 0;
  std::uint64_t triangle_count_a = 0;
  std::uint64_t triangle_count_b = 0;
  std::uint64_t edge_count_a = 0;
  std::uint64_t edge_count_b = 0;
  std::uint64_t point_triangle_total_pairs = 0;
  std::uint64_t point_triangle_kept_pairs = 0;
  std::uint64_t point_triangle_pruned_pairs = 0;
  std::uint64_t edge_edge_total_pairs = 0;
  std::uint64_t edge_edge_kept_pairs = 0;
  std::uint64_t edge_edge_pruned_pairs = 0;
};

struct MeshExactBuildResult {
  ExactCertificateQuery query;
  MeshExactBuildStats stats;
};

Status BuildMeshExactCertificateQuery(
    const Mesh& mesh_a,
    const std::array<double, 3>& translation_a_t0,
    const std::array<double, 3>& translation_a_t1,
    const Mesh& mesh_b,
    const std::array<double, 3>& translation_b_t0,
    const std::array<double, 3>& translation_b_t1,
    const ExactWorkItem& work_item,
    const CertificateEngineConfig& config,
    const MeshExactBuildConfig& build_config,
    MeshExactBuildResult* result);

Status ExtractUniqueMeshEdges(const Mesh& mesh,
                              std::vector<std::array<std::uint32_t, 2>>* edges);

}  // namespace p2cccd
