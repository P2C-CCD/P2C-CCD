#include "certificate/mesh_exact_query.h"

#include "common/validators.h"
#include "geometry/mesh_io.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <set>
#include <utility>
#include <vector>

namespace p2cccd {
namespace {

using Vec3 = std::array<double, 3>;

struct Aabb3 {
  Vec3 min{0.0, 0.0, 0.0};
  Vec3 max{0.0, 0.0, 0.0};
};

Vec3 Add(Vec3 lhs, Vec3 rhs) {
  return {lhs[0] + rhs[0], lhs[1] + rhs[1], lhs[2] + rhs[2]};
}

bool Overlaps(const Aabb3& lhs, const Aabb3& rhs) {
  for (std::size_t axis = 0; axis < 3; ++axis) {
    if (lhs.min[axis] > rhs.max[axis] || rhs.min[axis] > lhs.max[axis]) {
      return false;
    }
  }
  return true;
}

Aabb3 EmptyAabb() {
  return {
      {std::numeric_limits<double>::infinity(),
       std::numeric_limits<double>::infinity(),
       std::numeric_limits<double>::infinity()},
      {-std::numeric_limits<double>::infinity(),
       -std::numeric_limits<double>::infinity(),
       -std::numeric_limits<double>::infinity()},
  };
}

void ExpandWithPoint(Vec3 point, Aabb3* aabb) {
  for (std::size_t axis = 0; axis < 3; ++axis) {
    aabb->min[axis] = std::min(aabb->min[axis], point[axis]);
    aabb->max[axis] = std::max(aabb->max[axis], point[axis]);
  }
}

Aabb3 VertexSweptAabb(const Vec3& p0, const Vec3& p1) {
  Aabb3 aabb = EmptyAabb();
  ExpandWithPoint(p0, &aabb);
  ExpandWithPoint(p1, &aabb);
  return aabb;
}

std::size_t SafePrimitiveReserveHint(std::uint64_t total_pairs,
                                     std::uint64_t explicit_budget) {
  constexpr std::uint64_t kHeuristicReserve = 1ULL << 16;
  std::uint64_t hint = total_pairs;
  if (explicit_budget > 0) {
    hint = std::min(hint, explicit_budget);
  } else {
    hint = std::min(hint, kHeuristicReserve);
  }
  hint = std::min<std::uint64_t>(hint, std::numeric_limits<std::size_t>::max());
  return static_cast<std::size_t>(hint);
}

template <std::size_t N>
Aabb3 PrimitiveSweptAabb(const std::array<Vec3, N>& positions_t0,
                         const std::array<Vec3, N>& positions_t1) {
  Aabb3 aabb = EmptyAabb();
  for (std::size_t i = 0; i < N; ++i) {
    ExpandWithPoint(positions_t0[i], &aabb);
    ExpandWithPoint(positions_t1[i], &aabb);
  }
  return aabb;
}

std::vector<Vec3> TranslatedVertices(const Mesh& mesh, const Vec3& translation) {
  std::vector<Vec3> translated;
  translated.reserve(mesh.vertices_ref.size());
  for (const auto& vertex : mesh.vertices_ref) {
    translated.push_back(Add(vertex, translation));
  }
  return translated;
}

LinearVertexTrajectory MakeTrajectory(std::int64_t feature_id, const Vec3& p0, const Vec3& p1) {
  LinearVertexTrajectory trajectory;
  trajectory.feature_id = feature_id;
  trajectory.position_t0 = p0;
  trajectory.position_t1 = p1;
  return trajectory;
}

Status ValidateBuildConfig(const MeshExactBuildConfig& config) {
  if (config.max_point_triangle_primitives == std::numeric_limits<std::uint64_t>::max()) {
    return Status::Error("max_point_triangle_primitives must be smaller than uint64 max");
  }
  if (config.max_edge_edge_primitives == std::numeric_limits<std::uint64_t>::max()) {
    return Status::Error("max_edge_edge_primitives must be smaller than uint64 max");
  }
  return Status::Ok();
}

}  // namespace

Status ExtractUniqueMeshEdges(const Mesh& mesh, std::vector<std::array<std::uint32_t, 2>>* edges) {
  if (edges == nullptr) {
    return Status::Error("mesh edge output pointer is null");
  }
  if (auto status = ValidateTriangleMesh(mesh); !status.ok) {
    return status;
  }

  std::set<std::pair<std::uint32_t, std::uint32_t>> unique_edges;
  for (const auto& triangle : mesh.triangles) {
    const std::array<std::array<std::uint32_t, 2>, 3> triangle_edges{{
        {std::min(triangle[0], triangle[1]), std::max(triangle[0], triangle[1])},
        {std::min(triangle[1], triangle[2]), std::max(triangle[1], triangle[2])},
        {std::min(triangle[2], triangle[0]), std::max(triangle[2], triangle[0])},
    }};
    for (const auto& edge : triangle_edges) {
      unique_edges.emplace(edge[0], edge[1]);
    }
  }

  edges->clear();
  edges->reserve(unique_edges.size());
  for (const auto& edge : unique_edges) {
    edges->push_back({edge.first, edge.second});
  }
  return Status::Ok();
}

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
    MeshExactBuildResult* result) {
  if (result == nullptr) {
    return Status::Error("mesh exact build result output pointer is null");
  }
  if (auto status = ValidateTriangleMesh(mesh_a); !status.ok) {
    return status;
  }
  if (auto status = ValidateTriangleMesh(mesh_b); !status.ok) {
    return status;
  }
  if (auto status = ValidateExactWorkItem(work_item); !status.ok) {
    return status;
  }
  if (auto status = ValidateBuildConfig(build_config); !status.ok) {
    return status;
  }

  const Vec3 ta0 = translation_a_t0;
  const Vec3 ta1 = translation_a_t1;
  const Vec3 tb0 = translation_b_t0;
  const Vec3 tb1 = translation_b_t1;
  const std::vector<Vec3> vertices_a_t0 = TranslatedVertices(mesh_a, ta0);
  const std::vector<Vec3> vertices_a_t1 = TranslatedVertices(mesh_a, ta1);
  const std::vector<Vec3> vertices_b_t0 = TranslatedVertices(mesh_b, tb0);
  const std::vector<Vec3> vertices_b_t1 = TranslatedVertices(mesh_b, tb1);

  std::vector<std::array<std::uint32_t, 2>> edges_a;
  std::vector<std::array<std::uint32_t, 2>> edges_b;
  if (auto status = ExtractUniqueMeshEdges(mesh_a, &edges_a); !status.ok) {
    return status;
  }
  if (auto status = ExtractUniqueMeshEdges(mesh_b, &edges_b); !status.ok) {
    return status;
  }

  MeshExactBuildResult built;
  built.query.work_item = work_item;
  built.query.config = config;
  built.stats.vertex_count_a = mesh_a.vertices_ref.size();
  built.stats.vertex_count_b = mesh_b.vertices_ref.size();
  built.stats.triangle_count_a = mesh_a.triangles.size();
  built.stats.triangle_count_b = mesh_b.triangles.size();
  built.stats.edge_count_a = edges_a.size();
  built.stats.edge_count_b = edges_b.size();

  const std::uint64_t vertex_offset_b = mesh_a.vertices_ref.size();
  const std::uint64_t triangle_offset_b = mesh_a.triangles.size();
  const std::uint64_t edge_offset_b = edges_a.size();

  if ((work_item.feature_family_mask & kFeatureFamilyPointTriangle) != 0U) {
    built.stats.point_triangle_total_pairs =
        static_cast<std::uint64_t>(mesh_a.vertices_ref.size()) *
            static_cast<std::uint64_t>(mesh_b.triangles.size()) +
        static_cast<std::uint64_t>(mesh_b.vertices_ref.size()) *
            static_cast<std::uint64_t>(mesh_a.triangles.size());
    built.query.point_triangle_primitives.reserve(SafePrimitiveReserveHint(
        built.stats.point_triangle_total_pairs, build_config.max_point_triangle_primitives));

    for (std::size_t vertex_index = 0; vertex_index < mesh_a.vertices_ref.size(); ++vertex_index) {
      const Aabb3 point_aabb = VertexSweptAabb(vertices_a_t0[vertex_index], vertices_a_t1[vertex_index]);
      for (std::size_t triangle_index = 0; triangle_index < mesh_b.triangles.size(); ++triangle_index) {
        const auto& triangle = mesh_b.triangles[triangle_index];
        const std::array<Vec3, 3> triangle_t0{
            vertices_b_t0[triangle[0]], vertices_b_t0[triangle[1]], vertices_b_t0[triangle[2]]};
        const std::array<Vec3, 3> triangle_t1{
            vertices_b_t1[triangle[0]], vertices_b_t1[triangle[1]], vertices_b_t1[triangle[2]]};
        if (build_config.prune_by_swept_aabb &&
            !Overlaps(point_aabb, PrimitiveSweptAabb(triangle_t0, triangle_t1))) {
          ++built.stats.point_triangle_pruned_pairs;
          continue;
        }
        PointTriangleIntervalPrimitive primitive;
        primitive.point_id = static_cast<std::int64_t>(vertex_index);
        primitive.triangle_id = static_cast<std::int64_t>(triangle_offset_b + triangle_index);
        primitive.point = MakeTrajectory(primitive.point_id,
                                         vertices_a_t0[vertex_index],
                                         vertices_a_t1[vertex_index]);
        primitive.triangle_v0 = MakeTrajectory(static_cast<std::int64_t>(triangle_offset_b + triangle[0]),
                                               triangle_t0[0],
                                               triangle_t1[0]);
        primitive.triangle_v1 = MakeTrajectory(static_cast<std::int64_t>(triangle_offset_b + triangle[1]),
                                               triangle_t0[1],
                                               triangle_t1[1]);
        primitive.triangle_v2 = MakeTrajectory(static_cast<std::int64_t>(triangle_offset_b + triangle[2]),
                                               triangle_t0[2],
                                               triangle_t1[2]);
        built.query.point_triangle_primitives.push_back(std::move(primitive));
        ++built.stats.point_triangle_kept_pairs;
      }
    }

    for (std::size_t vertex_index = 0; vertex_index < mesh_b.vertices_ref.size(); ++vertex_index) {
      const Aabb3 point_aabb =
          VertexSweptAabb(vertices_b_t0[vertex_index], vertices_b_t1[vertex_index]);
      for (std::size_t triangle_index = 0; triangle_index < mesh_a.triangles.size(); ++triangle_index) {
        const auto& triangle = mesh_a.triangles[triangle_index];
        const std::array<Vec3, 3> triangle_t0{
            vertices_a_t0[triangle[0]], vertices_a_t0[triangle[1]], vertices_a_t0[triangle[2]]};
        const std::array<Vec3, 3> triangle_t1{
            vertices_a_t1[triangle[0]], vertices_a_t1[triangle[1]], vertices_a_t1[triangle[2]]};
        if (build_config.prune_by_swept_aabb &&
            !Overlaps(point_aabb, PrimitiveSweptAabb(triangle_t0, triangle_t1))) {
          ++built.stats.point_triangle_pruned_pairs;
          continue;
        }
        PointTriangleIntervalPrimitive primitive;
        primitive.point_id = static_cast<std::int64_t>(vertex_offset_b + vertex_index);
        primitive.triangle_id = static_cast<std::int64_t>(triangle_index);
        primitive.point = MakeTrajectory(primitive.point_id,
                                         vertices_b_t0[vertex_index],
                                         vertices_b_t1[vertex_index]);
        primitive.triangle_v0 =
            MakeTrajectory(static_cast<std::int64_t>(triangle[0]), triangle_t0[0], triangle_t1[0]);
        primitive.triangle_v1 =
            MakeTrajectory(static_cast<std::int64_t>(triangle[1]), triangle_t0[1], triangle_t1[1]);
        primitive.triangle_v2 =
            MakeTrajectory(static_cast<std::int64_t>(triangle[2]), triangle_t0[2], triangle_t1[2]);
        built.query.point_triangle_primitives.push_back(std::move(primitive));
        ++built.stats.point_triangle_kept_pairs;
      }
    }

    if (build_config.max_point_triangle_primitives > 0 &&
        built.stats.point_triangle_kept_pairs > build_config.max_point_triangle_primitives) {
      return Status::Error("point-triangle primitive budget exceeded");
    }
  }

  if ((work_item.feature_family_mask & kFeatureFamilyEdgeEdge) != 0U) {
    built.stats.edge_edge_total_pairs =
        static_cast<std::uint64_t>(edges_a.size()) * static_cast<std::uint64_t>(edges_b.size());
    built.query.edge_edge_primitives.reserve(SafePrimitiveReserveHint(
        built.stats.edge_edge_total_pairs, build_config.max_edge_edge_primitives));

    for (std::size_t edge_index_a = 0; edge_index_a < edges_a.size(); ++edge_index_a) {
      const auto& edge_a = edges_a[edge_index_a];
      const std::array<Vec3, 2> edge_a_t0{vertices_a_t0[edge_a[0]], vertices_a_t0[edge_a[1]]};
      const std::array<Vec3, 2> edge_a_t1{vertices_a_t1[edge_a[0]], vertices_a_t1[edge_a[1]]};
      const Aabb3 edge_aabb = PrimitiveSweptAabb(edge_a_t0, edge_a_t1);
      for (std::size_t edge_index_b = 0; edge_index_b < edges_b.size(); ++edge_index_b) {
        const auto& edge_b = edges_b[edge_index_b];
        const std::array<Vec3, 2> edge_b_t0{vertices_b_t0[edge_b[0]], vertices_b_t0[edge_b[1]]};
        const std::array<Vec3, 2> edge_b_t1{vertices_b_t1[edge_b[0]], vertices_b_t1[edge_b[1]]};
        if (build_config.prune_by_swept_aabb &&
            !Overlaps(edge_aabb, PrimitiveSweptAabb(edge_b_t0, edge_b_t1))) {
          ++built.stats.edge_edge_pruned_pairs;
          continue;
        }
        EdgeEdgeIntervalPrimitive primitive;
        primitive.edge_a_id = static_cast<std::int64_t>(edge_index_a);
        primitive.edge_b_id = static_cast<std::int64_t>(edge_offset_b + edge_index_b);
        primitive.edge_a0 =
            MakeTrajectory(static_cast<std::int64_t>(edge_a[0]), edge_a_t0[0], edge_a_t1[0]);
        primitive.edge_a1 =
            MakeTrajectory(static_cast<std::int64_t>(edge_a[1]), edge_a_t0[1], edge_a_t1[1]);
        primitive.edge_b0 = MakeTrajectory(static_cast<std::int64_t>(vertex_offset_b + edge_b[0]),
                                           edge_b_t0[0],
                                           edge_b_t1[0]);
        primitive.edge_b1 = MakeTrajectory(static_cast<std::int64_t>(vertex_offset_b + edge_b[1]),
                                           edge_b_t0[1],
                                           edge_b_t1[1]);
        built.query.edge_edge_primitives.push_back(std::move(primitive));
        ++built.stats.edge_edge_kept_pairs;
      }
    }

    if (build_config.max_edge_edge_primitives > 0 &&
        built.stats.edge_edge_kept_pairs > build_config.max_edge_edge_primitives) {
      return Status::Error("edge-edge primitive budget exceeded");
    }
  }

  *result = std::move(built);
  return Status::Ok();
}

}  // namespace p2cccd
