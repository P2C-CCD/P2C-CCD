#include "geometry/patch_builder.h"

#include "geometry/mesh_io.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <vector>

namespace p2cccd {
namespace {

using Vec3 = std::array<double, 3>;

Vec3 Add(Vec3 a, Vec3 b) {
  return {a[0] + b[0], a[1] + b[1], a[2] + b[2]};
}

Vec3 Sub(Vec3 a, Vec3 b) {
  return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}

Vec3 Scale(Vec3 a, double scale) {
  return {a[0] * scale, a[1] * scale, a[2] * scale};
}

Vec3 Cross(Vec3 a, Vec3 b) {
  return {
      a[1] * b[2] - a[2] * b[1],
      a[2] * b[0] - a[0] * b[2],
      a[0] * b[1] - a[1] * b[0],
  };
}

double Dot(Vec3 a, Vec3 b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

double Norm(Vec3 a) {
  return std::sqrt(Dot(a, a));
}

double Distance(Vec3 a, Vec3 b) {
  return Norm(Sub(a, b));
}

Vec3 TriangleCentroid(const Mesh& mesh, std::uint32_t triangle_id) {
  const auto& triangle = mesh.triangles[triangle_id];
  Vec3 centroid = Add(mesh.vertices_ref[triangle[0]], mesh.vertices_ref[triangle[1]]);
  centroid = Add(centroid, mesh.vertices_ref[triangle[2]]);
  return Scale(centroid, 1.0 / 3.0);
}

double TriangleArea(const Mesh& mesh, std::uint32_t triangle_id) {
  const auto& triangle = mesh.triangles[triangle_id];
  const Vec3 a = mesh.vertices_ref[triangle[0]];
  const Vec3 b = mesh.vertices_ref[triangle[1]];
  const Vec3 c = mesh.vertices_ref[triangle[2]];
  return 0.5 * Norm(Cross(Sub(b, a), Sub(c, a)));
}

Status RequirePatchOutput(std::vector<Patch>* patches) {
  if (patches == nullptr) {
    return Status::Error("patch output pointer is null");
  }
  return Status::Ok();
}

Status ValidatePatchTriangleIds(const Mesh& mesh, const Patch& patch) {
  if (patch.triangle_ids.empty()) {
    return Status::Error("patch has no triangles");
  }
  std::set<std::uint32_t> unique_triangle_ids;
  for (std::uint32_t triangle_id : patch.triangle_ids) {
    if (triangle_id >= mesh.triangles.size()) {
      return Status::Error("patch references a triangle outside the mesh triangle array");
    }
    if (!unique_triangle_ids.insert(triangle_id).second) {
      return Status::Error("patch contains duplicate triangle ids");
    }
  }
  return Status::Ok();
}

Status AddPatchFromTriangleIds(const Mesh& mesh,
                               std::uint32_t patch_id,
                               const std::vector<std::uint32_t>& triangle_ids,
                               std::vector<Patch>* patches) {
  Patch patch;
  patch.patch_id = patch_id;
  patch.triangle_ids = triangle_ids;
  if (auto status = ComputePatchStatistics(mesh, &patch); !status.ok) {
    return status;
  }
  patches->push_back(std::move(patch));
  return Status::Ok();
}

std::uint32_t LongestAxis(Vec3 min_corner, Vec3 max_corner) {
  const Vec3 extent = Sub(max_corner, min_corner);
  if (extent[1] > extent[0] && extent[1] >= extent[2]) {
    return 1;
  }
  if (extent[2] > extent[0] && extent[2] > extent[1]) {
    return 2;
  }
  return 0;
}

Status BuildBvhPatchesRecursive(const Mesh& mesh,
                                const BvhPatchBuildOptions& options,
                                std::vector<std::uint32_t> triangle_ids,
                                std::uint32_t depth,
                                std::vector<Patch>* patches) {
  if (triangle_ids.empty()) {
    return Status::Ok();
  }

  Vec3 min_corner{
      std::numeric_limits<double>::infinity(),
      std::numeric_limits<double>::infinity(),
      std::numeric_limits<double>::infinity(),
  };
  Vec3 max_corner{
      -std::numeric_limits<double>::infinity(),
      -std::numeric_limits<double>::infinity(),
      -std::numeric_limits<double>::infinity(),
  };
  for (std::uint32_t triangle_id : triangle_ids) {
    const Vec3 centroid = TriangleCentroid(mesh, triangle_id);
    for (std::uint32_t axis = 0; axis < 3; ++axis) {
      min_corner[axis] = std::min(min_corner[axis], centroid[axis]);
      max_corner[axis] = std::max(max_corner[axis], centroid[axis]);
    }
  }

  const Vec3 extent = Sub(max_corner, min_corner);
  const double max_extent = std::max({extent[0], extent[1], extent[2]});
  if (triangle_ids.size() <= options.max_triangles_per_leaf || depth >= options.max_depth) {
    std::sort(triangle_ids.begin(), triangle_ids.end());
    return AddPatchFromTriangleIds(mesh,
                                   static_cast<std::uint32_t>(patches->size()),
                                   triangle_ids,
                                   patches);
  }

  const std::uint32_t axis = LongestAxis(min_corner, max_corner);
  std::sort(triangle_ids.begin(), triangle_ids.end(), [&](std::uint32_t lhs, std::uint32_t rhs) {
    const Vec3 lhs_centroid = TriangleCentroid(mesh, lhs);
    const Vec3 rhs_centroid = TriangleCentroid(mesh, rhs);
    if (max_extent == 0.0 || lhs_centroid[axis] == rhs_centroid[axis]) {
      return lhs < rhs;
    }
    return lhs_centroid[axis] < rhs_centroid[axis];
  });

  const std::size_t split = triangle_ids.size() / 2;
  std::vector<std::uint32_t> left(triangle_ids.begin(), triangle_ids.begin() + split);
  std::vector<std::uint32_t> right(triangle_ids.begin() + split, triangle_ids.end());
  if (auto status = BuildBvhPatchesRecursive(mesh, options, std::move(left), depth + 1, patches);
      !status.ok) {
    return status;
  }
  return BuildBvhPatchesRecursive(mesh, options, std::move(right), depth + 1, patches);
}

}  // namespace

Status ComputePatchStatistics(const Mesh& mesh, Patch* patch) {
  if (patch == nullptr) {
    return Status::Error("patch pointer is null");
  }
  if (auto status = ValidateTriangleMesh(mesh); !status.ok) {
    return status;
  }
  if (auto status = ValidatePatchTriangleIds(mesh, *patch); !status.ok) {
    return status;
  }

  double total_area = 0.0;
  Vec3 weighted_center{0.0, 0.0, 0.0};
  Vec3 fallback_center{0.0, 0.0, 0.0};

  for (std::uint32_t triangle_id : patch->triangle_ids) {
    const double area = TriangleArea(mesh, triangle_id);
    const Vec3 centroid = TriangleCentroid(mesh, triangle_id);
    total_area += area;
    weighted_center = Add(weighted_center, Scale(centroid, area));
    fallback_center = Add(fallback_center, centroid);
  }

  if (total_area > 0.0) {
    patch->local_center = Scale(weighted_center, 1.0 / total_area);
  } else {
    patch->local_center = Scale(fallback_center, 1.0 / patch->triangle_ids.size());
  }

  double radius = 0.0;
  for (std::uint32_t triangle_id : patch->triangle_ids) {
    const auto& triangle = mesh.triangles[triangle_id];
    radius = std::max(radius, Distance(patch->local_center, mesh.vertices_ref[triangle[0]]));
    radius = std::max(radius, Distance(patch->local_center, mesh.vertices_ref[triangle[1]]));
    radius = std::max(radius, Distance(patch->local_center, mesh.vertices_ref[triangle[2]]));
  }

  patch->triangle_count = static_cast<std::uint32_t>(patch->triangle_ids.size());
  patch->area = total_area;
  patch->radius = radius;
  return Status::Ok();
}

Status ComputePatchStatisticsForAll(const Mesh& mesh, std::vector<Patch>* patches) {
  if (auto status = RequirePatchOutput(patches); !status.ok) {
    return status;
  }
  for (Patch& patch : *patches) {
    if (auto status = ComputePatchStatistics(mesh, &patch); !status.ok) {
      return status;
    }
  }
  return Status::Ok();
}

Status BuildPatchesFromRigidParts(const Mesh& mesh,
                                  const std::vector<std::uint32_t>& triangle_part_ids,
                                  std::vector<Patch>* patches) {
  if (auto status = RequirePatchOutput(patches); !status.ok) {
    return status;
  }
  if (auto status = ValidateTriangleMesh(mesh); !status.ok) {
    return status;
  }
  if (triangle_part_ids.size() != mesh.triangles.size()) {
    return Status::Error("triangle_part_ids must match mesh triangle count");
  }

  std::map<std::uint32_t, std::vector<std::uint32_t>> grouped_triangle_ids;
  for (std::uint32_t triangle_id = 0; triangle_id < triangle_part_ids.size(); ++triangle_id) {
    grouped_triangle_ids[triangle_part_ids[triangle_id]].push_back(triangle_id);
  }

  patches->clear();
  patches->reserve(grouped_triangle_ids.size());
  for (const auto& [part_id, triangle_ids] : grouped_triangle_ids) {
    if (auto status = AddPatchFromTriangleIds(mesh, part_id, triangle_ids, patches);
        !status.ok) {
      return status;
    }
  }
  return Status::Ok();
}

Status BuildPatchesFromMeshPatchIds(const Mesh& mesh, std::vector<Patch>* patches) {
  if (!mesh.patch_ids.empty()) {
    return BuildPatchesFromRigidParts(mesh, mesh.patch_ids, patches);
  }

  std::vector<std::uint32_t> single_part(mesh.triangles.size(), 0);
  return BuildPatchesFromRigidParts(mesh, single_part, patches);
}

Status BuildPatchesFromBvhLeafClusters(const Mesh& mesh,
                                       BvhPatchBuildOptions options,
                                       std::vector<Patch>* patches) {
  if (auto status = RequirePatchOutput(patches); !status.ok) {
    return status;
  }
  if (auto status = ValidateTriangleMesh(mesh); !status.ok) {
    return status;
  }
  if (options.max_triangles_per_leaf == 0) {
    return Status::Error("max_triangles_per_leaf must be positive");
  }
  if (options.max_depth == 0) {
    return Status::Error("max_depth must be positive");
  }

  std::vector<std::uint32_t> triangle_ids(mesh.triangles.size());
  std::iota(triangle_ids.begin(), triangle_ids.end(), 0);
  patches->clear();
  return BuildBvhPatchesRecursive(mesh, options, std::move(triangle_ids), 0, patches);
}

}  // namespace p2cccd
