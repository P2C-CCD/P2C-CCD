#pragma once

#include "common/status.h"
#include "geometry/mesh.h"
#include "geometry/patch.h"

#include <cstdint>
#include <vector>

namespace p2cccd {

struct BvhPatchBuildOptions {
  std::uint32_t max_triangles_per_leaf = 32;
  std::uint32_t max_depth = 32;
};

Status ComputePatchStatistics(const Mesh& mesh, Patch* patch);
Status ComputePatchStatisticsForAll(const Mesh& mesh, std::vector<Patch>* patches);

Status BuildPatchesFromRigidParts(const Mesh& mesh,
                                  const std::vector<std::uint32_t>& triangle_part_ids,
                                  std::vector<Patch>* patches);
Status BuildPatchesFromMeshPatchIds(const Mesh& mesh, std::vector<Patch>* patches);
Status BuildPatchesFromBvhLeafClusters(const Mesh& mesh,
                                       BvhPatchBuildOptions options,
                                       std::vector<Patch>* patches);

}  // namespace p2cccd
