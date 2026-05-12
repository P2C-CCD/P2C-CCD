#pragma once

#include "common/runtime_contracts.h"
#include "common/status.h"
#include "geometry/motion.h"
#include "geometry/motion_utils.h"
#include "geometry/patch.h"
#include "geometry/proxy.h"

#include <cstdint>
#include <vector>

namespace p2cccd {

struct ProxyObjectBuildInput {
  std::uint32_t object_id = 0;
  ProxyType proxy_type = ProxyType::kSweptAabb;
  std::vector<Patch> patches;
  std::vector<MotionSegment> motion_segments;
  std::uint32_t slabs_per_motion_segment = 1;
  double eps_proxy = 0.0;
};

struct ProxySceneBuildInput {
  std::uint64_t query_id = 0;
  std::vector<ProxyObjectBuildInput> objects;
};

struct ProxyPrimitive {
  std::uint32_t proxy_id = 0;
  std::uint32_t object_id = 0;
  std::uint32_t patch_id = 0;
  std::uint32_t slab_id = 0;
  std::uint32_t motion_segment_id = 0;
  ProxyType proxy_type = ProxyType::kUnknown;
  double t0 = 0.0;
  double t1 = 1.0;
  Aabb bounds;
  CapsuleProxy capsule;
  PatchMotionBound motion_bound;
};

struct ProxyScene {
  std::uint64_t query_id = 0;
  std::vector<ProxyPrimitive> primitives;
};

Status BuildProxyScene(const ProxySceneBuildInput& input, ProxyScene* scene);
Status ValidateProxyScene(const ProxyScene& scene);

}  // namespace p2cccd
