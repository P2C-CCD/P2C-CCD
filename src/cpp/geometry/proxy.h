#pragma once

#include "common/status.h"
#include "geometry/motion_utils.h"
#include "geometry/patch.h"

#include <array>
#include <cstdint>

namespace p2cccd {

struct Aabb {
  std::array<double, 3> min{0.0, 0.0, 0.0};
  std::array<double, 3> max{0.0, 0.0, 0.0};
};

struct EndpointSweptAabbProxy {
  std::uint32_t patch_id = 0;
  Aabb bounds;
  PatchMotionBound motion_bound;
};

struct CapsuleProxy {
  std::uint32_t patch_id = 0;
  std::array<double, 3> endpoint0{0.0, 0.0, 0.0};
  std::array<double, 3> endpoint1{0.0, 0.0, 0.0};
  double radius = 0.0;
  PatchMotionBound motion_bound;
};

Status InflateAabb(Aabb* aabb, double eps_proxy);
Status InflateCapsule(CapsuleProxy* capsule, double eps_proxy);

// Conservative broad-phase proxy over the center chord. It is intentionally not
// a tight swept-sphere representation of curved rotational center paths.
Status BuildEndpointSweptAabbProxy(const Patch& patch,
                                   const MotionSegment& segment,
                                   const TimeSlab& slab,
                                   double eps_proxy,
                                   EndpointSweptAabbProxy* proxy);
Status BuildCapsuleProxy(const Patch& patch,
                         const MotionSegment& segment,
                         const TimeSlab& slab,
                         double eps_proxy,
                         CapsuleProxy* proxy);

}  // namespace p2cccd
