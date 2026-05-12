#include "geometry/proxy.h"

#include <algorithm>
#include <cmath>
#include <string>

namespace p2cccd {
namespace {

bool IsFinite(double value) {
  return std::isfinite(value);
}

Status ValidateProxyEpsilon(double eps_proxy) {
  if (!IsFinite(eps_proxy) || eps_proxy < 0.0) {
    return Status::Error("eps_proxy must be finite and non-negative");
  }
  return Status::Ok();
}

Status ValidateAabb(const Aabb& aabb) {
  for (std::uint32_t axis = 0; axis < 3; ++axis) {
    if (!IsFinite(aabb.min[axis]) || !IsFinite(aabb.max[axis])) {
      return Status::Error("AABB contains a non-finite bound");
    }
    if (aabb.min[axis] > aabb.max[axis]) {
      return Status::Error("AABB min cannot exceed max");
    }
  }
  return Status::Ok();
}

}  // namespace

Status InflateAabb(Aabb* aabb, double eps_proxy) {
  if (aabb == nullptr) {
    return Status::Error("AABB pointer is null");
  }
  if (auto status = ValidateProxyEpsilon(eps_proxy); !status.ok) {
    return status;
  }
  if (auto status = ValidateAabb(*aabb); !status.ok) {
    return status;
  }

  for (std::uint32_t axis = 0; axis < 3; ++axis) {
    aabb->min[axis] -= eps_proxy;
    aabb->max[axis] += eps_proxy;
  }
  return Status::Ok();
}

Status InflateCapsule(CapsuleProxy* capsule, double eps_proxy) {
  if (capsule == nullptr) {
    return Status::Error("capsule pointer is null");
  }
  if (auto status = ValidateProxyEpsilon(eps_proxy); !status.ok) {
    return status;
  }
  if (!IsFinite(capsule->radius) || capsule->radius < 0.0) {
    return Status::Error("capsule radius must be finite and non-negative");
  }

  capsule->radius += eps_proxy;
  return Status::Ok();
}

Status BuildEndpointSweptAabbProxy(const Patch& patch,
                                   const MotionSegment& segment,
                                   const TimeSlab& slab,
                                   double eps_proxy,
                                   EndpointSweptAabbProxy* proxy) {
  if (proxy == nullptr) {
    return Status::Error("endpoint swept AABB proxy output pointer is null");
  }
  if (auto status = ValidateProxyEpsilon(eps_proxy); !status.ok) {
    return status;
  }

  PatchMotionBound bound;
  if (auto status = ComputePatchMotionBound(patch, segment, slab, &bound); !status.ok) {
    return status;
  }

  EndpointSweptAabbProxy result;
  result.patch_id = patch.patch_id;
  result.motion_bound = bound;

  const double radius = bound.conservative_radius;
  for (std::uint32_t axis = 0; axis < 3; ++axis) {
    result.bounds.min[axis] = std::min(bound.center_t0[axis], bound.center_t1[axis]) - radius;
    result.bounds.max[axis] = std::max(bound.center_t0[axis], bound.center_t1[axis]) + radius;
  }
  if (auto status = InflateAabb(&result.bounds, eps_proxy); !status.ok) {
    return status;
  }

  *proxy = result;
  return Status::Ok();
}

Status BuildCapsuleProxy(const Patch& patch,
                         const MotionSegment& segment,
                         const TimeSlab& slab,
                         double eps_proxy,
                         CapsuleProxy* proxy) {
  if (proxy == nullptr) {
    return Status::Error("capsule proxy output pointer is null");
  }
  if (auto status = ValidateProxyEpsilon(eps_proxy); !status.ok) {
    return status;
  }

  PatchMotionBound bound;
  if (auto status = ComputePatchMotionBound(patch, segment, slab, &bound); !status.ok) {
    return status;
  }

  CapsuleProxy result;
  result.patch_id = patch.patch_id;
  result.endpoint0 = bound.center_t0;
  result.endpoint1 = bound.center_t1;
  result.radius = bound.conservative_radius;
  result.motion_bound = bound;
  if (auto status = InflateCapsule(&result, eps_proxy); !status.ok) {
    return status;
  }

  *proxy = result;
  return Status::Ok();
}

}  // namespace p2cccd
