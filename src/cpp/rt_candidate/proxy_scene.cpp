#include "rt_candidate/proxy_scene.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace p2cccd {
namespace {

bool IsFinite(double value) {
  return std::isfinite(value);
}

bool IsConcreteProxyType(ProxyType proxy_type) {
  return proxy_type == ProxyType::kSweptAabb || proxy_type == ProxyType::kCapsule;
}

Aabb CapsuleBounds(const CapsuleProxy& capsule) {
  Aabb bounds;
  for (std::uint32_t axis = 0; axis < 3; ++axis) {
    bounds.min[axis] = std::min(capsule.endpoint0[axis], capsule.endpoint1[axis]) - capsule.radius;
    bounds.max[axis] = std::max(capsule.endpoint0[axis], capsule.endpoint1[axis]) + capsule.radius;
  }
  return bounds;
}

Status ValidateAabbBounds(const Aabb& bounds) {
  for (std::uint32_t axis = 0; axis < 3; ++axis) {
    if (!IsFinite(bounds.min[axis]) || !IsFinite(bounds.max[axis])) {
      return Status::Error("proxy bounds contain a non-finite value");
    }
    if (bounds.min[axis] > bounds.max[axis]) {
      return Status::Error("proxy bounds min cannot exceed max");
    }
  }
  return Status::Ok();
}

double TimeTolerance(double t0, double t1) {
  return 1.0e-12 * std::max({1.0, std::abs(t0), std::abs(t1), std::abs(t1 - t0)});
}

bool NearlySameTime(double lhs, double rhs, double scale_t0, double scale_t1) {
  const double tolerance = TimeTolerance(scale_t0, scale_t1);
  return std::abs(lhs - rhs) <= tolerance;
}

bool IntervalsOverlap(std::pair<double, double> lhs, std::pair<double, double> rhs) {
  const double tolerance =
      std::max(TimeTolerance(lhs.first, lhs.second), TimeTolerance(rhs.first, rhs.second));
  return lhs.first < rhs.second - tolerance && rhs.first < lhs.second - tolerance;
}

Status ValidateGlobalSlabPartition(const ProxyScene& scene) {
  std::map<std::uint32_t, std::pair<double, double>> slab_intervals;
  for (const ProxyPrimitive& primitive : scene.primitives) {
    const std::pair<double, double> interval{primitive.t0, primitive.t1};
    auto [iter, inserted] = slab_intervals.emplace(primitive.slab_id, interval);
    if (!inserted &&
        (!NearlySameTime(iter->second.first, primitive.t0, iter->second.first, iter->second.second) ||
         !NearlySameTime(iter->second.second, primitive.t1, iter->second.first, iter->second.second))) {
      return Status::Error("proxy scene slab_id must map to one global time interval");
    }
  }

  std::vector<std::pair<std::uint32_t, std::pair<double, double>>> ordered_intervals(
      slab_intervals.begin(), slab_intervals.end());
  std::sort(ordered_intervals.begin(), ordered_intervals.end(), [](const auto& lhs, const auto& rhs) {
    if (lhs.second.first == rhs.second.first) {
      return lhs.first < rhs.first;
    }
    return lhs.second.first < rhs.second.first;
  });

  for (std::size_t i = 1; i < ordered_intervals.size(); ++i) {
    if (IntervalsOverlap(ordered_intervals[i - 1].second, ordered_intervals[i].second)) {
      return Status::Error("proxy scene global slab intervals must not overlap");
    }
  }
  return Status::Ok();
}

Status AppendProxyForPatch(const Patch& patch,
                           const MotionSegment& segment,
                           const TimeSlab& slab,
                           const ProxyObjectBuildInput& object,
                           std::uint32_t motion_segment_id,
                           ProxyScene* scene) {
  ProxyPrimitive primitive;
  primitive.proxy_id = static_cast<std::uint32_t>(scene->primitives.size());
  primitive.object_id = object.object_id;
  primitive.patch_id = patch.patch_id;
  primitive.slab_id = slab.slab_id;
  primitive.motion_segment_id = motion_segment_id;
  primitive.proxy_type = object.proxy_type;
  primitive.t0 = slab.t0;
  primitive.t1 = slab.t1;

  if (object.proxy_type == ProxyType::kSweptAabb) {
    EndpointSweptAabbProxy proxy;
    if (auto status =
            BuildEndpointSweptAabbProxy(patch, segment, slab, object.eps_proxy, &proxy);
        !status.ok) {
      return status;
    }
    primitive.bounds = proxy.bounds;
    primitive.motion_bound = proxy.motion_bound;
  } else if (object.proxy_type == ProxyType::kCapsule) {
    CapsuleProxy proxy;
    if (auto status = BuildCapsuleProxy(patch, segment, slab, object.eps_proxy, &proxy);
        !status.ok) {
      return status;
    }
    primitive.capsule = proxy;
    primitive.bounds = CapsuleBounds(proxy);
    primitive.motion_bound = proxy.motion_bound;
  } else {
    return Status::Error("unsupported proxy type");
  }

  if (auto status = ValidateAabbBounds(primitive.bounds); !status.ok) {
    return status;
  }
  scene->primitives.push_back(std::move(primitive));
  return Status::Ok();
}

}  // namespace

Status ValidateProxyScene(const ProxyScene& scene) {
  if (scene.query_id == 0) {
    return Status::Error("ProxyScene.query_id is required");
  }
  for (std::uint32_t proxy_index = 0; proxy_index < scene.primitives.size(); ++proxy_index) {
    const ProxyPrimitive& primitive = scene.primitives[proxy_index];
    if (primitive.proxy_id != proxy_index) {
      return Status::Error("proxy_id must match primitive array index");
    }
    if (!IsConcreteProxyType(primitive.proxy_type)) {
      return Status::Error("proxy primitive has an unsupported proxy type");
    }
    if (!IsFinite(primitive.t0) || !IsFinite(primitive.t1) || primitive.t0 >= primitive.t1) {
      return Status::Error("proxy primitive must satisfy finite t0 < t1");
    }
    if (auto status = ValidateAabbBounds(primitive.bounds); !status.ok) {
      return status;
    }
  }
  return ValidateGlobalSlabPartition(scene);
}

Status BuildProxyScene(const ProxySceneBuildInput& input, ProxyScene* scene) {
  if (scene == nullptr) {
    return Status::Error("proxy scene output pointer is null");
  }
  if (input.query_id == 0) {
    return Status::Error("ProxySceneBuildInput.query_id is required");
  }
  if (input.objects.empty()) {
    return Status::Error("ProxySceneBuildInput.objects is empty");
  }

  ProxyScene built;
  built.query_id = input.query_id;

  for (const ProxyObjectBuildInput& object : input.objects) {
    if (!IsConcreteProxyType(object.proxy_type)) {
      return Status::Error("ProxyObjectBuildInput.proxy_type must be concrete");
    }
    if (object.patches.empty()) {
      return Status::Error("ProxyObjectBuildInput.patches is empty");
    }
    if (object.motion_segments.empty()) {
      return Status::Error("ProxyObjectBuildInput.motion_segments is empty");
    }
    if (object.slabs_per_motion_segment == 0) {
      return Status::Error("slabs_per_motion_segment must be positive");
    }
    if (!IsFinite(object.eps_proxy) || object.eps_proxy < 0.0) {
      return Status::Error("eps_proxy must be finite and non-negative");
    }

    for (std::uint32_t segment_id = 0; segment_id < object.motion_segments.size(); ++segment_id) {
      const MotionSegment& segment = object.motion_segments[segment_id];
      std::vector<TimeSlab> slabs;
      if (auto status =
              GenerateTimeSlabsForMotionSegment(segment, object.slabs_per_motion_segment, &slabs);
          !status.ok) {
        return status;
      }
      for (const TimeSlab& local_slab : slabs) {
        TimeSlab slab = local_slab;
        slab.slab_id = segment_id * object.slabs_per_motion_segment + local_slab.slab_id;
        for (const Patch& patch : object.patches) {
          if (auto status = AppendProxyForPatch(patch, segment, slab, object, segment_id, &built);
              !status.ok) {
            return status;
          }
        }
      }
    }
  }

  if (auto status = ValidateProxyScene(built); !status.ok) {
    return status;
  }
  *scene = std::move(built);
  return Status::Ok();
}

}  // namespace p2cccd
