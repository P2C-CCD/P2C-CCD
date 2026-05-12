#include "geometry/motion_utils.h"
#include "geometry/proxy.h"

#include <algorithm>
#include <array>
#include <cmath>
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

double Dot(Vec3 lhs, Vec3 rhs) {
  return lhs[0] * rhs[0] + lhs[1] * rhs[1] + lhs[2] * rhs[2];
}

Vec3 Sub(Vec3 lhs, Vec3 rhs) {
  return {lhs[0] - rhs[0], lhs[1] - rhs[1], lhs[2] - rhs[2]};
}

Vec3 Add(Vec3 lhs, Vec3 rhs) {
  return {lhs[0] + rhs[0], lhs[1] + rhs[1], lhs[2] + rhs[2]};
}

Vec3 Scale(Vec3 value, double scale) {
  return {value[0] * scale, value[1] * scale, value[2] * scale};
}

double Norm(Vec3 value) {
  return std::sqrt(Dot(value, value));
}

double DistancePointSegment(Vec3 point, Vec3 segment_a, Vec3 segment_b) {
  const Vec3 ab = Sub(segment_b, segment_a);
  const double length_sq = Dot(ab, ab);
  if (length_sq == 0.0) {
    return Norm(Sub(point, segment_a));
  }
  const double alpha = std::clamp(Dot(Sub(point, segment_a), ab) / length_sq, 0.0, 1.0);
  const Vec3 closest = Add(segment_a, Scale(ab, alpha));
  return Norm(Sub(point, closest));
}

bool AabbContainsSphere(const p2cccd::Aabb& bounds, Vec3 center, double radius) {
  for (std::uint32_t axis = 0; axis < 3; ++axis) {
    if (center[axis] - radius < bounds.min[axis] - 1.0e-10) {
      return false;
    }
    if (center[axis] + radius > bounds.max[axis] + 1.0e-10) {
      return false;
    }
  }
  return true;
}

p2cccd::Patch MakePatch() {
  p2cccd::Patch patch;
  patch.patch_id = 7;
  patch.triangle_ids = {7};
  patch.triangle_count = 1;
  patch.area = 1.0;
  patch.local_center = {1.5, 0.25, 0.0};
  patch.radius = 0.15;
  return patch;
}

p2cccd::MotionSegment MakeRotatingSegment() {
  const double angle = 0.8 * std::acos(-1.0);
  p2cccd::MotionSegment segment;
  segment.t0 = 0.0;
  segment.t1 = 1.0;
  segment.pose_t0.translation = {0.0, 0.0, 0.0};
  segment.pose_t1.translation = {0.4, -0.2, 0.3};
  segment.pose_t0.rotation_xyzw = {0.0, 0.0, 0.0, 1.0};
  segment.pose_t1.rotation_xyzw = {0.0, 0.0, std::sin(0.5 * angle), std::cos(0.5 * angle)};
  return segment;
}

void CheckProxyCoverage(const p2cccd::Patch& patch,
                        const p2cccd::MotionSegment& segment,
                        const p2cccd::TimeSlab& slab) {
  constexpr double kEpsProxy = 1.0e-4;
  p2cccd::EndpointSweptAabbProxy aabb_proxy;
  p2cccd::CapsuleProxy capsule_proxy;
  ExpectOk(p2cccd::BuildEndpointSweptAabbProxy(patch,
                                               segment,
                                               slab,
                                               kEpsProxy,
                                               &aabb_proxy),
           "build endpoint swept AABB proxy");
  ExpectOk(p2cccd::BuildCapsuleProxy(patch, segment, slab, kEpsProxy, &capsule_proxy),
           "build capsule proxy");

  constexpr std::uint32_t kSampleCount = 65;
  for (std::uint32_t i = 0; i < kSampleCount; ++i) {
    const double alpha = static_cast<double>(i) / static_cast<double>(kSampleCount - 1);
    const double t = slab.t0 + alpha * (slab.t1 - slab.t0);
    p2cccd::PoseSample pose;
    Vec3 center{};
    ExpectOk(p2cccd::InterpolateRigidMotion(segment, t, &pose), "sample pose");
    ExpectOk(p2cccd::TransformPoint(pose, patch.local_center, &center),
             "sample patch center");

    Expect(AabbContainsSphere(aabb_proxy.bounds, center, patch.radius),
           "swept AABB covers sampled patch sphere");
    const double center_to_capsule_axis =
        DistancePointSegment(center, capsule_proxy.endpoint0, capsule_proxy.endpoint1);
    Expect(center_to_capsule_axis + patch.radius <= capsule_proxy.radius + 1.0e-10,
           "capsule covers sampled patch sphere");
  }
}

}  // namespace

int main() {
  const p2cccd::Patch patch = MakePatch();
  const p2cccd::MotionSegment segment = MakeRotatingSegment();

  p2cccd::TimeSlab full_slab{0, segment.t0, segment.t1};
  CheckProxyCoverage(patch, segment, full_slab);

  std::vector<p2cccd::TimeSlab> slabs;
  ExpectOk(p2cccd::GenerateTimeSlabsForMotionSegment(segment, 4, &slabs),
           "generate coverage slabs");
  for (const p2cccd::TimeSlab& slab : slabs) {
    CheckProxyCoverage(patch, segment, slab);
  }

  return g_failures == 0 ? 0 : 1;
}
