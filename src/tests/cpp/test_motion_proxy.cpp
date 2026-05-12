#include "geometry/motion_utils.h"
#include "geometry/proxy.h"

#include <array>
#include <cmath>
#include <iostream>
#include <vector>

namespace {

int g_failures = 0;

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

void ExpectError(const p2cccd::Status& status, const char* label) {
  if (status.ok) {
    std::cerr << "FAIL " << label << ": expected error\n";
    ++g_failures;
  }
}

bool Near(double lhs, double rhs, double eps = 1.0e-12) {
  return std::abs(lhs - rhs) <= eps;
}

bool AabbContainsSphere(const p2cccd::Aabb& aabb,
                        const std::array<double, 3>& center,
                        double radius) {
  for (std::uint32_t axis = 0; axis < 3; ++axis) {
    if (center[axis] - radius < aabb.min[axis] - 1.0e-12) {
      return false;
    }
    if (center[axis] + radius > aabb.max[axis] + 1.0e-12) {
      return false;
    }
  }
  return true;
}

p2cccd::MotionSegment MakeQuarterTurnSegment() {
  const double half_angle = 0.25 * std::acos(-1.0);
  p2cccd::MotionSegment segment;
  segment.t0 = 0.0;
  segment.t1 = 1.0;
  segment.pose_t0.translation = {0.0, 0.0, 0.0};
  segment.pose_t0.rotation_xyzw = {0.0, 0.0, 0.0, 1.0};
  segment.pose_t1.translation = {1.0, 0.0, 0.0};
  segment.pose_t1.rotation_xyzw = {0.0, 0.0, std::sin(half_angle), std::cos(half_angle)};
  return segment;
}

}  // namespace

int main() {
  std::vector<p2cccd::TimeSlab> slabs;
  ExpectOk(p2cccd::GenerateUniformTimeSlabs(0.0, 1.0, 4, &slabs), "generate slabs");
  Expect(slabs.size() == 4, "slab count");
  Expect(Near(slabs.front().t0, 0.0), "slab starts at t0");
  Expect(Near(slabs.back().t1, 1.0), "slab ends at t1");
  for (std::uint32_t i = 0; i < slabs.size(); ++i) {
    Expect(slabs[i].slab_id == i, "slab id");
    Expect(slabs[i].t0 < slabs[i].t1, "positive slab width");
    if (i > 0) {
      Expect(Near(slabs[i - 1].t1, slabs[i].t0), "contiguous slabs");
    }
  }

  p2cccd::MotionSegment segment = MakeQuarterTurnSegment();
  std::vector<p2cccd::TimeSlab> segment_slabs;
  ExpectOk(p2cccd::GenerateTimeSlabsForMotionSegment(segment, 2, &segment_slabs),
           "generate segment slabs");
  Expect(Near(segment_slabs[0].t0, segment.t0), "segment slab starts at segment t0");
  Expect(Near(segment_slabs[1].t1, segment.t1), "segment slab ends at segment t1");

  p2cccd::PoseSample mid_pose;
  ExpectOk(p2cccd::InterpolateRigidMotion(segment, 0.5, &mid_pose), "interpolate midpoint");
  Expect(Near(mid_pose.translation[0], 0.5), "midpoint translation");

  std::array<double, 3> rotated{};
  ExpectOk(p2cccd::TransformPoint(mid_pose, {1.0, 0.0, 0.0}, &rotated), "transform midpoint");
  const double sqrt_half = std::sqrt(0.5);
  Expect(Near(rotated[0], 0.5 + sqrt_half, 1.0e-12), "slerp rotated x");
  Expect(Near(rotated[1], sqrt_half, 1.0e-12), "slerp rotated y");

  p2cccd::Patch patch;
  patch.patch_id = 3;
  patch.local_center = {2.0, 0.0, 0.0};
  patch.radius = 0.25;

  p2cccd::TimeSlab full_slab{0, 0.0, 1.0};
  p2cccd::PatchMotionBound bound;
  ExpectOk(p2cccd::ComputePatchMotionBound(patch, segment, full_slab, &bound),
           "patch motion bound");
  Expect(Near(bound.translation_bound, 1.0), "translation bound");
  Expect(Near(bound.rotation_angle, 0.5 * std::acos(-1.0), 1.0e-12), "rotation angle");
  const double expected_center_bound = 2.0 * 2.0 * std::sin(0.25 * std::acos(-1.0));
  Expect(Near(bound.center_rotation_bound, expected_center_bound, 1.0e-12),
         "center rotation bound");
  Expect(bound.radial_motion_bound >=
             bound.translation_bound + bound.center_rotation_bound + bound.surface_rotation_bound,
         "radial motion bound is conservative");

  const double eps_proxy = 1.0e-3;
  p2cccd::EndpointSweptAabbProxy aabb_proxy;
  ExpectOk(p2cccd::BuildEndpointSweptAabbProxy(patch, segment, full_slab, eps_proxy, &aabb_proxy),
           "endpoint swept AABB");
  ExpectError(p2cccd::BuildEndpointSweptAabbProxy(patch, segment, full_slab, -1.0e-3, &aabb_proxy),
              "reject negative eps_proxy");

  p2cccd::TimeSlab roundoff_slab{9, -5.0e-13, 1.0 + 5.0e-13};
  ExpectOk(p2cccd::ComputePatchMotionBound(patch, segment, roundoff_slab, &bound),
           "accept tiny external slab roundoff");

  p2cccd::TimeSlab outside_slab{10, -1.0e-6, 1.0};
  ExpectError(p2cccd::ComputePatchMotionBound(patch, segment, outside_slab, &bound),
              "reject slab outside tolerance");

  for (double t : {0.0, 0.25, 0.5, 0.75, 1.0}) {
    p2cccd::PoseSample pose;
    std::array<double, 3> center{};
    ExpectOk(p2cccd::InterpolateRigidMotion(segment, t, &pose), "sample pose");
    ExpectOk(p2cccd::TransformPoint(pose, patch.local_center, &center), "sample center");
    Expect(AabbContainsSphere(aabb_proxy.bounds, center, patch.radius),
           "AABB covers sampled rotating patch sphere");
  }

  p2cccd::CapsuleProxy capsule;
  ExpectOk(p2cccd::BuildCapsuleProxy(patch, segment, full_slab, eps_proxy, &capsule),
           "capsule proxy");
  Expect(Near(capsule.radius, bound.conservative_radius + eps_proxy, 1.0e-12),
         "capsule eps inflation");

  const double before_radius = capsule.radius;
  ExpectOk(p2cccd::InflateCapsule(&capsule, eps_proxy), "explicit capsule inflation");
  Expect(Near(capsule.radius, before_radius + eps_proxy, 1.0e-12),
         "explicit capsule inflation amount");

  p2cccd::MotionSegment translation_only;
  translation_only.t0 = 0.0;
  translation_only.t1 = 1.0;
  translation_only.pose_t0.translation = {0.0, 0.0, 0.0};
  translation_only.pose_t1.translation = {2.0, 0.0, 0.0};
  translation_only.pose_t0.rotation_xyzw = {0.0, 0.0, 0.0, 1.0};
  translation_only.pose_t1.rotation_xyzw = {0.0, 0.0, 0.0, 1.0};
  ExpectOk(p2cccd::ComputePatchMotionBound(patch, translation_only, full_slab, &bound),
           "translation-only bound");
  Expect(Near(bound.rotation_angle, 0.0), "translation-only rotation angle");
  Expect(Near(bound.conservative_radius, patch.radius), "translation-only conservative radius");

  return g_failures == 0 ? 0 : 1;
}
