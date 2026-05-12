#include "geometry/motion_utils.h"

#include <algorithm>
#include <cmath>
#include <string>

namespace p2cccd {
namespace {

using Vec3 = std::array<double, 3>;
using Quat = std::array<double, 4>;

double Dot(Vec3 a, Vec3 b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

double Dot(Quat a, Quat b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
}

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

double Norm(Vec3 a) {
  return std::sqrt(Dot(a, a));
}

double Norm(Quat q) {
  return std::sqrt(Dot(q, q));
}

double Distance(Vec3 a, Vec3 b) {
  return Norm(Sub(a, b));
}

bool IsFinite(double value) {
  return std::isfinite(value);
}

bool IsFinite(Vec3 values) {
  return IsFinite(values[0]) && IsFinite(values[1]) && IsFinite(values[2]);
}

bool IsFinite(Quat values) {
  return IsFinite(values[0]) && IsFinite(values[1]) && IsFinite(values[2]) &&
         IsFinite(values[3]);
}

Status NormalizeQuaternion(Quat input, Quat* output) {
  if (output == nullptr) {
    return Status::Error("quaternion output pointer is null");
  }
  if (!IsFinite(input)) {
    return Status::Error("quaternion contains a non-finite value");
  }
  const double norm = Norm(input);
  if (!IsFinite(norm) || norm <= 0.0) {
    return Status::Error("quaternion norm must be positive and finite");
  }
  *output = {input[0] / norm, input[1] / norm, input[2] / norm, input[3] / norm};
  return Status::Ok();
}

Status ValidatePose(const PoseSample& pose, const char* name) {
  if (!IsFinite(pose.translation)) {
    return Status::Error(std::string(name) + ".translation contains a non-finite value");
  }
  Quat normalized{};
  return NormalizeQuaternion(pose.rotation_xyzw, &normalized);
}

Quat LerpAndNormalize(Quat q0, Quat q1, double alpha) {
  Quat q{
      (1.0 - alpha) * q0[0] + alpha * q1[0],
      (1.0 - alpha) * q0[1] + alpha * q1[1],
      (1.0 - alpha) * q0[2] + alpha * q1[2],
      (1.0 - alpha) * q0[3] + alpha * q1[3],
  };
  const double norm = Norm(q);
  return {q[0] / norm, q[1] / norm, q[2] / norm, q[3] / norm};
}

Quat Slerp(Quat q0, Quat q1, double alpha) {
  double dot = Dot(q0, q1);
  if (dot < 0.0) {
    q1 = {-q1[0], -q1[1], -q1[2], -q1[3]};
    dot = -dot;
  }

  dot = std::clamp(dot, -1.0, 1.0);
  if (dot > 0.9995) {
    return LerpAndNormalize(q0, q1, alpha);
  }

  const double theta0 = std::acos(dot);
  const double theta = theta0 * alpha;
  const double sin_theta = std::sin(theta);
  const double sin_theta0 = std::sin(theta0);
  const double scale0 = std::cos(theta) - dot * sin_theta / sin_theta0;
  const double scale1 = sin_theta / sin_theta0;
  return {
      scale0 * q0[0] + scale1 * q1[0],
      scale0 * q0[1] + scale1 * q1[1],
      scale0 * q0[2] + scale1 * q1[2],
      scale0 * q0[3] + scale1 * q1[3],
  };
}

Vec3 RotateVector(Quat q, Vec3 value) {
  const Vec3 u{q[0], q[1], q[2]};
  const double s = q[3];
  const Vec3 term0 = Scale(u, 2.0 * Dot(u, value));
  const Vec3 term1 = Scale(value, s * s - Dot(u, u));
  const Vec3 term2 = Scale(Cross(u, value), 2.0 * s);
  return Add(Add(term0, term1), term2);
}

Status ValidateTimeSlab(const MotionSegment& segment, const TimeSlab& slab) {
  if (!IsFinite(slab.t0) || !IsFinite(slab.t1) || slab.t0 >= slab.t1) {
    return Status::Error("time slab must satisfy finite t0 < t1");
  }
  const double duration = segment.t1 - segment.t0;
  const double tolerance = 1.0e-12 * std::max(1.0, duration);
  if (slab.t0 < segment.t0 - tolerance || slab.t1 > segment.t1 + tolerance) {
    return Status::Error("time slab must lie inside the motion segment");
  }
  return Status::Ok();
}

double TimeTolerance(const MotionSegment& segment) {
  return 1.0e-12 * std::max(1.0, segment.t1 - segment.t0);
}

Status ValidatePatchForMotionBound(const Patch& patch) {
  if (!IsFinite(patch.local_center)) {
    return Status::Error("patch.local_center contains a non-finite value");
  }
  if (!IsFinite(patch.radius) || patch.radius < 0.0) {
    return Status::Error("patch.radius must be finite and non-negative");
  }
  return Status::Ok();
}

}  // namespace

Status ValidateMotionSegment(const MotionSegment& segment) {
  if (!IsFinite(segment.t0) || !IsFinite(segment.t1) || segment.t0 >= segment.t1) {
    return Status::Error("motion segment must satisfy finite t0 < t1");
  }
  if (auto status = ValidatePose(segment.pose_t0, "pose_t0"); !status.ok) {
    return status;
  }
  return ValidatePose(segment.pose_t1, "pose_t1");
}

Status GenerateUniformTimeSlabs(double t0,
                                double t1,
                                std::uint32_t slab_count,
                                std::vector<TimeSlab>* slabs) {
  if (slabs == nullptr) {
    return Status::Error("slab output pointer is null");
  }
  if (!IsFinite(t0) || !IsFinite(t1) || t0 >= t1) {
    return Status::Error("slab interval must satisfy finite t0 < t1");
  }
  if (slab_count == 0) {
    return Status::Error("slab_count must be positive");
  }

  slabs->clear();
  slabs->reserve(slab_count);
  const double dt = (t1 - t0) / static_cast<double>(slab_count);
  for (std::uint32_t slab_id = 0; slab_id < slab_count; ++slab_id) {
    const double slab_t0 = (slab_id == 0) ? t0 : t0 + dt * slab_id;
    const double slab_t1 = (slab_id + 1 == slab_count) ? t1 : t0 + dt * (slab_id + 1);
    slabs->push_back(TimeSlab{slab_id, slab_t0, slab_t1});
  }
  return Status::Ok();
}

Status GenerateTimeSlabsForMotionSegment(const MotionSegment& segment,
                                         std::uint32_t slab_count,
                                         std::vector<TimeSlab>* slabs) {
  if (auto status = ValidateMotionSegment(segment); !status.ok) {
    return status;
  }
  return GenerateUniformTimeSlabs(segment.t0, segment.t1, slab_count, slabs);
}

Status InterpolateRigidMotion(const MotionSegment& segment, double t, PoseSample* pose) {
  if (pose == nullptr) {
    return Status::Error("pose output pointer is null");
  }
  if (auto status = ValidateMotionSegment(segment); !status.ok) {
    return status;
  }
  const double tolerance = TimeTolerance(segment);
  if (!IsFinite(t) || t < segment.t0 - tolerance || t > segment.t1 + tolerance) {
    return Status::Error("interpolation time must lie inside the motion segment");
  }

  const double clamped_t = std::clamp(t, segment.t0, segment.t1);
  const double alpha = (clamped_t - segment.t0) / (segment.t1 - segment.t0);
  pose->translation = {
      (1.0 - alpha) * segment.pose_t0.translation[0] + alpha * segment.pose_t1.translation[0],
      (1.0 - alpha) * segment.pose_t0.translation[1] + alpha * segment.pose_t1.translation[1],
      (1.0 - alpha) * segment.pose_t0.translation[2] + alpha * segment.pose_t1.translation[2],
  };

  Quat q0{};
  Quat q1{};
  if (auto status = NormalizeQuaternion(segment.pose_t0.rotation_xyzw, &q0); !status.ok) {
    return status;
  }
  if (auto status = NormalizeQuaternion(segment.pose_t1.rotation_xyzw, &q1); !status.ok) {
    return status;
  }
  pose->rotation_xyzw = Slerp(q0, q1, alpha);
  return Status::Ok();
}

Status TransformPoint(const PoseSample& pose,
                      const std::array<double, 3>& local_point,
                      std::array<double, 3>* world_point) {
  if (world_point == nullptr) {
    return Status::Error("world point output pointer is null");
  }
  if (!IsFinite(local_point)) {
    return Status::Error("local point contains a non-finite value");
  }
  if (auto status = ValidatePose(pose, "pose"); !status.ok) {
    return status;
  }
  Quat q{};
  if (auto status = NormalizeQuaternion(pose.rotation_xyzw, &q); !status.ok) {
    return status;
  }
  *world_point = Add(RotateVector(q, local_point), pose.translation);
  return Status::Ok();
}

Status ComputeRotationAngle(const PoseSample& pose0, const PoseSample& pose1, double* angle) {
  if (angle == nullptr) {
    return Status::Error("angle output pointer is null");
  }
  Quat q0{};
  Quat q1{};
  if (auto status = NormalizeQuaternion(pose0.rotation_xyzw, &q0); !status.ok) {
    return status;
  }
  if (auto status = NormalizeQuaternion(pose1.rotation_xyzw, &q1); !status.ok) {
    return status;
  }
  const double dot = std::clamp(std::abs(Dot(q0, q1)), 0.0, 1.0);
  *angle = 2.0 * std::acos(dot);
  return Status::Ok();
}

Status ComputePatchMotionBound(const Patch& patch,
                               const MotionSegment& segment,
                               const TimeSlab& slab,
                               PatchMotionBound* bound) {
  if (bound == nullptr) {
    return Status::Error("motion bound output pointer is null");
  }
  if (auto status = ValidatePatchForMotionBound(patch); !status.ok) {
    return status;
  }
  if (auto status = ValidateMotionSegment(segment); !status.ok) {
    return status;
  }
  if (auto status = ValidateTimeSlab(segment, slab); !status.ok) {
    return status;
  }

  PoseSample pose0;
  PoseSample pose1;
  if (auto status = InterpolateRigidMotion(segment, slab.t0, &pose0); !status.ok) {
    return status;
  }
  if (auto status = InterpolateRigidMotion(segment, slab.t1, &pose1); !status.ok) {
    return status;
  }

  PatchMotionBound result;
  result.patch_id = patch.patch_id;
  result.t0 = slab.t0;
  result.t1 = slab.t1;
  if (auto status = TransformPoint(pose0, patch.local_center, &result.center_t0); !status.ok) {
    return status;
  }
  if (auto status = TransformPoint(pose1, patch.local_center, &result.center_t1); !status.ok) {
    return status;
  }
  if (auto status = ComputeRotationAngle(pose0, pose1, &result.rotation_angle); !status.ok) {
    return status;
  }

  result.translation_bound = Distance(pose0.translation, pose1.translation);
  const double half_angle_sin = std::sin(0.5 * result.rotation_angle);
  result.center_rotation_bound = 2.0 * Norm(patch.local_center) * half_angle_sin;
  result.surface_rotation_bound = 2.0 * patch.radius * half_angle_sin;
  result.radial_motion_bound =
      result.translation_bound + result.center_rotation_bound + result.surface_rotation_bound;
  result.conservative_radius = patch.radius + result.center_rotation_bound;
  *bound = result;
  return Status::Ok();
}

}  // namespace p2cccd
