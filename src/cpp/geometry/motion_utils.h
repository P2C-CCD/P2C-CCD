#pragma once

#include "common/status.h"
#include "geometry/motion.h"
#include "geometry/patch.h"

#include <array>
#include <cstdint>
#include <vector>

namespace p2cccd {

struct TimeSlab {
  std::uint32_t slab_id = 0;
  double t0 = 0.0;
  double t1 = 1.0;
};

struct PatchMotionBound {
  std::uint32_t patch_id = 0;
  double t0 = 0.0;
  double t1 = 1.0;
  std::array<double, 3> center_t0{0.0, 0.0, 0.0};
  std::array<double, 3> center_t1{0.0, 0.0, 0.0};
  double translation_bound = 0.0;
  double rotation_angle = 0.0;
  double center_rotation_bound = 0.0;
  double surface_rotation_bound = 0.0;
  double radial_motion_bound = 0.0;
  double conservative_radius = 0.0;
};

Status ValidateMotionSegment(const MotionSegment& segment);
Status GenerateUniformTimeSlabs(double t0,
                                double t1,
                                std::uint32_t slab_count,
                                std::vector<TimeSlab>* slabs);
Status GenerateTimeSlabsForMotionSegment(const MotionSegment& segment,
                                         std::uint32_t slab_count,
                                         std::vector<TimeSlab>* slabs);

Status InterpolateRigidMotion(const MotionSegment& segment, double t, PoseSample* pose);
Status TransformPoint(const PoseSample& pose,
                      const std::array<double, 3>& local_point,
                      std::array<double, 3>* world_point);
Status ComputeRotationAngle(const PoseSample& pose0, const PoseSample& pose1, double* angle);
Status ComputePatchMotionBound(const Patch& patch,
                               const MotionSegment& segment,
                               const TimeSlab& slab,
                               PatchMotionBound* bound);

}  // namespace p2cccd
