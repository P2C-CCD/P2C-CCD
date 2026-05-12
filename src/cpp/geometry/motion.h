#pragma once

#include <array>
#include <cstdint>
#include <vector>

namespace p2cccd {

struct PoseSample {
  std::array<double, 3> translation{0.0, 0.0, 0.0};
  std::array<double, 4> rotation_xyzw{0.0, 0.0, 0.0, 1.0};
};

struct MotionSegment {
  double t0 = 0.0;
  double t1 = 1.0;
  PoseSample pose_t0;
  PoseSample pose_t1;
};

struct CcdQuery {
  std::uint64_t query_id = 0;
  std::uint32_t object_a_id = 0;
  std::uint32_t object_b_id = 0;
  std::vector<MotionSegment> motion_segments_a;
  std::vector<MotionSegment> motion_segments_b;
};

}  // namespace p2cccd
