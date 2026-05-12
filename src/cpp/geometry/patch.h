#pragma once

#include <array>
#include <cstdint>
#include <vector>

namespace p2cccd {

struct Patch {
  std::uint32_t patch_id = 0;
  std::vector<std::uint32_t> triangle_ids;
  std::uint32_t triangle_count = 0;
  double area = 0.0;
  std::array<double, 3> local_center{0.0, 0.0, 0.0};
  double radius = 0.0;
};

}  // namespace p2cccd
