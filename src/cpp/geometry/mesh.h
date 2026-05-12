#pragma once

#include <array>
#include <cstdint>
#include <vector>

namespace p2cccd {

struct Mesh {
  std::vector<std::array<double, 3>> vertices_ref;
  std::vector<std::array<std::uint32_t, 3>> triangles;
  std::vector<std::uint32_t> patch_ids;
};

}  // namespace p2cccd
