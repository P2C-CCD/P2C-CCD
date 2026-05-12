#pragma once

#include "common/runtime_contracts.h"
#include "common/status.h"
#include "rt_candidate/proxy_scene.h"

#include <array>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace p2cccd {

enum RawCandidateFlags : std::uint32_t {
  kRawCandidateValid = 1U << 0U,
  kRawCandidateAabbOverlap = 1U << 1U,
};

struct alignas(16) RawCandidateHit {
  std::uint64_t query_id = 0;
  std::uint64_t pair_key = 0;
  std::uint32_t proxy_a_index = 0;
  std::uint32_t proxy_b_index = 0;
  std::uint32_t slab_id = 0;
  std::uint32_t rt_hit_count = 0;
  std::uint32_t flags = 0;
  std::uint32_t reserved0 = 0;
  std::array<float, 4> motion_bound{0.0f, 0.0f, 0.0f, 0.0f};
  std::uint32_t reserved1 = 0;
  std::uint32_t reserved2 = 0;
};

static_assert(sizeof(RawCandidateHit) % 16 == 0,
              "RawCandidateHit must keep 16-byte alignment for GPU writes.");
static_assert(std::is_standard_layout_v<RawCandidateHit>,
              "RawCandidateHit must be standard-layout for GPU writes.");
static_assert(std::is_trivially_copyable_v<RawCandidateHit>,
              "RawCandidateHit must be trivially copyable for GPU writes.");

struct RawCandidateBuffer {
  std::vector<RawCandidateHit> hits;
};

Status GenerateRawCandidatesCpu(const ProxyScene& scene,
                                std::uint64_t query_id,
                                RawCandidateBuffer* raw_buffer);
Status CompactRawCandidates(const ProxyScene& scene,
                            const RawCandidateBuffer& raw_buffer,
                            std::vector<CandidateRecord>* candidates);

}  // namespace p2cccd
