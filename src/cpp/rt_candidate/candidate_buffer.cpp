#include "rt_candidate/candidate_buffer.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <tuple>

namespace p2cccd {
namespace {

using CandidateKey =
    std::tuple<std::uint64_t, std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t,
               std::uint32_t, std::uint8_t, std::uint8_t>;

bool AabbOverlap(const Aabb& lhs, const Aabb& rhs) {
  for (std::uint32_t axis = 0; axis < 3; ++axis) {
    if (lhs.max[axis] < rhs.min[axis] || rhs.max[axis] < lhs.min[axis]) {
      return false;
    }
  }
  return true;
}

bool TimeOverlap(const ProxyPrimitive& lhs, const ProxyPrimitive& rhs) {
  return lhs.t0 < rhs.t1 && rhs.t0 < lhs.t1;
}

std::uint64_t PairKey(std::uint32_t proxy_a_index, std::uint32_t proxy_b_index) {
  const std::uint32_t lo = std::min(proxy_a_index, proxy_b_index);
  const std::uint32_t hi = std::max(proxy_a_index, proxy_b_index);
  return (static_cast<std::uint64_t>(lo) << 32U) | static_cast<std::uint64_t>(hi);
}

std::array<float, 4> MergeMotionBound(const ProxyPrimitive& lhs, const ProxyPrimitive& rhs) {
  return {
      static_cast<float>(std::max(lhs.motion_bound.translation_bound,
                                  rhs.motion_bound.translation_bound)),
      static_cast<float>(std::max(lhs.motion_bound.rotation_angle, rhs.motion_bound.rotation_angle)),
      static_cast<float>(std::max(lhs.motion_bound.radial_motion_bound,
                                  rhs.motion_bound.radial_motion_bound)),
      static_cast<float>(std::max(lhs.motion_bound.conservative_radius,
                                  rhs.motion_bound.conservative_radius)),
  };
}

RawCandidateHit MakeRawHit(const ProxyScene& scene,
                           std::uint64_t query_id,
                           std::uint32_t proxy_a_index,
                           std::uint32_t proxy_b_index) {
  const ProxyPrimitive& proxy_a = scene.primitives[proxy_a_index];
  const ProxyPrimitive& proxy_b = scene.primitives[proxy_b_index];

  RawCandidateHit hit;
  hit.query_id = query_id;
  hit.pair_key = PairKey(proxy_a_index, proxy_b_index);
  hit.proxy_a_index = proxy_a_index;
  hit.proxy_b_index = proxy_b_index;
  hit.slab_id = proxy_a.slab_id;
  hit.rt_hit_count = 1;
  hit.flags = kRawCandidateValid | kRawCandidateAabbOverlap;
  hit.motion_bound = MergeMotionBound(proxy_a, proxy_b);
  return hit;
}

Status ValidateRawHit(const ProxyScene& scene, const RawCandidateHit& hit) {
  if ((hit.flags & kRawCandidateValid) == 0U) {
    return Status::Error("raw candidate hit is not marked valid");
  }
  if (hit.query_id == 0) {
    return Status::Error("raw candidate query_id is required");
  }
  if (hit.query_id != scene.query_id) {
    return Status::Error("raw candidate query_id must match ProxyScene.query_id");
  }
  if (hit.proxy_a_index >= scene.primitives.size() ||
      hit.proxy_b_index >= scene.primitives.size()) {
    return Status::Error("raw candidate proxy index is out of range");
  }
  if (hit.proxy_a_index == hit.proxy_b_index) {
    return Status::Error("raw candidate cannot reference the same proxy twice");
  }
  if (hit.pair_key != PairKey(hit.proxy_a_index, hit.proxy_b_index)) {
    return Status::Error("raw candidate pair_key does not match proxy indices");
  }
  const ProxyPrimitive& proxy_a = scene.primitives[hit.proxy_a_index];
  const ProxyPrimitive& proxy_b = scene.primitives[hit.proxy_b_index];
  if (proxy_a.object_id == proxy_b.object_id) {
    return Status::Error("raw candidate must reference two different objects");
  }
  if (proxy_a.slab_id != proxy_b.slab_id || hit.slab_id != proxy_a.slab_id) {
    return Status::Error("raw candidate slab_id must match both proxy primitives");
  }
  for (float value : hit.motion_bound) {
    if (!std::isfinite(value) || value < 0.0f) {
      return Status::Error("raw candidate motion_bound must be finite and non-negative");
    }
  }
  return Status::Ok();
}

CandidateKey MakeCandidateKey(const ProxyPrimitive& proxy_a,
                              const ProxyPrimitive& proxy_b,
                              const RawCandidateHit& hit) {
  const ProxyPrimitive* first = &proxy_a;
  const ProxyPrimitive* second = &proxy_b;
  if (std::tie(first->object_id, first->patch_id, first->proxy_id) >
      std::tie(second->object_id, second->patch_id, second->proxy_id)) {
    std::swap(first, second);
  }

  return {
      hit.query_id,
      hit.slab_id,
      first->object_id,
      first->patch_id,
      second->object_id,
      second->patch_id,
      static_cast<std::uint8_t>(first->proxy_type),
      static_cast<std::uint8_t>(second->proxy_type),
  };
}

CandidateRecord MakeCandidateRecord(const CandidateKey& key, const RawCandidateHit& hit) {
  CandidateRecord record;
  record.query_id = std::get<0>(key);
  record.slab_id = std::get<1>(key);
  record.object_a_id = std::get<2>(key);
  record.patch_a_id = std::get<3>(key);
  record.object_b_id = std::get<4>(key);
  record.patch_b_id = std::get<5>(key);
  record.proxy_type_a = static_cast<ProxyType>(std::get<6>(key));
  record.proxy_type_b = static_cast<ProxyType>(std::get<7>(key));
  record.rt_hit_count = hit.rt_hit_count;
  record.motion_bound = hit.motion_bound;
  record.flags = hit.flags;
  return record;
}

}  // namespace

Status GenerateRawCandidatesCpu(const ProxyScene& scene,
                                std::uint64_t query_id,
                                RawCandidateBuffer* raw_buffer) {
  if (raw_buffer == nullptr) {
    return Status::Error("raw candidate buffer output pointer is null");
  }
  if (query_id == 0) {
    return Status::Error("query_id is required");
  }
  if (auto status = ValidateProxyScene(scene); !status.ok) {
    return status;
  }
  if (query_id != scene.query_id) {
    return Status::Error("query_id must match ProxyScene.query_id");
  }

  raw_buffer->hits.clear();
  for (std::uint32_t i = 0; i < scene.primitives.size(); ++i) {
    const ProxyPrimitive& lhs = scene.primitives[i];
    for (std::uint32_t j = i + 1; j < scene.primitives.size(); ++j) {
      const ProxyPrimitive& rhs = scene.primitives[j];
      if (lhs.object_id == rhs.object_id) {
        continue;
      }
      if (lhs.slab_id != rhs.slab_id) {
        continue;
      }
      if (!TimeOverlap(lhs, rhs)) {
        continue;
      }
      if (!AabbOverlap(lhs.bounds, rhs.bounds)) {
        continue;
      }
      raw_buffer->hits.push_back(MakeRawHit(scene, query_id, i, j));
    }
  }
  return Status::Ok();
}

Status CompactRawCandidates(const ProxyScene& scene,
                            const RawCandidateBuffer& raw_buffer,
                            std::vector<CandidateRecord>* candidates) {
  if (candidates == nullptr) {
    return Status::Error("candidate output pointer is null");
  }
  if (auto status = ValidateProxyScene(scene); !status.ok) {
    return status;
  }

  std::map<CandidateKey, CandidateRecord> compacted;
  for (const RawCandidateHit& hit : raw_buffer.hits) {
    if (auto status = ValidateRawHit(scene, hit); !status.ok) {
      return status;
    }
    const ProxyPrimitive& proxy_a = scene.primitives[hit.proxy_a_index];
    const ProxyPrimitive& proxy_b = scene.primitives[hit.proxy_b_index];
    const CandidateKey key = MakeCandidateKey(proxy_a, proxy_b, hit);

    auto iter = compacted.find(key);
    if (iter == compacted.end()) {
      compacted.emplace(key, MakeCandidateRecord(key, hit));
    } else {
      CandidateRecord& existing = iter->second;
      if (existing.rt_hit_count >
          std::numeric_limits<std::uint32_t>::max() - hit.rt_hit_count) {
        return Status::Error("candidate rt_hit_count overflow during compaction");
      }
      existing.rt_hit_count += hit.rt_hit_count;
      for (std::uint32_t i = 0; i < existing.motion_bound.size(); ++i) {
        existing.motion_bound[i] = std::max(existing.motion_bound[i], hit.motion_bound[i]);
      }
      existing.flags |= hit.flags;
    }
  }

  candidates->clear();
  candidates->reserve(compacted.size());
  std::uint64_t candidate_id = 1;
  for (auto& [key, record] : compacted) {
    record.candidate_id = candidate_id++;
    candidates->push_back(record);
  }
  return Status::Ok();
}

}  // namespace p2cccd
