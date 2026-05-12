#pragma once

#include "common/runtime_contracts.h"
#include "rt_candidate/candidate_buffer.h"

#include <cstdint>
#include <string>
#include <vector>

namespace p2cccd {

struct RtCandidateTiming {
  double build_ms = 0.0;
  double update_ms = 0.0;
  double trace_ms = 0.0;
  double compact_ms = 0.0;
  double stats_ms = 0.0;
  double total_ms = 0.0;
};

struct CandidateDensityStats {
  std::uint32_t schema_version = 1;
  std::uint64_t query_id = 0;
  std::uint64_t proxy_count = 0;
  std::uint64_t object_count = 0;
  std::uint64_t slab_count = 0;
  std::uint64_t cross_object_same_slab_pair_count = 0;
  std::uint64_t raw_hit_count = 0;
  std::uint64_t compact_candidate_count = 0;
  double raw_hits_per_proxy = 0.0;
  double candidates_per_proxy = 0.0;
  double candidates_per_slab = 0.0;
  double aabb_overlap_ratio = 0.0;
  double avg_rt_hits_per_candidate = 0.0;
  RtCandidateTiming timing;
  std::string backend_name;
};

struct CandidateGenerationResult {
  std::string backend_name;
  RawCandidateBuffer raw_buffer;
  std::vector<CandidateRecord> candidates;
  RtCandidateTiming timing;
  CandidateDensityStats density;
};

}  // namespace p2cccd
