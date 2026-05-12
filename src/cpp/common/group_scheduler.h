#pragma once

#include "common/status.h"

#include <cstdint>
#include <vector>

namespace p2cccd {

struct DenseGroupCandidateInput {
  std::uint64_t query_id = 0;
  std::uint64_t candidate_id = 0;
  double full_exact_cost = 1.0;
  double narrow_exact_cost = 1.0;
  double interval_t0 = 0.0;
  double interval_t1 = 1.0;
  double contact_t0 = 0.0;
  double contact_t1 = 0.0;
  float priority_score = 0.0F;
  float cost_score = 0.0F;
  float uncertainty_score = 0.0F;
  bool candidate_collides = false;
};

struct DenseGroupEarlyStopConfig {
  float uncertainty_fallback_threshold = 0.75F;
  std::uint32_t representative_attempt_limit = 3U;
  double interval_miss_penalty_scale = 0.22;
  bool preserve_input_order = false;
};

struct DenseGroupEarlyStopStats {
  std::uint64_t candidate_count = 0;
  std::uint64_t group_count = 0;
  std::uint64_t positive_group_count = 0;

  std::uint64_t no_proposal_exact_calls = 0;
  double no_proposal_exact_work = 0.0;

  std::uint64_t learned_exact_calls = 0;
  std::uint64_t learned_fallback_calls = 0;
  std::uint64_t learned_interval_hit_count = 0;
  std::uint64_t learned_interval_miss_count = 0;
  double learned_exact_work = 0.0;

  double first_positive_rank_sum = 0.0;
  double cost_before_first_positive_sum = 0.0;

  std::uint64_t tp = 0;
  std::uint64_t tn = 0;
  std::uint64_t fp = 0;
  std::uint64_t fn = 0;

  std::uint64_t reordered_count = 0;
  std::uint64_t high_uncertainty_group_count = 0;

  double schedule_ms = 0.0;
  double exact_ms = 0.0;
  double total_ms = 0.0;
};

Status RunDenseGroupExactEarlyStop(
    const std::vector<DenseGroupCandidateInput>& candidates,
    const DenseGroupEarlyStopConfig& config,
    DenseGroupEarlyStopStats* stats);

}  // namespace p2cccd
