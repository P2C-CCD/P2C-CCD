#include "common/group_scheduler.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <string>

namespace p2cccd {
namespace {

using Clock = std::chrono::steady_clock;

double ElapsedMs(const Clock::time_point start, const Clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

bool IsFiniteCandidate(const DenseGroupCandidateInput& candidate) {
  return std::isfinite(candidate.full_exact_cost) &&
         std::isfinite(candidate.narrow_exact_cost) &&
         std::isfinite(candidate.interval_t0) &&
         std::isfinite(candidate.interval_t1) &&
         std::isfinite(candidate.contact_t0) &&
         std::isfinite(candidate.contact_t1) &&
         std::isfinite(candidate.priority_score) &&
         std::isfinite(candidate.cost_score) &&
         std::isfinite(candidate.uncertainty_score);
}

double PositiveCost(const double value, const double fallback) {
  if (!std::isfinite(value) || value <= 0.0) {
    return fallback;
  }
  return value;
}

bool IntervalsOverlap(const double a0, const double a1, const double b0, const double b1) {
  return std::max(a0, b0) <= std::min(a1, b1);
}

bool PriorityBefore(const DenseGroupCandidateInput& lhs,
                    const DenseGroupCandidateInput& rhs,
                    const std::size_t lhs_index,
                    const std::size_t rhs_index) {
  if (lhs.query_id != rhs.query_id) {
    return lhs.query_id < rhs.query_id;
  }
  if (lhs.priority_score != rhs.priority_score) {
    return lhs.priority_score > rhs.priority_score;
  }
  if (lhs.cost_score != rhs.cost_score) {
    return lhs.cost_score < rhs.cost_score;
  }
  return lhs_index < rhs_index;
}

}  // namespace

Status RunDenseGroupExactEarlyStop(
    const std::vector<DenseGroupCandidateInput>& candidates,
    const DenseGroupEarlyStopConfig& config,
    DenseGroupEarlyStopStats* stats) {
  if (stats == nullptr) {
    return Status::Error("DenseGroupEarlyStopStats pointer is null");
  }
  if (config.representative_attempt_limit == 0U) {
    return Status::Error("representative_attempt_limit must be positive");
  }
  if (!std::isfinite(config.uncertainty_fallback_threshold) ||
      config.uncertainty_fallback_threshold < 0.0F) {
    return Status::Error("uncertainty_fallback_threshold must be finite and non-negative");
  }
  if (!std::isfinite(config.interval_miss_penalty_scale) ||
      config.interval_miss_penalty_scale < 0.0) {
    return Status::Error("interval_miss_penalty_scale must be finite and non-negative");
  }

  DenseGroupEarlyStopStats out;
  out.candidate_count = candidates.size();
  const auto total_start = Clock::now();

  for (const DenseGroupCandidateInput& candidate : candidates) {
    if (candidate.query_id == 0U || candidate.candidate_id == 0U) {
      return Status::Error("dense group candidate ids must be non-zero");
    }
    if (!IsFiniteCandidate(candidate)) {
      return Status::Error("dense group candidate contains a non-finite field");
    }
  }

  const auto schedule_start = Clock::now();
  std::vector<std::size_t> order(candidates.size());
  std::iota(order.begin(), order.end(), 0U);
  if (config.preserve_input_order) {
    std::stable_sort(order.begin(), order.end(), [&](const std::size_t lhs, const std::size_t rhs) {
      if (candidates[lhs].query_id != candidates[rhs].query_id) {
        return candidates[lhs].query_id < candidates[rhs].query_id;
      }
      return lhs < rhs;
    });
  } else {
    std::stable_sort(order.begin(), order.end(), [&](const std::size_t lhs, const std::size_t rhs) {
      return PriorityBefore(candidates[lhs], candidates[rhs], lhs, rhs);
    });
  }
  for (std::size_t i = 0; i < order.size(); ++i) {
    if (order[i] != i) {
      ++out.reordered_count;
    }
  }
  const auto schedule_end = Clock::now();
  out.schedule_ms = ElapsedMs(schedule_start, schedule_end);

  const auto exact_start = Clock::now();
  std::size_t group_begin = 0U;
  while (group_begin < order.size()) {
    std::size_t group_end = group_begin + 1U;
    const std::uint64_t query_id = candidates[order[group_begin]].query_id;
    while (group_end < order.size() && candidates[order[group_end]].query_id == query_id) {
      ++group_end;
    }

    ++out.group_count;
    bool truth_collides = false;
    double group_full_work = 0.0;
    for (std::size_t pos = group_begin; pos < group_end; ++pos) {
      const DenseGroupCandidateInput& candidate = candidates[order[pos]];
      truth_collides = truth_collides || candidate.candidate_collides;
      group_full_work += PositiveCost(candidate.full_exact_cost, 1.0);
    }
    if (truth_collides) {
      ++out.positive_group_count;
    }
    out.no_proposal_exact_calls += static_cast<std::uint64_t>(group_end - group_begin);
    out.no_proposal_exact_work += group_full_work;

    bool resolved = false;
    bool predicted_collision = false;
    std::uint32_t attempts = 0U;
    bool group_used_uncertainty_fallback = false;
    double work_before_first_positive = 0.0;

    auto run_full_fallback = [&](const std::size_t start_pos) {
      for (std::size_t fallback_pos = start_pos; fallback_pos < group_end; ++fallback_pos) {
        const DenseGroupCandidateInput& fallback_candidate = candidates[order[fallback_pos]];
        ++out.learned_exact_calls;
        ++out.learned_fallback_calls;
        const double full_cost = PositiveCost(fallback_candidate.full_exact_cost, 1.0);
        out.learned_exact_work += full_cost;
        if (fallback_candidate.candidate_collides) {
          predicted_collision = true;
          resolved = true;
          out.first_positive_rank_sum += static_cast<double>(fallback_pos - group_begin + 1U);
          out.cost_before_first_positive_sum += work_before_first_positive + full_cost;
          return;
        }
        work_before_first_positive += full_cost;
      }
      predicted_collision = false;
      resolved = true;
    };

    for (std::size_t pos = group_begin; pos < group_end && !resolved; ++pos) {
      const DenseGroupCandidateInput& candidate = candidates[order[pos]];
      ++attempts;
      const double full_cost = PositiveCost(candidate.full_exact_cost, 1.0);
      const double narrow_cost = PositiveCost(candidate.narrow_exact_cost, full_cost);
      const bool high_uncertainty =
          candidate.uncertainty_score >= config.uncertainty_fallback_threshold;

      if (high_uncertainty) {
        run_full_fallback(pos);
        group_used_uncertainty_fallback = true;
        break;
      }

      ++out.learned_exact_calls;
      out.learned_exact_work += narrow_cost;
      if (candidate.candidate_collides) {
        if (IntervalsOverlap(candidate.interval_t0,
                             candidate.interval_t1,
                             candidate.contact_t0,
                             candidate.contact_t1)) {
          ++out.learned_interval_hit_count;
          predicted_collision = true;
          resolved = true;
          out.first_positive_rank_sum += static_cast<double>(pos - group_begin + 1U);
          out.cost_before_first_positive_sum += work_before_first_positive + narrow_cost;
          break;
        }

        ++out.learned_interval_miss_count;
        out.learned_exact_work += full_cost * config.interval_miss_penalty_scale;
        if (attempts >= config.representative_attempt_limit) {
          ++out.learned_exact_calls;
          ++out.learned_fallback_calls;
          out.learned_exact_work += full_cost;
          predicted_collision = true;
          resolved = true;
          out.first_positive_rank_sum += static_cast<double>(pos - group_begin + 1U);
          out.cost_before_first_positive_sum +=
              work_before_first_positive + narrow_cost + full_cost;
          break;
        }
        work_before_first_positive += narrow_cost + (full_cost * config.interval_miss_penalty_scale);
        continue;
      }

      work_before_first_positive += narrow_cost;
    }

    if (!resolved) {
      predicted_collision = false;
      resolved = true;
    }
    if (group_used_uncertainty_fallback) {
      ++out.high_uncertainty_group_count;
    }

    if (truth_collides && predicted_collision) {
      ++out.tp;
    } else if (!truth_collides && !predicted_collision) {
      ++out.tn;
    } else if (!truth_collides && predicted_collision) {
      ++out.fp;
    } else {
      ++out.fn;
    }

    group_begin = group_end;
  }
  const auto exact_end = Clock::now();
  out.exact_ms = ElapsedMs(exact_start, exact_end);
  out.total_ms = ElapsedMs(total_start, exact_end);

  *stats = out;
  return Status::Ok();
}

}  // namespace p2cccd
