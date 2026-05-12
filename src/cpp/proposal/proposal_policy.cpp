#include "proposal/proposal_policy.h"

#include "common/validators.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <utility>

namespace p2cccd {
namespace {

constexpr std::uint32_t kRuntimeConservativeFeatureFamilyMask =
    kFeatureFamilyPointTriangle | kFeatureFamilyEdgeEdge;

struct RuntimeDensityDerivedStats {
  double candidates_per_proxy = 0.0;
  double aabb_overlap_ratio = 0.0;
  double avg_hits = 0.0;
};

Status BuildRuntimeFeatureRowFromDerived(const CandidateRecord& candidate,
                                         const RuntimeDensityDerivedStats& derived,
                                         std::uint32_t base_family_mask,
                                         ProposalFeatureRow* row);

float ClampRuntimeFeature(double value) {
  if (!std::isfinite(value)) {
    return 0.0F;
  }
  return static_cast<float>(std::clamp(value, -1.0e6, 1.0e6));
}

double SafeRuntimeRatio(std::uint64_t numerator, std::uint64_t denominator) {
  if (denominator == 0U) {
    return 0.0;
  }
  return static_cast<double>(numerator) / static_cast<double>(denominator);
}

RuntimeDensityDerivedStats MakeRuntimeDensityDerivedStats(const CandidateDensityStats& density) {
  const double candidates_per_proxy =
      SafeRuntimeRatio(density.compact_candidate_count, density.proxy_count);
  const std::uint64_t overlap_denominator =
      density.proxy_count >= 2U ? density.proxy_count / 2U : 1U;
  const double aabb_overlap_ratio =
      SafeRuntimeRatio(density.raw_hit_count, overlap_denominator);
  const double avg_hits =
      SafeRuntimeRatio(density.raw_hit_count, density.compact_candidate_count);
  return RuntimeDensityDerivedStats{
      .candidates_per_proxy = candidates_per_proxy,
      .aabb_overlap_ratio = aabb_overlap_ratio,
      .avg_hits = avg_hits,
  };
}

bool CandidateMatchesPrimitive(const CandidateRecord& candidate,
                               const ProxyPrimitive& primitive,
                               const bool first_side) {
  if (primitive.slab_id != candidate.slab_id) {
    return false;
  }
  if (first_side) {
    return primitive.object_id == candidate.object_a_id &&
           primitive.patch_id == candidate.patch_a_id &&
           primitive.proxy_type == candidate.proxy_type_a;
  }
  return primitive.object_id == candidate.object_b_id &&
         primitive.patch_id == candidate.patch_b_id &&
         primitive.proxy_type == candidate.proxy_type_b;
}

const ProxyPrimitive* FindCandidatePrimitive(const ProxyScene& scene,
                                             const CandidateRecord& candidate,
                                             const bool first_side) {
  for (const ProxyPrimitive& primitive : scene.primitives) {
    if (CandidateMatchesPrimitive(candidate, primitive, first_side)) {
      return &primitive;
    }
  }
  return nullptr;
}

bool IsValidSchedulingConfig(const ProposalSchedulingConfig& config, std::string* message) {
  if (config.first_work_item_id == 0) {
    *message = "ProposalSchedulingConfig.first_work_item_id must be non-zero";
    return false;
  }
  if (config.conservative_feature_family_mask == 0) {
    *message = "ProposalSchedulingConfig.conservative_feature_family_mask must be non-zero";
    return false;
  }
  if ((config.conservative_feature_family_mask & ~kRuntimeConservativeFeatureFamilyMask) != 0) {
    *message = "ProposalSchedulingConfig.conservative_feature_family_mask has unknown bits";
    return false;
  }
  if (!std::isfinite(config.fallback_interval_t0) ||
      !std::isfinite(config.fallback_interval_t1) || config.fallback_interval_t0 < 0.0 ||
      config.fallback_interval_t1 > 1.0 ||
      config.fallback_interval_t0 > config.fallback_interval_t1) {
    *message = "ProposalSchedulingConfig fallback interval must lie in [0, 1]";
    return false;
  }
  if (!std::isfinite(config.family_score_threshold)) {
    *message = "ProposalSchedulingConfig.family_score_threshold must be finite";
    return false;
  }
  if (!std::isfinite(config.uncertainty_fallback_threshold) ||
      config.uncertainty_fallback_threshold < 0.0F) {
    *message = "ProposalSchedulingConfig.uncertainty_fallback_threshold must be non-negative";
    return false;
  }
  if (!std::isfinite(config.ood_abs_feature_threshold) ||
      config.ood_abs_feature_threshold <= 0.0F) {
    *message = "ProposalSchedulingConfig.ood_abs_feature_threshold must be positive";
    return false;
  }
  return true;
}

std::uint32_t BaseFamilyMaskForRuntimeQuery(
    const std::unordered_map<std::uint64_t, std::uint32_t>& conservative_family_masks_by_query_id,
    const std::uint64_t query_id,
    const ProposalSchedulingConfig& config) {
  const auto mask_it = conservative_family_masks_by_query_id.find(query_id);
  if (mask_it != conservative_family_masks_by_query_id.end() && mask_it->second != 0U) {
    return mask_it->second;
  }
  return config.conservative_feature_family_mask;
}

std::uint32_t BaseFamilyMaskForRuntimeQuery(
    const std::unordered_map<std::uint64_t, std::uint32_t>& conservative_family_masks_by_query_id,
    const std::uint64_t query_id) {
  const auto mask_it = conservative_family_masks_by_query_id.find(query_id);
  if (mask_it != conservative_family_masks_by_query_id.end() && mask_it->second != 0U) {
    return mask_it->second;
  }
  return kRuntimeConservativeFeatureFamilyMask;
}

std::array<float, kProposalFamilyCount> RuntimeFamilyTargets(const std::uint32_t base_family_mask) {
  std::array<float, kProposalFamilyCount> family_targets{};
  if ((base_family_mask & kFeatureFamilyPointTriangle) != 0U) {
    family_targets[0] = 1.0F;
  }
  if ((base_family_mask & kFeatureFamilyEdgeEdge) != 0U) {
    family_targets[1] = 1.0F;
  }
  return family_targets;
}

std::array<float, kProposalFamilyCount> NormalizedDummyFamilyScores(
    const std::uint32_t base_family_mask) {
  std::array<float, kProposalFamilyCount> scores{};
  std::vector<std::size_t> active_indices;
  if ((base_family_mask & kFeatureFamilyPointTriangle) != 0U) {
    active_indices.push_back(0U);
  }
  if ((base_family_mask & kFeatureFamilyEdgeEdge) != 0U) {
    active_indices.push_back(1U);
  }
  if (active_indices.empty()) {
    active_indices = {0U, 1U};
  }
  const float weight = 1.0F / static_cast<float>(active_indices.size());
  for (const std::size_t index : active_indices) {
    scores[index] = weight;
  }
  return scores;
}

Status BuildRuntimeFeatureRow(const CandidateRecord& candidate,
                              const CandidateDensityStats& density,
                              const std::uint32_t base_family_mask,
                              ProposalFeatureRow* row) {
  const RuntimeDensityDerivedStats derived = MakeRuntimeDensityDerivedStats(density);
  return BuildRuntimeFeatureRowFromDerived(candidate, derived, base_family_mask, row);
}

Status BuildRuntimeFeatureRowFromDerived(const CandidateRecord& candidate,
                                         const RuntimeDensityDerivedStats& derived,
                                         const std::uint32_t base_family_mask,
                                         ProposalFeatureRow* row) {
  if (row == nullptr) {
    return Status::Error("proposal runtime feature row pointer is null");
  }
  if (auto status = ValidateCandidateRecord(candidate); !status.ok) {
    return status;
  }

  ProposalFeatureRow built;
  built.query_id = candidate.query_id;
  built.candidate_id = candidate.candidate_id;
  built.slab_id = candidate.slab_id;
  built.object_a_id = candidate.object_a_id;
  built.patch_a_id = candidate.patch_a_id;
  built.object_b_id = candidate.object_b_id;
  built.patch_b_id = candidate.patch_b_id;
  built.features[0] = 0.0F;
  built.features[1] = 1.0F;
  built.features[2] = 1.0F;
  built.features[3] = ClampRuntimeFeature(candidate.rt_hit_count);
  built.features[4] = ClampRuntimeFeature(static_cast<int>(candidate.proxy_type_a));
  built.features[5] = ClampRuntimeFeature(static_cast<int>(candidate.proxy_type_b));
  for (std::size_t i = 0; i < candidate.motion_bound.size(); ++i) {
    built.features[6 + i] = ClampRuntimeFeature(candidate.motion_bound[i]);
  }
  built.features[19] = ClampRuntimeFeature(std::min(1.0, derived.aabb_overlap_ratio));
  built.features[29] = ClampRuntimeFeature(derived.candidates_per_proxy);
  built.features[30] = ClampRuntimeFeature(derived.aabb_overlap_ratio);
  built.features[31] = ClampRuntimeFeature(derived.avg_hits);
  built.interval_targets.fill(0.0F);
  built.interval_targets[0] = 1.0F;
  built.family_targets = RuntimeFamilyTargets(base_family_mask);
  built.priority_target = ClampRuntimeFeature(
      std::min(1.0, 0.5 * static_cast<double>(candidate.rt_hit_count) + 0.5 * derived.candidates_per_proxy));
  built.cost_target =
      ClampRuntimeFeature(std::max(1.0, 1.0 + static_cast<double>(candidate.rt_hit_count)));
  built.uncertainty_target =
      base_family_mask == kRuntimeConservativeFeatureFamilyMask ? 0.25F : 0.0F;
  built.target_mask = (1U << 0U) | (1U << 1U) | (1U << 2U) | (1U << 3U) | (1U << 4U);

  *row = built;
  return Status::Ok();
}

float RuntimePriorityTarget(const CandidateRecord& candidate,
                            const RuntimeDensityDerivedStats& derived) {
  return ClampRuntimeFeature(
      std::min(1.0, 0.5 * static_cast<double>(candidate.rt_hit_count) + 0.5 * derived.candidates_per_proxy));
}

float RuntimeCostTarget(const CandidateRecord& candidate) {
  return ClampRuntimeFeature(std::max(1.0, 1.0 + static_cast<double>(candidate.rt_hit_count)));
}

float RuntimeUncertaintyTarget(const std::uint32_t base_family_mask) {
  return base_family_mask == kRuntimeConservativeFeatureFamilyMask ? 0.25F : 0.0F;
}

bool IsRuntimeCandidateOod(const CandidateRecord& candidate,
                           const RuntimeDensityDerivedStats& derived,
                           const ProposalSchedulingConfig& config) {
  const double capped_overlap_ratio = std::min(1.0, derived.aabb_overlap_ratio);
  const double feature_scalars[] = {
      1.0,
      static_cast<double>(candidate.rt_hit_count),
      static_cast<double>(static_cast<int>(candidate.proxy_type_a)),
      static_cast<double>(static_cast<int>(candidate.proxy_type_b)),
      capped_overlap_ratio,
      derived.candidates_per_proxy,
      derived.aabb_overlap_ratio,
      derived.avg_hits,
  };
  for (const double value : feature_scalars) {
    if (!std::isfinite(value) || std::abs(value) > config.ood_abs_feature_threshold) {
      return true;
    }
  }
  for (const float value : candidate.motion_bound) {
    if (!std::isfinite(value) || std::abs(value) > config.ood_abs_feature_threshold) {
      return true;
    }
  }
  return false;
}

bool NeedsPrioritySort(const std::vector<ExactWorkItem>& work_queue) {
  if (work_queue.size() < 2U) {
    return false;
  }
  for (std::size_t i = 1; i < work_queue.size(); ++i) {
    if (work_queue[i - 1].priority_score < work_queue[i].priority_score) {
      return true;
    }
  }
  return false;
}

std::uint32_t FamilyMaskFromScores(const ProposalOutput& output,
                                   const ProposalSchedulingConfig& config) {
  std::uint32_t mask = 0;
  if (output.family_scores[0] >= config.family_score_threshold) {
    mask |= kFeatureFamilyPointTriangle;
  }
  if (output.family_scores[1] >= config.family_score_threshold) {
    mask |= kFeatureFamilyEdgeEdge;
  }
  return mask | config.conservative_feature_family_mask;
}

std::uint32_t PredictedFamilyMaskFromScores(const ProposalOutput& output,
                                            const ProposalSchedulingConfig& config) {
  std::uint32_t mask = 0;
  if (output.family_scores[0] >= config.family_score_threshold) {
    mask |= kFeatureFamilyPointTriangle;
  }
  if (output.family_scores[1] >= config.family_score_threshold) {
    mask |= kFeatureFamilyEdgeEdge;
  }
  return mask;
}

double SlabIntervalT0(const ProxyScene& scene,
                      const CandidateRecord& candidate,
                      const ProposalSchedulingConfig& config) {
  const ProxyPrimitive* primitive = FindCandidatePrimitive(scene, candidate, true);
  return primitive != nullptr ? primitive->t0 : config.fallback_interval_t0;
}

double SlabIntervalT1(const ProxyScene& scene,
                      const CandidateRecord& candidate,
                      const ProposalSchedulingConfig& config) {
  const ProxyPrimitive* primitive = FindCandidatePrimitive(scene, candidate, true);
  return primitive != nullptr ? primitive->t1 : config.fallback_interval_t1;
}

bool HasNonFiniteProposal(const ProposalOutput& output) {
  for (float value : output.interval_scores) {
    if (!std::isfinite(value)) {
      return true;
    }
  }
  for (float value : output.family_scores) {
    if (!std::isfinite(value)) {
      return true;
    }
  }
  return !std::isfinite(output.priority_score) || !std::isfinite(output.cost_score) ||
         !std::isfinite(output.uncertainty_score);
}

struct ScheduleInput {
  CandidateRecord candidate;
  const ProposalFeatureRow* row = nullptr;
  const ProposalOutput* output = nullptr;
  bool missing_proposal = false;
  bool invalid_proposal = false;
  bool ood = false;
  bool high_uncertainty = false;
};

bool ShouldFallback(const ScheduleInput& input) {
  return input.missing_proposal || input.invalid_proposal || input.ood || input.high_uncertainty;
}

ExactWorkItem MakeWorkItem(const ProxyScene& scene,
                           const ScheduleInput& input,
                           const ProposalSchedulingConfig& config) {
  ExactWorkItem item;
  item.parent_candidate_id = input.candidate.candidate_id;
  item.query_id = input.candidate.query_id;
  item.slab_id = input.candidate.slab_id;
  item.patch_a_id = input.candidate.patch_a_id;
  item.patch_b_id = input.candidate.patch_b_id;
  item.interval_t0 = SlabIntervalT0(scene, input.candidate, config);
  item.interval_t1 = SlabIntervalT1(scene, input.candidate, config);
  item.feature_family_mask = config.conservative_feature_family_mask;
  item.priority_score = static_cast<float>(input.candidate.rt_hit_count);
  item.source = ProposalSource::kFallback;
  item.topk_feature_ids_offset = 0;

  if (!ShouldFallback(input) && input.output != nullptr) {
    item.feature_family_mask = FamilyMaskFromScores(*input.output, config);
    item.priority_score = input.output->priority_score;
    item.source = ProposalSource::kRaw;
    item.topk_feature_ids_offset = 0;
  }
  return item;
}

}  // namespace

Status BuildDummyProposalOutputs(const std::vector<ProposalFeatureRow>& rows,
                                 std::vector<ProposalOutput>* outputs) {
  if (outputs == nullptr) {
    return Status::Error("proposal output pointer is null");
  }
  outputs->clear();
  outputs->reserve(rows.size());
  for (const ProposalFeatureRow& row : rows) {
    ProposalOutput output;
    output.candidate_id = row.candidate_id;
    output.interval_scores = row.interval_targets;
    output.family_scores = row.family_targets;
    output.priority_score = row.priority_target;
    output.cost_score = row.cost_target;
    output.uncertainty_score = row.uncertainty_target;
    if (auto status = ValidateProposalOutput(output); !status.ok) {
      return status;
    }
    outputs->push_back(output);
  }
  return Status::Ok();
}

bool IsProposalFeatureRowOod(const ProposalFeatureRow& row,
                             const ProposalSchedulingConfig& config) {
  for (float value : row.features) {
    if (!std::isfinite(value) || std::abs(value) > config.ood_abs_feature_threshold) {
      return true;
    }
  }
  return false;
}

Status ScheduleExactWorkItemsFromProposals(const ProxyScene& scene,
                                           const RawCandidateQueue& raw_queue,
                                           const std::vector<ProposalFeatureRow>& rows,
                                           const std::vector<ProposalOutput>& proposal_outputs,
                                           const ProposalSchedulingConfig& config,
                                           std::vector<ExactWorkItem>* work_queue,
                                           ProposalScheduleStats* stats) {
  if (work_queue == nullptr) {
    return Status::Error("exact work queue output pointer is null");
  }
  std::string config_error;
  if (!IsValidSchedulingConfig(config, &config_error)) {
    return Status::Error(config_error);
  }
  if (auto status = ValidateProxyScene(scene); !status.ok) {
    return status;
  }
  if (raw_queue.query_id != scene.query_id) {
    return Status::Error("raw candidate queue query_id must match proxy scene query_id");
  }

  std::map<std::uint64_t, const ProposalFeatureRow*> rows_by_candidate;
  for (const ProposalFeatureRow& row : rows) {
    if (row.candidate_id == 0) {
      return Status::Error("ProposalFeatureRow.candidate_id is required");
    }
    if (!rows_by_candidate.emplace(row.candidate_id, &row).second) {
      return Status::Error("proposal feature rows contain duplicate candidate_id");
    }
  }

  std::map<std::uint64_t, const ProposalOutput*> outputs_by_candidate;
  for (const ProposalOutput& output : proposal_outputs) {
    if (output.candidate_id == 0) {
      return Status::Error("ProposalOutput.candidate_id is required");
    }
    if (!outputs_by_candidate.emplace(output.candidate_id, &output).second) {
      return Status::Error("proposal outputs contain duplicate candidate_id");
    }
  }

  ProposalScheduleStats local_stats;
  local_stats.raw_candidate_count = raw_queue.candidates.size();
  local_stats.proposal_output_count = proposal_outputs.size();

  std::vector<ExactWorkItem> scheduled;
  scheduled.reserve(raw_queue.candidates.size());
  for (const CandidateRecord& candidate : raw_queue.candidates) {
    if (auto status = ValidateCandidateRecord(candidate); !status.ok) {
      return status;
    }

    ScheduleInput input;
    input.candidate = candidate;
    const auto row_it = rows_by_candidate.find(candidate.candidate_id);
    input.row = row_it != rows_by_candidate.end() ? row_it->second : nullptr;
    const auto output_it = outputs_by_candidate.find(candidate.candidate_id);
    input.output = output_it != outputs_by_candidate.end() ? output_it->second : nullptr;
    input.missing_proposal = input.output == nullptr;
    input.ood = input.row == nullptr || IsProposalFeatureRowOod(*input.row, config);
    if (input.output != nullptr) {
      input.invalid_proposal = HasNonFiniteProposal(*input.output) ||
                               !ValidateProposalOutput(*input.output).ok;
      input.high_uncertainty = std::isfinite(input.output->uncertainty_score) &&
                               input.output->uncertainty_score >=
                                   config.uncertainty_fallback_threshold;
    }

    local_stats.missing_proposal_fallback_count += input.missing_proposal ? 1U : 0U;
    local_stats.invalid_proposal_fallback_count += input.invalid_proposal ? 1U : 0U;
    local_stats.ood_fallback_count += input.ood ? 1U : 0U;
    local_stats.high_uncertainty_fallback_count += input.high_uncertainty ? 1U : 0U;
    local_stats.fallback_count += ShouldFallback(input) ? 1U : 0U;

    ExactWorkItem item = MakeWorkItem(scene, input, config);
    scheduled.push_back(item);
  }

  if (!config.preserve_candidate_order) {
    if (NeedsPrioritySort(scheduled)) {
      std::stable_sort(scheduled.begin(),
                       scheduled.end(),
                       [](const ExactWorkItem& lhs, const ExactWorkItem& rhs) {
                         return lhs.priority_score > rhs.priority_score;
                       });
    }
  }

  for (std::size_t i = 0; i < scheduled.size(); ++i) {
    scheduled[i].work_item_id = config.first_work_item_id + i;
    if (i < raw_queue.candidates.size() &&
        scheduled[i].parent_candidate_id != raw_queue.candidates[i].candidate_id) {
      ++local_stats.reordered_count;
    }
  }

  if (auto status = ValidateProposalScheduleConservation(raw_queue, scheduled); !status.ok) {
    return status;
  }
  local_stats.work_item_count = scheduled.size();
  local_stats.monotonic_safe = true;
  *work_queue = std::move(scheduled);
  if (stats != nullptr) {
    *stats = local_stats;
  }
  return Status::Ok();
}

Status ValidateProposalScheduleConservation(const RawCandidateQueue& raw_queue,
                                            const std::vector<ExactWorkItem>& work_queue) {
  if (raw_queue.query_id == 0) {
    return Status::Error("raw candidate queue query_id is required");
  }
  if (raw_queue.candidates.size() != work_queue.size()) {
    return Status::Error("proposal schedule must conserve candidate count");
  }

  std::set<std::uint64_t> candidate_ids;
  for (const CandidateRecord& candidate : raw_queue.candidates) {
    if (auto status = ValidateCandidateRecord(candidate); !status.ok) {
      return status;
    }
    if (!candidate_ids.insert(candidate.candidate_id).second) {
      return Status::Error("raw candidate queue contains duplicate candidate_id");
    }
  }

  std::set<std::uint64_t> seen_candidate_ids;
  std::set<std::uint64_t> seen_work_item_ids;
  for (const ExactWorkItem& item : work_queue) {
    if (auto status = ValidateExactWorkItem(item); !status.ok) {
      return status;
    }
    if (item.query_id != raw_queue.query_id) {
      return Status::Error("scheduled work item query_id must match raw queue");
    }
    if (!candidate_ids.contains(item.parent_candidate_id)) {
      return Status::Error("scheduled work item references an unknown candidate");
    }
    if (!seen_candidate_ids.insert(item.parent_candidate_id).second) {
      return Status::Error("scheduled work queue duplicates a parent candidate");
    }
    if (!seen_work_item_ids.insert(item.work_item_id).second) {
      return Status::Error("scheduled work queue contains duplicate work_item_id");
    }
    if ((item.feature_family_mask &
         (kFeatureFamilyPointTriangle | kFeatureFamilyEdgeEdge)) == 0) {
      return Status::Error("scheduled work item has no exact feature family enabled");
    }
    if ((item.feature_family_mask & ~kRuntimeConservativeFeatureFamilyMask) != 0) {
      return Status::Error("scheduled work item has unknown exact feature-family bits");
    }
  }

  return Status::Ok();
}

Status BuildProposalFeatureRowsFromRuntimeCandidates(
    const std::vector<CandidateRecord>& candidates,
    const CandidateDensityStats& density,
    const std::unordered_map<std::uint64_t, std::uint32_t>& conservative_family_masks_by_query_id,
    std::vector<ProposalFeatureRow>* rows) {
  if (rows == nullptr) {
    return Status::Error("proposal runtime feature rows pointer is null");
  }
  if (density.proxy_count == 0U) {
    return Status::Error("CandidateDensityStats.proxy_count must be non-zero");
  }
  const RuntimeDensityDerivedStats derived = MakeRuntimeDensityDerivedStats(density);
  rows->clear();
  rows->reserve(candidates.size());
  for (const CandidateRecord& candidate : candidates) {
    ProposalFeatureRow row;
    const std::uint32_t base_family_mask =
        BaseFamilyMaskForRuntimeQuery(conservative_family_masks_by_query_id, candidate.query_id);
    if (auto status = BuildRuntimeFeatureRowFromDerived(candidate, derived, base_family_mask, &row);
        !status.ok) {
      return status;
    }
    rows->push_back(row);
  }
  return Status::Ok();
}

Status ScheduleRuntimeExactWorkItemsFromProposals(
    const std::vector<CandidateRecord>& candidates,
    const std::vector<ProposalFeatureRow>& rows,
    const std::vector<ProposalOutput>& proposal_outputs,
    const std::unordered_map<std::uint64_t, std::uint32_t>& conservative_family_masks_by_query_id,
    const ProposalSchedulingConfig& config,
    std::vector<ExactWorkItem>* work_queue,
    ProposalScheduleStats* stats) {
  if (work_queue == nullptr) {
    return Status::Error("runtime exact work queue pointer is null");
  }
  std::string config_error;
  if (!IsValidSchedulingConfig(config, &config_error)) {
    return Status::Error(config_error);
  }

  std::map<std::uint64_t, const ProposalFeatureRow*> rows_by_candidate;
  for (const ProposalFeatureRow& row : rows) {
    if (row.candidate_id == 0U) {
      return Status::Error("ProposalFeatureRow.candidate_id is required");
    }
    if (!rows_by_candidate.emplace(row.candidate_id, &row).second) {
      return Status::Error("proposal runtime feature rows contain duplicate candidate_id");
    }
  }

  std::map<std::uint64_t, const ProposalOutput*> outputs_by_candidate;
  for (const ProposalOutput& output : proposal_outputs) {
    if (output.candidate_id == 0U) {
      return Status::Error("ProposalOutput.candidate_id is required");
    }
    if (!outputs_by_candidate.emplace(output.candidate_id, &output).second) {
      return Status::Error("proposal runtime outputs contain duplicate candidate_id");
    }
  }

  ProposalScheduleStats local_stats;
  local_stats.raw_candidate_count = candidates.size();
  local_stats.proposal_output_count = proposal_outputs.size();
  local_stats.work_item_count = candidates.size();

  std::vector<ExactWorkItem> scheduled;
  scheduled.reserve(candidates.size());
  for (const CandidateRecord& candidate : candidates) {
    if (auto status = ValidateCandidateRecord(candidate); !status.ok) {
      return status;
    }

    const auto row_it = rows_by_candidate.find(candidate.candidate_id);
    const auto output_it = outputs_by_candidate.find(candidate.candidate_id);
    const ProposalFeatureRow* row = row_it != rows_by_candidate.end() ? row_it->second : nullptr;
    const ProposalOutput* output =
        output_it != outputs_by_candidate.end() ? output_it->second : nullptr;
    const bool missing_proposal = output == nullptr;
    const bool invalid_proposal =
        output != nullptr && (HasNonFiniteProposal(*output) || !ValidateProposalOutput(*output).ok);
    const bool ood = row == nullptr || IsProposalFeatureRowOod(*row, config);
    const bool high_uncertainty =
        output != nullptr && std::isfinite(output->uncertainty_score) &&
        output->uncertainty_score >= config.uncertainty_fallback_threshold;
    const bool fallback = missing_proposal || invalid_proposal || ood || high_uncertainty;

    local_stats.fallback_count += fallback ? 1U : 0U;
    local_stats.missing_proposal_fallback_count += missing_proposal ? 1U : 0U;
    local_stats.invalid_proposal_fallback_count += invalid_proposal ? 1U : 0U;
    local_stats.ood_fallback_count += ood ? 1U : 0U;
    local_stats.high_uncertainty_fallback_count += high_uncertainty ? 1U : 0U;

    const std::uint32_t base_family_mask = BaseFamilyMaskForRuntimeQuery(
        conservative_family_masks_by_query_id, candidate.query_id, config);
    ExactWorkItem item;
    item.parent_candidate_id = candidate.candidate_id;
    item.query_id = candidate.query_id;
    item.slab_id = candidate.slab_id;
    item.patch_a_id = candidate.patch_a_id;
    item.patch_b_id = candidate.patch_b_id;
    item.interval_t0 = config.fallback_interval_t0;
    item.interval_t1 = config.fallback_interval_t1;
    item.feature_family_mask = base_family_mask;
    item.topk_feature_ids_offset = 0U;
    item.depth = 0U;
    item.priority_score = static_cast<float>(candidate.rt_hit_count);
    item.source = ProposalSource::kFallback;

    if (!fallback && output != nullptr) {
      item.feature_family_mask = base_family_mask | PredictedFamilyMaskFromScores(*output, config);
      item.priority_score = output->priority_score;
      item.source = ProposalSource::kRefined;
    }
    scheduled.push_back(item);
  }

  if (!config.preserve_candidate_order) {
    if (NeedsPrioritySort(scheduled)) {
      std::stable_sort(scheduled.begin(),
                       scheduled.end(),
                       [](const ExactWorkItem& lhs, const ExactWorkItem& rhs) {
                         return lhs.priority_score > rhs.priority_score;
                       });
    }
  }
  for (std::size_t i = 0; i < scheduled.size(); ++i) {
    scheduled[i].work_item_id = config.first_work_item_id + i;
    if (i < candidates.size() &&
        scheduled[i].parent_candidate_id != candidates[i].candidate_id) {
      ++local_stats.reordered_count;
    }
    if (auto status = ValidateExactWorkItem(scheduled[i]); !status.ok) {
      return status;
    }
  }

  std::set<std::uint64_t> candidate_ids;
  for (const CandidateRecord& candidate : candidates) {
    if (!candidate_ids.insert(candidate.candidate_id).second) {
      return Status::Error("runtime candidates contain duplicate candidate_id");
    }
  }
  if (scheduled.size() != candidates.size()) {
    return Status::Error("runtime proposal schedule must conserve candidate count");
  }
  std::set<std::uint64_t> seen_candidate_ids;
  std::set<std::uint64_t> seen_work_item_ids;
  for (const ExactWorkItem& item : scheduled) {
    if (!candidate_ids.contains(item.parent_candidate_id)) {
      return Status::Error("runtime scheduled work item references an unknown candidate");
    }
    if (!seen_candidate_ids.insert(item.parent_candidate_id).second) {
      return Status::Error("runtime scheduled work queue duplicates a parent candidate");
    }
    if (!seen_work_item_ids.insert(item.work_item_id).second) {
      return Status::Error("runtime scheduled work queue contains duplicate work_item_id");
    }
    if ((item.feature_family_mask & kRuntimeConservativeFeatureFamilyMask) == 0U) {
      return Status::Error("runtime scheduled work item has no exact feature family enabled");
    }
    if ((item.feature_family_mask & ~kRuntimeConservativeFeatureFamilyMask) != 0U) {
      return Status::Error("runtime scheduled work item has unknown exact feature-family bits");
    }
  }

  local_stats.monotonic_safe = true;
  *work_queue = std::move(scheduled);
  if (stats != nullptr) {
    *stats = local_stats;
  }
  return Status::Ok();
}

Status RunDummyProposalScheduleFromRuntimeCandidates(
    const std::vector<CandidateRecord>& candidates,
    const CandidateDensityStats& density,
    const std::unordered_map<std::uint64_t, std::uint32_t>& conservative_family_masks_by_query_id,
    const ProposalSchedulingConfig& config,
    const bool materialize_artifacts,
    ProposalRuntimeScheduleResult* result) {
  if (result == nullptr) {
    return Status::Error("runtime dummy proposal result pointer is null");
  }
  std::string config_error;
  if (!IsValidSchedulingConfig(config, &config_error)) {
    return Status::Error(config_error);
  }
  if (density.proxy_count == 0U) {
    return Status::Error("CandidateDensityStats.proxy_count must be non-zero");
  }

  const RuntimeDensityDerivedStats derived = MakeRuntimeDensityDerivedStats(density);

  ProposalRuntimeScheduleResult built;
  built.work_queue.reserve(candidates.size());
  if (materialize_artifacts) {
    built.feature_rows.reserve(candidates.size());
    built.proposal_outputs.reserve(candidates.size());
  }

  built.stats.raw_candidate_count = candidates.size();
  built.stats.proposal_output_count = candidates.size();
  built.stats.work_item_count = candidates.size();

  for (const CandidateRecord& candidate : candidates) {
    if (auto status = ValidateCandidateRecord(candidate); !status.ok) {
      return status;
    }
    const std::uint32_t base_family_mask =
        BaseFamilyMaskForRuntimeQuery(conservative_family_masks_by_query_id, candidate.query_id);
    const bool ood = IsRuntimeCandidateOod(candidate, derived, config);
    built.stats.fallback_count += ood ? 1U : 0U;
    built.stats.ood_fallback_count += ood ? 1U : 0U;

    const auto family_scores = NormalizedDummyFamilyScores(base_family_mask);
    if (materialize_artifacts) {
      ProposalFeatureRow row;
      if (auto status =
              BuildRuntimeFeatureRowFromDerived(candidate, derived, base_family_mask, &row);
          !status.ok) {
        return status;
      }
      ProposalOutput output;
      output.candidate_id = candidate.candidate_id;
      output.interval_scores = row.interval_targets;
      output.family_scores = family_scores;
      output.priority_score = std::max(0.0F, row.priority_target);
      output.cost_score = std::max(0.0F, row.cost_target);
      output.uncertainty_score = ood ? 1.0F : std::max(0.0F, row.uncertainty_target);
      if (auto status = ValidateProposalOutput(output); !status.ok) {
        return status;
      }
      built.feature_rows.push_back(row);
      built.proposal_outputs.push_back(output);
    }

    ExactWorkItem item;
    item.parent_candidate_id = candidate.candidate_id;
    item.query_id = candidate.query_id;
    item.slab_id = candidate.slab_id;
    item.patch_a_id = candidate.patch_a_id;
    item.patch_b_id = candidate.patch_b_id;
    item.interval_t0 = config.fallback_interval_t0;
    item.interval_t1 = config.fallback_interval_t1;
    item.topk_feature_ids_offset = 0U;
    item.depth = 0U;
    if (ood) {
      item.feature_family_mask = base_family_mask;
      item.priority_score = static_cast<float>(candidate.rt_hit_count);
      item.source = ProposalSource::kFallback;
    } else {
      std::uint32_t feature_mask = base_family_mask;
      if (family_scores[0] >= config.family_score_threshold) {
        feature_mask |= kFeatureFamilyPointTriangle;
      }
      if (family_scores[1] >= config.family_score_threshold) {
        feature_mask |= kFeatureFamilyEdgeEdge;
      }
      item.feature_family_mask = feature_mask;
      item.priority_score = std::max(0.0F, RuntimePriorityTarget(candidate, derived));
      item.source = ProposalSource::kRefined;
    }
    built.work_queue.push_back(item);
  }

  if (!config.preserve_candidate_order) {
    if (NeedsPrioritySort(built.work_queue)) {
      std::stable_sort(built.work_queue.begin(),
                       built.work_queue.end(),
                       [](const ExactWorkItem& lhs, const ExactWorkItem& rhs) {
                         return lhs.priority_score > rhs.priority_score;
                       });
    }
  }
  for (std::size_t i = 0; i < built.work_queue.size(); ++i) {
    built.work_queue[i].work_item_id = config.first_work_item_id + i;
    if (i < candidates.size() &&
        built.work_queue[i].parent_candidate_id != candidates[i].candidate_id) {
      ++built.stats.reordered_count;
    }
    if (auto status = ValidateExactWorkItem(built.work_queue[i]); !status.ok) {
      return status;
    }
  }

  built.stats.monotonic_safe = true;
  *result = std::move(built);
  return Status::Ok();
}

}  // namespace p2cccd
