#pragma once

#include "common/runtime_contracts.h"
#include "common/status.h"
#include "proposal/proposal_features.h"

#include <cstdint>
#include <unordered_map>
#include <vector>

namespace p2cccd {

struct ProposalSchedulingConfig {
  std::uint64_t first_work_item_id = 1;
  std::uint32_t conservative_feature_family_mask =
      kFeatureFamilyPointTriangle | kFeatureFamilyEdgeEdge;
  double fallback_interval_t0 = 0.0;
  double fallback_interval_t1 = 1.0;
  float family_score_threshold = 0.5F;
  float uncertainty_fallback_threshold = 0.95F;
  float ood_abs_feature_threshold = 1.0e5F;
  bool preserve_candidate_order = false;
};

struct ProposalScheduleStats {
  std::uint64_t raw_candidate_count = 0;
  std::uint64_t proposal_output_count = 0;
  std::uint64_t work_item_count = 0;
  std::uint64_t fallback_count = 0;
  std::uint64_t missing_proposal_fallback_count = 0;
  std::uint64_t invalid_proposal_fallback_count = 0;
  std::uint64_t ood_fallback_count = 0;
  std::uint64_t high_uncertainty_fallback_count = 0;
  std::uint64_t reordered_count = 0;
  bool monotonic_safe = false;
};

struct ProposalRuntimeScheduleResult {
  std::vector<ProposalFeatureRow> feature_rows;
  std::vector<ProposalOutput> proposal_outputs;
  std::vector<ExactWorkItem> work_queue;
  ProposalScheduleStats stats;
};

Status BuildDummyProposalOutputs(const std::vector<ProposalFeatureRow>& rows,
                                 std::vector<ProposalOutput>* outputs);
Status ScheduleExactWorkItemsFromProposals(const ProxyScene& scene,
                                           const RawCandidateQueue& raw_queue,
                                           const std::vector<ProposalFeatureRow>& rows,
                                           const std::vector<ProposalOutput>& proposal_outputs,
                                           const ProposalSchedulingConfig& config,
                                           std::vector<ExactWorkItem>* work_queue,
                                           ProposalScheduleStats* stats);
Status ValidateProposalScheduleConservation(const RawCandidateQueue& raw_queue,
                                            const std::vector<ExactWorkItem>& work_queue);
bool IsProposalFeatureRowOod(const ProposalFeatureRow& row,
                             const ProposalSchedulingConfig& config);
Status BuildProposalFeatureRowsFromRuntimeCandidates(
    const std::vector<CandidateRecord>& candidates,
    const CandidateDensityStats& density,
    const std::unordered_map<std::uint64_t, std::uint32_t>& conservative_family_masks_by_query_id,
    std::vector<ProposalFeatureRow>* rows);
Status ScheduleRuntimeExactWorkItemsFromProposals(
    const std::vector<CandidateRecord>& candidates,
    const std::vector<ProposalFeatureRow>& rows,
    const std::vector<ProposalOutput>& proposal_outputs,
    const std::unordered_map<std::uint64_t, std::uint32_t>& conservative_family_masks_by_query_id,
    const ProposalSchedulingConfig& config,
    std::vector<ExactWorkItem>* work_queue,
    ProposalScheduleStats* stats);
Status RunDummyProposalScheduleFromRuntimeCandidates(
    const std::vector<CandidateRecord>& candidates,
    const CandidateDensityStats& density,
    const std::unordered_map<std::uint64_t, std::uint32_t>& conservative_family_masks_by_query_id,
    const ProposalSchedulingConfig& config,
    bool materialize_artifacts,
    ProposalRuntimeScheduleResult* result);

}  // namespace p2cccd
