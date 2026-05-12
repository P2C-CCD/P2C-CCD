#pragma once

#include "certificate/certificate_engine.h"
#include "common/runtime_contracts.h"
#include "common/status.h"
#include "rt_candidate/candidate_generation_result.h"
#include "rt_candidate/proxy_scene.h"

#include <array>
#include <cstdint>
#include <filesystem>
#include <vector>

namespace p2cccd {

constexpr std::uint32_t kProposalFeatureRowSchemaVersion = 1;
constexpr std::uint32_t kProposalFeatureDimension = 32;
constexpr std::uint32_t kProposalIntervalBinCount = 8;
constexpr std::uint32_t kProposalFamilyCount = 8;

struct RawCandidateQueue {
  std::uint64_t query_id = 0;
  std::vector<CandidateRecord> candidates;
  CandidateDensityStats density;
};

struct ProposalFeatureRow {
  std::uint32_t schema_version = kProposalFeatureRowSchemaVersion;
  std::uint64_t query_id = 0;
  std::uint64_t candidate_id = 0;
  std::uint32_t slab_id = 0;
  std::uint32_t object_a_id = 0;
  std::uint32_t patch_a_id = 0;
  std::uint32_t object_b_id = 0;
  std::uint32_t patch_b_id = 0;
  std::array<float, kProposalFeatureDimension> features{};
  std::array<float, kProposalIntervalBinCount> interval_targets{};
  std::array<float, kProposalFamilyCount> family_targets{};
  float priority_target = 0.0F;
  float cost_target = 0.0F;
  float uncertainty_target = 0.0F;
  std::uint32_t target_mask = 0;
};

struct ProposalQueueConfig {
  std::uint64_t first_work_item_id = 1;
  std::uint32_t feature_family_mask = kFeatureFamilyPointTriangle | kFeatureFamilyEdgeEdge;
  double fallback_interval_t0 = 0.0;
  double fallback_interval_t1 = 1.0;
};

struct ProposalDataFlow {
  RawCandidateQueue raw_candidate_queue;
  std::vector<ProposalFeatureRow> feature_rows;
  std::vector<ExactWorkItem> exact_work_queue;
};

Status BuildRawCandidateQueue(const CandidateGenerationResult& generation_result,
                              RawCandidateQueue* queue);
Status ExtractProposalFeatureRows(const ProxyScene& scene,
                                  const CandidateGenerationResult& generation_result,
                                  std::vector<ProposalFeatureRow>* rows);
Status BuildExactWorkQueuePassthrough(const ProxyScene& scene,
                                      const RawCandidateQueue& raw_queue,
                                      const ProposalQueueConfig& config,
                                      std::vector<ExactWorkItem>* work_queue);
Status BuildProposalDataFlow(const ProxyScene& scene,
                             const CandidateGenerationResult& generation_result,
                             const ProposalQueueConfig& config,
                             ProposalDataFlow* data_flow);
Status ValidateProposalDataFlow(const ProposalDataFlow& data_flow);

std::string ProposalFeatureCsvHeader();
Status WriteProposalFeatureRowsCsv(const std::filesystem::path& path,
                                   const std::vector<ProposalFeatureRow>& rows,
                                   bool append);

}  // namespace p2cccd
