#pragma once

#include "common/runtime_contracts.h"
#include "common/status.h"
#include "rt_candidate/candidate_generation_result.h"
#include "rt_candidate/candidate_generator.h"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace p2cccd {

enum class ExternalQueryFamily : std::uint8_t {
  kVertexFace = 0,
  kEdgeEdge = 1,
};

struct ExternalBatchQuery {
  std::uint64_t source_query_id = 0;
  std::uint32_t source_query_index = 0;
  ExternalQueryFamily family = ExternalQueryFamily::kVertexFace;
  std::array<std::array<double, 3>, 4> vertices_t0{};
  std::array<std::array<double, 3>, 4> vertices_t1{};
  bool has_ground_truth = false;
  bool ground_truth_collides = false;
  bool has_box_pair = false;
  std::array<std::uint32_t, 2> box_pair{1, 2};
};

struct RuntimeQueryIdMapping {
  std::uint64_t source_query_id = 0;
  std::uint64_t runtime_query_id = 0;
};

struct ExternalBatchCandidateResult {
  std::string backend_name;
  RtCandidateTiming timing;
  std::uint64_t primitive_count = 0;
  std::uint64_t raw_hit_count = 0;
  std::uint64_t compact_candidate_count = 0;
  double candidate_recall = 0.0;
  std::vector<CandidateRecord> candidates;
  std::vector<RuntimeQueryIdMapping> runtime_query_ids;
};

Status GenerateCandidatesForExternalBatch(const std::vector<ExternalBatchQuery>& queries,
                                          const CandidateGeneratorConfig& config,
                                          ExternalBatchCandidateResult* result);

}  // namespace p2cccd
