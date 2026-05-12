#pragma once

#include <array>
#include <cstdint>

namespace p2cccd {

enum class ProxyType : std::uint8_t {
  kUnknown = 0,
  kSweptAabb = 1,
  kCapsule = 2,
};

enum class ProposalSource : std::uint8_t {
  kRaw = 0,
  kRefined = 1,
  kFallback = 2,
};

enum class CertificateStatus : std::uint8_t {
  kCollision = 0,
  kSeparation = 1,
  kUndecided = 2,
};

enum class CertificateRefinementMode : std::uint8_t {
  kNone = 0,
  kBisectInterval = 1,
  kRequestGeometry = 2,
  kEscalatePrecision = 3,
};

enum class AuditStage : std::uint8_t {
  kRt = 0,
  kProposal = 1,
  kExact = 2,
  kRefine = 3,
  kCertify = 4,
};

struct CandidateRecord {
  std::uint32_t schema_version = 1;
  std::uint64_t candidate_id = 0;
  std::uint64_t query_id = 0;
  std::uint32_t slab_id = 0;
  std::uint32_t object_a_id = 0;
  std::uint32_t patch_a_id = 0;
  std::uint32_t object_b_id = 0;
  std::uint32_t patch_b_id = 0;
  ProxyType proxy_type_a = ProxyType::kUnknown;
  ProxyType proxy_type_b = ProxyType::kUnknown;
  std::uint32_t rt_hit_count = 0;
  std::array<float, 4> motion_bound{0.0f, 0.0f, 0.0f, 0.0f};
  std::uint32_t proxy_features_offset = 0;
  std::uint32_t flags = 0;
};

struct ProposalOutput {
  std::uint64_t candidate_id = 0;
  std::array<float, 8> interval_scores{};
  std::array<float, 8> family_scores{};
  float priority_score = 0.0f;
  float cost_score = 0.0f;
  float uncertainty_score = 0.0f;
};

struct ExactWorkItem {
  std::uint64_t work_item_id = 0;
  std::uint64_t parent_candidate_id = 0;
  std::uint64_t query_id = 0;
  std::uint32_t slab_id = 0;
  std::uint32_t patch_a_id = 0;
  std::uint32_t patch_b_id = 0;
  double interval_t0 = 0.0;
  double interval_t1 = 1.0;
  std::uint32_t feature_family_mask = 0;
  std::uint32_t topk_feature_ids_offset = 0;
  std::uint16_t depth = 0;
  float priority_score = 0.0f;
  ProposalSource source = ProposalSource::kRaw;
};

struct CertificateResult {
  std::uint64_t work_item_id = 0;
  std::uint64_t query_id = 0;
  CertificateStatus status = CertificateStatus::kUndecided;
  double interval_t0 = 0.0;
  double interval_t1 = 1.0;
  double toi_upper = 1.0;
  double safe_margin_lb = 0.0;
  std::uint8_t witness_family = 0;
  std::int64_t witness_id_a = -1;
  std::int64_t witness_id_b = -1;
  std::uint32_t covered_feature_mask = 0;
  double eps_time = 1.0e-4;
  double eps_space = 1.0e-6;
  std::uint16_t reason_code = 0;
  CertificateRefinementMode next_refinement_mode = CertificateRefinementMode::kNone;
};

struct AuditLogRow {
  std::uint64_t event_id = 0;
  std::uint64_t query_id = 0;
  std::uint64_t candidate_id = 0;
  std::uint64_t work_item_id = 0;
  AuditStage stage = AuditStage::kRt;
  std::uint16_t action = 0;
  std::uint16_t depth = 0;
  double interval_t0 = 0.0;
  double interval_t1 = 1.0;
  std::uint64_t timestamp_us = 0;
  double aux_value0 = 0.0;
  double aux_value1 = 0.0;
};

struct BenchmarkRow {
  std::uint64_t query_count = 0;
  std::uint64_t fn_count = 0;
  std::uint64_t fp_count = 0;
  double candidate_recall = 0.0;
  double avg_candidates = 0.0;
  double avg_exact_evals = 0.0;
  double avg_subdivision_depth = 0.0;
  double fallback_ratio = 0.0;
  double rt_ms = 0.0;
  double proposal_ms = 0.0;
  double exact_ms = 0.0;
  double total_ms = 0.0;
  double qps = 0.0;
};

}  // namespace p2cccd
