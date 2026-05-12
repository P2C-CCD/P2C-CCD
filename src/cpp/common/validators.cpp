#include "common/validators.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <string>

namespace p2cccd {
namespace {

constexpr std::uint32_t kRuntimeContractSchemaVersion = 1;

Status Require(bool condition, const char* message) {
  if (condition) {
    return Status::Ok();
  }
  return Status::Error(message);
}

Status RequireNonZero(std::uint64_t value, const char* field_name) {
  if (value != 0) {
    return Status::Ok();
  }
  return Status::Error(std::string(field_name) + " is required");
}

Status RequireFinite(double value, const char* field_name) {
  if (std::isfinite(value)) {
    return Status::Ok();
  }
  return Status::Error(std::string(field_name) + " must be finite");
}

Status RequireFinite(float value, const char* field_name) {
  if (std::isfinite(value)) {
    return Status::Ok();
  }
  return Status::Error(std::string(field_name) + " must be finite");
}

Status RequireNonNegative(double value, const char* field_name) {
  if (auto status = RequireFinite(value, field_name); !status.ok) {
    return status;
  }
  if (value >= 0.0) {
    return Status::Ok();
  }
  return Status::Error(std::string(field_name) + " must be non-negative");
}

Status RequirePositive(double value, const char* field_name) {
  if (auto status = RequireFinite(value, field_name); !status.ok) {
    return status;
  }
  if (value > 0.0) {
    return Status::Ok();
  }
  return Status::Error(std::string(field_name) + " must be positive");
}

Status ValidateUnitInterval(double t0, double t1) {
  if (auto status = RequireFinite(t0, "interval_t0"); !status.ok) {
    return status;
  }
  if (auto status = RequireFinite(t1, "interval_t1"); !status.ok) {
    return status;
  }
  if (t0 < 0.0 || t1 > 1.0 || t0 > t1) {
    return Status::Error("interval must satisfy 0 <= interval_t0 <= interval_t1 <= 1");
  }
  return Status::Ok();
}

Status ValidateRatio(double value, const char* field_name) {
  if (auto status = RequireFinite(value, field_name); !status.ok) {
    return status;
  }
  if (value >= 0.0 && value <= 1.0) {
    return Status::Ok();
  }
  return Status::Error(std::string(field_name) + " must be in [0, 1]");
}

template <std::size_t N>
Status ValidateFiniteArray(const std::array<float, N>& values, const char* field_name) {
  for (float value : values) {
    if (!std::isfinite(value)) {
      return Status::Error(std::string(field_name) + " contains a non-finite value");
    }
  }
  return Status::Ok();
}

template <std::size_t N>
Status ValidateNonNegativeArray(const std::array<float, N>& values, const char* field_name) {
  for (float value : values) {
    if (!std::isfinite(value) || value < 0.0f) {
      return Status::Error(std::string(field_name) + " must contain finite non-negative values");
    }
  }
  return Status::Ok();
}

bool IsKnownProxyType(ProxyType value) {
  switch (value) {
    case ProxyType::kSweptAabb:
    case ProxyType::kCapsule:
      return true;
    case ProxyType::kUnknown:
      return false;
  }
  return false;
}

bool IsKnownProposalSource(ProposalSource value) {
  switch (value) {
    case ProposalSource::kRaw:
    case ProposalSource::kRefined:
    case ProposalSource::kFallback:
      return true;
  }
  return false;
}

bool IsKnownCertificateStatus(CertificateStatus value) {
  switch (value) {
    case CertificateStatus::kCollision:
    case CertificateStatus::kSeparation:
    case CertificateStatus::kUndecided:
      return true;
  }
  return false;
}

bool IsKnownCertificateRefinementMode(CertificateRefinementMode value) {
  switch (value) {
    case CertificateRefinementMode::kNone:
    case CertificateRefinementMode::kBisectInterval:
    case CertificateRefinementMode::kRequestGeometry:
    case CertificateRefinementMode::kEscalatePrecision:
      return true;
  }
  return false;
}

bool IsKnownAuditStage(AuditStage value) {
  switch (value) {
    case AuditStage::kRt:
    case AuditStage::kProposal:
    case AuditStage::kExact:
    case AuditStage::kRefine:
    case AuditStage::kCertify:
      return true;
  }
  return false;
}

}  // namespace

Status ValidateCandidateRecord(const CandidateRecord& record) {
  if (record.schema_version != kRuntimeContractSchemaVersion) {
    return Status::Error("CandidateRecord.schema_version is unsupported");
  }
  if (auto status = RequireNonZero(record.candidate_id, "CandidateRecord.candidate_id");
      !status.ok) {
    return status;
  }
  if (auto status = RequireNonZero(record.query_id, "CandidateRecord.query_id"); !status.ok) {
    return status;
  }
  if (!IsKnownProxyType(record.proxy_type_a)) {
    return Status::Error("CandidateRecord.proxy_type_a must be a concrete proxy type");
  }
  if (!IsKnownProxyType(record.proxy_type_b)) {
    return Status::Error("CandidateRecord.proxy_type_b must be a concrete proxy type");
  }
  if (record.rt_hit_count == 0) {
    return Status::Error("CandidateRecord.rt_hit_count is required");
  }
  return ValidateNonNegativeArray(record.motion_bound, "CandidateRecord.motion_bound");
}

Status ValidateProposalOutput(const ProposalOutput& output) {
  if (auto status = RequireNonZero(output.candidate_id, "ProposalOutput.candidate_id");
      !status.ok) {
    return status;
  }
  if (auto status = ValidateFiniteArray(output.interval_scores, "ProposalOutput.interval_scores");
      !status.ok) {
    return status;
  }
  if (auto status = ValidateFiniteArray(output.family_scores, "ProposalOutput.family_scores");
      !status.ok) {
    return status;
  }
  if (auto status = RequireFinite(output.priority_score, "ProposalOutput.priority_score");
      !status.ok) {
    return status;
  }
  if (auto status = RequireNonNegative(output.cost_score, "ProposalOutput.cost_score");
      !status.ok) {
    return status;
  }
  return RequireNonNegative(output.uncertainty_score, "ProposalOutput.uncertainty_score");
}

Status ValidateExactWorkItem(const ExactWorkItem& item) {
  if (auto status = RequireNonZero(item.work_item_id, "ExactWorkItem.work_item_id");
      !status.ok) {
    return status;
  }
  if (auto status = RequireNonZero(item.parent_candidate_id, "ExactWorkItem.parent_candidate_id");
      !status.ok) {
    return status;
  }
  if (auto status = RequireNonZero(item.query_id, "ExactWorkItem.query_id"); !status.ok) {
    return status;
  }
  if (auto status = ValidateUnitInterval(item.interval_t0, item.interval_t1); !status.ok) {
    return status;
  }
  if (item.feature_family_mask == 0) {
    return Status::Error("ExactWorkItem.feature_family_mask is required");
  }
  if (auto status = RequireFinite(item.priority_score, "ExactWorkItem.priority_score");
      !status.ok) {
    return status;
  }
  return Require(IsKnownProposalSource(item.source), "ExactWorkItem.source is invalid");
}

Status ValidateCertificateResult(const CertificateResult& result) {
  if (auto status = RequireNonZero(result.work_item_id, "CertificateResult.work_item_id");
      !status.ok) {
    return status;
  }
  if (auto status = RequireNonZero(result.query_id, "CertificateResult.query_id"); !status.ok) {
    return status;
  }
  if (!IsKnownCertificateStatus(result.status)) {
    return Status::Error("CertificateResult.status is invalid");
  }
  if (auto status = ValidateUnitInterval(result.interval_t0, result.interval_t1); !status.ok) {
    return status;
  }
  if (auto status = RequireFinite(result.toi_upper, "CertificateResult.toi_upper"); !status.ok) {
    return status;
  }
  if (auto status = RequireFinite(result.safe_margin_lb, "CertificateResult.safe_margin_lb");
      !status.ok) {
    return status;
  }
  if (auto status = RequirePositive(result.eps_time, "CertificateResult.eps_time"); !status.ok) {
    return status;
  }
  if (auto status = RequirePositive(result.eps_space, "CertificateResult.eps_space"); !status.ok) {
    return status;
  }
  if (!IsKnownCertificateRefinementMode(result.next_refinement_mode)) {
    return Status::Error("CertificateResult.next_refinement_mode is invalid");
  }

  if (result.status == CertificateStatus::kCollision) {
    if (result.toi_upper < result.interval_t0 || result.toi_upper > result.interval_t1) {
      return Status::Error("CertificateResult.toi_upper must lie inside the certified interval");
    }
    if (result.witness_id_a < 0 || result.witness_id_b < 0) {
      return Status::Error("CertificateResult collision witnesses are required");
    }
    if (result.next_refinement_mode != CertificateRefinementMode::kNone) {
      return Status::Error("CertificateResult collision cannot request refinement");
    }
  }
  if (result.status == CertificateStatus::kSeparation) {
    if (result.safe_margin_lb < 0.0) {
      return Status::Error("CertificateResult.safe_margin_lb must be non-negative");
    }
    if (result.covered_feature_mask == 0) {
      return Status::Error("CertificateResult.covered_feature_mask is required for separation");
    }
    if (result.next_refinement_mode != CertificateRefinementMode::kNone) {
      return Status::Error("CertificateResult separation cannot request refinement");
    }
  }
  if (result.status == CertificateStatus::kUndecided && result.reason_code == 0) {
    return Status::Error("CertificateResult.reason_code is required for undecided results");
  }
  if (result.status == CertificateStatus::kUndecided &&
      result.next_refinement_mode == CertificateRefinementMode::kNone) {
    return Status::Error("CertificateResult.next_refinement_mode is required for undecided results");
  }
  return Status::Ok();
}

Status ValidateAuditLogRow(const AuditLogRow& row) {
  if (auto status = RequireNonZero(row.event_id, "AuditLogRow.event_id"); !status.ok) {
    return status;
  }
  if (auto status = RequireNonZero(row.query_id, "AuditLogRow.query_id"); !status.ok) {
    return status;
  }
  if (!IsKnownAuditStage(row.stage)) {
    return Status::Error("AuditLogRow.stage is invalid");
  }
  if (auto status = ValidateUnitInterval(row.interval_t0, row.interval_t1); !status.ok) {
    return status;
  }
  if (row.timestamp_us == 0) {
    return Status::Error("AuditLogRow.timestamp_us is required");
  }
  if (auto status = RequireFinite(row.aux_value0, "AuditLogRow.aux_value0"); !status.ok) {
    return status;
  }
  return RequireFinite(row.aux_value1, "AuditLogRow.aux_value1");
}

Status ValidateBenchmarkRow(const BenchmarkRow& row) {
  if (row.query_count == 0) {
    return Status::Error("BenchmarkRow.query_count is required");
  }
  if (row.fn_count > row.query_count) {
    return Status::Error("BenchmarkRow.fn_count cannot exceed query_count");
  }
  if (auto status = ValidateRatio(row.candidate_recall, "BenchmarkRow.candidate_recall");
      !status.ok) {
    return status;
  }
  if (auto status = ValidateRatio(row.fallback_ratio, "BenchmarkRow.fallback_ratio"); !status.ok) {
    return status;
  }
  if (auto status = RequireNonNegative(row.avg_candidates, "BenchmarkRow.avg_candidates");
      !status.ok) {
    return status;
  }
  if (auto status = RequireNonNegative(row.avg_exact_evals, "BenchmarkRow.avg_exact_evals");
      !status.ok) {
    return status;
  }
  if (auto status =
          RequireNonNegative(row.avg_subdivision_depth, "BenchmarkRow.avg_subdivision_depth");
      !status.ok) {
    return status;
  }
  if (auto status = RequireNonNegative(row.rt_ms, "BenchmarkRow.rt_ms"); !status.ok) {
    return status;
  }
  if (auto status = RequireNonNegative(row.proposal_ms, "BenchmarkRow.proposal_ms"); !status.ok) {
    return status;
  }
  if (auto status = RequireNonNegative(row.exact_ms, "BenchmarkRow.exact_ms"); !status.ok) {
    return status;
  }
  if (auto status = RequireNonNegative(row.total_ms, "BenchmarkRow.total_ms"); !status.ok) {
    return status;
  }
  return RequireNonNegative(row.qps, "BenchmarkRow.qps");
}

}  // namespace p2cccd
