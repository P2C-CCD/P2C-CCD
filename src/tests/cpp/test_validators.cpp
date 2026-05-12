#include "common/runtime_contracts.h"
#include "common/validators.h"

#include <iostream>

namespace {

int g_failures = 0;

void ExpectOk(const p2cccd::Status& status, const char* label) {
  if (!status.ok) {
    std::cerr << "FAIL " << label << ": " << status.message << '\n';
    ++g_failures;
  }
}

void ExpectError(const p2cccd::Status& status, const char* label) {
  if (status.ok) {
    std::cerr << "FAIL " << label << ": expected validation error\n";
    ++g_failures;
  }
}

}  // namespace

int main() {
  p2cccd::CandidateRecord candidate;
  candidate.candidate_id = 1;
  candidate.query_id = 7;
  candidate.proxy_type_a = p2cccd::ProxyType::kSweptAabb;
  candidate.proxy_type_b = p2cccd::ProxyType::kCapsule;
  candidate.rt_hit_count = 1;
  candidate.motion_bound = {0.1f, 0.2f, 0.3f, 0.4f};
  ExpectOk(p2cccd::ValidateCandidateRecord(candidate), "valid candidate");

  auto bad_candidate = candidate;
  bad_candidate.schema_version = 2;
  ExpectError(p2cccd::ValidateCandidateRecord(bad_candidate), "bad candidate schema");

  bad_candidate = candidate;
  bad_candidate.rt_hit_count = 0;
  ExpectError(p2cccd::ValidateCandidateRecord(bad_candidate), "zero candidate hit count");

  p2cccd::ProposalOutput proposal;
  proposal.candidate_id = 1;
  proposal.interval_scores.fill(0.25f);
  proposal.family_scores.fill(0.5f);
  proposal.priority_score = 0.75f;
  proposal.cost_score = 1.0f;
  proposal.uncertainty_score = 0.1f;
  ExpectOk(p2cccd::ValidateProposalOutput(proposal), "valid proposal");

  auto bad_proposal = proposal;
  bad_proposal.cost_score = -1.0f;
  ExpectError(p2cccd::ValidateProposalOutput(bad_proposal), "negative proposal cost");

  p2cccd::ExactWorkItem work;
  work.work_item_id = 10;
  work.parent_candidate_id = 1;
  work.query_id = 7;
  work.interval_t0 = 0.25;
  work.interval_t1 = 0.5;
  work.feature_family_mask = 1;
  work.priority_score = 0.4f;
  work.source = p2cccd::ProposalSource::kRaw;
  ExpectOk(p2cccd::ValidateExactWorkItem(work), "valid exact work item");

  auto bad_work = work;
  bad_work.interval_t0 = 0.75;
  bad_work.interval_t1 = 0.5;
  ExpectError(p2cccd::ValidateExactWorkItem(bad_work), "invalid exact interval");

  bad_work = work;
  bad_work.source = static_cast<p2cccd::ProposalSource>(99);
  ExpectError(p2cccd::ValidateExactWorkItem(bad_work), "invalid proposal source");

  p2cccd::CertificateResult certificate;
  certificate.work_item_id = 10;
  certificate.query_id = 7;
  certificate.status = p2cccd::CertificateStatus::kSeparation;
  certificate.interval_t0 = 0.25;
  certificate.interval_t1 = 0.5;
  certificate.toi_upper = 0.5;
  certificate.safe_margin_lb = 1.0e-3;
  certificate.covered_feature_mask = 1;
  certificate.eps_time = 1.0e-4;
  certificate.eps_space = 1.0e-6;
  ExpectOk(p2cccd::ValidateCertificateResult(certificate), "valid certificate");

  auto bad_certificate = certificate;
  bad_certificate.status = p2cccd::CertificateStatus::kUndecided;
  bad_certificate.reason_code = 0;
  bad_certificate.next_refinement_mode = p2cccd::CertificateRefinementMode::kBisectInterval;
  ExpectError(p2cccd::ValidateCertificateResult(bad_certificate), "undecided without reason");

  bad_certificate = certificate;
  bad_certificate.status = p2cccd::CertificateStatus::kUndecided;
  bad_certificate.reason_code = 1;
  bad_certificate.next_refinement_mode = p2cccd::CertificateRefinementMode::kNone;
  ExpectError(p2cccd::ValidateCertificateResult(bad_certificate),
              "undecided without refinement mode");

  bad_certificate = certificate;
  bad_certificate.status = p2cccd::CertificateStatus::kCollision;
  bad_certificate.witness_id_a = -1;
  bad_certificate.witness_id_b = 2;
  ExpectError(p2cccd::ValidateCertificateResult(bad_certificate),
              "collision without witnesses");

  bad_certificate = certificate;
  bad_certificate.status = static_cast<p2cccd::CertificateStatus>(99);
  ExpectError(p2cccd::ValidateCertificateResult(bad_certificate), "invalid certificate status");

  p2cccd::AuditLogRow audit;
  audit.event_id = 100;
  audit.query_id = 7;
  audit.candidate_id = 1;
  audit.work_item_id = 10;
  audit.stage = p2cccd::AuditStage::kRt;
  audit.interval_t0 = 0.0;
  audit.interval_t1 = 1.0;
  audit.timestamp_us = 1234;
  ExpectOk(p2cccd::ValidateAuditLogRow(audit), "valid audit row");

  auto bad_audit = audit;
  bad_audit.stage = static_cast<p2cccd::AuditStage>(99);
  ExpectError(p2cccd::ValidateAuditLogRow(bad_audit), "invalid audit stage");

  p2cccd::BenchmarkRow benchmark;
  benchmark.query_count = 100;
  benchmark.fn_count = 0;
  benchmark.fp_count = 3;
  benchmark.candidate_recall = 1.0;
  benchmark.avg_candidates = 4.5;
  benchmark.avg_exact_evals = 2.0;
  benchmark.avg_subdivision_depth = 1.2;
  benchmark.fallback_ratio = 0.05;
  benchmark.rt_ms = 1.0;
  benchmark.proposal_ms = 0.5;
  benchmark.exact_ms = 2.0;
  benchmark.total_ms = 3.5;
  benchmark.qps = 1000.0;
  ExpectOk(p2cccd::ValidateBenchmarkRow(benchmark), "valid benchmark row");

  auto bad_benchmark = benchmark;
  bad_benchmark.candidate_recall = 1.01;
  ExpectError(p2cccd::ValidateBenchmarkRow(bad_benchmark), "invalid recall");

  return g_failures == 0 ? 0 : 1;
}
