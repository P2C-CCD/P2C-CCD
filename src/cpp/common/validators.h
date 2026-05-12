#pragma once

#include "common/runtime_contracts.h"
#include "common/status.h"

namespace p2cccd {

Status ValidateCandidateRecord(const CandidateRecord& record);
Status ValidateProposalOutput(const ProposalOutput& output);
Status ValidateExactWorkItem(const ExactWorkItem& item);
Status ValidateCertificateResult(const CertificateResult& result);
Status ValidateAuditLogRow(const AuditLogRow& row);
Status ValidateBenchmarkRow(const BenchmarkRow& row);

}  // namespace p2cccd
