#pragma once

#include "common/runtime_contracts.h"
#include "common/status.h"

#include <array>
#include <cstdint>
#include <vector>

namespace p2cccd {

enum ExactFeatureFamilyMask : std::uint32_t {
  kFeatureFamilyPointTriangle = 1U << 0U,
  kFeatureFamilyEdgeEdge = 1U << 1U,
};

enum CertificateReasonCode : std::uint16_t {
  kCertificateReasonNone = 0,
  kCertificateReasonMissingGeometry = 1,
  kCertificateReasonMaxSubdivisionDepth = 2,
  kCertificateReasonInvalidInput = 3,
};

enum ExactAuditAction : std::uint16_t {
  kExactAuditDequeued = 1,
  kExactAuditCollision = 2,
  kExactAuditSeparation = 3,
  kExactAuditUndecided = 4,
  kExactAuditInvalidInput = 5,
};

struct LinearVertexTrajectory {
  std::int64_t feature_id = -1;
  std::array<double, 3> position_t0{0.0, 0.0, 0.0};
  std::array<double, 3> position_t1{0.0, 0.0, 0.0};
};

struct PointTriangleIntervalPrimitive {
  std::int64_t point_id = -1;
  std::int64_t triangle_id = -1;
  LinearVertexTrajectory point;
  LinearVertexTrajectory triangle_v0;
  LinearVertexTrajectory triangle_v1;
  LinearVertexTrajectory triangle_v2;
};

struct EdgeEdgeIntervalPrimitive {
  std::int64_t edge_a_id = -1;
  std::int64_t edge_b_id = -1;
  LinearVertexTrajectory edge_a0;
  LinearVertexTrajectory edge_a1;
  LinearVertexTrajectory edge_b0;
  LinearVertexTrajectory edge_b1;
};

struct CertificateEngineConfig {
  double eps_time = 1.0e-4;
  double eps_space = 1.0e-6;
  std::uint16_t max_subdivision_depth = 32;
};

struct ExactCertificateQuery {
  ExactWorkItem work_item;
  CertificateEngineConfig config;
  std::vector<PointTriangleIntervalPrimitive> point_triangle_primitives;
  std::vector<EdgeEdgeIntervalPrimitive> edge_edge_primitives;
};

struct PrimitiveIntervalResult {
  CertificateStatus status = CertificateStatus::kUndecided;
  std::uint32_t covered_feature_mask = 0;
  double interval_t0 = 0.0;
  double interval_t1 = 1.0;
  double toi_upper = 1.0;
  double safe_margin_lb = 0.0;
  std::uint8_t witness_family = 0;
  std::int64_t witness_id_a = -1;
  std::int64_t witness_id_b = -1;
  std::uint16_t reason_code = kCertificateReasonNone;
  CertificateRefinementMode next_refinement_mode = CertificateRefinementMode::kNone;
};

struct ExactWorkQueueConfig {
  std::uint64_t first_event_id = 1;
  std::uint64_t first_timestamp_us = 1;
  bool emit_dequeue_events = true;
};

struct ExactWorkQueueResult {
  std::vector<CertificateResult> certificates;
  std::vector<AuditLogRow> audit_log;
  std::uint64_t processed_count = 0;
};

struct ExactRefinementConfig {
  std::uint64_t first_child_work_item_id = 1;
  std::uint16_t max_child_depth = 64;
  double min_interval_width = 1.0e-4;
};

Status EvaluatePointTriangleInterval(const PointTriangleIntervalPrimitive& primitive,
                                     double interval_t0,
                                     double interval_t1,
                                     const CertificateEngineConfig& config,
                                     PrimitiveIntervalResult* result);
Status EvaluateEdgeEdgeInterval(const EdgeEdgeIntervalPrimitive& primitive,
                                double interval_t0,
                                double interval_t1,
                                const CertificateEngineConfig& config,
                                PrimitiveIntervalResult* result);
Status ProcessExactWorkQueueCpu(const std::vector<ExactCertificateQuery>& work_queue,
                                const ExactWorkQueueConfig& config,
                                ExactWorkQueueResult* result);
Status ValidateExactWorkQueueCoverage(const std::vector<ExactCertificateQuery>& work_queue,
                                      const ExactWorkQueueResult& result);
Status GenerateConservativeRefinementWorkItems(const ExactWorkItem& parent,
                                               const CertificateResult& certificate,
                                               const ExactRefinementConfig& config,
                                               std::vector<ExactWorkItem>* children);

class CertificateEngine {
 public:
  CertificateEngine() = default;

  CertificateResult Evaluate(const ExactWorkItem& work_item) const;
  Status Evaluate(const ExactCertificateQuery& query, CertificateResult* result) const;
};

}  // namespace p2cccd
