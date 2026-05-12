#include "certificate/certificate_engine.h"

#include "common/validators.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace p2cccd {
namespace {

using Vec3 = std::array<double, 3>;

constexpr double kNumericalEpsilon = 1.0e-14;

struct SampleDistance {
  double t = 0.0;
  double distance = 0.0;
};

Vec3 Add(Vec3 a, Vec3 b) {
  return {a[0] + b[0], a[1] + b[1], a[2] + b[2]};
}

Vec3 Sub(Vec3 a, Vec3 b) {
  return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}

Vec3 Scale(Vec3 a, double scale) {
  return {a[0] * scale, a[1] * scale, a[2] * scale};
}

double Dot(Vec3 a, Vec3 b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

Vec3 Cross(Vec3 a, Vec3 b) {
  return {
      a[1] * b[2] - a[2] * b[1],
      a[2] * b[0] - a[0] * b[2],
      a[0] * b[1] - a[1] * b[0],
  };
}

double SquaredNorm(Vec3 value) {
  return Dot(value, value);
}

double Norm(Vec3 value) {
  return std::sqrt(SquaredNorm(value));
}

bool IsFiniteVec3(Vec3 value) {
  return std::isfinite(value[0]) && std::isfinite(value[1]) && std::isfinite(value[2]);
}

bool IsFiniteTrajectory(const LinearVertexTrajectory& trajectory) {
  return IsFiniteVec3(trajectory.position_t0) && IsFiniteVec3(trajectory.position_t1);
}

Vec3 PositionAt(const LinearVertexTrajectory& trajectory, double t) {
  return Add(Scale(trajectory.position_t0, 1.0 - t), Scale(trajectory.position_t1, t));
}

double MaxDisplacementFromMidpoint(const LinearVertexTrajectory& trajectory,
                                   double interval_t0,
                                   double interval_t1) {
  const double interval_length = std::max(0.0, interval_t1 - interval_t0);
  return 0.5 * interval_length * Norm(Sub(trajectory.position_t1, trajectory.position_t0));
}

double DistancePointPoint(Vec3 a, Vec3 b) {
  return Norm(Sub(a, b));
}

double DistancePointSegment(Vec3 point, Vec3 a, Vec3 b) {
  const Vec3 ab = Sub(b, a);
  const double ab_squared = SquaredNorm(ab);
  if (ab_squared <= kNumericalEpsilon) {
    return DistancePointPoint(point, a);
  }
  const double t = std::clamp(Dot(Sub(point, a), ab) / ab_squared, 0.0, 1.0);
  return DistancePointPoint(point, Add(a, Scale(ab, t)));
}

double DistanceSegmentSegment(Vec3 p1, Vec3 q1, Vec3 p2, Vec3 q2) {
  const Vec3 d1 = Sub(q1, p1);
  const Vec3 d2 = Sub(q2, p2);
  const Vec3 r = Sub(p1, p2);
  const double a = SquaredNorm(d1);
  const double e = SquaredNorm(d2);
  const double f = Dot(d2, r);

  double s = 0.0;
  double t = 0.0;
  if (a <= kNumericalEpsilon && e <= kNumericalEpsilon) {
    return DistancePointPoint(p1, p2);
  }
  if (a <= kNumericalEpsilon) {
    t = std::clamp(f / e, 0.0, 1.0);
  } else {
    const double c = Dot(d1, r);
    if (e <= kNumericalEpsilon) {
      s = std::clamp(-c / a, 0.0, 1.0);
    } else {
      const double b = Dot(d1, d2);
      const double denom = a * e - b * b;
      if (denom != 0.0) {
        s = std::clamp((b * f - c * e) / denom, 0.0, 1.0);
      }
      t = (b * s + f) / e;
      if (t < 0.0) {
        t = 0.0;
        s = std::clamp(-c / a, 0.0, 1.0);
      } else if (t > 1.0) {
        t = 1.0;
        s = std::clamp((b - c) / a, 0.0, 1.0);
      }
    }
  }

  const Vec3 closest_a = Add(p1, Scale(d1, s));
  const Vec3 closest_b = Add(p2, Scale(d2, t));
  return DistancePointPoint(closest_a, closest_b);
}

double DistancePointTriangle(Vec3 point, Vec3 a, Vec3 b, Vec3 c) {
  const Vec3 ab = Sub(b, a);
  const Vec3 ac = Sub(c, a);
  const Vec3 normal = Cross(ab, ac);
  if (SquaredNorm(normal) <= kNumericalEpsilon) {
    return std::min({DistancePointSegment(point, a, b),
                     DistancePointSegment(point, b, c),
                     DistancePointSegment(point, c, a)});
  }

  const Vec3 ap = Sub(point, a);
  const double d1 = Dot(ab, ap);
  const double d2 = Dot(ac, ap);
  if (d1 <= 0.0 && d2 <= 0.0) {
    return DistancePointPoint(point, a);
  }

  const Vec3 bp = Sub(point, b);
  const double d3 = Dot(ab, bp);
  const double d4 = Dot(ac, bp);
  if (d3 >= 0.0 && d4 <= d3) {
    return DistancePointPoint(point, b);
  }

  const double vc = d1 * d4 - d3 * d2;
  if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
    const double v = d1 / (d1 - d3);
    return DistancePointPoint(point, Add(a, Scale(ab, v)));
  }

  const Vec3 cp = Sub(point, c);
  const double d5 = Dot(ab, cp);
  const double d6 = Dot(ac, cp);
  if (d6 >= 0.0 && d5 <= d6) {
    return DistancePointPoint(point, c);
  }

  const double vb = d5 * d2 - d1 * d6;
  if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
    const double w = d2 / (d2 - d6);
    return DistancePointPoint(point, Add(a, Scale(ac, w)));
  }

  const double va = d3 * d6 - d5 * d4;
  if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
    const double w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
    return DistancePointPoint(point, Add(b, Scale(Sub(c, b), w)));
  }

  const double signed_distance = Dot(ap, normal) / Norm(normal);
  return std::abs(signed_distance);
}

double PointTriangleDistanceAt(const PointTriangleIntervalPrimitive& primitive, double t) {
  return DistancePointTriangle(PositionAt(primitive.point, t),
                               PositionAt(primitive.triangle_v0, t),
                               PositionAt(primitive.triangle_v1, t),
                               PositionAt(primitive.triangle_v2, t));
}

double EdgeEdgeDistanceAt(const EdgeEdgeIntervalPrimitive& primitive, double t) {
  return DistanceSegmentSegment(PositionAt(primitive.edge_a0, t),
                                PositionAt(primitive.edge_a1, t),
                                PositionAt(primitive.edge_b0, t),
                                PositionAt(primitive.edge_b1, t));
}

double PointTriangleMotionRadius(const PointTriangleIntervalPrimitive& primitive,
                                 double interval_t0,
                                 double interval_t1) {
  const double point_displacement =
      MaxDisplacementFromMidpoint(primitive.point, interval_t0, interval_t1);
  const double triangle_displacement =
      std::max({MaxDisplacementFromMidpoint(primitive.triangle_v0, interval_t0, interval_t1),
                MaxDisplacementFromMidpoint(primitive.triangle_v1, interval_t0, interval_t1),
                MaxDisplacementFromMidpoint(primitive.triangle_v2, interval_t0, interval_t1)});
  return point_displacement + triangle_displacement;
}

double EdgeEdgeMotionRadius(const EdgeEdgeIntervalPrimitive& primitive,
                            double interval_t0,
                            double interval_t1) {
  const double edge_a_displacement =
      std::max(MaxDisplacementFromMidpoint(primitive.edge_a0, interval_t0, interval_t1),
               MaxDisplacementFromMidpoint(primitive.edge_a1, interval_t0, interval_t1));
  const double edge_b_displacement =
      std::max(MaxDisplacementFromMidpoint(primitive.edge_b0, interval_t0, interval_t1),
               MaxDisplacementFromMidpoint(primitive.edge_b1, interval_t0, interval_t1));
  return edge_a_displacement + edge_b_displacement;
}

Status ValidateConfig(const CertificateEngineConfig& config) {
  if (!std::isfinite(config.eps_time) || config.eps_time <= 0.0) {
    return Status::Error("CertificateEngineConfig.eps_time must be finite and positive");
  }
  if (!std::isfinite(config.eps_space) || config.eps_space <= 0.0) {
    return Status::Error("CertificateEngineConfig.eps_space must be finite and positive");
  }
  return Status::Ok();
}

Status ValidateInterval(double interval_t0, double interval_t1) {
  if (!std::isfinite(interval_t0) || !std::isfinite(interval_t1)) {
    return Status::Error("certificate interval endpoints must be finite");
  }
  if (interval_t0 < 0.0 || interval_t1 > 1.0 || interval_t0 > interval_t1) {
    return Status::Error("certificate interval must satisfy 0 <= t0 <= t1 <= 1");
  }
  return Status::Ok();
}

Status ValidatePointTrianglePrimitive(const PointTriangleIntervalPrimitive& primitive) {
  if (primitive.point_id < 0 || primitive.triangle_id < 0) {
    return Status::Error("point-triangle primitive ids are required");
  }
  if (!IsFiniteTrajectory(primitive.point) || !IsFiniteTrajectory(primitive.triangle_v0) ||
      !IsFiniteTrajectory(primitive.triangle_v1) || !IsFiniteTrajectory(primitive.triangle_v2)) {
    return Status::Error("point-triangle primitive trajectories must be finite");
  }
  return Status::Ok();
}

Status ValidateEdgeEdgePrimitive(const EdgeEdgeIntervalPrimitive& primitive) {
  if (primitive.edge_a_id < 0 || primitive.edge_b_id < 0) {
    return Status::Error("edge-edge primitive ids are required");
  }
  if (!IsFiniteTrajectory(primitive.edge_a0) || !IsFiniteTrajectory(primitive.edge_a1) ||
      !IsFiniteTrajectory(primitive.edge_b0) || !IsFiniteTrajectory(primitive.edge_b1)) {
    return Status::Error("edge-edge primitive trajectories must be finite");
  }
  return Status::Ok();
}

PrimitiveIntervalResult MakePrimitiveResult(CertificateStatus status,
                                            std::uint32_t family_mask,
                                            double interval_t0,
                                            double interval_t1) {
  PrimitiveIntervalResult result;
  result.status = status;
  result.covered_feature_mask = status == CertificateStatus::kSeparation ? family_mask : 0U;
  result.interval_t0 = interval_t0;
  result.interval_t1 = interval_t1;
  result.toi_upper = interval_t1;
  return result;
}

CertificateRefinementMode RefinementModeForReason(std::uint16_t reason_code) {
  switch (reason_code) {
    case kCertificateReasonMissingGeometry:
      return CertificateRefinementMode::kRequestGeometry;
    case kCertificateReasonMaxSubdivisionDepth:
      return CertificateRefinementMode::kBisectInterval;
    case kCertificateReasonInvalidInput:
      return CertificateRefinementMode::kRequestGeometry;
    case kCertificateReasonNone:
    default:
      return CertificateRefinementMode::kEscalatePrecision;
  }
}

void MarkUndecided(PrimitiveIntervalResult* result, std::uint16_t reason_code) {
  result->status = CertificateStatus::kUndecided;
  result->covered_feature_mask = 0U;
  result->reason_code = reason_code;
  result->next_refinement_mode = RefinementModeForReason(reason_code);
}

template <typename DistanceFn, typename LowerBoundFn>
PrimitiveIntervalResult EvaluateIntervalRecursive(double interval_t0,
                                                  double interval_t1,
                                                  const CertificateEngineConfig& config,
                                                  std::uint32_t family_mask,
                                                  std::uint8_t witness_family,
                                                  std::int64_t witness_id_a,
                                                  std::int64_t witness_id_b,
                                                  std::uint16_t depth,
                                                  DistanceFn distance_fn,
                                                  LowerBoundFn lower_bound_fn) {
  const double mid = 0.5 * (interval_t0 + interval_t1);
  const std::array<SampleDistance, 3> samples{{
      {interval_t0, distance_fn(interval_t0)},
      {mid, distance_fn(mid)},
      {interval_t1, distance_fn(interval_t1)},
  }};
  for (const SampleDistance& sample : samples) {
    if (sample.distance <= config.eps_space) {
      PrimitiveIntervalResult collision =
          MakePrimitiveResult(CertificateStatus::kCollision, family_mask, interval_t0, interval_t1);
      collision.toi_upper = sample.t;
      collision.witness_family = witness_family;
      collision.witness_id_a = witness_id_a;
      collision.witness_id_b = witness_id_b;
      collision.safe_margin_lb = 0.0;
      return collision;
    }
  }

  const double lower_bound = lower_bound_fn(interval_t0, interval_t1, mid);
  if (lower_bound > config.eps_space) {
    PrimitiveIntervalResult separation =
        MakePrimitiveResult(CertificateStatus::kSeparation, family_mask, interval_t0, interval_t1);
    separation.safe_margin_lb = std::max(0.0, lower_bound - config.eps_space);
    return separation;
  }

  if (depth >= config.max_subdivision_depth ||
      interval_t1 - interval_t0 <= config.eps_time) {
    PrimitiveIntervalResult undecided =
        MakePrimitiveResult(CertificateStatus::kUndecided, family_mask, interval_t0, interval_t1);
    MarkUndecided(&undecided, kCertificateReasonMaxSubdivisionDepth);
    return undecided;
  }

  PrimitiveIntervalResult left = EvaluateIntervalRecursive(interval_t0,
                                                           mid,
                                                           config,
                                                           family_mask,
                                                           witness_family,
                                                           witness_id_a,
                                                           witness_id_b,
                                                           static_cast<std::uint16_t>(depth + 1),
                                                           distance_fn,
                                                           lower_bound_fn);
  if (left.status == CertificateStatus::kCollision) {
    return left;
  }

  PrimitiveIntervalResult right = EvaluateIntervalRecursive(mid,
                                                            interval_t1,
                                                            config,
                                                            family_mask,
                                                            witness_family,
                                                            witness_id_a,
                                                            witness_id_b,
                                                            static_cast<std::uint16_t>(depth + 1),
                                                            distance_fn,
                                                            lower_bound_fn);
  if (right.status == CertificateStatus::kCollision) {
    return right;
  }

  if (left.status == CertificateStatus::kSeparation &&
      right.status == CertificateStatus::kSeparation) {
    PrimitiveIntervalResult separation =
        MakePrimitiveResult(CertificateStatus::kSeparation, family_mask, interval_t0, interval_t1);
    separation.safe_margin_lb = std::min(left.safe_margin_lb, right.safe_margin_lb);
    return separation;
  }

  PrimitiveIntervalResult undecided =
      MakePrimitiveResult(CertificateStatus::kUndecided, family_mask, interval_t0, interval_t1);
  MarkUndecided(&undecided, kCertificateReasonMaxSubdivisionDepth);
  return undecided;
}

CertificateResult MakeBaseCertificateResult(const ExactWorkItem& work_item,
                                            const CertificateEngineConfig& config) {
  CertificateResult result;
  result.work_item_id = work_item.work_item_id;
  result.query_id = work_item.query_id;
  result.interval_t0 = work_item.interval_t0;
  result.interval_t1 = work_item.interval_t1;
  result.toi_upper = work_item.interval_t1;
  result.eps_time = config.eps_time;
  result.eps_space = config.eps_space;
  result.status = CertificateStatus::kUndecided;
  result.reason_code = kCertificateReasonMissingGeometry;
  result.next_refinement_mode = CertificateRefinementMode::kRequestGeometry;
  return result;
}

void ApplyPrimitiveCollision(const PrimitiveIntervalResult& primitive_result,
                             CertificateResult* result) {
  result->status = CertificateStatus::kCollision;
  result->toi_upper = primitive_result.toi_upper;
  result->safe_margin_lb = 0.0;
  result->witness_family = primitive_result.witness_family;
  result->witness_id_a = primitive_result.witness_id_a;
  result->witness_id_b = primitive_result.witness_id_b;
  result->covered_feature_mask = 0U;
  result->reason_code = kCertificateReasonNone;
  result->next_refinement_mode = CertificateRefinementMode::kNone;
}

void ApplyUndecided(std::uint16_t reason_code, CertificateResult* result) {
  result->status = CertificateStatus::kUndecided;
  result->covered_feature_mask = 0U;
  result->safe_margin_lb = 0.0;
  result->reason_code = reason_code == kCertificateReasonNone ? kCertificateReasonMaxSubdivisionDepth
                                                               : reason_code;
  result->next_refinement_mode = RefinementModeForReason(result->reason_code);
}

CertificateResult MakeInvalidInputCertificate(const ExactWorkItem& work_item,
                                              const CertificateEngineConfig& config) {
  CertificateResult result = MakeBaseCertificateResult(work_item, config);
  ApplyUndecided(kCertificateReasonInvalidInput, &result);
  return result;
}

Status ValidateQueueConfig(const ExactWorkQueueConfig& config) {
  if (config.first_event_id == 0) {
    return Status::Error("ExactWorkQueueConfig.first_event_id must be non-zero");
  }
  if (config.first_timestamp_us == 0) {
    return Status::Error("ExactWorkQueueConfig.first_timestamp_us must be non-zero");
  }
  return Status::Ok();
}

std::uint16_t AuditActionForCertificate(const CertificateResult& result) {
  switch (result.status) {
    case CertificateStatus::kCollision:
      return kExactAuditCollision;
    case CertificateStatus::kSeparation:
      return kExactAuditSeparation;
    case CertificateStatus::kUndecided:
      return result.reason_code == kCertificateReasonInvalidInput ? kExactAuditInvalidInput
                                                                  : kExactAuditUndecided;
  }
  return kExactAuditUndecided;
}

AuditLogRow MakeAuditRow(std::uint64_t event_id,
                         std::uint64_t timestamp_us,
                         const ExactWorkItem& item,
                         AuditStage stage,
                         std::uint16_t action,
                         double aux_value0,
                         double aux_value1) {
  AuditLogRow row;
  row.event_id = event_id;
  row.query_id = item.query_id;
  row.candidate_id = item.parent_candidate_id;
  row.work_item_id = item.work_item_id;
  row.stage = stage;
  row.action = action;
  row.depth = item.depth;
  row.interval_t0 = item.interval_t0;
  row.interval_t1 = item.interval_t1;
  row.timestamp_us = timestamp_us;
  row.aux_value0 = aux_value0;
  row.aux_value1 = aux_value1;
  return row;
}

}  // namespace

Status EvaluatePointTriangleInterval(const PointTriangleIntervalPrimitive& primitive,
                                     double interval_t0,
                                     double interval_t1,
                                     const CertificateEngineConfig& config,
                                     PrimitiveIntervalResult* result) {
  if (result == nullptr) {
    return Status::Error("point-triangle interval result pointer is null");
  }
  if (auto status = ValidateConfig(config); !status.ok) {
    return status;
  }
  if (auto status = ValidateInterval(interval_t0, interval_t1); !status.ok) {
    return status;
  }
  if (auto status = ValidatePointTrianglePrimitive(primitive); !status.ok) {
    return status;
  }

  const auto distance_fn = [&](double t) { return PointTriangleDistanceAt(primitive, t); };
  const auto lower_bound_fn = [&](double t0, double t1, double mid) {
    return distance_fn(mid) - PointTriangleMotionRadius(primitive, t0, t1);
  };

  *result = EvaluateIntervalRecursive(interval_t0,
                                      interval_t1,
                                      config,
                                      kFeatureFamilyPointTriangle,
                                      static_cast<std::uint8_t>(kFeatureFamilyPointTriangle),
                                      primitive.point_id,
                                      primitive.triangle_id,
                                      0,
                                      distance_fn,
                                      lower_bound_fn);
  return Status::Ok();
}

Status EvaluateEdgeEdgeInterval(const EdgeEdgeIntervalPrimitive& primitive,
                                double interval_t0,
                                double interval_t1,
                                const CertificateEngineConfig& config,
                                PrimitiveIntervalResult* result) {
  if (result == nullptr) {
    return Status::Error("edge-edge interval result pointer is null");
  }
  if (auto status = ValidateConfig(config); !status.ok) {
    return status;
  }
  if (auto status = ValidateInterval(interval_t0, interval_t1); !status.ok) {
    return status;
  }
  if (auto status = ValidateEdgeEdgePrimitive(primitive); !status.ok) {
    return status;
  }

  const auto distance_fn = [&](double t) { return EdgeEdgeDistanceAt(primitive, t); };
  const auto lower_bound_fn = [&](double t0, double t1, double mid) {
    return distance_fn(mid) - EdgeEdgeMotionRadius(primitive, t0, t1);
  };

  *result = EvaluateIntervalRecursive(interval_t0,
                                      interval_t1,
                                      config,
                                      kFeatureFamilyEdgeEdge,
                                      static_cast<std::uint8_t>(kFeatureFamilyEdgeEdge),
                                      primitive.edge_a_id,
                                      primitive.edge_b_id,
                                      0,
                                      distance_fn,
                                      lower_bound_fn);
  return Status::Ok();
}

Status ValidateExactWorkQueueCoverage(const std::vector<ExactCertificateQuery>& work_queue,
                                      const ExactWorkQueueResult& result) {
  if (result.processed_count != work_queue.size()) {
    return Status::Error("exact work queue processed_count must match input size");
  }
  if (result.certificates.size() != work_queue.size()) {
    return Status::Error("each exact work item must produce exactly one certificate");
  }

  std::set<std::pair<std::uint64_t, std::uint64_t>> expected_items;
  for (const ExactCertificateQuery& query : work_queue) {
    if (auto status = ValidateExactWorkItem(query.work_item); !status.ok) {
      return status;
    }
    const auto key = std::make_pair(query.work_item.query_id, query.work_item.work_item_id);
    if (!expected_items.insert(key).second) {
      return Status::Error("exact work queue contains duplicate work_item_id for a query");
    }
  }

  std::set<std::pair<std::uint64_t, std::uint64_t>> covered_items;
  for (const CertificateResult& certificate : result.certificates) {
    if (auto status = ValidateCertificateResult(certificate); !status.ok) {
      return status;
    }
    const auto key = std::make_pair(certificate.query_id, certificate.work_item_id);
    if (!expected_items.contains(key)) {
      return Status::Error("certificate references a work item that was not queued");
    }
    if (!covered_items.insert(key).second) {
      return Status::Error("exact work item produced more than one certificate");
    }
  }

  if (covered_items.size() != expected_items.size()) {
    return Status::Error("at least one exact work item disappeared without certificate coverage");
  }
  return Status::Ok();
}

Status ProcessExactWorkQueueCpu(const std::vector<ExactCertificateQuery>& work_queue,
                                const ExactWorkQueueConfig& config,
                                ExactWorkQueueResult* result) {
  if (result == nullptr) {
    return Status::Error("exact work queue result output pointer is null");
  }
  if (auto status = ValidateQueueConfig(config); !status.ok) {
    return status;
  }

  ExactWorkQueueResult processed;
  processed.certificates.reserve(work_queue.size());
  processed.audit_log.reserve(work_queue.size() * (config.emit_dequeue_events ? 2U : 1U));

  CertificateEngine engine;
  std::uint64_t next_event_id = config.first_event_id;
  std::uint64_t next_timestamp_us = config.first_timestamp_us;
  for (const ExactCertificateQuery& query : work_queue) {
    if (auto status = ValidateExactWorkItem(query.work_item); !status.ok) {
      return status;
    }

    if (config.emit_dequeue_events) {
      AuditLogRow dequeue = MakeAuditRow(next_event_id++,
                                         next_timestamp_us++,
                                         query.work_item,
                                         AuditStage::kExact,
                                         kExactAuditDequeued,
                                         query.work_item.priority_score,
                                         static_cast<double>(query.work_item.feature_family_mask));
      if (auto status = ValidateAuditLogRow(dequeue); !status.ok) {
        return status;
      }
      processed.audit_log.push_back(dequeue);
    }

    CertificateResult certificate;
    if (auto status = engine.Evaluate(query, &certificate); !status.ok) {
      certificate = MakeInvalidInputCertificate(query.work_item, query.config);
    }
    if (auto status = ValidateCertificateResult(certificate); !status.ok) {
      return status;
    }

    AuditLogRow certify = MakeAuditRow(
        next_event_id++,
        next_timestamp_us++,
        query.work_item,
        AuditStage::kCertify,
        AuditActionForCertificate(certificate),
        certificate.status == CertificateStatus::kSeparation ? certificate.safe_margin_lb
                                                             : certificate.toi_upper,
        static_cast<double>(certificate.next_refinement_mode));
    if (auto status = ValidateAuditLogRow(certify); !status.ok) {
      return status;
    }

    processed.certificates.push_back(certificate);
    processed.audit_log.push_back(certify);
    ++processed.processed_count;
  }

  if (auto status = ValidateExactWorkQueueCoverage(work_queue, processed); !status.ok) {
    return status;
  }

  *result = std::move(processed);
  return Status::Ok();
}

Status GenerateConservativeRefinementWorkItems(const ExactWorkItem& parent,
                                               const CertificateResult& certificate,
                                               const ExactRefinementConfig& config,
                                               std::vector<ExactWorkItem>* children) {
  if (children == nullptr) {
    return Status::Error("refinement child output pointer is null");
  }
  children->clear();
  if (auto status = ValidateExactWorkItem(parent); !status.ok) {
    return status;
  }
  if (auto status = ValidateCertificateResult(certificate); !status.ok) {
    return status;
  }
  if (certificate.work_item_id != parent.work_item_id || certificate.query_id != parent.query_id) {
    return Status::Error("certificate must correspond to the parent exact work item");
  }
  if (config.first_child_work_item_id == 0) {
    return Status::Error("ExactRefinementConfig.first_child_work_item_id must be non-zero");
  }
  if (!std::isfinite(config.min_interval_width) || config.min_interval_width <= 0.0) {
    return Status::Error("ExactRefinementConfig.min_interval_width must be finite and positive");
  }
  if (parent.depth >= config.max_child_depth) {
    return Status::Ok();
  }

  const auto make_child = [&](std::uint64_t work_item_id, double t0, double t1) {
    ExactWorkItem child = parent;
    child.work_item_id = work_item_id;
    child.interval_t0 = t0;
    child.interval_t1 = t1;
    child.depth = static_cast<std::uint16_t>(parent.depth + 1);
    child.source = ProposalSource::kRefined;
    return child;
  };

  if (certificate.status == CertificateStatus::kCollision) {
    const double t0 = parent.interval_t0;
    const double t1 = std::clamp(certificate.toi_upper, parent.interval_t0, parent.interval_t1);
    if (t1 - t0 > config.min_interval_width) {
      children->push_back(make_child(config.first_child_work_item_id, t0, t1));
    }
    return Status::Ok();
  }

  if (certificate.status == CertificateStatus::kUndecided &&
      certificate.next_refinement_mode == CertificateRefinementMode::kBisectInterval) {
    if (parent.interval_t1 - parent.interval_t0 > config.min_interval_width) {
      const double mid = 0.5 * (parent.interval_t0 + parent.interval_t1);
      children->push_back(make_child(config.first_child_work_item_id, parent.interval_t0, mid));
      children->push_back(make_child(config.first_child_work_item_id + 1, mid, parent.interval_t1));
    }
    return Status::Ok();
  }

  return Status::Ok();
}

CertificateResult CertificateEngine::Evaluate(const ExactWorkItem& work_item) const {
  CertificateEngineConfig config;
  CertificateResult result = MakeBaseCertificateResult(work_item, config);
  result.reason_code = kCertificateReasonMissingGeometry;
  return result;
}

Status CertificateEngine::Evaluate(const ExactCertificateQuery& query,
                                   CertificateResult* result) const {
  if (result == nullptr) {
    return Status::Error("certificate result output pointer is null");
  }
  if (auto status = ValidateConfig(query.config); !status.ok) {
    return status;
  }
  if (auto status = ValidateExactWorkItem(query.work_item); !status.ok) {
    return status;
  }

  CertificateResult certificate = MakeBaseCertificateResult(query.work_item, query.config);
  double safe_margin_lb = std::numeric_limits<double>::infinity();
  std::uint32_t covered_feature_mask = 0U;
  bool evaluated_any_feature = false;
  bool has_undecided = false;
  std::uint16_t undecided_reason = kCertificateReasonNone;

  if ((query.work_item.feature_family_mask & kFeatureFamilyPointTriangle) != 0U) {
    if (query.point_triangle_primitives.empty()) {
      has_undecided = true;
      undecided_reason = kCertificateReasonMissingGeometry;
    }
    for (const PointTriangleIntervalPrimitive& primitive : query.point_triangle_primitives) {
      evaluated_any_feature = true;
      PrimitiveIntervalResult primitive_result;
      if (auto status = EvaluatePointTriangleInterval(primitive,
                                                      query.work_item.interval_t0,
                                                      query.work_item.interval_t1,
                                                      query.config,
                                                      &primitive_result);
          !status.ok) {
        return status;
      }
      if (primitive_result.status == CertificateStatus::kCollision) {
        ApplyPrimitiveCollision(primitive_result, &certificate);
        *result = certificate;
        return Status::Ok();
      }
      if (primitive_result.status == CertificateStatus::kSeparation) {
        covered_feature_mask |= primitive_result.covered_feature_mask;
        safe_margin_lb = std::min(safe_margin_lb, primitive_result.safe_margin_lb);
      } else {
        has_undecided = true;
        undecided_reason = primitive_result.reason_code;
      }
    }
  }

  if ((query.work_item.feature_family_mask & kFeatureFamilyEdgeEdge) != 0U) {
    if (query.edge_edge_primitives.empty()) {
      has_undecided = true;
      undecided_reason = kCertificateReasonMissingGeometry;
    }
    for (const EdgeEdgeIntervalPrimitive& primitive : query.edge_edge_primitives) {
      evaluated_any_feature = true;
      PrimitiveIntervalResult primitive_result;
      if (auto status = EvaluateEdgeEdgeInterval(primitive,
                                                 query.work_item.interval_t0,
                                                 query.work_item.interval_t1,
                                                 query.config,
                                                 &primitive_result);
          !status.ok) {
        return status;
      }
      if (primitive_result.status == CertificateStatus::kCollision) {
        ApplyPrimitiveCollision(primitive_result, &certificate);
        *result = certificate;
        return Status::Ok();
      }
      if (primitive_result.status == CertificateStatus::kSeparation) {
        covered_feature_mask |= primitive_result.covered_feature_mask;
        safe_margin_lb = std::min(safe_margin_lb, primitive_result.safe_margin_lb);
      } else {
        has_undecided = true;
        undecided_reason = primitive_result.reason_code;
      }
    }
  }

  if (!evaluated_any_feature) {
    ApplyUndecided(kCertificateReasonMissingGeometry, &certificate);
  } else if (has_undecided) {
    ApplyUndecided(undecided_reason, &certificate);
  } else {
    certificate.status = CertificateStatus::kSeparation;
    certificate.safe_margin_lb = std::isfinite(safe_margin_lb) ? safe_margin_lb : 0.0;
    certificate.covered_feature_mask = covered_feature_mask;
    certificate.reason_code = kCertificateReasonNone;
    certificate.next_refinement_mode = CertificateRefinementMode::kNone;
  }

  *result = certificate;
  return Status::Ok();
}

}  // namespace p2cccd
