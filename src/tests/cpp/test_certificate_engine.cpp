#include "certificate/certificate_engine.h"
#include "common/validators.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

namespace {

int g_failures = 0;

using Vec3 = std::array<double, 3>;

void Expect(bool condition, const char* label) {
  if (!condition) {
    std::cerr << "FAIL " << label << '\n';
    ++g_failures;
  }
}

void ExpectNear(double lhs, double rhs, double tolerance, const char* label) {
  if (std::abs(lhs - rhs) > tolerance) {
    std::cerr << "FAIL " << label << ": expected " << rhs << " got " << lhs << '\n';
    ++g_failures;
  }
}

void ExpectOk(const p2cccd::Status& status, const char* label) {
  if (!status.ok) {
    std::cerr << "FAIL " << label << ": " << status.message << '\n';
    ++g_failures;
  }
}

p2cccd::LinearVertexTrajectory Vertex(std::int64_t id, Vec3 p0, Vec3 p1) {
  p2cccd::LinearVertexTrajectory trajectory;
  trajectory.feature_id = id;
  trajectory.position_t0 = p0;
  trajectory.position_t1 = p1;
  return trajectory;
}

p2cccd::PointTriangleIntervalPrimitive MovingPointStaticTriangle(double z0,
                                                                 double z1,
                                                                 std::int64_t point_id = 10,
                                                                 std::int64_t triangle_id = 20) {
  p2cccd::PointTriangleIntervalPrimitive primitive;
  primitive.point_id = point_id;
  primitive.triangle_id = triangle_id;
  primitive.point = Vertex(point_id, {0.25, 0.25, z0}, {0.25, 0.25, z1});
  primitive.triangle_v0 = Vertex(101, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0});
  primitive.triangle_v1 = Vertex(102, {1.0, 0.0, 0.0}, {1.0, 0.0, 0.0});
  primitive.triangle_v2 = Vertex(103, {0.0, 1.0, 0.0}, {0.0, 1.0, 0.0});
  return primitive;
}

p2cccd::EdgeEdgeIntervalPrimitive MovingEdgeThroughStaticEdge(double z0,
                                                              double z1,
                                                              std::int64_t edge_a_id = 30,
                                                              std::int64_t edge_b_id = 40) {
  p2cccd::EdgeEdgeIntervalPrimitive primitive;
  primitive.edge_a_id = edge_a_id;
  primitive.edge_b_id = edge_b_id;
  primitive.edge_a0 = Vertex(201, {-1.0, 0.0, 0.0}, {-1.0, 0.0, 0.0});
  primitive.edge_a1 = Vertex(202, {1.0, 0.0, 0.0}, {1.0, 0.0, 0.0});
  primitive.edge_b0 = Vertex(301, {0.0, -1.0, z0}, {0.0, -1.0, z1});
  primitive.edge_b1 = Vertex(302, {0.0, 1.0, z0}, {0.0, 1.0, z1});
  return primitive;
}

p2cccd::ExactWorkItem WorkItem(std::uint32_t feature_family_mask) {
  p2cccd::ExactWorkItem item;
  item.work_item_id = 9001;
  item.parent_candidate_id = 77;
  item.query_id = 1234;
  item.slab_id = 0;
  item.patch_a_id = 1;
  item.patch_b_id = 2;
  item.interval_t0 = 0.0;
  item.interval_t1 = 1.0;
  item.feature_family_mask = feature_family_mask;
  item.priority_score = 1.0F;
  item.source = p2cccd::ProposalSource::kRaw;
  return item;
}

p2cccd::CertificateEngineConfig Config() {
  p2cccd::CertificateEngineConfig config;
  config.eps_time = 1.0e-5;
  config.eps_space = 1.0e-6;
  config.max_subdivision_depth = 32;
  return config;
}

void TestPointTriangleOracleCollisionUsesSubdivision() {
  p2cccd::PrimitiveIntervalResult result;
  ExpectOk(p2cccd::EvaluatePointTriangleInterval(MovingPointStaticTriangle(1.0, -3.0),
                                                 0.0,
                                                 1.0,
                                                 Config(),
                                                 &result),
           "point-triangle collision oracle");
  Expect(result.status == p2cccd::CertificateStatus::kCollision,
         "point-triangle collision status");
  ExpectNear(result.toi_upper, 0.25, 1.0e-12, "point-triangle recursive toi");
  Expect(result.witness_family == p2cccd::kFeatureFamilyPointTriangle,
         "point-triangle witness family");
  Expect(result.witness_id_a == 10, "point-triangle point witness id");
  Expect(result.witness_id_b == 20, "point-triangle triangle witness id");
}

void TestPointTriangleOracleSeparation() {
  p2cccd::PrimitiveIntervalResult result;
  ExpectOk(p2cccd::EvaluatePointTriangleInterval(MovingPointStaticTriangle(2.0, 2.0),
                                                 0.0,
                                                 1.0,
                                                 Config(),
                                                 &result),
           "point-triangle separation oracle");
  Expect(result.status == p2cccd::CertificateStatus::kSeparation,
         "point-triangle separation status");
  Expect(result.covered_feature_mask == p2cccd::kFeatureFamilyPointTriangle,
         "point-triangle separation mask");
  Expect(result.safe_margin_lb > 1.9, "point-triangle safe margin");
}

void TestEdgeEdgeOracleCollisionUsesSubdivision() {
  p2cccd::PrimitiveIntervalResult result;
  ExpectOk(p2cccd::EvaluateEdgeEdgeInterval(MovingEdgeThroughStaticEdge(1.0, -3.0),
                                            0.0,
                                            1.0,
                                            Config(),
                                            &result),
           "edge-edge collision oracle");
  Expect(result.status == p2cccd::CertificateStatus::kCollision,
         "edge-edge collision status");
  ExpectNear(result.toi_upper, 0.25, 1.0e-12, "edge-edge recursive toi");
  Expect(result.witness_family == p2cccd::kFeatureFamilyEdgeEdge,
         "edge-edge witness family");
  Expect(result.witness_id_a == 30, "edge-edge first witness id");
  Expect(result.witness_id_b == 40, "edge-edge second witness id");
}

void TestEdgeEdgeOracleSeparation() {
  p2cccd::PrimitiveIntervalResult result;
  ExpectOk(p2cccd::EvaluateEdgeEdgeInterval(MovingEdgeThroughStaticEdge(2.0, 2.0),
                                            0.0,
                                            1.0,
                                            Config(),
                                            &result),
           "edge-edge separation oracle");
  Expect(result.status == p2cccd::CertificateStatus::kSeparation,
         "edge-edge separation status");
  Expect(result.covered_feature_mask == p2cccd::kFeatureFamilyEdgeEdge,
         "edge-edge separation mask");
  Expect(result.safe_margin_lb > 1.9, "edge-edge safe margin");
}

void TestCertificateEngineCollisionOutput() {
  p2cccd::ExactCertificateQuery query;
  query.work_item = WorkItem(p2cccd::kFeatureFamilyPointTriangle);
  query.config = Config();
  query.point_triangle_primitives = {MovingPointStaticTriangle(1.0, -3.0)};

  p2cccd::CertificateResult result;
  p2cccd::CertificateEngine engine;
  ExpectOk(engine.Evaluate(query, &result), "engine collision evaluate");
  ExpectOk(p2cccd::ValidateCertificateResult(result), "engine collision validates");
  Expect(result.status == p2cccd::CertificateStatus::kCollision, "engine collision status");
  ExpectNear(result.toi_upper, 0.25, 1.0e-12, "engine collision toi upper");
  Expect(result.witness_family == p2cccd::kFeatureFamilyPointTriangle,
         "engine collision witness family");
  Expect(result.witness_id_a == 10 && result.witness_id_b == 20,
         "engine collision witness ids");
  Expect(result.next_refinement_mode == p2cccd::CertificateRefinementMode::kNone,
         "engine collision has no refinement");
}

void TestCertificateEngineSeparationOutput() {
  p2cccd::ExactCertificateQuery query;
  query.work_item =
      WorkItem(p2cccd::kFeatureFamilyPointTriangle | p2cccd::kFeatureFamilyEdgeEdge);
  query.config = Config();
  query.point_triangle_primitives = {MovingPointStaticTriangle(2.0, 2.0)};
  query.edge_edge_primitives = {MovingEdgeThroughStaticEdge(2.0, 2.0)};

  p2cccd::CertificateResult result;
  p2cccd::CertificateEngine engine;
  ExpectOk(engine.Evaluate(query, &result), "engine separation evaluate");
  ExpectOk(p2cccd::ValidateCertificateResult(result), "engine separation validates");
  Expect(result.status == p2cccd::CertificateStatus::kSeparation,
         "engine separation status");
  Expect(result.covered_feature_mask ==
             (p2cccd::kFeatureFamilyPointTriangle | p2cccd::kFeatureFamilyEdgeEdge),
         "engine separation covered mask");
  Expect(result.safe_margin_lb > 1.9, "engine separation safe margin");
  Expect(result.next_refinement_mode == p2cccd::CertificateRefinementMode::kNone,
         "engine separation has no refinement");
}

void TestCertificateEngineUndecidedOutput() {
  p2cccd::ExactCertificateQuery query;
  query.work_item = WorkItem(p2cccd::kFeatureFamilyPointTriangle);
  query.config = Config();

  p2cccd::CertificateResult result;
  p2cccd::CertificateEngine engine;
  ExpectOk(engine.Evaluate(query, &result), "engine undecided evaluate");
  ExpectOk(p2cccd::ValidateCertificateResult(result), "engine undecided validates");
  Expect(result.status == p2cccd::CertificateStatus::kUndecided, "engine undecided status");
  Expect(result.reason_code == p2cccd::kCertificateReasonMissingGeometry,
         "engine undecided reason");
  Expect(result.next_refinement_mode == p2cccd::CertificateRefinementMode::kRequestGeometry,
         "engine undecided refinement mode");
}

void TestLegacyInterfaceStaysUndecided() {
  p2cccd::CertificateEngine engine;
  const p2cccd::CertificateResult result = engine.Evaluate(WorkItem(p2cccd::kFeatureFamilyPointTriangle));
  ExpectOk(p2cccd::ValidateCertificateResult(result), "legacy result validates");
  Expect(result.status == p2cccd::CertificateStatus::kUndecided,
         "legacy interface remains undecided");
  Expect(result.next_refinement_mode == p2cccd::CertificateRefinementMode::kRequestGeometry,
         "legacy interface requests geometry");
}

void TestExactWorkQueueProcessingAuditAndCoverage() {
  p2cccd::ExactCertificateQuery collision_query;
  collision_query.work_item = WorkItem(p2cccd::kFeatureFamilyPointTriangle);
  collision_query.config = Config();
  collision_query.point_triangle_primitives = {MovingPointStaticTriangle(1.0, -3.0)};

  p2cccd::ExactCertificateQuery separation_query;
  separation_query.work_item = WorkItem(p2cccd::kFeatureFamilyEdgeEdge);
  separation_query.work_item.work_item_id = 9002;
  separation_query.config = Config();
  separation_query.edge_edge_primitives = {MovingEdgeThroughStaticEdge(2.0, 2.0)};

  p2cccd::ExactCertificateQuery undecided_query;
  undecided_query.work_item = WorkItem(p2cccd::kFeatureFamilyPointTriangle);
  undecided_query.work_item.work_item_id = 9003;
  undecided_query.config = Config();

  const std::vector<p2cccd::ExactCertificateQuery> queue{
      collision_query,
      separation_query,
      undecided_query,
  };
  p2cccd::ExactWorkQueueConfig queue_config;
  queue_config.first_event_id = 500;
  queue_config.first_timestamp_us = 800;

  p2cccd::ExactWorkQueueResult result;
  ExpectOk(p2cccd::ProcessExactWorkQueueCpu(queue, queue_config, &result),
           "process exact work queue");
  Expect(result.processed_count == queue.size(), "queue processed count");
  Expect(result.certificates.size() == queue.size(), "queue certificate count");
  Expect(result.audit_log.size() == queue.size() * 2, "queue audit count");
  ExpectOk(p2cccd::ValidateExactWorkQueueCoverage(queue, result), "queue coverage guard");
  Expect(result.certificates[0].status == p2cccd::CertificateStatus::kCollision,
         "queue collision branch");
  Expect(result.certificates[1].status == p2cccd::CertificateStatus::kSeparation,
         "queue separation branch");
  Expect(result.certificates[2].status == p2cccd::CertificateStatus::kUndecided,
         "queue undecided branch");
  Expect(result.certificates[2].next_refinement_mode ==
             p2cccd::CertificateRefinementMode::kRequestGeometry,
         "queue undecided refinement mode");
  Expect(result.audit_log.front().action == p2cccd::kExactAuditDequeued,
         "queue dequeue audit action");
  Expect(result.audit_log[1].action == p2cccd::kExactAuditCollision,
         "queue collision audit action");
}

void TestExactWorkQueueCoverageRejectsDroppedItem() {
  p2cccd::ExactCertificateQuery query;
  query.work_item = WorkItem(p2cccd::kFeatureFamilyPointTriangle);
  query.config = Config();
  query.point_triangle_primitives = {MovingPointStaticTriangle(1.0, -3.0)};

  p2cccd::ExactWorkQueueResult incomplete;
  incomplete.processed_count = 1;
  Expect(!p2cccd::ValidateExactWorkQueueCoverage({query}, incomplete).ok,
         "coverage guard rejects missing certificate");
}

void TestConservativeRefinementHeuristics() {
  p2cccd::ExactWorkItem parent = WorkItem(p2cccd::kFeatureFamilyPointTriangle);
  parent.interval_t0 = 0.0;
  parent.interval_t1 = 1.0;

  p2cccd::CertificateResult collision;
  collision.work_item_id = parent.work_item_id;
  collision.query_id = parent.query_id;
  collision.status = p2cccd::CertificateStatus::kCollision;
  collision.interval_t0 = parent.interval_t0;
  collision.interval_t1 = parent.interval_t1;
  collision.toi_upper = 0.25;
  collision.witness_family = p2cccd::kFeatureFamilyPointTriangle;
  collision.witness_id_a = 10;
  collision.witness_id_b = 20;

  p2cccd::ExactRefinementConfig config;
  config.first_child_work_item_id = 10001;
  config.min_interval_width = 1.0e-5;
  std::vector<p2cccd::ExactWorkItem> children;
  ExpectOk(p2cccd::GenerateConservativeRefinementWorkItems(parent,
                                                           collision,
                                                           config,
                                                           &children),
           "collision TOI refinement");
  Expect(children.size() == 1, "collision refinement child count");
  ExpectNear(children.front().interval_t0, 0.0, 0.0, "collision child interval start");
  ExpectNear(children.front().interval_t1, 0.25, 0.0, "collision child interval end");
  Expect(children.front().source == p2cccd::ProposalSource::kRefined,
         "collision child source");

  p2cccd::CertificateResult undecided;
  undecided.work_item_id = parent.work_item_id;
  undecided.query_id = parent.query_id;
  undecided.status = p2cccd::CertificateStatus::kUndecided;
  undecided.interval_t0 = parent.interval_t0;
  undecided.interval_t1 = parent.interval_t1;
  undecided.toi_upper = parent.interval_t1;
  undecided.reason_code = p2cccd::kCertificateReasonMaxSubdivisionDepth;
  undecided.next_refinement_mode = p2cccd::CertificateRefinementMode::kBisectInterval;
  ExpectOk(p2cccd::GenerateConservativeRefinementWorkItems(parent,
                                                           undecided,
                                                           config,
                                                           &children),
           "undecided interval bisection refinement");
  Expect(children.size() == 2, "undecided refinement child count");
  ExpectNear(children[0].interval_t1, 0.5, 0.0, "undecided first child end");
  ExpectNear(children[1].interval_t0, 0.5, 0.0, "undecided second child start");
}

}  // namespace

int main() {
  TestPointTriangleOracleCollisionUsesSubdivision();
  TestPointTriangleOracleSeparation();
  TestEdgeEdgeOracleCollisionUsesSubdivision();
  TestEdgeEdgeOracleSeparation();
  TestCertificateEngineCollisionOutput();
  TestCertificateEngineSeparationOutput();
  TestCertificateEngineUndecidedOutput();
  TestLegacyInterfaceStaysUndecided();
  TestExactWorkQueueProcessingAuditAndCoverage();
  TestExactWorkQueueCoverageRejectsDroppedItem();
  TestConservativeRefinementHeuristics();
  return g_failures == 0 ? 0 : 1;
}
