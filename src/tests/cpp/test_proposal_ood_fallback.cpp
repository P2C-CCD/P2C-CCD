#include "proposal/proposal_features.h"
#include "proposal/proposal_policy.h"
#include "rt_candidate/candidate_generator.h"
#include "rt_candidate/proxy_scene.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <vector>

namespace {

int g_failures = 0;

void Expect(bool condition, const char* label) {
  if (!condition) {
    std::cerr << "FAIL " << label << '\n';
    ++g_failures;
  }
}

void ExpectOk(const p2cccd::Status& status, const char* label) {
  if (!status.ok) {
    std::cerr << "FAIL " << label << ": " << status.message << '\n';
    ++g_failures;
  }
}

p2cccd::Patch MakePatch(std::uint32_t patch_id, double x, double radius = 0.2) {
  p2cccd::Patch patch;
  patch.patch_id = patch_id;
  patch.triangle_ids = {patch_id};
  patch.triangle_count = 1;
  patch.area = 1.0;
  patch.local_center = {x, 0.0, 0.0};
  patch.radius = radius;
  return patch;
}

p2cccd::MotionSegment MakeStaticMotion() {
  p2cccd::MotionSegment motion;
  motion.t0 = 0.0;
  motion.t1 = 1.0;
  motion.pose_t0.rotation_xyzw = {0.0, 0.0, 0.0, 1.0};
  motion.pose_t1.rotation_xyzw = {0.0, 0.0, 0.0, 1.0};
  return motion;
}

p2cccd::ProxySceneBuildInput MakeSceneInput() {
  p2cccd::ProxySceneBuildInput input;
  input.query_id = 6700;

  p2cccd::ProxyObjectBuildInput object_a;
  object_a.object_id = 10;
  object_a.proxy_type = p2cccd::ProxyType::kSweptAabb;
  object_a.patches = {MakePatch(1, 0.0), MakePatch(2, 4.0)};
  object_a.motion_segments = {MakeStaticMotion()};
  object_a.slabs_per_motion_segment = 2;
  object_a.eps_proxy = 0.05;

  p2cccd::ProxyObjectBuildInput object_b;
  object_b.object_id = 20;
  object_b.proxy_type = p2cccd::ProxyType::kCapsule;
  object_b.patches = {MakePatch(3, 0.25), MakePatch(4, 8.0)};
  object_b.motion_segments = {MakeStaticMotion()};
  object_b.slabs_per_motion_segment = 2;
  object_b.eps_proxy = 0.05;

  input.objects = {object_a, object_b};
  return input;
}

}  // namespace

int main() {
  p2cccd::ProxyScene scene;
  ExpectOk(p2cccd::BuildProxyScene(MakeSceneInput(), &scene), "build proxy scene");

  p2cccd::CandidateGenerator generator;
  p2cccd::CandidateGenerationResult generation_result;
  ExpectOk(generator.GenerateCandidates(scene, scene.query_id, &generation_result),
           "generate candidates");

  p2cccd::RawCandidateQueue raw_queue;
  ExpectOk(p2cccd::BuildRawCandidateQueue(generation_result, &raw_queue),
           "build raw candidate queue");
  std::vector<p2cccd::ProposalFeatureRow> rows;
  ExpectOk(p2cccd::ExtractProposalFeatureRows(scene, generation_result, &rows),
           "extract proposal feature rows");
  std::vector<p2cccd::ProposalOutput> outputs;
  ExpectOk(p2cccd::BuildDummyProposalOutputs(rows, &outputs), "build dummy proposal outputs");

  rows.front().features[1] = 50.0F;
  outputs.back().priority_score = std::numeric_limits<float>::quiet_NaN();
  outputs.front().uncertainty_score = 0.99F;

  p2cccd::ProposalSchedulingConfig config;
  config.first_work_item_id = 900;
  config.ood_abs_feature_threshold = 10.0F;
  config.uncertainty_fallback_threshold = 0.95F;

  std::vector<p2cccd::ExactWorkItem> scheduled;
  p2cccd::ProposalScheduleStats stats;
  ExpectOk(p2cccd::ScheduleExactWorkItemsFromProposals(scene,
                                                        raw_queue,
                                                        rows,
                                                        outputs,
                                                        config,
                                                        &scheduled,
                                                        &stats),
           "schedule with OOD, high uncertainty, and invalid proposal");
  ExpectOk(p2cccd::ValidateProposalScheduleConservation(raw_queue, scheduled),
           "OOD fallback conserves work queue");
  Expect(stats.fallback_count == raw_queue.candidates.size(), "all risky rows fallback");
  Expect(stats.ood_fallback_count == 1, "OOD fallback counted once");
  Expect(stats.high_uncertainty_fallback_count == 1, "uncertainty fallback counted once");
  Expect(stats.invalid_proposal_fallback_count == 1, "invalid proposal fallback counted once");
  Expect(stats.monotonic_safe, "OOD fallback monotonic safe");
  for (const p2cccd::ExactWorkItem& item : scheduled) {
    Expect(item.source == p2cccd::ProposalSource::kFallback,
           "risky proposal routed through fallback");
    Expect(item.feature_family_mask ==
               (p2cccd::kFeatureFamilyPointTriangle | p2cccd::kFeatureFamilyEdgeEdge),
           "fallback keeps all exact feature families");
  }

  return g_failures == 0 ? 0 : 1;
}
