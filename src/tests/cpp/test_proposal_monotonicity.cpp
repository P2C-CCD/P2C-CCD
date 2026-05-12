#include "proposal/proposal_features.h"
#include "proposal/proposal_policy.h"
#include "rt_candidate/candidate_generator.h"
#include "rt_candidate/proxy_scene.h"

#include <cstdint>
#include <iostream>
#include <set>
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
  input.query_id = 6100;

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

void BuildFixture(p2cccd::ProxyScene* scene,
                  p2cccd::RawCandidateQueue* raw_queue,
                  std::vector<p2cccd::ProposalFeatureRow>* rows,
                  std::vector<p2cccd::ProposalOutput>* outputs) {
  ExpectOk(p2cccd::BuildProxyScene(MakeSceneInput(), scene), "build proxy scene");
  p2cccd::CandidateGenerator generator;
  p2cccd::CandidateGenerationResult generation_result;
  ExpectOk(generator.GenerateCandidates(*scene, scene->query_id, &generation_result),
           "generate candidates");
  ExpectOk(p2cccd::BuildRawCandidateQueue(generation_result, raw_queue),
           "build raw candidate queue");
  ExpectOk(p2cccd::ExtractProposalFeatureRows(*scene, generation_result, rows),
           "extract proposal feature rows");
  ExpectOk(p2cccd::BuildDummyProposalOutputs(*rows, outputs), "build dummy proposal outputs");
}

}  // namespace

int main() {
  p2cccd::ProxyScene scene;
  p2cccd::RawCandidateQueue raw_queue;
  std::vector<p2cccd::ProposalFeatureRow> rows;
  std::vector<p2cccd::ProposalOutput> outputs;
  BuildFixture(&scene, &raw_queue, &rows, &outputs);
  Expect(raw_queue.candidates.size() == 2, "fixture candidate count");

  p2cccd::ProposalSchedulingConfig config;
  config.first_work_item_id = 700;
  config.ood_abs_feature_threshold = 100.0F;

  std::vector<p2cccd::ExactWorkItem> scheduled;
  p2cccd::ProposalScheduleStats stats;
  ExpectOk(p2cccd::ScheduleExactWorkItemsFromProposals(scene,
                                                        raw_queue,
                                                        rows,
                                                        outputs,
                                                        config,
                                                        &scheduled,
                                                        &stats),
           "schedule from dummy outputs");
  ExpectOk(p2cccd::ValidateProposalScheduleConservation(raw_queue, scheduled),
           "validate monotonic schedule");
  Expect(stats.monotonic_safe, "stats monotonic safe");
  Expect(stats.fallback_count == 0, "no fallback on in-distribution dummy outputs");
  for (const p2cccd::ExactWorkItem& item : scheduled) {
    Expect(item.topk_feature_ids_offset == 0, "scheduler does not overload topk offset");
  }

  rows.front().features[0] = 1.0e6F;
  outputs.pop_back();
  scheduled.clear();
  stats = p2cccd::ProposalScheduleStats{};
  ExpectOk(p2cccd::ScheduleExactWorkItemsFromProposals(scene,
                                                        raw_queue,
                                                        rows,
                                                        outputs,
                                                        config,
                                                        &scheduled,
                                                        &stats),
           "schedule with OOD and missing proposal");
  Expect(scheduled.size() == raw_queue.candidates.size(), "fallback still conserves count");
  Expect(stats.fallback_count == 2, "two candidates fallback");
  Expect(stats.ood_fallback_count == 1, "OOD fallback counted");
  Expect(stats.missing_proposal_fallback_count == 1, "missing proposal fallback counted");
  Expect(stats.monotonic_safe, "fallback schedule monotonic safe");
  ExpectOk(p2cccd::ValidateProposalScheduleConservation(raw_queue, scheduled),
           "validate fallback monotonic schedule");

  std::set<std::uint64_t> parent_ids;
  for (const p2cccd::ExactWorkItem& item : scheduled) {
    parent_ids.insert(item.parent_candidate_id);
    Expect(item.feature_family_mask ==
               (p2cccd::kFeatureFamilyPointTriangle | p2cccd::kFeatureFamilyEdgeEdge),
           "fallback keeps conservative family mask");
    Expect(item.topk_feature_ids_offset == 0, "fallback keeps topk offset unused");
  }
  for (const p2cccd::CandidateRecord& candidate : raw_queue.candidates) {
    Expect(parent_ids.contains(candidate.candidate_id), "candidate appears in schedule");
  }

  return g_failures == 0 ? 0 : 1;
}
