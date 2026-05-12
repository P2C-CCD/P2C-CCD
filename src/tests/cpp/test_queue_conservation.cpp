#include "proposal/proposal_features.h"
#include "proposal/proposal_policy.h"
#include "rt_candidate/candidate_generator.h"
#include "rt_candidate/proxy_scene.h"

#include <cstdint>
#include <iostream>
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
  input.query_id = 6600;

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
                  std::vector<p2cccd::ExactWorkItem>* scheduled) {
  ExpectOk(p2cccd::BuildProxyScene(MakeSceneInput(), scene), "build proxy scene");
  p2cccd::CandidateGenerator generator;
  p2cccd::CandidateGenerationResult generation_result;
  ExpectOk(generator.GenerateCandidates(*scene, scene->query_id, &generation_result),
           "generate candidates");
  ExpectOk(p2cccd::BuildRawCandidateQueue(generation_result, raw_queue),
           "build raw candidate queue");
  std::vector<p2cccd::ProposalFeatureRow> rows;
  ExpectOk(p2cccd::ExtractProposalFeatureRows(*scene, generation_result, &rows),
           "extract proposal feature rows");
  std::vector<p2cccd::ProposalOutput> outputs;
  ExpectOk(p2cccd::BuildDummyProposalOutputs(rows, &outputs), "build dummy proposal outputs");
  p2cccd::ProposalSchedulingConfig config;
  config.first_work_item_id = 800;
  ExpectOk(p2cccd::ScheduleExactWorkItemsFromProposals(*scene,
                                                        *raw_queue,
                                                        rows,
                                                        outputs,
                                                        config,
                                                        scheduled,
                                                        nullptr),
           "schedule proposal outputs");
}

}  // namespace

int main() {
  p2cccd::ProxyScene scene;
  p2cccd::RawCandidateQueue raw_queue;
  std::vector<p2cccd::ExactWorkItem> scheduled;
  BuildFixture(&scene, &raw_queue, &scheduled);
  ExpectOk(p2cccd::ValidateProposalScheduleConservation(raw_queue, scheduled),
           "valid schedule conserves candidates");

  std::vector<p2cccd::ExactWorkItem> dropped = scheduled;
  dropped.pop_back();
  Expect(!p2cccd::ValidateProposalScheduleConservation(raw_queue, dropped).ok,
         "dropped work item rejected");

  std::vector<p2cccd::ExactWorkItem> duplicate_parent = scheduled;
  duplicate_parent.back().parent_candidate_id = duplicate_parent.front().parent_candidate_id;
  Expect(!p2cccd::ValidateProposalScheduleConservation(raw_queue, duplicate_parent).ok,
         "duplicate parent rejected");

  std::vector<p2cccd::ExactWorkItem> unknown_parent = scheduled;
  unknown_parent.back().parent_candidate_id = 999999;
  Expect(!p2cccd::ValidateProposalScheduleConservation(raw_queue, unknown_parent).ok,
         "unknown parent rejected");

  std::vector<p2cccd::ExactWorkItem> duplicate_work_id = scheduled;
  duplicate_work_id.back().work_item_id = duplicate_work_id.front().work_item_id;
  Expect(!p2cccd::ValidateProposalScheduleConservation(raw_queue, duplicate_work_id).ok,
         "duplicate work item id rejected");

  std::vector<p2cccd::ExactWorkItem> no_family = scheduled;
  no_family.front().feature_family_mask = 0;
  Expect(!p2cccd::ValidateProposalScheduleConservation(raw_queue, no_family).ok,
         "missing feature family rejected");

  std::vector<p2cccd::ExactWorkItem> unknown_family = scheduled;
  unknown_family.front().feature_family_mask =
      p2cccd::kFeatureFamilyPointTriangle | (1U << 8U);
  Expect(!p2cccd::ValidateProposalScheduleConservation(raw_queue, unknown_family).ok,
         "unknown feature family bit rejected");

  std::vector<p2cccd::ProposalFeatureRow> rows;
  p2cccd::CandidateGenerationResult generation_result;
  p2cccd::CandidateGenerator generator;
  ExpectOk(generator.GenerateCandidates(scene, scene.query_id, &generation_result),
           "regenerate candidates");
  ExpectOk(p2cccd::ExtractProposalFeatureRows(scene, generation_result, &rows),
           "extract rows for config validation");
  std::vector<p2cccd::ProposalOutput> outputs;
  ExpectOk(p2cccd::BuildDummyProposalOutputs(rows, &outputs),
           "build outputs for config validation");
  p2cccd::ProposalSchedulingConfig bad_config;
  bad_config.conservative_feature_family_mask = 1U << 8U;
  std::vector<p2cccd::ExactWorkItem> rejected_schedule;
  Expect(!p2cccd::ScheduleExactWorkItemsFromProposals(scene,
                                                       raw_queue,
                                                       rows,
                                                       outputs,
                                                       bad_config,
                                                       &rejected_schedule,
                                                       nullptr)
              .ok,
         "unknown family mask config rejected");

  return g_failures == 0 ? 0 : 1;
}
