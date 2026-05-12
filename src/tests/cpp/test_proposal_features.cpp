#include "proposal/proposal_features.h"
#include "rt_candidate/candidate_generator.h"
#include "rt_candidate/proxy_scene.h"

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
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

std::string ReadTextFile(const std::filesystem::path& path) {
  std::ifstream stream(path);
  return std::string(std::istreambuf_iterator<char>(stream),
                     std::istreambuf_iterator<char>());
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
  input.query_id = 5700;

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
  Expect(generation_result.candidates.size() == 2, "candidate count");

  p2cccd::RawCandidateQueue raw_queue;
  ExpectOk(p2cccd::BuildRawCandidateQueue(generation_result, &raw_queue),
           "build raw candidate queue");
  Expect(raw_queue.candidates.size() == generation_result.candidates.size(),
         "raw queue preserves candidate count");

  std::vector<p2cccd::ProposalFeatureRow> rows;
  ExpectOk(p2cccd::ExtractProposalFeatureRows(scene, generation_result, &rows),
           "extract proposal feature rows");
  Expect(rows.size() == generation_result.candidates.size(), "feature row count");
  for (const p2cccd::ProposalFeatureRow& row : rows) {
    Expect(row.schema_version == p2cccd::kProposalFeatureRowSchemaVersion, "row schema");
    Expect(row.query_id == scene.query_id, "row query id");
    float interval_target_sum = 0.0F;
    for (float value : row.interval_targets) {
      interval_target_sum += value;
    }
    Expect(interval_target_sum == 1.0F, "interval target one-hot");
    Expect(row.family_targets[0] == 1.0F && row.family_targets[1] == 1.0F,
           "family target defaults");
    Expect(row.target_mask != 0, "target mask set");
    for (float feature : row.features) {
      Expect(std::isfinite(feature), "feature finite");
    }
  }

  p2cccd::ProposalQueueConfig config;
  config.first_work_item_id = 900;
  p2cccd::ProposalDataFlow data_flow;
  ExpectOk(p2cccd::BuildProposalDataFlow(scene, generation_result, config, &data_flow),
           "build proposal data flow");
  ExpectOk(p2cccd::ValidateProposalDataFlow(data_flow), "validate proposal data flow");
  Expect(data_flow.exact_work_queue.size() == data_flow.raw_candidate_queue.candidates.size(),
         "exact work queue count");
  Expect(data_flow.exact_work_queue.front().work_item_id == 900, "first work item id");
  Expect(data_flow.exact_work_queue.front().feature_family_mask ==
             (p2cccd::kFeatureFamilyPointTriangle | p2cccd::kFeatureFamilyEdgeEdge),
         "feature family mask");

  p2cccd::ProposalDataFlow broken = data_flow;
  broken.exact_work_queue.pop_back();
  Expect(!p2cccd::ValidateProposalDataFlow(broken).ok,
         "data flow rejects dropped exact work item");

  const std::filesystem::path csv_path =
      std::filesystem::temp_directory_path() / "p2cccd_proposal_features" /
      "proposal_features.csv";
  std::filesystem::remove(csv_path);
  ExpectOk(p2cccd::WriteProposalFeatureRowsCsv(csv_path, rows, false),
           "write proposal feature rows csv");
  const std::string csv_text = ReadTextFile(csv_path);
  Expect(csv_text.find(p2cccd::ProposalFeatureCsvHeader()) != std::string::npos,
         "proposal CSV header");
  Expect(csv_text.find("feature_31") != std::string::npos, "proposal CSV feature columns");

  return g_failures == 0 ? 0 : 1;
}
