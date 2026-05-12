#include "rt_candidate/patch_granularity_tuning.h"

#include <cmath>
#include <cstdint>
#include <iostream>

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

void ExpectError(const p2cccd::Status& status, const char* label) {
  if (status.ok) {
    std::cerr << "FAIL " << label << ": expected error\n";
    ++g_failures;
  }
}

p2cccd::Mesh MakeTwoIslandMesh() {
  p2cccd::Mesh mesh;
  mesh.vertices_ref = {
      {0.0, 0.0, 0.0},
      {1.0, 0.0, 0.0},
      {0.0, 1.0, 0.0},
      {1.0, 1.0, 0.0},
      {10.0, 0.0, 0.0},
      {11.0, 0.0, 0.0},
      {10.0, 1.0, 0.0},
      {11.0, 1.0, 0.0},
  };
  mesh.triangles = {
      {0, 1, 2},
      {1, 3, 2},
      {4, 5, 6},
      {5, 7, 6},
  };
  return mesh;
}

p2cccd::Mesh MakeProbeMesh() {
  p2cccd::Mesh mesh;
  mesh.vertices_ref = {
      {5.0, 0.0, 0.0},
      {6.0, 0.0, 0.0},
      {5.0, 1.0, 0.0},
      {6.0, 1.0, 0.0},
  };
  mesh.triangles = {
      {0, 1, 2},
      {1, 3, 2},
  };
  return mesh;
}

p2cccd::MotionSegment MakeStaticMotion() {
  p2cccd::MotionSegment motion;
  motion.t0 = 0.0;
  motion.t1 = 1.0;
  motion.pose_t0.rotation_xyzw = {0.0, 0.0, 0.0, 1.0};
  motion.pose_t1.rotation_xyzw = {0.0, 0.0, 0.0, 1.0};
  return motion;
}

p2cccd::PatchGranularityTuningInput MakeTuningInput() {
  p2cccd::PatchGranularityTuningInput input;
  input.query_id = 4040;
  input.slabs_per_motion_segment = 1;
  input.options = {
      {8, 16},
      {2, 16},
  };
  input.min_oracle_recall = 1.0;
  input.candidate_weight = 10.0;
  input.raw_hit_weight = 1.0;
  input.proxy_weight = 0.1;

  p2cccd::PatchGranularityTuningObject object_a;
  object_a.object_id = 10;
  object_a.mesh = MakeTwoIslandMesh();
  object_a.motion_segments = {MakeStaticMotion()};
  object_a.proxy_type = p2cccd::ProxyType::kSweptAabb;

  p2cccd::PatchGranularityTuningObject object_b;
  object_b.object_id = 20;
  object_b.mesh = MakeProbeMesh();
  object_b.motion_segments = {MakeStaticMotion()};
  object_b.proxy_type = p2cccd::ProxyType::kSweptAabb;

  input.objects = {object_a, object_b};
  return input;
}

}  // namespace

int main() {
  p2cccd::PatchGranularityTuningInput input = MakeTuningInput();
  p2cccd::PatchGranularityTuningResult result;
  ExpectOk(p2cccd::TunePatchGranularity(input, &result), "tune patch granularity");

  Expect(result.evaluations.size() == 2, "evaluation count");
  Expect(result.best_index == 1, "fine granularity selected");
  Expect(result.evaluations[0].option.max_triangles_per_leaf == 8, "coarse option order");
  Expect(result.evaluations[1].option.max_triangles_per_leaf == 2, "fine option order");
  Expect(result.evaluations[0].compact_candidate_count == 1, "coarse option has false candidate");
  Expect(result.evaluations[1].compact_candidate_count == 0, "fine option removes false candidate");
  Expect(result.evaluations[0].oracle_recall == 1.0, "coarse oracle recall");
  Expect(result.evaluations[1].oracle_recall == 1.0, "fine oracle recall");
  Expect(result.evaluations[1].selected, "selected flag set on best evaluation");
  Expect(result.evaluations[1].score < result.evaluations[0].score, "score improves");

  p2cccd::ProxyScene scene;
  std::vector<std::vector<p2cccd::Patch>> object_patches;
  ExpectOk(p2cccd::BuildProxySceneForPatchGranularity(input,
                                                      result.evaluations[1].option,
                                                      &scene,
                                                      &object_patches),
           "build selected granularity proxy scene");
  Expect(scene.primitives.size() == 3, "selected proxy scene primitive count");
  Expect(object_patches.size() == 2, "selected object patch groups");
  Expect(object_patches[0].size() == 2, "two-island mesh split into two patches");
  Expect(object_patches[1].size() == 1, "probe mesh stays as one patch");

  input.options.clear();
  ExpectError(p2cccd::TunePatchGranularity(input, &result), "reject empty option sweep");

  return g_failures == 0 ? 0 : 1;
}
