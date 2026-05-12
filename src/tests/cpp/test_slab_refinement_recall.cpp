#include "rt_candidate/candidate_generator.h"
#include "rt_candidate/proxy_scene.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <set>
#include <tuple>
#include <vector>

namespace {

int g_failures = 0;

struct SlabPairKey {
  std::uint32_t slab_id = 0;
  std::uint32_t patch_a_id = 0;
  std::uint32_t patch_b_id = 0;

  bool operator<(const SlabPairKey& rhs) const {
    return std::tie(slab_id, patch_a_id, patch_b_id) <
           std::tie(rhs.slab_id, rhs.patch_a_id, rhs.patch_b_id);
  }
};

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

bool AabbOverlap(const p2cccd::Aabb& lhs, const p2cccd::Aabb& rhs) {
  for (std::uint32_t axis = 0; axis < 3; ++axis) {
    if (lhs.max[axis] < rhs.min[axis] || rhs.max[axis] < lhs.min[axis]) {
      return false;
    }
  }
  return true;
}

std::set<SlabPairKey> ComputeOracleSlabPairs(const p2cccd::ProxyScene& scene) {
  std::set<SlabPairKey> keys;
  for (std::uint32_t i = 0; i < scene.primitives.size(); ++i) {
    const p2cccd::ProxyPrimitive& lhs = scene.primitives[i];
    for (std::uint32_t j = i + 1; j < scene.primitives.size(); ++j) {
      const p2cccd::ProxyPrimitive& rhs = scene.primitives[j];
      if (lhs.object_id == rhs.object_id || lhs.slab_id != rhs.slab_id) {
        continue;
      }
      if (!AabbOverlap(lhs.bounds, rhs.bounds)) {
        continue;
      }
      keys.insert(SlabPairKey{lhs.slab_id,
                              std::min(lhs.patch_id, rhs.patch_id),
                              std::max(lhs.patch_id, rhs.patch_id)});
    }
  }
  return keys;
}

std::set<SlabPairKey> MakeGeneratedSlabPairs(
    const std::vector<p2cccd::CandidateRecord>& candidates) {
  std::set<SlabPairKey> keys;
  for (const p2cccd::CandidateRecord& candidate : candidates) {
    keys.insert(SlabPairKey{candidate.slab_id,
                            std::min(candidate.patch_a_id, candidate.patch_b_id),
                            std::max(candidate.patch_a_id, candidate.patch_b_id)});
  }
  return keys;
}

double Recall(const std::set<SlabPairKey>& expected,
              const std::set<SlabPairKey>& generated) {
  if (expected.empty()) {
    return 1.0;
  }
  std::uint64_t matched = 0;
  for (const SlabPairKey& key : expected) {
    if (generated.contains(key)) {
      ++matched;
    }
  }
  return static_cast<double>(matched) / static_cast<double>(expected.size());
}

p2cccd::Patch MakePatch(std::uint32_t patch_id, double x) {
  p2cccd::Patch patch;
  patch.patch_id = patch_id;
  patch.triangle_ids = {patch_id};
  patch.triangle_count = 1;
  patch.area = 1.0;
  patch.local_center = {x, 0.0, 0.0};
  patch.radius = 0.25;
  return patch;
}

p2cccd::MotionSegment MakeSharedMotion() {
  p2cccd::MotionSegment motion;
  motion.t0 = 0.0;
  motion.t1 = 1.0;
  motion.pose_t0.translation = {0.0, 0.0, 0.0};
  motion.pose_t1.translation = {0.5, 0.0, 0.0};
  motion.pose_t0.rotation_xyzw = {0.0, 0.0, 0.0, 1.0};
  motion.pose_t1.rotation_xyzw = {0.0, 0.0, 0.0, 1.0};
  return motion;
}

p2cccd::ProxySceneBuildInput MakeInput(std::uint32_t slabs_per_motion_segment) {
  p2cccd::ProxySceneBuildInput input;
  input.query_id = 7200 + slabs_per_motion_segment;

  p2cccd::ProxyObjectBuildInput object_a;
  object_a.object_id = 1;
  object_a.proxy_type = p2cccd::ProxyType::kSweptAabb;
  object_a.patches = {MakePatch(11, 0.0)};
  object_a.motion_segments = {MakeSharedMotion()};
  object_a.slabs_per_motion_segment = slabs_per_motion_segment;
  object_a.eps_proxy = 0.0;

  p2cccd::ProxyObjectBuildInput object_b;
  object_b.object_id = 2;
  object_b.proxy_type = p2cccd::ProxyType::kCapsule;
  object_b.patches = {MakePatch(22, 0.2)};
  object_b.motion_segments = {MakeSharedMotion()};
  object_b.slabs_per_motion_segment = slabs_per_motion_segment;
  object_b.eps_proxy = 0.0;

  input.objects = {object_a, object_b};
  return input;
}

}  // namespace

int main() {
  p2cccd::CandidateGenerator generator;
  for (std::uint32_t slab_count : {1U, 2U, 4U, 8U, 16U}) {
    p2cccd::ProxyScene scene;
    ExpectOk(p2cccd::BuildProxyScene(MakeInput(slab_count), &scene),
             "build refined slab scene");
    const std::set<SlabPairKey> oracle = ComputeOracleSlabPairs(scene);
    Expect(oracle.size() == slab_count, "oracle keeps one positive pair per slab");

    p2cccd::CandidateGenerationResult result;
    ExpectOk(generator.GenerateCandidates(scene, scene.query_id, &result),
             "generate refined slab candidates");
    const std::set<SlabPairKey> generated = MakeGeneratedSlabPairs(result.candidates);
    Expect(Recall(oracle, generated) == 1.0, "slab refinement recall remains 1.0");
    Expect(result.density.slab_count == slab_count, "density slab count matches refinement");
  }

  return g_failures == 0 ? 0 : 1;
}
