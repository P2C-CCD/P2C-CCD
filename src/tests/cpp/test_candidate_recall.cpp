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

struct CandidateKey {
  std::uint64_t query_id = 0;
  std::uint32_t slab_id = 0;
  std::uint32_t object_a_id = 0;
  std::uint32_t patch_a_id = 0;
  std::uint32_t object_b_id = 0;
  std::uint32_t patch_b_id = 0;
  p2cccd::ProxyType proxy_type_a = p2cccd::ProxyType::kUnknown;
  p2cccd::ProxyType proxy_type_b = p2cccd::ProxyType::kUnknown;

  bool operator<(const CandidateKey& rhs) const {
    return std::tie(query_id,
                    slab_id,
                    object_a_id,
                    patch_a_id,
                    object_b_id,
                    patch_b_id,
                    proxy_type_a,
                    proxy_type_b) <
           std::tie(rhs.query_id,
                    rhs.slab_id,
                    rhs.object_a_id,
                    rhs.patch_a_id,
                    rhs.object_b_id,
                    rhs.patch_b_id,
                    rhs.proxy_type_a,
                    rhs.proxy_type_b);
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

bool TimeOverlap(const p2cccd::ProxyPrimitive& lhs, const p2cccd::ProxyPrimitive& rhs) {
  return lhs.t0 < rhs.t1 && rhs.t0 < lhs.t1;
}

CandidateKey MakeKey(const p2cccd::ProxyPrimitive& lhs,
                     const p2cccd::ProxyPrimitive& rhs,
                     std::uint64_t query_id) {
  const p2cccd::ProxyPrimitive* first = &lhs;
  const p2cccd::ProxyPrimitive* second = &rhs;
  if (std::tie(first->object_id, first->patch_id, first->proxy_id) >
      std::tie(second->object_id, second->patch_id, second->proxy_id)) {
    std::swap(first, second);
  }

  CandidateKey key;
  key.query_id = query_id;
  key.slab_id = first->slab_id;
  key.object_a_id = first->object_id;
  key.patch_a_id = first->patch_id;
  key.object_b_id = second->object_id;
  key.patch_b_id = second->patch_id;
  key.proxy_type_a = first->proxy_type;
  key.proxy_type_b = second->proxy_type;
  return key;
}

CandidateKey MakeKey(const p2cccd::CandidateRecord& candidate) {
  CandidateKey key;
  key.query_id = candidate.query_id;
  key.slab_id = candidate.slab_id;
  key.object_a_id = candidate.object_a_id;
  key.patch_a_id = candidate.patch_a_id;
  key.object_b_id = candidate.object_b_id;
  key.patch_b_id = candidate.patch_b_id;
  key.proxy_type_a = candidate.proxy_type_a;
  key.proxy_type_b = candidate.proxy_type_b;
  return key;
}

std::set<CandidateKey> ComputeOracleCandidates(const p2cccd::ProxyScene& scene) {
  std::set<CandidateKey> keys;
  for (std::uint32_t i = 0; i < scene.primitives.size(); ++i) {
    const p2cccd::ProxyPrimitive& lhs = scene.primitives[i];
    for (std::uint32_t j = i + 1; j < scene.primitives.size(); ++j) {
      const p2cccd::ProxyPrimitive& rhs = scene.primitives[j];
      if (lhs.object_id == rhs.object_id || lhs.slab_id != rhs.slab_id) {
        continue;
      }
      if (!TimeOverlap(lhs, rhs) || !AabbOverlap(lhs.bounds, rhs.bounds)) {
        continue;
      }
      keys.insert(MakeKey(lhs, rhs, scene.query_id));
    }
  }
  return keys;
}

std::set<CandidateKey> MakeGeneratedKeySet(
    const std::vector<p2cccd::CandidateRecord>& candidates) {
  std::set<CandidateKey> keys;
  for (const p2cccd::CandidateRecord& candidate : candidates) {
    keys.insert(MakeKey(candidate));
  }
  return keys;
}

double Recall(const std::set<CandidateKey>& expected,
              const std::set<CandidateKey>& generated) {
  if (expected.empty()) {
    return 1.0;
  }
  std::uint64_t matched = 0;
  for (const CandidateKey& key : expected) {
    if (generated.contains(key)) {
      ++matched;
    }
  }
  return static_cast<double>(matched) / static_cast<double>(expected.size());
}

p2cccd::Patch MakePatch(std::uint32_t patch_id,
                        double x,
                        double radius = 0.2) {
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
  motion.pose_t0.translation = {0.0, 0.0, 0.0};
  motion.pose_t1.translation = {0.0, 0.0, 0.0};
  motion.pose_t0.rotation_xyzw = {0.0, 0.0, 0.0, 1.0};
  motion.pose_t1.rotation_xyzw = {0.0, 0.0, 0.0, 1.0};
  return motion;
}

p2cccd::ProxySceneBuildInput MakeRecallSceneInput() {
  p2cccd::ProxySceneBuildInput input;
  input.query_id = 9001;

  p2cccd::ProxyObjectBuildInput object_a;
  object_a.object_id = 10;
  object_a.proxy_type = p2cccd::ProxyType::kSweptAabb;
  object_a.patches = {MakePatch(1, 0.0), MakePatch(2, 3.0)};
  object_a.motion_segments = {MakeStaticMotion()};
  object_a.slabs_per_motion_segment = 3;
  object_a.eps_proxy = 0.01;

  p2cccd::ProxyObjectBuildInput object_b;
  object_b.object_id = 20;
  object_b.proxy_type = p2cccd::ProxyType::kCapsule;
  object_b.patches = {MakePatch(3, 0.25), MakePatch(4, 7.0)};
  object_b.motion_segments = {MakeStaticMotion()};
  object_b.slabs_per_motion_segment = 3;
  object_b.eps_proxy = 0.01;

  p2cccd::ProxyObjectBuildInput object_c;
  object_c.object_id = 30;
  object_c.proxy_type = p2cccd::ProxyType::kSweptAabb;
  object_c.patches = {MakePatch(5, 3.15)};
  object_c.motion_segments = {MakeStaticMotion()};
  object_c.slabs_per_motion_segment = 3;
  object_c.eps_proxy = 0.01;

  input.objects = {object_a, object_b, object_c};
  return input;
}

}  // namespace

int main() {
  p2cccd::ProxyScene scene;
  ExpectOk(p2cccd::BuildProxyScene(MakeRecallSceneInput(), &scene), "build recall scene");

  const std::set<CandidateKey> oracle_keys = ComputeOracleCandidates(scene);
  Expect(!oracle_keys.empty(), "oracle has positive candidates");

  p2cccd::CandidateGenerator generator;
  p2cccd::CandidateGenerationResult result;
  ExpectOk(generator.GenerateCandidates(scene, scene.query_id, &result),
           "generate CPU reference candidates");
  const std::set<CandidateKey> generated_keys = MakeGeneratedKeySet(result.candidates);
  Expect(Recall(oracle_keys, generated_keys) == 1.0, "CPU reference recall is 1.0");

  p2cccd::CandidateGeneratorConfig optix_fallback_config;
  optix_fallback_config.backend = p2cccd::CandidateBackend::kOptix;
  optix_fallback_config.allow_optix_cpu_fallback = true;
  p2cccd::CandidateGenerator optix_fallback_generator(optix_fallback_config);
  ExpectOk(optix_fallback_generator.GenerateCandidates(scene, scene.query_id, &result),
           "generate OptiX fallback candidates");
  const std::set<CandidateKey> fallback_keys = MakeGeneratedKeySet(result.candidates);
  Expect(Recall(oracle_keys, fallback_keys) == 1.0, "OptiX fallback recall is 1.0");

  return g_failures == 0 ? 0 : 1;
}
