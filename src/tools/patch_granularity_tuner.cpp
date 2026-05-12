#include "rt_candidate/patch_granularity_tuning.h"

#include <cstdint>
#include <filesystem>
#include <iostream>

namespace {

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
      {1, 16},
  };
  input.min_oracle_recall = 1.0;
  input.candidate_weight = 10.0;
  input.raw_hit_weight = 1.0;
  input.proxy_weight = 0.1;
  input.radius_weight = 0.0;

  p2cccd::PatchGranularityTuningObject object_a;
  object_a.object_id = 10;
  object_a.mesh = MakeTwoIslandMesh();
  object_a.motion_segments = {MakeStaticMotion()};
  object_a.proxy_type = p2cccd::ProxyType::kSweptAabb;
  object_a.eps_proxy = 0.0;

  p2cccd::PatchGranularityTuningObject object_b;
  object_b.object_id = 20;
  object_b.mesh = MakeProbeMesh();
  object_b.motion_segments = {MakeStaticMotion()};
  object_b.proxy_type = p2cccd::ProxyType::kSweptAabb;
  object_b.eps_proxy = 0.0;

  input.objects = {object_a, object_b};
  return input;
}

}  // namespace

int main(int argc, char** argv) {
  const std::filesystem::path output_path =
      argc >= 2 ? std::filesystem::path(argv[1])
                : std::filesystem::path("outputs/patch_granularity_tuning.csv");

  p2cccd::PatchGranularityTuningResult result;
  if (auto status = p2cccd::TunePatchGranularity(MakeTuningInput(), &result); !status.ok) {
    std::cerr << "TunePatchGranularity failed: " << status.message << '\n';
    return 1;
  }
  if (auto status = p2cccd::WritePatchGranularityTuningCsv(output_path, result); !status.ok) {
    std::cerr << "WritePatchGranularityTuningCsv failed: " << status.message << '\n';
    return 1;
  }

  const p2cccd::PatchGranularityEvaluation& best =
      result.evaluations[static_cast<std::size_t>(result.best_index)];
  std::cout << "wrote " << std::filesystem::absolute(output_path).string() << '\n';
  std::cout << "best max_triangles_per_leaf=" << best.option.max_triangles_per_leaf
            << ", candidates=" << best.compact_candidate_count
            << ", proxies=" << best.proxy_count << ", score=" << best.score << '\n';
  return 0;
}
