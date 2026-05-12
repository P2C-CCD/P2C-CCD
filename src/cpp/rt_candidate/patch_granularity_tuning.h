#pragma once

#include "common/status.h"
#include "geometry/mesh.h"
#include "geometry/motion.h"
#include "geometry/patch_builder.h"
#include "rt_candidate/candidate_generation_result.h"
#include "rt_candidate/candidate_generator.h"
#include "rt_candidate/proxy_scene.h"

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace p2cccd {

struct PatchGranularityOption {
  std::uint32_t max_triangles_per_leaf = 32;
  std::uint32_t max_depth = 32;
};

struct PatchGranularityTuningObject {
  std::uint32_t object_id = 0;
  Mesh mesh;
  std::vector<MotionSegment> motion_segments;
  ProxyType proxy_type = ProxyType::kSweptAabb;
  double eps_proxy = 0.0;
};

struct PatchGranularityTuningInput {
  std::uint64_t query_id = 0;
  std::vector<PatchGranularityTuningObject> objects;
  std::vector<PatchGranularityOption> options;
  std::uint32_t slabs_per_motion_segment = 1;
  CandidateGeneratorConfig candidate_config;
  double min_oracle_recall = 1.0;
  double candidate_weight = 1.0;
  double raw_hit_weight = 0.05;
  double proxy_weight = 0.01;
  double radius_weight = 0.0;
};

struct PatchGranularityEvaluation {
  PatchGranularityOption option;
  bool feasible = false;
  bool selected = false;
  std::string backend_name;
  std::uint64_t patch_count = 0;
  std::uint64_t proxy_count = 0;
  std::uint64_t raw_hit_count = 0;
  std::uint64_t compact_candidate_count = 0;
  double oracle_recall = 0.0;
  double avg_patch_radius = 0.0;
  double avg_patch_area = 0.0;
  double score = 0.0;
  CandidateDensityStats density;
};

struct PatchGranularityTuningResult {
  std::vector<PatchGranularityEvaluation> evaluations;
  int best_index = -1;
};

Status BuildProxySceneForPatchGranularity(const PatchGranularityTuningInput& input,
                                          const PatchGranularityOption& option,
                                          ProxyScene* scene,
                                          std::vector<std::vector<Patch>>* object_patches);
Status TunePatchGranularity(const PatchGranularityTuningInput& input,
                            PatchGranularityTuningResult* result);

std::string PatchGranularityTuningCsvHeader();
Status WritePatchGranularityTuningCsv(const std::filesystem::path& path,
                                      const PatchGranularityTuningResult& result);

}  // namespace p2cccd
