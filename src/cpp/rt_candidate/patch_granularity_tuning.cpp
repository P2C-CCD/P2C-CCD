#include "rt_candidate/patch_granularity_tuning.h"

#include "geometry/mesh_io.h"
#include "rt_candidate/candidate_buffer.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <set>
#include <sstream>
#include <tuple>
#include <utility>

namespace p2cccd {
namespace {

struct CandidateKey {
  std::uint64_t query_id = 0;
  std::uint32_t slab_id = 0;
  std::uint32_t object_a_id = 0;
  std::uint32_t patch_a_id = 0;
  std::uint32_t object_b_id = 0;
  std::uint32_t patch_b_id = 0;
  ProxyType proxy_type_a = ProxyType::kUnknown;
  ProxyType proxy_type_b = ProxyType::kUnknown;

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

Status RequireFiniteNonNegative(double value, const char* field_name) {
  if (!std::isfinite(value) || value < 0.0) {
    return Status::Error(std::string(field_name) + " must be finite and non-negative");
  }
  return Status::Ok();
}

bool IsConcreteProxyType(ProxyType proxy_type) {
  return proxy_type == ProxyType::kSweptAabb || proxy_type == ProxyType::kCapsule;
}

CandidateKey MakeKey(const CandidateRecord& candidate) {
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

std::set<CandidateKey> MakeKeySet(const std::vector<CandidateRecord>& candidates) {
  std::set<CandidateKey> keys;
  for (const CandidateRecord& candidate : candidates) {
    keys.insert(MakeKey(candidate));
  }
  return keys;
}

double RecallAgainstOracle(const std::vector<CandidateRecord>& oracle,
                           const std::vector<CandidateRecord>& generated) {
  const std::set<CandidateKey> oracle_keys = MakeKeySet(oracle);
  if (oracle_keys.empty()) {
    return 1.0;
  }

  const std::set<CandidateKey> generated_keys = MakeKeySet(generated);
  std::uint64_t matched = 0;
  for (const CandidateKey& key : oracle_keys) {
    if (generated_keys.contains(key)) {
      ++matched;
    }
  }
  return static_cast<double>(matched) / static_cast<double>(oracle_keys.size());
}

Status ValidateInput(const PatchGranularityTuningInput& input) {
  if (input.query_id == 0) {
    return Status::Error("PatchGranularityTuningInput.query_id is required");
  }
  if (input.objects.size() < 2) {
    return Status::Error("patch granularity tuning requires at least two objects");
  }
  if (input.options.empty()) {
    return Status::Error("patch granularity tuning requires at least one option");
  }
  if (input.slabs_per_motion_segment == 0) {
    return Status::Error("slabs_per_motion_segment must be positive");
  }
  if (!std::isfinite(input.min_oracle_recall) || input.min_oracle_recall < 0.0 ||
      input.min_oracle_recall > 1.0) {
    return Status::Error("min_oracle_recall must be in [0, 1]");
  }
  if (auto status = RequireFiniteNonNegative(input.candidate_weight, "candidate_weight");
      !status.ok) {
    return status;
  }
  if (auto status = RequireFiniteNonNegative(input.raw_hit_weight, "raw_hit_weight");
      !status.ok) {
    return status;
  }
  if (auto status = RequireFiniteNonNegative(input.proxy_weight, "proxy_weight"); !status.ok) {
    return status;
  }
  if (auto status = RequireFiniteNonNegative(input.radius_weight, "radius_weight"); !status.ok) {
    return status;
  }

  for (const PatchGranularityOption& option : input.options) {
    if (option.max_triangles_per_leaf == 0) {
      return Status::Error("PatchGranularityOption.max_triangles_per_leaf must be positive");
    }
    if (option.max_depth == 0) {
      return Status::Error("PatchGranularityOption.max_depth must be positive");
    }
  }

  for (const PatchGranularityTuningObject& object : input.objects) {
    if (object.object_id == 0) {
      return Status::Error("PatchGranularityTuningObject.object_id is required");
    }
    if (!IsConcreteProxyType(object.proxy_type)) {
      return Status::Error("PatchGranularityTuningObject.proxy_type must be concrete");
    }
    if (object.motion_segments.empty()) {
      return Status::Error("PatchGranularityTuningObject.motion_segments is empty");
    }
    if (!std::isfinite(object.eps_proxy) || object.eps_proxy < 0.0) {
      return Status::Error("PatchGranularityTuningObject.eps_proxy must be finite and non-negative");
    }
    if (auto status = ValidateTriangleMesh(object.mesh); !status.ok) {
      return status;
    }
  }

  return Status::Ok();
}

PatchGranularityEvaluation MakeEvaluation(const PatchGranularityTuningInput& input,
                                          const PatchGranularityOption& option,
                                          const std::vector<std::vector<Patch>>& object_patches,
                                          const CandidateGenerationResult& generated,
                                          double oracle_recall) {
  PatchGranularityEvaluation evaluation;
  evaluation.option = option;
  evaluation.feasible = oracle_recall + 1.0e-12 >= input.min_oracle_recall;
  evaluation.backend_name = generated.backend_name;
  evaluation.proxy_count = generated.density.proxy_count;
  evaluation.raw_hit_count = generated.density.raw_hit_count;
  evaluation.compact_candidate_count = generated.density.compact_candidate_count;
  evaluation.oracle_recall = oracle_recall;
  evaluation.density = generated.density;

  double radius_sum = 0.0;
  double area_sum = 0.0;
  for (const std::vector<Patch>& patches : object_patches) {
    evaluation.patch_count += patches.size();
    for (const Patch& patch : patches) {
      radius_sum += patch.radius;
      area_sum += patch.area;
    }
  }
  if (evaluation.patch_count != 0) {
    evaluation.avg_patch_radius = radius_sum / static_cast<double>(evaluation.patch_count);
    evaluation.avg_patch_area = area_sum / static_cast<double>(evaluation.patch_count);
  }

  evaluation.score =
      input.candidate_weight * static_cast<double>(evaluation.compact_candidate_count) +
      input.raw_hit_weight * static_cast<double>(evaluation.raw_hit_count) +
      input.proxy_weight * static_cast<double>(evaluation.proxy_count) +
      input.radius_weight * evaluation.avg_patch_radius;
  return evaluation;
}

bool IsBetterEvaluation(const PatchGranularityEvaluation& candidate,
                        const PatchGranularityEvaluation& incumbent) {
  constexpr double kTieTolerance = 1.0e-12;
  if (candidate.score < incumbent.score - kTieTolerance) {
    return true;
  }
  if (std::abs(candidate.score - incumbent.score) > kTieTolerance) {
    return false;
  }
  if (candidate.compact_candidate_count != incumbent.compact_candidate_count) {
    return candidate.compact_candidate_count < incumbent.compact_candidate_count;
  }
  if (candidate.raw_hit_count != incumbent.raw_hit_count) {
    return candidate.raw_hit_count < incumbent.raw_hit_count;
  }
  if (candidate.proxy_count != incumbent.proxy_count) {
    return candidate.proxy_count < incumbent.proxy_count;
  }
  return candidate.option.max_triangles_per_leaf > incumbent.option.max_triangles_per_leaf;
}

Status EnsureParentDirectory(const std::filesystem::path& path) {
  const std::filesystem::path parent = path.parent_path();
  if (parent.empty()) {
    return Status::Ok();
  }

  std::error_code ec;
  std::filesystem::create_directories(parent, ec);
  if (ec) {
    return Status::Error("failed to create patch granularity tuning export directory: " +
                         ec.message());
  }
  return Status::Ok();
}

void WriteEvaluationCsvRow(std::ostream& stream,
                           int evaluation_index,
                           const PatchGranularityEvaluation& row) {
  stream << evaluation_index << ',' << (row.selected ? 1 : 0) << ','
         << (row.feasible ? 1 : 0) << ',' << row.option.max_triangles_per_leaf << ','
         << row.option.max_depth << ',' << row.patch_count << ',' << row.proxy_count << ','
         << row.raw_hit_count << ',' << row.compact_candidate_count << ','
         << row.oracle_recall << ',' << row.avg_patch_radius << ',' << row.avg_patch_area << ','
         << row.score << ',' << row.density.aabb_overlap_ratio << ','
         << row.density.candidates_per_proxy << ',' << row.density.candidates_per_slab << ','
         << row.density.timing.build_ms << ',' << row.density.timing.update_ms << ','
         << row.density.timing.trace_ms << ',' << row.density.timing.compact_ms << ','
         << row.density.timing.stats_ms << ',' << row.density.timing.total_ms << ','
         << '"' << row.backend_name << '"' << '\n';
}

}  // namespace

Status BuildProxySceneForPatchGranularity(const PatchGranularityTuningInput& input,
                                          const PatchGranularityOption& option,
                                          ProxyScene* scene,
                                          std::vector<std::vector<Patch>>* object_patches) {
  if (scene == nullptr) {
    return Status::Error("proxy scene output pointer is null");
  }
  if (object_patches == nullptr) {
    return Status::Error("object patch output pointer is null");
  }
  if (auto status = ValidateInput(input); !status.ok) {
    return status;
  }
  if (option.max_triangles_per_leaf == 0 || option.max_depth == 0) {
    return Status::Error("patch granularity option must be positive");
  }

  ProxySceneBuildInput scene_input;
  scene_input.query_id = input.query_id;
  object_patches->clear();
  object_patches->reserve(input.objects.size());

  for (const PatchGranularityTuningObject& tuning_object : input.objects) {
    BvhPatchBuildOptions bvh_options;
    bvh_options.max_triangles_per_leaf = option.max_triangles_per_leaf;
    bvh_options.max_depth = option.max_depth;

    std::vector<Patch> patches;
    if (auto status =
            BuildPatchesFromBvhLeafClusters(tuning_object.mesh, bvh_options, &patches);
        !status.ok) {
      return status;
    }

    ProxyObjectBuildInput object;
    object.object_id = tuning_object.object_id;
    object.proxy_type = tuning_object.proxy_type;
    object.patches = patches;
    object.motion_segments = tuning_object.motion_segments;
    object.slabs_per_motion_segment = input.slabs_per_motion_segment;
    object.eps_proxy = tuning_object.eps_proxy;
    scene_input.objects.push_back(std::move(object));
    object_patches->push_back(std::move(patches));
  }

  return BuildProxyScene(scene_input, scene);
}

Status TunePatchGranularity(const PatchGranularityTuningInput& input,
                            PatchGranularityTuningResult* result) {
  if (result == nullptr) {
    return Status::Error("patch granularity tuning result output pointer is null");
  }
  if (auto status = ValidateInput(input); !status.ok) {
    return status;
  }

  PatchGranularityTuningResult tuned;
  CandidateGenerator generator(input.candidate_config);

  int best_index = -1;
  for (const PatchGranularityOption& option : input.options) {
    ProxyScene scene;
    std::vector<std::vector<Patch>> object_patches;
    if (auto status =
            BuildProxySceneForPatchGranularity(input, option, &scene, &object_patches);
        !status.ok) {
      return status;
    }

    CandidateGenerationResult generated;
    if (auto status = generator.GenerateCandidates(scene, scene.query_id, &generated);
        !status.ok) {
      return status;
    }

    RawCandidateBuffer oracle_raw;
    if (auto status = GenerateRawCandidatesCpu(scene, scene.query_id, &oracle_raw); !status.ok) {
      return status;
    }
    std::vector<CandidateRecord> oracle_candidates;
    if (auto status = CompactRawCandidates(scene, oracle_raw, &oracle_candidates); !status.ok) {
      return status;
    }

    PatchGranularityEvaluation evaluation =
        MakeEvaluation(input,
                       option,
                       object_patches,
                       generated,
                       RecallAgainstOracle(oracle_candidates, generated.candidates));
    tuned.evaluations.push_back(std::move(evaluation));
    const int current_index = static_cast<int>(tuned.evaluations.size() - 1);
    if (tuned.evaluations.back().feasible &&
        (best_index < 0 ||
         IsBetterEvaluation(tuned.evaluations.back(), tuned.evaluations[best_index]))) {
      best_index = current_index;
    }
  }

  if (best_index < 0) {
    return Status::Error("no patch granularity option satisfied the minimum oracle recall");
  }

  tuned.best_index = best_index;
  tuned.evaluations[best_index].selected = true;
  *result = std::move(tuned);
  return Status::Ok();
}

std::string PatchGranularityTuningCsvHeader() {
  return "evaluation_index,selected,feasible,max_triangles_per_leaf,max_depth,"
         "patch_count,proxy_count,raw_hit_count,compact_candidate_count,"
         "oracle_recall,avg_patch_radius,avg_patch_area,score,aabb_overlap_ratio,"
         "candidates_per_proxy,candidates_per_slab,build_ms,update_ms,trace_ms,"
         "compact_ms,stats_ms,total_ms,backend_name";
}

Status WritePatchGranularityTuningCsv(const std::filesystem::path& path,
                                      const PatchGranularityTuningResult& result) {
  if (auto status = EnsureParentDirectory(path); !status.ok) {
    return status;
  }

  std::ofstream stream(path, std::ios::trunc);
  if (!stream) {
    return Status::Error("failed to open patch granularity tuning CSV export");
  }

  stream << std::setprecision(17);
  stream << PatchGranularityTuningCsvHeader() << '\n';
  for (std::size_t i = 0; i < result.evaluations.size(); ++i) {
    WriteEvaluationCsvRow(stream, static_cast<int>(i), result.evaluations[i]);
  }
  return Status::Ok();
}

}  // namespace p2cccd
