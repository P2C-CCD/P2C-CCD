#include "rt_candidate/external_batch_candidate.h"

#include "common/validators.h"
#include "rt_candidate/candidate_generator.h"
#include "rt_candidate/proxy_scene.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace p2cccd {
namespace {

using Clock = std::chrono::steady_clock;

constexpr std::uint64_t kBatchedSceneQueryId = 1;
constexpr double kBatchSlabStride = 2.0;
constexpr double kBatchXMargin = 1.0e-2;
constexpr std::uint64_t kCandidateIdStride = 1'000'003ULL;

struct QueryBuildInfo {
  std::uint64_t source_query_id = 0;
  std::uint64_t runtime_query_id = 0;
  std::uint32_t source_query_index = 0;
  ExternalQueryFamily family = ExternalQueryFamily::kVertexFace;
  bool has_ground_truth = false;
  bool ground_truth_collides = false;
  std::uint32_t object_a_id = 1;
  std::uint32_t object_b_id = 2;
  std::uint32_t lhs_patch_id = 0;
  std::uint32_t rhs_patch_id = 0;
};

double ElapsedMs(const Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

bool IsFinitePoint(const std::array<double, 3>& point) {
  return std::isfinite(point[0]) && std::isfinite(point[1]) && std::isfinite(point[2]);
}

Status ValidateExternalBatchQuery(const ExternalBatchQuery& query) {
  for (const std::array<double, 3>& point : query.vertices_t0) {
    if (!IsFinitePoint(point)) {
      return Status::Error("external batch query vertices_t0 must be finite");
    }
  }
  for (const std::array<double, 3>& point : query.vertices_t1) {
    if (!IsFinitePoint(point)) {
      return Status::Error("external batch query vertices_t1 must be finite");
    }
  }
  return Status::Ok();
}

std::vector<std::uint32_t> LhsFeatureIndices(const ExternalQueryFamily family) {
  if (family == ExternalQueryFamily::kVertexFace) {
    return {0};
  }
  return {0, 1};
}

std::vector<std::uint32_t> RhsFeatureIndices(const ExternalQueryFamily family) {
  if (family == ExternalQueryFamily::kVertexFace) {
    return {1, 2, 3};
  }
  return {2, 3};
}

Aabb AabbFromFeatureIndices(const ExternalBatchQuery& query,
                            const std::vector<std::uint32_t>& indices) {
  Aabb bounds;
  bounds.min = {std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity(),
                std::numeric_limits<double>::infinity()};
  bounds.max = {-std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity(),
                -std::numeric_limits<double>::infinity()};
  for (const std::uint32_t index : indices) {
    for (const std::array<double, 3>& point : {query.vertices_t0[index], query.vertices_t1[index]}) {
      for (std::uint32_t axis = 0; axis < 3; ++axis) {
        bounds.min[axis] = std::min(bounds.min[axis], point[axis]);
        bounds.max[axis] = std::max(bounds.max[axis], point[axis]);
      }
    }
  }
  return bounds;
}

Aabb TranslateAabbX(const Aabb& bounds, const double x_offset) {
  Aabb translated = bounds;
  translated.min[0] += x_offset;
  translated.max[0] += x_offset;
  return translated;
}

PatchMotionBound MotionBoundForAabb(const std::uint32_t patch_id,
                                    const double t0,
                                    const double t1,
                                    const Aabb& bounds) {
  PatchMotionBound bound;
  bound.patch_id = patch_id;
  bound.t0 = t0;
  bound.t1 = t1;
  for (std::uint32_t axis = 0; axis < 3; ++axis) {
    const double center = 0.5 * (bounds.min[axis] + bounds.max[axis]);
    bound.center_t0[axis] = center;
    bound.center_t1[axis] = center;
  }
  bound.conservative_radius =
      0.5 * std::max({bounds.max[0] - bounds.min[0], bounds.max[1] - bounds.min[1],
                      bounds.max[2] - bounds.min[2]});
  return bound;
}

std::uint64_t MakeCandidateId(const std::uint64_t runtime_query_id, const std::uint64_t ordinal) {
  return runtime_query_id * kCandidateIdStride + ordinal + 1ULL;
}

Status BuildBatchedProxyScene(const std::vector<ExternalBatchQuery>& queries,
                              std::vector<QueryBuildInfo>* infos,
                              std::unordered_map<std::uint32_t, std::size_t>* query_index_by_patch_id,
                              ProxyScene* scene) {
  if (infos == nullptr || query_index_by_patch_id == nullptr || scene == nullptr) {
    return Status::Error("external batch proxy scene outputs must be non-null");
  }
  infos->clear();
  query_index_by_patch_id->clear();

  std::unordered_set<std::uint64_t> seen_source_ids;
  std::unordered_set<std::uint64_t> used_runtime_ids;

  ProxyScene built;
  built.query_id = kBatchedSceneQueryId;

  double x_cursor = 0.0;
  infos->reserve(queries.size());
  built.primitives.reserve(queries.size() * 2U);

  for (std::size_t query_index = 0; query_index < queries.size(); ++query_index) {
    const ExternalBatchQuery& query = queries[query_index];
    if (auto status = ValidateExternalBatchQuery(query); !status.ok) {
      return status;
    }
    if (!seen_source_ids.insert(query.source_query_id).second) {
      return Status::Error("external batch source_query_id values must be unique");
    }

    std::uint64_t runtime_query_id =
        query.source_query_id > 0 ? query.source_query_id : static_cast<std::uint64_t>(query_index + 1U);
    while (used_runtime_ids.contains(runtime_query_id)) {
      ++runtime_query_id;
    }
    used_runtime_ids.insert(runtime_query_id);

    const std::vector<std::uint32_t> lhs_indices = LhsFeatureIndices(query.family);
    const std::vector<std::uint32_t> rhs_indices = RhsFeatureIndices(query.family);
    const Aabb lhs_bounds = AabbFromFeatureIndices(query, lhs_indices);
    const Aabb rhs_bounds = AabbFromFeatureIndices(query, rhs_indices);
    const double query_min_x = std::min(lhs_bounds.min[0], rhs_bounds.min[0]);
    const double query_max_x = std::max(lhs_bounds.max[0], rhs_bounds.max[0]);
    const double x_offset = x_cursor - query_min_x;
    const double t0 = static_cast<double>(query_index) * kBatchSlabStride;
    const double t1 = t0 + 1.0;
    const std::uint32_t lhs_patch_id = static_cast<std::uint32_t>(query_index * 2U + 1U);
    const std::uint32_t rhs_patch_id = static_cast<std::uint32_t>(query_index * 2U + 2U);
    const std::uint32_t slab_id = static_cast<std::uint32_t>(query_index);
    const std::array<std::uint32_t, 2> box_pair = query.has_box_pair ? query.box_pair : std::array<std::uint32_t, 2>{1U, 2U};

    QueryBuildInfo info;
    info.source_query_id = query.source_query_id;
    info.runtime_query_id = runtime_query_id;
    info.source_query_index = query.source_query_index;
    info.family = query.family;
    info.has_ground_truth = query.has_ground_truth;
    info.ground_truth_collides = query.ground_truth_collides;
    info.object_a_id = box_pair[0];
    info.object_b_id = box_pair[1];
    info.lhs_patch_id = lhs_patch_id;
    info.rhs_patch_id = rhs_patch_id;
    infos->push_back(info);

    const Aabb lhs_translated = TranslateAabbX(lhs_bounds, x_offset);
    const Aabb rhs_translated = TranslateAabbX(rhs_bounds, x_offset);

    ProxyPrimitive lhs;
    lhs.proxy_id = static_cast<std::uint32_t>(built.primitives.size());
    lhs.object_id = 1;
    lhs.patch_id = lhs_patch_id;
    lhs.slab_id = slab_id;
    lhs.motion_segment_id = 0;
    lhs.proxy_type = ProxyType::kSweptAabb;
    lhs.t0 = t0;
    lhs.t1 = t1;
    lhs.bounds = lhs_translated;
    lhs.motion_bound = MotionBoundForAabb(lhs_patch_id, t0, t1, lhs_translated);
    built.primitives.push_back(lhs);
    (*query_index_by_patch_id)[lhs_patch_id] = query_index;

    ProxyPrimitive rhs;
    rhs.proxy_id = static_cast<std::uint32_t>(built.primitives.size());
    rhs.object_id = 2;
    rhs.patch_id = rhs_patch_id;
    rhs.slab_id = slab_id;
    rhs.motion_segment_id = 0;
    rhs.proxy_type = ProxyType::kSweptAabb;
    rhs.t0 = t0;
    rhs.t1 = t1;
    rhs.bounds = rhs_translated;
    rhs.motion_bound = MotionBoundForAabb(rhs_patch_id, t0, t1, rhs_translated);
    built.primitives.push_back(rhs);
    (*query_index_by_patch_id)[rhs_patch_id] = query_index;

    x_cursor += (query_max_x - query_min_x) + kBatchXMargin;
  }

  if (auto status = ValidateProxyScene(built); !status.ok) {
    return status;
  }
  *scene = std::move(built);
  return Status::Ok();
}

}  // namespace

Status GenerateCandidatesForExternalBatch(const std::vector<ExternalBatchQuery>& queries,
                                          const CandidateGeneratorConfig& config,
                                          ExternalBatchCandidateResult* result) {
  if (result == nullptr) {
    return Status::Error("external batch candidate result output pointer is null");
  }
  if (queries.empty()) {
    return Status::Error("external batch candidate generation requires at least one query");
  }

  const Clock::time_point build_start = Clock::now();
  std::vector<QueryBuildInfo> infos;
  std::unordered_map<std::uint32_t, std::size_t> query_index_by_patch_id;
  ProxyScene scene;
  if (auto status = BuildBatchedProxyScene(queries, &infos, &query_index_by_patch_id, &scene);
      !status.ok) {
    return status;
  }
  const double encode_build_ms = ElapsedMs(build_start);

  CandidateGenerator generator(config);
  CandidateGenerationResult generation;
  if (auto status = generator.GenerateCandidates(scene, scene.query_id, &generation); !status.ok) {
    return status;
  }

  const Clock::time_point compact_start = Clock::now();
  std::vector<CandidateRecord> translated_candidates;
  translated_candidates.reserve(generation.candidates.size());
  std::unordered_set<std::size_t> active_query_indices;
  for (const CandidateRecord& candidate : generation.candidates) {
    const auto lhs_iter = query_index_by_patch_id.find(candidate.patch_a_id);
    const auto rhs_iter = query_index_by_patch_id.find(candidate.patch_b_id);
    if (lhs_iter == query_index_by_patch_id.end() || rhs_iter == query_index_by_patch_id.end()) {
      return Status::Error("external batch candidate references an unknown patch_id");
    }
    if (lhs_iter->second != rhs_iter->second) {
      return Status::Error("external batch candidate crossed query boundaries");
    }

    const QueryBuildInfo& info = infos[lhs_iter->second];
    CandidateRecord translated = candidate;
    translated.query_id = info.runtime_query_id;
    translated.object_a_id = info.object_a_id;
    translated.patch_a_id =
        info.family == ExternalQueryFamily::kVertexFace ? 0U : 1U;
    translated.object_b_id = info.object_b_id;
    translated.patch_b_id = info.source_query_index + 1U;
    translated.proxy_type_a = ProxyType::kSweptAabb;
    translated.proxy_type_b = ProxyType::kSweptAabb;
    translated_candidates.push_back(translated);
    active_query_indices.insert(lhs_iter->second);
  }
  std::sort(translated_candidates.begin(),
            translated_candidates.end(),
            [](const CandidateRecord& lhs, const CandidateRecord& rhs) {
              if (lhs.query_id != rhs.query_id) {
                return lhs.query_id < rhs.query_id;
              }
              if (lhs.patch_a_id != rhs.patch_a_id) {
                return lhs.patch_a_id < rhs.patch_a_id;
              }
              return lhs.patch_b_id < rhs.patch_b_id;
            });
  for (std::size_t ordinal = 0; ordinal < translated_candidates.size(); ++ordinal) {
    translated_candidates[ordinal].candidate_id =
        MakeCandidateId(translated_candidates[ordinal].query_id, ordinal);
    if (auto status = ValidateCandidateRecord(translated_candidates[ordinal]); !status.ok) {
      return status;
    }
  }
  const double compact_ms = ElapsedMs(compact_start);

  std::uint64_t positive_total = 0;
  std::uint64_t positive_covered = 0;
  for (std::size_t query_index = 0; query_index < infos.size(); ++query_index) {
    const QueryBuildInfo& info = infos[query_index];
    if (!info.has_ground_truth || !info.ground_truth_collides) {
      continue;
    }
    ++positive_total;
    if (active_query_indices.contains(query_index)) {
      ++positive_covered;
    }
  }

  result->backend_name = generation.backend_name;
  result->primitive_count = scene.primitives.size();
  result->raw_hit_count = generation.raw_buffer.hits.size();
  result->compact_candidate_count = translated_candidates.size();
  result->candidate_recall =
      positive_total == 0 ? 1.0 : static_cast<double>(positive_covered) / static_cast<double>(positive_total);
  result->candidates = std::move(translated_candidates);
  result->runtime_query_ids.clear();
  result->runtime_query_ids.reserve(infos.size());
  for (const QueryBuildInfo& info : infos) {
    result->runtime_query_ids.push_back({info.source_query_id, info.runtime_query_id});
  }

  result->timing.build_ms = encode_build_ms + generation.timing.build_ms;
  result->timing.update_ms = generation.timing.update_ms;
  result->timing.trace_ms =
      generation.timing.trace_ms + generation.timing.compact_ms + generation.timing.stats_ms;
  result->timing.compact_ms = compact_ms;
  result->timing.stats_ms = 0.0;
  result->timing.total_ms = result->timing.build_ms + result->timing.update_ms +
                            result->timing.trace_ms + result->timing.compact_ms;
  return Status::Ok();
}

}  // namespace p2cccd
