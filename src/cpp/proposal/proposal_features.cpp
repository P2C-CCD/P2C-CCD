#include "proposal/proposal_features.h"

#include "common/validators.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>

namespace p2cccd {
namespace {

constexpr std::uint32_t kTargetMaskInterval = 1U << 0U;
constexpr std::uint32_t kTargetMaskFamily = 1U << 1U;
constexpr std::uint32_t kTargetMaskPriority = 1U << 2U;
constexpr std::uint32_t kTargetMaskCost = 1U << 3U;
constexpr std::uint32_t kTargetMaskUncertainty = 1U << 4U;

double SafeRatio(std::uint64_t numerator, std::uint64_t denominator) {
  if (denominator == 0) {
    return 0.0;
  }
  return static_cast<double>(numerator) / static_cast<double>(denominator);
}

double AabbExtent(const Aabb& bounds, std::uint32_t axis) {
  return std::max(0.0, bounds.max[axis] - bounds.min[axis]);
}

double AabbVolume(const Aabb& bounds) {
  return AabbExtent(bounds, 0) * AabbExtent(bounds, 1) * AabbExtent(bounds, 2);
}

double AabbSurfaceArea(const Aabb& bounds) {
  const double x = AabbExtent(bounds, 0);
  const double y = AabbExtent(bounds, 1);
  const double z = AabbExtent(bounds, 2);
  return 2.0 * (x * y + y * z + z * x);
}

std::array<double, 3> AabbCenter(const Aabb& bounds) {
  return {
      0.5 * (bounds.min[0] + bounds.max[0]),
      0.5 * (bounds.min[1] + bounds.max[1]),
      0.5 * (bounds.min[2] + bounds.max[2]),
  };
}

double Distance(std::array<double, 3> lhs, std::array<double, 3> rhs) {
  const double dx = lhs[0] - rhs[0];
  const double dy = lhs[1] - rhs[1];
  const double dz = lhs[2] - rhs[2];
  return std::sqrt(dx * dx + dy * dy + dz * dz);
}

double OverlapExtent(const Aabb& lhs, const Aabb& rhs, std::uint32_t axis) {
  return std::max(0.0, std::min(lhs.max[axis], rhs.max[axis]) -
                           std::max(lhs.min[axis], rhs.min[axis]));
}

double OverlapVolume(const Aabb& lhs, const Aabb& rhs) {
  return OverlapExtent(lhs, rhs, 0) * OverlapExtent(lhs, rhs, 1) *
         OverlapExtent(lhs, rhs, 2);
}

double UnionVolume(const Aabb& lhs, const Aabb& rhs) {
  const double lhs_volume = AabbVolume(lhs);
  const double rhs_volume = AabbVolume(rhs);
  const double overlap_volume = OverlapVolume(lhs, rhs);
  return std::max(0.0, lhs_volume + rhs_volume - overlap_volume);
}

float ClampFeature(double value) {
  if (!std::isfinite(value)) {
    return 0.0F;
  }
  const double clamped = std::clamp(value, -1.0e6, 1.0e6);
  return static_cast<float>(clamped);
}

std::uint32_t IntervalBin(double t0, double t1) {
  const double center = 0.5 * (t0 + t1);
  const double clamped = std::clamp(center, 0.0, std::nextafter(1.0, 0.0));
  return static_cast<std::uint32_t>(clamped * kProposalIntervalBinCount);
}

bool CandidateMatchesPrimitive(const CandidateRecord& candidate,
                               const ProxyPrimitive& primitive,
                               bool first_side) {
  if (primitive.slab_id != candidate.slab_id) {
    return false;
  }
  if (first_side) {
    return primitive.object_id == candidate.object_a_id &&
           primitive.patch_id == candidate.patch_a_id &&
           primitive.proxy_type == candidate.proxy_type_a;
  }
  return primitive.object_id == candidate.object_b_id &&
         primitive.patch_id == candidate.patch_b_id &&
         primitive.proxy_type == candidate.proxy_type_b;
}

const ProxyPrimitive* FindCandidatePrimitive(const ProxyScene& scene,
                                             const CandidateRecord& candidate,
                                             bool first_side) {
  for (const ProxyPrimitive& primitive : scene.primitives) {
    if (CandidateMatchesPrimitive(candidate, primitive, first_side)) {
      return &primitive;
    }
  }
  return nullptr;
}

Status FillFeatureRow(const ProxyScene& scene,
                      const CandidateGenerationResult& generation_result,
                      const CandidateRecord& candidate,
                      ProposalFeatureRow* row) {
  if (row == nullptr) {
    return Status::Error("proposal feature row pointer is null");
  }
  if (auto status = ValidateCandidateRecord(candidate); !status.ok) {
    return status;
  }

  const ProxyPrimitive* proxy_a = FindCandidatePrimitive(scene, candidate, true);
  const ProxyPrimitive* proxy_b = FindCandidatePrimitive(scene, candidate, false);
  if (proxy_a == nullptr || proxy_b == nullptr) {
    return Status::Error("candidate does not map to proxy primitives");
  }

  ProposalFeatureRow built;
  built.query_id = candidate.query_id;
  built.candidate_id = candidate.candidate_id;
  built.slab_id = candidate.slab_id;
  built.object_a_id = candidate.object_a_id;
  built.patch_a_id = candidate.patch_a_id;
  built.object_b_id = candidate.object_b_id;
  built.patch_b_id = candidate.patch_b_id;

  const Aabb& a = proxy_a->bounds;
  const Aabb& b = proxy_b->bounds;
  const double volume_a = AabbVolume(a);
  const double volume_b = AabbVolume(b);
  const double overlap_volume = OverlapVolume(a, b);
  const double union_volume = UnionVolume(a, b);
  const double center_distance = Distance(AabbCenter(a), AabbCenter(b));
  const double slab_width = std::max(0.0, proxy_a->t1 - proxy_a->t0);
  const CandidateDensityStats& density = generation_result.density;

  std::array<float, kProposalFeatureDimension> features{};
  features[0] = ClampFeature(proxy_a->t0);
  features[1] = ClampFeature(proxy_a->t1);
  features[2] = ClampFeature(slab_width);
  features[3] = ClampFeature(candidate.rt_hit_count);
  features[4] = ClampFeature(static_cast<std::uint8_t>(candidate.proxy_type_a));
  features[5] = ClampFeature(static_cast<std::uint8_t>(candidate.proxy_type_b));
  features[6] = ClampFeature(candidate.motion_bound[0]);
  features[7] = ClampFeature(candidate.motion_bound[1]);
  features[8] = ClampFeature(candidate.motion_bound[2]);
  features[9] = ClampFeature(candidate.motion_bound[3]);
  features[10] = ClampFeature(AabbExtent(a, 0));
  features[11] = ClampFeature(AabbExtent(a, 1));
  features[12] = ClampFeature(AabbExtent(a, 2));
  features[13] = ClampFeature(AabbExtent(b, 0));
  features[14] = ClampFeature(AabbExtent(b, 1));
  features[15] = ClampFeature(AabbExtent(b, 2));
  features[16] = ClampFeature(volume_a);
  features[17] = ClampFeature(volume_b);
  features[18] = ClampFeature(overlap_volume);
  features[19] = ClampFeature(union_volume > 0.0 ? overlap_volume / union_volume : 0.0);
  features[20] = ClampFeature(center_distance);
  features[21] = ClampFeature(AabbSurfaceArea(a));
  features[22] = ClampFeature(AabbSurfaceArea(b));
  features[23] = ClampFeature(proxy_a->motion_bound.conservative_radius);
  features[24] = ClampFeature(proxy_b->motion_bound.conservative_radius);
  features[25] = ClampFeature(proxy_a->motion_bound.translation_bound);
  features[26] = ClampFeature(proxy_b->motion_bound.translation_bound);
  features[27] = ClampFeature(proxy_a->motion_bound.rotation_angle);
  features[28] = ClampFeature(proxy_b->motion_bound.rotation_angle);
  features[29] = ClampFeature(density.candidates_per_proxy);
  features[30] = ClampFeature(density.aabb_overlap_ratio);
  features[31] = ClampFeature(density.avg_rt_hits_per_candidate);
  built.features = features;

  built.interval_targets.fill(0.0F);
  built.interval_targets[IntervalBin(proxy_a->t0, proxy_a->t1)] = 1.0F;
  built.family_targets.fill(0.0F);
  built.family_targets[0] = 1.0F;
  built.family_targets[1] = 1.0F;
  built.priority_target =
      ClampFeature(0.5 * static_cast<double>(candidate.rt_hit_count) +
                   0.5 * density.candidates_per_proxy);
  built.cost_target = ClampFeature(1.0 + candidate.motion_bound[3]);
  built.uncertainty_target = ClampFeature(1.0 - std::min(1.0, static_cast<double>(features[19])));
  built.target_mask = kTargetMaskInterval | kTargetMaskFamily | kTargetMaskPriority |
                      kTargetMaskCost | kTargetMaskUncertainty;

  *row = built;
  return Status::Ok();
}

Status EnsureParentDirectory(const std::filesystem::path& path) {
  const std::filesystem::path parent = path.parent_path();
  if (parent.empty()) {
    return Status::Ok();
  }
  std::error_code ec;
  std::filesystem::create_directories(parent, ec);
  if (ec) {
    return Status::Error("failed to create proposal feature export directory: " +
                         ec.message());
  }
  return Status::Ok();
}

void WriteFeatureArrayCsv(std::ostream& out, const auto& values) {
  for (float value : values) {
    out << ',' << value;
  }
}

}  // namespace

Status BuildRawCandidateQueue(const CandidateGenerationResult& generation_result,
                              RawCandidateQueue* queue) {
  if (queue == nullptr) {
    return Status::Error("raw candidate queue pointer is null");
  }
  if (generation_result.density.query_id == 0) {
    return Status::Error("candidate generation density query_id is required");
  }
  for (const CandidateRecord& candidate : generation_result.candidates) {
    if (auto status = ValidateCandidateRecord(candidate); !status.ok) {
      return status;
    }
  }

  RawCandidateQueue built;
  built.query_id = generation_result.density.query_id;
  built.candidates = generation_result.candidates;
  built.density = generation_result.density;
  *queue = std::move(built);
  return Status::Ok();
}

Status ExtractProposalFeatureRows(const ProxyScene& scene,
                                  const CandidateGenerationResult& generation_result,
                                  std::vector<ProposalFeatureRow>* rows) {
  if (rows == nullptr) {
    return Status::Error("proposal feature row output pointer is null");
  }
  if (auto status = ValidateProxyScene(scene); !status.ok) {
    return status;
  }
  if (generation_result.density.query_id != scene.query_id) {
    return Status::Error("candidate density query_id must match proxy scene query_id");
  }

  rows->clear();
  rows->reserve(generation_result.candidates.size());
  for (const CandidateRecord& candidate : generation_result.candidates) {
    ProposalFeatureRow row;
    if (auto status = FillFeatureRow(scene, generation_result, candidate, &row); !status.ok) {
      return status;
    }
    rows->push_back(row);
  }
  return Status::Ok();
}

Status BuildExactWorkQueuePassthrough(const ProxyScene& scene,
                                      const RawCandidateQueue& raw_queue,
                                      const ProposalQueueConfig& config,
                                      std::vector<ExactWorkItem>* work_queue) {
  if (work_queue == nullptr) {
    return Status::Error("exact work queue output pointer is null");
  }
  if (auto status = ValidateProxyScene(scene); !status.ok) {
    return status;
  }
  if (raw_queue.query_id != scene.query_id) {
    return Status::Error("raw candidate queue query_id must match proxy scene query_id");
  }
  if (config.first_work_item_id == 0) {
    return Status::Error("ProposalQueueConfig.first_work_item_id must be non-zero");
  }
  if (config.feature_family_mask == 0) {
    return Status::Error("ProposalQueueConfig.feature_family_mask must be non-zero");
  }
  if (!std::isfinite(config.fallback_interval_t0) ||
      !std::isfinite(config.fallback_interval_t1) ||
      config.fallback_interval_t0 < 0.0 || config.fallback_interval_t1 > 1.0 ||
      config.fallback_interval_t0 > config.fallback_interval_t1) {
    return Status::Error("ProposalQueueConfig fallback interval must lie in [0, 1]");
  }

  work_queue->clear();
  work_queue->reserve(raw_queue.candidates.size());
  std::uint64_t next_work_item_id = config.first_work_item_id;
  for (const CandidateRecord& candidate : raw_queue.candidates) {
    if (auto status = ValidateCandidateRecord(candidate); !status.ok) {
      return status;
    }

    const ProxyPrimitive* primitive = FindCandidatePrimitive(scene, candidate, true);
    const double interval_t0 = primitive != nullptr ? primitive->t0 : config.fallback_interval_t0;
    const double interval_t1 = primitive != nullptr ? primitive->t1 : config.fallback_interval_t1;

    ExactWorkItem item;
    item.work_item_id = next_work_item_id++;
    item.parent_candidate_id = candidate.candidate_id;
    item.query_id = candidate.query_id;
    item.slab_id = candidate.slab_id;
    item.patch_a_id = candidate.patch_a_id;
    item.patch_b_id = candidate.patch_b_id;
    item.interval_t0 = interval_t0;
    item.interval_t1 = interval_t1;
    item.feature_family_mask = config.feature_family_mask;
    item.priority_score = static_cast<float>(candidate.rt_hit_count);
    item.source = ProposalSource::kRaw;
    if (auto status = ValidateExactWorkItem(item); !status.ok) {
      return status;
    }
    work_queue->push_back(item);
  }
  return Status::Ok();
}

Status BuildProposalDataFlow(const ProxyScene& scene,
                             const CandidateGenerationResult& generation_result,
                             const ProposalQueueConfig& config,
                             ProposalDataFlow* data_flow) {
  if (data_flow == nullptr) {
    return Status::Error("proposal data flow output pointer is null");
  }

  ProposalDataFlow built;
  if (auto status = BuildRawCandidateQueue(generation_result, &built.raw_candidate_queue);
      !status.ok) {
    return status;
  }
  if (auto status = ExtractProposalFeatureRows(scene, generation_result, &built.feature_rows);
      !status.ok) {
    return status;
  }
  if (auto status = BuildExactWorkQueuePassthrough(scene,
                                                   built.raw_candidate_queue,
                                                   config,
                                                   &built.exact_work_queue);
      !status.ok) {
    return status;
  }
  if (auto status = ValidateProposalDataFlow(built); !status.ok) {
    return status;
  }
  *data_flow = std::move(built);
  return Status::Ok();
}

Status ValidateProposalDataFlow(const ProposalDataFlow& data_flow) {
  const std::size_t candidate_count = data_flow.raw_candidate_queue.candidates.size();
  if (data_flow.raw_candidate_queue.query_id == 0) {
    return Status::Error("raw candidate queue query_id is required");
  }
  if (data_flow.feature_rows.size() != candidate_count) {
    return Status::Error("proposal feature row count must match candidate count");
  }
  if (data_flow.exact_work_queue.size() != candidate_count) {
    return Status::Error("exact work queue count must match candidate count");
  }

  std::set<std::uint64_t> candidate_ids;
  std::set<std::uint64_t> work_item_ids;
  for (const CandidateRecord& candidate : data_flow.raw_candidate_queue.candidates) {
    if (auto status = ValidateCandidateRecord(candidate); !status.ok) {
      return status;
    }
    if (!candidate_ids.insert(candidate.candidate_id).second) {
      return Status::Error("raw candidate queue contains duplicate candidate_id");
    }
  }

  for (std::size_t i = 0; i < candidate_count; ++i) {
    const CandidateRecord& candidate = data_flow.raw_candidate_queue.candidates[i];
    const ProposalFeatureRow& row = data_flow.feature_rows[i];
    const ExactWorkItem& item = data_flow.exact_work_queue[i];
    if (row.schema_version != kProposalFeatureRowSchemaVersion) {
      return Status::Error("ProposalFeatureRow.schema_version is unsupported");
    }
    if (row.query_id != candidate.query_id || row.candidate_id != candidate.candidate_id) {
      return Status::Error("ProposalFeatureRow must align with raw candidate order");
    }
    for (float value : row.features) {
      if (!std::isfinite(value)) {
        return Status::Error("ProposalFeatureRow features must be finite");
      }
    }
    if (auto status = ValidateExactWorkItem(item); !status.ok) {
      return status;
    }
    if (item.parent_candidate_id != candidate.candidate_id || item.query_id != candidate.query_id) {
      return Status::Error("ExactWorkItem must reference its parent candidate");
    }
    if (!work_item_ids.insert(item.work_item_id).second) {
      return Status::Error("exact work queue contains duplicate work_item_id");
    }
  }

  return Status::Ok();
}

std::string ProposalFeatureCsvHeader() {
  std::ostringstream header;
  header << "schema_version,query_id,candidate_id,slab_id,object_a_id,patch_a_id,"
         << "object_b_id,patch_b_id";
  for (std::uint32_t i = 0; i < kProposalFeatureDimension; ++i) {
    header << ",feature_" << i;
  }
  for (std::uint32_t i = 0; i < kProposalIntervalBinCount; ++i) {
    header << ",interval_target_" << i;
  }
  for (std::uint32_t i = 0; i < kProposalFamilyCount; ++i) {
    header << ",family_target_" << i;
  }
  header << ",priority_target,cost_target,uncertainty_target,target_mask";
  return header.str();
}

Status WriteProposalFeatureRowsCsv(const std::filesystem::path& path,
                                   const std::vector<ProposalFeatureRow>& rows,
                                   const bool append) {
  if (auto status = EnsureParentDirectory(path); !status.ok) {
    return status;
  }

  const bool write_header = !append || !std::filesystem::exists(path) ||
                            std::filesystem::file_size(path) == 0;
  std::ofstream out(path, append ? std::ios::app : std::ios::trunc);
  if (!out) {
    return Status::Error("failed to open proposal feature CSV export");
  }

  out << std::setprecision(9);
  if (write_header) {
    out << ProposalFeatureCsvHeader() << '\n';
  }
  for (const ProposalFeatureRow& row : rows) {
    out << row.schema_version << ',' << row.query_id << ',' << row.candidate_id << ','
        << row.slab_id << ',' << row.object_a_id << ',' << row.patch_a_id << ','
        << row.object_b_id << ',' << row.patch_b_id;
    WriteFeatureArrayCsv(out, row.features);
    WriteFeatureArrayCsv(out, row.interval_targets);
    WriteFeatureArrayCsv(out, row.family_targets);
    out << ',' << row.priority_target << ',' << row.cost_target << ','
        << row.uncertainty_target << ',' << row.target_mask << '\n';
  }
  return Status::Ok();
}

}  // namespace p2cccd
