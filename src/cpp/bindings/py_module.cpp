#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "certificate/certificate_cuda.h"
#include "certificate/certificate_engine.h"
#include "certificate/mesh_exact_query.h"
#include "common/group_scheduler.h"
#include "common/runtime_contracts.h"
#include "common/validators.h"
#include "geometry/motion.h"
#include "geometry/mesh_io.h"
#include "geometry/motion_utils.h"
#include "geometry/patch.h"
#include "geometry/proxy.h"
#include "proposal/proposal_policy.h"
#include "rt_candidate/candidate_buffer.h"
#include "rt_candidate/candidate_generation_result.h"
#include "rt_candidate/candidate_generator.h"
#include "rt_candidate/external_batch_candidate.h"
#include "rt_candidate/proxy_scene.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace py = pybind11;

namespace p2cccd {
namespace {

void ThrowIfError(const Status& status) {
  if (!status.ok) {
    throw py::value_error(status.message);
  }
}

Mesh LoadTriangleMeshOrThrow(const py::handle& path) {
  const std::filesystem::path native_path(py::cast<std::wstring>(path));
  Mesh mesh;
  ThrowIfError(LoadTriangleMesh(native_path, &mesh));
  return mesh;
}

bool ValidateTriangleMeshOrThrow(const Mesh& mesh) {
  ThrowIfError(ValidateTriangleMesh(mesh));
  return true;
}

py::tuple CenterMeshAtAabbCenterOrThrow(const Mesh& mesh) {
  Mesh centered;
  std::array<double, 3> center{};
  ThrowIfError(CenterMeshAtAabbCenter(mesh, &centered, &center));
  return py::make_tuple(centered, center);
}

MeshExactBuildResult BuildMeshExactCertificateQueryOrThrow(
    const Mesh& mesh_a,
    const std::array<double, 3>& translation_a_t0,
    const std::array<double, 3>& translation_a_t1,
    const Mesh& mesh_b,
    const std::array<double, 3>& translation_b_t0,
    const std::array<double, 3>& translation_b_t1,
    const ExactWorkItem& work_item,
    const CertificateEngineConfig& config,
    const MeshExactBuildConfig& build_config) {
  MeshExactBuildResult result;
  ThrowIfError(BuildMeshExactCertificateQuery(mesh_a,
                                              translation_a_t0,
                                              translation_a_t1,
                                              mesh_b,
                                              translation_b_t0,
                                              translation_b_t1,
                                              work_item,
                                              config,
                                              build_config,
                                              &result));
  return result;
}

CandidateBackend ParseCandidateBackend(const std::string& backend_name) {
  if (backend_name == "cpu_reference" || backend_name == "cpu_reference_rt") {
    return CandidateBackend::kCpuReference;
  }
  if (backend_name == "optix" || backend_name == "optix_compatible" || backend_name == "optix_rt") {
    return CandidateBackend::kOptix;
  }
  throw py::value_error("unsupported candidate backend: " + backend_name);
}

ProxyScene BuildProxySceneOrThrow(const ProxySceneBuildInput& input) {
  ProxyScene scene;
  ThrowIfError(BuildProxyScene(input, &scene));
  return scene;
}

CandidateGenerationResult GenerateCandidatesForProxyScene(
    const ProxyScene& scene,
    std::uint64_t query_id,
    const std::string& backend_name,
    bool allow_optix_cpu_fallback) {
  CandidateGeneratorConfig config;
  config.backend = ParseCandidateBackend(backend_name);
  config.allow_optix_cpu_fallback = allow_optix_cpu_fallback;

  CandidateGenerationResult result;
  CandidateGenerator generator(config);
  ThrowIfError(generator.GenerateCandidates(scene, query_id == 0 ? scene.query_id : query_id, &result));
  return result;
}

ExternalQueryFamily ParseExternalQueryFamily(const py::handle& family) {
  const std::string value = py::str(family);
  if (value == "vf") {
    return ExternalQueryFamily::kVertexFace;
  }
  if (value == "ee") {
    return ExternalQueryFamily::kEdgeEdge;
  }
  throw py::value_error("unsupported external CCD query family: " + value);
}

std::array<std::array<double, 3>, 4> ParseVertexArray(const py::handle& vertices,
                                                      const char* field_name) {
  const py::sequence sequence = py::cast<py::sequence>(vertices);
  if (py::len(sequence) != 4) {
    throw py::value_error(std::string(field_name) + " must contain four vertices");
  }
  std::array<std::array<double, 3>, 4> parsed{};
  for (std::size_t i = 0; i < parsed.size(); ++i) {
    const py::sequence vertex = py::cast<py::sequence>(sequence[i]);
    if (py::len(vertex) != 3) {
      throw py::value_error(std::string(field_name) + " vertices must be 3D");
    }
    for (std::size_t axis = 0; axis < 3; ++axis) {
      parsed[i][axis] = py::cast<double>(vertex[axis]);
    }
  }
  return parsed;
}

std::vector<ExternalBatchQuery> ParseExternalBatchQueries(const py::object& batch) {
  const py::sequence queries = py::cast<py::sequence>(batch.attr("queries"));
  std::vector<ExternalBatchQuery> parsed;
  parsed.reserve(py::len(queries));
  for (const py::handle& item : queries) {
    ExternalBatchQuery query;
    query.source_query_id = py::cast<std::uint64_t>(item.attr("query_id"));
    query.source_query_index = py::cast<std::uint32_t>(item.attr("source_query_index"));
    query.family = ParseExternalQueryFamily(item.attr("family"));
    query.vertices_t0 = ParseVertexArray(item.attr("vertices_t0"), "vertices_t0");
    query.vertices_t1 = ParseVertexArray(item.attr("vertices_t1"), "vertices_t1");

    py::object ground_truth = item.attr("ground_truth_collides");
    if (!ground_truth.is_none()) {
      query.has_ground_truth = true;
      query.ground_truth_collides = py::cast<bool>(ground_truth);
    }

    py::object box_pair = item.attr("box_pair");
    if (!box_pair.is_none()) {
      const py::sequence pair = py::cast<py::sequence>(box_pair);
      if (py::len(pair) != 2) {
        throw py::value_error("box_pair must contain exactly two object ids");
      }
      query.has_box_pair = true;
      query.box_pair = {
          py::cast<std::uint32_t>(pair[0]),
          py::cast<std::uint32_t>(pair[1]),
      };
    }
    parsed.push_back(query);
  }
  return parsed;
}

ExternalBatchCandidateResult GenerateCandidatesForExternalBatchOrThrow(
    const py::object& batch,
    const std::string& backend_name,
    bool allow_optix_cpu_fallback) {
  CandidateGeneratorConfig config;
  config.backend = ParseCandidateBackend(backend_name);
  config.allow_optix_cpu_fallback = allow_optix_cpu_fallback;

  const std::vector<ExternalBatchQuery> queries = ParseExternalBatchQueries(batch);
  ExternalBatchCandidateResult result;
  Status status;
  {
    py::gil_scoped_release release;
    status = GenerateCandidatesForExternalBatch(queries, config, &result);
  }
  ThrowIfError(status);
  return result;
}

struct ExternalBatchDummyProposalScheduleResult {
  ExternalBatchCandidateResult candidate_result;
  std::vector<ProposalFeatureRow> feature_rows;
  std::vector<ProposalOutput> proposal_outputs;
  std::vector<ExactWorkItem> work_queue;
  ProposalScheduleStats stats;
  double proposal_elapsed_ms = 0.0;
};

std::uint32_t ConservativeFamilyMaskForExternalQueryFamily(const ExternalQueryFamily family) {
  switch (family) {
    case ExternalQueryFamily::kVertexFace:
      return kFeatureFamilyPointTriangle;
    case ExternalQueryFamily::kEdgeEdge:
      return kFeatureFamilyEdgeEdge;
  }
  return kFeatureFamilyPointTriangle | kFeatureFamilyEdgeEdge;
}

std::unordered_map<std::uint64_t, std::uint32_t> ConservativeFamilyMasksByRuntimeQueryId(
    const std::vector<ExternalBatchQuery>& queries,
    const std::vector<RuntimeQueryIdMapping>& runtime_query_ids) {
  std::unordered_map<std::uint64_t, std::uint32_t> source_masks;
  source_masks.reserve(queries.size());
  for (const ExternalBatchQuery& query : queries) {
    source_masks.emplace(query.source_query_id,
                         ConservativeFamilyMaskForExternalQueryFamily(query.family));
  }

  std::unordered_map<std::uint64_t, std::uint32_t> runtime_masks;
  runtime_masks.reserve(runtime_query_ids.size());
  for (const RuntimeQueryIdMapping& mapping : runtime_query_ids) {
    const auto mask_it = source_masks.find(mapping.source_query_id);
    const std::uint32_t family_mask =
        mask_it != source_masks.end()
            ? mask_it->second
            : (kFeatureFamilyPointTriangle | kFeatureFamilyEdgeEdge);
    runtime_masks.emplace(mapping.runtime_query_id, family_mask);
  }
  return runtime_masks;
}

bool HasSingleExactFamily(const std::uint32_t family_mask) {
  return family_mask != 0U && (family_mask & (family_mask - 1U)) == 0U;
}

bool CanBypassDummyProposalSchedule(
    const std::vector<CandidateRecord>& candidates,
    const std::unordered_map<std::uint64_t, std::uint32_t>& runtime_masks) {
  std::unordered_map<std::uint64_t, std::uint8_t> seen_query_ids;
  seen_query_ids.reserve(candidates.size());
  for (const CandidateRecord& candidate : candidates) {
    if (!seen_query_ids.emplace(candidate.query_id, 1U).second) {
      return false;
    }
    const auto mask_it = runtime_masks.find(candidate.query_id);
    const std::uint32_t family_mask =
        mask_it != runtime_masks.end()
            ? mask_it->second
            : (kFeatureFamilyPointTriangle | kFeatureFamilyEdgeEdge);
    if (!HasSingleExactFamily(family_mask)) {
      return false;
    }
  }
  return true;
}

ExternalBatchDummyProposalScheduleResult GenerateExternalBatchDummyProposalScheduleOrThrow(
    const py::object& batch,
    const std::string& backend_name,
    bool allow_optix_cpu_fallback,
    const ProposalSchedulingConfig& scheduling_config,
    const bool materialize_artifacts) {
  CandidateGeneratorConfig candidate_config;
  candidate_config.backend = ParseCandidateBackend(backend_name);
  candidate_config.allow_optix_cpu_fallback = allow_optix_cpu_fallback;

  const std::vector<ExternalBatchQuery> queries = ParseExternalBatchQueries(batch);
  ExternalBatchDummyProposalScheduleResult result;
  Status status;
  {
    py::gil_scoped_release release;
    status = GenerateCandidatesForExternalBatch(queries, candidate_config, &result.candidate_result);
    if (status.ok) {
      const auto runtime_masks = ConservativeFamilyMasksByRuntimeQueryId(
          queries, result.candidate_result.runtime_query_ids);
      const auto proposal_start = std::chrono::steady_clock::now();
      ProposalRuntimeScheduleResult schedule_result;
      if (CanBypassDummyProposalSchedule(result.candidate_result.candidates, runtime_masks)) {
        schedule_result.work_queue.reserve(result.candidate_result.candidates.size());
        schedule_result.stats.raw_candidate_count = result.candidate_result.candidates.size();
        schedule_result.stats.work_item_count = result.candidate_result.candidates.size();
        schedule_result.stats.monotonic_safe = true;
        for (std::size_t i = 0; i < result.candidate_result.candidates.size(); ++i) {
          const CandidateRecord& candidate = result.candidate_result.candidates[i];
          ExactWorkItem item;
          item.work_item_id = scheduling_config.first_work_item_id + i;
          item.parent_candidate_id = candidate.candidate_id;
          item.query_id = candidate.query_id;
          item.slab_id = candidate.slab_id;
          item.patch_a_id = candidate.patch_a_id;
          item.patch_b_id = candidate.patch_b_id;
          item.interval_t0 = scheduling_config.fallback_interval_t0;
          item.interval_t1 = scheduling_config.fallback_interval_t1;
          item.feature_family_mask = runtime_masks.at(candidate.query_id);
          item.topk_feature_ids_offset = 0U;
          item.depth = 0U;
          item.priority_score = static_cast<float>(candidate.rt_hit_count);
          item.source = ProposalSource::kRaw;
          status = ValidateExactWorkItem(item);
          if (!status.ok) {
            break;
          }
          schedule_result.work_queue.push_back(item);
        }
      } else {
        CandidateDensityStats density;
        density.query_id =
            result.candidate_result.candidates.empty()
                ? 1U
                : result.candidate_result.candidates.front().query_id;
        density.proxy_count = result.candidate_result.primitive_count;
        density.raw_hit_count = result.candidate_result.raw_hit_count;
        density.compact_candidate_count = result.candidate_result.compact_candidate_count;
        density.candidates_per_proxy =
            result.candidate_result.primitive_count == 0U
                ? 0.0
                : static_cast<double>(result.candidate_result.compact_candidate_count) /
                      static_cast<double>(result.candidate_result.primitive_count);
        const std::uint64_t overlap_denominator =
            result.candidate_result.primitive_count >= 2U
                ? result.candidate_result.primitive_count / 2U
                : 1U;
        density.aabb_overlap_ratio =
            static_cast<double>(result.candidate_result.raw_hit_count) /
            static_cast<double>(overlap_denominator);
        density.avg_rt_hits_per_candidate =
            result.candidate_result.compact_candidate_count == 0U
                ? 0.0
                : static_cast<double>(result.candidate_result.raw_hit_count) /
                      static_cast<double>(result.candidate_result.compact_candidate_count);
        status = RunDummyProposalScheduleFromRuntimeCandidates(result.candidate_result.candidates,
                                                               density,
                                                               runtime_masks,
                                                               scheduling_config,
                                                               materialize_artifacts,
                                                               &schedule_result);
      }
      result.proposal_elapsed_ms =
          std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() -
                                                    proposal_start)
              .count();
      if (status.ok) {
        result.feature_rows = std::move(schedule_result.feature_rows);
        result.proposal_outputs = std::move(schedule_result.proposal_outputs);
        result.work_queue = std::move(schedule_result.work_queue);
        result.stats = schedule_result.stats;
      }
    }
  }
  ThrowIfError(status);
  return result;
}

RawCandidateBuffer GenerateRawCandidatesCpuOrThrow(const ProxyScene& scene,
                                                   std::uint64_t query_id) {
  RawCandidateBuffer buffer;
  ThrowIfError(GenerateRawCandidatesCpu(scene, query_id == 0 ? scene.query_id : query_id, &buffer));
  return buffer;
}

std::vector<CandidateRecord> CompactRawCandidatesOrThrow(
    const ProxyScene& scene,
    const RawCandidateBuffer& raw_buffer) {
  std::vector<CandidateRecord> candidates;
  ThrowIfError(CompactRawCandidates(scene, raw_buffer, &candidates));
  return candidates;
}

PrimitiveIntervalResult EvaluatePointTriangleIntervalOrThrow(
    const PointTriangleIntervalPrimitive& primitive,
    double interval_t0,
    double interval_t1,
    const CertificateEngineConfig& config) {
  PrimitiveIntervalResult result;
  ThrowIfError(EvaluatePointTriangleInterval(primitive, interval_t0, interval_t1, config, &result));
  return result;
}

PrimitiveIntervalResult EvaluateEdgeEdgeIntervalOrThrow(
    const EdgeEdgeIntervalPrimitive& primitive,
    double interval_t0,
    double interval_t1,
    const CertificateEngineConfig& config) {
  PrimitiveIntervalResult result;
  ThrowIfError(EvaluateEdgeEdgeInterval(primitive, interval_t0, interval_t1, config, &result));
  return result;
}

std::vector<PrimitiveIntervalResult> EvaluatePointTriangleBatchCudaOrThrow(
    const std::vector<PointTriangleIntervalPrimitive>& primitives,
    double interval_t0,
    double interval_t1,
    const CertificateEngineConfig& config) {
  std::vector<PrimitiveIntervalResult> results;
  ThrowIfError(
      EvaluatePointTriangleBatchCuda(primitives, interval_t0, interval_t1, config, &results));
  return results;
}

std::vector<PrimitiveIntervalResult> EvaluateEdgeEdgeBatchCudaOrThrow(
    const std::vector<EdgeEdgeIntervalPrimitive>& primitives,
    double interval_t0,
    double interval_t1,
    const CertificateEngineConfig& config) {
  std::vector<PrimitiveIntervalResult> results;
  ThrowIfError(EvaluateEdgeEdgeBatchCuda(primitives, interval_t0, interval_t1, config, &results));
  return results;
}

bool CrossCheckCpuCudaExactOrThrow(
    const std::vector<PointTriangleIntervalPrimitive>& point_triangles,
    const std::vector<EdgeEdgeIntervalPrimitive>& edge_edges,
    double interval_t0,
    double interval_t1,
    const CertificateEngineConfig& config,
    double eps_cert) {
  ThrowIfError(
      CrossCheckCpuCudaExact(point_triangles, edge_edges, interval_t0, interval_t1, config, eps_cert));
  return true;
}

py::dict CudaBindingStatus() {
  py::dict status;
  status["cuda_exact_built"] = py::bool_(IsCudaExactBuilt());
  status["host_batch_exact_api"] = py::bool_(true);
  status["device_pointer_abi"] = py::bool_(false);
  status["backend_name"] = py::str(IsCudaExactBuilt() ? "cuda_exact" : "cuda_exact_stub");
  status["safety_policy"] = py::str(
      "Python bindings expose host-owned batch exact APIs only; raw CUDA device pointer "
      "ABI remains disabled until ownership and lifetime contracts are certified.");
  return status;
}

CertificateResult EvaluateCertificateQueryCpuOrThrow(const ExactCertificateQuery& query) {
  CertificateEngine engine;
  CertificateResult result;
  ThrowIfError(engine.Evaluate(query, &result));
  return result;
}

ExactWorkQueueResult ProcessExactWorkQueueCpuOrThrow(
    const std::vector<ExactCertificateQuery>& work_queue,
    const ExactWorkQueueConfig& config) {
  ExactWorkQueueResult result;
  ThrowIfError(ProcessExactWorkQueueCpu(work_queue, config, &result));
  return result;
}

std::vector<ExactWorkItem> GenerateConservativeRefinementWorkItemsOrThrow(
    const ExactWorkItem& parent,
    const CertificateResult& certificate,
    const ExactRefinementConfig& config) {
  std::vector<ExactWorkItem> children;
  ThrowIfError(GenerateConservativeRefinementWorkItems(parent, certificate, config, &children));
  return children;
}

bool ValidateAuditLogRowsOrThrow(const std::vector<AuditLogRow>& rows) {
  for (const AuditLogRow& row : rows) {
    ThrowIfError(ValidateAuditLogRow(row));
  }
  return true;
}

std::vector<AuditLogRow> AuditLogRowsForQuery(const std::vector<AuditLogRow>& rows,
                                              std::uint64_t query_id) {
  std::vector<AuditLogRow> filtered;
  for (const AuditLogRow& row : rows) {
    if (row.query_id == query_id) {
      filtered.push_back(row);
    }
  }
  return filtered;
}

bool ValidateCandidateRecordOrThrow(const CandidateRecord& record) {
  ThrowIfError(ValidateCandidateRecord(record));
  return true;
}

bool ValidateExactWorkItemOrThrow(const ExactWorkItem& item) {
  ThrowIfError(ValidateExactWorkItem(item));
  return true;
}

bool ValidateCertificateResultOrThrow(const CertificateResult& result) {
  ThrowIfError(ValidateCertificateResult(result));
  return true;
}

bool ValidateBenchmarkRowOrThrow(const BenchmarkRow& row) {
  ThrowIfError(ValidateBenchmarkRow(row));
  return true;
}

CandidateDensityStats MakeRuntimeDensityStats(
    const std::vector<CandidateRecord>& candidates,
    const std::uint64_t primitive_count,
    const std::uint64_t raw_hit_count,
    const std::uint64_t compact_candidate_count) {
  CandidateDensityStats density;
  density.query_id = candidates.empty() ? 1U : candidates.front().query_id;
  density.proxy_count = primitive_count;
  density.raw_hit_count = raw_hit_count;
  density.compact_candidate_count = compact_candidate_count;
  density.candidates_per_proxy =
      primitive_count == 0U ? 0.0 : static_cast<double>(compact_candidate_count) /
                                      static_cast<double>(primitive_count);
  const std::uint64_t overlap_denominator = primitive_count >= 2U ? primitive_count / 2U : 1U;
  density.aabb_overlap_ratio =
      overlap_denominator == 0U
          ? 0.0
          : static_cast<double>(raw_hit_count) / static_cast<double>(overlap_denominator);
  density.avg_rt_hits_per_candidate =
      compact_candidate_count == 0U
          ? 0.0
          : static_cast<double>(raw_hit_count) / static_cast<double>(compact_candidate_count);
  return density;
}

std::array<float, 4> ParseMotionBound(const py::handle& motion_bound) {
  const py::sequence values = py::cast<py::sequence>(motion_bound);
  if (py::len(values) != 4) {
    throw py::value_error("CandidateRecord.motion_bound must have length 4");
  }
  std::array<float, 4> parsed{};
  for (std::size_t i = 0; i < parsed.size(); ++i) {
    parsed[i] = py::cast<float>(values[i]);
  }
  return parsed;
}

template <std::size_t N>
std::array<float, N> ParseFloatArray(const py::handle& values_handle, const char* field_name) {
  const py::sequence values = py::cast<py::sequence>(values_handle);
  if (py::len(values) != static_cast<py::ssize_t>(N)) {
    throw py::value_error(std::string(field_name) + " must have length " + std::to_string(N));
  }
  std::array<float, N> parsed{};
  for (std::size_t i = 0; i < parsed.size(); ++i) {
    parsed[i] = py::cast<float>(values[i]);
  }
  return parsed;
}

CandidateRecord ParseRuntimeCandidate(const py::handle& item) {
  CandidateRecord candidate;
  candidate.schema_version = py::cast<std::uint32_t>(item.attr("schema_version"));
  candidate.candidate_id = py::cast<std::uint64_t>(item.attr("candidate_id"));
  candidate.query_id = py::cast<std::uint64_t>(item.attr("query_id"));
  candidate.slab_id = py::cast<std::uint32_t>(item.attr("slab_id"));
  candidate.object_a_id = py::cast<std::uint32_t>(item.attr("object_a_id"));
  candidate.patch_a_id = py::cast<std::uint32_t>(item.attr("patch_a_id"));
  candidate.object_b_id = py::cast<std::uint32_t>(item.attr("object_b_id"));
  candidate.patch_b_id = py::cast<std::uint32_t>(item.attr("patch_b_id"));
  candidate.proxy_type_a = static_cast<ProxyType>(py::cast<int>(item.attr("proxy_type_a")));
  candidate.proxy_type_b = static_cast<ProxyType>(py::cast<int>(item.attr("proxy_type_b")));
  candidate.rt_hit_count = py::cast<std::uint32_t>(item.attr("rt_hit_count"));
  candidate.motion_bound = ParseMotionBound(item.attr("motion_bound"));
  candidate.proxy_features_offset = py::cast<std::uint32_t>(item.attr("proxy_features_offset"));
  candidate.flags = py::cast<std::uint32_t>(item.attr("flags"));
  return candidate;
}

std::vector<CandidateRecord> ParseRuntimeCandidates(const py::sequence& candidates) {
  std::vector<CandidateRecord> parsed;
  parsed.reserve(py::len(candidates));
  for (const py::handle& item : candidates) {
    parsed.push_back(ParseRuntimeCandidate(item));
  }
  return parsed;
}

ProposalFeatureRow ParseProposalFeatureRow(const py::handle& item) {
  ProposalFeatureRow row;
  row.schema_version = py::cast<std::uint32_t>(item.attr("schema_version"));
  row.query_id = py::cast<std::uint64_t>(item.attr("query_id"));
  row.candidate_id = py::cast<std::uint64_t>(item.attr("candidate_id"));
  row.slab_id = py::cast<std::uint32_t>(item.attr("slab_id"));
  row.object_a_id = py::cast<std::uint32_t>(item.attr("object_a_id"));
  row.patch_a_id = py::cast<std::uint32_t>(item.attr("patch_a_id"));
  row.object_b_id = py::cast<std::uint32_t>(item.attr("object_b_id"));
  row.patch_b_id = py::cast<std::uint32_t>(item.attr("patch_b_id"));
  row.features = ParseFloatArray<kProposalFeatureDimension>(item.attr("features"), "ProposalFeatureRow.features");
  row.interval_targets = ParseFloatArray<kProposalIntervalBinCount>(
      item.attr("interval_targets"), "ProposalFeatureRow.interval_targets");
  row.family_targets = ParseFloatArray<kProposalFamilyCount>(
      item.attr("family_targets"), "ProposalFeatureRow.family_targets");
  row.priority_target = py::cast<float>(item.attr("priority_target"));
  row.cost_target = py::cast<float>(item.attr("cost_target"));
  row.uncertainty_target = py::cast<float>(item.attr("uncertainty_target"));
  row.target_mask = py::cast<std::uint32_t>(item.attr("target_mask"));
  return row;
}

std::vector<ProposalFeatureRow> ParseProposalFeatureRows(const py::sequence& rows) {
  std::vector<ProposalFeatureRow> parsed;
  parsed.reserve(py::len(rows));
  for (const py::handle& item : rows) {
    parsed.push_back(ParseProposalFeatureRow(item));
  }
  return parsed;
}

ProposalOutput ParseProposalOutput(const py::handle& item) {
  ProposalOutput output;
  output.candidate_id = py::cast<std::uint64_t>(item.attr("candidate_id"));
  output.interval_scores =
      ParseFloatArray<kProposalIntervalBinCount>(item.attr("interval_scores"), "ProposalOutput.interval_scores");
  output.family_scores =
      ParseFloatArray<kProposalFamilyCount>(item.attr("family_scores"), "ProposalOutput.family_scores");
  output.priority_score = py::cast<float>(item.attr("priority_score"));
  output.cost_score = py::cast<float>(item.attr("cost_score"));
  output.uncertainty_score = py::cast<float>(item.attr("uncertainty_score"));
  return output;
}

std::vector<ProposalOutput> ParseProposalOutputs(const py::sequence& outputs) {
  std::vector<ProposalOutput> parsed;
  parsed.reserve(py::len(outputs));
  for (const py::handle& item : outputs) {
    parsed.push_back(ParseProposalOutput(item));
  }
  return parsed;
}

std::vector<ProposalFeatureRow> BuildRuntimeProposalFeatureRowsOrThrow(
    const py::sequence& candidate_objects,
    const std::uint64_t primitive_count,
    const std::uint64_t raw_hit_count,
    const std::uint64_t compact_candidate_count,
    const std::unordered_map<std::uint64_t, std::uint32_t>& conservative_family_masks_by_query_id) {
  const std::vector<CandidateRecord> candidates = ParseRuntimeCandidates(candidate_objects);
  const CandidateDensityStats density =
      MakeRuntimeDensityStats(candidates, primitive_count, raw_hit_count, compact_candidate_count);
  std::vector<ProposalFeatureRow> rows;
  ThrowIfError(BuildProposalFeatureRowsFromRuntimeCandidates(
      candidates, density, conservative_family_masks_by_query_id, &rows));
  return rows;
}

py::dict BuildRuntimeProposalFeatureArraysOrThrow(
    const py::sequence& candidate_objects,
    const std::uint64_t primitive_count,
    const std::uint64_t raw_hit_count,
    const std::uint64_t compact_candidate_count,
    const std::unordered_map<std::uint64_t, std::uint32_t>& conservative_family_masks_by_query_id) {
  const std::vector<CandidateRecord> candidates = ParseRuntimeCandidates(candidate_objects);
  const CandidateDensityStats density =
      MakeRuntimeDensityStats(candidates, primitive_count, raw_hit_count, compact_candidate_count);
  std::vector<ProposalFeatureRow> rows;
  Status status;
  {
    py::gil_scoped_release release;
    status = BuildProposalFeatureRowsFromRuntimeCandidates(
        candidates, density, conservative_family_masks_by_query_id, &rows);
  }
  ThrowIfError(status);

  const py::ssize_t row_count = static_cast<py::ssize_t>(rows.size());
  py::array_t<std::uint64_t> query_id(row_count);
  py::array_t<std::uint64_t> candidate_id(row_count);
  py::array_t<std::uint32_t> slab_id(row_count);
  py::array_t<std::uint32_t> object_a_id(row_count);
  py::array_t<std::uint32_t> patch_a_id(row_count);
  py::array_t<std::uint32_t> object_b_id(row_count);
  py::array_t<std::uint32_t> patch_b_id(row_count);
  py::array_t<float> features({row_count, static_cast<py::ssize_t>(kProposalFeatureDimension)});
  py::array_t<float> interval_targets(
      {row_count, static_cast<py::ssize_t>(kProposalIntervalBinCount)});
  py::array_t<float> family_targets(
      {row_count, static_cast<py::ssize_t>(kProposalFamilyCount)});
  py::array_t<float> priority_target(row_count);
  py::array_t<float> cost_target(row_count);
  py::array_t<float> uncertainty_target(row_count);
  py::array_t<std::uint32_t> target_mask(row_count);

  auto query_id_view = query_id.mutable_unchecked<1>();
  auto candidate_id_view = candidate_id.mutable_unchecked<1>();
  auto slab_id_view = slab_id.mutable_unchecked<1>();
  auto object_a_id_view = object_a_id.mutable_unchecked<1>();
  auto patch_a_id_view = patch_a_id.mutable_unchecked<1>();
  auto object_b_id_view = object_b_id.mutable_unchecked<1>();
  auto patch_b_id_view = patch_b_id.mutable_unchecked<1>();
  auto features_view = features.mutable_unchecked<2>();
  auto interval_targets_view = interval_targets.mutable_unchecked<2>();
  auto family_targets_view = family_targets.mutable_unchecked<2>();
  auto priority_target_view = priority_target.mutable_unchecked<1>();
  auto cost_target_view = cost_target.mutable_unchecked<1>();
  auto uncertainty_target_view = uncertainty_target.mutable_unchecked<1>();
  auto target_mask_view = target_mask.mutable_unchecked<1>();

  for (py::ssize_t i = 0; i < row_count; ++i) {
    const ProposalFeatureRow& row = rows[static_cast<std::size_t>(i)];
    query_id_view(i) = row.query_id;
    candidate_id_view(i) = row.candidate_id;
    slab_id_view(i) = row.slab_id;
    object_a_id_view(i) = row.object_a_id;
    patch_a_id_view(i) = row.patch_a_id;
    object_b_id_view(i) = row.object_b_id;
    patch_b_id_view(i) = row.patch_b_id;
    for (py::ssize_t feature_index = 0;
         feature_index < static_cast<py::ssize_t>(kProposalFeatureDimension);
         ++feature_index) {
      features_view(i, feature_index) = row.features[static_cast<std::size_t>(feature_index)];
    }
    for (py::ssize_t interval_index = 0;
         interval_index < static_cast<py::ssize_t>(kProposalIntervalBinCount);
         ++interval_index) {
      interval_targets_view(i, interval_index) =
          row.interval_targets[static_cast<std::size_t>(interval_index)];
    }
    for (py::ssize_t family_index = 0;
         family_index < static_cast<py::ssize_t>(kProposalFamilyCount);
         ++family_index) {
      family_targets_view(i, family_index) =
          row.family_targets[static_cast<std::size_t>(family_index)];
    }
    priority_target_view(i) = row.priority_target;
    cost_target_view(i) = row.cost_target;
    uncertainty_target_view(i) = row.uncertainty_target;
    target_mask_view(i) = row.target_mask;
  }

  py::dict result;
  result["schema_version"] = py::int_(row_count > 0 ? rows.front().schema_version : 1U);
  result["query_id"] = std::move(query_id);
  result["candidate_id"] = std::move(candidate_id);
  result["slab_id"] = std::move(slab_id);
  result["object_a_id"] = std::move(object_a_id);
  result["patch_a_id"] = std::move(patch_a_id);
  result["object_b_id"] = std::move(object_b_id);
  result["patch_b_id"] = std::move(patch_b_id);
  result["features"] = std::move(features);
  result["interval_targets"] = std::move(interval_targets);
  result["family_targets"] = std::move(family_targets);
  result["priority_target"] = std::move(priority_target);
  result["cost_target"] = std::move(cost_target);
  result["uncertainty_target"] = std::move(uncertainty_target);
  result["target_mask"] = std::move(target_mask);
  return result;
}

template <typename T>
py::array_t<T, py::array::c_style | py::array::forcecast> RequireArray1D(
    const py::dict& arrays,
    const char* key,
    const py::ssize_t expected_size) {
  py::array_t<T, py::array::c_style | py::array::forcecast> array =
      py::array_t<T, py::array::c_style | py::array::forcecast>::ensure(arrays[key]);
  if (!array) {
    throw py::value_error(std::string("missing or invalid array field: ") + key);
  }
  if (array.ndim() != 1 || array.shape(0) != expected_size) {
    throw py::value_error(std::string(key) + " must have shape [" +
                          std::to_string(expected_size) + "]");
  }
  return array;
}

template <typename T>
py::array_t<T, py::array::c_style | py::array::forcecast> RequireArray2D(
    const py::dict& arrays,
    const char* key,
    const py::ssize_t expected_dim0,
    const py::ssize_t expected_dim1) {
  py::array_t<T, py::array::c_style | py::array::forcecast> array =
      py::array_t<T, py::array::c_style | py::array::forcecast>::ensure(arrays[key]);
  if (!array) {
    throw py::value_error(std::string("missing or invalid array field: ") + key);
  }
  if (array.ndim() != 2 || array.shape(0) != expected_dim0 || array.shape(1) != expected_dim1) {
    throw py::value_error(std::string(key) + " must have shape [" +
                          std::to_string(expected_dim0) + ", " +
                          std::to_string(expected_dim1) + "]");
  }
  return array;
}

std::vector<ProposalFeatureRow> ParseProposalFeatureRowsFromArrays(const py::dict& arrays) {
  const std::uint32_t schema_version = py::cast<std::uint32_t>(arrays["schema_version"]);
  const auto candidate_id = RequireArray1D<std::uint64_t>(arrays, "candidate_id", py::len(arrays["query_id"]));
  const py::ssize_t row_count = candidate_id.shape(0);
  const auto query_id = RequireArray1D<std::uint64_t>(arrays, "query_id", row_count);
  const auto slab_id = RequireArray1D<std::uint32_t>(arrays, "slab_id", row_count);
  const auto object_a_id = RequireArray1D<std::uint32_t>(arrays, "object_a_id", row_count);
  const auto patch_a_id = RequireArray1D<std::uint32_t>(arrays, "patch_a_id", row_count);
  const auto object_b_id = RequireArray1D<std::uint32_t>(arrays, "object_b_id", row_count);
  const auto patch_b_id = RequireArray1D<std::uint32_t>(arrays, "patch_b_id", row_count);
  const auto features =
      RequireArray2D<float>(arrays, "features", row_count, static_cast<py::ssize_t>(kProposalFeatureDimension));
  const auto interval_targets =
      RequireArray2D<float>(arrays, "interval_targets", row_count, static_cast<py::ssize_t>(kProposalIntervalBinCount));
  const auto family_targets =
      RequireArray2D<float>(arrays, "family_targets", row_count, static_cast<py::ssize_t>(kProposalFamilyCount));
  const auto priority_target = RequireArray1D<float>(arrays, "priority_target", row_count);
  const auto cost_target = RequireArray1D<float>(arrays, "cost_target", row_count);
  const auto uncertainty_target = RequireArray1D<float>(arrays, "uncertainty_target", row_count);
  const auto target_mask = RequireArray1D<std::uint32_t>(arrays, "target_mask", row_count);

  const auto query_id_view = query_id.unchecked<1>();
  const auto candidate_id_view = candidate_id.unchecked<1>();
  const auto slab_id_view = slab_id.unchecked<1>();
  const auto object_a_id_view = object_a_id.unchecked<1>();
  const auto patch_a_id_view = patch_a_id.unchecked<1>();
  const auto object_b_id_view = object_b_id.unchecked<1>();
  const auto patch_b_id_view = patch_b_id.unchecked<1>();
  const auto features_view = features.unchecked<2>();
  const auto interval_targets_view = interval_targets.unchecked<2>();
  const auto family_targets_view = family_targets.unchecked<2>();
  const auto priority_target_view = priority_target.unchecked<1>();
  const auto cost_target_view = cost_target.unchecked<1>();
  const auto uncertainty_target_view = uncertainty_target.unchecked<1>();
  const auto target_mask_view = target_mask.unchecked<1>();

  std::vector<ProposalFeatureRow> rows;
  rows.reserve(static_cast<std::size_t>(row_count));
  for (py::ssize_t i = 0; i < row_count; ++i) {
    ProposalFeatureRow row;
    row.schema_version = schema_version;
    row.query_id = query_id_view(i);
    row.candidate_id = candidate_id_view(i);
    row.slab_id = slab_id_view(i);
    row.object_a_id = object_a_id_view(i);
    row.patch_a_id = patch_a_id_view(i);
    row.object_b_id = object_b_id_view(i);
    row.patch_b_id = patch_b_id_view(i);
    for (py::ssize_t feature_index = 0;
         feature_index < static_cast<py::ssize_t>(kProposalFeatureDimension);
         ++feature_index) {
      row.features[static_cast<std::size_t>(feature_index)] = features_view(i, feature_index);
    }
    for (py::ssize_t interval_index = 0;
         interval_index < static_cast<py::ssize_t>(kProposalIntervalBinCount);
         ++interval_index) {
      row.interval_targets[static_cast<std::size_t>(interval_index)] =
          interval_targets_view(i, interval_index);
    }
    for (py::ssize_t family_index = 0;
         family_index < static_cast<py::ssize_t>(kProposalFamilyCount);
         ++family_index) {
      row.family_targets[static_cast<std::size_t>(family_index)] =
          family_targets_view(i, family_index);
    }
    row.priority_target = priority_target_view(i);
    row.cost_target = cost_target_view(i);
    row.uncertainty_target = uncertainty_target_view(i);
    row.target_mask = target_mask_view(i);
    rows.push_back(row);
  }
  return rows;
}

std::vector<ProposalOutput> ParseProposalOutputsFromArrays(
    const py::dict& feature_arrays,
    const py::dict& prediction_arrays) {
  const auto candidate_id =
      RequireArray1D<std::uint64_t>(feature_arrays, "candidate_id", py::len(feature_arrays["query_id"]));
  const py::ssize_t row_count = candidate_id.shape(0);
  const auto interval_scores =
      RequireArray2D<float>(prediction_arrays,
                            "interval_scores",
                            row_count,
                            static_cast<py::ssize_t>(kProposalIntervalBinCount));
  const auto family_scores =
      RequireArray2D<float>(prediction_arrays,
                            "family_scores",
                            row_count,
                            static_cast<py::ssize_t>(kProposalFamilyCount));
  const auto priority_score = RequireArray1D<float>(prediction_arrays, "priority_score", row_count);
  const auto cost_score = RequireArray1D<float>(prediction_arrays, "cost_score", row_count);
  const auto uncertainty_score =
      RequireArray1D<float>(prediction_arrays, "uncertainty_score", row_count);

  const auto candidate_id_view = candidate_id.unchecked<1>();
  const auto interval_scores_view = interval_scores.unchecked<2>();
  const auto family_scores_view = family_scores.unchecked<2>();
  const auto priority_score_view = priority_score.unchecked<1>();
  const auto cost_score_view = cost_score.unchecked<1>();
  const auto uncertainty_score_view = uncertainty_score.unchecked<1>();

  std::vector<ProposalOutput> outputs;
  outputs.reserve(static_cast<std::size_t>(row_count));
  for (py::ssize_t i = 0; i < row_count; ++i) {
    ProposalOutput output;
    output.candidate_id = candidate_id_view(i);
    for (py::ssize_t interval_index = 0;
         interval_index < static_cast<py::ssize_t>(kProposalIntervalBinCount);
         ++interval_index) {
      output.interval_scores[static_cast<std::size_t>(interval_index)] =
          interval_scores_view(i, interval_index);
    }
    for (py::ssize_t family_index = 0;
         family_index < static_cast<py::ssize_t>(kProposalFamilyCount);
         ++family_index) {
      output.family_scores[static_cast<std::size_t>(family_index)] =
          family_scores_view(i, family_index);
    }
    output.priority_score = priority_score_view(i);
    output.cost_score = cost_score_view(i);
    output.uncertainty_score = uncertainty_score_view(i);
    outputs.push_back(output);
  }
  return outputs;
}

constexpr std::uint32_t kPybindConservativeFeatureFamilyMask =
    kFeatureFamilyPointTriangle | kFeatureFamilyEdgeEdge;

bool IsValidPybindSchedulingConfig(const ProposalSchedulingConfig& config, std::string* message) {
  if (config.first_work_item_id == 0U) {
    *message = "ProposalSchedulingConfig.first_work_item_id must be non-zero";
    return false;
  }
  if (config.conservative_feature_family_mask == 0U) {
    *message = "ProposalSchedulingConfig.conservative_feature_family_mask must be non-zero";
    return false;
  }
  if ((config.conservative_feature_family_mask & ~kPybindConservativeFeatureFamilyMask) != 0U) {
    *message = "ProposalSchedulingConfig.conservative_feature_family_mask has unknown bits";
    return false;
  }
  if (!std::isfinite(config.fallback_interval_t0) ||
      !std::isfinite(config.fallback_interval_t1) ||
      config.fallback_interval_t0 < 0.0 || config.fallback_interval_t1 > 1.0 ||
      config.fallback_interval_t0 > config.fallback_interval_t1) {
    *message = "ProposalSchedulingConfig fallback interval must lie in [0, 1]";
    return false;
  }
  if (!std::isfinite(config.family_score_threshold)) {
    *message = "ProposalSchedulingConfig.family_score_threshold must be finite";
    return false;
  }
  if (!std::isfinite(config.uncertainty_fallback_threshold) ||
      config.uncertainty_fallback_threshold < 0.0F) {
    *message = "ProposalSchedulingConfig.uncertainty_fallback_threshold must be non-negative";
    return false;
  }
  if (!std::isfinite(config.ood_abs_feature_threshold) ||
      config.ood_abs_feature_threshold <= 0.0F) {
    *message = "ProposalSchedulingConfig.ood_abs_feature_threshold must be positive";
    return false;
  }
  return true;
}

bool HasNonFinitePybindProposal(const ProposalOutput& output) {
  for (const float value : output.interval_scores) {
    if (!std::isfinite(value)) {
      return true;
    }
  }
  for (const float value : output.family_scores) {
    if (!std::isfinite(value)) {
      return true;
    }
  }
  return !std::isfinite(output.priority_score) || !std::isfinite(output.cost_score) ||
         !std::isfinite(output.uncertainty_score);
}

std::uint32_t PybindBaseFamilyMaskForRuntimeQuery(
    const std::unordered_map<std::uint64_t, std::uint32_t>& conservative_family_masks_by_query_id,
    const std::uint64_t query_id,
    const ProposalSchedulingConfig& config) {
  const auto mask_it = conservative_family_masks_by_query_id.find(query_id);
  if (mask_it != conservative_family_masks_by_query_id.end() && mask_it->second != 0U) {
    return mask_it->second;
  }
  return config.conservative_feature_family_mask;
}

std::uint32_t PybindPredictedFamilyMaskFromScores(const ProposalOutput& output,
                                                  const ProposalSchedulingConfig& config) {
  std::uint32_t mask = 0U;
  if (output.family_scores[0] >= config.family_score_threshold) {
    mask |= kFeatureFamilyPointTriangle;
  }
  if (output.family_scores[1] >= config.family_score_threshold) {
    mask |= kFeatureFamilyEdgeEdge;
  }
  return mask;
}

bool PybindNeedsPrioritySort(const std::vector<ExactWorkItem>& work_queue) {
  if (work_queue.size() < 2U) {
    return false;
  }
  for (std::size_t i = 1; i < work_queue.size(); ++i) {
    if (work_queue[i - 1].priority_score < work_queue[i].priority_score) {
      return true;
    }
  }
  return false;
}

ProposalRuntimeScheduleResult RunDummyProposalScheduleFromRuntimeCandidatesOrThrow(
    const py::sequence& candidate_objects,
    const std::uint64_t primitive_count,
    const std::uint64_t raw_hit_count,
    const std::uint64_t compact_candidate_count,
    const std::unordered_map<std::uint64_t, std::uint32_t>& conservative_family_masks_by_query_id,
    const ProposalSchedulingConfig& config,
    const bool materialize_artifacts) {
  const std::vector<CandidateRecord> candidates = ParseRuntimeCandidates(candidate_objects);
  const CandidateDensityStats density =
      MakeRuntimeDensityStats(candidates, primitive_count, raw_hit_count, compact_candidate_count);
  ProposalRuntimeScheduleResult result;
  Status status;
  {
    py::gil_scoped_release release;
    status = RunDummyProposalScheduleFromRuntimeCandidates(candidates,
                                                           density,
                                                           conservative_family_masks_by_query_id,
                                                           config,
                                                           materialize_artifacts,
                                                           &result);
  }
  ThrowIfError(status);
  return result;
}

py::tuple ScheduleRuntimeExactWorkItemsFromProposalsOrThrow(
    const py::sequence& candidate_objects,
    const py::sequence& row_objects,
    const py::sequence& proposal_output_objects,
    const std::unordered_map<std::uint64_t, std::uint32_t>& conservative_family_masks_by_query_id,
    const ProposalSchedulingConfig& config) {
  const std::vector<CandidateRecord> candidates = ParseRuntimeCandidates(candidate_objects);
  const std::vector<ProposalFeatureRow> rows = ParseProposalFeatureRows(row_objects);
  const std::vector<ProposalOutput> proposal_outputs = ParseProposalOutputs(proposal_output_objects);
  std::vector<ExactWorkItem> work_queue;
  ProposalScheduleStats stats;
  Status status;
  {
    py::gil_scoped_release release;
    status = ScheduleRuntimeExactWorkItemsFromProposals(candidates,
                                                        rows,
                                                        proposal_outputs,
                                                        conservative_family_masks_by_query_id,
                                                        config,
                                                        &work_queue,
                                                        &stats);
  }
  ThrowIfError(status);
  return py::make_tuple(work_queue, stats);
}

py::tuple ScheduleRuntimeExactWorkItemsFromArraysOrThrow(
    const py::sequence& candidate_objects,
    const py::dict& feature_arrays,
    const py::dict& prediction_arrays,
    const std::unordered_map<std::uint64_t, std::uint32_t>& conservative_family_masks_by_query_id,
    const ProposalSchedulingConfig& config) {
  const std::vector<CandidateRecord> candidates = ParseRuntimeCandidates(candidate_objects);
  const std::vector<ProposalFeatureRow> rows = ParseProposalFeatureRowsFromArrays(feature_arrays);
  const std::vector<ProposalOutput> proposal_outputs =
      ParseProposalOutputsFromArrays(feature_arrays, prediction_arrays);
  std::vector<ExactWorkItem> work_queue;
  ProposalScheduleStats stats;
  Status status;
  {
    py::gil_scoped_release release;
    status = ScheduleRuntimeExactWorkItemsFromProposals(candidates,
                                                        rows,
                                                        proposal_outputs,
                                                        conservative_family_masks_by_query_id,
                                                        config,
                                                        &work_queue,
                                                        &stats);
  }
  ThrowIfError(status);
  return py::make_tuple(work_queue, stats);
}

py::tuple ScheduleRuntimeExactWorkItemsFromProposalArraysOrThrow(
    const py::dict& feature_arrays,
    const py::dict& prediction_arrays,
    const std::unordered_map<std::uint64_t, std::uint32_t>& conservative_family_masks_by_query_id,
    const ProposalSchedulingConfig& config) {
  const std::vector<ProposalFeatureRow> rows = ParseProposalFeatureRowsFromArrays(feature_arrays);
  const std::vector<ProposalOutput> proposal_outputs =
      ParseProposalOutputsFromArrays(feature_arrays, prediction_arrays);
  if (rows.size() != proposal_outputs.size()) {
    throw py::value_error("feature row and proposal output array lengths must match");
  }

  std::vector<ExactWorkItem> work_queue;
  ProposalScheduleStats stats;
  Status status = Status::Ok();
  {
    py::gil_scoped_release release;
    std::string config_error;
    if (!IsValidPybindSchedulingConfig(config, &config_error)) {
      status = Status::Error(config_error);
    } else {
      stats.raw_candidate_count = rows.size();
      stats.proposal_output_count = proposal_outputs.size();
      stats.work_item_count = rows.size();
      work_queue.reserve(rows.size());
      for (std::size_t i = 0; status.ok && i < rows.size(); ++i) {
        const ProposalFeatureRow& row = rows[i];
        const ProposalOutput& output = proposal_outputs[i];
        if (row.candidate_id == 0U || output.candidate_id != row.candidate_id) {
          status = Status::Error("proposal arrays must preserve candidate_id order");
          break;
        }
        const bool invalid_proposal =
            HasNonFinitePybindProposal(output) || !ValidateProposalOutput(output).ok;
        const bool ood = IsProposalFeatureRowOod(row, config);
        const bool high_uncertainty = std::isfinite(output.uncertainty_score) &&
                                      output.uncertainty_score >=
                                          config.uncertainty_fallback_threshold;
        const bool fallback = invalid_proposal || ood || high_uncertainty;
        stats.invalid_proposal_fallback_count += invalid_proposal ? 1U : 0U;
        stats.ood_fallback_count += ood ? 1U : 0U;
        stats.high_uncertainty_fallback_count += high_uncertainty ? 1U : 0U;
        stats.fallback_count += fallback ? 1U : 0U;

        const std::uint32_t base_family_mask = PybindBaseFamilyMaskForRuntimeQuery(
            conservative_family_masks_by_query_id, row.query_id, config);
        ExactWorkItem item;
        item.parent_candidate_id = row.candidate_id;
        item.query_id = row.query_id;
        item.slab_id = row.slab_id;
        item.patch_a_id = row.patch_a_id;
        item.patch_b_id = row.patch_b_id;
        item.interval_t0 = config.fallback_interval_t0;
        item.interval_t1 = config.fallback_interval_t1;
        item.feature_family_mask = base_family_mask;
        item.topk_feature_ids_offset = 0U;
        item.depth = 0U;
        item.priority_score = row.features[3];
        item.source = ProposalSource::kFallback;
        if (!fallback) {
          item.feature_family_mask = base_family_mask | PybindPredictedFamilyMaskFromScores(output, config);
          item.priority_score = output.priority_score;
          item.source = ProposalSource::kRefined;
        }
        work_queue.push_back(item);
      }

      if (status.ok && !config.preserve_candidate_order && PybindNeedsPrioritySort(work_queue)) {
        std::stable_sort(work_queue.begin(),
                         work_queue.end(),
                         [](const ExactWorkItem& lhs, const ExactWorkItem& rhs) {
                           return lhs.priority_score > rhs.priority_score;
                         });
      }
      for (std::size_t i = 0; status.ok && i < work_queue.size(); ++i) {
        work_queue[i].work_item_id = config.first_work_item_id + i;
        if (work_queue[i].parent_candidate_id != rows[i].candidate_id) {
          ++stats.reordered_count;
        }
        if (auto validate_status = ValidateExactWorkItem(work_queue[i]); !validate_status.ok) {
          status = validate_status;
          break;
        }
      }
      stats.monotonic_safe = status.ok;
    }
  }
  ThrowIfError(status);
  return py::make_tuple(work_queue, stats);
}

py::dict RunNativeDenseGroupExactEarlyStopOrThrow(
    const py::dict& feature_arrays,
    const py::dict& prediction_arrays,
    const float uncertainty_fallback_threshold,
    const std::uint32_t representative_attempt_limit,
    const double interval_miss_penalty_scale,
    const bool preserve_input_order) {
  const auto parse_start = std::chrono::steady_clock::now();
  const auto query_id =
      RequireArray1D<std::uint64_t>(feature_arrays, "query_id", py::len(feature_arrays["query_id"]));
  const py::ssize_t row_count = query_id.shape(0);
  const auto candidate_id = RequireArray1D<std::uint64_t>(feature_arrays, "candidate_id", row_count);
  const auto cost_target = RequireArray1D<float>(feature_arrays, "cost_target", row_count);
  const auto oracle_trace = RequireArray2D<double>(feature_arrays, "oracle_trace", row_count, 8);
  const auto interval_scores =
      RequireArray2D<float>(prediction_arrays,
                            "interval_scores",
                            row_count,
                            static_cast<py::ssize_t>(kProposalIntervalBinCount));
  const auto priority_score = RequireArray1D<float>(prediction_arrays, "priority_score", row_count);
  const auto pred_cost_score = RequireArray1D<float>(prediction_arrays, "cost_score", row_count);
  const auto uncertainty_score =
      RequireArray1D<float>(prediction_arrays, "uncertainty_score", row_count);

  const auto query_id_view = query_id.unchecked<1>();
  const auto candidate_id_view = candidate_id.unchecked<1>();
  const auto cost_target_view = cost_target.unchecked<1>();
  const auto oracle_trace_view = oracle_trace.unchecked<2>();
  const auto interval_scores_view = interval_scores.unchecked<2>();
  const auto priority_score_view = priority_score.unchecked<1>();
  const auto pred_cost_score_view = pred_cost_score.unchecked<1>();
  const auto uncertainty_score_view = uncertainty_score.unchecked<1>();

  std::vector<DenseGroupCandidateInput> candidates;
  candidates.reserve(static_cast<std::size_t>(row_count));
  for (py::ssize_t i = 0; i < row_count; ++i) {
    py::ssize_t best_interval = 0;
    float best_score = interval_scores_view(i, 0);
    for (py::ssize_t interval_index = 1;
         interval_index < static_cast<py::ssize_t>(kProposalIntervalBinCount);
         ++interval_index) {
      const float score = interval_scores_view(i, interval_index);
      if (score > best_score) {
        best_score = score;
        best_interval = interval_index;
      }
    }
    const double interval_width = 1.0 / static_cast<double>(kProposalIntervalBinCount);
    const double full_exact_cost = oracle_trace_view(i, 5);
    const double cost_ratio = cost_target_view(i);

    DenseGroupCandidateInput input;
    input.query_id = query_id_view(i);
    input.candidate_id = candidate_id_view(i);
    input.full_exact_cost = full_exact_cost;
    input.narrow_exact_cost =
        std::isfinite(cost_ratio) && cost_ratio > 0.0
            ? std::max(1.0e-12, full_exact_cost * static_cast<double>(cost_ratio))
            : full_exact_cost;
    input.interval_t0 = interval_width * static_cast<double>(best_interval);
    input.interval_t1 = std::min(1.0, input.interval_t0 + interval_width);
    input.contact_t0 = oracle_trace_view(i, 6);
    input.contact_t1 = oracle_trace_view(i, 7);
    input.priority_score = priority_score_view(i);
    input.cost_score = pred_cost_score_view(i);
    input.uncertainty_score = uncertainty_score_view(i);
    input.candidate_collides = oracle_trace_view(i, 0) > 0.5;
    candidates.push_back(input);
  }
  const auto parse_end = std::chrono::steady_clock::now();

  DenseGroupEarlyStopConfig config;
  config.uncertainty_fallback_threshold = uncertainty_fallback_threshold;
  config.representative_attempt_limit = representative_attempt_limit;
  config.interval_miss_penalty_scale = interval_miss_penalty_scale;
  config.preserve_input_order = preserve_input_order;

  DenseGroupEarlyStopStats stats;
  Status status;
  {
    py::gil_scoped_release release;
    status = RunDenseGroupExactEarlyStop(candidates, config, &stats);
  }
  ThrowIfError(status);

  const double parse_ms =
      std::chrono::duration<double, std::milli>(parse_end - parse_start).count();
  const double native_total_ms = parse_ms + stats.total_ms;
  const double exact_call_reduction =
      stats.no_proposal_exact_calls == 0U
          ? 0.0
          : 1.0 - (static_cast<double>(stats.learned_exact_calls) /
                   static_cast<double>(stats.no_proposal_exact_calls));
  const double exact_work_reduction =
      stats.no_proposal_exact_work <= 0.0
          ? 0.0
          : 1.0 - (stats.learned_exact_work / stats.no_proposal_exact_work);

  py::dict result;
  result["candidate_count"] = py::int_(stats.candidate_count);
  result["group_count"] = py::int_(stats.group_count);
  result["positive_group_count"] = py::int_(stats.positive_group_count);
  result["no_proposal_exact_calls"] = py::int_(stats.no_proposal_exact_calls);
  result["no_proposal_exact_work"] = py::float_(stats.no_proposal_exact_work);
  result["learned_exact_calls"] = py::int_(stats.learned_exact_calls);
  result["learned_fallback_calls"] = py::int_(stats.learned_fallback_calls);
  result["learned_interval_hit_count"] = py::int_(stats.learned_interval_hit_count);
  result["learned_interval_miss_count"] = py::int_(stats.learned_interval_miss_count);
  result["learned_exact_work"] = py::float_(stats.learned_exact_work);
  result["first_positive_rank_mean"] =
      stats.positive_group_count == 0U
          ? py::float_(0.0)
          : py::float_(stats.first_positive_rank_sum /
                       static_cast<double>(stats.positive_group_count));
  result["cost_weighted_first_positive_mean"] =
      stats.positive_group_count == 0U
          ? py::float_(0.0)
          : py::float_(stats.cost_before_first_positive_sum /
                       static_cast<double>(stats.positive_group_count));
  result["tp"] = py::int_(stats.tp);
  result["tn"] = py::int_(stats.tn);
  result["fp"] = py::int_(stats.fp);
  result["fn"] = py::int_(stats.fn);
  result["reordered_count"] = py::int_(stats.reordered_count);
  result["high_uncertainty_group_count"] = py::int_(stats.high_uncertainty_group_count);
  result["parse_ms"] = py::float_(parse_ms);
  result["schedule_ms"] = py::float_(stats.schedule_ms);
  result["exact_ms"] = py::float_(stats.exact_ms);
  result["native_driver_ms"] = py::float_(stats.total_ms);
  result["native_total_ms"] = py::float_(native_total_ms);
  result["exact_call_reduction"] = py::float_(exact_call_reduction);
  result["exact_work_reduction"] = py::float_(exact_work_reduction);
  return result;
}

bool ValidateProxySceneOrThrow(const ProxyScene& scene) {
  ThrowIfError(ValidateProxyScene(scene));
  return true;
}

bool ValidateExactWorkQueueCoverageOrThrow(
    const std::vector<ExactCertificateQuery>& work_queue,
    const ExactWorkQueueResult& result) {
  ThrowIfError(ValidateExactWorkQueueCoverage(work_queue, result));
  return true;
}

void BindRuntimeContracts(py::module_& m) {
  py::enum_<ProxyType>(m, "ProxyType")
      .value("UNKNOWN", ProxyType::kUnknown)
      .value("SWEPT_AABB", ProxyType::kSweptAabb)
      .value("CAPSULE", ProxyType::kCapsule);

  py::enum_<ProposalSource>(m, "ProposalSource")
      .value("RAW", ProposalSource::kRaw)
      .value("REFINED", ProposalSource::kRefined)
      .value("FALLBACK", ProposalSource::kFallback);

  py::enum_<CertificateStatus>(m, "CertificateStatus")
      .value("COLLISION", CertificateStatus::kCollision)
      .value("SEPARATION", CertificateStatus::kSeparation)
      .value("UNDECIDED", CertificateStatus::kUndecided)
      .value("Collision", CertificateStatus::kCollision)
      .value("Separation", CertificateStatus::kSeparation)
      .value("Undecided", CertificateStatus::kUndecided);

  py::enum_<CertificateRefinementMode>(m, "CertificateRefinementMode")
      .value("NONE", CertificateRefinementMode::kNone)
      .value("BISECT_INTERVAL", CertificateRefinementMode::kBisectInterval)
      .value("REQUEST_GEOMETRY", CertificateRefinementMode::kRequestGeometry)
      .value("ESCALATE_PRECISION", CertificateRefinementMode::kEscalatePrecision);

  py::enum_<AuditStage>(m, "AuditStage")
      .value("RT", AuditStage::kRt)
      .value("PROPOSAL", AuditStage::kProposal)
      .value("EXACT", AuditStage::kExact)
      .value("REFINE", AuditStage::kRefine)
      .value("CERTIFY", AuditStage::kCertify);

  py::class_<CandidateRecord>(m, "CandidateRecord")
      .def(py::init<>())
      .def_readwrite("schema_version", &CandidateRecord::schema_version)
      .def_readwrite("candidate_id", &CandidateRecord::candidate_id)
      .def_readwrite("query_id", &CandidateRecord::query_id)
      .def_readwrite("slab_id", &CandidateRecord::slab_id)
      .def_readwrite("object_a_id", &CandidateRecord::object_a_id)
      .def_readwrite("patch_a_id", &CandidateRecord::patch_a_id)
      .def_readwrite("object_b_id", &CandidateRecord::object_b_id)
      .def_readwrite("patch_b_id", &CandidateRecord::patch_b_id)
      .def_readwrite("proxy_type_a", &CandidateRecord::proxy_type_a)
      .def_readwrite("proxy_type_b", &CandidateRecord::proxy_type_b)
      .def_readwrite("rt_hit_count", &CandidateRecord::rt_hit_count)
      .def_readwrite("motion_bound", &CandidateRecord::motion_bound)
      .def_readwrite("proxy_features_offset", &CandidateRecord::proxy_features_offset)
      .def_readwrite("flags", &CandidateRecord::flags);

  py::class_<ProposalOutput>(m, "ProposalOutput")
      .def(py::init<>())
      .def_readwrite("candidate_id", &ProposalOutput::candidate_id)
      .def_readwrite("interval_scores", &ProposalOutput::interval_scores)
      .def_readwrite("family_scores", &ProposalOutput::family_scores)
      .def_readwrite("priority_score", &ProposalOutput::priority_score)
      .def_readwrite("cost_score", &ProposalOutput::cost_score)
      .def_readwrite("uncertainty_score", &ProposalOutput::uncertainty_score);

  py::class_<ExactWorkItem>(m, "ExactWorkItem")
      .def(py::init<>())
      .def_readwrite("work_item_id", &ExactWorkItem::work_item_id)
      .def_readwrite("parent_candidate_id", &ExactWorkItem::parent_candidate_id)
      .def_readwrite("query_id", &ExactWorkItem::query_id)
      .def_readwrite("slab_id", &ExactWorkItem::slab_id)
      .def_readwrite("patch_a_id", &ExactWorkItem::patch_a_id)
      .def_readwrite("patch_b_id", &ExactWorkItem::patch_b_id)
      .def_readwrite("interval_t0", &ExactWorkItem::interval_t0)
      .def_readwrite("interval_t1", &ExactWorkItem::interval_t1)
      .def_readwrite("feature_family_mask", &ExactWorkItem::feature_family_mask)
      .def_readwrite("topk_feature_ids_offset", &ExactWorkItem::topk_feature_ids_offset)
      .def_readwrite("depth", &ExactWorkItem::depth)
      .def_readwrite("priority_score", &ExactWorkItem::priority_score)
      .def_readwrite("source", &ExactWorkItem::source);

  py::class_<CertificateResult>(m, "CertificateResult")
      .def(py::init<>())
      .def_readwrite("work_item_id", &CertificateResult::work_item_id)
      .def_readwrite("query_id", &CertificateResult::query_id)
      .def_readwrite("status", &CertificateResult::status)
      .def_readwrite("interval_t0", &CertificateResult::interval_t0)
      .def_readwrite("interval_t1", &CertificateResult::interval_t1)
      .def_readwrite("toi_upper", &CertificateResult::toi_upper)
      .def_readwrite("safe_margin_lb", &CertificateResult::safe_margin_lb)
      .def_readwrite("witness_family", &CertificateResult::witness_family)
      .def_readwrite("witness_id_a", &CertificateResult::witness_id_a)
      .def_readwrite("witness_id_b", &CertificateResult::witness_id_b)
      .def_readwrite("covered_feature_mask", &CertificateResult::covered_feature_mask)
      .def_readwrite("eps_time", &CertificateResult::eps_time)
      .def_readwrite("eps_space", &CertificateResult::eps_space)
      .def_readwrite("reason_code", &CertificateResult::reason_code)
      .def_readwrite("next_refinement_mode", &CertificateResult::next_refinement_mode);

  py::class_<AuditLogRow>(m, "AuditLogRow")
      .def(py::init<>())
      .def_readwrite("event_id", &AuditLogRow::event_id)
      .def_readwrite("query_id", &AuditLogRow::query_id)
      .def_readwrite("candidate_id", &AuditLogRow::candidate_id)
      .def_readwrite("work_item_id", &AuditLogRow::work_item_id)
      .def_readwrite("stage", &AuditLogRow::stage)
      .def_readwrite("action", &AuditLogRow::action)
      .def_readwrite("depth", &AuditLogRow::depth)
      .def_readwrite("interval_t0", &AuditLogRow::interval_t0)
      .def_readwrite("interval_t1", &AuditLogRow::interval_t1)
      .def_readwrite("timestamp_us", &AuditLogRow::timestamp_us)
      .def_readwrite("aux_value0", &AuditLogRow::aux_value0)
      .def_readwrite("aux_value1", &AuditLogRow::aux_value1);

  py::class_<BenchmarkRow>(m, "BenchmarkRow")
      .def(py::init<>())
      .def_readwrite("query_count", &BenchmarkRow::query_count)
      .def_readwrite("fn_count", &BenchmarkRow::fn_count)
      .def_readwrite("fp_count", &BenchmarkRow::fp_count)
      .def_readwrite("candidate_recall", &BenchmarkRow::candidate_recall)
      .def_readwrite("avg_candidates", &BenchmarkRow::avg_candidates)
      .def_readwrite("avg_exact_evals", &BenchmarkRow::avg_exact_evals)
      .def_readwrite("avg_subdivision_depth", &BenchmarkRow::avg_subdivision_depth)
      .def_readwrite("fallback_ratio", &BenchmarkRow::fallback_ratio)
      .def_readwrite("rt_ms", &BenchmarkRow::rt_ms)
      .def_readwrite("proposal_ms", &BenchmarkRow::proposal_ms)
      .def_readwrite("exact_ms", &BenchmarkRow::exact_ms)
      .def_readwrite("total_ms", &BenchmarkRow::total_ms)
      .def_readwrite("qps", &BenchmarkRow::qps);
}

void BindProposalTypes(py::module_& m) {
  py::class_<ProposalFeatureRow>(m, "ProposalFeatureRow")
      .def(py::init<>())
      .def_readwrite("schema_version", &ProposalFeatureRow::schema_version)
      .def_readwrite("query_id", &ProposalFeatureRow::query_id)
      .def_readwrite("candidate_id", &ProposalFeatureRow::candidate_id)
      .def_readwrite("slab_id", &ProposalFeatureRow::slab_id)
      .def_readwrite("object_a_id", &ProposalFeatureRow::object_a_id)
      .def_readwrite("patch_a_id", &ProposalFeatureRow::patch_a_id)
      .def_readwrite("object_b_id", &ProposalFeatureRow::object_b_id)
      .def_readwrite("patch_b_id", &ProposalFeatureRow::patch_b_id)
      .def_readwrite("features", &ProposalFeatureRow::features)
      .def_readwrite("interval_targets", &ProposalFeatureRow::interval_targets)
      .def_readwrite("family_targets", &ProposalFeatureRow::family_targets)
      .def_readwrite("priority_target", &ProposalFeatureRow::priority_target)
      .def_readwrite("cost_target", &ProposalFeatureRow::cost_target)
      .def_readwrite("uncertainty_target", &ProposalFeatureRow::uncertainty_target)
      .def_readwrite("target_mask", &ProposalFeatureRow::target_mask);

  py::class_<ProposalSchedulingConfig>(m, "ProposalSchedulingConfig")
      .def(py::init<>())
      .def_readwrite("first_work_item_id", &ProposalSchedulingConfig::first_work_item_id)
      .def_readwrite("conservative_feature_family_mask",
                     &ProposalSchedulingConfig::conservative_feature_family_mask)
      .def_readwrite("fallback_interval_t0", &ProposalSchedulingConfig::fallback_interval_t0)
      .def_readwrite("fallback_interval_t1", &ProposalSchedulingConfig::fallback_interval_t1)
      .def_readwrite("family_score_threshold", &ProposalSchedulingConfig::family_score_threshold)
      .def_readwrite("uncertainty_fallback_threshold",
                     &ProposalSchedulingConfig::uncertainty_fallback_threshold)
      .def_readwrite("ood_abs_feature_threshold",
                     &ProposalSchedulingConfig::ood_abs_feature_threshold)
      .def_readwrite("preserve_candidate_order",
                     &ProposalSchedulingConfig::preserve_candidate_order);

  py::class_<ProposalScheduleStats>(m, "ProposalScheduleStats")
      .def(py::init<>())
      .def_readwrite("raw_candidate_count", &ProposalScheduleStats::raw_candidate_count)
      .def_readwrite("proposal_output_count", &ProposalScheduleStats::proposal_output_count)
      .def_readwrite("work_item_count", &ProposalScheduleStats::work_item_count)
      .def_readwrite("fallback_count", &ProposalScheduleStats::fallback_count)
      .def_readwrite("missing_proposal_fallback_count",
                     &ProposalScheduleStats::missing_proposal_fallback_count)
      .def_readwrite("invalid_proposal_fallback_count",
                     &ProposalScheduleStats::invalid_proposal_fallback_count)
      .def_readwrite("ood_fallback_count", &ProposalScheduleStats::ood_fallback_count)
      .def_readwrite("high_uncertainty_fallback_count",
                     &ProposalScheduleStats::high_uncertainty_fallback_count)
      .def_readwrite("reordered_count", &ProposalScheduleStats::reordered_count)
      .def_readwrite("monotonic_safe", &ProposalScheduleStats::monotonic_safe);

  py::class_<ProposalRuntimeScheduleResult>(m, "ProposalRuntimeScheduleResult")
      .def(py::init<>())
      .def_readwrite("feature_rows", &ProposalRuntimeScheduleResult::feature_rows)
      .def_readwrite("proposal_outputs", &ProposalRuntimeScheduleResult::proposal_outputs)
      .def_readwrite("work_queue", &ProposalRuntimeScheduleResult::work_queue)
      .def_readwrite("stats", &ProposalRuntimeScheduleResult::stats);
}

void BindGeometryAndCandidateTypes(py::module_& m) {
  py::class_<Mesh>(m, "Mesh")
      .def(py::init<>())
      .def_readwrite("vertices_ref", &Mesh::vertices_ref)
      .def_readwrite("triangles", &Mesh::triangles)
      .def_readwrite("patch_ids", &Mesh::patch_ids);

  py::class_<ObjLoadOptions>(m, "ObjLoadOptions")
      .def(py::init<>())
      .def_readwrite("triangulate_polygon_faces", &ObjLoadOptions::triangulate_polygon_faces)
      .def_readwrite("use_object_groups_as_patch_ids", &ObjLoadOptions::use_object_groups_as_patch_ids);

  py::class_<PoseSample>(m, "PoseSample")
      .def(py::init<>())
      .def_readwrite("translation", &PoseSample::translation)
      .def_readwrite("rotation_xyzw", &PoseSample::rotation_xyzw);

  py::class_<MotionSegment>(m, "MotionSegment")
      .def(py::init<>())
      .def_readwrite("t0", &MotionSegment::t0)
      .def_readwrite("t1", &MotionSegment::t1)
      .def_readwrite("pose_t0", &MotionSegment::pose_t0)
      .def_readwrite("pose_t1", &MotionSegment::pose_t1);

  py::class_<Patch>(m, "Patch")
      .def(py::init<>())
      .def_readwrite("patch_id", &Patch::patch_id)
      .def_readwrite("triangle_ids", &Patch::triangle_ids)
      .def_readwrite("triangle_count", &Patch::triangle_count)
      .def_readwrite("area", &Patch::area)
      .def_readwrite("local_center", &Patch::local_center)
      .def_readwrite("radius", &Patch::radius);

  py::class_<Aabb>(m, "Aabb")
      .def(py::init<>())
      .def_readwrite("min", &Aabb::min)
      .def_readwrite("max", &Aabb::max);

  py::class_<PatchMotionBound>(m, "PatchMotionBound")
      .def(py::init<>())
      .def_readwrite("patch_id", &PatchMotionBound::patch_id)
      .def_readwrite("t0", &PatchMotionBound::t0)
      .def_readwrite("t1", &PatchMotionBound::t1)
      .def_readwrite("center_t0", &PatchMotionBound::center_t0)
      .def_readwrite("center_t1", &PatchMotionBound::center_t1)
      .def_readwrite("translation_bound", &PatchMotionBound::translation_bound)
      .def_readwrite("rotation_angle", &PatchMotionBound::rotation_angle)
      .def_readwrite("center_rotation_bound", &PatchMotionBound::center_rotation_bound)
      .def_readwrite("surface_rotation_bound", &PatchMotionBound::surface_rotation_bound)
      .def_readwrite("radial_motion_bound", &PatchMotionBound::radial_motion_bound)
      .def_readwrite("conservative_radius", &PatchMotionBound::conservative_radius);

  py::class_<CapsuleProxy>(m, "CapsuleProxy")
      .def(py::init<>())
      .def_readwrite("patch_id", &CapsuleProxy::patch_id)
      .def_readwrite("endpoint0", &CapsuleProxy::endpoint0)
      .def_readwrite("endpoint1", &CapsuleProxy::endpoint1)
      .def_readwrite("radius", &CapsuleProxy::radius)
      .def_readwrite("motion_bound", &CapsuleProxy::motion_bound);

  py::class_<ProxyObjectBuildInput>(m, "ProxyObjectBuildInput")
      .def(py::init<>())
      .def_readwrite("object_id", &ProxyObjectBuildInput::object_id)
      .def_readwrite("proxy_type", &ProxyObjectBuildInput::proxy_type)
      .def_readwrite("patches", &ProxyObjectBuildInput::patches)
      .def_readwrite("motion_segments", &ProxyObjectBuildInput::motion_segments)
      .def_readwrite("slabs_per_motion_segment", &ProxyObjectBuildInput::slabs_per_motion_segment)
      .def_readwrite("eps_proxy", &ProxyObjectBuildInput::eps_proxy);

  py::class_<ProxySceneBuildInput>(m, "ProxySceneBuildInput")
      .def(py::init<>())
      .def_readwrite("query_id", &ProxySceneBuildInput::query_id)
      .def_readwrite("objects", &ProxySceneBuildInput::objects);

  py::class_<ProxyPrimitive>(m, "ProxyPrimitive")
      .def(py::init<>())
      .def_readwrite("proxy_id", &ProxyPrimitive::proxy_id)
      .def_readwrite("object_id", &ProxyPrimitive::object_id)
      .def_readwrite("patch_id", &ProxyPrimitive::patch_id)
      .def_readwrite("slab_id", &ProxyPrimitive::slab_id)
      .def_readwrite("motion_segment_id", &ProxyPrimitive::motion_segment_id)
      .def_readwrite("proxy_type", &ProxyPrimitive::proxy_type)
      .def_readwrite("t0", &ProxyPrimitive::t0)
      .def_readwrite("t1", &ProxyPrimitive::t1)
      .def_readwrite("bounds", &ProxyPrimitive::bounds)
      .def_readwrite("capsule", &ProxyPrimitive::capsule)
      .def_readwrite("motion_bound", &ProxyPrimitive::motion_bound);

  py::class_<ProxyScene>(m, "ProxyScene")
      .def(py::init<>())
      .def_readwrite("query_id", &ProxyScene::query_id)
      .def_readwrite("primitives", &ProxyScene::primitives);

  py::class_<RawCandidateHit>(m, "RawCandidateHit")
      .def(py::init<>())
      .def_readwrite("query_id", &RawCandidateHit::query_id)
      .def_readwrite("pair_key", &RawCandidateHit::pair_key)
      .def_readwrite("proxy_a_index", &RawCandidateHit::proxy_a_index)
      .def_readwrite("proxy_b_index", &RawCandidateHit::proxy_b_index)
      .def_readwrite("slab_id", &RawCandidateHit::slab_id)
      .def_readwrite("rt_hit_count", &RawCandidateHit::rt_hit_count)
      .def_readwrite("flags", &RawCandidateHit::flags)
      .def_readwrite("motion_bound", &RawCandidateHit::motion_bound);

  py::class_<RawCandidateBuffer>(m, "RawCandidateBuffer")
      .def(py::init<>())
      .def_readwrite("hits", &RawCandidateBuffer::hits);

  py::class_<RtCandidateTiming>(m, "RtCandidateTiming")
      .def(py::init<>())
      .def_readwrite("build_ms", &RtCandidateTiming::build_ms)
      .def_readwrite("update_ms", &RtCandidateTiming::update_ms)
      .def_readwrite("trace_ms", &RtCandidateTiming::trace_ms)
      .def_readwrite("compact_ms", &RtCandidateTiming::compact_ms)
      .def_readwrite("stats_ms", &RtCandidateTiming::stats_ms)
      .def_readwrite("total_ms", &RtCandidateTiming::total_ms);

  py::class_<CandidateDensityStats>(m, "CandidateDensityStats")
      .def(py::init<>())
      .def_readwrite("schema_version", &CandidateDensityStats::schema_version)
      .def_readwrite("query_id", &CandidateDensityStats::query_id)
      .def_readwrite("proxy_count", &CandidateDensityStats::proxy_count)
      .def_readwrite("object_count", &CandidateDensityStats::object_count)
      .def_readwrite("slab_count", &CandidateDensityStats::slab_count)
      .def_readwrite("cross_object_same_slab_pair_count",
                     &CandidateDensityStats::cross_object_same_slab_pair_count)
      .def_readwrite("raw_hit_count", &CandidateDensityStats::raw_hit_count)
      .def_readwrite("compact_candidate_count", &CandidateDensityStats::compact_candidate_count)
      .def_readwrite("raw_hits_per_proxy", &CandidateDensityStats::raw_hits_per_proxy)
      .def_readwrite("candidates_per_proxy", &CandidateDensityStats::candidates_per_proxy)
      .def_readwrite("candidates_per_slab", &CandidateDensityStats::candidates_per_slab)
      .def_readwrite("aabb_overlap_ratio", &CandidateDensityStats::aabb_overlap_ratio)
      .def_readwrite("avg_rt_hits_per_candidate", &CandidateDensityStats::avg_rt_hits_per_candidate)
      .def_readwrite("timing", &CandidateDensityStats::timing)
      .def_readwrite("backend_name", &CandidateDensityStats::backend_name);

  py::class_<CandidateGenerationResult>(m, "CandidateGenerationResult")
      .def(py::init<>())
      .def_readwrite("backend_name", &CandidateGenerationResult::backend_name)
      .def_readwrite("raw_buffer", &CandidateGenerationResult::raw_buffer)
      .def_readwrite("candidates", &CandidateGenerationResult::candidates)
      .def_readwrite("timing", &CandidateGenerationResult::timing)
      .def_readwrite("density", &CandidateGenerationResult::density);

  py::class_<RuntimeQueryIdMapping>(m, "RuntimeQueryIdMapping")
      .def(py::init<>())
      .def_readwrite("source_query_id", &RuntimeQueryIdMapping::source_query_id)
      .def_readwrite("runtime_query_id", &RuntimeQueryIdMapping::runtime_query_id);

  py::class_<ExternalBatchCandidateResult>(m, "ExternalBatchCandidateResult")
      .def(py::init<>())
      .def_readwrite("backend_name", &ExternalBatchCandidateResult::backend_name)
      .def_readwrite("timing", &ExternalBatchCandidateResult::timing)
      .def_readwrite("primitive_count", &ExternalBatchCandidateResult::primitive_count)
      .def_readwrite("raw_hit_count", &ExternalBatchCandidateResult::raw_hit_count)
      .def_readwrite("compact_candidate_count",
                     &ExternalBatchCandidateResult::compact_candidate_count)
      .def_readwrite("candidate_recall", &ExternalBatchCandidateResult::candidate_recall)
      .def_readwrite("candidates", &ExternalBatchCandidateResult::candidates)
      .def_readwrite("runtime_query_ids", &ExternalBatchCandidateResult::runtime_query_ids);

  py::class_<ExternalBatchDummyProposalScheduleResult>(
      m, "ExternalBatchDummyProposalScheduleResult")
      .def(py::init<>())
      .def_readwrite("candidate_result",
                     &ExternalBatchDummyProposalScheduleResult::candidate_result)
      .def_readwrite("feature_rows",
                     &ExternalBatchDummyProposalScheduleResult::feature_rows)
      .def_readwrite("proposal_outputs",
                     &ExternalBatchDummyProposalScheduleResult::proposal_outputs)
      .def_readwrite("work_queue", &ExternalBatchDummyProposalScheduleResult::work_queue)
      .def_readwrite("stats", &ExternalBatchDummyProposalScheduleResult::stats)
      .def_readwrite("proposal_elapsed_ms",
                     &ExternalBatchDummyProposalScheduleResult::proposal_elapsed_ms);

  py::class_<MeshExactBuildConfig>(m, "MeshExactBuildConfig")
      .def(py::init<>())
      .def_readwrite("prune_by_swept_aabb", &MeshExactBuildConfig::prune_by_swept_aabb)
      .def_readwrite("max_point_triangle_primitives",
                     &MeshExactBuildConfig::max_point_triangle_primitives)
      .def_readwrite("max_edge_edge_primitives",
                     &MeshExactBuildConfig::max_edge_edge_primitives);

  py::class_<MeshExactBuildStats>(m, "MeshExactBuildStats")
      .def(py::init<>())
      .def_readwrite("vertex_count_a", &MeshExactBuildStats::vertex_count_a)
      .def_readwrite("vertex_count_b", &MeshExactBuildStats::vertex_count_b)
      .def_readwrite("triangle_count_a", &MeshExactBuildStats::triangle_count_a)
      .def_readwrite("triangle_count_b", &MeshExactBuildStats::triangle_count_b)
      .def_readwrite("edge_count_a", &MeshExactBuildStats::edge_count_a)
      .def_readwrite("edge_count_b", &MeshExactBuildStats::edge_count_b)
      .def_readwrite("point_triangle_total_pairs",
                     &MeshExactBuildStats::point_triangle_total_pairs)
      .def_readwrite("point_triangle_kept_pairs",
                     &MeshExactBuildStats::point_triangle_kept_pairs)
      .def_readwrite("point_triangle_pruned_pairs",
                     &MeshExactBuildStats::point_triangle_pruned_pairs)
      .def_readwrite("edge_edge_total_pairs", &MeshExactBuildStats::edge_edge_total_pairs)
      .def_readwrite("edge_edge_kept_pairs", &MeshExactBuildStats::edge_edge_kept_pairs)
      .def_readwrite("edge_edge_pruned_pairs", &MeshExactBuildStats::edge_edge_pruned_pairs);

  py::class_<MeshExactBuildResult>(m, "MeshExactBuildResult")
      .def(py::init<>())
      .def_readwrite("query", &MeshExactBuildResult::query)
      .def_readwrite("stats", &MeshExactBuildResult::stats);
}

void BindCertificateTypes(py::module_& m) {
  py::class_<LinearVertexTrajectory>(m, "LinearVertexTrajectory")
      .def(py::init<>())
      .def_readwrite("feature_id", &LinearVertexTrajectory::feature_id)
      .def_readwrite("position_t0", &LinearVertexTrajectory::position_t0)
      .def_readwrite("position_t1", &LinearVertexTrajectory::position_t1);

  py::class_<PointTriangleIntervalPrimitive>(m, "PointTriangleIntervalPrimitive")
      .def(py::init<>())
      .def_readwrite("point_id", &PointTriangleIntervalPrimitive::point_id)
      .def_readwrite("triangle_id", &PointTriangleIntervalPrimitive::triangle_id)
      .def_readwrite("point", &PointTriangleIntervalPrimitive::point)
      .def_readwrite("triangle_v0", &PointTriangleIntervalPrimitive::triangle_v0)
      .def_readwrite("triangle_v1", &PointTriangleIntervalPrimitive::triangle_v1)
      .def_readwrite("triangle_v2", &PointTriangleIntervalPrimitive::triangle_v2);

  py::class_<EdgeEdgeIntervalPrimitive>(m, "EdgeEdgeIntervalPrimitive")
      .def(py::init<>())
      .def_readwrite("edge_a_id", &EdgeEdgeIntervalPrimitive::edge_a_id)
      .def_readwrite("edge_b_id", &EdgeEdgeIntervalPrimitive::edge_b_id)
      .def_readwrite("edge_a0", &EdgeEdgeIntervalPrimitive::edge_a0)
      .def_readwrite("edge_a1", &EdgeEdgeIntervalPrimitive::edge_a1)
      .def_readwrite("edge_b0", &EdgeEdgeIntervalPrimitive::edge_b0)
      .def_readwrite("edge_b1", &EdgeEdgeIntervalPrimitive::edge_b1);

  py::class_<CertificateEngineConfig>(m, "CertificateEngineConfig")
      .def(py::init<>())
      .def_readwrite("eps_time", &CertificateEngineConfig::eps_time)
      .def_readwrite("eps_space", &CertificateEngineConfig::eps_space)
      .def_readwrite("max_subdivision_depth", &CertificateEngineConfig::max_subdivision_depth);

  py::class_<ExactCertificateQuery>(m, "ExactCertificateQuery")
      .def(py::init<>())
      .def_readwrite("work_item", &ExactCertificateQuery::work_item)
      .def_readwrite("config", &ExactCertificateQuery::config)
      .def_readwrite("point_triangle_primitives",
                     &ExactCertificateQuery::point_triangle_primitives)
      .def_readwrite("edge_edge_primitives", &ExactCertificateQuery::edge_edge_primitives);

  py::class_<PrimitiveIntervalResult>(m, "PrimitiveIntervalResult")
      .def(py::init<>())
      .def_readwrite("status", &PrimitiveIntervalResult::status)
      .def_readwrite("covered_feature_mask", &PrimitiveIntervalResult::covered_feature_mask)
      .def_readwrite("interval_t0", &PrimitiveIntervalResult::interval_t0)
      .def_readwrite("interval_t1", &PrimitiveIntervalResult::interval_t1)
      .def_readwrite("toi_upper", &PrimitiveIntervalResult::toi_upper)
      .def_readwrite("safe_margin_lb", &PrimitiveIntervalResult::safe_margin_lb)
      .def_readwrite("witness_family", &PrimitiveIntervalResult::witness_family)
      .def_readwrite("witness_id_a", &PrimitiveIntervalResult::witness_id_a)
      .def_readwrite("witness_id_b", &PrimitiveIntervalResult::witness_id_b)
      .def_readwrite("reason_code", &PrimitiveIntervalResult::reason_code)
      .def_readwrite("next_refinement_mode", &PrimitiveIntervalResult::next_refinement_mode);

  py::class_<ExactWorkQueueConfig>(m, "ExactWorkQueueConfig")
      .def(py::init<>())
      .def_readwrite("first_event_id", &ExactWorkQueueConfig::first_event_id)
      .def_readwrite("first_timestamp_us", &ExactWorkQueueConfig::first_timestamp_us)
      .def_readwrite("emit_dequeue_events", &ExactWorkQueueConfig::emit_dequeue_events);

  py::class_<ExactWorkQueueResult>(m, "ExactWorkQueueResult")
      .def(py::init<>())
      .def_readwrite("certificates", &ExactWorkQueueResult::certificates)
      .def_readwrite("audit_log", &ExactWorkQueueResult::audit_log)
      .def_readwrite("processed_count", &ExactWorkQueueResult::processed_count);

  py::class_<ExactRefinementConfig>(m, "ExactRefinementConfig")
      .def(py::init<>())
      .def_readwrite("first_child_work_item_id", &ExactRefinementConfig::first_child_work_item_id)
      .def_readwrite("max_child_depth", &ExactRefinementConfig::max_child_depth)
      .def_readwrite("min_interval_width", &ExactRefinementConfig::min_interval_width);

  py::class_<CertificateEngine>(m, "CertificateEngine")
      .def(py::init<>())
      .def("evaluate_work_item",
           static_cast<CertificateResult (CertificateEngine::*)(const ExactWorkItem&) const>(
               &CertificateEngine::Evaluate))
      .def("evaluate_query", [](const CertificateEngine& engine, const ExactCertificateQuery& query) {
        CertificateResult result;
        ThrowIfError(engine.Evaluate(query, &result));
        return result;
      });
}

}  // namespace
}  // namespace p2cccd

using namespace p2cccd;

PYBIND11_MODULE(p2cccd_cpp, m) {
  m.doc() = "P2CCCD CPU pybind11 bindings for contracts, RT candidates, exact certificates, and audit replay.";

  BindRuntimeContracts(m);
  BindGeometryAndCandidateTypes(m);
  BindCertificateTypes(m);
  BindProposalTypes(m);

  py::enum_<CandidateBackend>(m, "CandidateBackend")
      .value("CPU_REFERENCE", CandidateBackend::kCpuReference)
      .value("OPTIX", CandidateBackend::kOptix);

  py::class_<CandidateGeneratorConfig>(m, "CandidateGeneratorConfig")
      .def(py::init<>())
      .def_readwrite("backend", &CandidateGeneratorConfig::backend)
      .def_readwrite("allow_optix_cpu_fallback",
                     &CandidateGeneratorConfig::allow_optix_cpu_fallback);

  py::class_<CandidateGenerator>(m, "CandidateGenerator")
      .def(py::init<>())
      .def(py::init<CandidateGeneratorConfig>())
      .def("trace_candidates_for_proxy_scene",
           [](const CandidateGenerator& generator, const ProxyScene& scene, std::uint64_t query_id) {
             return generator.TraceCandidates(scene, query_id == 0 ? scene.query_id : query_id);
           },
           py::arg("scene"),
           py::arg("query_id") = 0);

  m.attr("FEATURE_FAMILY_POINT_TRIANGLE") =
      py::int_(static_cast<std::uint32_t>(kFeatureFamilyPointTriangle));
  m.attr("FEATURE_FAMILY_EDGE_EDGE") =
      py::int_(static_cast<std::uint32_t>(kFeatureFamilyEdgeEdge));
  m.attr("RAW_CANDIDATE_VALID") = py::int_(static_cast<std::uint32_t>(kRawCandidateValid));
  m.attr("RAW_CANDIDATE_AABB_OVERLAP") =
      py::int_(static_cast<std::uint32_t>(kRawCandidateAabbOverlap));
  m.attr("CERTIFICATE_REASON_NONE") = py::int_(static_cast<std::uint16_t>(kCertificateReasonNone));
  m.attr("CERTIFICATE_REASON_MISSING_GEOMETRY") =
      py::int_(static_cast<std::uint16_t>(kCertificateReasonMissingGeometry));
  m.attr("CERTIFICATE_REASON_MAX_SUBDIVISION_DEPTH") =
      py::int_(static_cast<std::uint16_t>(kCertificateReasonMaxSubdivisionDepth));
  m.attr("CERTIFICATE_REASON_INVALID_INPUT") =
      py::int_(static_cast<std::uint16_t>(kCertificateReasonInvalidInput));
  m.attr("EXACT_AUDIT_DEQUEUED") = py::int_(static_cast<std::uint16_t>(kExactAuditDequeued));
  m.attr("EXACT_AUDIT_COLLISION") = py::int_(static_cast<std::uint16_t>(kExactAuditCollision));
  m.attr("EXACT_AUDIT_SEPARATION") = py::int_(static_cast<std::uint16_t>(kExactAuditSeparation));
  m.attr("EXACT_AUDIT_UNDECIDED") = py::int_(static_cast<std::uint16_t>(kExactAuditUndecided));
  m.attr("EXACT_AUDIT_INVALID_INPUT") =
      py::int_(static_cast<std::uint16_t>(kExactAuditInvalidInput));
  m.attr("CUDA_DEVICE_POINTER_ABI_ENABLED") = py::bool_(false);

  m.def("validate_candidate_record", &ValidateCandidateRecordOrThrow, py::arg("record"));
  m.def("validate_exact_work_item", &ValidateExactWorkItemOrThrow, py::arg("item"));
  m.def("validate_certificate_result", &ValidateCertificateResultOrThrow, py::arg("result"));
  m.def("validate_audit_log_row",
        [](const AuditLogRow& row) {
          ThrowIfError(ValidateAuditLogRow(row));
          return true;
        },
        py::arg("row"));
  m.def("validate_benchmark_row", &ValidateBenchmarkRowOrThrow, py::arg("row"));
  m.def("validate_proxy_scene", &ValidateProxySceneOrThrow, py::arg("scene"));

  m.def("load_triangle_mesh", &LoadTriangleMeshOrThrow, py::arg("path"));
  m.def("validate_triangle_mesh", &ValidateTriangleMeshOrThrow, py::arg("mesh"));
  m.def("center_mesh_at_aabb_center", &CenterMeshAtAabbCenterOrThrow, py::arg("mesh"));
  m.def("build_proxy_scene", &BuildProxySceneOrThrow, py::arg("input"));
  m.def("generate_raw_candidates_cpu",
        &GenerateRawCandidatesCpuOrThrow,
        py::arg("scene"),
        py::arg("query_id") = 0);
  m.def("compact_raw_candidates",
        &CompactRawCandidatesOrThrow,
        py::arg("scene"),
        py::arg("raw_buffer"));
  m.def("generate_candidates_for_proxy_scene",
        &GenerateCandidatesForProxyScene,
        py::arg("scene"),
        py::arg("query_id") = 0,
        py::arg("backend_name") = "cpu_reference",
        py::arg("allow_optix_cpu_fallback") = false);
  m.def("generate_candidates_for_external_batch",
        &GenerateCandidatesForExternalBatchOrThrow,
        py::arg("batch"),
        py::arg("backend_name") = "cpu_reference",
        py::arg("allow_optix_cpu_fallback") = false);
  m.def("generate_external_batch_dummy_proposal_schedule",
        &GenerateExternalBatchDummyProposalScheduleOrThrow,
        py::arg("batch"),
        py::arg("backend_name") = "cpu_reference",
        py::arg("allow_optix_cpu_fallback") = false,
        py::arg("config"),
        py::arg("materialize_artifacts") = false);
  m.def("build_runtime_proposal_feature_rows",
        &BuildRuntimeProposalFeatureRowsOrThrow,
        py::arg("candidates"),
        py::arg("primitive_count"),
        py::arg("raw_hit_count"),
        py::arg("compact_candidate_count"),
        py::arg("conservative_family_masks_by_query_id"));
  m.def("build_runtime_proposal_feature_arrays",
        &BuildRuntimeProposalFeatureArraysOrThrow,
        py::arg("candidates"),
        py::arg("primitive_count"),
        py::arg("raw_hit_count"),
        py::arg("compact_candidate_count"),
        py::arg("conservative_family_masks_by_query_id"));
  m.def("run_dummy_runtime_proposal_schedule",
        &RunDummyProposalScheduleFromRuntimeCandidatesOrThrow,
        py::arg("candidates"),
        py::arg("primitive_count"),
        py::arg("raw_hit_count"),
        py::arg("compact_candidate_count"),
        py::arg("conservative_family_masks_by_query_id"),
        py::arg("config"),
        py::arg("materialize_artifacts") = true);
  m.def("schedule_runtime_exact_work_items",
        &ScheduleRuntimeExactWorkItemsFromProposalsOrThrow,
        py::arg("candidates"),
        py::arg("rows"),
        py::arg("proposal_outputs"),
        py::arg("conservative_family_masks_by_query_id"),
        py::arg("config"));
  m.def("schedule_runtime_exact_work_items_from_arrays",
        &ScheduleRuntimeExactWorkItemsFromArraysOrThrow,
        py::arg("candidates"),
        py::arg("feature_arrays"),
        py::arg("prediction_arrays"),
        py::arg("conservative_family_masks_by_query_id"),
        py::arg("config"));
  m.def("schedule_runtime_exact_work_items_from_proposal_arrays",
        &ScheduleRuntimeExactWorkItemsFromProposalArraysOrThrow,
        py::arg("feature_arrays"),
        py::arg("prediction_arrays"),
        py::arg("conservative_family_masks_by_query_id"),
        py::arg("config"));
  m.def("run_native_dense_group_exact_early_stop",
        &RunNativeDenseGroupExactEarlyStopOrThrow,
        py::arg("feature_arrays"),
        py::arg("prediction_arrays"),
        py::arg("uncertainty_fallback_threshold") = 0.75F,
        py::arg("representative_attempt_limit") = 3U,
        py::arg("interval_miss_penalty_scale") = 0.22,
        py::arg("preserve_input_order") = false);
  m.def("build_mesh_exact_certificate_query",
        &BuildMeshExactCertificateQueryOrThrow,
        py::arg("mesh_a"),
        py::arg("translation_a_t0"),
        py::arg("translation_a_t1"),
        py::arg("mesh_b"),
        py::arg("translation_b_t0"),
        py::arg("translation_b_t1"),
        py::arg("work_item"),
        py::arg("config"),
        py::arg("build_config"));

  m.def("evaluate_point_triangle_interval",
        &EvaluatePointTriangleIntervalOrThrow,
        py::arg("primitive"),
        py::arg("interval_t0"),
        py::arg("interval_t1"),
        py::arg("config"));
  m.def("evaluate_edge_edge_interval",
        &EvaluateEdgeEdgeIntervalOrThrow,
        py::arg("primitive"),
        py::arg("interval_t0"),
        py::arg("interval_t1"),
        py::arg("config"));
  m.def("is_cuda_exact_built", &IsCudaExactBuilt);
  m.def("cuda_binding_status", &CudaBindingStatus);
  m.def("evaluate_point_triangle_batch_cuda",
        &EvaluatePointTriangleBatchCudaOrThrow,
        py::arg("primitives"),
        py::arg("interval_t0"),
        py::arg("interval_t1"),
        py::arg("config"));
  m.def("evaluate_edge_edge_batch_cuda",
        &EvaluateEdgeEdgeBatchCudaOrThrow,
        py::arg("primitives"),
        py::arg("interval_t0"),
        py::arg("interval_t1"),
        py::arg("config"));
  m.def("cross_check_cpu_cuda_exact",
        &CrossCheckCpuCudaExactOrThrow,
        py::arg("point_triangles"),
        py::arg("edge_edges"),
        py::arg("interval_t0"),
        py::arg("interval_t1"),
        py::arg("config"),
        py::arg("eps_cert"));
  m.def("evaluate_certificate_query_cpu",
        &EvaluateCertificateQueryCpuOrThrow,
        py::arg("query"));
  m.def("process_exact_work_queue_cpu",
        &ProcessExactWorkQueueCpuOrThrow,
        py::arg("work_queue"),
        py::arg("config"));
  m.def("validate_exact_work_queue_coverage",
        &ValidateExactWorkQueueCoverageOrThrow,
        py::arg("work_queue"),
        py::arg("result"));
  m.def("generate_conservative_refinement_work_items",
        &GenerateConservativeRefinementWorkItemsOrThrow,
        py::arg("parent"),
        py::arg("certificate"),
        py::arg("config"));
  m.def("validate_audit_log_rows",
        &ValidateAuditLogRowsOrThrow,
        py::arg("rows"));
  m.def("audit_log_rows_for_query",
        &AuditLogRowsForQuery,
        py::arg("rows"),
        py::arg("query_id"));
}
