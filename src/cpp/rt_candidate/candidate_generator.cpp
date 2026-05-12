#include "rt_candidate/candidate_generator.h"

#include "rt_candidate/cpu_reference_candidate_generator.h"
#include "rt_candidate/optix_candidate_tracer.h"

#include <utility>

namespace p2cccd {

CandidateGenerator::CandidateGenerator(CandidateGeneratorConfig config)
    : config_(std::move(config)) {}

Status CandidateGenerator::GenerateCandidates(const ProxyScene& scene,
                                              const std::uint64_t query_id,
                                              CandidateGenerationResult* result) const {
  if (result == nullptr) {
    return Status::Error("candidate generation result output pointer is null");
  }
  if (query_id == 0) {
    return Status::Error("query_id is required");
  }
  if (query_id != scene.query_id) {
    return Status::Error("query_id must match ProxyScene.query_id");
  }

  if (config_.backend == CandidateBackend::kOptix) {
    OptixCandidateTracer tracer({config_.allow_optix_cpu_fallback});
    return tracer.Generate(scene, result);
  }

  CpuReferenceCandidateGenerator generator;
  return generator.Generate(scene, result);
}

std::vector<CandidateRecord> CandidateGenerator::TraceCandidates(const CcdQuery& query) const {
  (void)query;
  return {};
}

std::vector<CandidateRecord> CandidateGenerator::TraceCandidates(const ProxyScene& scene,
                                                                 std::uint64_t query_id) const {
  CandidateGenerationResult result;
  if (auto status = GenerateCandidates(scene, query_id, &result); !status.ok) {
    return {};
  }
  return result.candidates;
}

}  // namespace p2cccd
