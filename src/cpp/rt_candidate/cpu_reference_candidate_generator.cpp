#include "rt_candidate/cpu_reference_candidate_generator.h"

#include "rt_candidate/candidate_stats.h"

#include <chrono>
#include <utility>

namespace p2cccd {
namespace {

using Clock = std::chrono::steady_clock;

double ElapsedMilliseconds(const Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

}  // namespace

Status CpuReferenceCandidateGenerator::Generate(const ProxyScene& scene,
                                                CandidateGenerationResult* result) const {
  if (result == nullptr) {
    return Status::Error("candidate generation result output pointer is null");
  }

  CandidateGenerationResult generated;
  generated.backend_name = "cpu_reference";

  const Clock::time_point total_start = Clock::now();

  Clock::time_point stage_start = Clock::now();
  if (auto status = ValidateProxyScene(scene); !status.ok) {
    return status;
  }
  generated.timing.build_ms = ElapsedMilliseconds(stage_start);

  stage_start = Clock::now();
  generated.timing.update_ms = ElapsedMilliseconds(stage_start);

  stage_start = Clock::now();
  if (auto status = GenerateRawCandidatesCpu(scene, scene.query_id, &generated.raw_buffer);
      !status.ok) {
    return status;
  }
  generated.timing.trace_ms = ElapsedMilliseconds(stage_start);

  stage_start = Clock::now();
  if (auto status = CompactRawCandidates(scene, generated.raw_buffer, &generated.candidates);
      !status.ok) {
    return status;
  }
  generated.timing.compact_ms = ElapsedMilliseconds(stage_start);

  stage_start = Clock::now();
  generated.timing.total_ms = ElapsedMilliseconds(total_start);
  if (auto status = ComputeCandidateDensityStats(scene,
                                                 generated.raw_buffer,
                                                 generated.candidates,
                                                 generated.timing,
                                                 generated.backend_name,
                                                 &generated.density);
      !status.ok) {
    return status;
  }
  generated.timing.stats_ms = ElapsedMilliseconds(stage_start);
  generated.timing.total_ms = ElapsedMilliseconds(total_start);
  generated.density.timing = generated.timing;

  *result = std::move(generated);
  return Status::Ok();
}

}  // namespace p2cccd
