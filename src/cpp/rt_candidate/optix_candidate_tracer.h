#pragma once

#include "common/status.h"
#include "rt_candidate/candidate_generation_result.h"
#include "rt_candidate/proxy_scene.h"

namespace p2cccd {

struct OptixCandidateTracerConfig {
  bool allow_cpu_fallback = false;
};

class OptixCandidateTracer {
 public:
  explicit OptixCandidateTracer(OptixCandidateTracerConfig config = {});

  static bool IsBuildEnabled();
  Status Generate(const ProxyScene& scene, CandidateGenerationResult* result) const;

 private:
  OptixCandidateTracerConfig config_;
};

}  // namespace p2cccd
