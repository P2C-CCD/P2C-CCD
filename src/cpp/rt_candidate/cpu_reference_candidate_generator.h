#pragma once

#include "common/status.h"
#include "rt_candidate/candidate_generation_result.h"
#include "rt_candidate/proxy_scene.h"

namespace p2cccd {

class CpuReferenceCandidateGenerator {
 public:
  Status Generate(const ProxyScene& scene, CandidateGenerationResult* result) const;
};

}  // namespace p2cccd
