#pragma once

#include <vector>

#include "common/runtime_contracts.h"
#include "common/status.h"
#include "geometry/motion.h"
#include "rt_candidate/candidate_generation_result.h"
#include "rt_candidate/proxy_scene.h"

namespace p2cccd {

enum class CandidateBackend {
  kCpuReference,
  kOptix,
};

struct CandidateGeneratorConfig {
  CandidateBackend backend = CandidateBackend::kCpuReference;
  bool allow_optix_cpu_fallback = false;
};

class CandidateGenerator {
 public:
  CandidateGenerator() = default;
  explicit CandidateGenerator(CandidateGeneratorConfig config);

  Status GenerateCandidates(const ProxyScene& scene,
                            std::uint64_t query_id,
                            CandidateGenerationResult* result) const;
  std::vector<CandidateRecord> TraceCandidates(const CcdQuery& query) const;
  std::vector<CandidateRecord> TraceCandidates(const ProxyScene& scene,
                                               std::uint64_t query_id) const;

 private:
  CandidateGeneratorConfig config_;
};

}  // namespace p2cccd
