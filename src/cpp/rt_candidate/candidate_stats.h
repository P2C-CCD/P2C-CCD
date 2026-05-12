#pragma once

#include "common/status.h"
#include "rt_candidate/candidate_generation_result.h"
#include "rt_candidate/proxy_scene.h"

#include <filesystem>
#include <string>
#include <vector>

namespace p2cccd {

Status ComputeCandidateDensityStats(const ProxyScene& scene,
                                    const RawCandidateBuffer& raw_buffer,
                                    const std::vector<CandidateRecord>& candidates,
                                    const RtCandidateTiming& timing,
                                    const std::string& backend_name,
                                    CandidateDensityStats* stats);

std::string CandidateDensityCsvHeader();
Status WriteCandidateDensityCsv(const std::filesystem::path& path,
                                const std::vector<CandidateDensityStats>& rows,
                                bool append);
Status WriteCandidateDensityJsonl(const std::filesystem::path& path,
                                  const std::vector<CandidateDensityStats>& rows,
                                  bool append);

}  // namespace p2cccd
