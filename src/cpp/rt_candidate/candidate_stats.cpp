#include "rt_candidate/candidate_stats.h"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <set>
#include <sstream>
#include <tuple>
#include <utility>

namespace p2cccd {
namespace {

double SafeRatio(const std::uint64_t numerator, const std::uint64_t denominator) {
  if (denominator == 0) {
    return 0.0;
  }
  return static_cast<double>(numerator) / static_cast<double>(denominator);
}

std::uint64_t SumRtHits(const std::vector<CandidateRecord>& candidates) {
  std::uint64_t sum = 0;
  for (const CandidateRecord& candidate : candidates) {
    sum += candidate.rt_hit_count;
  }
  return sum;
}

std::uint64_t CountCrossObjectSameSlabPairs(const ProxyScene& scene) {
  std::uint64_t pair_count = 0;
  for (std::uint32_t i = 0; i < scene.primitives.size(); ++i) {
    const ProxyPrimitive& lhs = scene.primitives[i];
    for (std::uint32_t j = i + 1; j < scene.primitives.size(); ++j) {
      const ProxyPrimitive& rhs = scene.primitives[j];
      if (lhs.object_id != rhs.object_id && lhs.slab_id == rhs.slab_id) {
        ++pair_count;
      }
    }
  }
  return pair_count;
}

Status EnsureParentDirectory(const std::filesystem::path& path) {
  const std::filesystem::path parent = path.parent_path();
  if (parent.empty()) {
    return Status::Ok();
  }

  std::error_code ec;
  std::filesystem::create_directories(parent, ec);
  if (ec) {
    return Status::Error("failed to create candidate density export directory: " +
                         ec.message());
  }
  return Status::Ok();
}

std::string EscapeJsonString(const std::string& input) {
  std::ostringstream stream;
  for (const char ch : input) {
    switch (ch) {
      case '\\':
        stream << "\\\\";
        break;
      case '"':
        stream << "\\\"";
        break;
      case '\n':
        stream << "\\n";
        break;
      case '\r':
        stream << "\\r";
        break;
      case '\t':
        stream << "\\t";
        break;
      default:
        stream << ch;
        break;
    }
  }
  return stream.str();
}

void WriteCsvRow(std::ostream& stream, const CandidateDensityStats& row) {
  stream << row.schema_version << ',' << row.query_id << ',' << row.proxy_count << ','
         << row.object_count << ',' << row.slab_count << ','
         << row.cross_object_same_slab_pair_count << ',' << row.raw_hit_count << ','
         << row.compact_candidate_count << ',' << row.raw_hits_per_proxy << ','
         << row.candidates_per_proxy << ',' << row.candidates_per_slab << ','
         << row.aabb_overlap_ratio << ',' << row.avg_rt_hits_per_candidate << ','
         << row.timing.build_ms << ',' << row.timing.update_ms << ','
         << row.timing.trace_ms << ',' << row.timing.compact_ms << ','
         << row.timing.stats_ms << ',' << row.timing.total_ms << ','
         << '"' << row.backend_name << '"' << '\n';
}

void WriteJsonRow(std::ostream& stream, const CandidateDensityStats& row) {
  stream << '{'
         << "\"schema_version\":" << row.schema_version << ','
         << "\"query_id\":" << row.query_id << ','
         << "\"proxy_count\":" << row.proxy_count << ','
         << "\"object_count\":" << row.object_count << ','
         << "\"slab_count\":" << row.slab_count << ','
         << "\"cross_object_same_slab_pair_count\":"
         << row.cross_object_same_slab_pair_count << ','
         << "\"raw_hit_count\":" << row.raw_hit_count << ','
         << "\"compact_candidate_count\":" << row.compact_candidate_count << ','
         << "\"raw_hits_per_proxy\":" << row.raw_hits_per_proxy << ','
         << "\"candidates_per_proxy\":" << row.candidates_per_proxy << ','
         << "\"candidates_per_slab\":" << row.candidates_per_slab << ','
         << "\"aabb_overlap_ratio\":" << row.aabb_overlap_ratio << ','
         << "\"avg_rt_hits_per_candidate\":" << row.avg_rt_hits_per_candidate << ','
         << "\"build_ms\":" << row.timing.build_ms << ','
         << "\"update_ms\":" << row.timing.update_ms << ','
         << "\"trace_ms\":" << row.timing.trace_ms << ','
         << "\"compact_ms\":" << row.timing.compact_ms << ','
         << "\"stats_ms\":" << row.timing.stats_ms << ','
         << "\"total_ms\":" << row.timing.total_ms << ','
         << "\"backend_name\":\"" << EscapeJsonString(row.backend_name) << "\""
         << "}\n";
}

}  // namespace

Status ComputeCandidateDensityStats(const ProxyScene& scene,
                                    const RawCandidateBuffer& raw_buffer,
                                    const std::vector<CandidateRecord>& candidates,
                                    const RtCandidateTiming& timing,
                                    const std::string& backend_name,
                                    CandidateDensityStats* stats) {
  if (stats == nullptr) {
    return Status::Error("candidate density stats output pointer is null");
  }
  if (auto status = ValidateProxyScene(scene); !status.ok) {
    return status;
  }

  std::set<std::uint32_t> object_ids;
  std::set<std::uint32_t> slab_ids;
  for (const ProxyPrimitive& primitive : scene.primitives) {
    object_ids.insert(primitive.object_id);
    slab_ids.insert(primitive.slab_id);
  }

  CandidateDensityStats computed;
  computed.query_id = scene.query_id;
  computed.proxy_count = scene.primitives.size();
  computed.object_count = object_ids.size();
  computed.slab_count = slab_ids.size();
  computed.cross_object_same_slab_pair_count = CountCrossObjectSameSlabPairs(scene);
  computed.raw_hit_count = raw_buffer.hits.size();
  computed.compact_candidate_count = candidates.size();
  computed.raw_hits_per_proxy = SafeRatio(computed.raw_hit_count, computed.proxy_count);
  computed.candidates_per_proxy =
      SafeRatio(computed.compact_candidate_count, computed.proxy_count);
  computed.candidates_per_slab =
      SafeRatio(computed.compact_candidate_count, computed.slab_count);
  computed.aabb_overlap_ratio =
      SafeRatio(computed.raw_hit_count, computed.cross_object_same_slab_pair_count);
  computed.avg_rt_hits_per_candidate =
      SafeRatio(SumRtHits(candidates), computed.compact_candidate_count);
  computed.timing = timing;
  computed.backend_name = backend_name;

  *stats = std::move(computed);
  return Status::Ok();
}

std::string CandidateDensityCsvHeader() {
  return "schema_version,query_id,proxy_count,object_count,slab_count,"
         "cross_object_same_slab_pair_count,raw_hit_count,compact_candidate_count,"
         "raw_hits_per_proxy,candidates_per_proxy,candidates_per_slab,"
         "aabb_overlap_ratio,avg_rt_hits_per_candidate,build_ms,update_ms,"
         "trace_ms,compact_ms,stats_ms,total_ms,backend_name";
}

Status WriteCandidateDensityCsv(const std::filesystem::path& path,
                                const std::vector<CandidateDensityStats>& rows,
                                const bool append) {
  if (auto status = EnsureParentDirectory(path); !status.ok) {
    return status;
  }

  const bool write_header = !append || !std::filesystem::exists(path) ||
                            std::filesystem::file_size(path) == 0;
  std::ofstream stream(path, append ? std::ios::app : std::ios::trunc);
  if (!stream) {
    return Status::Error("failed to open candidate density CSV export");
  }

  stream << std::setprecision(17);
  if (write_header) {
    stream << CandidateDensityCsvHeader() << '\n';
  }
  for (const CandidateDensityStats& row : rows) {
    WriteCsvRow(stream, row);
  }
  return Status::Ok();
}

Status WriteCandidateDensityJsonl(const std::filesystem::path& path,
                                  const std::vector<CandidateDensityStats>& rows,
                                  const bool append) {
  if (auto status = EnsureParentDirectory(path); !status.ok) {
    return status;
  }

  std::ofstream stream(path, append ? std::ios::app : std::ios::trunc);
  if (!stream) {
    return Status::Error("failed to open candidate density JSONL export");
  }

  stream << std::setprecision(17);
  for (const CandidateDensityStats& row : rows) {
    WriteJsonRow(stream, row);
  }
  return Status::Ok();
}

}  // namespace p2cccd
