#include <tight_inclusion/ccd.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct Args {
  fs::path dataset_root;
  fs::path learned_schedule;
  fs::path random_schedule;
  fs::path output_json;
  fs::path output_md;
  fs::path output_csv;
  double ms = 0.0;
  double tolerance = 1.0e-6;
  double t_max = 1.0;
  long max_itr = 1000000;
};

struct ScheduledCandidate {
  std::uint64_t group_id = 0;
  std::string case_name;
  std::string kind;
  std::string csv_path;
  std::uint64_t query_index = 0;
  double score = 0.0;
};

struct QueryBlock {
  std::array<ticcd::Vector3, 8> vertices;
  bool truth = false;
};

struct MethodStats {
  std::string method;
  std::uint64_t group_count = 0;
  std::uint64_t candidate_count = 0;
  std::uint64_t positive_group_count = 0;
  std::uint64_t exact_calls = 0;
  std::uint64_t skipped_exact_calls = 0;
  std::uint64_t tp = 0;
  std::uint64_t tn = 0;
  std::uint64_t fp = 0;
  std::uint64_t fn = 0;
  double exact_ms = 0.0;
  double wall_ms = 0.0;
  double first_positive_rank_sum = 0.0;
};

std::string NormalizeSlashes(std::string value) {
  std::replace(value.begin(), value.end(), '\\', '/');
  return value;
}

std::string JsonEscape(const std::string& value) {
  std::string out;
  out.reserve(value.size() + 8);
  for (const char ch : value) {
    if (ch == '"' || ch == '\\') {
      out.push_back('\\');
    }
    out.push_back(ch);
  }
  return out;
}

std::vector<std::string> SplitCSVLine(const std::string& line) {
  std::vector<std::string> parts;
  std::string token;
  std::istringstream stream(line);
  while (std::getline(stream, token, ',')) {
    while (!token.empty() &&
           (token.back() == '\r' || token.back() == ' ' || token.back() == '\t')) {
      token.pop_back();
    }
    parts.push_back(token);
  }
  return parts;
}

std::string CandidateKey(const ScheduledCandidate& candidate) {
  return candidate.kind + "|" + candidate.csv_path + "|" + std::to_string(candidate.query_index);
}

std::string CandidateKey(const std::string& kind,
                         const std::string& csv_path,
                         const std::uint64_t query_index) {
  return kind + "|" + csv_path + "|" + std::to_string(query_index);
}

ticcd::Vector3 ParseVertex(const std::string& line, bool* truth) {
  const std::vector<std::string> parts = SplitCSVLine(line);
  if (parts.size() != 7) {
    throw std::runtime_error("Tight-Inclusion CSV row must contain 7 columns");
  }
  const long double x = std::stold(parts[0]) / std::stold(parts[1]);
  const long double y = std::stold(parts[2]) / std::stold(parts[3]);
  const long double z = std::stold(parts[4]) / std::stold(parts[5]);
  if (parts[6] != "0" && parts[6] != "1") {
    throw std::runtime_error("truth column must be 0 or 1");
  }
  *truth = parts[6] == "1";
  return ticcd::Vector3(static_cast<ticcd::Scalar>(x),
                        static_cast<ticcd::Scalar>(y),
                        static_cast<ticcd::Scalar>(z));
}

Args ParseArgs(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    auto need_value = [&](const std::string& name) -> std::string {
      if (i + 1 >= argc) {
        throw std::runtime_error("missing value for " + name);
      }
      return argv[++i];
    };
    if (key == "--dataset-root") {
      args.dataset_root = need_value(key);
    } else if (key == "--learned-schedule") {
      args.learned_schedule = need_value(key);
    } else if (key == "--random-schedule") {
      args.random_schedule = need_value(key);
    } else if (key == "--output-json") {
      args.output_json = need_value(key);
    } else if (key == "--output-md") {
      args.output_md = need_value(key);
    } else if (key == "--output-csv") {
      args.output_csv = need_value(key);
    } else if (key == "--ms") {
      args.ms = std::stod(need_value(key));
    } else if (key == "--tolerance") {
      args.tolerance = std::stod(need_value(key));
    } else if (key == "--t-max") {
      args.t_max = std::stod(need_value(key));
    } else if (key == "--max-itr") {
      args.max_itr = std::stol(need_value(key));
    } else {
      throw std::runtime_error("unknown argument: " + key);
    }
  }
  if (args.dataset_root.empty() || args.learned_schedule.empty() || args.random_schedule.empty() ||
      args.output_json.empty() || args.output_md.empty() || args.output_csv.empty()) {
    throw std::runtime_error(
        "--dataset-root, --learned-schedule, --random-schedule, --output-json, --output-md and --output-csv are required");
  }
  return args;
}

std::vector<ScheduledCandidate> LoadSchedule(const fs::path& path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("failed to open schedule " + path.string());
  }
  std::vector<ScheduledCandidate> out;
  std::string line;
  bool first_line = true;
  while (std::getline(in, line)) {
    if (line.empty()) {
      continue;
    }
    if (first_line && line.find("group_id") != std::string::npos) {
      first_line = false;
      continue;
    }
    first_line = false;
    const std::vector<std::string> parts = SplitCSVLine(line);
    if (parts.size() != 6) {
      throw std::runtime_error("schedule row must be group_id,case,kind,csv_path,query_index,score");
    }
    ScheduledCandidate item;
    item.group_id = static_cast<std::uint64_t>(std::stoull(parts[0]));
    item.case_name = parts[1];
    item.kind = parts[2];
    item.csv_path = NormalizeSlashes(parts[3]);
    item.query_index = static_cast<std::uint64_t>(std::stoull(parts[4]));
    item.score = std::stod(parts[5]);
    out.push_back(item);
  }
  return out;
}

void SortByGroupAndScore(std::vector<ScheduledCandidate>* schedule) {
  std::stable_sort(schedule->begin(), schedule->end(), [](const auto& lhs, const auto& rhs) {
    if (lhs.group_id != rhs.group_id) {
      return lhs.group_id < rhs.group_id;
    }
    if (lhs.score != rhs.score) {
      return lhs.score > rhs.score;
    }
    if (lhs.csv_path != rhs.csv_path) {
      return lhs.csv_path < rhs.csv_path;
    }
    return lhs.query_index < rhs.query_index;
  });
}

void SortByGroupOnly(std::vector<ScheduledCandidate>* schedule) {
  std::stable_sort(schedule->begin(), schedule->end(), [](const auto& lhs, const auto& rhs) {
    if (lhs.group_id != rhs.group_id) {
      return lhs.group_id < rhs.group_id;
    }
    if (lhs.csv_path != rhs.csv_path) {
      return lhs.csv_path < rhs.csv_path;
    }
    return lhs.query_index < rhs.query_index;
  });
}

std::unordered_map<std::string, QueryBlock> PreloadQueryBlocks(
    const fs::path& dataset_root,
    const std::vector<ScheduledCandidate>& learned,
    const std::vector<ScheduledCandidate>& random_schedule,
    double* preload_ms) {
  const auto begin = std::chrono::steady_clock::now();
  std::map<std::string, std::set<std::uint64_t>> wanted;
  std::unordered_map<std::string, std::string> kind_for_path_query;
  auto add_schedule = [&](const std::vector<ScheduledCandidate>& schedule) {
    for (const ScheduledCandidate& candidate : schedule) {
      wanted[candidate.csv_path].insert(candidate.query_index);
      kind_for_path_query[candidate.csv_path + "|" + std::to_string(candidate.query_index)] =
          candidate.kind;
    }
  };
  add_schedule(learned);
  add_schedule(random_schedule);

  std::unordered_map<std::string, QueryBlock> blocks;
  blocks.reserve(learned.size() + random_schedule.size());
  for (const auto& [relative_csv, indices] : wanted) {
    const fs::path csv_path = dataset_root / fs::path(relative_csv);
    std::ifstream in(csv_path);
    if (!in) {
      throw std::runtime_error("failed to open " + csv_path.string());
    }
    std::string line;
    std::uint64_t current_query = 0;
    auto next_it = indices.begin();
    while (next_it != indices.end()) {
      const std::uint64_t target = *next_it;
      while (current_query < target) {
        for (int row = 0; row < 8; ++row) {
          if (!std::getline(in, line)) {
            throw std::runtime_error("query index out of range in " + csv_path.string());
          }
        }
        ++current_query;
      }
      QueryBlock block;
      bool first_truth = false;
      for (std::size_t row = 0; row < 8; ++row) {
        if (!std::getline(in, line)) {
          throw std::runtime_error("truncated query in " + csv_path.string());
        }
        bool truth = false;
        block.vertices[row] = ParseVertex(line, &truth);
        if (row == 0) {
          first_truth = truth;
        } else if (truth != first_truth) {
          throw std::runtime_error("truth changes inside query block in " + csv_path.string());
        }
      }
      block.truth = first_truth;
      const std::string kind = kind_for_path_query.at(relative_csv + "|" + std::to_string(target));
      blocks[CandidateKey(kind, relative_csv, target)] = block;
      ++current_query;
      ++next_it;
    }
  }
  const auto end = std::chrono::steady_clock::now();
  *preload_ms = std::chrono::duration<double, std::milli>(end - begin).count();
  return blocks;
}

bool RunTightInclusionQuery(const std::string& kind,
                            const std::array<ticcd::Vector3, 8>& vertices,
                            const Args& args) {
  const ticcd::Array3 err(-1, -1, -1);
  ticcd::Scalar toi = std::numeric_limits<ticcd::Scalar>::infinity();
  ticcd::Scalar output_tolerance = static_cast<ticcd::Scalar>(args.tolerance);
  if (kind == "edge-edge") {
    return ticcd::edgeEdgeCCD(vertices[0], vertices[1], vertices[2], vertices[3],
                              vertices[4], vertices[5], vertices[6], vertices[7],
                              err, static_cast<ticcd::Scalar>(args.ms), toi,
                              static_cast<ticcd::Scalar>(args.tolerance),
                              static_cast<ticcd::Scalar>(args.t_max), args.max_itr,
                              output_tolerance, false,
                              ticcd::CCDRootFindingMethod::BREADTH_FIRST_SEARCH);
  }
  return ticcd::vertexFaceCCD(vertices[0], vertices[1], vertices[2], vertices[3],
                              vertices[4], vertices[5], vertices[6], vertices[7],
                              err, static_cast<ticcd::Scalar>(args.ms), toi,
                              static_cast<ticcd::Scalar>(args.tolerance),
                              static_cast<ticcd::Scalar>(args.t_max), args.max_itr,
                              output_tolerance, false,
                              ticcd::CCDRootFindingMethod::BREADTH_FIRST_SEARCH);
}

MethodStats EvaluateMethod(const Args& args,
                           std::vector<ScheduledCandidate> schedule,
                           const std::unordered_map<std::string, QueryBlock>& blocks,
                           const std::string& method,
                           const bool early_stop) {
  if (early_stop) {
    SortByGroupAndScore(&schedule);
  } else {
    SortByGroupOnly(&schedule);
  }
  MethodStats stats;
  stats.method = method;
  stats.candidate_count = schedule.size();
  const auto wall_begin = std::chrono::steady_clock::now();

  std::size_t group_begin = 0;
  while (group_begin < schedule.size()) {
    std::size_t group_end = group_begin + 1;
    while (group_end < schedule.size() && schedule[group_end].group_id == schedule[group_begin].group_id) {
      ++group_end;
    }
    ++stats.group_count;
    bool truth = false;
    bool predicted = false;
    bool first_positive_recorded = false;
    std::uint64_t rank = 0;
    for (std::size_t pos = group_begin; pos < group_end; ++pos) {
      const auto block_it = blocks.find(CandidateKey(schedule[pos]));
      if (block_it == blocks.end()) {
        throw std::runtime_error("candidate query block was not preloaded");
      }
      truth = truth || block_it->second.truth;
    }
    if (truth) {
      ++stats.positive_group_count;
    }

    for (std::size_t pos = group_begin; pos < group_end; ++pos) {
      ++rank;
      ++stats.exact_calls;
      const auto block_it = blocks.find(CandidateKey(schedule[pos]));
      const auto exact_begin = std::chrono::steady_clock::now();
      const bool result = RunTightInclusionQuery(schedule[pos].kind, block_it->second.vertices, args);
      const auto exact_end = std::chrono::steady_clock::now();
      stats.exact_ms += std::chrono::duration<double, std::milli>(exact_end - exact_begin).count();
      predicted = predicted || result;
      if (result && !first_positive_recorded) {
        stats.first_positive_rank_sum += static_cast<double>(rank);
        first_positive_recorded = true;
      }
      if (early_stop && result) {
        break;
      }
    }

    const std::uint64_t group_size = static_cast<std::uint64_t>(group_end - group_begin);
    if (stats.exact_calls <= stats.candidate_count) {
      // Filled after loop using global counts.
    }
    if (!first_positive_recorded && truth) {
      stats.first_positive_rank_sum += static_cast<double>(group_size + 1);
    }
    if (truth && predicted) {
      ++stats.tp;
    } else if (!truth && !predicted) {
      ++stats.tn;
    } else if (!truth && predicted) {
      ++stats.fp;
    } else {
      ++stats.fn;
    }
    group_begin = group_end;
  }
  const auto wall_end = std::chrono::steady_clock::now();
  stats.wall_ms = std::chrono::duration<double, std::milli>(wall_end - wall_begin).count();
  stats.skipped_exact_calls = stats.candidate_count - stats.exact_calls;
  return stats;
}

double Recall(const MethodStats& stats) {
  return static_cast<double>(stats.tp) /
         static_cast<double>(std::max<std::uint64_t>(1, stats.tp + stats.fn));
}

double Precision(const MethodStats& stats) {
  return static_cast<double>(stats.tp) /
         static_cast<double>(std::max<std::uint64_t>(1, stats.tp + stats.fp));
}

double CallReduction(const MethodStats& stats) {
  return 1.0 - static_cast<double>(stats.exact_calls) /
                   static_cast<double>(std::max<std::uint64_t>(1, stats.candidate_count));
}

double FirstRank(const MethodStats& stats) {
  return stats.positive_group_count == 0
             ? 0.0
             : stats.first_positive_rank_sum / static_cast<double>(stats.positive_group_count);
}

void WriteCsv(const fs::path& path, const std::vector<MethodStats>& rows) {
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("failed to write " + path.string());
  }
  out << "method,groups,candidates,positive_groups,exact_calls,skipped_exact_calls,"
         "exact_call_reduction,tp,tn,fp,fn,recall,precision,first_positive_rank_mean,exact_ms,wall_ms\n";
  out << std::setprecision(12);
  for (const MethodStats& stats : rows) {
    out << stats.method << "," << stats.group_count << "," << stats.candidate_count << ","
        << stats.positive_group_count << "," << stats.exact_calls << ","
        << stats.skipped_exact_calls << "," << CallReduction(stats) << "," << stats.tp << ","
        << stats.tn << "," << stats.fp << "," << stats.fn << "," << Recall(stats) << ","
        << Precision(stats) << "," << FirstRank(stats) << "," << stats.exact_ms << ","
        << stats.wall_ms << "\n";
  }
}

void WriteJson(const fs::path& path,
               const Args& args,
               const std::vector<MethodStats>& rows,
               const double preload_ms,
               const std::uint64_t unique_query_blocks) {
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("failed to write " + path.string());
  }
  out << std::setprecision(12);
  out << "{\n"
      << "  \"runner\": \"tight_inclusion_dense_group_three_method_benchmark\",\n"
      << "  \"dataset_root\": \"" << JsonEscape(args.dataset_root.generic_string()) << "\",\n"
      << "  \"learned_schedule\": \"" << JsonEscape(args.learned_schedule.generic_string()) << "\",\n"
      << "  \"random_schedule\": \"" << JsonEscape(args.random_schedule.generic_string()) << "\",\n"
      << "  \"unique_query_blocks\": " << unique_query_blocks << ",\n"
      << "  \"preload_ms\": " << preload_ms << ",\n"
      << "  \"parameters\": {\"ms\": " << args.ms << ", \"tolerance\": " << args.tolerance
      << ", \"t_max\": " << args.t_max << ", \"max_itr\": " << args.max_itr << "},\n"
      << "  \"methods\": [\n";
  for (std::size_t i = 0; i < rows.size(); ++i) {
    const MethodStats& stats = rows[i];
    out << "    {\n"
        << "      \"method\": \"" << JsonEscape(stats.method) << "\",\n"
        << "      \"group_count\": " << stats.group_count << ",\n"
        << "      \"candidate_count\": " << stats.candidate_count << ",\n"
        << "      \"positive_group_count\": " << stats.positive_group_count << ",\n"
        << "      \"exact_calls\": " << stats.exact_calls << ",\n"
        << "      \"skipped_exact_calls\": " << stats.skipped_exact_calls << ",\n"
        << "      \"exact_call_reduction\": " << CallReduction(stats) << ",\n"
        << "      \"tp\": " << stats.tp << ",\n"
        << "      \"tn\": " << stats.tn << ",\n"
        << "      \"fp\": " << stats.fp << ",\n"
        << "      \"fn\": " << stats.fn << ",\n"
        << "      \"recall\": " << Recall(stats) << ",\n"
        << "      \"precision\": " << Precision(stats) << ",\n"
        << "      \"first_positive_rank_mean\": " << FirstRank(stats) << ",\n"
        << "      \"exact_ms\": " << stats.exact_ms << ",\n"
        << "      \"wall_ms\": " << stats.wall_ms << ",\n"
        << "      \"wall_ms_with_shared_preload\": " << stats.wall_ms + preload_ms << "\n"
        << "    }" << (i + 1 == rows.size() ? "\n" : ",\n");
  }
  out << "  ]\n}\n";
}

void WriteMarkdown(const fs::path& path,
                   const Args& args,
                   const std::vector<MethodStats>& rows,
                   const double preload_ms,
                   const std::uint64_t unique_query_blocks) {
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("failed to write " + path.string());
  }
  out << std::setprecision(6);
  out << "# Large Selected-real Tight-Inclusion Dense Group Benchmark\n\n"
      << "## Scope\n\n"
      << "- Exact payload: native `ticcd::vertexFaceCCD` / `ticcd::edgeEdgeCCD` with the same parameters as the Tight-Inclusion baseline.\n"
      << "- Compared methods: `NoProposal+TI`, `Random+TI`, and `RTSTPFExact+TI`.\n"
      << "- STPF only changes the order of exact work items. It does not delete candidates or output collision truth.\n"
      << "- Negative or uncertain groups are evaluated to exhaustion; positive groups stop only after a certified TI hit.\n"
      << "- Shared preload reads the selected CSV query blocks once and is reported separately so the table measures native scheduling and exact certification.\n\n"
      << "## Inputs\n\n"
      << "- Dataset root: `" << args.dataset_root.generic_string() << "`\n"
      << "- Learned schedule: `" << args.learned_schedule.generic_string() << "`\n"
      << "- Random schedule: `" << args.random_schedule.generic_string() << "`\n"
      << "- Unique query blocks preloaded: `" << unique_query_blocks << "`\n"
      << "- Shared preload time: `" << preload_ms << " ms`\n"
      << "- TI parameters: `ms=" << args.ms << ", tolerance=" << args.tolerance
      << ", t_max=" << args.t_max << ", max_itr=" << args.max_itr << "`\n\n"
      << "## Results\n\n"
      << "| Method | Groups | Candidates | Positive groups | Exact calls | Skipped calls | Call reduction | TP | TN | FP | FN | Recall | Precision | First positive rank | Exact ms | Wall ms | Wall + shared preload ms |\n"
      << "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n";
  for (const MethodStats& stats : rows) {
    out << "| " << stats.method << " | " << stats.group_count << " | " << stats.candidate_count
        << " | " << stats.positive_group_count << " | " << stats.exact_calls << " | "
        << stats.skipped_exact_calls << " | " << CallReduction(stats) * 100.0 << "% | "
        << stats.tp << " | " << stats.tn << " | " << stats.fp << " | " << stats.fn
        << " | " << Recall(stats) << " | " << Precision(stats) << " | " << FirstRank(stats)
        << " | " << stats.exact_ms << " | " << stats.wall_ms << " | "
        << stats.wall_ms + preload_ms << " |\n";
  }
  out << "\n## Interpretation\n\n"
      << "- `FN=0` is the required correctness condition. Any nonzero FP is conservative and is reported.\n"
      << "- `NoProposal+TI` is the all-candidate exact baseline.\n"
      << "- `Random+TI` tests whether early-stop alone explains the gain.\n"
      << "- `RTSTPFExact+TI` is the learned route: the same candidates are ordered by the learned STPF policy before TI certification.\n";
}

void ValidateCandidateSets(const std::vector<ScheduledCandidate>& learned,
                           const std::vector<ScheduledCandidate>& random_schedule) {
  if (learned.size() != random_schedule.size()) {
    throw std::runtime_error("learned and random schedules have different candidate counts");
  }
  std::multiset<std::string> lhs;
  std::multiset<std::string> rhs;
  for (const ScheduledCandidate& candidate : learned) {
    lhs.insert(std::to_string(candidate.group_id) + "|" + CandidateKey(candidate));
  }
  for (const ScheduledCandidate& candidate : random_schedule) {
    rhs.insert(std::to_string(candidate.group_id) + "|" + CandidateKey(candidate));
  }
  if (lhs != rhs) {
    throw std::runtime_error("learned and random schedules do not contain the same grouped candidates");
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Args args = ParseArgs(argc, argv);
    std::vector<ScheduledCandidate> learned = LoadSchedule(args.learned_schedule);
    std::vector<ScheduledCandidate> random_schedule = LoadSchedule(args.random_schedule);
    ValidateCandidateSets(learned, random_schedule);

    double preload_ms = 0.0;
    const std::unordered_map<std::string, QueryBlock> blocks =
        PreloadQueryBlocks(args.dataset_root, learned, random_schedule, &preload_ms);

    std::vector<ScheduledCandidate> no_proposal = learned;
    std::vector<MethodStats> rows;
    rows.push_back(EvaluateMethod(args, no_proposal, blocks, "NoProposal+TI", false));
    rows.push_back(EvaluateMethod(args, random_schedule, blocks, "Random+TI", true));
    rows.push_back(EvaluateMethod(args, learned, blocks, "RTSTPFExact+TI", true));

    WriteCsv(args.output_csv, rows);
    WriteJson(args.output_json, args, rows, preload_ms, static_cast<std::uint64_t>(blocks.size()));
    WriteMarkdown(args.output_md, args, rows, preload_ms, static_cast<std::uint64_t>(blocks.size()));
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << "\n";
    return 1;
  }
  return 0;
}
