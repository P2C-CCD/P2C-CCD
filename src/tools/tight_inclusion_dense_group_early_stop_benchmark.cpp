#include <tight_inclusion/ccd.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct Args {
  fs::path dataset_root;
  fs::path schedule_csv;
  fs::path output_json;
  fs::path output_md;
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

struct Stats {
  std::uint64_t group_count = 0;
  std::uint64_t candidate_count = 0;
  std::uint64_t positive_group_count = 0;
  std::uint64_t no_proposal_exact_calls = 0;
  std::uint64_t learned_exact_calls = 0;
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
    } else if (key == "--schedule") {
      args.schedule_csv = need_value(key);
    } else if (key == "--output-json") {
      args.output_json = need_value(key);
    } else if (key == "--output-md") {
      args.output_md = need_value(key);
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
  if (args.dataset_root.empty() || args.schedule_csv.empty() || args.output_json.empty() ||
      args.output_md.empty()) {
    throw std::runtime_error("--dataset-root, --schedule, --output-json and --output-md are required");
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

QueryBlock ReadQueryBlock(const fs::path& dataset_root, const ScheduledCandidate& candidate) {
  const fs::path path = dataset_root / fs::path(candidate.csv_path);
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("failed to open " + path.string());
  }
  const std::uint64_t first_line = candidate.query_index * 8U;
  std::string line;
  for (std::uint64_t i = 0; i < first_line; ++i) {
    if (!std::getline(in, line)) {
      throw std::runtime_error("query index out of range in " + path.string());
    }
  }
  QueryBlock block;
  bool first_truth = false;
  for (std::size_t row = 0; row < 8; ++row) {
    if (!std::getline(in, line)) {
      throw std::runtime_error("truncated query in " + path.string());
    }
    bool truth = false;
    block.vertices[row] = ParseVertex(line, &truth);
    if (row == 0) {
      first_truth = truth;
    } else if (truth != first_truth) {
      throw std::runtime_error("truth changes inside query block in " + path.string());
    }
  }
  block.truth = first_truth;
  return block;
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

Stats RunBenchmark(const Args& args, std::vector<ScheduledCandidate> schedule) {
  std::stable_sort(schedule.begin(), schedule.end(), [](const auto& lhs, const auto& rhs) {
    if (lhs.group_id != rhs.group_id) {
      return lhs.group_id < rhs.group_id;
    }
    return lhs.score > rhs.score;
  });
  Stats stats;
  stats.candidate_count = schedule.size();
  stats.no_proposal_exact_calls = schedule.size();

  const auto wall_begin = std::chrono::steady_clock::now();
  std::size_t group_begin = 0;
  while (group_begin < schedule.size()) {
    std::size_t group_end = group_begin + 1;
    while (group_end < schedule.size() &&
           schedule[group_end].group_id == schedule[group_begin].group_id) {
      ++group_end;
    }
    ++stats.group_count;
    bool truth = false;
    bool predicted = false;
    std::vector<QueryBlock> blocks;
    blocks.reserve(group_end - group_begin);
    for (std::size_t pos = group_begin; pos < group_end; ++pos) {
      blocks.push_back(ReadQueryBlock(args.dataset_root, schedule[pos]));
      truth = truth || blocks.back().truth;
    }
    std::uint64_t rank = 0;
    for (std::size_t pos = group_begin; pos < group_end; ++pos) {
      ++rank;
      ++stats.learned_exact_calls;
      const auto exact_begin = std::chrono::steady_clock::now();
      const bool result = RunTightInclusionQuery(
          schedule[pos].kind,
          blocks[pos - group_begin].vertices,
          args);
      const auto exact_end = std::chrono::steady_clock::now();
      stats.exact_ms += std::chrono::duration<double, std::milli>(exact_end - exact_begin).count();
      if (result) {
        predicted = true;
        stats.first_positive_rank_sum += static_cast<double>(rank);
        break;
      }
    }
    if (truth) {
      ++stats.positive_group_count;
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
  return stats;
}

void WriteJson(const fs::path& path, const Stats& stats, const Args& args) {
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("failed to write " + path.string());
  }
  const double call_reduction =
      1.0 - static_cast<double>(stats.learned_exact_calls) /
                static_cast<double>(std::max<std::uint64_t>(1, stats.no_proposal_exact_calls));
  const double recall =
      static_cast<double>(stats.tp) / static_cast<double>(std::max<std::uint64_t>(1, stats.tp + stats.fn));
  const double first_rank =
      stats.positive_group_count == 0 ? 0.0
                                      : stats.first_positive_rank_sum /
                                            static_cast<double>(stats.positive_group_count);
  out << "{\n"
      << "  \"method\": \"RTSTPFExact+TI-native-dense-group\",\n"
      << "  \"schedule\": \"" << JsonEscape(args.schedule_csv.generic_string()) << "\",\n"
      << "  \"group_count\": " << stats.group_count << ",\n"
      << "  \"candidate_count\": " << stats.candidate_count << ",\n"
      << "  \"positive_group_count\": " << stats.positive_group_count << ",\n"
      << "  \"no_proposal_exact_calls\": " << stats.no_proposal_exact_calls << ",\n"
      << "  \"learned_exact_calls\": " << stats.learned_exact_calls << ",\n"
      << "  \"exact_call_reduction\": " << call_reduction << ",\n"
      << "  \"tp\": " << stats.tp << ",\n"
      << "  \"tn\": " << stats.tn << ",\n"
      << "  \"fp\": " << stats.fp << ",\n"
      << "  \"fn\": " << stats.fn << ",\n"
      << "  \"recall\": " << recall << ",\n"
      << "  \"first_positive_rank_mean\": " << first_rank << ",\n"
      << "  \"exact_ms\": " << stats.exact_ms << ",\n"
      << "  \"wall_ms\": " << stats.wall_ms << "\n"
      << "}\n";
}

void WriteMarkdown(const fs::path& path, const Stats& stats, const Args& args) {
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("failed to write " + path.string());
  }
  const double call_reduction =
      1.0 - static_cast<double>(stats.learned_exact_calls) /
                static_cast<double>(std::max<std::uint64_t>(1, stats.no_proposal_exact_calls));
  const double recall =
      static_cast<double>(stats.tp) / static_cast<double>(std::max<std::uint64_t>(1, stats.tp + stats.fn));
  const double first_rank =
      stats.positive_group_count == 0 ? 0.0
                                      : stats.first_positive_rank_sum /
                                            static_cast<double>(stats.positive_group_count);
  out << "# Tight-Inclusion Native Dense Group Early-stop Benchmark\n\n"
      << "- Exact payload: real `ticcd::vertexFaceCCD` / `ticcd::edgeEdgeCCD`\n"
      << "- Schedule: `" << args.schedule_csv.generic_string() << "`\n"
      << "- Parameters: `ms=" << args.ms << ", tolerance=" << args.tolerance
      << ", t_max=" << args.t_max << ", max_itr=" << args.max_itr << "`\n\n"
      << "| Groups | Candidates | Positive groups | NoProposal exact calls | RTSTPFExact+TI exact calls | Exact-call reduction | TP | TN | FP | FN | Recall | First-positive rank mean | Exact ms | Wall ms |\n"
      << "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n"
      << "| " << stats.group_count << " | " << stats.candidate_count << " | "
      << stats.positive_group_count << " | " << stats.no_proposal_exact_calls << " | "
      << stats.learned_exact_calls << " | " << call_reduction * 100.0 << "% | " << stats.tp
      << " | " << stats.tn << " | " << stats.fp << " | " << stats.fn << " | " << recall
      << " | " << first_rank << " | " << stats.exact_ms << " | " << stats.wall_ms << " |\n\n"
      << "This table replaces the previous proxy-oracle dense group exact payload for this smoke workload. "
      << "It still uses the same group schedule interface, but every evaluated primitive candidate is certified by Tight-Inclusion.\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Args args = ParseArgs(argc, argv);
    const std::vector<ScheduledCandidate> schedule = LoadSchedule(args.schedule_csv);
    const Stats stats = RunBenchmark(args, schedule);
    WriteJson(args.output_json, stats, args);
    WriteMarkdown(args.output_md, stats, args);
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << "\n";
    return 1;
  }
  return 0;
}
