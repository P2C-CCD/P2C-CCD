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
  fs::path manifest;
  fs::path dataset_root;
  fs::path output_jsonl;
  fs::path output_md;
  fs::path selection;
  std::string selection_mode = "include";
  std::string method = "TightInclusion";
  std::string split = "heldout_test";
  std::set<std::string> cases;
  std::set<std::string> kinds;
  std::uint64_t max_queries = 0;
  double ms = 0.0;
  double tolerance = 1.0e-6;
  double t_max = 1.0;
  long max_itr = 1000000;
};

struct FileSpec {
  std::string case_name;
  std::string kind;
  std::string csv_path;
  std::string split;
  std::uint64_t query_count = 0;
};

struct Stats {
  std::string method;
  std::string split;
  std::string case_name;
  std::string kind;
  std::uint64_t file_count = 0;
  std::uint64_t query_count = 0;
  std::uint64_t exact_calls = 0;
  std::uint64_t skipped_exact_calls = 0;
  std::uint64_t positive_count = 0;
  std::uint64_t negative_count = 0;
  std::uint64_t tp = 0;
  std::uint64_t tn = 0;
  std::uint64_t fp = 0;
  std::uint64_t fn = 0;
  double exact_us = 0.0;
  double wall_us = 0.0;
  std::vector<double> exact_latency_us;
  std::vector<double> wall_latency_us;
};

std::string ReadTextFile(const fs::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("failed to open " + path.string());
  }
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

std::string NormalizeSlashes(std::string value) {
  std::replace(value.begin(), value.end(), '\\', '/');
  return value;
}

std::vector<std::string> SplitList(const std::string& value) {
  std::vector<std::string> out;
  std::string token;
  std::istringstream stream(value);
  while (std::getline(stream, token, ',')) {
    token.erase(token.begin(), std::find_if(token.begin(), token.end(), [](unsigned char ch) {
      return !std::isspace(ch);
    }));
    token.erase(std::find_if(token.rbegin(), token.rend(), [](unsigned char ch) {
      return !std::isspace(ch);
    }).base(), token.end());
    if (!token.empty()) {
      out.push_back(token);
    }
  }
  return out;
}

std::string JsonStringValue(const std::string& object, const std::string& key) {
  const std::string needle = "\"" + key + "\"";
  std::size_t pos = object.find(needle);
  if (pos == std::string::npos) {
    return "";
  }
  pos = object.find(':', pos);
  if (pos == std::string::npos) {
    return "";
  }
  pos = object.find('"', pos);
  if (pos == std::string::npos) {
    return "";
  }
  ++pos;
  std::string value;
  bool escape = false;
  for (; pos < object.size(); ++pos) {
    const char ch = object[pos];
    if (escape) {
      value.push_back(ch);
      escape = false;
      continue;
    }
    if (ch == '\\') {
      escape = true;
      continue;
    }
    if (ch == '"') {
      break;
    }
    value.push_back(ch);
  }
  return value;
}

std::uint64_t JsonUIntValue(const std::string& object, const std::string& key) {
  const std::string needle = "\"" + key + "\"";
  std::size_t pos = object.find(needle);
  if (pos == std::string::npos) {
    return 0;
  }
  pos = object.find(':', pos);
  if (pos == std::string::npos) {
    return 0;
  }
  ++pos;
  while (pos < object.size() && std::isspace(static_cast<unsigned char>(object[pos]))) {
    ++pos;
  }
  std::size_t end = pos;
  while (end < object.size() && std::isdigit(static_cast<unsigned char>(object[end]))) {
    ++end;
  }
  if (end == pos) {
    return 0;
  }
  return static_cast<std::uint64_t>(std::stoull(object.substr(pos, end - pos)));
}

std::vector<FileSpec> ParseManifestFiles(const std::string& text) {
  std::vector<FileSpec> files;
  std::size_t files_pos = text.find("\"files\"");
  if (files_pos == std::string::npos) {
    throw std::runtime_error("manifest does not contain a files array");
  }
  std::size_t array_pos = text.find('[', files_pos);
  if (array_pos == std::string::npos) {
    throw std::runtime_error("manifest files field is not an array");
  }
  std::size_t pos = array_pos + 1;
  while (true) {
    std::size_t begin = text.find('{', pos);
    if (begin == std::string::npos) {
      break;
    }
    std::size_t end = text.find('}', begin);
    if (end == std::string::npos) {
      throw std::runtime_error("unterminated file object in manifest");
    }
    const std::string object = text.substr(begin, end - begin + 1);
    FileSpec spec;
    spec.case_name = JsonStringValue(object, "case");
    spec.kind = JsonStringValue(object, "kind");
    spec.csv_path = NormalizeSlashes(JsonStringValue(object, "csv_path"));
    spec.split = JsonStringValue(object, "split");
    spec.query_count = JsonUIntValue(object, "query_count");
    if (!spec.case_name.empty() && !spec.kind.empty() && !spec.csv_path.empty()) {
      files.push_back(spec);
    }
    pos = end + 1;
  }
  return files;
}

std::string ParseManifestDatasetRoot(const std::string& text) {
  return NormalizeSlashes(JsonStringValue(text, "dataset_root"));
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
    if (key == "--manifest") {
      args.manifest = need_value(key);
    } else if (key == "--dataset-root") {
      args.dataset_root = need_value(key);
    } else if (key == "--output-jsonl") {
      args.output_jsonl = need_value(key);
    } else if (key == "--output-md") {
      args.output_md = need_value(key);
    } else if (key == "--selection") {
      args.selection = need_value(key);
    } else if (key == "--selection-mode") {
      args.selection_mode = need_value(key);
      if (args.selection_mode != "include" && args.selection_mode != "exclude") {
        throw std::runtime_error("--selection-mode must be include or exclude");
      }
    } else if (key == "--method") {
      args.method = need_value(key);
    } else if (key == "--split") {
      args.split = need_value(key);
    } else if (key == "--case" || key == "--cases") {
      for (const std::string& item : SplitList(need_value(key))) {
        args.cases.insert(item);
      }
    } else if (key == "--kind" || key == "--kinds") {
      for (const std::string& item : SplitList(need_value(key))) {
        args.kinds.insert(item);
      }
    } else if (key == "--max-queries") {
      args.max_queries = static_cast<std::uint64_t>(std::stoull(need_value(key)));
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
  if (args.manifest.empty()) {
    throw std::runtime_error("--manifest is required");
  }
  if (args.output_jsonl.empty()) {
    throw std::runtime_error("--output-jsonl is required");
  }
  if (args.output_md.empty()) {
    throw std::runtime_error("--output-md is required");
  }
  return args;
}

std::unordered_map<std::string, std::unordered_set<std::uint64_t>> LoadSelection(const fs::path& path) {
  std::unordered_map<std::string, std::unordered_set<std::uint64_t>> selected;
  if (path.empty()) {
    return selected;
  }
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("failed to open selection file " + path.string());
  }
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty()) {
      continue;
    }
    const std::size_t comma = line.rfind(',');
    if (comma == std::string::npos) {
      throw std::runtime_error("selection line must be csv_path,query_index: " + line);
    }
    const std::string csv_path = NormalizeSlashes(line.substr(0, comma));
    const std::uint64_t query_index = static_cast<std::uint64_t>(std::stoull(line.substr(comma + 1)));
    selected[csv_path].insert(query_index);
  }
  return selected;
}

std::vector<std::string> SplitCSVLine(const std::string& line) {
  std::vector<std::string> parts;
  parts.reserve(7);
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

bool ParseTruthFast(const std::string& line) {
  const std::size_t comma = line.rfind(',');
  if (comma == std::string::npos || comma + 1 >= line.size()) {
    throw std::runtime_error("Tight-Inclusion CSV row must contain a truth column");
  }
  std::string token = line.substr(comma + 1);
  while (!token.empty() &&
         (token.back() == '\r' || token.back() == ' ' || token.back() == '\t')) {
    token.pop_back();
  }
  if (token != "0" && token != "1") {
    throw std::runtime_error("truth column must be 0 or 1");
  }
  return token == "1";
}

std::string StatKey(const std::string& method,
                    const std::string& split,
                    const std::string& case_name,
                    const std::string& kind) {
  return method + "|" + split + "|" + case_name + "|" + kind;
}

void UpdateCorrectness(Stats* stats, bool expected, bool result) {
  stats->positive_count += expected ? 1U : 0U;
  stats->negative_count += expected ? 0U : 1U;
  if (expected && result) {
    ++stats->tp;
  } else if (!expected && !result) {
    ++stats->tn;
  } else if (!expected && result) {
    ++stats->fp;
  } else {
    ++stats->fn;
  }
}

double Percentile(std::vector<double> values, double percentile) {
  if (values.empty()) {
    return 0.0;
  }
  std::sort(values.begin(), values.end());
  const double rank = percentile * static_cast<double>(values.size() - 1);
  const auto lower = static_cast<std::size_t>(std::floor(rank));
  const auto upper = static_cast<std::size_t>(std::ceil(rank));
  if (lower == upper) {
    return values[lower];
  }
  const double weight = rank - static_cast<double>(lower);
  return values[lower] * (1.0 - weight) + values[upper] * weight;
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

void BenchmarkCSV(const fs::path& dataset_root,
                  const FileSpec& file,
                  const Args& args,
                  const std::unordered_map<std::string, std::unordered_set<std::uint64_t>>& selection,
                  bool selection_enabled,
                  Stats* stats,
                  std::uint64_t* global_queries) {
  const fs::path csv_path = dataset_root / fs::path(file.csv_path);
  std::ifstream in(csv_path);
  if (!in) {
    throw std::runtime_error("failed to open CSV " + csv_path.string());
  }
  ++stats->file_count;
  std::array<std::string, 8> lines;
  std::uint64_t query_index = 0;
  const auto file_selection_it = selection.find(file.csv_path);
  const std::unordered_set<std::uint64_t>* selected_indices =
      file_selection_it == selection.end() ? nullptr : &file_selection_it->second;
  const auto wall_begin = std::chrono::steady_clock::now();
  while (true) {
    bool have_block = true;
    for (int row = 0; row < 8; ++row) {
      if (!std::getline(in, lines[row])) {
        have_block = false;
        break;
      }
    }
    if (!have_block) {
      break;
    }
    if (args.max_queries != 0 && *global_queries >= args.max_queries) {
      break;
    }
    const auto query_begin = std::chrono::steady_clock::now();
    bool selected = true;
    if (selection_enabled) {
      const bool in_selection =
          selected_indices != nullptr && selected_indices->find(query_index) != selected_indices->end();
      selected = args.selection_mode == "include" ? in_selection : !in_selection;
    }
    bool expected = false;
    std::array<ticcd::Vector3, 8> vertices;
    if (selected) {
      bool first_truth = false;
      for (int row = 0; row < 8; ++row) {
        bool truth = false;
        vertices[row] = ParseVertex(lines[row], &truth);
        if (row == 0) {
          first_truth = truth;
        } else if (truth != first_truth) {
          throw std::runtime_error("truth label changes inside query block in " + csv_path.string());
        }
      }
      expected = first_truth;
    } else {
      expected = ParseTruthFast(lines[0]);
    }
    bool result = false;
    if (selected) {
      const auto exact_begin = std::chrono::steady_clock::now();
      result = RunTightInclusionQuery(file.kind, vertices, args);
      const auto exact_end = std::chrono::steady_clock::now();
      const double exact_query_us = std::chrono::duration<double, std::micro>(exact_end - exact_begin).count();
      stats->exact_us += exact_query_us;
      stats->exact_latency_us.push_back(exact_query_us);
      ++stats->exact_calls;
    } else {
      ++stats->skipped_exact_calls;
    }
    ++stats->query_count;
    ++(*global_queries);
    UpdateCorrectness(stats, expected, result);
    const auto query_end = std::chrono::steady_clock::now();
    stats->wall_latency_us.push_back(std::chrono::duration<double, std::micro>(query_end - query_begin).count());
    ++query_index;
  }
  const auto wall_end = std::chrono::steady_clock::now();
  stats->wall_us += std::chrono::duration<double, std::micro>(wall_end - wall_begin).count();
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

void WriteJsonRow(std::ostream& out, const Stats& stats, const Args& args) {
  const double recall = static_cast<double>(stats.tp) / std::max<std::uint64_t>(1, stats.tp + stats.fn);
  const double precision = static_cast<double>(stats.tp) / std::max<std::uint64_t>(1, stats.tp + stats.fp);
  const double exact_reduction = 1.0 - static_cast<double>(stats.exact_calls) /
                                           static_cast<double>(std::max<std::uint64_t>(1, stats.query_count));
  const double wall_seconds = stats.wall_us / 1000000.0;
  out << "{"
      << "\"method\":\"" << JsonEscape(stats.method) << "\","
      << "\"split\":\"" << JsonEscape(stats.split) << "\","
      << "\"case\":\"" << JsonEscape(stats.case_name) << "\","
      << "\"kind\":\"" << JsonEscape(stats.kind) << "\","
      << "\"file_count\":" << stats.file_count << ","
      << "\"query_count\":" << stats.query_count << ","
      << "\"exact_calls\":" << stats.exact_calls << ","
      << "\"skipped_exact_calls\":" << stats.skipped_exact_calls << ","
      << "\"exact_call_reduction\":" << exact_reduction << ","
      << "\"positive_count\":" << stats.positive_count << ","
      << "\"negative_count\":" << stats.negative_count << ","
      << "\"tp\":" << stats.tp << ","
      << "\"tn\":" << stats.tn << ","
      << "\"fp\":" << stats.fp << ","
      << "\"fn\":" << stats.fn << ","
      << "\"recall\":" << recall << ","
      << "\"precision\":" << precision << ","
      << "\"exact_us\":" << stats.exact_us << ","
      << "\"wall_us\":" << stats.wall_us << ","
      << "\"queries_per_second\":" << (static_cast<double>(stats.query_count) / std::max(1.0e-12, wall_seconds)) << ","
      << "\"avg_exact_us_per_exact_call\":" << (stats.exact_us / std::max<std::uint64_t>(1, stats.exact_calls)) << ","
      << "\"avg_wall_us_per_query\":" << (stats.wall_us / std::max<std::uint64_t>(1, stats.query_count)) << ","
      << "\"wall_p50_us\":" << Percentile(stats.wall_latency_us, 0.50) << ","
      << "\"wall_p90_us\":" << Percentile(stats.wall_latency_us, 0.90) << ","
      << "\"wall_p99_us\":" << Percentile(stats.wall_latency_us, 0.99) << ","
      << "\"exact_p50_us\":" << Percentile(stats.exact_latency_us, 0.50) << ","
      << "\"exact_p90_us\":" << Percentile(stats.exact_latency_us, 0.90) << ","
      << "\"exact_p99_us\":" << Percentile(stats.exact_latency_us, 0.99) << ","
      << "\"ms\":" << args.ms << ","
      << "\"tolerance\":" << args.tolerance << ","
      << "\"t_max\":" << args.t_max << ","
      << "\"max_itr\":" << args.max_itr
      << "}\n";
}

void WriteMarkdown(const fs::path& path, const std::vector<Stats>& rows, const Args& args) {
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("failed to write markdown " + path.string());
  }
  out << "# Tight-Inclusion Full-Query C++ Benchmark\n\n";
  out << "- Method: `" << args.method << "`\n";
  out << "- Split: `" << args.split << "`\n";
  out << "- Parameters: `ms=" << args.ms << ", tolerance=" << args.tolerance
      << ", t_max=" << args.t_max << ", max_itr=" << args.max_itr
      << ", no_zero_toi=false, root=BREADTH_FIRST_SEARCH`\n\n";
  out << "| Case | Kind | Queries | Exact calls | Reduction | TP | TN | FP | FN | Recall | Exact ms | Wall ms | QPS | Avg us/query | p50 us | p90 us | p99 us |\n";
  out << "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n";
  for (const Stats& row : rows) {
    const double recall = static_cast<double>(row.tp) / std::max<std::uint64_t>(1, row.tp + row.fn);
    const double exact_reduction = 1.0 - static_cast<double>(row.exact_calls) /
                                             static_cast<double>(std::max<std::uint64_t>(1, row.query_count));
    const double wall_seconds = row.wall_us / 1000000.0;
    out << "| `" << row.case_name << "` | `" << row.kind << "` | `" << row.query_count << "` | `"
        << row.exact_calls << "` | `" << (100.0 * exact_reduction) << "%` | `"
        << row.tp << "` | `" << row.tn << "` | `" << row.fp << "` | `" << row.fn << "` | `"
        << recall << "` | `" << (row.exact_us / 1000.0) << "` | `" << (row.wall_us / 1000.0)
        << "` | `" << (static_cast<double>(row.query_count) / std::max(1.0e-12, wall_seconds))
        << "` | `" << (row.wall_us / std::max<std::uint64_t>(1, row.query_count))
        << "` | `" << Percentile(row.wall_latency_us, 0.50)
        << "` | `" << Percentile(row.wall_latency_us, 0.90)
        << "` | `" << Percentile(row.wall_latency_us, 0.99) << "` |\n";
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    Args args = ParseArgs(argc, argv);
    const std::string manifest_text = ReadTextFile(args.manifest);
    if (args.dataset_root.empty()) {
      args.dataset_root = fs::path(ParseManifestDatasetRoot(manifest_text));
    }
    if (args.dataset_root.empty()) {
      throw std::runtime_error("dataset root missing; pass --dataset-root or include it in manifest");
    }
    const std::vector<FileSpec> files = ParseManifestFiles(manifest_text);
    const std::unordered_map<std::string, std::unordered_set<std::uint64_t>> selection =
        LoadSelection(args.selection);
    const bool selection_enabled = !args.selection.empty();
    std::unordered_map<std::string, Stats> grouped;
    std::uint64_t global_queries = 0;
    for (const FileSpec& file : files) {
      if (args.split != "full_stress" && file.split != args.split) {
        continue;
      }
      if (!args.cases.empty() && args.cases.find(file.case_name) == args.cases.end()) {
        continue;
      }
      if (!args.kinds.empty() && args.kinds.find(file.kind) == args.kinds.end()) {
        continue;
      }
      if (args.max_queries != 0 && global_queries >= args.max_queries) {
        break;
      }
      const std::string key = StatKey(args.method, args.split, file.case_name, file.kind);
      Stats& stats = grouped[key];
      stats.method = args.method;
      stats.split = args.split;
      stats.case_name = file.case_name;
      stats.kind = file.kind;
      BenchmarkCSV(args.dataset_root, file, args, selection, selection_enabled, &stats, &global_queries);
    }
    std::vector<Stats> rows;
    rows.reserve(grouped.size());
    for (const auto& item : grouped) {
      rows.push_back(item.second);
    }
    std::sort(rows.begin(), rows.end(), [](const Stats& a, const Stats& b) {
      if (a.case_name != b.case_name) {
        return a.case_name < b.case_name;
      }
      return a.kind < b.kind;
    });
    fs::create_directories(args.output_jsonl.parent_path());
    std::ofstream jsonl(args.output_jsonl);
    if (!jsonl) {
      throw std::runtime_error("failed to write JSONL " + args.output_jsonl.string());
    }
    for (const Stats& row : rows) {
      WriteJsonRow(jsonl, row, args);
    }
    fs::create_directories(args.output_md.parent_path());
    WriteMarkdown(args.output_md, rows, args);
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "tight_inclusion_full_query_benchmark failed: " << ex.what() << "\n";
    return 1;
  }
}
