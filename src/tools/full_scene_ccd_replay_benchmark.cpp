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
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct Args {
  fs::path manifest;
  fs::path schedule;
  fs::path output_jsonl;
  std::string method = "NaturalAnyHit";
  double ms = 0.0;
  double tolerance = 1.0e-6;
  double t_max = 1.0;
  long max_itr = 1000000;
  std::uint64_t max_files = 0;
  std::uint64_t max_queries_per_file = 0;
};

struct FileSpec {
  std::string scene;
  std::string kind;
  std::string csv_path;
  std::string bool_json;
  std::string frame0;
  std::string frame1;
  std::string split;
  int timestep = 0;
  std::uint64_t query_count = 0;
  std::uint64_t positive_count = 0;
};

struct Query {
  std::array<ticcd::Vector3, 8> vertices;
  bool label = false;
};

struct Row {
  std::string method;
  std::string timing_scope;
  FileSpec file;
  std::uint64_t query_count = 0;
  std::uint64_t positive_count = 0;
  std::uint64_t exact_calls = 0;
  std::uint64_t skipped_candidates = 0;
  std::uint64_t executed_label_positives = 0;
  std::uint64_t executed_label_negatives = 0;
  std::uint64_t exact_positive_count = 0;
  std::uint64_t exact_false_positive_count = 0;
  std::uint64_t exact_false_negative_count = 0;
  bool expected_scene_hit = false;
  bool detected_scene_hit = false;
  std::uint64_t scene_tp = 0;
  std::uint64_t scene_tn = 0;
  std::uint64_t scene_fp = 0;
  std::uint64_t scene_fn = 0;
  double load_us = 0.0;
  double schedule_us = 0.0;
  double exact_us = 0.0;
  double detection_wall_us = 0.0;
  double total_wall_us = 0.0;
  double first_hit_toi = std::numeric_limits<double>::quiet_NaN();
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

std::string Trim(std::string value) {
  value.erase(value.begin(), std::find_if(value.begin(), value.end(), [](unsigned char ch) {
    return !std::isspace(ch);
  }));
  value.erase(std::find_if(value.rbegin(), value.rend(), [](unsigned char ch) {
    return !std::isspace(ch);
  }).base(), value.end());
  return value;
}

std::vector<std::string> SplitCSVLine(const std::string& line) {
  std::vector<std::string> out;
  std::string token;
  std::istringstream stream(line);
  while (std::getline(stream, token, ',')) {
    out.push_back(Trim(token));
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
  const std::size_t files_pos = text.find("\"files\"");
  if (files_pos == std::string::npos) {
    throw std::runtime_error("manifest does not contain files");
  }
  const std::size_t array_pos = text.find('[', files_pos);
  if (array_pos == std::string::npos) {
    throw std::runtime_error("manifest files is not an array");
  }
  std::size_t pos = array_pos + 1;
  while (true) {
    const std::size_t begin = text.find('{', pos);
    if (begin == std::string::npos) {
      break;
    }
    const std::size_t end = text.find('}', begin);
    if (end == std::string::npos) {
      throw std::runtime_error("unterminated file object in manifest");
    }
    const std::string object = text.substr(begin, end - begin + 1);
    FileSpec spec;
    spec.scene = JsonStringValue(object, "scene");
    spec.kind = JsonStringValue(object, "kind");
    spec.csv_path = NormalizeSlashes(JsonStringValue(object, "csv_path"));
    spec.bool_json = NormalizeSlashes(JsonStringValue(object, "bool_json"));
    spec.frame0 = NormalizeSlashes(JsonStringValue(object, "frame0"));
    spec.frame1 = NormalizeSlashes(JsonStringValue(object, "frame1"));
    spec.split = JsonStringValue(object, "split");
    spec.timestep = static_cast<int>(JsonUIntValue(object, "timestep"));
    spec.query_count = JsonUIntValue(object, "query_count");
    spec.positive_count = JsonUIntValue(object, "positive_count");
    if (!spec.scene.empty() && !spec.kind.empty() && !spec.csv_path.empty()) {
      files.push_back(spec);
    }
    pos = end + 1;
  }
  return files;
}

std::vector<bool> ParseBoolJsonArray(const fs::path& path) {
  const std::string text = ReadTextFile(path);
  std::vector<bool> out;
  for (std::size_t pos = 0; pos < text.size();) {
    if (text.compare(pos, 4, "true") == 0) {
      out.push_back(true);
      pos += 4;
    } else if (text.compare(pos, 5, "false") == 0) {
      out.push_back(false);
      pos += 5;
    } else if (text[pos] == '0' || text[pos] == '1') {
      out.push_back(text[pos] == '1');
      ++pos;
    } else {
      ++pos;
    }
  }
  return out;
}

ticcd::Vector3 ParseRationalVertex(const std::string& line, bool* embedded_truth, bool* has_truth) {
  const std::vector<std::string> parts = SplitCSVLine(line);
  if (parts.size() != 6 && parts.size() != 7) {
    throw std::runtime_error("query CSV row must contain 6 rational columns plus optional truth");
  }
  const long double x = std::stold(parts[0]) / std::stold(parts[1]);
  const long double y = std::stold(parts[2]) / std::stold(parts[3]);
  const long double z = std::stold(parts[4]) / std::stold(parts[5]);
  if (parts.size() == 7) {
    *has_truth = true;
    *embedded_truth = parts[6] == "1" || parts[6] == "true";
  }
  return ticcd::Vector3(static_cast<ticcd::Scalar>(x),
                        static_cast<ticcd::Scalar>(y),
                        static_cast<ticcd::Scalar>(z));
}

std::vector<Query> LoadQueries(const FileSpec& file) {
  std::ifstream in(file.csv_path);
  if (!in) {
    throw std::runtime_error("failed to open query CSV " + file.csv_path);
  }
  std::vector<bool> labels;
  if (!file.bool_json.empty()) {
    labels = ParseBoolJsonArray(file.bool_json);
  }
  std::vector<Query> queries;
  std::array<std::string, 8> lines;
  std::uint64_t query_index = 0;
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
    Query query;
    bool has_embedded_truth = false;
    bool embedded_truth = false;
    bool first_embedded_truth = false;
    for (int row = 0; row < 8; ++row) {
      bool row_truth = false;
      bool row_has_truth = false;
      query.vertices[row] = ParseRationalVertex(lines[row], &row_truth, &row_has_truth);
      if (row_has_truth) {
        if (!has_embedded_truth) {
          has_embedded_truth = true;
          first_embedded_truth = row_truth;
        } else if (row_truth != first_embedded_truth) {
          throw std::runtime_error("embedded truth changes inside query block");
        }
      }
    }
    if (!labels.empty()) {
      if (query_index >= labels.size()) {
        throw std::runtime_error("bool_json has fewer labels than query CSV");
      }
      query.label = labels[query_index];
    } else if (has_embedded_truth) {
      query.label = first_embedded_truth;
    } else {
      throw std::runtime_error("query CSV has no embedded truth and manifest has no bool_json");
    }
    queries.push_back(query);
    ++query_index;
  }
  if (!labels.empty() && labels.size() != queries.size()) {
    throw std::runtime_error("bool_json label count does not match query count for " + file.csv_path);
  }
  return queries;
}

std::uint64_t StableHash(const std::string& value) {
  std::uint64_t h = 1469598103934665603ull;
  for (unsigned char ch : value) {
    h ^= static_cast<std::uint64_t>(ch);
    h *= 1099511628211ull;
  }
  return h;
}

std::unordered_map<std::string, std::vector<std::pair<std::uint64_t, double>>> LoadSchedule(const fs::path& path) {
  std::unordered_map<std::string, std::vector<std::pair<std::uint64_t, double>>> schedules;
  if (path.empty()) {
    return schedules;
  }
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("failed to open schedule " + path.string());
  }
  std::string line;
  while (std::getline(in, line)) {
    if (Trim(line).empty()) {
      continue;
    }
    const std::size_t last_comma = line.rfind(',');
    if (last_comma == std::string::npos) {
      continue;
    }
    const std::size_t second_last = line.rfind(',', last_comma - 1);
    if (second_last == std::string::npos) {
      continue;
    }
    const std::string csv_path = NormalizeSlashes(line.substr(0, second_last));
    const std::string index_text = Trim(line.substr(second_last + 1, last_comma - second_last - 1));
    const std::string score_text = Trim(line.substr(last_comma + 1));
    if (csv_path == "csv_path" || index_text == "query_index") {
      continue;
    }
    try {
      const std::uint64_t index = static_cast<std::uint64_t>(std::stoull(index_text));
      const double score = std::stod(score_text);
      schedules[csv_path].push_back({index, score});
    } catch (const std::exception&) {
      continue;
    }
  }
  return schedules;
}

std::vector<std::uint64_t> BuildOrder(
    const std::string& method,
    const FileSpec& file,
    const std::vector<Query>& queries,
    const std::unordered_map<std::string, std::vector<std::pair<std::uint64_t, double>>>& schedules) {
  std::vector<std::uint64_t> order(queries.size());
  for (std::uint64_t i = 0; i < order.size(); ++i) {
    order[i] = i;
  }
  if (method.find("Oracle") != std::string::npos) {
    std::stable_sort(order.begin(), order.end(), [&](std::uint64_t a, std::uint64_t b) {
      return static_cast<int>(queries[a].label) > static_cast<int>(queries[b].label);
    });
    return order;
  }
  if (method.find("Random") != std::string::npos && schedules.empty()) {
    std::mt19937_64 rng(StableHash(file.csv_path) ^ 20260505ull);
    std::shuffle(order.begin(), order.end(), rng);
    return order;
  }
  if (!schedules.empty()) {
    const auto it = schedules.find(file.csv_path);
    if (it == schedules.end()) {
      return order;
    }
    std::vector<std::pair<std::uint64_t, double>> scored = it->second;
    std::stable_sort(scored.begin(), scored.end(), [](const auto& a, const auto& b) {
      return a.second > b.second;
    });
    order.clear();
    order.reserve(queries.size());
    std::vector<unsigned char> seen(queries.size(), 0);
    for (const auto& item : scored) {
      if (item.first < queries.size() && !seen[item.first]) {
        order.push_back(item.first);
        seen[item.first] = 1;
      }
    }
    for (std::uint64_t i = 0; i < queries.size(); ++i) {
      if (!seen[i]) {
        order.push_back(i);
      }
    }
  }
  return order;
}

bool IsEdgeEdgeKind(const std::string& kind) {
  return kind == "ee" || kind == "edge-edge" || kind == "edge_edge";
}

bool RunTightInclusionQuery(const std::string& kind,
                            const std::array<ticcd::Vector3, 8>& vertices,
                            const Args& args,
                            double* toi_out) {
  const ticcd::Array3 err(-1, -1, -1);
  ticcd::Scalar toi = std::numeric_limits<ticcd::Scalar>::infinity();
  ticcd::Scalar output_tolerance = static_cast<ticcd::Scalar>(args.tolerance);
  bool hit = false;
  if (IsEdgeEdgeKind(kind)) {
    hit = ticcd::edgeEdgeCCD(vertices[0], vertices[1], vertices[2], vertices[3],
                             vertices[4], vertices[5], vertices[6], vertices[7],
                             err, static_cast<ticcd::Scalar>(args.ms), toi,
                             static_cast<ticcd::Scalar>(args.tolerance),
                             static_cast<ticcd::Scalar>(args.t_max), args.max_itr,
                             output_tolerance, false,
                             ticcd::CCDRootFindingMethod::BREADTH_FIRST_SEARCH);
  } else {
    hit = ticcd::vertexFaceCCD(vertices[0], vertices[1], vertices[2], vertices[3],
                               vertices[4], vertices[5], vertices[6], vertices[7],
                               err, static_cast<ticcd::Scalar>(args.ms), toi,
                               static_cast<ticcd::Scalar>(args.tolerance),
                               static_cast<ticcd::Scalar>(args.t_max), args.max_itr,
                               output_tolerance, false,
                               ticcd::CCDRootFindingMethod::BREADTH_FIRST_SEARCH);
  }
  *toi_out = static_cast<double>(toi);
  return hit;
}

bool IsAllExactMethod(const std::string& method) {
  return method.find("AllExact") != std::string::npos || method.find("Enumeration") != std::string::npos;
}

Row BenchmarkFile(
    const Args& args,
    const FileSpec& file,
    const std::unordered_map<std::string, std::vector<std::pair<std::uint64_t, double>>>& schedules) {
  Row row;
  row.method = args.method;
  row.timing_scope = IsAllExactMethod(args.method)
                         ? "native_full_scene_replay_all_candidate_enumeration"
                         : "native_full_scene_replay_any_hit_detection";
  row.file = file;
  const auto total_begin = std::chrono::steady_clock::now();
  const auto load_begin = total_begin;
  std::vector<Query> queries = LoadQueries(file);
  if (args.max_queries_per_file != 0 && queries.size() > args.max_queries_per_file) {
    queries.resize(static_cast<std::size_t>(args.max_queries_per_file));
  }
  const auto load_end = std::chrono::steady_clock::now();
  const auto schedule_begin = load_end;
  std::vector<std::uint64_t> order = BuildOrder(args.method, file, queries, schedules);
  const auto schedule_end = std::chrono::steady_clock::now();
  row.load_us = std::chrono::duration<double, std::micro>(load_end - load_begin).count();
  row.schedule_us = std::chrono::duration<double, std::micro>(schedule_end - schedule_begin).count();
  row.query_count = queries.size();
  for (const Query& query : queries) {
    row.positive_count += query.label ? 1 : 0;
  }
  row.expected_scene_hit = row.positive_count > 0;
  const auto detect_begin = schedule_end;
  const bool all_exact = IsAllExactMethod(args.method);
  for (const std::uint64_t index : order) {
    if (index >= queries.size()) {
      continue;
    }
    const Query& query = queries[index];
    double toi = std::numeric_limits<double>::quiet_NaN();
    const auto exact_begin = std::chrono::steady_clock::now();
    const bool hit = RunTightInclusionQuery(file.kind, query.vertices, args, &toi);
    const auto exact_end = std::chrono::steady_clock::now();
    row.exact_us += std::chrono::duration<double, std::micro>(exact_end - exact_begin).count();
    ++row.exact_calls;
    row.executed_label_positives += query.label ? 1 : 0;
    row.executed_label_negatives += query.label ? 0 : 1;
    row.exact_positive_count += hit ? 1 : 0;
    if (hit && !query.label) {
      ++row.exact_false_positive_count;
    }
    if (!hit && query.label) {
      ++row.exact_false_negative_count;
    }
    if (hit) {
      row.detected_scene_hit = true;
      row.first_hit_toi = toi;
      if (!all_exact) {
        break;
      }
    }
  }
  const auto detect_end = std::chrono::steady_clock::now();
  row.skipped_candidates = row.query_count - row.exact_calls;
  row.detection_wall_us =
      row.schedule_us + std::chrono::duration<double, std::micro>(detect_end - detect_begin).count();
  const auto total_end = detect_end;
  row.total_wall_us = std::chrono::duration<double, std::micro>(total_end - total_begin).count();
  if (row.expected_scene_hit && row.detected_scene_hit) {
    row.scene_tp = 1;
  } else if (!row.expected_scene_hit && !row.detected_scene_hit) {
    row.scene_tn = 1;
  } else if (!row.expected_scene_hit && row.detected_scene_hit) {
    row.scene_fp = 1;
  } else {
    row.scene_fn = 1;
  }
  return row;
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
    } else if (key == "--schedule") {
      args.schedule = need_value(key);
    } else if (key == "--output-jsonl") {
      args.output_jsonl = need_value(key);
    } else if (key == "--method") {
      args.method = need_value(key);
    } else if (key == "--ms") {
      args.ms = std::stod(need_value(key));
    } else if (key == "--tolerance") {
      args.tolerance = std::stod(need_value(key));
    } else if (key == "--t-max") {
      args.t_max = std::stod(need_value(key));
    } else if (key == "--max-itr") {
      args.max_itr = std::stol(need_value(key));
    } else if (key == "--max-files") {
      args.max_files = static_cast<std::uint64_t>(std::stoull(need_value(key)));
    } else if (key == "--max-queries-per-file") {
      args.max_queries_per_file = static_cast<std::uint64_t>(std::stoull(need_value(key)));
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
  return args;
}

std::string JsonEscape(const std::string& value) {
  std::string out;
  for (const char ch : value) {
    if (ch == '"' || ch == '\\') {
      out.push_back('\\');
    }
    if (ch == '\n') {
      out += "\\n";
    } else if (ch == '\r') {
      out += "\\r";
    } else {
      out.push_back(ch);
    }
  }
  return out;
}

void WriteRow(std::ostream& out, const Row& row) {
  const double call_reduction =
      1.0 - static_cast<double>(row.exact_calls) / static_cast<double>(std::max<std::uint64_t>(1, row.query_count));
  const double scene_recall =
      static_cast<double>(row.scene_tp) / static_cast<double>(std::max<std::uint64_t>(1, row.scene_tp + row.scene_fn));
  out << "{"
      << "\"method\":\"" << JsonEscape(row.method) << "\","
      << "\"timing_scope\":\"" << JsonEscape(row.timing_scope) << "\","
      << "\"scene\":\"" << JsonEscape(row.file.scene) << "\","
      << "\"kind\":\"" << JsonEscape(row.file.kind) << "\","
      << "\"split\":\"" << JsonEscape(row.file.split) << "\","
      << "\"timestep\":" << row.file.timestep << ","
      << "\"csv_path\":\"" << JsonEscape(row.file.csv_path) << "\","
      << "\"frame0\":\"" << JsonEscape(row.file.frame0) << "\","
      << "\"frame1\":\"" << JsonEscape(row.file.frame1) << "\","
      << "\"query_count\":" << row.query_count << ","
      << "\"positive_count\":" << row.positive_count << ","
      << "\"exact_calls\":" << row.exact_calls << ","
      << "\"skipped_candidates\":" << row.skipped_candidates << ","
      << "\"call_reduction\":" << call_reduction << ","
      << "\"executed_label_positives\":" << row.executed_label_positives << ","
      << "\"executed_label_negatives\":" << row.executed_label_negatives << ","
      << "\"exact_positive_count\":" << row.exact_positive_count << ","
      << "\"exact_false_positive_count\":" << row.exact_false_positive_count << ","
      << "\"exact_false_negative_count\":" << row.exact_false_negative_count << ","
      << "\"expected_scene_hit\":" << (row.expected_scene_hit ? "true" : "false") << ","
      << "\"detected_scene_hit\":" << (row.detected_scene_hit ? "true" : "false") << ","
      << "\"scene_tp\":" << row.scene_tp << ","
      << "\"scene_tn\":" << row.scene_tn << ","
      << "\"scene_fp\":" << row.scene_fp << ","
      << "\"scene_fn\":" << row.scene_fn << ","
      << "\"scene_recall\":" << scene_recall << ","
      << "\"load_us\":" << row.load_us << ","
      << "\"schedule_us\":" << row.schedule_us << ","
      << "\"exact_us\":" << row.exact_us << ","
      << "\"detection_wall_us\":" << row.detection_wall_us << ","
      << "\"total_wall_us\":" << row.total_wall_us << ","
      << "\"avg_detection_us_per_query\":" << (row.detection_wall_us / std::max<std::uint64_t>(1, row.query_count)) << ","
      << "\"avg_detection_us_per_exact_call\":" << (row.detection_wall_us / std::max<std::uint64_t>(1, row.exact_calls)) << ","
      << "\"first_hit_toi\":" << row.first_hit_toi
      << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Args args = ParseArgs(argc, argv);
    const std::vector<FileSpec> files = ParseManifestFiles(ReadTextFile(args.manifest));
    const auto schedules = LoadSchedule(args.schedule);
    fs::create_directories(args.output_jsonl.parent_path());
    std::ofstream out(args.output_jsonl);
    if (!out) {
      throw std::runtime_error("failed to write " + args.output_jsonl.string());
    }
    std::uint64_t file_count = 0;
    for (const FileSpec& file : files) {
      if (args.max_files != 0 && file_count >= args.max_files) {
        break;
      }
      const Row row = BenchmarkFile(args, file, schedules);
      WriteRow(out, row);
      ++file_count;
    }
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "full_scene_ccd_replay_benchmark failed: " << ex.what() << "\n";
    return 1;
  }
}
