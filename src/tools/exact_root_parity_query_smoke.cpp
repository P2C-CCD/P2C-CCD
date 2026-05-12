#include <CCD/ccd.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct Args {
  fs::path dataset_root;
  fs::path output_json;
  fs::path output_md;
  std::vector<std::string> cases;
  std::uint64_t max_queries = 2000;
};

struct QueryBlock {
  std::array<ccd::Vector3d, 8> vertices;
  bool truth = false;
};

struct Stats {
  std::uint64_t queries = 0;
  std::uint64_t tp = 0;
  std::uint64_t tn = 0;
  std::uint64_t fp = 0;
  std::uint64_t fn = 0;
  double exact_ms = 0.0;
};

std::vector<std::string> Split(const std::string& text, char delimiter) {
  std::vector<std::string> out;
  std::stringstream stream(text);
  std::string item;
  while (std::getline(stream, item, delimiter)) {
    if (!item.empty()) {
      out.push_back(item);
    }
  }
  return out;
}

Args ParseArgs(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    const std::string key(argv[i]);
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) {
        throw std::runtime_error("missing value for " + key);
      }
      return std::string(argv[++i]);
    };
    if (key == "--dataset-root") {
      args.dataset_root = next();
    } else if (key == "--output-json") {
      args.output_json = next();
    } else if (key == "--output-md") {
      args.output_md = next();
    } else if (key == "--cases") {
      args.cases = Split(next(), ',');
    } else if (key == "--max-queries") {
      args.max_queries = static_cast<std::uint64_t>(std::stoull(next()));
    } else if (key == "--help" || key == "-h") {
      std::cout
          << "Usage: exact_root_parity_query_smoke --dataset-root ROOT "
          << "--output-json out.json --output-md out.md --cases unit-tests,golf-ball "
          << "[--max-queries 2000]\n";
      std::exit(0);
    } else {
      throw std::runtime_error("unknown argument: " + key);
    }
  }
  if (args.dataset_root.empty() || args.output_json.empty() || args.output_md.empty() ||
      args.cases.empty()) {
    throw std::runtime_error("dataset root, outputs, and cases are required");
  }
  return args;
}

std::array<std::string, 7> SplitRow(const std::string& line) {
  std::array<std::string, 7> parts{};
  std::stringstream stream(line);
  for (std::size_t i = 0; i < parts.size(); ++i) {
    if (!std::getline(stream, parts[i], ',')) {
      throw std::runtime_error("expected seven CSV columns");
    }
  }
  return parts;
}

ccd::Vector3d ParseVertex(const std::string& line, bool* truth_out) {
  const auto parts = SplitRow(line);
  const double x = std::stod(parts[0]) / std::stod(parts[1]);
  const double y = std::stod(parts[2]) / std::stod(parts[3]);
  const double z = std::stod(parts[4]) / std::stod(parts[5]);
  const int truth = std::stoi(parts[6]);
  if (truth != 0 && truth != 1) {
    throw std::runtime_error("truth must be 0 or 1");
  }
  *truth_out = truth == 1;
  return ccd::Vector3d(x, y, z);
}

bool ReadQuery(std::ifstream& in, QueryBlock* block) {
  std::string line;
  bool first_truth = false;
  for (std::size_t row = 0; row < 8; ++row) {
    if (!std::getline(in, line)) {
      return row == 0 ? false : throw std::runtime_error("truncated query block");
    }
    bool truth = false;
    block->vertices[row] = ParseVertex(line, &truth);
    if (row == 0) {
      first_truth = truth;
    } else if (truth != first_truth) {
      throw std::runtime_error("truth changes inside query block");
    }
  }
  block->truth = first_truth;
  return true;
}

bool RunERP(const std::string& kind, const QueryBlock& block) {
  if (kind == "edge-edge") {
    return ccd::edgeEdgeCCD(block.vertices[0], block.vertices[1], block.vertices[2], block.vertices[3],
                            block.vertices[4], block.vertices[5], block.vertices[6], block.vertices[7]);
  }
  return ccd::vertexFaceCCD(block.vertices[0], block.vertices[1], block.vertices[2], block.vertices[3],
                            block.vertices[4], block.vertices[5], block.vertices[6], block.vertices[7]);
}

std::vector<fs::path> DiscoverCSVs(const fs::path& root, const std::vector<std::string>& cases) {
  std::vector<fs::path> paths;
  for (const auto& case_name : cases) {
    for (const std::string kind : {"edge-edge", "vertex-face"}) {
      const fs::path dir = root / case_name / kind;
      if (!fs::exists(dir)) {
        continue;
      }
      for (const auto& entry : fs::directory_iterator(dir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".csv") {
          paths.push_back(entry.path());
        }
      }
    }
  }
  std::sort(paths.begin(), paths.end());
  return paths;
}

std::string CaseName(const fs::path& csv_path) {
  return csv_path.parent_path().parent_path().filename().generic_string();
}

std::string KindName(const fs::path& csv_path) {
  return csv_path.parent_path().filename().generic_string();
}

Stats Run(const Args& args, std::map<std::string, Stats>* by_case_kind) {
  Stats total;
  for (const fs::path& csv_path : DiscoverCSVs(args.dataset_root, args.cases)) {
    if (total.queries >= args.max_queries) {
      break;
    }
    const std::string case_name = CaseName(csv_path);
    const std::string kind = KindName(csv_path);
    const std::string key = case_name + "/" + kind;
    std::ifstream in(csv_path);
    if (!in) {
      throw std::runtime_error("failed to read " + csv_path.string());
    }
    QueryBlock block;
    while (total.queries < args.max_queries && ReadQuery(in, &block)) {
      const auto start = std::chrono::steady_clock::now();
      const bool result = RunERP(kind, block);
      const auto end = std::chrono::steady_clock::now();
      const double ms = std::chrono::duration<double, std::milli>(end - start).count();
      auto update = [&](Stats& stats) {
        ++stats.queries;
        stats.exact_ms += ms;
        if (block.truth && result) {
          ++stats.tp;
        } else if (!block.truth && !result) {
          ++stats.tn;
        } else if (!block.truth && result) {
          ++stats.fp;
        } else {
          ++stats.fn;
        }
      };
      update(total);
      update((*by_case_kind)[key]);
    }
  }
  return total;
}

double Recall(const Stats& stats) {
  return static_cast<double>(stats.tp) /
         static_cast<double>(std::max<std::uint64_t>(1, stats.tp + stats.fn));
}

void WriteJson(const fs::path& path, const Stats& total, const std::map<std::string, Stats>& by_case_kind) {
  std::ofstream out(path);
  out << "{\n"
      << "  \"method\": \"Exact-Root-Parity-CCD\",\n"
      << "  \"queries\": " << total.queries << ",\n"
      << "  \"tp\": " << total.tp << ", \"tn\": " << total.tn << ", \"fp\": " << total.fp
      << ", \"fn\": " << total.fn << ",\n"
      << "  \"recall\": " << Recall(total) << ",\n"
      << "  \"exact_ms\": " << total.exact_ms << ",\n"
      << "  \"by_case_kind\": [\n";
  bool first = true;
  for (const auto& [key, stats] : by_case_kind) {
    if (!first) {
      out << ",\n";
    }
    first = false;
    out << "    {\"case_kind\": \"" << key << "\", \"queries\": " << stats.queries
        << ", \"tp\": " << stats.tp << ", \"tn\": " << stats.tn << ", \"fp\": "
        << stats.fp << ", \"fn\": " << stats.fn << ", \"recall\": " << Recall(stats)
        << ", \"exact_ms\": " << stats.exact_ms << "}";
  }
  out << "\n  ]\n}\n";
}

void WriteMarkdown(const fs::path& path, const Stats& total, const std::map<std::string, Stats>& by_case_kind) {
  std::ofstream out(path);
  out << "# Exact-Root-Parity-CCD CSV Correctness Smoke\n\n"
      << "- Dataset: NYU/Tight-Inclusion primitive CSV format\n"
      << "- Tolerance: Exact-Root-Parity has no runtime tolerance knob; this smoke compares binary output against the CSV/Tight-Inclusion reference label.\n\n"
      << "| Scope | Queries | TP | TN | FP | FN | Recall | Exact ms |\n"
      << "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n"
      << "| total | " << total.queries << " | " << total.tp << " | " << total.tn << " | "
      << total.fp << " | " << total.fn << " | " << Recall(total) << " | "
      << total.exact_ms << " |\n";
  for (const auto& [key, stats] : by_case_kind) {
    out << "| `" << key << "` | " << stats.queries << " | " << stats.tp << " | "
        << stats.tn << " | " << stats.fp << " | " << stats.fn << " | "
        << Recall(stats) << " | " << stats.exact_ms << " |\n";
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Args args = ParseArgs(argc, argv);
    std::map<std::string, Stats> by_case_kind;
    const Stats total = Run(args, &by_case_kind);
    WriteJson(args.output_json, total, by_case_kind);
    WriteMarkdown(args.output_md, total, by_case_kind);
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << "\n";
    return 1;
  }
  return 0;
}
