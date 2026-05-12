#include "proposal/proposal_features.h"
#include "proposal/proposal_policy.h"
#include "rt_candidate/candidate_generator.h"
#include "rt_candidate/proxy_scene.h"

#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;

std::string HtmlEscape(const std::string& value) {
  std::string escaped;
  escaped.reserve(value.size());
  for (const char c : value) {
    switch (c) {
      case '&':
        escaped += "&amp;";
        break;
      case '<':
        escaped += "&lt;";
        break;
      case '>':
        escaped += "&gt;";
        break;
      case '"':
        escaped += "&quot;";
        break;
      default:
        escaped += c;
        break;
    }
  }
  return escaped;
}

std::string FormatDouble(const double value, const int precision = 3) {
  std::ostringstream stream;
  stream << std::fixed << std::setprecision(precision) << value;
  return stream.str();
}

std::string SourceName(const p2cccd::ProposalSource source) {
  switch (source) {
    case p2cccd::ProposalSource::kRaw:
      return "proposal";
    case p2cccd::ProposalSource::kRefined:
      return "refined";
    case p2cccd::ProposalSource::kFallback:
      return "fallback";
  }
  return "invalid";
}

std::string FamilyMaskName(const std::uint32_t mask) {
  std::vector<std::string> names;
  if ((mask & p2cccd::kFeatureFamilyPointTriangle) != 0) {
    names.emplace_back("PT");
  }
  if ((mask & p2cccd::kFeatureFamilyEdgeEdge) != 0) {
    names.emplace_back("EE");
  }
  if (names.empty()) {
    return "none";
  }
  std::ostringstream stream;
  for (std::size_t i = 0; i < names.size(); ++i) {
    if (i > 0) {
      stream << "+";
    }
    stream << names[i];
  }
  return stream.str();
}

p2cccd::Patch MakePatch(const std::uint32_t patch_id,
                        const double x,
                        const double y,
                        const double radius,
                        const double area = 1.0) {
  p2cccd::Patch patch;
  patch.patch_id = patch_id;
  patch.triangle_ids = {patch_id * 2U, patch_id * 2U + 1U};
  patch.triangle_count = static_cast<std::uint32_t>(patch.triangle_ids.size());
  patch.area = area;
  patch.local_center = {x, y, 0.0};
  patch.radius = radius;
  return patch;
}

p2cccd::MotionSegment MakePlanarMotion(const double tx0,
                                       const double ty0,
                                       const double tx1,
                                       const double ty1,
                                       const double angle_t1) {
  const double half_angle = 0.5 * angle_t1;
  p2cccd::MotionSegment motion;
  motion.t0 = 0.0;
  motion.t1 = 1.0;
  motion.pose_t0.translation = {tx0, ty0, 0.0};
  motion.pose_t1.translation = {tx1, ty1, 0.0};
  motion.pose_t0.rotation_xyzw = {0.0, 0.0, 0.0, 1.0};
  motion.pose_t1.rotation_xyzw = {0.0, 0.0, std::sin(half_angle), std::cos(half_angle)};
  return motion;
}

p2cccd::ProxySceneBuildInput MakeSceneInput() {
  p2cccd::ProxySceneBuildInput input;
  input.query_id = 6800;

  p2cccd::ProxyObjectBuildInput object_a;
  object_a.object_id = 10;
  object_a.proxy_type = p2cccd::ProxyType::kSweptAabb;
  object_a.patches = {MakePatch(1, 0.00, 0.00, 0.24, 0.72),
                      MakePatch(2, 2.60, 0.08, 0.20, 0.50)};
  object_a.motion_segments = {MakePlanarMotion(0.00, 0.00, 0.64, 0.18, 0.20 * kPi)};
  object_a.slabs_per_motion_segment = 4;
  object_a.eps_proxy = 0.04;

  p2cccd::ProxyObjectBuildInput object_b;
  object_b.object_id = 20;
  object_b.proxy_type = p2cccd::ProxyType::kCapsule;
  object_b.patches = {MakePatch(3, 0.32, 0.05, 0.23, 0.62),
                      MakePatch(4, 4.20, 0.00, 0.18, 0.38)};
  object_b.motion_segments = {MakePlanarMotion(0.04, 0.03, 0.44, 0.15, 0.08 * kPi)};
  object_b.slabs_per_motion_segment = 4;
  object_b.eps_proxy = 0.04;

  input.objects = {object_a, object_b};
  return input;
}

bool CheckOk(const p2cccd::Status& status, const char* label) {
  if (status.ok) {
    return true;
  }
  std::cerr << label << " failed: " << status.message << '\n';
  return false;
}

p2cccd::Status EnsureParentDirectory(const std::filesystem::path& path) {
  const std::filesystem::path parent = path.parent_path();
  if (parent.empty()) {
    return p2cccd::Status::Ok();
  }
  std::error_code ec;
  std::filesystem::create_directories(parent, ec);
  if (ec) {
    return p2cccd::Status::Error("failed to create output directory: " + ec.message());
  }
  return p2cccd::Status::Ok();
}

double ProposalMicrobenchmarkUs(const p2cccd::ProxyScene& scene,
                                const p2cccd::CandidateGenerationResult& generation_result,
                                const int iterations) {
  using Clock = std::chrono::steady_clock;
  const auto start = Clock::now();
  for (int i = 0; i < iterations; ++i) {
    p2cccd::RawCandidateQueue raw_queue;
    std::vector<p2cccd::ProposalFeatureRow> rows;
    std::vector<p2cccd::ProposalOutput> outputs;
    std::vector<p2cccd::ExactWorkItem> work_queue;
    p2cccd::ProposalScheduleStats stats;
    p2cccd::ProposalSchedulingConfig config;
    config.first_work_item_id = 1000;
    if (!p2cccd::BuildRawCandidateQueue(generation_result, &raw_queue).ok ||
        !p2cccd::ExtractProposalFeatureRows(scene, generation_result, &rows).ok ||
        !p2cccd::BuildDummyProposalOutputs(rows, &outputs).ok ||
        !p2cccd::ScheduleExactWorkItemsFromProposals(scene,
                                                      raw_queue,
                                                      rows,
                                                      outputs,
                                                      config,
                                                      &work_queue,
                                                      &stats)
             .ok) {
      return -1.0;
    }
  }
  const auto elapsed = std::chrono::duration<double, std::micro>(Clock::now() - start).count();
  return elapsed / static_cast<double>(iterations);
}

void WriteCards(std::ostream& out,
                const p2cccd::ProxyScene& scene,
                const p2cccd::CandidateGenerationResult& generation_result,
                const std::vector<p2cccd::ProposalFeatureRow>& rows,
                const std::vector<p2cccd::ExactWorkItem>& work_queue,
                const p2cccd::ProposalScheduleStats& normal_stats,
                const p2cccd::ProposalScheduleStats& stress_stats,
                const double proposal_us) {
  out << "<section class=\"cards\">";
  out << "<div class=\"card\"><b>Proxy primitives</b><span>" << scene.primitives.size()
      << "</span></div>";
  out << "<div class=\"card\"><b>Raw RT hits</b><span>"
      << generation_result.density.raw_hit_count << "</span></div>";
  out << "<div class=\"card\"><b>Compact candidates</b><span>"
      << generation_result.candidates.size() << "</span></div>";
  out << "<div class=\"card\"><b>Feature rows</b><span>" << rows.size() << "</span></div>";
  out << "<div class=\"card\"><b>Exact work items</b><span>" << work_queue.size()
      << "</span></div>";
  out << "<div class=\"card\"><b>Normal fallback</b><span>" << normal_stats.fallback_count
      << "</span></div>";
  out << "<div class=\"card warn\"><b>Stress fallback</b><span>" << stress_stats.fallback_count
      << "</span></div>";
  out << "<div class=\"card\"><b>Proposal smoke avg</b><span>"
      << FormatDouble(proposal_us, 2) << " us</span></div>";
  out << "</section>";
}

void WritePipeline(std::ostream& out) {
  out << R"HTML(
<section class="panel">
<h2>No-Drop Proposal Pipeline</h2>
<svg viewBox="0 0 960 170" role="img" aria-label="proposal pipeline">
  <defs>
    <marker id="arrow" markerWidth="10" markerHeight="10" refX="7" refY="3" orient="auto">
      <path d="M0,0 L0,6 L8,3 z" fill="#f2c14e"/>
    </marker>
  </defs>
  <g class="node"><rect x="25" y="45" width="150" height="70"/><text x="100" y="76">Raw Candidate</text><text x="100" y="98">Queue</text></g>
  <g class="node"><rect x="215" y="45" width="150" height="70"/><text x="290" y="76">32D Feature</text><text x="290" y="98">Rows</text></g>
  <g class="node accent"><rect x="405" y="45" width="150" height="70"/><text x="480" y="76">Dummy / STPF</text><text x="480" y="98">Policy</text></g>
  <g class="node"><rect x="595" y="45" width="150" height="70"/><text x="670" y="76">Monotonic</text><text x="670" y="98">Scheduler</text></g>
  <g class="node safe"><rect x="785" y="45" width="150" height="70"/><text x="860" y="76">Exact Work</text><text x="860" y="98">Queue</text></g>
  <path class="arrow" d="M175 80 H215" marker-end="url(#arrow)"/>
  <path class="arrow" d="M365 80 H405" marker-end="url(#arrow)"/>
  <path class="arrow" d="M555 80 H595" marker-end="url(#arrow)"/>
  <path class="arrow" d="M745 80 H785" marker-end="url(#arrow)"/>
  <text class="caption" x="480" y="145">Invariant: every raw candidate maps to exactly one uncertified exact work item.</text>
</svg>
</section>
)HTML";
}

void WriteCandidateTable(std::ostream& out,
                         const std::vector<p2cccd::ProposalFeatureRow>& rows,
                         const std::vector<p2cccd::ExactWorkItem>& work_queue) {
  std::map<std::uint64_t, p2cccd::ExactWorkItem> by_candidate;
  for (const p2cccd::ExactWorkItem& item : work_queue) {
    by_candidate[item.parent_candidate_id] = item;
  }

  out << "<section class=\"panel\"><h2>Candidate To Work-Item Mapping</h2><table>";
  out << "<tr><th>candidate</th><th>slab</th><th>patch pair</th><th>overlap feature</th>"
      << "<th>priority target</th><th>work item</th><th>source</th><th>families</th></tr>";
  for (const p2cccd::ProposalFeatureRow& row : rows) {
    const auto item_it = by_candidate.find(row.candidate_id);
    out << "<tr><td>" << row.candidate_id << "</td><td>" << row.slab_id << "</td><td>"
        << row.patch_a_id << " x " << row.patch_b_id << "</td><td>"
        << FormatDouble(row.features[19], 4) << "</td><td>"
        << FormatDouble(row.priority_target, 3) << "</td>";
    if (item_it != by_candidate.end()) {
      out << "<td>" << item_it->second.work_item_id << "</td><td>"
          << HtmlEscape(SourceName(item_it->second.source)) << "</td><td>"
          << HtmlEscape(FamilyMaskName(item_it->second.feature_family_mask)) << "</td>";
    } else {
      out << "<td colspan=\"3\">missing</td>";
    }
    out << "</tr>";
  }
  out << "</table></section>";
}

void WriteStressTable(std::ostream& out,
                      const p2cccd::ProposalScheduleStats& normal_stats,
                      const p2cccd::ProposalScheduleStats& stress_stats) {
  out << "<section class=\"panel\"><h2>Fallback Stress Check</h2><table>";
  out << "<tr><th>case</th><th>raw</th><th>work</th><th>fallback</th>"
      << "<th>missing</th><th>invalid</th><th>OOD</th><th>high uncertainty</th>"
      << "<th>monotonic</th></tr>";
  auto row = [&out](const char* name, const p2cccd::ProposalScheduleStats& stats) {
    out << "<tr><td>" << name << "</td><td>" << stats.raw_candidate_count << "</td><td>"
        << stats.work_item_count << "</td><td>" << stats.fallback_count << "</td><td>"
        << stats.missing_proposal_fallback_count << "</td><td>"
        << stats.invalid_proposal_fallback_count << "</td><td>" << stats.ood_fallback_count
        << "</td><td>" << stats.high_uncertainty_fallback_count << "</td><td>"
        << (stats.monotonic_safe ? "yes" : "no") << "</td></tr>";
  };
  row("normal", normal_stats);
  row("stress", stress_stats);
  out << "</table></section>";
}

bool WriteHtml(const std::filesystem::path& output_path,
               const p2cccd::ProxyScene& scene,
               const p2cccd::CandidateGenerationResult& generation_result,
               const p2cccd::RawCandidateQueue& raw_queue,
               const std::vector<p2cccd::ProposalFeatureRow>& rows,
               const std::vector<p2cccd::ExactWorkItem>& work_queue,
               const p2cccd::ProposalScheduleStats& normal_stats,
               const p2cccd::ProposalScheduleStats& stress_stats,
               const double proposal_us) {
  if (!CheckOk(EnsureParentDirectory(output_path), "EnsureParentDirectory")) {
    return false;
  }
  std::ofstream out(output_path);
  if (!out) {
    std::cerr << "failed to open " << output_path.string() << '\n';
    return false;
  }
  out << R"HTML(<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>P2CCCD Proposal Queue And STPF</title>
<style>
  :root { --bg:#10130f; --panel:#191e17; --ink:#f1eddf; --muted:#aab09f; --gold:#f2c14e; --safe:#74b49b; --warn:#d46a4c; --line:#343b2f; }
  body { margin:0; background:radial-gradient(circle at 12% 8%, rgba(242,193,78,.16), transparent 28%), linear-gradient(135deg,#10130f,#151914 48%,#0e1415); color:var(--ink); font-family: Georgia, 'Times New Roman', serif; }
  main { width:min(1180px, calc(100% - 48px)); margin:0 auto; padding:38px 0 56px; }
  h1 { margin:0 0 8px; font-size:34px; letter-spacing:.01em; }
  h2 { margin:0 0 16px; font-size:20px; }
  p { color:var(--muted); line-height:1.55; }
  code { color:var(--gold); }
  .cards { display:grid; grid-template-columns:repeat(4,1fr); gap:14px; margin:24px 0; }
  .card { background:linear-gradient(180deg, rgba(255,255,255,.055), rgba(255,255,255,.02)); border:1px solid var(--line); border-radius:16px; padding:16px; min-height:78px; box-shadow:0 16px 36px rgba(0,0,0,.18); }
  .card b { display:block; color:var(--muted); font-size:13px; font-weight:400; }
  .card span { display:block; margin-top:12px; font-size:27px; color:var(--ink); }
  .card.warn span { color:var(--warn); }
  .panel { background:rgba(25,30,23,.88); border:1px solid var(--line); border-radius:20px; padding:22px; margin:18px 0; box-shadow:0 18px 46px rgba(0,0,0,.24); }
  svg { width:100%; height:auto; }
  .node rect { fill:#242b20; stroke:#53604b; stroke-width:1.3; rx:13; }
  .node.accent rect { fill:#3a2f18; stroke:#f2c14e; }
  .node.safe rect { fill:#1d332d; stroke:#74b49b; }
  .node text { fill:var(--ink); font-size:16px; text-anchor:middle; font-family:Georgia, 'Times New Roman', serif; }
  .arrow { stroke:#f2c14e; stroke-width:3; fill:none; }
  .caption { fill:var(--muted); text-anchor:middle; font-size:15px; }
  table { width:100%; border-collapse:collapse; font-size:14px; }
  th, td { border-bottom:1px solid var(--line); padding:10px 8px; text-align:left; }
  th { color:var(--gold); font-weight:500; }
  .note { border-left:4px solid var(--gold); padding:12px 14px; background:rgba(242,193,78,.08); border-radius:10px; }
  @media (max-width: 900px) { .cards { grid-template-columns:repeat(2,1fr); } }
</style>
</head>
<body><main>
)HTML";
  out << "<h1>P2CCCD Proposal Queue And STPF</h1>";
  out << "<p>Generated from the current C++ pipeline. Query <code>" << raw_queue.query_id
      << "</code>, backend <code>" << HtmlEscape(generation_result.backend_name)
      << "</code>. This is a smoke visualization and micro timing, not the formal paper benchmark.</p>";
  out << "<p class=\"note\">Formal benchmark status: TODO 79-93 now have Python baseline, ablation, style-comparison, or downstream runners. This C++ view remains a smoke visualization for candidate density and proposal-queue behavior, not the paper benchmark exporter.</p>";
  WriteCards(out, scene, generation_result, rows, work_queue, normal_stats, stress_stats, proposal_us);
  WritePipeline(out);
  WriteCandidateTable(out, rows, work_queue);
  WriteStressTable(out, normal_stats, stress_stats);
  out << "<section class=\"panel\"><h2>Timing Snapshot</h2><table>"
      << "<tr><th>stage</th><th>value</th></tr>"
      << "<tr><td>RT candidate total</td><td>" << FormatDouble(generation_result.timing.total_ms, 5)
      << " ms</td></tr>"
      << "<tr><td>RT trace</td><td>" << FormatDouble(generation_result.timing.trace_ms, 5)
      << " ms</td></tr>"
      << "<tr><td>RT compaction</td><td>" << FormatDouble(generation_result.timing.compact_ms, 5)
      << " ms</td></tr>"
      << "<tr><td>Proposal feature + dummy policy + scheduler smoke average</td><td>"
      << FormatDouble(proposal_us, 3) << " us / iteration</td></tr>"
      << "</table></section>";
  out << "</main></body></html>\n";
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  const std::filesystem::path output_path =
      argc >= 2 ? std::filesystem::path(argv[1])
                : std::filesystem::path("outputs/proposal_queue_stpf_overview.html");

  p2cccd::ProxyScene scene;
  if (!CheckOk(p2cccd::BuildProxyScene(MakeSceneInput(), &scene), "BuildProxyScene")) {
    return 1;
  }

  p2cccd::CandidateGenerationResult generation_result;
  p2cccd::CandidateGenerator generator;
  if (!CheckOk(generator.GenerateCandidates(scene, scene.query_id, &generation_result),
               "GenerateCandidates")) {
    return 1;
  }

  p2cccd::RawCandidateQueue raw_queue;
  std::vector<p2cccd::ProposalFeatureRow> rows;
  std::vector<p2cccd::ProposalOutput> outputs;
  std::vector<p2cccd::ExactWorkItem> work_queue;
  p2cccd::ProposalScheduleStats normal_stats;
  p2cccd::ProposalSchedulingConfig config;
  config.first_work_item_id = 2000;

  if (!CheckOk(p2cccd::BuildRawCandidateQueue(generation_result, &raw_queue),
               "BuildRawCandidateQueue") ||
      !CheckOk(p2cccd::ExtractProposalFeatureRows(scene, generation_result, &rows),
               "ExtractProposalFeatureRows") ||
      !CheckOk(p2cccd::BuildDummyProposalOutputs(rows, &outputs), "BuildDummyProposalOutputs") ||
      !CheckOk(p2cccd::ScheduleExactWorkItemsFromProposals(scene,
                                                            raw_queue,
                                                            rows,
                                                            outputs,
                                                            config,
                                                            &work_queue,
                                                            &normal_stats),
               "ScheduleExactWorkItemsFromProposals")) {
    return 1;
  }

  std::vector<p2cccd::ProposalFeatureRow> stress_rows = rows;
  std::vector<p2cccd::ProposalOutput> stress_outputs = outputs;
  if (!stress_rows.empty()) {
    stress_rows.front().features[0] = 1.0e6F;
  }
  if (!stress_outputs.empty()) {
    stress_outputs.front().uncertainty_score = 0.99F;
  }
  if (stress_outputs.size() > 1) {
    stress_outputs.pop_back();
  }
  std::vector<p2cccd::ExactWorkItem> stress_work_queue;
  p2cccd::ProposalScheduleStats stress_stats;
  p2cccd::ProposalSchedulingConfig stress_config = config;
  stress_config.ood_abs_feature_threshold = 100.0F;
  if (!CheckOk(p2cccd::ScheduleExactWorkItemsFromProposals(scene,
                                                            raw_queue,
                                                            stress_rows,
                                                            stress_outputs,
                                                            stress_config,
                                                            &stress_work_queue,
                                                            &stress_stats),
               "ScheduleExactWorkItemsFromProposals stress")) {
    return 1;
  }

  const double proposal_us = ProposalMicrobenchmarkUs(scene, generation_result, 2000);
  if (proposal_us < 0.0) {
    std::cerr << "ProposalMicrobenchmark failed\n";
    return 1;
  }

  if (!WriteHtml(output_path,
                 scene,
                 generation_result,
                 raw_queue,
                 rows,
                 work_queue,
                 normal_stats,
                 stress_stats,
                 proposal_us)) {
    return 1;
  }

  std::cout << "wrote " << std::filesystem::absolute(output_path).string() << '\n';
  std::cout << "candidates=" << generation_result.candidates.size()
            << ", work_items=" << work_queue.size()
            << ", normal_fallback=" << normal_stats.fallback_count
            << ", stress_fallback=" << stress_stats.fallback_count
            << ", proposal_us=" << proposal_us << '\n';
  return 0;
}
