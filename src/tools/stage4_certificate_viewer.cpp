#include "certificate/certificate_engine.h"
#include "common/validators.h"

#include <array>
#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct WorkItemView {
  std::string label;
  std::string family;
  p2cccd::ExactWorkItem item;
  p2cccd::CertificateResult result;
  p2cccd::Status item_status;
  p2cccd::Status result_status;
};

using Vec3 = std::array<double, 3>;

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

std::string FormatDouble(double value, int precision = 3) {
  std::ostringstream stream;
  stream << std::fixed << std::setprecision(precision) << value;
  return stream.str();
}

std::string CertificateStatusName(p2cccd::CertificateStatus status) {
  switch (status) {
    case p2cccd::CertificateStatus::kCollision:
      return "collision";
    case p2cccd::CertificateStatus::kSeparation:
      return "separation";
    case p2cccd::CertificateStatus::kUndecided:
      return "undecided";
  }
  return "invalid";
}

std::string RefinementModeName(p2cccd::CertificateRefinementMode mode) {
  switch (mode) {
    case p2cccd::CertificateRefinementMode::kNone:
      return "none";
    case p2cccd::CertificateRefinementMode::kBisectInterval:
      return "bisect_interval";
    case p2cccd::CertificateRefinementMode::kRequestGeometry:
      return "request_geometry";
    case p2cccd::CertificateRefinementMode::kEscalatePrecision:
      return "escalate_precision";
  }
  return "invalid";
}

p2cccd::ExactWorkItem MakeWorkItem(std::uint64_t work_item_id,
                                   std::uint64_t candidate_id,
                                   std::uint32_t slab_id,
                                   std::uint32_t patch_a_id,
                                   std::uint32_t patch_b_id,
                                   double t0,
                                   double t1,
                                   std::uint32_t feature_family_mask,
                                   float priority_score) {
  p2cccd::ExactWorkItem item;
  item.work_item_id = work_item_id;
  item.parent_candidate_id = candidate_id;
  item.query_id = 4100;
  item.slab_id = slab_id;
  item.patch_a_id = patch_a_id;
  item.patch_b_id = patch_b_id;
  item.interval_t0 = t0;
  item.interval_t1 = t1;
  item.feature_family_mask = feature_family_mask;
  item.priority_score = priority_score;
  item.source = p2cccd::ProposalSource::kRaw;
  return item;
}

p2cccd::LinearVertexTrajectory Vertex(std::int64_t id, Vec3 p0, Vec3 p1) {
  p2cccd::LinearVertexTrajectory trajectory;
  trajectory.feature_id = id;
  trajectory.position_t0 = p0;
  trajectory.position_t1 = p1;
  return trajectory;
}

p2cccd::PointTriangleIntervalPrimitive MovingPointStaticTriangle(double z0,
                                                                 double z1,
                                                                 std::int64_t point_id,
                                                                 std::int64_t triangle_id) {
  p2cccd::PointTriangleIntervalPrimitive primitive;
  primitive.point_id = point_id;
  primitive.triangle_id = triangle_id;
  primitive.point = Vertex(point_id, {0.25, 0.25, z0}, {0.25, 0.25, z1});
  primitive.triangle_v0 = Vertex(101, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0});
  primitive.triangle_v1 = Vertex(102, {1.0, 0.0, 0.0}, {1.0, 0.0, 0.0});
  primitive.triangle_v2 = Vertex(103, {0.0, 1.0, 0.0}, {0.0, 1.0, 0.0});
  return primitive;
}

p2cccd::EdgeEdgeIntervalPrimitive MovingEdgeThroughStaticEdge(double z0,
                                                              double z1,
                                                              std::int64_t edge_a_id,
                                                              std::int64_t edge_b_id) {
  p2cccd::EdgeEdgeIntervalPrimitive primitive;
  primitive.edge_a_id = edge_a_id;
  primitive.edge_b_id = edge_b_id;
  primitive.edge_a0 = Vertex(201, {-1.0, 0.0, 0.0}, {-1.0, 0.0, 0.0});
  primitive.edge_a1 = Vertex(202, {1.0, 0.0, 0.0}, {1.0, 0.0, 0.0});
  primitive.edge_b0 = Vertex(301, {0.0, -1.0, z0}, {0.0, -1.0, z1});
  primitive.edge_b1 = Vertex(302, {0.0, 1.0, z0}, {0.0, 1.0, z1});
  return primitive;
}

void AppendEvaluatedQuery(std::vector<WorkItemView>* views,
                          std::string label,
                          std::string family,
                          p2cccd::ExactCertificateQuery query) {
  p2cccd::CertificateEngine engine;
  WorkItemView view;
  view.label = std::move(label);
  view.family = std::move(family);
  view.item = query.work_item;
  view.item_status = p2cccd::ValidateExactWorkItem(query.work_item);
  const p2cccd::Status evaluate_status = engine.Evaluate(query, &view.result);
  view.result_status = evaluate_status.ok ? p2cccd::ValidateCertificateResult(view.result)
                                          : evaluate_status;
  views->push_back(std::move(view));
}

std::vector<WorkItemView> BuildWorkItemViews() {
  std::vector<WorkItemView> views;
  views.reserve(4);

  p2cccd::CertificateEngineConfig config;
  config.eps_time = 1.0e-5;
  config.eps_space = 1.0e-6;
  config.max_subdivision_depth = 32;

  p2cccd::ExactCertificateQuery point_triangle_collision;
  point_triangle_collision.work_item =
      MakeWorkItem(1, 101, 0, 11, 21, 0.00, 1.00, p2cccd::kFeatureFamilyPointTriangle, 0.91F);
  point_triangle_collision.config = config;
  point_triangle_collision.point_triangle_primitives =
      {MovingPointStaticTriangle(1.0, -3.0, 10, 20)};
  AppendEvaluatedQuery(&views, "Point-triangle collision", "PT", point_triangle_collision);

  p2cccd::ExactCertificateQuery edge_edge_collision;
  edge_edge_collision.work_item =
      MakeWorkItem(2, 102, 1, 12, 22, 0.00, 1.00, p2cccd::kFeatureFamilyEdgeEdge, 0.76F);
  edge_edge_collision.config = config;
  edge_edge_collision.edge_edge_primitives =
      {MovingEdgeThroughStaticEdge(1.0, -3.0, 30, 40)};
  AppendEvaluatedQuery(&views, "Edge-edge collision", "EE", edge_edge_collision);

  p2cccd::ExactCertificateQuery separation;
  separation.work_item =
      MakeWorkItem(3,
                   103,
                   2,
                   13,
                   23,
                   0.00,
                   1.00,
                   p2cccd::kFeatureFamilyPointTriangle | p2cccd::kFeatureFamilyEdgeEdge,
                   0.64F);
  separation.config = config;
  separation.point_triangle_primitives = {MovingPointStaticTriangle(2.0, 2.0, 50, 60)};
  separation.edge_edge_primitives = {MovingEdgeThroughStaticEdge(2.0, 2.0, 70, 80)};
  AppendEvaluatedQuery(&views, "Combined separation", "PT+EE", separation);

  p2cccd::ExactCertificateQuery missing_geometry;
  missing_geometry.work_item =
      MakeWorkItem(4, 104, 3, 14, 24, 0.00, 1.00, p2cccd::kFeatureFamilyPointTriangle, 0.42F);
  missing_geometry.config = config;
  AppendEvaluatedQuery(&views, "Missing geometry fallback", "PT", missing_geometry);

  return views;
}

std::string PillClassForStatus(p2cccd::CertificateStatus status) {
  switch (status) {
    case p2cccd::CertificateStatus::kCollision:
      return "bad";
    case p2cccd::CertificateStatus::kSeparation:
      return "ok";
    case p2cccd::CertificateStatus::kUndecided:
      return "warn";
  }
  return "warn";
}

void WriteHtmlPrefix(std::ostream& out) {
  out << "<!doctype html>\n"
      << "<html lang=\"en\">\n"
      << "<head>\n"
      << "<meta charset=\"utf-8\">\n"
      << "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
      << "<title>P2CCCD Stage 4 Certificate Overview</title>\n"
      << "<style>\n"
      << ":root{--bg:#0b1218;--panel:#14202b;--ink:#f7edd8;--muted:#aab5bd;"
      << "--line:#314353;--ok:#5fd3a5;--warn:#ffd166;--bad:#ff6b6b;--blue:#6bb7ff;}\n"
      << "body{margin:0;background:linear-gradient(135deg,#101923 0%,#0b1218 48%,#15100b 100%);"
      << "color:var(--ink);font-family:\"Iowan Old Style\",\"Palatino Linotype\",Georgia,serif;}\n"
      << "main{max-width:1200px;margin:0 auto;padding:38px 26px 54px;}\n"
      << "h1{font-size:42px;line-height:1;margin:0 0 10px;letter-spacing:-.035em;}\n"
      << "p{margin:0;color:var(--muted);line-height:1.55;font-size:16px;}\n"
      << ".grid{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin:26px 0;}\n"
      << ".card{background:linear-gradient(180deg,rgba(255,255,255,.065),rgba(255,255,255,.025));"
      << "border:1px solid var(--line);border-radius:22px;padding:18px;box-shadow:0 20px 44px rgba(0,0,0,.22);}\n"
      << ".card h2{font-size:16px;margin:0 0 10px;text-transform:uppercase;letter-spacing:.12em;color:#e4edf3;}\n"
      << ".metric{font-size:34px;font-weight:700;margin:2px 0;color:var(--ink);}\n"
      << ".label{font-size:13px;color:var(--muted);}\n"
      << ".viz{background:#0d1720;border:1px solid var(--line);border-radius:28px;padding:18px;margin:18px 0;}\n"
      << "svg{display:block;width:100%;height:auto;}\n"
      << "table{width:100%;border-collapse:collapse;margin-top:12px;font-size:14px;}\n"
      << "th,td{border-bottom:1px solid var(--line);padding:10px 8px;text-align:left;vertical-align:top;}\n"
      << "th{color:#e4edf3;font-size:12px;text-transform:uppercase;letter-spacing:.11em;}\n"
      << ".pill{display:inline-block;border-radius:999px;padding:3px 9px;font-size:12px;font-weight:700;}\n"
      << ".ok{background:rgba(95,211,165,.14);color:var(--ok);border:1px solid rgba(95,211,165,.46);}\n"
      << ".warn{background:rgba(255,209,102,.13);color:var(--warn);border:1px solid rgba(255,209,102,.45);}\n"
      << ".bad{background:rgba(255,107,107,.14);color:var(--bad);border:1px solid rgba(255,107,107,.45);}\n"
      << ".mono{font-family:\"Cascadia Code\",\"Fira Code\",Consolas,monospace;}\n"
      << ".note{border-left:3px solid var(--warn);padding:12px 14px;margin-top:16px;background:rgba(255,209,102,.07);color:#e6d8ae;}\n"
      << "@media(max-width:860px){.grid{grid-template-columns:1fr;}h1{font-size:32px;}main{padding:24px 14px 38px;}}\n"
      << "</style>\n"
      << "</head>\n"
      << "<body><main>\n";
}

void WriteCards(std::ostream& out, const std::vector<WorkItemView>& views) {
  const auto valid_results = std::count_if(views.begin(), views.end(), [](const WorkItemView& view) {
    return view.result_status.ok;
  });
  const auto undecided_results =
      std::count_if(views.begin(), views.end(), [](const WorkItemView& view) {
        return view.result.status == p2cccd::CertificateStatus::kUndecided;
      });

  out << "<div class=\"grid\">\n";
  out << "<section class=\"card\"><h2>Stage 4 / CPU+CUDA Exact</h2><div class=\"metric\">15/16</div>"
      << "<div class=\"label\">TODO items complete in Exact Local Certificate Engine</div></section>\n";
  out << "<section class=\"card\"><h2>Contract Validation</h2><div class=\"metric\">"
      << valid_results << "/" << views.size()
      << "</div><div class=\"label\">CertificateResult rows accepted by validators</div></section>\n";
  out << "<section class=\"card\"><h2>Certified Decisions</h2><div class=\"metric\">"
      << (views.size() - undecided_results)
      << "</div><div class=\"label\">collision or separation certificates emitted</div></section>\n";
  out << "</div>\n";
}

void WriteSvg(std::ostream& out, const std::vector<WorkItemView>& views) {
  constexpr double width = 1120.0;
  constexpr double height = 460.0;
  constexpr double left = 86.0;
  constexpr double right = 1040.0;
  constexpr double timeline_y = 110.0;
  constexpr double row_start_y = 180.0;
  constexpr double row_gap = 74.0;

  const auto project_t = [&](double t) {
    return left + std::clamp(t, 0.0, 1.0) * (right - left);
  };

  out << "<svg viewBox=\"0 0 " << width << ' ' << height
      << "\" role=\"img\" aria-label=\"P2CCCD stage 4 certificate placeholder results\">\n";
  out << "<rect x=\"0\" y=\"0\" width=\"" << width
      << "\" height=\"" << height << "\" rx=\"20\" fill=\"#0b141d\"/>\n";
  out << "<text x=\"30\" y=\"38\" fill=\"#f7edd8\" font-size=\"18\" font-weight=\"700\">"
      << "Stage 4 exact certificate path: CPU baseline behavior</text>\n";

  out << "<line x1=\"" << left << "\" y1=\"" << timeline_y << "\" x2=\"" << right
      << "\" y2=\"" << timeline_y << "\" stroke=\"#6bb7ff\" stroke-width=\"3\"/>\n";
  for (int tick = 0; tick <= 4; ++tick) {
    const double t = static_cast<double>(tick) / 4.0;
    const double x = project_t(t);
    out << "<line x1=\"" << x << "\" y1=\"" << (timeline_y - 9.0) << "\" x2=\""
        << x << "\" y2=\"" << (timeline_y + 9.0)
        << "\" stroke=\"#6bb7ff\" stroke-width=\"2\"/>\n";
    out << "<text x=\"" << (x - 10.0) << "\" y=\"" << (timeline_y - 18.0)
        << "\" fill=\"#aab5bd\" font-size=\"13\">t=" << FormatDouble(t, 2) << "</text>\n";
  }

  for (std::size_t i = 0; i < views.size(); ++i) {
    const WorkItemView& view = views[i];
    const double y = row_start_y + static_cast<double>(i) * row_gap;
    const double x0 = project_t(view.item.interval_t0);
    const double x1 = project_t(view.item.interval_t1);
    const double badge_x = std::min(x1 + 72.0, right - 150.0);
    out << "<text x=\"30\" y=\"" << (y + 5.0) << "\" fill=\"#f7edd8\" font-size=\"15\">"
        << HtmlEscape(view.family) << "</text>\n";
    out << "<rect x=\"" << x0 << "\" y=\"" << (y - 18.0) << "\" width=\""
        << std::max(2.0, x1 - x0) << "\" height=\"36\" rx=\"12\" fill=\"#ffd166\" fill-opacity=\"0.22\""
        << " stroke=\"#ffd166\" stroke-width=\"1.7\"/>\n";
    out << "<line x1=\"" << x1 << "\" y1=\"" << y << "\" x2=\"" << (badge_x - 4.0)
        << "\" y2=\"" << y << "\" stroke=\"#ffd166\" stroke-width=\"2\" stroke-dasharray=\"6 5\"/>\n";
    out << "<rect x=\"" << badge_x << "\" y=\"" << (y - 18.0)
        << "\" width=\"142\" height=\"36\" rx=\"18\" fill=\"#3a3020\" stroke=\"#ffd166\"/>\n";
    out << "<text x=\"" << (badge_x + 19.0) << "\" y=\"" << (y + 5.0)
        << "\" fill=\"#ffd166\" font-size=\"14\" font-weight=\"700\">"
        << HtmlEscape(CertificateStatusName(view.result.status)) << "</text>\n";
    out << "<text x=\"" << (x0 + 10.0) << "\" y=\"" << (y + 5.0)
        << "\" fill=\"#f7edd8\" font-size=\"14\">" << HtmlEscape(view.label) << "</text>\n";
  }

  out << "<g transform=\"translate(30,394)\">\n";
  out << "<rect x=\"0\" y=\"0\" width=\"1060\" height=\"42\" rx=\"12\" fill=\"rgba(255,209,102,.08)\" stroke=\"#6f5a2e\"/>\n";
  out << "<text x=\"16\" y=\"26\" fill=\"#e6d8ae\" font-size=\"14\">"
      << "PT/EE CPU interval oracles, work queue, audit, refinement guard, and optional CUDA batch kernels are now implemented; stronger root isolation remains future work.</text>\n";
  out << "</g>\n";
  out << "</svg>\n";
}

void WriteTable(std::ostream& out, const std::vector<WorkItemView>& views) {
  out << "<section class=\"card\"><h2>Exact Work Item Results</h2>\n";
  out << "<table><thead><tr><th>Work</th><th>Interval</th><th>Input Contract</th>"
      << "<th>Certificate Status</th><th>Result Contract</th><th>Reason / Next</th></tr></thead><tbody>\n";
  for (const WorkItemView& view : views) {
    out << "<tr><td>" << HtmlEscape(view.label) << "<br><span class=\"mono\">family="
        << HtmlEscape(view.family) << ", patches=" << view.item.patch_a_id << "/"
        << view.item.patch_b_id << "</span></td><td class=\"mono\">["
        << FormatDouble(view.item.interval_t0, 2) << ", "
        << FormatDouble(view.item.interval_t1, 2) << "]</td><td><span class=\"pill "
        << (view.item_status.ok ? "ok" : "bad") << "\">"
        << (view.item_status.ok ? "PASS" : "FAIL") << "</span></td><td><span class=\"pill "
        << PillClassForStatus(view.result.status) << "\">"
        << HtmlEscape(CertificateStatusName(view.result.status))
        << "</span></td><td><span class=\"pill "
        << (view.result_status.ok ? "ok" : "bad") << "\">"
        << (view.result_status.ok ? "PASS" : "FAIL") << "</span></td><td class=\"mono\">"
        << "reason_code=" << view.result.reason_code << "<br>next="
        << HtmlEscape(RefinementModeName(view.result.next_refinement_mode)) << "</td></tr>\n";
  }
  out << "</tbody></table></section>\n";
}

void WriteTodoTable(std::ostream& out) {
  struct TodoRow {
    int id = 0;
    const char* status = "";
    const char* content = "";
  };
  const std::vector<TodoRow> rows = {
      {41, "done", "CertificateEngine interface exists"},
      {42, "done", "CPU point-triangle interval oracle"},
      {43, "done", "CPU edge-edge interval oracle"},
      {44, "done", "Recursive interval subdivision"},
      {45, "done", "Collision certificate with witness and TOI upper bound"},
      {46, "done", "Separation certificate with feature coverage and safe margin"},
      {47, "done", "Undecided reason and refinement mode"},
      {48, "done", "CPU exact work queue processing"},
      {49, "done", "Audit log emission"},
      {50, "done", "CPU regression cases"},
      {51, "done", "Optional CUDA batched point-triangle kernel"},
      {52, "done", "Optional CUDA batched edge-edge kernel"},
      {53, "done", "CUDA exact result transfer to host"},
      {54, "done", "CPU versus CUDA cross-check"},
      {55, "done", "Illegal termination coverage guard"},
      {56, "done", "Conservative TOI and interval refinement heuristics"},
  };

  out << "<section class=\"card\"><h2>Stage 4 Implementation Coverage</h2>\n";
  out << "<table><thead><tr><th>ID</th><th>Status</th><th>Scope</th></tr></thead><tbody>\n";
  for (const TodoRow& row : rows) {
    out << "<tr><td class=\"mono\">" << row.id << "</td><td><span class=\"pill "
        << (std::string(row.status) == "done" ? "ok" : "warn") << "\">"
        << row.status << "</span></td><td>" << HtmlEscape(row.content) << "</td></tr>\n";
  }
  out << "</tbody></table></section>\n";
}

void WriteHtml(std::ostream& out, const std::vector<WorkItemView>& views) {
  WriteHtmlPrefix(out);
  out << "<h1>P2CCCD Stage 4 Result View</h1>\n";
  out << "<p>This page is generated from the current C++ certificate module. It runs point-triangle and edge-edge CPU interval checks through the CertificateEngine.</p>\n";
  WriteCards(out, views);
  out << "<section class=\"viz\">\n";
  WriteSvg(out, views);
  out << "</section>\n";
  out << "<div class=\"note\">Current result is a CPU baseline plus optional CUDA batch path. It is conservative and test-covered for simple linear feature trajectories, but it is not yet the final high-order root-isolation CCD kernel.</div>\n";
  out << "<div class=\"grid\">\n";
  WriteTable(out, views);
  WriteTodoTable(out);
  out << "<section class=\"card\"><h2>Remaining Exact Work</h2>"
      << "<table><tbody>"
      << "<tr><td>Root isolation</td><td>replace sampled subdivision with stronger algebraic TOI isolation</td></tr>"
      << "<tr><td>Degeneracy suite</td><td>expand coplanar, parallel, zero-length, and grazing cases</td></tr>"
      << "<tr><td>Queue integration</td><td>wire exact queue into proposal scheduler and end-to-end monotonicity tests</td></tr>"
      << "<tr><td>Performance</td><td>benchmark CPU vs CUDA batch throughput on real candidate traces</td></tr>"
      << "</tbody></table></section>\n";
  out << "</div>\n";
  out << "</main></body></html>\n";
}

}  // namespace

int main(int argc, char** argv) {
  const std::filesystem::path output_path =
      argc >= 2 ? std::filesystem::path(argv[1])
                : std::filesystem::path("outputs/stage_4_certificate_overview.html");

  const std::vector<WorkItemView> views = BuildWorkItemViews();
  if (!output_path.parent_path().empty()) {
    std::filesystem::create_directories(output_path.parent_path());
  }

  std::ofstream out(output_path);
  if (!out) {
    std::cerr << "failed to open output path: " << output_path.string() << '\n';
    return 1;
  }

  WriteHtml(out, views);
  std::cout << "wrote " << std::filesystem::absolute(output_path).string() << '\n';
  return 0;
}
