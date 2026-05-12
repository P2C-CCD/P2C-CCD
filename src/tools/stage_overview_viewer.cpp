#include "common/validators.h"
#include "geometry/motion_utils.h"
#include "geometry/patch.h"
#include "rt_candidate/candidate_generator.h"
#include "rt_candidate/proxy_scene.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

using Vec3 = std::array<double, 3>;

constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr double kSvgWidth = 1120.0;
constexpr double kSvgHeight = 540.0;
constexpr double kSvgPadding = 56.0;

struct ValidationRow {
  std::string contract_name;
  bool ok = false;
  std::string message;
};

struct PatchPath {
  std::uint32_t object_id = 0;
  std::uint32_t patch_id = 0;
  std::string color;
  std::vector<Vec3> points;
};

struct Bounds2D {
  double min_x = 0.0;
  double max_x = 1.0;
  double min_y = 0.0;
  double max_y = 1.0;
  bool initialized = false;

  void Add(double x, double y) {
    if (!initialized) {
      min_x = max_x = x;
      min_y = max_y = y;
      initialized = true;
      return;
    }
    min_x = std::min(min_x, x);
    max_x = std::max(max_x, x);
    min_y = std::min(min_y, y);
    max_y = std::max(max_y, y);
  }
};

struct Projector {
  Bounds2D bounds;
  double scale = 1.0;
  double offset_x = 0.0;
  double offset_y = 0.0;

  std::array<double, 2> Project(const Vec3& point) const {
    return {offset_x + (point[0] - bounds.min_x) * scale,
            offset_y + (bounds.max_y - point[1]) * scale};
  }

  std::array<double, 2> Project(double x, double y) const {
    return Project(Vec3{x, y, 0.0});
  }
};

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

std::string ProxyTypeName(p2cccd::ProxyType proxy_type) {
  switch (proxy_type) {
    case p2cccd::ProxyType::kSweptAabb:
      return "swept AABB";
    case p2cccd::ProxyType::kCapsule:
      return "capsule";
    case p2cccd::ProxyType::kUnknown:
      return "unknown";
  }
  return "invalid";
}

p2cccd::Patch MakePatch(std::uint32_t patch_id,
                        double x,
                        double y,
                        double radius,
                        double area = 1.0) {
  p2cccd::Patch patch;
  patch.patch_id = patch_id;
  patch.triangle_ids = {patch_id * 2U, patch_id * 2U + 1U};
  patch.triangle_count = static_cast<std::uint32_t>(patch.triangle_ids.size());
  patch.area = area;
  patch.local_center = {x, y, 0.0};
  patch.radius = radius;
  return patch;
}

p2cccd::MotionSegment MakePlanarMotion(double tx0,
                                       double ty0,
                                       double tx1,
                                       double ty1,
                                       double angle_t1) {
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

p2cccd::ProxySceneBuildInput MakeDemoSceneInput() {
  p2cccd::ProxySceneBuildInput input;
  input.query_id = 3103;

  p2cccd::ProxyObjectBuildInput object_a;
  object_a.object_id = 10;
  object_a.proxy_type = p2cccd::ProxyType::kSweptAabb;
  object_a.patches = {MakePatch(1, 0.00, 0.00, 0.26, 0.72),
                      MakePatch(2, 2.20, 0.10, 0.22, 0.54)};
  object_a.motion_segments = {MakePlanarMotion(0.00, 0.00, 0.80, 0.25, 0.26 * kPi)};
  object_a.slabs_per_motion_segment = 4;
  object_a.eps_proxy = 0.035;

  p2cccd::ProxyObjectBuildInput object_b;
  object_b.object_id = 20;
  object_b.proxy_type = p2cccd::ProxyType::kCapsule;
  object_b.patches = {MakePatch(3, 0.34, 0.08, 0.24, 0.62),
                      MakePatch(4, 4.45, 0.00, 0.18, 0.38)};
  object_b.motion_segments = {MakePlanarMotion(0.08, 0.05, 0.52, 0.20, 0.08 * kPi)};
  object_b.slabs_per_motion_segment = 4;
  object_b.eps_proxy = 0.035;

  p2cccd::ProxyObjectBuildInput object_c;
  object_c.object_id = 30;
  object_c.proxy_type = p2cccd::ProxyType::kSweptAabb;
  object_c.patches = {MakePatch(5, 2.55, 0.72, 0.25, 0.66)};
  object_c.motion_segments = {MakePlanarMotion(0.00, 0.00, 0.28, 0.12, 0.02 * kPi)};
  object_c.slabs_per_motion_segment = 4;
  object_c.eps_proxy = 0.035;

  input.objects = {object_a, object_b, object_c};
  return input;
}

void AppendValidationRow(std::vector<ValidationRow>* rows,
                         std::string name,
                         const p2cccd::Status& status) {
  rows->push_back({std::move(name), status.ok, status.ok ? "pass" : status.message});
}

std::vector<ValidationRow> BuildValidationRows(const p2cccd::CandidateGenerationResult& result) {
  std::vector<ValidationRow> rows;

  p2cccd::CandidateRecord candidate;
  if (!result.candidates.empty()) {
    candidate = result.candidates.front();
  } else {
    candidate.candidate_id = 1;
    candidate.query_id = 3103;
    candidate.slab_id = 0;
    candidate.object_a_id = 10;
    candidate.patch_a_id = 1;
    candidate.object_b_id = 20;
    candidate.patch_b_id = 3;
    candidate.proxy_type_a = p2cccd::ProxyType::kSweptAabb;
    candidate.proxy_type_b = p2cccd::ProxyType::kCapsule;
    candidate.rt_hit_count = 1;
    candidate.motion_bound = {0.1F, 0.1F, 0.1F, 0.4F};
  }
  AppendValidationRow(&rows, "CandidateRecord", p2cccd::ValidateCandidateRecord(candidate));

  p2cccd::ProposalOutput proposal;
  proposal.candidate_id = candidate.candidate_id;
  proposal.interval_scores = {0.62F, 0.38F, 0.20F, 0.08F, 0.03F, 0.02F, 0.01F, 0.00F};
  proposal.family_scores = {0.70F, 0.44F, 0.33F, 0.12F, 0.04F, 0.01F, 0.00F, 0.00F};
  proposal.priority_score = 0.82F;
  proposal.cost_score = 0.24F;
  proposal.uncertainty_score = 0.18F;
  AppendValidationRow(&rows, "ProposalOutput", p2cccd::ValidateProposalOutput(proposal));

  p2cccd::ExactWorkItem item;
  item.work_item_id = 7001;
  item.parent_candidate_id = candidate.candidate_id;
  item.query_id = candidate.query_id;
  item.slab_id = candidate.slab_id;
  item.patch_a_id = candidate.patch_a_id;
  item.patch_b_id = candidate.patch_b_id;
  item.interval_t0 = 0.0;
  item.interval_t1 = 0.25;
  item.feature_family_mask = 0x3U;
  item.priority_score = proposal.priority_score;
  item.source = p2cccd::ProposalSource::kRaw;
  AppendValidationRow(&rows, "ExactWorkItem", p2cccd::ValidateExactWorkItem(item));

  p2cccd::CertificateResult certificate;
  certificate.work_item_id = item.work_item_id;
  certificate.query_id = item.query_id;
  certificate.status = p2cccd::CertificateStatus::kSeparation;
  certificate.interval_t0 = item.interval_t0;
  certificate.interval_t1 = item.interval_t1;
  certificate.toi_upper = item.interval_t1;
  certificate.safe_margin_lb = 0.0025;
  certificate.covered_feature_mask = item.feature_family_mask;
  certificate.eps_time = 1.0e-4;
  certificate.eps_space = 1.0e-6;
  AppendValidationRow(&rows, "CertificateResult", p2cccd::ValidateCertificateResult(certificate));

  p2cccd::AuditLogRow audit;
  audit.event_id = 9001;
  audit.query_id = item.query_id;
  audit.candidate_id = candidate.candidate_id;
  audit.work_item_id = item.work_item_id;
  audit.stage = p2cccd::AuditStage::kRt;
  audit.action = 1;
  audit.interval_t0 = item.interval_t0;
  audit.interval_t1 = item.interval_t1;
  audit.timestamp_us = 123456;
  audit.aux_value0 = static_cast<double>(result.candidates.size());
  audit.aux_value1 = result.timing.total_ms;
  AppendValidationRow(&rows, "AuditLogRow", p2cccd::ValidateAuditLogRow(audit));

  p2cccd::BenchmarkRow benchmark;
  benchmark.query_count = 1;
  benchmark.fn_count = 0;
  benchmark.fp_count = result.candidates.size();
  benchmark.candidate_recall = 1.0;
  benchmark.avg_candidates = static_cast<double>(result.candidates.size());
  benchmark.avg_exact_evals = 1.0;
  benchmark.avg_subdivision_depth = 0.0;
  benchmark.fallback_ratio = 0.0;
  benchmark.rt_ms = result.timing.trace_ms;
  benchmark.proposal_ms = 0.0;
  benchmark.exact_ms = 0.0;
  benchmark.total_ms = result.timing.total_ms;
  benchmark.qps = result.timing.total_ms > 0.0 ? 1000.0 / result.timing.total_ms : 0.0;
  AppendValidationRow(&rows, "BenchmarkRow", p2cccd::ValidateBenchmarkRow(benchmark));

  return rows;
}

std::string ObjectColor(std::uint32_t object_id) {
  switch (object_id) {
    case 10:
      return "#ffb454";
    case 20:
      return "#62d2a2";
    case 30:
      return "#64b5f6";
    default:
      return "#d8dee9";
  }
}

p2cccd::Status BuildPatchPaths(const p2cccd::ProxySceneBuildInput& input,
                               std::vector<PatchPath>* paths) {
  paths->clear();
  for (const p2cccd::ProxyObjectBuildInput& object : input.objects) {
    if (object.motion_segments.empty()) {
      return p2cccd::Status::Error("object motion segment is missing");
    }
    const p2cccd::MotionSegment& motion = object.motion_segments.front();
    for (const p2cccd::Patch& patch : object.patches) {
      PatchPath path;
      path.object_id = object.object_id;
      path.patch_id = patch.patch_id;
      path.color = ObjectColor(object.object_id);
      for (std::uint32_t i = 0; i <= 32; ++i) {
        const double t = static_cast<double>(i) / 32.0;
        p2cccd::PoseSample pose;
        Vec3 point{};
        if (auto status = p2cccd::InterpolateRigidMotion(motion, t, &pose); !status.ok) {
          return status;
        }
        if (auto status = p2cccd::TransformPoint(pose, patch.local_center, &point); !status.ok) {
          return status;
        }
        path.points.push_back(point);
      }
      paths->push_back(std::move(path));
    }
  }
  return p2cccd::Status::Ok();
}

Bounds2D ComputeBounds(const p2cccd::ProxyScene& scene, const std::vector<PatchPath>& paths) {
  Bounds2D bounds;
  for (const p2cccd::ProxyPrimitive& primitive : scene.primitives) {
    bounds.Add(primitive.bounds.min[0], primitive.bounds.min[1]);
    bounds.Add(primitive.bounds.max[0], primitive.bounds.max[1]);
  }
  for (const PatchPath& path : paths) {
    for (const Vec3& point : path.points) {
      bounds.Add(point[0], point[1]);
    }
  }
  if (!bounds.initialized) {
    bounds.Add(-1.0, -1.0);
    bounds.Add(1.0, 1.0);
  }
  const double margin_x = std::max(0.25, 0.10 * (bounds.max_x - bounds.min_x));
  const double margin_y = std::max(0.25, 0.10 * (bounds.max_y - bounds.min_y));
  bounds.min_x -= margin_x;
  bounds.max_x += margin_x;
  bounds.min_y -= margin_y;
  bounds.max_y += margin_y;
  return bounds;
}

Projector MakeProjector(const Bounds2D& bounds) {
  Projector projector;
  projector.bounds = bounds;
  const double width = std::max(1.0, bounds.max_x - bounds.min_x);
  const double height = std::max(1.0, bounds.max_y - bounds.min_y);
  projector.scale =
      std::min((kSvgWidth - 2.0 * kSvgPadding) / width,
               (kSvgHeight - 2.0 * kSvgPadding) / height);
  const double drawn_width = width * projector.scale;
  const double drawn_height = height * projector.scale;
  projector.offset_x = 0.5 * (kSvgWidth - drawn_width);
  projector.offset_y = 0.5 * (kSvgHeight - drawn_height);
  return projector;
}

Vec3 PrimitiveCenter(const p2cccd::ProxyPrimitive& primitive) {
  return {0.5 * (primitive.bounds.min[0] + primitive.bounds.max[0]),
          0.5 * (primitive.bounds.min[1] + primitive.bounds.max[1]),
          0.5 * (primitive.bounds.min[2] + primitive.bounds.max[2])};
}

std::string PolylinePoints(const Projector& projector, const std::vector<Vec3>& points) {
  std::ostringstream stream;
  for (std::size_t i = 0; i < points.size(); ++i) {
    const auto projected = projector.Project(points[i]);
    if (i != 0) {
      stream << ' ';
    }
    stream << FormatDouble(projected[0], 1) << ',' << FormatDouble(projected[1], 1);
  }
  return stream.str();
}

void WriteHtmlPrefix(std::ostream& out) {
  out << "<!doctype html>\n"
      << "<html lang=\"en\">\n"
      << "<head>\n"
      << "<meta charset=\"utf-8\">\n"
      << "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
      << "<title>P2CCCD Stage 1-3 Overview</title>\n"
      << "<style>\n"
      << ":root{--bg:#101922;--panel:#162231;--ink:#f6f0dc;--muted:#aeb8c2;"
      << "--line:#314257;--accent:#ffb454;--ok:#62d2a2;--warn:#ffd166;--bad:#ff6b6b;}\n"
      << "body{margin:0;background:radial-gradient(circle at 18% 8%,#223b4c 0,#101922 34%,#091016 100%);"
      << "color:var(--ink);font-family:\"Iowan Old Style\",\"Palatino Linotype\",Georgia,serif;}\n"
      << "main{max-width:1240px;margin:0 auto;padding:38px 28px 54px;}\n"
      << "h1{font-size:42px;line-height:1;margin:0 0 10px;letter-spacing:-.03em;}\n"
      << "p{color:var(--muted);font-size:16px;line-height:1.55;margin:0;}\n"
      << ".grid{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin:26px 0;}\n"
      << ".card{background:linear-gradient(180deg,rgba(255,255,255,.06),rgba(255,255,255,.025));"
      << "border:1px solid var(--line);border-radius:22px;padding:18px;box-shadow:0 18px 42px rgba(0,0,0,.22);}\n"
      << ".card h2{font-size:17px;margin:0 0 10px;text-transform:uppercase;letter-spacing:.12em;color:#dce7ef;}\n"
      << ".metric{font-size:34px;font-weight:700;margin:4px 0;color:var(--ink);}\n"
      << ".label{font-size:13px;color:var(--muted);}\n"
      << ".viz{background:#0e1720;border:1px solid var(--line);border-radius:28px;padding:18px;margin-top:18px;}\n"
      << "svg{width:100%;height:auto;display:block;}\n"
      << "table{width:100%;border-collapse:collapse;margin-top:12px;font-size:14px;}\n"
      << "th,td{border-bottom:1px solid var(--line);padding:10px 8px;text-align:left;}\n"
      << "th{color:#dce7ef;font-size:12px;text-transform:uppercase;letter-spacing:.11em;}\n"
      << ".pill{display:inline-block;border-radius:999px;padding:3px 9px;font-size:12px;font-weight:700;}\n"
      << ".pass{background:rgba(98,210,162,.14);color:var(--ok);border:1px solid rgba(98,210,162,.45);}\n"
      << ".fail{background:rgba(255,107,107,.14);color:var(--bad);border:1px solid rgba(255,107,107,.45);}\n"
      << ".legend{display:flex;gap:18px;flex-wrap:wrap;margin-top:12px;color:var(--muted);font-size:14px;}\n"
      << ".swatch{display:inline-block;width:12px;height:12px;border-radius:3px;margin-right:7px;vertical-align:-1px;}\n"
      << ".mono{font-family:\"Cascadia Code\",\"Fira Code\",Consolas,monospace;}\n"
      << "@media(max-width:860px){.grid{grid-template-columns:1fr;}h1{font-size:32px;}main{padding:24px 14px 38px;}}\n"
      << "</style>\n"
      << "</head>\n"
      << "<body><main>\n";
}

void WriteStageCards(std::ostream& out,
                     const std::vector<ValidationRow>& rows,
                     const p2cccd::ProxyScene& scene,
                     const p2cccd::CandidateGenerationResult& result) {
  const std::uint64_t validation_pass_count =
      static_cast<std::uint64_t>(std::count_if(rows.begin(), rows.end(), [](const auto& row) {
        return row.ok;
      }));

  out << "<div class=\"grid\">\n";
  out << "<section class=\"card\"><h2>Stage 1 / Contracts</h2><div class=\"metric\">"
      << validation_pass_count << "/" << rows.size()
      << "</div><div class=\"label\">C++ runtime contract validators passing</div></section>\n";
  out << "<section class=\"card\"><h2>Stage 2 / Geometry Core</h2><div class=\"metric\">"
      << scene.primitives.size()
      << "</div><div class=\"label\">slab-local conservative proxy primitives</div></section>\n";
  out << "<section class=\"card\"><h2>Stage 3 / RT Candidates</h2><div class=\"metric\">"
      << result.candidates.size()
      << "</div><div class=\"label\">compact candidate records from "
      << HtmlEscape(result.backend_name) << "</div></section>\n";
  out << "</div>\n";
}

void WriteSceneSvg(std::ostream& out,
                   const p2cccd::ProxyScene& scene,
                   const p2cccd::CandidateGenerationResult& result,
                   const std::vector<PatchPath>& paths) {
  const Projector projector = MakeProjector(ComputeBounds(scene, paths));
  out << "<svg viewBox=\"0 0 " << FormatDouble(kSvgWidth, 0) << ' '
      << FormatDouble(kSvgHeight, 0) << "\" role=\"img\" aria-label=\"P2CCCD proxy and candidate scene\">\n";
  out << "<rect x=\"0\" y=\"0\" width=\"" << FormatDouble(kSvgWidth, 0) << "\" height=\""
      << FormatDouble(kSvgHeight, 0) << "\" fill=\"#0b141d\" rx=\"20\"/>\n";
  out << "<g opacity=\"0.18\" stroke=\"#dce7ef\" stroke-width=\"1\">\n";
  for (std::uint32_t grid = 0; grid <= 10; ++grid) {
    const double x = 40.0 + grid * (kSvgWidth - 80.0) / 10.0;
    out << "<line x1=\"" << FormatDouble(x, 1) << "\" y1=\"42\" x2=\""
        << FormatDouble(x, 1) << "\" y2=\"" << FormatDouble(kSvgHeight - 42.0, 1) << "\"/>\n";
  }
  for (std::uint32_t grid = 0; grid <= 5; ++grid) {
    const double y = 42.0 + grid * (kSvgHeight - 84.0) / 5.0;
    out << "<line x1=\"40\" y1=\"" << FormatDouble(y, 1) << "\" x2=\""
        << FormatDouble(kSvgWidth - 40.0, 1) << "\" y2=\"" << FormatDouble(y, 1) << "\"/>\n";
  }
  out << "</g>\n";

  out << "<g id=\"proxies\">\n";
  for (const p2cccd::ProxyPrimitive& primitive : scene.primitives) {
    const std::string color = ObjectColor(primitive.object_id);
    const auto min_corner = projector.Project(primitive.bounds.min[0], primitive.bounds.min[1]);
    const auto max_corner = projector.Project(primitive.bounds.max[0], primitive.bounds.max[1]);
    const double x = std::min(min_corner[0], max_corner[0]);
    const double y = std::min(min_corner[1], max_corner[1]);
    const double width = std::abs(max_corner[0] - min_corner[0]);
    const double height = std::abs(max_corner[1] - min_corner[1]);
    const double opacity = 0.11 + 0.035 * static_cast<double>(primitive.slab_id);
    out << "<rect x=\"" << FormatDouble(x, 1) << "\" y=\"" << FormatDouble(y, 1)
        << "\" width=\"" << FormatDouble(width, 1) << "\" height=\"" << FormatDouble(height, 1)
        << "\" fill=\"" << color << "\" fill-opacity=\"" << FormatDouble(opacity, 3)
        << "\" stroke=\"" << color << "\" stroke-opacity=\"0.46\" stroke-width=\"1.3\" rx=\"5\">\n";
    out << "<title>object " << primitive.object_id << ", patch " << primitive.patch_id
        << ", slab " << primitive.slab_id << ", " << ProxyTypeName(primitive.proxy_type)
        << "</title></rect>\n";

    if (primitive.proxy_type == p2cccd::ProxyType::kCapsule) {
      const auto p0 = projector.Project(primitive.capsule.endpoint0);
      const auto p1 = projector.Project(primitive.capsule.endpoint1);
      out << "<line x1=\"" << FormatDouble(p0[0], 1) << "\" y1=\""
          << FormatDouble(p0[1], 1) << "\" x2=\"" << FormatDouble(p1[0], 1)
          << "\" y2=\"" << FormatDouble(p1[1], 1) << "\" stroke=\"" << color
          << "\" stroke-width=\"" << FormatDouble(std::max(2.0, 2.0 * primitive.capsule.radius * projector.scale), 1)
          << "\" stroke-opacity=\"0.25\" stroke-linecap=\"round\"/>\n";
      out << "<line x1=\"" << FormatDouble(p0[0], 1) << "\" y1=\""
          << FormatDouble(p0[1], 1) << "\" x2=\"" << FormatDouble(p1[0], 1)
          << "\" y2=\"" << FormatDouble(p1[1], 1) << "\" stroke=\"" << color
          << "\" stroke-width=\"2.2\" stroke-opacity=\"0.9\" stroke-linecap=\"round\"/>\n";
    }
  }
  out << "</g>\n";

  out << "<g id=\"candidate-links\" stroke=\"#ffd166\" stroke-width=\"2.4\" stroke-opacity=\"0.82\" fill=\"none\">\n";
  for (const p2cccd::RawCandidateHit& hit : result.raw_buffer.hits) {
    if (hit.proxy_a_index >= scene.primitives.size() || hit.proxy_b_index >= scene.primitives.size()) {
      continue;
    }
    const Vec3 center_a = PrimitiveCenter(scene.primitives[hit.proxy_a_index]);
    const Vec3 center_b = PrimitiveCenter(scene.primitives[hit.proxy_b_index]);
    const auto a = projector.Project(center_a);
    const auto b = projector.Project(center_b);
    out << "<line x1=\"" << FormatDouble(a[0], 1) << "\" y1=\"" << FormatDouble(a[1], 1)
        << "\" x2=\"" << FormatDouble(b[0], 1) << "\" y2=\"" << FormatDouble(b[1], 1)
        << "\"><title>raw candidate, slab " << hit.slab_id << "</title></line>\n";
  }
  out << "</g>\n";

  out << "<g id=\"paths\" fill=\"none\" stroke-linecap=\"round\" stroke-linejoin=\"round\">\n";
  for (const PatchPath& path : paths) {
    if (path.points.empty()) {
      continue;
    }
    out << "<polyline points=\"" << PolylinePoints(projector, path.points) << "\" stroke=\""
        << path.color << "\" stroke-width=\"4\" stroke-opacity=\"0.92\"/>\n";
    const auto start = projector.Project(path.points.front());
    const auto end = projector.Project(path.points.back());
    out << "<circle cx=\"" << FormatDouble(start[0], 1) << "\" cy=\"" << FormatDouble(start[1], 1)
        << "\" r=\"5\" fill=\"#0b141d\" stroke=\"" << path.color
        << "\" stroke-width=\"2\"><title>object " << path.object_id << ", patch "
        << path.patch_id << " t0</title></circle>\n";
    out << "<circle cx=\"" << FormatDouble(end[0], 1) << "\" cy=\"" << FormatDouble(end[1], 1)
        << "\" r=\"6\" fill=\"" << path.color << "\"><title>object " << path.object_id
        << ", patch " << path.patch_id << " t1</title></circle>\n";
  }
  out << "</g>\n";

  out << "<text x=\"26\" y=\"34\" fill=\"#f6f0dc\" font-size=\"18\" font-weight=\"700\">"
      << "Stage 2 proxy bounds + Stage 3 candidate links</text>\n";
  out << "<text x=\"26\" y=\"" << FormatDouble(kSvgHeight - 18.0, 1)
      << "\" fill=\"#aeb8c2\" font-size=\"13\">Transparent rectangles: slab-local AABBs. Thick green lines: capsule axes. Yellow links: raw candidate hits.</text>\n";
  out << "</svg>\n";
}

void WriteValidationTable(std::ostream& out, const std::vector<ValidationRow>& rows) {
  out << "<section class=\"card\"><h2>Stage 1 Contract Checks</h2>\n";
  out << "<table><thead><tr><th>Contract</th><th>Status</th><th>Validator result</th></tr></thead><tbody>\n";
  for (const ValidationRow& row : rows) {
    out << "<tr><td class=\"mono\">" << HtmlEscape(row.contract_name) << "</td><td><span class=\"pill "
        << (row.ok ? "pass" : "fail") << "\">" << (row.ok ? "PASS" : "FAIL")
        << "</span></td><td>" << HtmlEscape(row.message) << "</td></tr>\n";
  }
  out << "</tbody></table></section>\n";
}

void WriteCandidateTable(std::ostream& out, const p2cccd::CandidateGenerationResult& result) {
  out << "<section class=\"card\"><h2>Stage 3 Candidate Records</h2>\n";
  out << "<table><thead><tr><th>ID</th><th>Slab</th><th>Object/Patch Pair</th><th>Proxy Pair</th><th>RT Hits</th></tr></thead><tbody>\n";
  for (const p2cccd::CandidateRecord& candidate : result.candidates) {
    out << "<tr><td class=\"mono\">" << candidate.candidate_id << "</td><td>"
        << candidate.slab_id << "</td><td class=\"mono\">" << candidate.object_a_id << "/"
        << candidate.patch_a_id << " -> " << candidate.object_b_id << "/"
        << candidate.patch_b_id << "</td><td>" << ProxyTypeName(candidate.proxy_type_a)
        << " / " << ProxyTypeName(candidate.proxy_type_b) << "</td><td>"
        << candidate.rt_hit_count << "</td></tr>\n";
  }
  out << "</tbody></table></section>\n";
}

void WriteDensityStats(std::ostream& out, const p2cccd::CandidateGenerationResult& result) {
  const p2cccd::CandidateDensityStats& density = result.density;
  out << "<section class=\"card\"><h2>Density And Timing</h2>\n";
  out << "<table><tbody>\n";
  out << "<tr><td>backend</td><td class=\"mono\">" << HtmlEscape(density.backend_name)
      << "</td></tr>\n";
  out << "<tr><td>proxy_count</td><td>" << density.proxy_count << "</td></tr>\n";
  out << "<tr><td>object_count</td><td>" << density.object_count << "</td></tr>\n";
  out << "<tr><td>slab_count</td><td>" << density.slab_count << "</td></tr>\n";
  out << "<tr><td>cross_object_same_slab_pair_count</td><td>"
      << density.cross_object_same_slab_pair_count << "</td></tr>\n";
  out << "<tr><td>raw_hit_count</td><td>" << density.raw_hit_count << "</td></tr>\n";
  out << "<tr><td>compact_candidate_count</td><td>" << density.compact_candidate_count
      << "</td></tr>\n";
  out << "<tr><td>aabb_overlap_ratio</td><td>" << FormatDouble(density.aabb_overlap_ratio, 4)
      << "</td></tr>\n";
  out << "<tr><td>trace_ms</td><td>" << FormatDouble(result.timing.trace_ms, 6) << "</td></tr>\n";
  out << "<tr><td>compact_ms</td><td>" << FormatDouble(result.timing.compact_ms, 6)
      << "</td></tr>\n";
  out << "<tr><td>total_ms</td><td>" << FormatDouble(result.timing.total_ms, 6) << "</td></tr>\n";
  out << "</tbody></table></section>\n";
}

void WriteHtml(std::ostream& out,
               const std::vector<ValidationRow>& rows,
               const p2cccd::ProxyScene& scene,
               const p2cccd::CandidateGenerationResult& result,
               const std::vector<PatchPath>& paths) {
  WriteHtmlPrefix(out);
  out << "<h1>P2CCCD Stage 1-3 Result View</h1>\n";
  out << "<p>This page is generated from the current C++ implementation. It exercises runtime contracts, geometry proxy construction, and the CPU reference RT-candidate path in one deterministic scene.</p>\n";
  WriteStageCards(out, rows, scene, result);
  out << "<section class=\"viz\">\n";
  WriteSceneSvg(out, scene, result, paths);
  out << "<div class=\"legend\">"
      << "<span><span class=\"swatch\" style=\"background:#ffb454\"></span>object 10 / swept AABB</span>"
      << "<span><span class=\"swatch\" style=\"background:#62d2a2\"></span>object 20 / capsule</span>"
      << "<span><span class=\"swatch\" style=\"background:#64b5f6\"></span>object 30 / swept AABB</span>"
      << "<span><span class=\"swatch\" style=\"background:#ffd166\"></span>candidate links</span>"
      << "</div>\n";
  out << "</section>\n";
  out << "<div class=\"grid\">\n";
  WriteValidationTable(out, rows);
  WriteDensityStats(out, result);
  WriteCandidateTable(out, result);
  out << "</div>\n";
  out << "</main></body></html>\n";
}

}  // namespace

int main(int argc, char** argv) {
  const std::filesystem::path output_path =
      argc >= 2 ? std::filesystem::path(argv[1])
                : std::filesystem::path("outputs/stage_1_3_overview.html");

  const p2cccd::ProxySceneBuildInput input = MakeDemoSceneInput();

  p2cccd::ProxyScene scene;
  if (auto status = p2cccd::BuildProxyScene(input, &scene); !status.ok) {
    std::cerr << "BuildProxyScene failed: " << status.message << '\n';
    return 1;
  }

  p2cccd::CandidateGenerator generator;
  p2cccd::CandidateGenerationResult result;
  if (auto status = generator.GenerateCandidates(scene, scene.query_id, &result); !status.ok) {
    std::cerr << "GenerateCandidates failed: " << status.message << '\n';
    return 1;
  }

  std::vector<PatchPath> paths;
  if (auto status = BuildPatchPaths(input, &paths); !status.ok) {
    std::cerr << "BuildPatchPaths failed: " << status.message << '\n';
    return 1;
  }

  const std::vector<ValidationRow> validation_rows = BuildValidationRows(result);

  if (!output_path.parent_path().empty()) {
    std::filesystem::create_directories(output_path.parent_path());
  }
  std::ofstream out(output_path);
  if (!out) {
    std::cerr << "failed to open output path: " << output_path.string() << '\n';
    return 1;
  }

  WriteHtml(out, validation_rows, scene, result, paths);
  std::cout << "wrote " << std::filesystem::absolute(output_path).string() << '\n';
  return 0;
}
