#include "common/validators.h"
#include "geometry/patch.h"
#include "rt_candidate/candidate_buffer.h"
#include "rt_candidate/candidate_generator.h"
#include "rt_candidate/candidate_stats.h"
#include "rt_candidate/cpu_reference_candidate_generator.h"
#include "rt_candidate/optix_candidate_tracer.h"
#include "rt_candidate/external_batch_candidate.h"
#include "rt_candidate/proxy_scene.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

void Expect(bool condition, const char* label) {
  if (!condition) {
    std::cerr << "FAIL " << label << '\n';
    ++g_failures;
  }
}

void ExpectOk(const p2cccd::Status& status, const char* label) {
  if (!status.ok) {
    std::cerr << "FAIL " << label << ": " << status.message << '\n';
    ++g_failures;
  }
}

void ExpectError(const p2cccd::Status& status, const char* label) {
  if (status.ok) {
    std::cerr << "FAIL " << label << ": expected error\n";
    ++g_failures;
  }
}

std::string ReadTextFile(const std::filesystem::path& path) {
  std::ifstream stream(path);
  return std::string(std::istreambuf_iterator<char>(stream),
                     std::istreambuf_iterator<char>());
}

p2cccd::Patch MakePatch(std::uint32_t patch_id, double x) {
  p2cccd::Patch patch;
  patch.patch_id = patch_id;
  patch.triangle_ids = {patch_id};
  patch.triangle_count = 1;
  patch.area = 1.0;
  patch.local_center = {x, 0.0, 0.0};
  patch.radius = 0.2;
  return patch;
}

p2cccd::MotionSegment MakeStaticMotion(double x) {
  p2cccd::MotionSegment motion;
  motion.t0 = 0.0;
  motion.t1 = 1.0;
  motion.pose_t0.translation = {x, 0.0, 0.0};
  motion.pose_t1.translation = {x, 0.0, 0.0};
  motion.pose_t0.rotation_xyzw = {0.0, 0.0, 0.0, 1.0};
  motion.pose_t1.rotation_xyzw = {0.0, 0.0, 0.0, 1.0};
  return motion;
}

p2cccd::ProxySceneBuildInput MakeOverlappingSceneInput() {
  p2cccd::ProxySceneBuildInput input;
  input.query_id = 42;

  p2cccd::ProxyObjectBuildInput object_a;
  object_a.object_id = 10;
  object_a.proxy_type = p2cccd::ProxyType::kSweptAabb;
  object_a.patches = {MakePatch(1, 0.0)};
  object_a.motion_segments = {MakeStaticMotion(0.0)};
  object_a.slabs_per_motion_segment = 2;
  object_a.eps_proxy = 0.05;

  p2cccd::ProxyObjectBuildInput object_b;
  object_b.object_id = 20;
  object_b.proxy_type = p2cccd::ProxyType::kCapsule;
  object_b.patches = {MakePatch(2, 0.25)};
  object_b.motion_segments = {MakeStaticMotion(0.0)};
  object_b.slabs_per_motion_segment = 2;
  object_b.eps_proxy = 0.05;

  input.objects = {object_a, object_b};
  return input;
}

}  // namespace

int main() {
  p2cccd::ProxyScene scene;
  ExpectOk(p2cccd::BuildProxyScene(MakeOverlappingSceneInput(), &scene), "build proxy scene");
  Expect(scene.query_id == 42, "scene query id");
  Expect(scene.primitives.size() == 4, "proxy primitive count");
  ExpectOk(p2cccd::ValidateProxyScene(scene), "validate proxy scene");

  p2cccd::RawCandidateBuffer raw_buffer;
  ExpectOk(p2cccd::GenerateRawCandidatesCpu(scene, scene.query_id, &raw_buffer),
           "generate raw candidates");
  Expect(raw_buffer.hits.size() == 2, "raw overlap count per slab");
  for (const p2cccd::RawCandidateHit& hit : raw_buffer.hits) {
    Expect(hit.query_id == scene.query_id, "raw query id");
    Expect((hit.flags & p2cccd::kRawCandidateValid) != 0, "raw valid flag");
    Expect((hit.flags & p2cccd::kRawCandidateAabbOverlap) != 0, "raw overlap flag");
    Expect(hit.motion_bound[3] > 0.0f, "raw conservative radius");
  }

  std::vector<p2cccd::CandidateRecord> candidates;
  ExpectOk(p2cccd::CompactRawCandidates(scene, raw_buffer, &candidates), "compact candidates");
  Expect(candidates.size() == 2, "compact candidate count");
  for (const p2cccd::CandidateRecord& candidate : candidates) {
    ExpectOk(p2cccd::ValidateCandidateRecord(candidate), "validate compact candidate");
    Expect(candidate.candidate_id != 0, "candidate id assigned");
    Expect(candidate.query_id == scene.query_id, "candidate query id");
    Expect(candidate.object_a_id == 10, "candidate object a id");
    Expect(candidate.object_b_id == 20, "candidate object b id");
    Expect(candidate.rt_hit_count == 1, "candidate hit count");
  }

  p2cccd::RawCandidateBuffer duplicate_buffer = raw_buffer;
  duplicate_buffer.hits.push_back(raw_buffer.hits.front());
  ExpectOk(p2cccd::CompactRawCandidates(scene, duplicate_buffer, &candidates),
           "compact duplicate raw candidates");
  Expect(candidates.size() == 2, "duplicate compaction keeps key count");
  Expect(candidates.front().rt_hit_count == 2, "duplicate compaction accumulates hit count");

  p2cccd::CandidateGenerator generator;
  p2cccd::CandidateGenerationResult generation_result;
  ExpectOk(generator.GenerateCandidates(scene, scene.query_id, &generation_result),
           "CandidateGenerator result API");
  Expect(generation_result.backend_name == "cpu_reference", "default backend name");
  Expect(generation_result.raw_buffer.hits.size() == 2, "result raw count");
  Expect(generation_result.candidates.size() == 2, "result candidate count");
  Expect(generation_result.timing.build_ms >= 0.0, "result build timing");
  Expect(generation_result.timing.update_ms >= 0.0, "result update timing");
  Expect(generation_result.timing.trace_ms >= 0.0, "result trace timing");
  Expect(generation_result.timing.compact_ms >= 0.0, "result compact timing");
  Expect(generation_result.timing.stats_ms >= 0.0, "result stats timing");
  Expect(generation_result.timing.total_ms >= 0.0, "result total timing");
  Expect(generation_result.density.query_id == scene.query_id, "density query id");
  Expect(generation_result.density.proxy_count == 4, "density proxy count");
  Expect(generation_result.density.object_count == 2, "density object count");
  Expect(generation_result.density.slab_count == 2, "density slab count");
  Expect(generation_result.density.cross_object_same_slab_pair_count == 2,
         "density possible pair count");
  Expect(generation_result.density.raw_hit_count == 2, "density raw hit count");
  Expect(generation_result.density.compact_candidate_count == 2,
         "density compact candidate count");
  Expect(generation_result.density.aabb_overlap_ratio == 1.0, "density overlap ratio");

  p2cccd::CpuReferenceCandidateGenerator cpu_reference_generator;
  p2cccd::CandidateGenerationResult cpu_reference_result;
  ExpectOk(cpu_reference_generator.Generate(scene, &cpu_reference_result),
           "CPU reference candidate generator");
  Expect(cpu_reference_result.candidates.size() == 2, "CPU reference candidate count");

  std::vector<p2cccd::ExternalBatchQuery> external_queries;
  p2cccd::ExternalBatchQuery colliding_query;
  colliding_query.source_query_id = 5;
  colliding_query.source_query_index = 5;
  colliding_query.family = p2cccd::ExternalQueryFamily::kVertexFace;
  colliding_query.vertices_t0 = {{
      {0.25, 0.25, 1.0},
      {0.0, 0.0, 0.0},
      {1.0, 0.0, 0.0},
      {0.0, 1.0, 0.0},
  }};
  colliding_query.vertices_t1 = {{
      {0.25, 0.25, -1.0},
      {0.0, 0.0, 0.0},
      {1.0, 0.0, 0.0},
      {0.0, 1.0, 0.0},
  }};
  colliding_query.has_ground_truth = true;
  colliding_query.ground_truth_collides = true;
  external_queries.push_back(colliding_query);

  p2cccd::ExternalBatchQuery separated_query = colliding_query;
  separated_query.source_query_id = 6;
  separated_query.source_query_index = 6;
  separated_query.vertices_t0[0] = {0.25, 0.25, 2.0};
  separated_query.vertices_t1[0] = {0.25, 0.25, 2.0};
  separated_query.ground_truth_collides = false;
  external_queries.push_back(separated_query);

  p2cccd::CandidateGeneratorConfig external_config;
  external_config.backend = p2cccd::CandidateBackend::kCpuReference;
  p2cccd::ExternalBatchCandidateResult external_result;
  ExpectOk(p2cccd::GenerateCandidatesForExternalBatch(external_queries, external_config, &external_result),
           "generate candidates for external batch");
  Expect(external_result.backend_name == "cpu_reference", "external batch backend name");
  Expect(external_result.primitive_count == 4, "external batch primitive count");
  Expect(external_result.raw_hit_count == 1, "external batch raw hit count");
  Expect(external_result.compact_candidate_count == 1, "external batch compact candidate count");
  Expect(external_result.candidate_recall == 1.0, "external batch candidate recall");
  Expect(external_result.runtime_query_ids.size() == 2, "external batch runtime id count");
  Expect(external_result.candidates.size() == 1, "external batch translated candidate count");
  if (!external_result.candidates.empty()) {
    const p2cccd::CandidateRecord& external_candidate = external_result.candidates.front();
    Expect(external_candidate.query_id == 5, "external batch candidate runtime query id");
    Expect(external_candidate.patch_a_id == 0, "external batch patch a id");
    Expect(external_candidate.patch_b_id == 6, "external batch patch b id");
    Expect(external_candidate.object_a_id == 1, "external batch object a id");
    Expect(external_candidate.object_b_id == 2, "external batch object b id");
  }

  const std::filesystem::path export_dir =
      std::filesystem::temp_directory_path() / "p2cccd_rt_candidate_tests";
  const std::filesystem::path csv_path = export_dir / "candidate_density.csv";
  const std::filesystem::path jsonl_path = export_dir / "candidate_density.jsonl";
  std::filesystem::remove(csv_path);
  std::filesystem::remove(jsonl_path);
  ExpectOk(p2cccd::WriteCandidateDensityCsv(csv_path, {generation_result.density}, false),
           "write candidate density csv");
  ExpectOk(p2cccd::WriteCandidateDensityJsonl(jsonl_path, {generation_result.density}, false),
           "write candidate density jsonl");
  const std::string csv_text = ReadTextFile(csv_path);
  const std::string json_text = ReadTextFile(jsonl_path);
  Expect(csv_text.find(p2cccd::CandidateDensityCsvHeader()) != std::string::npos,
         "csv density header");
  Expect(csv_text.find("cpu_reference") != std::string::npos, "csv backend name");
  Expect(json_text.find("\"backend_name\":\"cpu_reference\"") != std::string::npos,
         "jsonl backend name");

  if (!p2cccd::OptixCandidateTracer::IsBuildEnabled()) {
    p2cccd::OptixCandidateTracer optix_tracer;
    ExpectError(optix_tracer.Generate(scene, &generation_result), "optix disabled branch");
  }

  p2cccd::CandidateGeneratorConfig optix_fallback_config;
  optix_fallback_config.backend = p2cccd::CandidateBackend::kOptix;
  optix_fallback_config.allow_optix_cpu_fallback = true;
  p2cccd::CandidateGenerator optix_fallback_generator(optix_fallback_config);
  ExpectOk(optix_fallback_generator.GenerateCandidates(scene, scene.query_id, &generation_result),
           "CandidateGenerator optix CPU fallback");
#if P2CCCD_HAS_OPTIX
  Expect(generation_result.backend_name == "optix_rt",
         "CandidateGenerator optix backend name");
#else
  Expect(generation_result.backend_name == "optix_cpu_fallback",
         "CandidateGenerator optix fallback backend name");
#endif
  Expect(generation_result.candidates.size() == 2,
         "CandidateGenerator optix candidate count");

  const std::vector<p2cccd::CandidateRecord> generated =
      generator.TraceCandidates(scene, scene.query_id);
  Expect(generated.size() == 2, "CandidateGenerator scene overload");

  p2cccd::RawCandidateBuffer bad_buffer = raw_buffer;
  bad_buffer.hits.front().proxy_a_index = 999;
  ExpectError(p2cccd::CompactRawCandidates(scene, bad_buffer, &candidates),
              "reject out-of-range raw proxy index");

  bad_buffer = raw_buffer;
  bad_buffer.hits.front().pair_key = 0;
  ExpectError(p2cccd::CompactRawCandidates(scene, bad_buffer, &candidates),
              "reject mismatched raw pair key");

  bad_buffer = raw_buffer;
  bad_buffer.hits.front().query_id = 777;
  ExpectError(p2cccd::CompactRawCandidates(scene, bad_buffer, &candidates),
              "reject mismatched raw query id");

  bad_buffer = raw_buffer;
  bad_buffer.hits.front().slab_id = 99;
  ExpectError(p2cccd::CompactRawCandidates(scene, bad_buffer, &candidates),
              "reject mismatched raw slab id");

  p2cccd::ProxySceneBuildInput bad_input = MakeOverlappingSceneInput();
  bad_input.objects.front().proxy_type = p2cccd::ProxyType::kUnknown;
  ExpectError(p2cccd::BuildProxyScene(bad_input, &scene), "reject unknown proxy type");

  bad_input = MakeOverlappingSceneInput();
  bad_input.objects.back().slabs_per_motion_segment = 3;
  ExpectError(p2cccd::BuildProxyScene(bad_input, &scene), "reject non-global slab grid");

  return g_failures == 0 ? 0 : 1;
}
