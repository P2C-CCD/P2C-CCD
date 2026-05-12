#include "rt_candidate/optix_candidate_tracer.h"

#include "rt_candidate/candidate_stats.h"
#include "rt_candidate/cpu_reference_candidate_generator.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <mutex>
#include <set>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#if P2CCCD_HAS_OPTIX
#include <cuda.h>
#include <cuda_runtime.h>
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>
#endif

namespace p2cccd {
namespace {

using Clock = std::chrono::steady_clock;

double ElapsedMilliseconds(const Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

Status GenerateCpuFallback(const ProxyScene& scene,
                           const RtCandidateTiming& optix_timing,
                           CandidateGenerationResult* result) {
  CpuReferenceCandidateGenerator fallback;
  CandidateGenerationResult generated;
  if (auto status = fallback.Generate(scene, &generated); !status.ok) {
    return status;
  }
  generated.backend_name = "optix_cpu_fallback";
  generated.timing.build_ms += optix_timing.build_ms;
  generated.timing.update_ms += optix_timing.update_ms;
  generated.timing.trace_ms += optix_timing.trace_ms;
  generated.timing.total_ms += optix_timing.total_ms;
  generated.density.backend_name = generated.backend_name;
  generated.density.timing = generated.timing;
  *result = std::move(generated);
  return Status::Ok();
}

#if P2CCCD_HAS_OPTIX

#ifndef P2CCCD_OPTIX_CANDIDATE_PTX_PATH
#define P2CCCD_OPTIX_CANDIDATE_PTX_PATH ""
#endif

constexpr const char* kOptixBackendName = "optix_rt";

struct DeviceProxyPrimitive {
  float bounds_min[3];
  float bounds_max[3];
  std::uint32_t object_id = 0;
  std::uint32_t patch_id = 0;
  std::uint32_t slab_id = 0;
  std::uint32_t proxy_type = 0;
  float motion_bound[4];
};

struct LaunchParams {
  OptixTraversableHandle traversable = 0;
  DeviceProxyPrimitive* proxies = nullptr;
  RawCandidateHit* hits = nullptr;
  unsigned int* hit_count = nullptr;
  unsigned int proxy_count = 0;
  unsigned int hit_capacity = 0;
  std::uint64_t query_id = 0;
};

template <typename T>
struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) SbtRecord {
  char header[OPTIX_SBT_RECORD_HEADER_SIZE];
  T data;
};

struct EmptySbtData {};

using EmptySbtRecord = SbtRecord<EmptySbtData>;

static_assert(std::is_standard_layout_v<DeviceProxyPrimitive>);
static_assert(std::is_trivially_copyable_v<DeviceProxyPrimitive>);

void FreeIfAllocated(void* ptr);

struct OptixRuntimeCache {
  OptixDeviceContext context = nullptr;
  OptixModule module = nullptr;
  OptixProgramGroup raygen_group = nullptr;
  OptixProgramGroup miss_group = nullptr;
  OptixProgramGroup hit_group = nullptr;
  OptixPipeline pipeline = nullptr;
  OptixShaderBindingTable sbt = {};
  EmptySbtRecord* raygen_record = nullptr;
  EmptySbtRecord* miss_record = nullptr;
  EmptySbtRecord* hit_record = nullptr;
  cudaStream_t stream = nullptr;

  DeviceProxyPrimitive* device_proxies = nullptr;
  std::size_t proxy_capacity = 0;
  OptixAabb* device_aabbs = nullptr;
  std::size_t aabb_capacity = 0;
  RawCandidateHit* device_hits = nullptr;
  std::size_t hit_capacity = 0;
  unsigned int* device_hit_count = nullptr;
  std::size_t hit_count_capacity = 0;
  LaunchParams* device_params = nullptr;
  std::size_t params_capacity = 0;

  void* device_temp = nullptr;
  std::size_t temp_capacity_bytes = 0;
  void* device_gas_output = nullptr;
  std::size_t gas_output_capacity_bytes = 0;
  OptixTraversableHandle gas_traversable = 0;
  std::size_t gas_primitive_count = 0;
  bool gas_update_ready = false;

  std::mutex mutex;

  ~OptixRuntimeCache() {
    FreeIfAllocated(device_params);
    FreeIfAllocated(device_hit_count);
    FreeIfAllocated(device_hits);
    FreeIfAllocated(device_proxies);
    FreeIfAllocated(device_aabbs);
    FreeIfAllocated(raygen_record);
    FreeIfAllocated(miss_record);
    FreeIfAllocated(hit_record);
    FreeIfAllocated(device_temp);
    FreeIfAllocated(device_gas_output);
    if (pipeline != nullptr) {
      optixPipelineDestroy(pipeline);
    }
    if (raygen_group != nullptr) {
      optixProgramGroupDestroy(raygen_group);
    }
    if (miss_group != nullptr) {
      optixProgramGroupDestroy(miss_group);
    }
    if (hit_group != nullptr) {
      optixProgramGroupDestroy(hit_group);
    }
    if (module != nullptr) {
      optixModuleDestroy(module);
    }
    if (stream != nullptr) {
      cudaStreamDestroy(stream);
    }
    if (context != nullptr) {
      optixDeviceContextDestroy(context);
    }
  }
};

Status CheckCuda(const cudaError_t result, const char* expression) {
  if (result == cudaSuccess) {
    return Status::Ok();
  }
  return Status::Error(std::string(expression) + " failed: " + cudaGetErrorString(result));
}

Status CheckOptix(const OptixResult result, const char* expression) {
  if (result == OPTIX_SUCCESS) {
    return Status::Ok();
  }
  return Status::Error(std::string(expression) + " failed with OptiX error code " +
                       std::to_string(static_cast<int>(result)));
}

void ContextLogCallback(unsigned int, const char*, const char*, void*) {}

std::string ReadTextFile(const std::filesystem::path& path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) {
    return {};
  }
  return std::string(std::istreambuf_iterator<char>(stream),
                     std::istreambuf_iterator<char>());
}

std::uint64_t CountPossibleCrossObjectSameSlabPairs(const ProxyScene& scene) {
  std::uint64_t count = 0;
  for (std::uint32_t i = 0; i < scene.primitives.size(); ++i) {
    for (std::uint32_t j = i + 1; j < scene.primitives.size(); ++j) {
      const ProxyPrimitive& lhs = scene.primitives[i];
      const ProxyPrimitive& rhs = scene.primitives[j];
      if (lhs.object_id != rhs.object_id && lhs.slab_id == rhs.slab_id) {
        ++count;
      }
    }
  }
  return count;
}

DeviceProxyPrimitive MakeDeviceProxy(const ProxyPrimitive& primitive) {
  DeviceProxyPrimitive proxy;
  for (std::uint32_t axis = 0; axis < 3; ++axis) {
    proxy.bounds_min[axis] = static_cast<float>(primitive.bounds.min[axis]);
    proxy.bounds_max[axis] = static_cast<float>(primitive.bounds.max[axis]);
  }
  proxy.object_id = primitive.object_id;
  proxy.patch_id = primitive.patch_id;
  proxy.slab_id = primitive.slab_id;
  proxy.proxy_type = static_cast<std::uint32_t>(primitive.proxy_type);
  proxy.motion_bound[0] = static_cast<float>(primitive.motion_bound.translation_bound);
  proxy.motion_bound[1] = static_cast<float>(primitive.motion_bound.rotation_angle);
  proxy.motion_bound[2] = static_cast<float>(primitive.motion_bound.radial_motion_bound);
  proxy.motion_bound[3] = static_cast<float>(primitive.motion_bound.conservative_radius);
  return proxy;
}

std::array<float, 3> MaxProxyHalfExtents(const std::vector<DeviceProxyPrimitive>& proxies) {
  std::array<float, 3> half_extents{0.0f, 0.0f, 0.0f};
  for (const DeviceProxyPrimitive& proxy : proxies) {
    for (std::uint32_t axis = 0; axis < 3; ++axis) {
      half_extents[axis] =
          (std::max)(half_extents[axis],
                     0.5f * (std::max)(0.0f, proxy.bounds_max[axis] - proxy.bounds_min[axis]));
    }
  }
  return half_extents;
}

std::vector<OptixAabb> BuildTraversalAabbs(const std::vector<DeviceProxyPrimitive>& proxies) {
  const std::array<float, 3> half_extents = MaxProxyHalfExtents(proxies);
  constexpr float kTraversalEpsilon = 1.0e-4f;
  std::vector<OptixAabb> aabbs;
  aabbs.reserve(proxies.size());
  for (const DeviceProxyPrimitive& proxy : proxies) {
    OptixAabb aabb = {};
    aabb.minX = proxy.bounds_min[0] - kTraversalEpsilon;
    aabb.maxX = proxy.bounds_max[0] + kTraversalEpsilon;
    aabb.minY = proxy.bounds_min[1] - half_extents[1] - kTraversalEpsilon;
    aabb.maxY = proxy.bounds_max[1] + half_extents[1] + kTraversalEpsilon;
    aabb.minZ = proxy.bounds_min[2] - half_extents[2] - kTraversalEpsilon;
    aabb.maxZ = proxy.bounds_max[2] + half_extents[2] + kTraversalEpsilon;
    aabbs.push_back(aabb);
  }
  return aabbs;
}

template <typename T>
Status AllocateAndCopyToDevice(const std::vector<T>& host, T** device, const char* label) {
  if (device == nullptr) {
    return Status::Error(std::string(label) + " output pointer is null");
  }
  *device = nullptr;
  if (host.empty()) {
    return Status::Ok();
  }
  const std::size_t bytes = sizeof(T) * host.size();
  if (auto status = CheckCuda(cudaMalloc(reinterpret_cast<void**>(device), bytes), label);
      !status.ok) {
    return status;
  }
  return CheckCuda(cudaMemcpy(*device, host.data(), bytes, cudaMemcpyHostToDevice), label);
}

void FreeIfAllocated(void* ptr) {
  if (ptr != nullptr) {
    cudaFree(ptr);
  }
}

std::size_t GrownCapacity(const std::size_t required) {
  std::size_t capacity = 1;
  while (capacity < required && capacity < ((std::numeric_limits<std::size_t>::max)() / 2U)) {
    capacity *= 2U;
  }
  return (std::max)(capacity, required);
}

template <typename T>
Status EnsureDeviceArrayCapacity(T** device,
                                 std::size_t* capacity,
                                 const std::size_t required,
                                 const char* label) {
  if (device == nullptr || capacity == nullptr) {
    return Status::Error(std::string(label) + " capacity output pointer is null");
  }
  if (required == 0) {
    return Status::Ok();
  }
  if (*device != nullptr && *capacity >= required) {
    return Status::Ok();
  }
  FreeIfAllocated(*device);
  *device = nullptr;
  *capacity = 0;
  const std::size_t next_capacity = GrownCapacity(required);
  if (auto status = CheckCuda(cudaMalloc(reinterpret_cast<void**>(device),
                                         sizeof(T) * next_capacity),
                              label);
      !status.ok) {
    return status;
  }
  *capacity = next_capacity;
  return Status::Ok();
}

Status EnsureDeviceByteCapacity(void** device,
                                std::size_t* capacity_bytes,
                                const std::size_t required_bytes,
                                const char* label) {
  if (device == nullptr || capacity_bytes == nullptr) {
    return Status::Error(std::string(label) + " capacity output pointer is null");
  }
  if (required_bytes == 0) {
    return Status::Ok();
  }
  if (*device != nullptr && *capacity_bytes >= required_bytes) {
    return Status::Ok();
  }
  FreeIfAllocated(*device);
  *device = nullptr;
  *capacity_bytes = 0;
  const std::size_t next_capacity = GrownCapacity(required_bytes);
  if (auto status = CheckCuda(cudaMalloc(device, next_capacity), label); !status.ok) {
    return status;
  }
  *capacity_bytes = next_capacity;
  return Status::Ok();
}

OptixRuntimeCache& GetOptixRuntimeCache() {
  static OptixRuntimeCache cache;
  return cache;
}

Status BuildOptixContext(OptixDeviceContext* context) {
  if (auto status = CheckCuda(cudaFree(nullptr), "cudaFree(nullptr)"); !status.ok) {
    return status;
  }
  if (auto status = CheckOptix(optixInit(), "optixInit()"); !status.ok) {
    return status;
  }

  OptixDeviceContextOptions options = {};
  options.logCallbackFunction = &ContextLogCallback;
  options.logCallbackLevel = 3;
  return CheckOptix(optixDeviceContextCreate(nullptr, &options, context),
                    "optixDeviceContextCreate()");
}

Status CreateModule(OptixDeviceContext context,
                    OptixPipelineCompileOptions* pipeline_compile_options,
                    OptixModule* module) {
  const std::filesystem::path ptx_path(P2CCCD_OPTIX_CANDIDATE_PTX_PATH);
  const std::string ptx = ReadTextFile(ptx_path);
  if (ptx.empty()) {
    return Status::Error("failed to read OptiX candidate PTX from " + ptx_path.string());
  }

  OptixModuleCompileOptions module_options = {};
  module_options.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
  module_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
#if OPTIX_VERSION >= 70400
  module_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
#else
  module_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;
#endif

  *pipeline_compile_options = {};
  pipeline_compile_options->usesMotionBlur = false;
  pipeline_compile_options->traversableGraphFlags =
      OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
  pipeline_compile_options->numPayloadValues = 1;
  pipeline_compile_options->numAttributeValues = 2;
  pipeline_compile_options->exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
  pipeline_compile_options->pipelineLaunchParamsVariableName = "params";
#if OPTIX_VERSION >= 70400
  pipeline_compile_options->usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM;
#endif

  char log[4096];
  std::size_t log_size = sizeof(log);
#if OPTIX_VERSION >= 70700
  return CheckOptix(optixModuleCreate(context,
                                      &module_options,
                                      pipeline_compile_options,
                                      ptx.c_str(),
                                      ptx.size(),
                                      log,
                                      &log_size,
                                      module),
                    "optixModuleCreate()");
#else
  return CheckOptix(optixModuleCreateFromPTX(context,
                                            &module_options,
                                            pipeline_compile_options,
                                            ptx.c_str(),
                                            ptx.size(),
                                            log,
                                            &log_size,
                                            module),
                    "optixModuleCreateFromPTX()");
#endif
}

Status CreateProgramGroups(OptixDeviceContext context,
                           OptixModule module,
                           OptixProgramGroup* raygen_group,
                           OptixProgramGroup* miss_group,
                           OptixProgramGroup* hit_group) {
  OptixProgramGroupOptions options = {};
  char log[4096];
  std::size_t log_size = sizeof(log);

  OptixProgramGroupDesc raygen_desc = {};
  raygen_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
  raygen_desc.raygen.module = module;
  raygen_desc.raygen.entryFunctionName = "__raygen__p2cccd_candidates";
  if (auto status = CheckOptix(optixProgramGroupCreate(context,
                                                       &raygen_desc,
                                                       1,
                                                       &options,
                                                       log,
                                                       &log_size,
                                                       raygen_group),
                               "optixProgramGroupCreate(raygen)");
      !status.ok) {
    return status;
  }

  log_size = sizeof(log);
  OptixProgramGroupDesc miss_desc = {};
  miss_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
  miss_desc.miss.module = module;
  miss_desc.miss.entryFunctionName = "__miss__p2cccd_candidates";
  if (auto status = CheckOptix(optixProgramGroupCreate(context,
                                                       &miss_desc,
                                                       1,
                                                       &options,
                                                       log,
                                                       &log_size,
                                                       miss_group),
                               "optixProgramGroupCreate(miss)");
      !status.ok) {
    return status;
  }

  log_size = sizeof(log);
  OptixProgramGroupDesc hit_desc = {};
  hit_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
  hit_desc.hitgroup.moduleIS = module;
  hit_desc.hitgroup.entryFunctionNameIS = "__intersection__p2cccd_candidates";
  return CheckOptix(optixProgramGroupCreate(context,
                                           &hit_desc,
                                           1,
                                           &options,
                                           log,
                                           &log_size,
                                           hit_group),
                    "optixProgramGroupCreate(hitgroup)");
}

Status CreatePipeline(OptixDeviceContext context,
                      const OptixPipelineCompileOptions& pipeline_compile_options,
                      const std::array<OptixProgramGroup, 3>& groups,
                      OptixPipeline* pipeline) {
  OptixPipelineLinkOptions link_options = {};
  link_options.maxTraceDepth = 1;
#if OPTIX_VERSION < 70700
  link_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
#endif
  char log[4096];
  std::size_t log_size = sizeof(log);
  if (auto status = CheckOptix(optixPipelineCreate(context,
                                                   &pipeline_compile_options,
                                                   &link_options,
                                                   groups.data(),
                                                   static_cast<unsigned int>(groups.size()),
                                                   log,
                                                   &log_size,
                                                   pipeline),
                               "optixPipelineCreate()");
      !status.ok) {
    return status;
  }
  return CheckOptix(optixPipelineSetStackSize(*pipeline, 2048, 2048, 2048, 1),
                    "optixPipelineSetStackSize()");
}

Status BuildGas(OptixDeviceContext context,
                cudaStream_t stream,
                OptixRuntimeCache* runtime,
                const std::vector<OptixAabb>& host_aabbs,
                OptixTraversableHandle* traversable,
                void** device_gas_output,
                bool* used_update) {
  if (runtime == nullptr) {
    return Status::Error("OptiX runtime cache is null");
  }
  if (auto status = EnsureDeviceArrayCapacity(&runtime->device_aabbs,
                                              &runtime->aabb_capacity,
                                              host_aabbs.size(),
                                              "cudaMalloc AABBs");
      !status.ok) {
    return status;
  }
  if (auto status = CheckCuda(cudaMemcpy(runtime->device_aabbs,
                                         host_aabbs.data(),
                                         sizeof(OptixAabb) * host_aabbs.size(),
                                         cudaMemcpyHostToDevice),
                              "cudaMemcpy AABBs");
      !status.ok) {
    return status;
  }

  CUdeviceptr aabb_buffer = reinterpret_cast<CUdeviceptr>(runtime->device_aabbs);
  unsigned int geometry_flags = OPTIX_GEOMETRY_FLAG_NONE;
  OptixBuildInput build_input = {};
  build_input.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
  build_input.customPrimitiveArray.aabbBuffers = &aabb_buffer;
  build_input.customPrimitiveArray.numPrimitives =
      static_cast<unsigned int>(host_aabbs.size());
  build_input.customPrimitiveArray.flags = &geometry_flags;
  build_input.customPrimitiveArray.numSbtRecords = 1;

  const bool can_update = runtime->gas_update_ready && runtime->device_gas_output != nullptr &&
                          runtime->gas_traversable != 0 &&
                          runtime->gas_primitive_count == host_aabbs.size();
  OptixAccelBuildOptions accel_options = {};
  accel_options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE | OPTIX_BUILD_FLAG_ALLOW_UPDATE;
  accel_options.operation = can_update ? OPTIX_BUILD_OPERATION_UPDATE : OPTIX_BUILD_OPERATION_BUILD;

  OptixAccelBufferSizes buffer_sizes = {};
  Status status = CheckOptix(optixAccelComputeMemoryUsage(context,
                                                          &accel_options,
                                                          &build_input,
                                                          1,
                                                          &buffer_sizes),
                             "optixAccelComputeMemoryUsage()");
  if (!status.ok) {
    return status;
  }

  std::size_t temp_bytes = buffer_sizes.tempSizeInBytes;
  if (can_update) {
    temp_bytes = buffer_sizes.tempUpdateSizeInBytes > 0
                     ? buffer_sizes.tempUpdateSizeInBytes
                     : buffer_sizes.tempSizeInBytes;
  }
  if (auto cuda_status =
          EnsureDeviceByteCapacity(&runtime->device_temp,
                                   &runtime->temp_capacity_bytes,
                                   temp_bytes,
                                   "cudaMalloc GAS temp");
      !cuda_status.ok) {
    return cuda_status;
  }
  if (auto cuda_status =
          EnsureDeviceByteCapacity(&runtime->device_gas_output,
                                   &runtime->gas_output_capacity_bytes,
                                   buffer_sizes.outputSizeInBytes,
                                   "cudaMalloc GAS output");
      !cuda_status.ok) {
    return cuda_status;
  }
  *device_gas_output = runtime->device_gas_output;

  status = CheckOptix(optixAccelBuild(context,
                                      stream,
                                      &accel_options,
                                      &build_input,
                                      1,
                                      reinterpret_cast<CUdeviceptr>(runtime->device_temp),
                                      temp_bytes,
                                      reinterpret_cast<CUdeviceptr>(runtime->device_gas_output),
                                      buffer_sizes.outputSizeInBytes,
                                      traversable,
                                      nullptr,
                                      0),
                       "optixAccelBuild()");
  if (!status.ok) {
    return status;
  }
  runtime->gas_traversable = *traversable;
  runtime->gas_primitive_count = host_aabbs.size();
  runtime->gas_update_ready = true;
  if (used_update != nullptr) {
    *used_update = can_update;
  }
  return Status::Ok();
}

Status UploadSbtRecord(const OptixProgramGroup group, EmptySbtRecord** device_record) {
  EmptySbtRecord host_record = {};
  if (auto status = CheckOptix(optixSbtRecordPackHeader(group, &host_record),
                               "optixSbtRecordPackHeader()");
      !status.ok) {
    return status;
  }
  *device_record = nullptr;
  if (auto status = CheckCuda(cudaMalloc(reinterpret_cast<void**>(device_record),
                                         sizeof(EmptySbtRecord)),
                              "cudaMalloc SBT record");
      !status.ok) {
    return status;
  }
  return CheckCuda(cudaMemcpy(*device_record,
                              &host_record,
                              sizeof(EmptySbtRecord),
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy SBT record");
}

Status BuildSbt(const OptixProgramGroup raygen_group,
                const OptixProgramGroup miss_group,
                const OptixProgramGroup hit_group,
                OptixShaderBindingTable* sbt,
                EmptySbtRecord** raygen_record,
                EmptySbtRecord** miss_record,
                EmptySbtRecord** hit_record) {
  if (auto status = UploadSbtRecord(raygen_group, raygen_record); !status.ok) {
    return status;
  }
  if (auto status = UploadSbtRecord(miss_group, miss_record); !status.ok) {
    return status;
  }
  if (auto status = UploadSbtRecord(hit_group, hit_record); !status.ok) {
    return status;
  }

  *sbt = {};
  sbt->raygenRecord = reinterpret_cast<CUdeviceptr>(*raygen_record);
  sbt->missRecordBase = reinterpret_cast<CUdeviceptr>(*miss_record);
  sbt->missRecordStrideInBytes = sizeof(EmptySbtRecord);
  sbt->missRecordCount = 1;
  sbt->hitgroupRecordBase = reinterpret_cast<CUdeviceptr>(*hit_record);
  sbt->hitgroupRecordStrideInBytes = sizeof(EmptySbtRecord);
  sbt->hitgroupRecordCount = 1;
  return Status::Ok();
}

Status LaunchOptixCandidateProgram(const ProxyScene& scene,
                                   RawCandidateBuffer* raw_buffer,
                                   RtCandidateTiming* timing) {
  std::vector<DeviceProxyPrimitive> host_proxies;
  host_proxies.reserve(scene.primitives.size());
  for (const ProxyPrimitive& primitive : scene.primitives) {
    host_proxies.push_back(MakeDeviceProxy(primitive));
  }

  const std::uint64_t hit_capacity64 = CountPossibleCrossObjectSameSlabPairs(scene);
  if (host_proxies.empty() || hit_capacity64 == 0) {
    raw_buffer->hits.clear();
    timing->build_ms = 0.0;
    timing->total_ms = 0.0;
    return Status::Ok();
  }
  if (hit_capacity64 >
      static_cast<std::uint64_t>((std::numeric_limits<unsigned int>::max)())) {
    return Status::Error("OptiX candidate hit capacity exceeds 32-bit launch limit");
  }
  if (host_proxies.size() >
      static_cast<std::size_t>((std::numeric_limits<unsigned int>::max)())) {
    return Status::Error("OptiX proxy count exceeds 32-bit launch limit");
  }

  OptixDeviceContext context = nullptr;
  OptixRuntimeCache& runtime = GetOptixRuntimeCache();
  std::lock_guard<std::mutex> runtime_lock(runtime.mutex);

  Status status = Status::Ok();
  if (runtime.context == nullptr) {
    status = BuildOptixContext(&runtime.context);
    if (status.ok) {
      status = CheckCuda(cudaStreamCreate(&runtime.stream), "cudaStreamCreate()");
    }
    OptixPipelineCompileOptions pipeline_compile_options = {};
    if (status.ok) {
      status = CreateModule(runtime.context, &pipeline_compile_options, &runtime.module);
    }
    if (status.ok) {
      status = CreateProgramGroups(
          runtime.context, runtime.module, &runtime.raygen_group, &runtime.miss_group, &runtime.hit_group);
    }
    if (status.ok) {
      const std::array<OptixProgramGroup, 3> groups{
          runtime.raygen_group, runtime.miss_group, runtime.hit_group};
      status = CreatePipeline(runtime.context, pipeline_compile_options, groups, &runtime.pipeline);
    }
    if (status.ok) {
      status = BuildSbt(runtime.raygen_group,
                        runtime.miss_group,
                        runtime.hit_group,
                        &runtime.sbt,
                        &runtime.raygen_record,
                        &runtime.miss_record,
                        &runtime.hit_record);
    }
  }
  const Clock::time_point total_start = Clock::now();
  const Clock::time_point accel_start = Clock::now();

  const std::vector<OptixAabb> host_aabbs = BuildTraversalAabbs(host_proxies);
  OptixTraversableHandle traversable = 0;
  void* device_gas_output = nullptr;
  bool used_update = false;
  if (status.ok) {
    context = runtime.context;
    status = BuildGas(
        runtime.context,
        runtime.stream,
        &runtime,
        host_aabbs,
        &traversable,
        &device_gas_output,
        &used_update);
  }
  if (status.ok) {
    status = EnsureDeviceArrayCapacity(
        &runtime.device_proxies, &runtime.proxy_capacity, host_proxies.size(), "cudaMalloc proxies");
  }
  if (status.ok) {
    status = CheckCuda(cudaMemcpy(runtime.device_proxies,
                                  host_proxies.data(),
                                  sizeof(DeviceProxyPrimitive) * host_proxies.size(),
                                  cudaMemcpyHostToDevice),
                       "cudaMemcpy proxies");
  }
  const unsigned int hit_capacity = static_cast<unsigned int>(hit_capacity64);
  if (status.ok) {
    status = EnsureDeviceArrayCapacity(
        &runtime.device_hits, &runtime.hit_capacity, hit_capacity, "cudaMalloc raw hits");
  }
  if (status.ok) {
    status = EnsureDeviceArrayCapacity(
        &runtime.device_hit_count, &runtime.hit_count_capacity, 1, "cudaMalloc hit count");
  }
  if (status.ok) {
    status = EnsureDeviceArrayCapacity(
        &runtime.device_params, &runtime.params_capacity, 1, "cudaMalloc launch params");
  }
  if (status.ok) {
    status = CheckCuda(cudaMemset(runtime.device_hit_count, 0, sizeof(unsigned int)),
                       "cudaMemset hit count");
  }

  const double accel_ms = ElapsedMilliseconds(accel_start);
  timing->build_ms = used_update ? 0.0 : accel_ms;
  timing->update_ms = used_update ? accel_ms : 0.0;

  const Clock::time_point trace_start = Clock::now();
  if (status.ok) {
    LaunchParams params;
    params.traversable = traversable;
    params.proxies = runtime.device_proxies;
    params.hits = runtime.device_hits;
    params.hit_count = runtime.device_hit_count;
    params.proxy_count = static_cast<unsigned int>(host_proxies.size());
    params.hit_capacity = hit_capacity;
    params.query_id = scene.query_id;
    status = CheckCuda(cudaMemcpy(runtime.device_params,
                                  &params,
                                  sizeof(LaunchParams),
                                  cudaMemcpyHostToDevice),
                       "cudaMemcpy launch params");
    if (status.ok) {
      status = CheckOptix(optixLaunch(runtime.pipeline,
                                      runtime.stream,
                                      reinterpret_cast<CUdeviceptr>(runtime.device_params),
                                      sizeof(LaunchParams),
                                      &runtime.sbt,
                                      static_cast<unsigned int>(host_proxies.size()),
                                      1,
                                      1),
                           "optixLaunch()");
    }
    if (status.ok) {
      status = CheckCuda(cudaStreamSynchronize(runtime.stream), "cudaStreamSynchronize(optixLaunch)");
    }
  }
  timing->trace_ms = ElapsedMilliseconds(trace_start);

  if (status.ok) {
    unsigned int host_hit_count = 0;
    status = CheckCuda(cudaMemcpy(&host_hit_count,
                                  runtime.device_hit_count,
                                  sizeof(unsigned int),
                                  cudaMemcpyDeviceToHost),
                       "cudaMemcpy hit count D2H");
    if (status.ok && host_hit_count > hit_capacity) {
      status = Status::Error("OptiX candidate hit count exceeded allocated capacity");
    }
    if (status.ok) {
      raw_buffer->hits.resize(host_hit_count);
      if (host_hit_count > 0) {
        status = CheckCuda(cudaMemcpy(raw_buffer->hits.data(),
                                      runtime.device_hits,
                                      sizeof(RawCandidateHit) * host_hit_count,
                                      cudaMemcpyDeviceToHost),
                           "cudaMemcpy raw hits D2H");
      }
    }
  }

  timing->total_ms = ElapsedMilliseconds(total_start);
  return status;
}

#endif

}  // namespace

OptixCandidateTracer::OptixCandidateTracer(OptixCandidateTracerConfig config)
    : config_(std::move(config)) {}

bool OptixCandidateTracer::IsBuildEnabled() {
#if P2CCCD_HAS_OPTIX
  return true;
#else
  return false;
#endif
}

Status OptixCandidateTracer::Generate(const ProxyScene& scene,
                                      CandidateGenerationResult* result) const {
  if (result == nullptr) {
    return Status::Error("candidate generation result output pointer is null");
  }
  if (auto status = ValidateProxyScene(scene); !status.ok) {
    return status;
  }

#if P2CCCD_HAS_OPTIX
  CandidateGenerationResult generated;
  generated.backend_name = kOptixBackendName;

  if (auto status = LaunchOptixCandidateProgram(scene, &generated.raw_buffer, &generated.timing);
      !status.ok) {
    if (!config_.allow_cpu_fallback) {
      return status;
    }
    return GenerateCpuFallback(scene, generated.timing, result);
  }

  const Clock::time_point compact_start = Clock::now();
  if (auto status = CompactRawCandidates(scene, generated.raw_buffer, &generated.candidates);
      !status.ok) {
    if (!config_.allow_cpu_fallback) {
      return status;
    }
    return GenerateCpuFallback(scene, generated.timing, result);
  }
  generated.timing.compact_ms = ElapsedMilliseconds(compact_start);

  const Clock::time_point stats_start = Clock::now();
  generated.timing.total_ms += generated.timing.compact_ms;
  if (auto status = ComputeCandidateDensityStats(scene,
                                                 generated.raw_buffer,
                                                 generated.candidates,
                                                 generated.timing,
                                                 generated.backend_name,
                                                 &generated.density);
      !status.ok) {
    return status;
  }
  generated.timing.stats_ms = ElapsedMilliseconds(stats_start);
  generated.timing.total_ms += generated.timing.stats_ms;
  generated.density.timing = generated.timing;

  *result = std::move(generated);
  return Status::Ok();
#else
  if (config_.allow_cpu_fallback) {
    return GenerateCpuFallback(scene, {}, result);
  }
  return Status::Error(
      "P2CCCD was built without OptiX support; configure with -DP2CCCD_ENABLE_OPTIX=ON");
#endif
}

}  // namespace p2cccd
