#include <optix.h>
#include <optix_device.h>

#include <cstdint>

struct DeviceProxyPrimitive {
  float bounds_min[3];
  float bounds_max[3];
  std::uint32_t object_id;
  std::uint32_t patch_id;
  std::uint32_t slab_id;
  std::uint32_t proxy_type;
  float motion_bound[4];
};

struct alignas(16) DeviceRawCandidateHit {
  std::uint64_t query_id;
  std::uint64_t pair_key;
  std::uint32_t proxy_a_index;
  std::uint32_t proxy_b_index;
  std::uint32_t slab_id;
  std::uint32_t rt_hit_count;
  std::uint32_t flags;
  std::uint32_t reserved0;
  float motion_bound[4];
  std::uint32_t reserved1;
  std::uint32_t reserved2;
};

struct LaunchParams {
  OptixTraversableHandle traversable;
  const DeviceProxyPrimitive* proxies;
  DeviceRawCandidateHit* hits;
  unsigned int* hit_count;
  unsigned int proxy_count;
  unsigned int hit_capacity;
  std::uint64_t query_id;
};

extern "C" {
__constant__ LaunchParams params;
}

static __forceinline__ __device__ bool AabbOverlap(const DeviceProxyPrimitive& lhs,
                                                   const DeviceProxyPrimitive& rhs) {
  for (unsigned int axis = 0; axis < 3; ++axis) {
    if (lhs.bounds_max[axis] < rhs.bounds_min[axis] ||
        rhs.bounds_max[axis] < lhs.bounds_min[axis]) {
      return false;
    }
  }
  return true;
}

static __forceinline__ __device__ std::uint64_t PairKey(const unsigned int lhs,
                                                        const unsigned int rhs) {
  const unsigned int lo = lhs < rhs ? lhs : rhs;
  const unsigned int hi = lhs < rhs ? rhs : lhs;
  return (static_cast<std::uint64_t>(lo) << 32U) | static_cast<std::uint64_t>(hi);
}

static __forceinline__ __device__ float MaxFloat(const float lhs, const float rhs) {
  return lhs > rhs ? lhs : rhs;
}

extern "C" __global__ void __raygen__p2cccd_candidates() {
  const unsigned int proxy_index = optixGetLaunchIndex().x;
  if (proxy_index >= params.proxy_count) {
    return;
  }

  const DeviceProxyPrimitive proxy = params.proxies[proxy_index];
  const float center_y = 0.5f * (proxy.bounds_min[1] + proxy.bounds_max[1]);
  const float center_z = 0.5f * (proxy.bounds_min[2] + proxy.bounds_max[2]);
  const float epsilon = 1.0e-4f;
  const float3 origin = make_float3(proxy.bounds_min[0] - epsilon, center_y, center_z);
  const float3 direction = make_float3(1.0f, 0.0f, 0.0f);
  const float tmax = MaxFloat(proxy.bounds_max[0] - proxy.bounds_min[0] + 2.0f * epsilon,
                              epsilon);

  unsigned int payload0 = proxy_index;
  optixTrace(params.traversable,
             origin,
             direction,
             0.0f,
             tmax,
             0.0f,
             OptixVisibilityMask(255),
             OPTIX_RAY_FLAG_DISABLE_ANYHIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
             0,
             1,
             0,
             payload0);
}

extern "C" __global__ void __intersection__p2cccd_candidates() {
  const unsigned int proxy_a_index = optixGetPayload_0();
  const unsigned int proxy_b_index = optixGetPrimitiveIndex();
  if (proxy_a_index >= params.proxy_count || proxy_b_index >= params.proxy_count ||
      proxy_a_index >= proxy_b_index) {
    return;
  }

  const DeviceProxyPrimitive proxy_a = params.proxies[proxy_a_index];
  const DeviceProxyPrimitive proxy_b = params.proxies[proxy_b_index];
  if (proxy_a.object_id == proxy_b.object_id || proxy_a.slab_id != proxy_b.slab_id) {
    return;
  }
  if (!AabbOverlap(proxy_a, proxy_b)) {
    return;
  }

  const unsigned int hit_index = atomicAdd(params.hit_count, 1U);
  if (hit_index >= params.hit_capacity) {
    return;
  }

  DeviceRawCandidateHit hit = {};
  hit.query_id = params.query_id;
  hit.pair_key = PairKey(proxy_a_index, proxy_b_index);
  hit.proxy_a_index = proxy_a_index;
  hit.proxy_b_index = proxy_b_index;
  hit.slab_id = proxy_a.slab_id;
  hit.rt_hit_count = 1U;
  hit.flags = (1U << 0U) | (1U << 1U);
  for (unsigned int i = 0; i < 4; ++i) {
    hit.motion_bound[i] = MaxFloat(proxy_a.motion_bound[i], proxy_b.motion_bound[i]);
  }
  params.hits[hit_index] = hit;
}

extern "C" __global__ void __miss__p2cccd_candidates() {}
