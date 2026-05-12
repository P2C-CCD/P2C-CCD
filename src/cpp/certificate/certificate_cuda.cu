#include "certificate/certificate_cuda.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

namespace p2cccd {
namespace {

constexpr int kCudaStatusCollision = 0;
constexpr int kCudaStatusSeparation = 1;
constexpr int kCudaStatusUndecided = 2;
constexpr int kCudaFamilyPointTriangle = 1;
constexpr int kCudaFamilyEdgeEdge = 2;
constexpr int kCudaReasonNone = 0;
constexpr int kCudaReasonMaxSubdivisionDepth = 2;
constexpr int kCudaRefinementNone = 0;
constexpr int kCudaRefinementBisectInterval = 1;
constexpr double kDeviceEpsilon = 1.0e-14;

struct CudaVec3 {
  double x;
  double y;
  double z;
};

struct CudaTrajectory {
  std::int64_t feature_id;
  CudaVec3 position_t0;
  CudaVec3 position_t1;
};

struct CudaPointTrianglePrimitive {
  std::int64_t point_id;
  std::int64_t triangle_id;
  CudaTrajectory point;
  CudaTrajectory triangle_v0;
  CudaTrajectory triangle_v1;
  CudaTrajectory triangle_v2;
};

struct CudaEdgeEdgePrimitive {
  std::int64_t edge_a_id;
  std::int64_t edge_b_id;
  CudaTrajectory edge_a0;
  CudaTrajectory edge_a1;
  CudaTrajectory edge_b0;
  CudaTrajectory edge_b1;
};

struct CudaPrimitiveResult {
  int status;
  std::uint32_t covered_feature_mask;
  double interval_t0;
  double interval_t1;
  double toi_upper;
  double safe_margin_lb;
  std::uint8_t witness_family;
  std::int64_t witness_id_a;
  std::int64_t witness_id_b;
  std::uint16_t reason_code;
  std::uint8_t next_refinement_mode;
};

struct CudaConfig {
  double eps_time;
  double eps_space;
  std::uint16_t max_subdivision_depth;
};

struct CudaInterval {
  double t0;
  double t1;
  std::uint16_t depth;
};

__device__ CudaVec3 Add(CudaVec3 a, CudaVec3 b) {
  return {a.x + b.x, a.y + b.y, a.z + b.z};
}

__device__ CudaVec3 Sub(CudaVec3 a, CudaVec3 b) {
  return {a.x - b.x, a.y - b.y, a.z - b.z};
}

__device__ CudaVec3 Scale(CudaVec3 a, double scale) {
  return {a.x * scale, a.y * scale, a.z * scale};
}

__device__ double Dot(CudaVec3 a, CudaVec3 b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ CudaVec3 Cross(CudaVec3 a, CudaVec3 b) {
  return {
      a.y * b.z - a.z * b.y,
      a.z * b.x - a.x * b.z,
      a.x * b.y - a.y * b.x,
  };
}

__device__ double SquaredNorm(CudaVec3 value) {
  return Dot(value, value);
}

__device__ double Norm(CudaVec3 value) {
  return sqrt(SquaredNorm(value));
}

__device__ CudaVec3 PositionAt(const CudaTrajectory& trajectory, double t) {
  return Add(Scale(trajectory.position_t0, 1.0 - t), Scale(trajectory.position_t1, t));
}

__device__ double MaxDisplacementFromMidpoint(const CudaTrajectory& trajectory,
                                              double interval_t0,
                                              double interval_t1) {
  const double interval_length = fmax(0.0, interval_t1 - interval_t0);
  return 0.5 * interval_length * Norm(Sub(trajectory.position_t1, trajectory.position_t0));
}

__device__ double DistancePointPoint(CudaVec3 a, CudaVec3 b) {
  return Norm(Sub(a, b));
}

__device__ double DistancePointSegment(CudaVec3 point, CudaVec3 a, CudaVec3 b) {
  const CudaVec3 ab = Sub(b, a);
  const double ab_squared = SquaredNorm(ab);
  if (ab_squared <= kDeviceEpsilon) {
    return DistancePointPoint(point, a);
  }
  const double t = fmin(1.0, fmax(0.0, Dot(Sub(point, a), ab) / ab_squared));
  return DistancePointPoint(point, Add(a, Scale(ab, t)));
}

__device__ double DistanceSegmentSegment(CudaVec3 p1, CudaVec3 q1, CudaVec3 p2, CudaVec3 q2) {
  const CudaVec3 d1 = Sub(q1, p1);
  const CudaVec3 d2 = Sub(q2, p2);
  const CudaVec3 r = Sub(p1, p2);
  const double a = SquaredNorm(d1);
  const double e = SquaredNorm(d2);
  const double f = Dot(d2, r);

  double s = 0.0;
  double t = 0.0;
  if (a <= kDeviceEpsilon && e <= kDeviceEpsilon) {
    return DistancePointPoint(p1, p2);
  }
  if (a <= kDeviceEpsilon) {
    t = fmin(1.0, fmax(0.0, f / e));
  } else {
    const double c = Dot(d1, r);
    if (e <= kDeviceEpsilon) {
      s = fmin(1.0, fmax(0.0, -c / a));
    } else {
      const double b = Dot(d1, d2);
      const double denom = a * e - b * b;
      if (denom != 0.0) {
        s = fmin(1.0, fmax(0.0, (b * f - c * e) / denom));
      }
      t = (b * s + f) / e;
      if (t < 0.0) {
        t = 0.0;
        s = fmin(1.0, fmax(0.0, -c / a));
      } else if (t > 1.0) {
        t = 1.0;
        s = fmin(1.0, fmax(0.0, (b - c) / a));
      }
    }
  }

  return DistancePointPoint(Add(p1, Scale(d1, s)), Add(p2, Scale(d2, t)));
}

__device__ double DistancePointTriangle(CudaVec3 point, CudaVec3 a, CudaVec3 b, CudaVec3 c) {
  const CudaVec3 ab = Sub(b, a);
  const CudaVec3 ac = Sub(c, a);
  const CudaVec3 normal = Cross(ab, ac);
  if (SquaredNorm(normal) <= kDeviceEpsilon) {
    return fmin(DistancePointSegment(point, a, b),
                fmin(DistancePointSegment(point, b, c),
                     DistancePointSegment(point, c, a)));
  }

  const CudaVec3 ap = Sub(point, a);
  const double d1 = Dot(ab, ap);
  const double d2 = Dot(ac, ap);
  if (d1 <= 0.0 && d2 <= 0.0) {
    return DistancePointPoint(point, a);
  }

  const CudaVec3 bp = Sub(point, b);
  const double d3 = Dot(ab, bp);
  const double d4 = Dot(ac, bp);
  if (d3 >= 0.0 && d4 <= d3) {
    return DistancePointPoint(point, b);
  }

  const double vc = d1 * d4 - d3 * d2;
  if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
    return DistancePointPoint(point, Add(a, Scale(ab, d1 / (d1 - d3))));
  }

  const CudaVec3 cp = Sub(point, c);
  const double d5 = Dot(ab, cp);
  const double d6 = Dot(ac, cp);
  if (d6 >= 0.0 && d5 <= d6) {
    return DistancePointPoint(point, c);
  }

  const double vb = d5 * d2 - d1 * d6;
  if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
    return DistancePointPoint(point, Add(a, Scale(ac, d2 / (d2 - d6))));
  }

  const double va = d3 * d6 - d5 * d4;
  if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
    const double w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
    return DistancePointPoint(point, Add(b, Scale(Sub(c, b), w)));
  }

  return fabs(Dot(ap, normal) / Norm(normal));
}

__device__ double PointTriangleDistanceAt(const CudaPointTrianglePrimitive& primitive, double t) {
  return DistancePointTriangle(PositionAt(primitive.point, t),
                               PositionAt(primitive.triangle_v0, t),
                               PositionAt(primitive.triangle_v1, t),
                               PositionAt(primitive.triangle_v2, t));
}

__device__ double EdgeEdgeDistanceAt(const CudaEdgeEdgePrimitive& primitive, double t) {
  return DistanceSegmentSegment(PositionAt(primitive.edge_a0, t),
                                PositionAt(primitive.edge_a1, t),
                                PositionAt(primitive.edge_b0, t),
                                PositionAt(primitive.edge_b1, t));
}

__device__ double PointTriangleMotionRadius(const CudaPointTrianglePrimitive& primitive,
                                            double interval_t0,
                                            double interval_t1) {
  const double point_displacement =
      MaxDisplacementFromMidpoint(primitive.point, interval_t0, interval_t1);
  const double triangle_displacement =
      fmax(MaxDisplacementFromMidpoint(primitive.triangle_v0, interval_t0, interval_t1),
           fmax(MaxDisplacementFromMidpoint(primitive.triangle_v1, interval_t0, interval_t1),
                MaxDisplacementFromMidpoint(primitive.triangle_v2, interval_t0, interval_t1)));
  return point_displacement + triangle_displacement;
}

__device__ double EdgeEdgeMotionRadius(const CudaEdgeEdgePrimitive& primitive,
                                       double interval_t0,
                                       double interval_t1) {
  const double edge_a_displacement =
      fmax(MaxDisplacementFromMidpoint(primitive.edge_a0, interval_t0, interval_t1),
           MaxDisplacementFromMidpoint(primitive.edge_a1, interval_t0, interval_t1));
  const double edge_b_displacement =
      fmax(MaxDisplacementFromMidpoint(primitive.edge_b0, interval_t0, interval_t1),
           MaxDisplacementFromMidpoint(primitive.edge_b1, interval_t0, interval_t1));
  return edge_a_displacement + edge_b_displacement;
}

__device__ CudaPrimitiveResult MakeCudaResult(int status,
                                             std::uint32_t family_mask,
                                             double interval_t0,
                                             double interval_t1) {
  CudaPrimitiveResult result{};
  result.status = status;
  result.covered_feature_mask = status == kCudaStatusSeparation ? family_mask : 0U;
  result.interval_t0 = interval_t0;
  result.interval_t1 = interval_t1;
  result.toi_upper = interval_t1;
  result.safe_margin_lb = 0.0;
  result.witness_family = 0;
  result.witness_id_a = -1;
  result.witness_id_b = -1;
  result.reason_code = kCudaReasonNone;
  result.next_refinement_mode = kCudaRefinementNone;
  return result;
}

template <typename Primitive, typename DistanceFn, typename MotionRadiusFn>
__device__ CudaPrimitiveResult EvaluateIntervalIterative(const Primitive& primitive,
                                                        double interval_t0,
                                                        double interval_t1,
                                                        CudaConfig config,
                                                        std::uint32_t family_mask,
                                                        std::uint8_t witness_family,
                                                        std::int64_t witness_id_a,
                                                        std::int64_t witness_id_b,
                                                        DistanceFn distance_fn,
                                                        MotionRadiusFn motion_radius_fn) {
  CudaInterval stack[128];
  int stack_size = 0;
  stack[stack_size++] = {interval_t0, interval_t1, 0};
  double safe_margin_lb = 1.0e300;

  while (stack_size > 0) {
    const CudaInterval interval = stack[--stack_size];
    const double mid = 0.5 * (interval.t0 + interval.t1);
    const double distance0 = distance_fn(primitive, interval.t0);
    const double distance_mid = distance_fn(primitive, mid);
    const double distance1 = distance_fn(primitive, interval.t1);
    if (distance0 <= config.eps_space || distance_mid <= config.eps_space ||
        distance1 <= config.eps_space) {
      CudaPrimitiveResult collision =
          MakeCudaResult(kCudaStatusCollision, family_mask, interval.t0, interval.t1);
      collision.toi_upper = distance0 <= config.eps_space
                                ? interval.t0
                                : (distance_mid <= config.eps_space ? mid : interval.t1);
      collision.witness_family = witness_family;
      collision.witness_id_a = witness_id_a;
      collision.witness_id_b = witness_id_b;
      return collision;
    }

    const double lower_bound =
        distance_mid - motion_radius_fn(primitive, interval.t0, interval.t1);
    if (lower_bound > config.eps_space) {
      safe_margin_lb = fmin(safe_margin_lb, fmax(0.0, lower_bound - config.eps_space));
      continue;
    }

    if (interval.depth >= config.max_subdivision_depth ||
        interval.t1 - interval.t0 <= config.eps_time || stack_size + 2 >= 128) {
      CudaPrimitiveResult undecided =
          MakeCudaResult(kCudaStatusUndecided, family_mask, interval.t0, interval.t1);
      undecided.reason_code = kCudaReasonMaxSubdivisionDepth;
      undecided.next_refinement_mode = kCudaRefinementBisectInterval;
      return undecided;
    }

    const std::uint16_t child_depth = static_cast<std::uint16_t>(interval.depth + 1);
    stack[stack_size++] = {mid, interval.t1, child_depth};
    stack[stack_size++] = {interval.t0, mid, child_depth};
  }

  CudaPrimitiveResult separation =
      MakeCudaResult(kCudaStatusSeparation, family_mask, interval_t0, interval_t1);
  separation.safe_margin_lb = safe_margin_lb < 1.0e299 ? safe_margin_lb : 0.0;
  return separation;
}

__global__ void PointTriangleKernel(const CudaPointTrianglePrimitive* primitives,
                                    int primitive_count,
                                    double interval_t0,
                                    double interval_t1,
                                    CudaConfig config,
                                    CudaPrimitiveResult* results) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= primitive_count) {
    return;
  }
  const CudaPointTrianglePrimitive primitive = primitives[index];
  results[index] = EvaluateIntervalIterative(primitive,
                                             interval_t0,
                                             interval_t1,
                                             config,
                                             kCudaFamilyPointTriangle,
                                             kCudaFamilyPointTriangle,
                                             primitive.point_id,
                                             primitive.triangle_id,
                                             PointTriangleDistanceAt,
                                             PointTriangleMotionRadius);
}

__global__ void EdgeEdgeKernel(const CudaEdgeEdgePrimitive* primitives,
                              int primitive_count,
                              double interval_t0,
                              double interval_t1,
                              CudaConfig config,
                              CudaPrimitiveResult* results) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= primitive_count) {
    return;
  }
  const CudaEdgeEdgePrimitive primitive = primitives[index];
  results[index] = EvaluateIntervalIterative(primitive,
                                             interval_t0,
                                             interval_t1,
                                             config,
                                             kCudaFamilyEdgeEdge,
                                             kCudaFamilyEdgeEdge,
                                             primitive.edge_a_id,
                                             primitive.edge_b_id,
                                             EdgeEdgeDistanceAt,
                                             EdgeEdgeMotionRadius);
}

Status CudaStatus(cudaError_t error, const char* label) {
  if (error == cudaSuccess) {
    return Status::Ok();
  }
  return Status::Error(std::string(label) + ": " + cudaGetErrorString(error));
}

CudaVec3 ToCudaVec3(const std::array<double, 3>& value) {
  return {value[0], value[1], value[2]};
}

CudaTrajectory ToCudaTrajectory(const LinearVertexTrajectory& trajectory) {
  return {
      trajectory.feature_id,
      ToCudaVec3(trajectory.position_t0),
      ToCudaVec3(trajectory.position_t1),
  };
}

CudaPointTrianglePrimitive ToCudaPrimitive(const PointTriangleIntervalPrimitive& primitive) {
  return {
      primitive.point_id,
      primitive.triangle_id,
      ToCudaTrajectory(primitive.point),
      ToCudaTrajectory(primitive.triangle_v0),
      ToCudaTrajectory(primitive.triangle_v1),
      ToCudaTrajectory(primitive.triangle_v2),
  };
}

CudaEdgeEdgePrimitive ToCudaPrimitive(const EdgeEdgeIntervalPrimitive& primitive) {
  return {
      primitive.edge_a_id,
      primitive.edge_b_id,
      ToCudaTrajectory(primitive.edge_a0),
      ToCudaTrajectory(primitive.edge_a1),
      ToCudaTrajectory(primitive.edge_b0),
      ToCudaTrajectory(primitive.edge_b1),
  };
}

PrimitiveIntervalResult FromCudaResult(const CudaPrimitiveResult& cuda_result) {
  PrimitiveIntervalResult result;
  result.status = static_cast<CertificateStatus>(cuda_result.status);
  result.covered_feature_mask = cuda_result.covered_feature_mask;
  result.interval_t0 = cuda_result.interval_t0;
  result.interval_t1 = cuda_result.interval_t1;
  result.toi_upper = cuda_result.toi_upper;
  result.safe_margin_lb = cuda_result.safe_margin_lb;
  result.witness_family = cuda_result.witness_family;
  result.witness_id_a = cuda_result.witness_id_a;
  result.witness_id_b = cuda_result.witness_id_b;
  result.reason_code = cuda_result.reason_code;
  result.next_refinement_mode =
      static_cast<CertificateRefinementMode>(cuda_result.next_refinement_mode);
  return result;
}

CudaConfig ToCudaConfig(const CertificateEngineConfig& config) {
  return {config.eps_time, config.eps_space, config.max_subdivision_depth};
}

template <typename HostPrimitive, typename CudaPrimitive>
Status RunCudaBatch(const std::vector<HostPrimitive>& primitives,
                    double interval_t0,
                    double interval_t1,
                    const CertificateEngineConfig& config,
                    std::vector<PrimitiveIntervalResult>* results) {
  if (results == nullptr) {
    return Status::Error("CUDA exact result output pointer is null");
  }
  results->clear();
  if (primitives.empty()) {
    return Status::Ok();
  }

  std::vector<CudaPrimitive> host_primitives;
  host_primitives.reserve(primitives.size());
  for (const HostPrimitive& primitive : primitives) {
    host_primitives.push_back(ToCudaPrimitive(primitive));
  }
  std::vector<CudaPrimitiveResult> host_results(primitives.size());

  CudaPrimitive* device_primitives = nullptr;
  CudaPrimitiveResult* device_results = nullptr;
  const std::size_t primitive_bytes = sizeof(CudaPrimitive) * host_primitives.size();
  const std::size_t result_bytes = sizeof(CudaPrimitiveResult) * host_results.size();

  if (auto status = CudaStatus(cudaMalloc(&device_primitives, primitive_bytes),
                               "cudaMalloc primitives");
      !status.ok) {
    return status;
  }
  if (auto status = CudaStatus(cudaMalloc(&device_results, result_bytes),
                               "cudaMalloc results");
      !status.ok) {
    cudaFree(device_primitives);
    return status;
  }
  if (auto status = CudaStatus(cudaMemcpy(device_primitives,
                                          host_primitives.data(),
                                          primitive_bytes,
                                          cudaMemcpyHostToDevice),
                               "cudaMemcpy primitives H2D");
      !status.ok) {
    cudaFree(device_primitives);
    cudaFree(device_results);
    return status;
  }

  constexpr int kBlockSize = 128;
  const int primitive_count = static_cast<int>(host_primitives.size());
  const int grid_size = (primitive_count + kBlockSize - 1) / kBlockSize;
  if constexpr (std::is_same_v<CudaPrimitive, CudaPointTrianglePrimitive>) {
    PointTriangleKernel<<<grid_size, kBlockSize>>>(device_primitives,
                                                  primitive_count,
                                                  interval_t0,
                                                  interval_t1,
                                                  ToCudaConfig(config),
                                                  device_results);
  } else {
    EdgeEdgeKernel<<<grid_size, kBlockSize>>>(device_primitives,
                                             primitive_count,
                                             interval_t0,
                                             interval_t1,
                                             ToCudaConfig(config),
                                             device_results);
  }
  if (auto status = CudaStatus(cudaGetLastError(), "CUDA exact kernel launch"); !status.ok) {
    cudaFree(device_primitives);
    cudaFree(device_results);
    return status;
  }
  if (auto status = CudaStatus(cudaDeviceSynchronize(), "CUDA exact kernel synchronize");
      !status.ok) {
    cudaFree(device_primitives);
    cudaFree(device_results);
    return status;
  }
  if (auto status = CudaStatus(cudaMemcpy(host_results.data(),
                                          device_results,
                                          result_bytes,
                                          cudaMemcpyDeviceToHost),
                               "cudaMemcpy results D2H");
      !status.ok) {
    cudaFree(device_primitives);
    cudaFree(device_results);
    return status;
  }

  cudaFree(device_primitives);
  cudaFree(device_results);

  results->reserve(host_results.size());
  for (const CudaPrimitiveResult& cuda_result : host_results) {
    results->push_back(FromCudaResult(cuda_result));
  }
  return Status::Ok();
}

Status ComparePrimitiveResults(const PrimitiveIntervalResult& cpu,
                               const PrimitiveIntervalResult& gpu,
                               double eps_cert,
                               const char* label) {
  if (cpu.status != gpu.status) {
    return Status::Error(std::string(label) + " status mismatch");
  }
  if (cpu.covered_feature_mask != gpu.covered_feature_mask) {
    return Status::Error(std::string(label) + " covered feature mask mismatch");
  }
  if (cpu.status == CertificateStatus::kCollision) {
    if (std::abs(cpu.toi_upper - gpu.toi_upper) > eps_cert) {
      return Status::Error(std::string(label) + " TOI mismatch");
    }
    if (cpu.witness_family != gpu.witness_family || cpu.witness_id_a != gpu.witness_id_a ||
        cpu.witness_id_b != gpu.witness_id_b) {
      return Status::Error(std::string(label) + " witness mismatch");
    }
  }
  if (cpu.status == CertificateStatus::kSeparation &&
      std::abs(cpu.safe_margin_lb - gpu.safe_margin_lb) > eps_cert) {
    return Status::Error(std::string(label) + " safe margin mismatch");
  }
  if (cpu.status == CertificateStatus::kUndecided &&
      (cpu.reason_code != gpu.reason_code ||
       cpu.next_refinement_mode != gpu.next_refinement_mode)) {
    return Status::Error(std::string(label) + " undecided refinement mismatch");
  }
  return Status::Ok();
}

}  // namespace

bool IsCudaExactBuilt() {
  return true;
}

Status EvaluatePointTriangleBatchCuda(const std::vector<PointTriangleIntervalPrimitive>& primitives,
                                      double interval_t0,
                                      double interval_t1,
                                      const CertificateEngineConfig& config,
                                      std::vector<PrimitiveIntervalResult>* results) {
  return RunCudaBatch<PointTriangleIntervalPrimitive, CudaPointTrianglePrimitive>(
      primitives,
      interval_t0,
      interval_t1,
      config,
      results);
}

Status EvaluateEdgeEdgeBatchCuda(const std::vector<EdgeEdgeIntervalPrimitive>& primitives,
                                 double interval_t0,
                                 double interval_t1,
                                 const CertificateEngineConfig& config,
                                 std::vector<PrimitiveIntervalResult>* results) {
  return RunCudaBatch<EdgeEdgeIntervalPrimitive, CudaEdgeEdgePrimitive>(primitives,
                                                                       interval_t0,
                                                                       interval_t1,
                                                                       config,
                                                                       results);
}

Status CrossCheckCpuCudaExact(const std::vector<PointTriangleIntervalPrimitive>& point_triangles,
                              const std::vector<EdgeEdgeIntervalPrimitive>& edge_edges,
                              double interval_t0,
                              double interval_t1,
                              const CertificateEngineConfig& config,
                              double eps_cert) {
  if (!std::isfinite(eps_cert) || eps_cert < 0.0) {
    return Status::Error("eps_cert must be finite and non-negative");
  }

  std::vector<PrimitiveIntervalResult> gpu_point_triangle_results;
  if (auto status = EvaluatePointTriangleBatchCuda(point_triangles,
                                                   interval_t0,
                                                   interval_t1,
                                                   config,
                                                   &gpu_point_triangle_results);
      !status.ok) {
    return status;
  }
  for (std::size_t i = 0; i < point_triangles.size(); ++i) {
    PrimitiveIntervalResult cpu_result;
    if (auto status = EvaluatePointTriangleInterval(point_triangles[i],
                                                    interval_t0,
                                                    interval_t1,
                                                    config,
                                                    &cpu_result);
        !status.ok) {
      return status;
    }
    if (auto status = ComparePrimitiveResults(cpu_result,
                                              gpu_point_triangle_results[i],
                                              eps_cert,
                                              "point-triangle CPU/CUDA cross-check");
        !status.ok) {
      return status;
    }
  }

  std::vector<PrimitiveIntervalResult> gpu_edge_edge_results;
  if (auto status = EvaluateEdgeEdgeBatchCuda(edge_edges,
                                              interval_t0,
                                              interval_t1,
                                              config,
                                              &gpu_edge_edge_results);
      !status.ok) {
    return status;
  }
  for (std::size_t i = 0; i < edge_edges.size(); ++i) {
    PrimitiveIntervalResult cpu_result;
    if (auto status =
            EvaluateEdgeEdgeInterval(edge_edges[i], interval_t0, interval_t1, config, &cpu_result);
        !status.ok) {
      return status;
    }
    if (auto status = ComparePrimitiveResults(cpu_result,
                                              gpu_edge_edge_results[i],
                                              eps_cert,
                                              "edge-edge CPU/CUDA cross-check");
        !status.ok) {
      return status;
    }
  }
  return Status::Ok();
}

}  // namespace p2cccd
