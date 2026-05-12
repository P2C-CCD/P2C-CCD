#include "certificate/certificate_cuda.h"

namespace p2cccd {

bool IsCudaExactBuilt() {
  return false;
}

Status EvaluatePointTriangleBatchCuda(
    const std::vector<PointTriangleIntervalPrimitive>& /*primitives*/,
    double /*interval_t0*/,
    double /*interval_t1*/,
    const CertificateEngineConfig& /*config*/,
    std::vector<PrimitiveIntervalResult>* /*results*/) {
  return Status::Error("CUDA exact backend was not built");
}

Status EvaluateEdgeEdgeBatchCuda(const std::vector<EdgeEdgeIntervalPrimitive>& /*primitives*/,
                                 double /*interval_t0*/,
                                 double /*interval_t1*/,
                                 const CertificateEngineConfig& /*config*/,
                                 std::vector<PrimitiveIntervalResult>* /*results*/) {
  return Status::Error("CUDA exact backend was not built");
}

Status CrossCheckCpuCudaExact(
    const std::vector<PointTriangleIntervalPrimitive>& /*point_triangles*/,
    const std::vector<EdgeEdgeIntervalPrimitive>& /*edge_edges*/,
    double /*interval_t0*/,
    double /*interval_t1*/,
    const CertificateEngineConfig& /*config*/,
    double /*eps_cert*/) {
  return Status::Error("CUDA exact backend was not built");
}

}  // namespace p2cccd
