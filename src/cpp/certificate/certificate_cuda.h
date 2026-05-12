#pragma once

#include "certificate/certificate_engine.h"
#include "common/status.h"

#include <vector>

namespace p2cccd {

bool IsCudaExactBuilt();
Status EvaluatePointTriangleBatchCuda(const std::vector<PointTriangleIntervalPrimitive>& primitives,
                                      double interval_t0,
                                      double interval_t1,
                                      const CertificateEngineConfig& config,
                                      std::vector<PrimitiveIntervalResult>* results);
Status EvaluateEdgeEdgeBatchCuda(const std::vector<EdgeEdgeIntervalPrimitive>& primitives,
                                 double interval_t0,
                                 double interval_t1,
                                 const CertificateEngineConfig& config,
                                 std::vector<PrimitiveIntervalResult>* results);
Status CrossCheckCpuCudaExact(const std::vector<PointTriangleIntervalPrimitive>& point_triangles,
                              const std::vector<EdgeEdgeIntervalPrimitive>& edge_edges,
                              double interval_t0,
                              double interval_t1,
                              const CertificateEngineConfig& config,
                              double eps_cert);

}  // namespace p2cccd
