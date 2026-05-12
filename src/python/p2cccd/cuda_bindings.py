from __future__ import annotations

from dataclasses import dataclass
import importlib
from typing import Any, Sequence


CUDA_BINDING_ENTRYPOINTS = (
    "is_cuda_exact_built",
    "cuda_binding_status",
    "evaluate_point_triangle_batch_cuda",
    "evaluate_edge_edge_batch_cuda",
    "cross_check_cpu_cuda_exact",
)


@dataclass(frozen=True, slots=True)
class CudaBindingStatus:
    cpp_module_available: bool
    cuda_exact_built: bool
    host_batch_exact_api: bool
    device_pointer_abi: bool
    backend_name: str
    fallback_reason: str

    @property
    def ready_for_host_batches(self) -> bool:
        return self.cpp_module_available and self.host_batch_exact_api

    @property
    def ready_for_cuda_execution(self) -> bool:
        return self.ready_for_host_batches and self.cuda_exact_built


def _load_cpp_module() -> Any | None:
    try:
        return importlib.import_module("p2cccd_cpp")
    except ImportError:
        return None


def _has_entrypoints(module: Any) -> bool:
    return all(callable(getattr(module, name, None)) for name in CUDA_BINDING_ENTRYPOINTS)


def get_cuda_binding_status(cpp_module: Any | None = None) -> CudaBindingStatus:
    module = cpp_module if cpp_module is not None else _load_cpp_module()
    if module is None:
        return CudaBindingStatus(
            cpp_module_available=False,
            cuda_exact_built=False,
            host_batch_exact_api=False,
            device_pointer_abi=False,
            backend_name="unavailable",
            fallback_reason="p2cccd_cpp is not importable",
        )
    if not _has_entrypoints(module):
        return CudaBindingStatus(
            cpp_module_available=True,
            cuda_exact_built=False,
            host_batch_exact_api=False,
            device_pointer_abi=False,
            backend_name="missing_entrypoints",
            fallback_reason="p2cccd_cpp is missing CUDA-aware binding entrypoints",
        )
    raw_status = dict(module.cuda_binding_status())
    cuda_exact_built = bool(raw_status.get("cuda_exact_built", module.is_cuda_exact_built()))
    return CudaBindingStatus(
        cpp_module_available=True,
        cuda_exact_built=cuda_exact_built,
        host_batch_exact_api=bool(raw_status.get("host_batch_exact_api", True)),
        device_pointer_abi=bool(raw_status.get("device_pointer_abi", False)),
        backend_name=str(raw_status.get("backend_name", "cuda_exact" if cuda_exact_built else "cuda_exact_stub")),
        fallback_reason="" if cuda_exact_built else "CUDA exact backend is not built; use CPU exact fallback",
    )


def is_cuda_exact_binding_available(*, require_built: bool = True) -> bool:
    status = get_cuda_binding_status()
    if require_built:
        return status.ready_for_cuda_execution
    return status.ready_for_host_batches


def _require_cuda_module(cpp_module: Any | None, *, require_built: bool) -> Any:
    module = cpp_module if cpp_module is not None else _load_cpp_module()
    status = get_cuda_binding_status(module)
    if not status.ready_for_host_batches:
        raise RuntimeError(status.fallback_reason)
    if require_built and not status.cuda_exact_built:
        raise RuntimeError(status.fallback_reason)
    return module


def evaluate_point_triangle_batch_cuda(
    primitives: Sequence[Any],
    interval_t0: float,
    interval_t1: float,
    config: Any,
    *,
    cpp_module: Any | None = None,
    require_built: bool = True,
) -> tuple[Any, ...]:
    module = _require_cuda_module(cpp_module, require_built=require_built)
    return tuple(
        module.evaluate_point_triangle_batch_cuda(
            list(primitives),
            float(interval_t0),
            float(interval_t1),
            config,
        )
    )


def evaluate_edge_edge_batch_cuda(
    primitives: Sequence[Any],
    interval_t0: float,
    interval_t1: float,
    config: Any,
    *,
    cpp_module: Any | None = None,
    require_built: bool = True,
) -> tuple[Any, ...]:
    module = _require_cuda_module(cpp_module, require_built=require_built)
    return tuple(
        module.evaluate_edge_edge_batch_cuda(
            list(primitives),
            float(interval_t0),
            float(interval_t1),
            config,
        )
    )


def cross_check_cpu_cuda_exact(
    point_triangles: Sequence[Any],
    edge_edges: Sequence[Any],
    interval_t0: float,
    interval_t1: float,
    config: Any,
    eps_cert: float,
    *,
    cpp_module: Any | None = None,
    require_built: bool = True,
) -> bool:
    module = _require_cuda_module(cpp_module, require_built=require_built)
    return bool(
        module.cross_check_cpu_cuda_exact(
            list(point_triangles),
            list(edge_edges),
            float(interval_t0),
            float(interval_t1),
            config,
            float(eps_cert),
        )
    )
