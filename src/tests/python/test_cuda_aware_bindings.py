from __future__ import annotations

import pytest

import p2cccd.cuda_bindings as cuda_bindings
from p2cccd.cuda_bindings import (
    cross_check_cpu_cuda_exact,
    evaluate_edge_edge_batch_cuda,
    evaluate_point_triangle_batch_cuda,
    get_cuda_binding_status,
)


class _FakeCudaModule:
    def __init__(self, *, built: bool) -> None:
        self._built = built

    def is_cuda_exact_built(self) -> bool:
        return self._built

    def cuda_binding_status(self) -> dict[str, object]:
        return {
            "cuda_exact_built": self._built,
            "host_batch_exact_api": True,
            "device_pointer_abi": False,
            "backend_name": "cuda_exact" if self._built else "cuda_exact_stub",
        }

    def evaluate_point_triangle_batch_cuda(self, primitives, interval_t0, interval_t1, config):
        return [("pt", len(primitives), interval_t0, interval_t1, config)]

    def evaluate_edge_edge_batch_cuda(self, primitives, interval_t0, interval_t1, config):
        return [("ee", len(primitives), interval_t0, interval_t1, config)]

    def cross_check_cpu_cuda_exact(self, point_triangles, edge_edges, interval_t0, interval_t1, config, eps_cert):
        return (
            len(point_triangles),
            len(edge_edges),
            interval_t0,
            interval_t1,
            config,
            eps_cert,
        ) == (1, 1, 0.0, 1.0, "cfg", 1.0e-6)


class _MissingEntrypointsModule:
    pass


def test_cuda_binding_status_reports_missing_cpp_module(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cuda_bindings, "_load_cpp_module", lambda: None)
    status = get_cuda_binding_status()
    assert status.cpp_module_available is False
    assert status.ready_for_host_batches is False
    assert "not importable" in status.fallback_reason


def test_cuda_binding_status_rejects_incomplete_cpp_module() -> None:
    status = get_cuda_binding_status(cpp_module=_MissingEntrypointsModule())
    assert status.cpp_module_available is True
    assert status.host_batch_exact_api is False
    assert status.backend_name == "missing_entrypoints"


def test_cuda_binding_wrappers_keep_stub_build_safe() -> None:
    fake = _FakeCudaModule(built=False)
    status = get_cuda_binding_status(cpp_module=fake)
    assert status.ready_for_host_batches is True
    assert status.ready_for_cuda_execution is False
    assert status.device_pointer_abi is False

    with pytest.raises(RuntimeError, match="not built"):
        evaluate_point_triangle_batch_cuda(["primitive"], 0.0, 1.0, "cfg", cpp_module=fake)


def test_cuda_binding_wrappers_forward_host_batches_when_built() -> None:
    fake = _FakeCudaModule(built=True)
    assert get_cuda_binding_status(cpp_module=fake).ready_for_cuda_execution is True

    assert evaluate_point_triangle_batch_cuda(["p0", "p1"], 0.0, 1.0, "cfg", cpp_module=fake) == (
        ("pt", 2, 0.0, 1.0, "cfg"),
    )
    assert evaluate_edge_edge_batch_cuda(["e0"], 0.25, 0.75, "cfg", cpp_module=fake) == (
        ("ee", 1, 0.25, 0.75, "cfg"),
    )
    assert (
        cross_check_cpu_cuda_exact(["pt"], ["ee"], 0.0, 1.0, "cfg", 1.0e-6, cpp_module=fake)
        is True
    )
