# GPU / OptiX Environment Notes

This document separates large GPU benchmark reproduction from the CPU smoke
checks in the repository root `environment.yml`.

## Scope

`environment.yml` is intentionally a CPU smoke / CI environment. It supports
Python contracts, quality-gate inventory checks, manifest verification, and C++
CTest smoke builds.

Full GPU benchmark reproduction additionally needs local vendor SDKs and
ignored model/data artifacts. Do not treat a passing CPU smoke run as evidence
that CUDA, OptiX, TensorRT, or ONNX Runtime TensorRT execution is available.

## Required Local Components

- CUDA toolkit and a compatible NVIDIA driver for CUDA benchmarks.
- OptiX SDK for optional RT candidate traversal builds.
- TensorRT SDK when using ONNX Runtime TensorRT execution providers.
- ONNX Runtime build/package compatible with the intended execution provider.
- Checkpoints and shards listed in `model_artifacts_manifest.md`.

## Path Conventions

Prefer environment variables over machine-specific absolute paths:

```powershell
$env:P2CCCD_OPTIX_ROOT = "<path-to-optix-sdk>"
$env:P2CCCD_TENSORRT_ROOT = "<path-to-tensorrt-sdk>"
$env:P2CCCD_PYTHON = "python"
```

Python commands should be launched after:

```powershell
conda activate cudadev
```

PowerShell benchmark wrappers default to `python`, so the activated environment
controls the interpreter. Override `-Python` or `-PythonExe` only when running a
known alternate interpreter.

## Verification Boundary

Before reporting GPU/OptiX results, record:

- GPU model and driver version.
- CUDA toolkit version.
- OptiX SDK version if used.
- TensorRT version if used.
- ONNX Runtime provider list.
- Artifact hashes from `model_artifacts_manifest.json`.
