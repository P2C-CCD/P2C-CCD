# P2 Ablation Suite

Run identifier: run_id

## Artifacts

- P2-1: `p2_training_source_ablation_run_id.*`
- P2-2: `p2_patch_slab_proxy_native_run_id.*`
- P2-3: `p2_backend_stack_ablation_run_id.*`

## Guardrails

- P2-1 is sampled retraining with CPU PyTorch on this workstation.
- P2-2 uses the production C++/OptiX proxy-candidate wrapper where available, with analytic swept-sphere oracle certificates.
- P2-3 reports actual ORT provider names; TensorRT preference is not counted as TensorRT unless the active provider is `TensorrtExecutionProvider`.

## Headline Checks

- P2-1 variants: 8; per-source rows: 32; max FN: 0.
- P2-2 selected safe rows: 9; min selected recall: 1.0000; max selected FN: 0.
- P2-3 rows: 12; C++ binding available: True.
