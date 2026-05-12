# Near-Term Execution Gate

This note covers TODO 127-128.

## Purpose

The near-term execution gate verifies the first ordered end-to-end pipeline:

1. Run `RTExact` from conservative RT candidates to exact certificates without STPF.
2. Require `final_fn_zero`, `candidate_recall == 1.0`, queue conservation, and certificate coverage.
3. Only after `RTExact` passes, run `RTSTPFExact` with the lightweight STPF v1 MLP.
4. Require the same safety gates for STPF v1, plus monotonic scheduling safety.

## Runner

```text
src/python/p2cccd/bench/near_term_execution.py
```

Public entry points:

```python
from p2cccd.bench import (
    NearTermExecutionConfig,
    run_near_term_execution_gate,
    write_near_term_execution_gate_json,
)
```

## Generated Output

The current local output is:

```text
src/outputs/near_term_execution_gate.json
```

The output records:

- dataset seed and query count,
- `RTExact` pass/fail,
- STPF v1 run/pass status,
- candidate, work item, and certificate counts,
- collision/separation/undecided certificate counts,
- `final_fn_zero`,
- `candidate_recall`,
- queue conservation,
- monotonic safety,
- RT/proposal/exact/total timing breakdown.

## Reproduce

From the repository root:

```powershell
conda activate cudadev
python -m pytest src\tests\python\test_near_term_execution_order.py -q
```

To regenerate the JSON manually:

```powershell
conda activate cudadev
python - <<'PY'
from pathlib import Path
import sys
project = Path("src").resolve()
sys.path.insert(0, str(project / "python"))
from p2cccd.bench import NearTermExecutionConfig, run_near_term_execution_gate, write_near_term_execution_gate_json
result = run_near_term_execution_gate(NearTermExecutionConfig())
write_near_term_execution_gate_json(project / "outputs" / "near_term_execution_gate.json", result)
PY
```
