# P2-3 Backend Stack Ablation

## Scope

P2-3 backend stack timing on common_modeling_large and fusion360_full_assembly dense eval shards.

## Results

| case | backend | scope | provider_actual | row_count | proposal_ms | schedule_ms | total_detection_ms | rows_per_second | fn | exact_calls |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| common_modeling_large | python_rows_torch_reference | sampled_reference | torch_cpu | 65536 | 1624.0 | 0.0000 | 1624.0 | 40355.8 |  |  |
| common_modeling_large | torch_array_cpu | full_dense_eval | torch_cpu_cuda_available=False | 663552 | 824.3946 | 0.0000 | 824.3946 | 804896.1 |  |  |
| common_modeling_large | ort_cpu | full_dense_eval | CPUExecutionProvider | 663552 | 3352.1 | 0.0000 | 3352.1 | 197952.8 |  |  |
| common_modeling_large | ort_cuda | full_dense_eval | CUDAExecutionProvider | 663552 | 486.0709 | 0.0000 | 486.0709 | 1365134.2 |  |  |
| common_modeling_large | ort_tensorrt_preferred | full_dense_eval | CUDAExecutionProvider | 663552 | 281.4994 | 0.0000 | 281.4994 | 2357205.7 |  |  |
| common_modeling_large | ort_cuda_plus_cpp_scheduler | full_dense_eval | CUDAExecutionProvider | 663552 | 243.7831 | 114.1076 | 357.8907 | 1854063.3 | 0 | 110893 |
| fusion360_full_assembly | python_rows_torch_reference | sampled_reference | torch_cpu | 65536 | 1044.5 | 0.0000 | 1044.5 | 62743.6 |  |  |
| fusion360_full_assembly | torch_array_cpu | full_dense_eval | torch_cpu_cuda_available=False | 1048576 | 1285.6 | 0.0000 | 1285.6 | 815634.3 |  |  |
| fusion360_full_assembly | ort_cpu | full_dense_eval | CPUExecutionProvider | 1048576 | 4788.1 | 0.0000 | 4788.1 | 218997.7 |  |  |
| fusion360_full_assembly | ort_cuda | full_dense_eval | CUDAExecutionProvider | 1048576 | 746.4289 | 0.0000 | 746.4289 | 1404790.2 |  |  |
| fusion360_full_assembly | ort_tensorrt_preferred | full_dense_eval | CUDAExecutionProvider | 1048576 | 426.2231 | 0.0000 | 426.2231 | 2460157.6 |  |  |
| fusion360_full_assembly | ort_cuda_plus_cpp_scheduler | full_dense_eval | CUDAExecutionProvider | 1048576 | 376.4062 | 261.8594 | 638.2656 | 1642852.1 | 0 | 263974 |
