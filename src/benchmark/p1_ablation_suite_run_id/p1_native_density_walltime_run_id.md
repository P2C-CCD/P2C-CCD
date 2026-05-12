# P1-3 Native Density Wall-Time Sweep

## Scope

- Reuses the existing density-advantage checkpoint; no retraining is performed in this P1 run.
- Rebuilds the four-source eval workload at each density and runs ORT inference plus native C++ dense-group early-stop.
- Reports native replay/detection wall-time for the dense oracle driver, not full physical simulation solver wall-time.

## Results

| Density | Eval queries | Candidates | Head | ORT provider | Inference ms | C++ ms | E2E ms | Calls | Call red. | Work red. | FN | Break-even ms/work |
| ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `16` | `528` | `8448` | `cost_aware` | `CUDAExecutionProvider` | `13.837` | `0.775` | `14.612` | `2448` | `71.02%` | `93.02%` | `0` | `0.000062661` |
| `32` | `528` | `16896` | `cost_aware` | `CUDAExecutionProvider` | `29.958` | `1.854` | `31.812` | `4429` | `73.79%` | `95.34%` | `0` | `0.000066551` |
| `64` | `528` | `33792` | `cost_aware` | `CUDAExecutionProvider` | `93.470` | `3.654` | `97.124` | `8370` | `75.23%` | `94.61%` | `0` | `0.000102371` |
| `128` | `528` | `67584` | `cost_aware` | `CUDAExecutionProvider` | `256.717` | `7.724` | `264.441` | `16297` | `75.89%` | `96.00%` | `0` | `0.000137347` |
| `256` | `528` | `135168` | `risk_proximity_hybrid` | `CUDAExecutionProvider` | `272.381` | `19.176` | `291.557` | `32166` | `76.20%` | `95.97%` | `0` | `0.000075743` |
| `512` | `528` | `270336` | `risk_proximity_hybrid` | `CUDAExecutionProvider` | `391.705` | `35.389` | `427.094` | `63907` | `76.36%` | `87.58%` | `0` | `0.000060788` |
| `1024` | `528` | `540672` | `risk_proximity_hybrid` | `CUDAExecutionProvider` | `383.717` | `80.193` | `463.910` | `127386` | `76.44%` | `84.34%` | `0` | `0.000034281` |
| `2304` | `528` | `1216512` | `risk_proximity_hybrid` | `CUDAExecutionProvider` | `771.475` | `251.688` | `1023.163` | `286218` | `76.47%` | `83.54%` | `0` | `0.000033929` |

## Conclusion

- First density with FN=0 and exact-work reduction >= 99%: `None`.
- The native C++ early-stop path preserves FN=0 for all reported density rows.
- Wall-time remains inference dominated at the largest densities, which supports keeping ORT/TensorRT and C++ scheduling in the final pipeline.
