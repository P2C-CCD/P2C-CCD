# Native dense group scheduling and exact early-stop wall-time table

## Protocol

- Python is responsible only fordescription, load shard, call ORT andsummarizereport; candidatedescription, group early-stop, fallback statisticsin `p2cccd_cpp.run_native_dense_group_exact_early_stop` description.
- Proposal inference descriptionuse ORT, defaultdescription `TensorrtExecutionProvider`, automatically on failure CUDA/CPU fallback.
- Exact layercurrentdescriptionuse dense shard within analytic proxy exact oracle / exact-cost trace as native early-stop driver; description TI/CUDA primitive exact hot path descriptioninsameconnectdescriptionunderdescription exact payload.
- STPF only determines candidate evaluation order / fallback policy, does not directly output final collision truth.
- default learned policy descriptionuse validation/source-aware head selection: descriptiondata sourcein `priority_only`, `cost_aware`, `risk_proximity_hybrid` descriptionfixedselect, avoid hard-negative group ondescription cost-aware head description random description.

## description

| Dataset | Policy head | Groups | Candidates | Positive groups | ORT provider | Inference ms | C++ native ms | E2E RTSTPF ms | Exact calls | Call reduction | Work reduction | FN | Break-even ms/work |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `common_modeling_large` | `priority_only` | `192` | `663552` | `160` | `CUDAExecutionProvider` | `290.051` | `115.736` | `405.787` | `110893` | `83.2880%` | `92.4183%` | `0` | `0.000037028` |
| `fusion360_full_assembly` | `cost_aware` | `1024` | `1048576` | `768` | `CUDAExecutionProvider` | `482.572` | `209.098` | `691.671` | `263974` | `74.8255%` | `72.6747%` | `0` | `0.000031908` |
| `rtstpf_advantage_v4` | `priority_only` | `864` | `1990656` | `689` | `CUDAExecutionProvider` | `950.475` | `434.190` | `1384.666` | `404847` | `79.6626%` | `84.8089%` | `0` | `0.000037991` |
| `shapenet_ood_dense` | `cost_aware` | `648` | `1492992` | `378` | `CUDAExecutionProvider` | `668.486` | `310.285` | `978.771` | `622944` | `58.2755%` | `69.6289%` | `0` | `0.000061826` |

## splitdescriptionMetrics

| Dataset | Parse ms | Schedule ms | Early-stop exact ms | Native total ms | NoProposal calls | NoProposal work | Learned work | Fallback calls | Interval miss | TP/TN/FP/FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `common_modeling_large` | `14.155` | `59.581` | `10.837` | `113.397` | `663552` | `11789673.8657` | `893854.1917` | `48264` | `141` | `160/32/0/0` |
| `fusion360_full_assembly` | `21.558` | `118.608` | `18.245` | `204.856` | `1048576` | `29644420.1402` | `8100434.4739` | `259070` | `1062` | `768/256/0/0` |
| `rtstpf_advantage_v4` | `50.648` | `264.758` | `37.061` | `429.926` | `1990656` | `42843311.4314` | `6508352.2407` | `382091` | `958` | `689/175/0/0` |
| `shapenet_ood_dense` | `36.933` | `203.697` | `6.951` | `304.922` | `1492992` | `22611854.9354` | `6867474.1820` | `583362` | `486` | `378/270/0/0` |

## Conclusion

- thisdescription P0 No. 1 description native dense group hot path: selection Filewritedescriptionand Python per-candidate scheduling descriptionfromdescriptionPathdescription.
- current exact driver descriptionis proxy oracle/cost trace, thereforedescriptionas native scheduling + early-stop realdescription; if used as final SOTA primitive wall-time description, description exact payload description Tight-Inclusion or CUDA primitive exact kernel.
- `Break-even ms/work` descriptionreal exact kernel each work unit descriptionwhen, learned early-stop description exact work descriptionwithdescription ORT + C++ scheduling overhead.
