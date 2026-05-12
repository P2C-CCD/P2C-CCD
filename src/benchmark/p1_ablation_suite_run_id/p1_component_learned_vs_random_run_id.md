# Learned-vs-random ablation reinforcement report

## Protocol

- Purpose: description `learned STPF isdescriptionbetter than random scheduling`.
- descriptionoriginal dense groups: if positive group descriptionall candidate descriptionis positive, descriptionanydescription rank=1, description learned better than random.
- thereforedescription balanced hard-negative rank challenge: each group fixeddescription positive candidate anddescription high-cost negative candidate, descriptionusesame heldout shard, same checkpoint, same ORT TensorRT Output.
- descriptionis group-level scheduling/ranking, description STPF descriptionconnectdescription collision; description zero-FN description conservative exact scan/fallback guarantee.

## original Dense Group descriptionsplitdescription

| Dataset | Groups | Candidates | Positive groups | Mixed groups | Pure-positive groups | Positive fraction in positive groups | Informative for rank ablation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `common_modeling_large` | `192` | `663552` | `160` | `0` | `160` | `1.0000` | `False` |
| `fusion360_full_assembly` | `1024` | `1048576` | `768` | `0` | `768` | `1.0000` | `False` |
| `rtstpf_advantage_v4` | `864` | `1990656` | `689` | `0` | `689` | `1.0000` | `False` |
| `shapenet_ood_dense` | `648` | `1492992` | `378` | `0` | `378` | `1.0000` | `False` |

Conclusion: currentdescription dense shard original group descriptionis not candidate-level mixed group; descriptionconnectusedescription learned descriptionbetter than random. underdescription balanced hard-negative challenge descriptionis learned-vs-random hasdescriptionProtocol.

## Balanced Hard-Negative Rank Challenge

| Dataset | Groups | Candidates/group | Pos/group | Method | Exact calls | Call reduction | Work reduction | First-positive rank mean | p90 | Cost-weighted mean | FN |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `common_modeling_large` | `217` | `512` | `4` | `ValidationSelectedFullSTPF` | `13024` | `88.2776%` | `90.5403%` | `60.018` | `215.400` | `619.144` | `0` |
| `common_modeling_large` | `217` | `512` | `4` | `LearnedCostAware` | `13070` | `88.2362%` | `90.5050%` | `60.230` | `216.400` | `621.456` | `0` |
| `common_modeling_large` | `217` | `512` | `4` | `LearnedPriorityOnly` | `13024` | `88.2776%` | `90.5403%` | `60.018` | `215.400` | `619.144` | `0` |
| `common_modeling_large` | `217` | `512` | `4` | `IntervalOnly` | `32944` | `70.3485%` | `61.0662%` | `151.816` | `224.000` | `2548.252` | `0` |
| `common_modeling_large` | `217` | `512` | `4` | `RankingOnly` | `32042` | `71.1604%` | `61.8629%` | `147.659` | `210.000` | `2496.103` | `0` |
| `common_modeling_large` | `217` | `512` | `4` | `LearnedCalibrated` | `18706` | `83.1635%` | `86.6644%` | `86.203` | `265.000` | `872.825` | `0` |
| `common_modeling_large` | `217` | `512` | `4` | `RandomUniform(mean over seeds)` | `22728` | `79.5439%` | `79.5272%` | `104.735` | `224.900` | `1339.964` | `0` |
| `common_modeling_large` | `217` | `512` | `4` | `HeuristicCostLow` | `90969` | `18.1227%` | `25.6912%` | `419.212` | `447.800` | `4863.575` | `0` |
| `common_modeling_large` | `217` | `512` | `4` | `HeuristicCostHigh` | `3649` | `96.7157%` | `95.1536%` | `16.816` | `52.000` | `317.202` | `0` |
| `common_modeling_large` | `217` | `512` | `4` | `UncertaintyOnly` | `12999` | `88.3002%` | `90.5572%` | `59.903` | `214.800` | `618.038` | `0` |
| `common_modeling_large` | `217` | `512` | `4` | `OracleUpperBound` | `217` | `99.8047%` | `99.7428%` | `1.000` | `1.000` | `16.833` | `0` |
| `fusion360_full_assembly` | `512` | `512` | `4` | `ValidationSelectedFullSTPF` | `1694` | `99.3538%` | `98.9244%` | `3.309` | `6.000` | `170.591` | `0` |
| `fusion360_full_assembly` | `512` | `512` | `4` | `LearnedCostAware` | `1694` | `99.3538%` | `98.9244%` | `3.309` | `6.000` | `170.591` | `0` |
| `fusion360_full_assembly` | `512` | `512` | `4` | `LearnedPriorityOnly` | `1934` | `99.2622%` | `99.5905%` | `3.777` | `7.000` | `64.945` | `0` |
| `fusion360_full_assembly` | `512` | `512` | `4` | `IntervalOnly` | `229531` | `12.4409%` | `4.7097%` | `448.303` | `494.000` | `15113.230` | `0` |
| `fusion360_full_assembly` | `512` | `512` | `4` | `RankingOnly` | `185472` | `29.2480%` | `17.0823%` | `362.250` | `483.000` | `13150.913` | `0` |
| `fusion360_full_assembly` | `512` | `512` | `4` | `LearnedCalibrated` | `1985` | `99.2428%` | `99.6543%` | `3.877` | `7.000` | `54.824` | `0` |
| `fusion360_full_assembly` | `512` | `512` | `4` | `RandomUniform(mean over seeds)` | `52791` | `79.8620%` | `79.8392%` | `103.107` | `222.790` | `3197.545` | `0` |
| `fusion360_full_assembly` | `512` | `512` | `4` | `HeuristicCostLow` | `195752` | `25.3265%` | `73.2928%` | `382.328` | `413.000` | `4235.819` | `0` |
| `fusion360_full_assembly` | `512` | `512` | `4` | `HeuristicCostHigh` | `23885` | `90.8886%` | `37.4645%` | `46.650` | `87.900` | `9918.249` | `0` |
| `fusion360_full_assembly` | `512` | `512` | `4` | `UncertaintyOnly` | `1901` | `99.2748%` | `99.6060%` | `3.713` | `7.000` | `62.485` | `0` |
| `fusion360_full_assembly` | `512` | `512` | `4` | `OracleUpperBound` | `512` | `99.8047%` | `99.9026%` | `1.000` | `1.000` | `15.441` | `0` |
| `rtstpf_advantage_v4` | `512` | `512` | `4` | `ValidationSelectedFullSTPF` | `9885` | `96.2292%` | `96.6878%` | `19.307` | `23.000` | `282.743` | `0` |
| `rtstpf_advantage_v4` | `512` | `512` | `4` | `LearnedCostAware` | `7124` | `97.2824%` | `97.5810%` | `13.914` | `22.000` | `206.497` | `0` |
| `rtstpf_advantage_v4` | `512` | `512` | `4` | `LearnedPriorityOnly` | `9885` | `96.2292%` | `96.6878%` | `19.307` | `23.000` | `282.743` | `0` |
| `rtstpf_advantage_v4` | `512` | `512` | `4` | `IntervalOnly` | `127491` | `51.3660%` | `32.8540%` | `249.006` | `305.000` | `5731.868` | `0` |
| `rtstpf_advantage_v4` | `512` | `512` | `4` | `RankingOnly` | `93205` | `64.4451%` | `54.7757%` | `182.041` | `322.900` | `3860.540` | `0` |
| `rtstpf_advantage_v4` | `512` | `512` | `4` | `LearnedCalibrated` | `12465` | `95.2450%` | `96.9351%` | `24.346` | `24.900` | `261.629` | `0` |
| `rtstpf_advantage_v4` | `512` | `512` | `4` | `RandomUniform(mean over seeds)` | `52082` | `80.1324%` | `80.1083%` | `101.722` | `223.777` | `1698.041` | `0` |
| `rtstpf_advantage_v4` | `512` | `512` | `4` | `HeuristicCostLow` | `171559` | `34.5554%` | `58.2770%` | `335.076` | `377.000` | `3561.649` | `0` |
| `rtstpf_advantage_v4` | `512` | `512` | `4` | `HeuristicCostHigh` | `33620` | `87.1750%` | `68.6496%` | `65.664` | `122.000` | `2676.208` | `0` |
| `rtstpf_advantage_v4` | `512` | `512` | `4` | `UncertaintyOnly` | `7901` | `96.9860%` | `97.0613%` | `15.432` | `23.000` | `250.856` | `0` |
| `rtstpf_advantage_v4` | `512` | `512` | `4` | `OracleUpperBound` | `512` | `99.8047%` | `99.8226%` | `1.000` | `1.000` | `15.147` | `0` |
| `shapenet_ood_dense` | `512` | `512` | `4` | `ValidationSelectedFullSTPF` | `16796` | `93.5928%` | `93.5062%` | `32.805` | `40.000` | `388.076` | `0` |
| `shapenet_ood_dense` | `512` | `512` | `4` | `LearnedCostAware` | `16796` | `93.5928%` | `93.5062%` | `32.805` | `40.000` | `388.076` | `0` |
| `shapenet_ood_dense` | `512` | `512` | `4` | `LearnedPriorityOnly` | `10340` | `96.0556%` | `95.9756%` | `20.195` | `37.000` | `240.502` | `0` |
| `shapenet_ood_dense` | `512` | `512` | `4` | `IntervalOnly` | `167429` | `36.1309%` | `36.0942%` | `327.010` | `486.000` | `3819.085` | `0` |
| `shapenet_ood_dense` | `512` | `512` | `4` | `RankingOnly` | `167772` | `36.0001%` | `35.9687%` | `327.680` | `481.000` | `3826.588` | `0` |
| `shapenet_ood_dense` | `512` | `512` | `4` | `LearnedCalibrated` | `32067` | `87.7674%` | `87.7137%` | `62.631` | `42.000` | `734.245` | `0` |
| `shapenet_ood_dense` | `512` | `512` | `4` | `RandomUniform(mean over seeds)` | `52702` | `79.8959%` | `79.8764%` | `102.933` | `223.833` | `1202.611` | `0` |
| `shapenet_ood_dense` | `512` | `512` | `4` | `HeuristicCostLow` | `260608` | `0.5859%` | `0.8994%` | `509.000` | `509.000` | `5922.370` | `0` |
| `shapenet_ood_dense` | `512` | `512` | `4` | `HeuristicCostHigh` | `512` | `99.8047%` | `99.6908%` | `1.000` | `1.000` | `18.476` | `0` |
| `shapenet_ood_dense` | `512` | `512` | `4` | `UncertaintyOnly` | `11749` | `95.5181%` | `95.4475%` | `22.947` | `37.000` | `272.060` | `0` |
| `shapenet_ood_dense` | `512` | `512` | `4` | `OracleUpperBound` | `512` | `99.8047%` | `99.7169%` | `1.000` | `1.000` | `16.919` | `0` |

## LearnedCostAware vs RandomUniform

| Dataset | Random seeds | Learned work | Random mean work | Work speedup | Work reduction delta | Learned calls | Random mean calls | Call speedup | Win rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `common_modeling_large` | `30` | `134855.987` | `290772.160` | `2.156x` | `10.9778%` | `13070.0` | `22727.6` | `1.739x` | `1.000` |
| `fusion360_full_assembly` | `30` | `87342.832` | `1637143.014` | `18.744x` | `19.0852%` | `1694.0` | `52790.6` | `31.163x` | `1.000` |
| `rtstpf_advantage_v4` | `30` | `105726.517` | `869396.987` | `8.223x` | `17.4727%` | `7124.0` | `52081.8` | `7.311x` | `1.000` |
| `shapenet_ood_dense` | `30` | `198695.141` | `615736.849` | `3.099x` | `13.6298%` | `16796.0` | `52701.7` | `3.138x` | `1.000` |

## Best Learned Head vs RandomUniform

description: `Best learned head` isdescription, usedescriptionModelisdescriptionhashasusedescription; descriptionconnectasdescriptiondefaultMethod, descriptionafterdescriptionuse validation split fixed head-selection description.

| Dataset | Best learned head | Learned work | Random mean work | Work speedup | Work reduction delta | Learned calls | Random mean calls | Call speedup | Win rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `common_modeling_large` | `LearnedPriorityOnly` | `134354.280` | `290772.160` | `2.164x` | `11.0131%` | `13024.0` | `22727.6` | `1.745x` | `1.000` |
| `fusion360_full_assembly` | `LearnedCalibrated` | `28069.676` | `1637143.014` | `58.324x` | `19.8151%` | `1985.0` | `52790.6` | `26.595x` | `1.000` |
| `rtstpf_advantage_v4` | `LearnedCostAware` | `105726.517` | `869396.987` | `8.223x` | `17.4727%` | `7124.0` | `52081.8` | `7.311x` | `1.000` |
| `shapenet_ood_dense` | `LearnedPriorityOnly` | `123136.923` | `615736.849` | `5.000x` | `16.0992%` | `10340.0` | `52701.7` | `5.097x` | `1.000` |

## RandomCostMatched description

`RandomCostMatched` fixeddescriptionuseand `LearnedCostAware` description per-group exact budget; if budget descriptionto positive, descriptionas FN. thisdescriptionused fordescription ranking quality, is not used as final conservative CCD Method.

| Dataset | Budget source | Random seeds | Mean calls | Mean work | Mean FN | Max FN | Mean recall |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `common_modeling_large` | `LearnedCostAware first-positive rank` | `30` | `13070.0` | `167102.202` | `154.867` | `166` | `0.2863` |
| `fusion360_full_assembly` | `LearnedCostAware first-positive rank` | `30` | `1694.0` | `51713.868` | `498.933` | `506` | `0.0255` |
| `rtstpf_advantage_v4` | `LearnedCostAware first-positive rank` | `30` | `7124.0` | `118520.210` | `458.633` | `470` | `0.1042` |
| `shapenet_ood_dense` | `LearnedCostAware first-positive rank` | `30` | `16796.0` | `196032.109` | `421.400` | `437` | `0.1770` |

## descriptionConclusiondescription

- if `LearnedCostAware` indescription case ondescription random mean and win-rate connectdescription 1, descriptionwithwrite learned ranker description dense hard-negative scheduling hasdescription.
- ifadvantagedescription, descriptionas: RTSTPFExact is correctness-preserving learned scheduling/proposal layer; advantagedescriptionfrom dense group early-stop + conservative fallback, rather thandescriptioninalldistributionondescriptionbetter than random.
- `OracleUpperBound` descriptionasdescriptionondescription, descriptionasdescription baseline.
