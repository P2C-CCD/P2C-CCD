# Scene/Object Conservative Envelope Native TI Wall-time (scene_object_envelope_strong_native_run_id)

## Scope

- Input is real adjacent full-scene mesh frames, not synthetic dense groups and not pre-expanded query CSV replay.
- The native C++ runner partitions connected components as objects, builds swept object AABB envelopes, then builds swept VF/EE primitive AABB candidates inside those object envelopes.
- Exact backend is native Tight-Inclusion `vertexFaceCCD` / `edgeEdgeCCD`.
- `native_exact_backend_ms` times only native exact CCD calls. `envelope_ms` reports conservative envelope/candidate construction separately.
- `EnvelopeAllExact+TI` enumerates every generated envelope candidate. Scheduled rows rank candidates within each scene/object-envelope/kind group, test a bounded top-K proposal set, and conservatively fall back to full exact replay when no hit is certified.
- `FairFrontier*+TI` rows use the same fast native frontier, the same frontier-K, the same proposal top-K, and the same partial-sort/fallback replay. Only the ranking score inside the shared frontier changes.
- `OptimizedFrozenLearnedAnyHit+TI` uses a fast native geometric frontier and then applies the frozen STPF model only inside that frontier before exact replay; high object-count scenes use a fixed low-overhead gate instead of full learned scoring. Its scheduler backend time is `ordering_ms + native_exact_backend_ms`.
- `BestFixedHeuristicOracle+TI` is a retrospective non-deployable oracle that picks the better fixed heuristic per primitive family after evaluation.
- This is CCD detection wall-time over a conservative scene/object envelope; it is not full simulation/contact-solver wall-time and is not Scalable-CCD kernel time.

## Overall

- Scenes: `2`
- All-exact envelope candidates: `252295029`
- All-exact native exact calls: `252295029`
- All-exact native exact backend wall-time: `102122.040 ms`
- Conservative envelope construction wall-time: `1094097.831 ms`
- Proposal top-K before fallback: `32`
- Optimized learned frontier-K: `128`
- Optimized learned scan limit per group: `4096`
- Optimized learned random gate object count: `0`
- Frozen STPF checkpoint: `src/outputs/stpf_training/scene_object_envelope_strong_native_run_id/model_state.pt`

## Main Scheduler Table

| Method | Groups | Candidates | Exact calls | Call reduction | Native backend ms | Ordering ms | Scheduler backend ms | TP/TN/FP/FN | Coverage / fallback |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| AllExact+TI | 22 | 252295029 | 252295029 | 0.000000 | 102122.040 | 0.000 | 102122.040 | 22/0/0/0 | 1.000 / 0.000 |
| FairFrontierLearnedAnyHit+TI | 22 | 252295029 | 233221 | 0.999076 | 257.653 | 1035.315 | 1292.968 | 22/0/0/0 | 1.000 / 0.682 |
| FairFrontierRandomAnyHit+TI | 22 | 252295029 | 236895 | 0.999061 | 300.026 | 1016.551 | 1316.577 | 22/0/0/0 | 1.000 / 0.909 |
| FairFrontierProximityAnyHit+TI | 22 | 252295029 | 232242 | 0.999079 | 296.947 | 1018.898 | 1315.845 | 22/0/0/0 | 1.000 / 0.682 |
| FairFrontierMotionAnyHit+TI | 22 | 252295029 | 232478 | 0.999079 | 284.521 | 1016.571 | 1301.092 | 22/0/0/0 | 1.000 / 0.682 |
| FairFrontierBestHeuristicOracle+TI | 22 | 252295029 | 231611 | 0.999082 | 297.649 | 1023.443 | 1321.092 | 22/0/0/0 | 1.000 / 0.636 |
| OptimizedFrozenLearnedAnyHit+TI | 22 | 252295029 | 232219 | 0.999080 | 284.450 | 1013.036 | 1297.486 | 22/0/0/0 | 1.000 / 0.682 |
| FrozenLearnedAnyHit+TI | 22 | 252295029 | 242049 | 0.999041 | 307.086 | 287921.085 | 288228.171 | 22/0/0/0 | 1.000 / 1.000 |
| RandomAnyHit+TI | 22 | 252295029 | 241389 | 0.999043 | 306.211 | 1914.780 | 2220.992 | 22/0/0/0 | 1.000 / 0.955 |
| ProximityHeuristicAnyHit+TI | 22 | 252295029 | 181591 | 0.999280 | 241.641 | 19580.116 | 19821.757 | 22/0/0/0 | 1.000 / 0.545 |
| MotionHeuristicAnyHit+TI | 22 | 252295029 | 238223 | 0.999056 | 294.546 | 19700.067 | 19994.613 | 22/0/0/0 | 1.000 / 0.909 |
| BestFixedHeuristicOracle+TI | 22 | 252295029 | 181591 | 0.999280 | 241.641 | 19580.116 | 19821.757 | 22/0/0/0 | 1.000 / 0.545 |

## Fair Frontier Ranking Diagnostics

These rows isolate the learned-vs-heuristic ranking question under the same native frontier, the same top-K proposal budget, and the same partial-sort/order path. `Positive proposal hits` counts positive groups certified inside the bounded proposal stage before conservative fallback.

| Method | Positive groups | Positive proposal hits | Proposal hit rate | Positive exact calls | Positive proposal calls | Positive fallback calls | Mean exact calls to first positive | Total exact calls | Scheduler backend ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FairFrontierLearnedAnyHit+TI | 22 | 7 | 0.318182 | 233221 | 583 | 232638 | 10600.955 | 233221 | 1292.968 |
| FairFrontierRandomAnyHit+TI | 22 | 2 | 0.090909 | 236895 | 668 | 236227 | 10767.955 | 236895 | 1316.577 |
| FairFrontierProximityAnyHit+TI | 22 | 7 | 0.318182 | 232242 | 602 | 231640 | 10556.455 | 232242 | 1315.845 |
| FairFrontierMotionAnyHit+TI | 22 | 7 | 0.318182 | 232478 | 560 | 231918 | 10567.182 | 232478 | 1301.092 |
| FairFrontierBestHeuristicOracle+TI | 22 | 8 | 0.363636 | 231611 | 573 | 231038 | 10527.773 | 231611 | 1321.092 |

## Summary

| Method | Kind | Groups | Candidates | Exact calls | Positive exact calls | Positive proposal hits | Native hits | TP/TN/FP/FN | Fallback rate | Call reduction | Envelope ms | Ordering ms | Native exact backend ms | Scheduler backend ms | Total wall ms | Calls/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| EnvelopeAllExact+TI | ee | 11 | 180269041 | 180269041 | 0 | 0 | 66214 | 11/0/0/0 | 0.000000 | 0.000000 | 907062.041 | 0.000 | 74278.000 | 74278.000 | 981855.900 | 2426950.658 |
| FairFrontierLearnedAnyHit+TI | ee | 11 | 180269041 | 152459 | 152459 | 4 | 11 | 11/0/0/0 | 0.636364 | 0.999154 | 907062.041 | 676.277 | 150.940 | 827.217 | 908404.337 | 1010066.278 |
| FairFrontierMotionAnyHit+TI | ee | 11 | 180269041 | 152191 | 152191 | 3 | 11 | 11/0/0/0 | 0.727273 | 0.999156 | 907062.041 | 663.821 | 177.200 | 841.021 | 908418.128 | 858866.173 |
| FairFrontierProximityAnyHit+TI | ee | 11 | 180269041 | 151324 | 151324 | 4 | 11 | 11/0/0/0 | 0.636364 | 0.999161 | 907062.041 | 670.693 | 190.328 | 861.022 | 908438.340 | 795069.146 |
| FairFrontierRandomAnyHit+TI | ee | 11 | 180269041 | 153438 | 153438 | 2 | 11 | 11/0/0/0 | 0.818182 | 0.999149 | 907062.041 | 668.188 | 188.944 | 857.132 | 908434.560 | 812080.597 |
| FrozenLearnedAnyHit+TI | ee | 11 | 180269041 | 158435 | 158435 | 0 | 11 | 11/0/0/0 | 1.000000 | 0.999121 | 907062.041 | 209381.500 | 198.048 | 209579.548 | 1117159.890 | 799982.832 |
| MotionHeuristicAnyHit+TI | ee | 11 | 180269041 | 156512 | 156512 | 1 | 11 | 11/0/0/0 | 0.909091 | 0.999132 | 907062.041 | 14122.187 | 188.974 | 14311.161 | 921888.250 | 828220.193 |
| OptimizedFrozenLearnedAnyHit+TI | ee | 11 | 180269041 | 151305 | 151305 | 4 | 11 | 11/0/0/0 | 0.636364 | 0.999161 | 907062.041 | 659.434 | 178.825 | 838.259 | 908415.745 | 846107.475 |
| ProximityHeuristicAnyHit+TI | ee | 11 | 180269041 | 139662 | 139662 | 3 | 11 | 11/0/0/0 | 0.727273 | 0.999225 | 907062.041 | 13755.445 | 189.735 | 13945.180 | 921522.710 | 736088.646 |
| RandomAnyHit+TI | ee | 11 | 180269041 | 158439 | 158439 | 0 | 11 | 11/0/0/0 | 1.000000 | 0.999121 | 907062.041 | 1261.609 | 200.070 | 1461.679 | 909039.490 | 791916.641 |
| EnvelopeAllExact+TI | vf | 11 | 72025988 | 72025988 | 0 | 0 | 16963 | 11/0/0/0 | 0.000000 | 0.000000 | 187035.790 | 0.000 | 27844.040 | 27844.040 | 215394.460 | 2586764.995 |
| FairFrontierLearnedAnyHit+TI | vf | 11 | 72025988 | 80762 | 80762 | 3 | 11 | 11/0/0/0 | 0.727273 | 0.998879 | 187035.790 | 359.038 | 106.714 | 465.752 | 188016.247 | 756809.335 |
| FairFrontierMotionAnyHit+TI | vf | 11 | 72025988 | 80287 | 80287 | 4 | 11 | 11/0/0/0 | 0.636364 | 0.998885 | 187035.790 | 352.750 | 107.321 | 460.071 | 188011.006 | 748104.278 |
| FairFrontierProximityAnyHit+TI | vf | 11 | 72025988 | 80918 | 80918 | 3 | 11 | 11/0/0/0 | 0.727273 | 0.998877 | 187035.790 | 348.204 | 106.619 | 454.823 | 188005.796 | 758946.116 |
| FairFrontierRandomAnyHit+TI | vf | 11 | 72025988 | 83457 | 83457 | 0 | 11 | 11/0/0/0 | 1.000000 | 0.998841 | 187035.790 | 348.363 | 111.082 | 459.445 | 188009.626 | 751309.843 |
| FrozenLearnedAnyHit+TI | vf | 11 | 72025988 | 83614 | 83614 | 0 | 11 | 11/0/0/0 | 1.000000 | 0.998839 | 187035.790 | 78539.585 | 109.038 | 78648.623 | 266199.209 | 766832.878 |
| MotionHeuristicAnyHit+TI | vf | 11 | 72025988 | 81711 | 81711 | 1 | 11 | 11/0/0/0 | 0.909091 | 0.998866 | 187035.790 | 5577.880 | 105.572 | 5683.452 | 193234.528 | 773981.433 |
| OptimizedFrozenLearnedAnyHit+TI | vf | 11 | 72025988 | 80914 | 80914 | 3 | 11 | 11/0/0/0 | 0.727273 | 0.998877 | 187035.790 | 353.602 | 105.625 | 459.226 | 188009.993 | 766051.155 |
| ProximityHeuristicAnyHit+TI | vf | 11 | 72025988 | 41929 | 41929 | 7 | 11 | 11/0/0/0 | 0.363636 | 0.999418 | 187035.790 | 5824.671 | 51.905 | 5876.576 | 193427.281 | 807796.491 |
| RandomAnyHit+TI | vf | 11 | 72025988 | 82950 | 82950 | 1 | 11 | 11/0/0/0 | 0.909091 | 0.998848 | 187035.790 | 653.172 | 106.141 | 759.312 | 188309.504 | 781507.617 |

## Per-scene Rows

| Scene | Kind | Method | Objects | Object envelopes | Groups | Candidates | Exact calls | TP/TN/FP/FN | Fallback rate | Capped | Envelope ms | Ordering ms | Native exact backend ms | Total wall ms |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- | ---: | ---: | ---: | ---: |
| puffer-ball | vf | EnvelopeAllExact+TI | 1 | 1 | 1 | 71577718 | 71577718 | 1/0/0/0 | 0.000000 | False | 186941.000 | 0.000 | 25640.100 | 213064.000 |
| puffer-ball | vf | FairFrontierLearnedAnyHit+TI | 1 | 1 | 1 | 71577718 | 9887 | 1/0/0/0 | 1.000000 | False | 186941.000 | 354.031 | 2.989 | 187781.000 |
| puffer-ball | vf | FairFrontierRandomAnyHit+TI | 1 | 1 | 1 | 71577718 | 9887 | 1/0/0/0 | 1.000000 | False | 186941.000 | 344.515 | 3.820 | 187772.000 |
| puffer-ball | vf | FairFrontierProximityAnyHit+TI | 1 | 1 | 1 | 71577718 | 9887 | 1/0/0/0 | 1.000000 | False | 186941.000 | 344.529 | 3.014 | 187772.000 |
| puffer-ball | vf | FairFrontierMotionAnyHit+TI | 1 | 1 | 1 | 71577718 | 9887 | 1/0/0/0 | 1.000000 | False | 186941.000 | 348.631 | 2.949 | 187776.000 |
| puffer-ball | vf | OptimizedFrozenLearnedAnyHit+TI | 1 | 1 | 1 | 71577718 | 9887 | 1/0/0/0 | 1.000000 | False | 186941.000 | 348.781 | 2.968 | 187776.000 |
| puffer-ball | vf | FrozenLearnedAnyHit+TI | 1 | 1 | 1 | 71577718 | 9919 | 1/0/0/0 | 1.000000 | False | 186941.000 | 78071.900 | 3.030 | 265499.000 |
| puffer-ball | vf | RandomAnyHit+TI | 1 | 1 | 1 | 71577718 | 9919 | 1/0/0/0 | 1.000000 | False | 186941.000 | 649.268 | 3.056 | 188076.000 |
| puffer-ball | vf | ProximityHeuristicAnyHit+TI | 1 | 1 | 1 | 71577718 | 9919 | 1/0/0/0 | 1.000000 | False | 186941.000 | 5791.850 | 2.961 | 193219.000 |
| puffer-ball | vf | MotionHeuristicAnyHit+TI | 1 | 1 | 1 | 71577718 | 9919 | 1/0/0/0 | 1.000000 | False | 186941.000 | 5543.380 | 3.059 | 192971.000 |
| puffer-ball | ee | EnvelopeAllExact+TI | 1 | 1 | 1 | 178227707 | 178227707 | 1/0/0/0 | 0.000000 | False | 906274.000 | 0.000 | 54722.900 | 961481.000 |
| puffer-ball | ee | FairFrontierLearnedAnyHit+TI | 1 | 1 | 1 | 178227707 | 40754 | 1/0/0/0 | 1.000000 | False | 906274.000 | 665.183 | 11.463 | 907434.000 |
| puffer-ball | ee | FairFrontierRandomAnyHit+TI | 1 | 1 | 1 | 178227707 | 40754 | 1/0/0/0 | 1.000000 | False | 906274.000 | 658.481 | 11.860 | 907428.000 |
| puffer-ball | ee | FairFrontierProximityAnyHit+TI | 1 | 1 | 1 | 178227707 | 40754 | 1/0/0/0 | 1.000000 | False | 906274.000 | 660.920 | 11.527 | 907430.000 |
| puffer-ball | ee | FairFrontierMotionAnyHit+TI | 1 | 1 | 1 | 178227707 | 40754 | 1/0/0/0 | 1.000000 | False | 906274.000 | 654.073 | 11.586 | 907423.000 |
| puffer-ball | ee | OptimizedFrozenLearnedAnyHit+TI | 1 | 1 | 1 | 178227707 | 40754 | 1/0/0/0 | 1.000000 | False | 906274.000 | 648.781 | 11.499 | 907418.000 |
| puffer-ball | ee | FrozenLearnedAnyHit+TI | 1 | 1 | 1 | 178227707 | 40786 | 1/0/0/0 | 1.000000 | False | 906274.000 | 207228.000 | 11.429 | 1114000.000 |
| puffer-ball | ee | RandomAnyHit+TI | 1 | 1 | 1 | 178227707 | 40786 | 1/0/0/0 | 1.000000 | False | 906274.000 | 1246.470 | 11.487 | 908016.000 |
| puffer-ball | ee | ProximityHeuristicAnyHit+TI | 1 | 1 | 1 | 178227707 | 40786 | 1/0/0/0 | 1.000000 | False | 906274.000 | 13609.600 | 11.638 | 920379.000 |
| puffer-ball | ee | MotionHeuristicAnyHit+TI | 1 | 1 | 1 | 178227707 | 40786 | 1/0/0/0 | 1.000000 | False | 906274.000 | 13973.800 | 11.881 | 920743.000 |
| rod-twist | vf | EnvelopeAllExact+TI | 4 | 10 | 10 | 448270 | 448270 | 10/0/0/0 | 0.000000 | False | 94.790 | 0.000 | 2203.940 | 2330.460 |
| rod-twist | vf | FairFrontierLearnedAnyHit+TI | 4 | 10 | 10 | 448270 | 70875 | 10/0/0/0 | 0.700000 | False | 94.790 | 5.007 | 103.725 | 235.247 |
| rod-twist | vf | FairFrontierRandomAnyHit+TI | 4 | 10 | 10 | 448270 | 73570 | 10/0/0/0 | 1.000000 | False | 94.790 | 3.848 | 107.262 | 237.626 |
| rod-twist | vf | FairFrontierProximityAnyHit+TI | 4 | 10 | 10 | 448270 | 71031 | 10/0/0/0 | 0.700000 | False | 94.790 | 3.675 | 103.605 | 233.796 |
| rod-twist | vf | FairFrontierMotionAnyHit+TI | 4 | 10 | 10 | 448270 | 70400 | 10/0/0/0 | 0.600000 | False | 94.790 | 4.119 | 104.372 | 235.006 |
| rod-twist | vf | OptimizedFrozenLearnedAnyHit+TI | 4 | 10 | 10 | 448270 | 71027 | 10/0/0/0 | 0.700000 | False | 94.790 | 4.821 | 102.657 | 233.993 |
| rod-twist | vf | FrozenLearnedAnyHit+TI | 4 | 10 | 10 | 448270 | 73695 | 10/0/0/0 | 1.000000 | False | 94.790 | 467.685 | 106.008 | 700.209 |
| rod-twist | vf | RandomAnyHit+TI | 4 | 10 | 10 | 448270 | 73031 | 10/0/0/0 | 0.900000 | False | 94.790 | 3.904 | 103.085 | 233.504 |
| rod-twist | vf | ProximityHeuristicAnyHit+TI | 4 | 10 | 10 | 448270 | 32010 | 10/0/0/0 | 0.300000 | False | 94.790 | 32.821 | 48.945 | 208.281 |
| rod-twist | vf | MotionHeuristicAnyHit+TI | 4 | 10 | 10 | 448270 | 71792 | 10/0/0/0 | 0.900000 | False | 94.790 | 34.500 | 102.513 | 263.528 |
| rod-twist | ee | EnvelopeAllExact+TI | 4 | 10 | 10 | 2041334 | 2041334 | 10/0/0/0 | 0.000000 | False | 788.041 | 0.000 | 19555.100 | 20374.900 |
| rod-twist | ee | FairFrontierLearnedAnyHit+TI | 4 | 10 | 10 | 2041334 | 111705 | 10/0/0/0 | 0.600000 | False | 788.041 | 11.094 | 139.477 | 970.337 |
| rod-twist | ee | FairFrontierRandomAnyHit+TI | 4 | 10 | 10 | 2041334 | 112684 | 10/0/0/0 | 0.800000 | False | 788.041 | 9.707 | 177.084 | 1006.560 |
| rod-twist | ee | FairFrontierProximityAnyHit+TI | 4 | 10 | 10 | 2041334 | 110570 | 10/0/0/0 | 0.600000 | False | 788.041 | 9.774 | 178.801 | 1008.340 |
| rod-twist | ee | FairFrontierMotionAnyHit+TI | 4 | 10 | 10 | 2041334 | 111437 | 10/0/0/0 | 0.700000 | False | 788.041 | 9.748 | 165.614 | 995.128 |
| rod-twist | ee | OptimizedFrozenLearnedAnyHit+TI | 4 | 10 | 10 | 2041334 | 110551 | 10/0/0/0 | 0.600000 | False | 788.041 | 10.653 | 167.326 | 997.745 |
| rod-twist | ee | FrozenLearnedAnyHit+TI | 4 | 10 | 10 | 2041334 | 117649 | 10/0/0/0 | 1.000000 | False | 788.041 | 2153.500 | 186.619 | 3159.890 |
| rod-twist | ee | RandomAnyHit+TI | 4 | 10 | 10 | 2041334 | 117653 | 10/0/0/0 | 1.000000 | False | 788.041 | 15.139 | 188.583 | 1023.490 |
| rod-twist | ee | ProximityHeuristicAnyHit+TI | 4 | 10 | 10 | 2041334 | 98876 | 10/0/0/0 | 0.700000 | False | 788.041 | 145.845 | 178.097 | 1143.710 |
| rod-twist | ee | MotionHeuristicAnyHit+TI | 4 | 10 | 10 | 2041334 | 115726 | 10/0/0/0 | 0.900000 | False | 788.041 | 148.387 | 177.093 | 1145.250 |

## Artifacts

- Native runner: `src/build_tools/scene_object_envelope_native_ti_walltime.exe`
- C++ source: `src/tools/scene_object_envelope_native_ti_walltime.cpp`
- Raw JSONL dir: `src/benchmark/scene_object_envelope_strong_native_run_id/raw_jsonl`
- Summary CSV: `src/benchmark/scene_object_envelope_strong_native_run_id/scene_object_envelope_strong_native_run_id_summary.csv`
- Main scheduler CSV: `src/benchmark/scene_object_envelope_strong_native_run_id/scene_object_envelope_strong_native_run_id_main_scheduler_table.csv`
- Fair frontier ranking CSV: `src/benchmark/scene_object_envelope_strong_native_run_id/scene_object_envelope_strong_native_run_id_fair_frontier_ranking_table.csv`
- Row CSV: `src/benchmark/scene_object_envelope_strong_native_run_id/scene_object_envelope_strong_native_run_id_rows.csv`

## Reproduction

```powershell
conda activate cudadev
python src/tools/run_scene_object_envelope_native_ti_walltime.py --run-name scene_object_envelope_strong_native_run_id --proposal-top-k 32 --optimized-frontier-k 128 --optimized-scan-limit-per-group 4096 --optimized-random-gate-object-count 0
```
