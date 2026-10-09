# Generated summary tables

Written by `scripts/build_summary.py`; do not edit by hand.

## arms

| Arm | Added in | What differs | Seeds | Test BPB | Δ vs Dense-prior | Head params | V-dependent head params |
|---|---|---|---:|---:|---:|---:|---:|
| **Dense-prior** | P08 | Dense + trainable per-token bias initialised to log-unigram counts | 4 | 1.3987 | — | 29,365,659 | 29,365,659 |
| **KAS-U64** | P06a, P07 | KAS + learned prior + stored rank-64 correction | 4 | 1.4155 | +0.0167 | 7,947,900 | 3,720,795 |
| **Dense** | P03, P07 | untied d x V head on the same Kronecker input; no output bias | 4 | 1.4405 | +0.0418 | 29,308,416 | 29,308,416 |
| KAS-U16 | P03 | KAS + learned prior + stored rank-16 correction (K2 re-implemented) | 2 | 1.4613 | +0.0625 | 5,175,660 | 973,131 |
| KAS-U16-dropU25 | P06a | KAS-U16; each step 25% of tokens lose u_v (vocabulary dropout) | 2 | 1.4844 | +0.0856 | 5,175,660 | 973,131 |
| KAS-G | P03 | KAS + prior + correction generated from bytes by a shared ByteCNN | 2 | 1.5138 | +0.1151 | 4,954,418 | 0 |
| KAS-G-shuf | P03 | KAS-G, generator reads a fixed other token's bytes | 2 | 1.5279 | +0.1291 | 4,954,418 | 0 |
| KAS-G-occ | P06 | KAS-G with the reference report's occupancy-MLP generator | 2 | 1.5522 | +0.1535 | 4,957,866 | 0 |
| KAS-U16 (c_zero) | P03 (ablation) | KAS-U16 with the broken C = 0 initialisation | 2 | 1.5830 | +0.1843 | 5,175,660 | 973,131 |
| **KAS-P** | P03 | KAS + fixed log-unigram prior; no per-token trainable term | 2 | 1.6308 | +0.2321 | 4,194,337 | 0 |
| KAS-U16-drop25 | P06 | KAS-U16; each step 25% of tokens lose beta_v and u_v | 2 | 1.6347 | +0.2360 | 5,175,660 | 973,131 |
| KAS-0 | P03 | Kronecker additive head (KAS) alone | 2 | 1.9398 | +0.5410 | 4,194,337 | 0 |

## verdicts

| ID | Protocol | Question | Key number (test, 95% interval) | Pre-registered verdict |
|---|---|---|---|---|
| H1 | P03 | Does a correction generated from bytes recover ≥ 0.5 of a stored rank-16 correction's gain? | ρ = 0.690 [+0.661, +0.718] | SUPPORTED |
| H2 | P03 | Do generated parameters mint held-out merges with lower regret than stored ones? | Δregret −0.377 [−0.543, −0.200] bits/site | SUPPORTED (relative cost only) |
| H3 | P03 | Does the byte head's deficit change sign across frequency buckets? | Spearman ρ = 0.143 | INCONCLUSIVE |
| H4 | P03 | Does the minting advantage need real spelling (selected rule)? | −0.099 [−0.246, +0.052] bits/site | REJECTED |
| K2 | P03 | Does Dense's lead over KAS-U16 grow from 12M to 100M tokens (dev)? | +0.0029 → +0.0211 BPB | REPRODUCED |
| Q1/H5 | P06 | Vocabulary dropout of prior and correction: quality cost; minting | +0.1734 [+0.1680, +0.1789] BPB | costly; H5 SUPPORTED |
| Q1/H5′ | P06a | Vocabulary dropout of the correction only (post-hoc reading) | +0.0231 [+0.0207, +0.0254] BPB | H5′ SUPPORTED |
| H7 | P06 | Does the reference report's negative result for byte-computed coordinates replicate? | KAS-P − KAS-G-occ +0.0786 [+0.0773, +0.0797] BPB (report: −0.24) | DOES NOT REPLICATE |
| H8 | P06/06a | Does a rank-64 stored correction train (and how does it compare with bias-free Dense)? | KAS-U64 − Dense −0.0281 [−0.0337, −0.0229] BPB (2 seeds) | TRAINS |
| P07 | P07 | Does KAS-U64 < bias-free Dense hold on fresh seeds 3 and 4? | −0.0219 BPB; Welch [−0.0883, +0.0444] | WEAKENED |
| P08 | P08 | Does KAS-U64 still beat Dense once Dense gets the same log-unigram bias? | KAS-U64 − Dense-prior +0.0167 [+0.0114, +0.0224] BPB | band D: prior-matched Dense beats KAS-U64 |

## protocol08

| Seed | Dense | Dense-prior | KAS-U64 | Dense − Dense-prior | KAS-U64 − Dense-prior |
|---|---:|---:|---:|---:|---:|
| 1 | 1.4452 | 1.3986 | 1.4168 | +0.0466 | +0.0182 |
| 2 | 1.4403 | 1.3990 | 1.4124 | +0.0413 | +0.0134 |
| 3 | 1.4396 | 1.3969 | 1.4101 | +0.0427 | +0.0131 |
| 4 | 1.4369 | 1.4004 | 1.4226 | +0.0364 | +0.0221 |
| **mean** | **1.4405** | **1.3987** | **1.4155** | **+0.0418** | **+0.0167** |

Dense − Dense-prior: paired t (df 3) [+0.0351, +0.0484]. KAS-U64 − Dense-prior: two-level bootstrap [+0.0114, +0.0224], Welch (df ≈ 3.4) [+0.0083, +0.0252]. KAS-U64 is lower in 0 of 16 (seed, seed) pairs. Share of the earlier KAS-U64-vs-Dense gap removed by giving Dense the bias: 1.67 [1.37, 2.18]. Source: `results/phase4/prior_verdict.json`.

## projection

| Teacher / split | Non-EOS targets | Fitted teacher KL (fixed log-unigram bias) | Mimic target-NLL gap | Realised KAS-P gap | KAS-P minus mimic |
|---|---:|---:|---:|---:|---:|
| seed 1 / dev | 4,091 | 0.742673 | 0.732321 | 0.830475 | 0.098108 |
| seed 2 / dev | 4,091 | 0.743360 | 0.732049 | 0.837669 | 0.105650 |
| seed 1 / test | 4,092 | 0.741189 | 0.785448 | 0.795471 | 0.010001 |
| seed 2 / test | 4,092 | 0.741865 | 0.780921 | 0.792246 | 0.011305 |

## families

| Fixed feature family | Extra features | Bias in the fit | Achieved teacher KL, nats (4 grids) | Mimic target-NLL gap, nats (4 grids) |
|---|---:|---|---:|---:|
| KAS (exact codec family) | 0 | fixed log-unigram (= KAS-P's prior) | 0.741–0.743 | 0.732–0.785 |
| KAS (exact codec family) | 0 | teacher's learned bias, fixed | 0.740–0.743 | 0.731–0.784 |
| KAS (exact codec family) | 0 | free per-token bias, refit per 512-context chunk (relaxation) | 0.614–0.631 | 0.554–0.609 |
| KAS + end-anchored occupancy cells | 1,773 | teacher's learned bias, fixed | 0.539–0.541 | 0.531–0.544 |
| KAS + private cell for 4,000 most frequent tokens | 4,000 | teacher's learned bias, fixed | 0.281–0.287 | 0.280–0.294 |
| KAS + 8,192 hashed byte-trigram cells | 8,192 | teacher's learned bias, fixed | 0.201–0.207 | 0.160–0.176 |
| KAS + 8,192 signed hashed identity cells (8 per token) | 8,192 | teacher's learned bias, fixed | 0.138–0.142 | 0.113–0.150 |

## params

| Output head (d = 512, V = 57,243) | Trainable parameters | FP32 Adam moments, MiB | FP32 weights + grads + moments, MiB |
|---|---:|---:|---:|
| KAS-P | 4,194,337 | 32.000 | 64.001 |
| KAS + 8,192 hashed trigrams (hypothetical, not trained) | 8,388,641 | 64.000 | 128.001 |
| KAS-U16 | 5,175,660 | 39.487 | 78.974 |
| KAS-U64 | 7,947,900 | 60.638 | 121.275 |
| Dense-prior | 29,365,659 | 224.042 | 448.084 |

## convergence

| Fit | Iterations | Mean achieved KL, nats |
|---|---|---|
| KAS, fixed log-unigram bias; first 512 dev contexts (original fitter) | LBFGS 200 / 400 / 800 | 0.808016 / 0.805567 / 0.804929 |
| KAS, fixed log-unigram bias; 4 preselected dev contexts (independent float64 fitter) | saved 200 (float32) → independent 800 (float64) | 0.721957 → 0.717770 (−0.004187; per context 0.0026–0.0073) |
| KAS + 8,192 trigrams, teacher bias; 4 preselected dev contexts (independent float64 fitter) | saved 200 (float32) → independent 800 (float64) | 0.194032 → 0.161651 (−0.032382; per context 0.0180–0.0427) |

## static

| Target distribution | Contexts | Achieved KL of the KAS family, nats | Bias |
|---|---:|---:|---|
| training unigram (context-free) | 1 | 1.3482 (LBFGS); 1.3475 (Adam) | none |
| bigram conditionals of the 128 most frequent previous tokens (45.86% of training positions) | 128 | 0.5525 | fixed log-unigram |
| same | 128 | 0.4807 | free per-token, joint over contexts |

## rowsets

| Fixed rows (d = 512, free context vector) | Achieved teacher KL, nats (1,024 dev contexts) |
|---|---:|
| Dense-prior's own rows and bias (sanity: exact optimum is 0) | −0.0000955 |
| bias-free Dense rows, teacher's bias supplied | 0.029067 |
| trained KAS-U64 rows, own bias | 0.119284 |
| trained KAS-U16 rows, own bias | 0.249541 |
| tied input projection P·κ_v (a KAS subfamily) | 0.779072 |
