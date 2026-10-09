### Bits per byte (test, end of training; seed mean, per-seed values, spread)

| Arm | bpb | seeds | spread | head params | V-dependent |
|---|---:|---|---:|---:|---:|
| Dense | 1.4427 | 1.4452, 1.4403 | 0.0048 | 29,308,416 | 29,308,416 |
| KAS-0 | 1.9398 | 1.9396, 1.9399 | 0.0003 | 4,194,337 | 0 |
| KAS-P | 1.6308 | 1.6307, 1.6308 | 0.0001 | 4,194,337 | 0 |
| KAS-U16 | 1.4613 | 1.4608, 1.4618 | 0.0010 | 5,175,660 | 973,131 |
| KAS-G | 1.5138 | 1.5181, 1.5095 | 0.0086 | 4,954,418 | 0 |
| KAS-G-shuf | 1.5279 | 1.5293, 1.5265 | 0.0028 | 4,954,418 | 0 |
| KAS-U16 (c_zero) | 1.5830 | 1.5828, 1.5832 | 0.0004 | 5,175,660 | 973,131 |

Largest seed spread: 0.0086 bpb.

### Paired differences (bpb, 95% two-level bootstrap)

| Difference | estimate [95% CI] |
|---|---|
| KAS-0 - KAS-P | +0.3090 [+0.3071, +0.3108] |
| KAS-P - KAS-U16 | +0.1695 [+0.1678, +0.1711] |
| KAS-U16 - Dense | +0.0185 [+0.0138, +0.0231] |
| KAS-P - Dense | +0.1880 [+0.1827, +0.1933] |
| KAS-G - KAS-U16 | +0.0525 [+0.0475, +0.0575] |
| KAS-P - KAS-G | +0.1170 [+0.1119, +0.1220] |
| KAS-G-shuf - KAS-G | +0.0141 [+0.0083, +0.0200] |
| KAS-P - KAS-G-shuf | +0.1029 [+0.1007, +0.1051] |

### H1

* Denominator bpb(KAS-P) − bpb(KAS-U16): +0.1695 [+0.1678, +0.1711]
* ρ = [bpb(KAS-P) − bpb(KAS-G)] / [bpb(KAS-P) − bpb(KAS-U16)]: +0.690 [+0.661, +0.718]
* Verdict: **SUPPORTED** (band: success)

### H2

* Rules selected on dev: KAS-U16 `inherit`, KAS-G `inherit`
* Δregret (KAS-G − KAS-U16), bits/site: -0.377 [-0.543, -0.200] over 2,110 sites
* Δleak (KAS-G − KAS-U16), bpb: -0.002549 [-0.002688, -0.002418]
* KAS-U16 'inherit' beats every KAS-G rule: False
* Verdict: **SUPPORTED**

### H4

* Δspell (KAS-G − KAS-G-shuf), bits/site: -0.099 [-0.246, +0.052]
* Δshuf (KAS-G-shuf − KAS-U16), bits/site: -0.277 [-0.424, -0.133]
* Verdict: **REJECTED**

### H3 — NLL(Dense) − NLL(KAS), nats per target (primary: KAS-U16)

| training count | targets | KAS-P | KAS-U16 | KAS-G |
|---|---:|---|---|---|
| [0,1) | 1 | -3.890 [-4.819, -2.961] | -1.106 [-2.190, -0.022] | -1.599 [-3.520, +0.321] |
| [1,10) | 17 | +0.240 [-0.908, +1.251] | -0.598 [-1.399, +0.140] | +0.397 [-0.560, +1.348] |
| [10,100) | 1,191 | +1.429 [+1.053, +1.803] | +0.471 [+0.148, +0.822] | +1.307 [+0.924, +1.690] |
| [100,1000) | 66,564 | -1.026 [-1.091, -0.962] | -0.108 [-0.159, -0.059] | -0.832 [-0.888, -0.774] |
| [1000,10000) | 213,347 | -1.150 [-1.187, -1.114] | -0.099 [-0.125, -0.072] | -0.569 [-0.609, -0.526] |
| [10000,100000) | 300,202 | -0.889 [-0.916, -0.860] | -0.099 [-0.121, -0.077] | -0.244 [-0.275, -0.213] |
| [100000,1000000) | 200,537 | -0.465 [-0.484, -0.446] | -0.075 [-0.090, -0.059] | -0.108 [-0.133, -0.083] |
| [1000000,inf) | 348,442 | -0.057 [-0.068, -0.046] | +0.012 [+0.004, +0.019] | +0.024 [+0.008, +0.041] |

Spearman ρ (bucket order vs d, KAS-U16): +0.143. Verdict: **INCONCLUSIVE**

### Minting every held-out merge (test; seed mean)

| Arm | rule | regret bits/site | beats prefix | median rank | leak bpb |
|---|---|---:|---:|---:|---:|
| Dense | mean | +7.697 | 0.001 | 26,668 | 0.00014 |
| Dense | inherit | +0.766 | 0.000 | 1,306 | 0.00057 |
| KAS-0 | bytes | -1.704 | 0.238 | 2,872 | 0.00328 |
| KAS-P | floor | +9.524 | 0.000 | 35,990 | 0.00000 |
| KAS-P | zero | +0.252 | 0.142 | 3,474 | 0.00073 |
| KAS-P | count | +0.258 | 0.144 | 3,394 | 0.00060 |
| KAS-P | inherit | +0.258 | 0.144 | 3,394 | 0.00060 |
| KAS-U16 | floor | +10.895 | 0.001 | 30,332 | 0.00000 |
| KAS-U16 | zero | +1.660 | 0.115 | 2,140 | 0.00268 |
| KAS-U16 | count | +1.663 | 0.121 | 2,110 | 0.00220 |
| KAS-U16 | inherit | +1.047 | 0.153 | 1,374 | 0.00329 |
| KAS-U16 | inherit_prior | +1.674 | 0.121 | 2,100 | 0.00218 |
| KAS-U16 | count_cal | +1.707 | 0.120 | 2,156 | 0.00214 |
| KAS-G | floor | +18.437 | 0.003 | 54,586 | 0.00005 |
| KAS-G | zero | +9.201 | 0.047 | 31,206 | 0.00234 |
| KAS-G | count | +9.201 | 0.044 | 30,693 | 0.00202 |
| KAS-G | inherit | +0.670 | 0.104 | 2,296 | 0.00074 |
| KAS-G | gen_floor | +10.105 | 0.000 | 32,002 | 0.00000 |
| KAS-G | gen_zero | +0.833 | 0.134 | 2,572 | 0.00067 |
| KAS-G | gen_count | +0.839 | 0.140 | 2,476 | 0.00055 |
| KAS-G | gen_inherit | +0.839 | 0.140 | 2,476 | 0.00055 |
| KAS-G | count_cal | +11.298 | 0.022 | 38,476 | 0.00092 |
| KAS-G-shuf | floor | +18.090 | 0.005 | 53,908 | 0.00022 |
| KAS-G-shuf | zero | +8.909 | 0.083 | 31,499 | 0.00653 |
| KAS-G-shuf | count | +8.899 | 0.080 | 31,630 | 0.00522 |
| KAS-G-shuf | inherit | +0.770 | 0.115 | 2,744 | 0.00059 |
| KAS-G-shuf | gen_floor | +10.426 | 0.000 | 35,565 | 0.00000 |
| KAS-G-shuf | gen_zero | +1.154 | 0.114 | 3,268 | 0.00075 |
| KAS-G-shuf | gen_count | +1.159 | 0.117 | 3,283 | 0.00062 |
| KAS-G-shuf | gen_inherit | +1.159 | 0.117 | 3,283 | 0.00062 |
| KAS-G-shuf | count_cal | +10.550 | 0.060 | 36,282 | 0.00299 |
| KAS-U16 (c_zero) | floor | +9.870 | 0.000 | 35,834 | 0.00000 |
| KAS-U16 (c_zero) | zero | +0.598 | 0.135 | 3,327 | 0.00068 |
| KAS-U16 (c_zero) | count | +0.603 | 0.140 | 3,277 | 0.00057 |
| KAS-U16 (c_zero) | inherit | +0.446 | 0.143 | 2,966 | 0.00065 |
| KAS-U16 (c_zero) | inherit_prior | +0.617 | 0.139 | 3,360 | 0.00056 |
| KAS-U16 (c_zero) | count_cal | +0.699 | 0.134 | 3,441 | 0.00053 |

### Net effect: bpb on test text re-tokenised with the 1,000 held-out merges

| Arm | rule | standard bpb (same text) | Δ bpb ×10⁻³ | tokens saved |
|---|---|---:|---:|---:|
| Dense | mean | 1.4428 | +3.501 | 0.18% |
| Dense | inherit | 1.4428 | +1.325 | 0.18% |
| KAS-0 | bytes | 1.9398 | +2.424 | 0.18% |
| KAS-P | count | 1.6308 | +0.317 | 0.18% |
| KAS-P | inherit | 1.6308 | +0.317 | 0.18% |
| KAS-U16 | count | 1.4613 | +2.626 | 0.18% |
| KAS-U16 | inherit | 1.4613 | +3.464 | 0.18% |
| KAS-U16 | inherit_prior | 1.4613 | +2.607 | 0.18% |
| KAS-G | count | 1.5139 | +5.083 | 0.18% |
| KAS-G | inherit | 1.5139 | +0.564 | 0.18% |
| KAS-G | gen_count | 1.5139 | +0.400 | 0.18% |
| KAS-G | gen_inherit | 1.5139 | +0.400 | 0.18% |
| KAS-G-shuf | count | 1.5279 | +8.284 | 0.18% |
| KAS-G-shuf | inherit | 1.5279 | +0.507 | 0.18% |
| KAS-G-shuf | gen_count | 1.5279 | +0.679 | 0.18% |
| KAS-G-shuf | gen_inherit | 1.5279 | +0.679 | 0.18% |
| KAS-U16 (c_zero) | count | 1.5830 | +0.453 | 0.18% |
| KAS-U16 (c_zero) | inherit | 1.5830 | +0.489 | 0.18% |
| KAS-U16 (c_zero) | inherit_prior | 1.5830 | +0.455 | 0.18% |

### K2 pattern (dev): bpb(KAS-U16) − bpb(Dense) first +0.0029 [+0.0002, +0.0057], last +0.0211 [+0.0168, +0.0255]; reproduced: True
