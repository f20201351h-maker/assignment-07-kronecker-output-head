# Protocol 02 — R2 pilot at K2's 20M-token point (written before the run)

Kernel `kq5-r2-pilot` (T4×2, two independent workers). Backbone and KAS head learning-rate
multiplier come from R1 under the rules of Protocol 01: **BACKBONE = 8L×512 (8 heads), head_lr_mult = 0.1** (R1 + R1b, research-log 15:05 entry):
Protocol 01 rule 1 rejected 6L×768 because its slowest arm (KAS-G v1) ran at 14.1K tok/s < 15K.
Parameters: Dense 59.2M (head 29.3M); KAS-P 34.1M (head 4.19M); KAS-U16 35.1M; KAS-G v1 34.3M;
KAS-G v3 34.8M. Head and non-head gradients are clipped separately at 1.0.

## Runs (20,971,520 tokens each = 320 steps × 64 seq × 1024; warmup 1,048,576; cosine to 10%)
Same data order (data_seed 1234) for every run; seeds vary initialisation only.

| run | head | seed | generator |
|---|---|---|---|
| r2-dense-s1, r2-dense-s2 | Dense | 1, 2 | — |
| r2-kas0-s1 | KAS-0 | 1 | — |
| r2-kasp-s1, r2-kasp-s2 | KAS-P | 1, 2 | — |
| r2-kasu-s1, r2-kasu-s2 | KAS-U16 | 1, 2 | — |
| r2-kasg-v1-s1 | KAS-G | 1 | v1: ByteCNN emb64/ch128/2 layers/hidden256 (173,073 params), gen_lr ×1 |
| r2-kasg-v2-s1 | KAS-G | 1 | v2: v1 with gen_lr ×0.1 |
| r2-kasg-v3-s1 | KAS-G | 1 | v3: ByteCNN emb64/ch256/3 layers/hidden512 (751,889 params), gen_lr ×1 |
| r2-kasgshuf-v1-s1 | KAS-G-shuf | 1 | v1, shuffled spellings (fixed derangement, seed 4242) |

KAS-U16 stored V-sized budget: β (57,243) + U (57,243×16) = 973,131. All generators are below it
and none depends on V. After training, each worker runs `kq5.analysis` on the dev split:
minting under every rule, re-tokenised bpb for a subset of rules, logit-term shares,
correction ablations, counterfactual spelling (KAS-G).

## What this pilot decides
1. **K2 ordering.** Expected (K2 at 20M): KAS-0 ≫ KAS-P > KAS-U16 ≈ Dense. Criterion for
   "reproduced": bpb(KAS-0) − bpb(KAS-P) > 0.1, bpb(KAS-P) − bpb(KAS-U16) > 2 × seed spread,
   |bpb(KAS-U16) − bpb(Dense)| < bpb(KAS-P) − bpb(KAS-U16). If not reproduced, diagnose before R3.
2. **Seed spread.** |s1 − s2| on full-dev bpb for Dense, KAS-P, KAS-U16; the largest is the
   noise floor used in R3 planning.
3. **Generator choice.** The variant with the lowest full-dev bpb among v1–v3 is used in R3
   (and its shuffled twin). If all three are within seed spread of KAS-P, the generator search
   stops here (planning rule) and R3 keeps v1 as a declared long shot.
4. **Rule set and site count** for minting in R3 are fixed from the pilot's dev analysis
   (exploratory here; the confirmatory comparison is on test in R3).

## Compute note
The generator costs extra compute per step (R1b, 8L×512: KAS-P 21.8K tok/s, KAS-G v1 18.8K,
v3 15.5K). Any KAS-G gain must therefore be weighed against the shuffled control, which has
identical compute and parameter count.

## Predictions (made now)
* KAS-0 worst by ≥ 0.3 bpb; KAS-P recovers most of it.
* KAS-U16 beats KAS-P by 0.03–0.10 bpb; Dense within ±0.03 of KAS-U16 at 20M.
* KAS-G recovers less than half of the KAS-P→KAS-U16 gap (H1 prior ≈ 40%); v2 (slower
  generator) ≥ v1; v3 not better than v1 by more than seed spread.
* KAS-G-shuf ≈ KAS-G on in-vocabulary bpb (a hashed embedding can memorise some of what U
  stores) but clearly worse on held-out-merge regret.
* Minting: "floor" has the worst regret for every byte-derived head; "count" or "inherit"
  best among training-free rules; KAS-U16's stored correction gives nothing for minted tokens.
