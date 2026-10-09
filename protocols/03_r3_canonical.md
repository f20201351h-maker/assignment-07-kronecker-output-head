# Protocol 03 — R3 canonical runs and pre-registered verdict rules (written before the run)

Locked before any R3 run (commit recorded in research-log). The only item not yet known is the
generator variant, which is fixed mechanically by Protocol 02 rule 3 (lowest full-dev bpb among
v1–v3 in the pilot) and used only by kernel G.

## Setup (identical for every arm)
* Backbone 8L × 512, 8 heads, T = 1024, learned positions, pre-LN, GELU, no biases; Kronecker
  input (reference codec, pos_dim 32, D = 8192) with W_in ~ N(0, 1/D).
* 1,526 steps × 64 sequences × 1024 tokens = **100,007,936 training tokens**, one epoch over a
  fixed window order (data_seed 1234, sha256 logged per run). AdamW (0.9, 0.95), lr 1e-3,
  2,097,152-token warmup, cosine to 10%, weight decay 0.1 on ≥2-D tensors, fp16 autocast +
  GradScaler, loss in fp32, SDPA efficient-attention backend, head and non-head gradients
  clipped separately at 1.0. KAS head-parameter lr ×0.1; Dense head ×1.
* Generator for KAS-G and KAS-G-shuf: the pilot's winner under Protocol 02 rule 3 (v1, v2 or v3).
* Evaluations: full dev (1,197 windows of 1024 targets) at 12M/24M/46M/68M tokens (K2's points)
  with per-window sums saved; full dev and full **test** at 100M with per-position NLL saved.
  Test is read once, here.
* After each run, inference-only analysis on dev and test: minting under every rule, leak,
  beats-prefix, rank, re-tokenised bpb, logit-term shares, correction ablations, counterfactual
  spelling (KAS-G).

## Runs
Kernel X (`kq5-r3-canonical-x`, launched while the pilot is still running, because none of these
arms depends on the generator choice): GPU0 Dense s1, KAS-P s1, KAS-U16 s1, KAS-0 s1, then
vocabulary growth on those four checkpoints | GPU1 Dense s2, KAS-P s2, KAS-U16 s2, KAS-0 s2.
Kernel G (`kq5-r3-canonical-g`, after the pilot fixes the generator): GPU0 KAS-G s1,
KAS-G-shuf s1, growth | GPU1 KAS-G s2, KAS-G-shuf s2, head-cost benchmark.
Seeds change initialisation only. A run that fails for a code bug may be rerun at most twice
(planning limit); reruns are logged.
Gate kept from Protocol 02: if the pilot reveals a defect that affects kernel X's arms (e.g. K2's
ordering fails for a reason traced to code or configuration), kernel X's runs are discarded and
rerun after the fix; this is logged, not hidden.

## Uncertainty
Two-level bootstrap (B = 2,000): evaluation windows resampled jointly across all arms (paired),
seeds resampled with replacement within each arm. 95% percentile intervals. Minting statistics
resample windows as clusters of sites. Seed spread (|s1 − s2|) is reported for every arm; no
difference smaller than the largest measured seed spread is described as a difference.

## Pre-registered verdicts (all on TEST at 100M tokens unless stated)

**H1 (generated correction recovers ≥ half of the stored one).**
ρ = [bpb(KAS-P) − bpb(KAS-G)] / [bpb(KAS-P) − bpb(KAS-U16)], seed-mean bpb per arm.
* Precondition: the denominator's 95% CI excludes 0. Otherwise H1 is UNDEFINED and reported so.
* SUPPORTED if the CI of ρ lies entirely ≥ 0.5; REJECTED if entirely < 0.5; else INCONCLUSIVE.
* Outcome band from the plan: ρ ≥ 0.5 success; 0.2–0.5 partial; < 0.2 informative failure.
* Falsifier: a CI entirely below 0.5 (the experiment can produce it: ρ = 0 if KAS-G = KAS-P).

**H2 (generated parameters mint better than KAS-U16's best training-free rule).**
Sites = test positions whose next two targets are a held-out merge's pieces (a, b).
Per-site regret (bits) averaged over the two seeds of each arm. KAS-G's rule and KAS-U16's rule
are each the one with the lowest mean regret on DEV (KAS-U16 candidates: floor, zero, count,
inherit, inherit_prior; KAS-G candidates: all of its rules).
Δregret = mean_sites[regret(KAS-G) − regret(KAS-U16)]; Δleak = leak_bpb(KAS-G) − leak_bpb(KAS-U16).
* SUPPORTED if Δregret's CI < 0 entirely and Δleak's CI lower bound ≤ 0 (leak not worse).
* REJECTED if Δregret's CI ≥ 0 entirely, or if Δregret < 0 but Δleak's CI > 0 entirely
  (reported as a trade-off).
* INCONCLUSIVE otherwise.
* If KAS-U16's best rule is "inherit" and it beats every KAS-G rule, the report says so plainly.

**H3 (the gap lives in frequent tokens).** d = NLL(Dense) − NLL(KAS) per target (nats),
seed-averaged, bucketed by the target's training count:
[0,1), [1,10), [10,100), [100,1K), [1K,10K), [10K,100K), [100K,1M), [1M,∞).
Buckets with < 1,000 test targets are reported but not used for the verdict.
Primary KAS arm: KAS-U16 (K2's comparison); KAS-P and KAS-G reported alongside.
* SUPPORTED if at least one bucket has CI(d) > 0 entirely (KAS better), at least one has
  CI(d) < 0 entirely (Dense better), and every KAS-better bucket is rarer than every
  Dense-better bucket.
* PARTIAL if d decreases with bucket frequency (Spearman ρ < 0 across qualifying buckets) and the
  most frequent qualifying bucket has CI(d) < 0, but no bucket has CI(d) > 0.
* REJECTED if all qualifying buckets share one sign with CIs excluding 0 and no trend, or the
  pattern is reversed (KAS better on frequent, worse on rare).
* INCONCLUSIVE otherwise.

**H4 (structure, not capacity).** Using KAS-G's selected rule for KAS-G-shuf as well:
Δshuf = mean_sites[regret(KAS-G-shuf) − regret(KAS-U16)] and Δspell = mean_sites[regret(KAS-G) −
regret(KAS-G-shuf)].
* If H2 is SUPPORTED: H4 SUPPORTED if Δspell's CI < 0 entirely and Δshuf's CI includes 0 or is
  > 0; REJECTED if Δspell's CI includes 0 (shuffled spelling mints as well).
* If H2 is not SUPPORTED, H4 is NOT APPLICABLE as stated; Δspell is still reported as the direct
  test of whether the generator's minting depends on real spelling.

**Secondary, pre-registered (K2 replication).** Dense − KAS-U16 dev bpb gap at 12/24/46/68/100M:
K2's pattern is reproduced if the gap at 100M is > 0 with CI excluding 0 and larger than at 12M.

## Predictions (made 2026-10-01 15:45, before any pilot result was seen)
* Ordering at 100M (test bpb): KAS-0 worst by > 0.3; KAS-P next; KAS-U16 below KAS-P by
  0.03–0.08; Dense below KAS-U16 by 0.01–0.04, and the Dense − KAS-U16 gap larger at 100M than
  at 12M (K2's pattern).
* H1: ρ between 0.1 and 0.4 → REJECTED or INCONCLUSIVE (planning prior for ρ ≥ 0.5 ≈ 40%;
  R0b shows spelling explains < 30% of log-frequency variance and ~8% of GPT-2 output-row variance
  linearly).
* H2: KAS-U16's best training-free rule is "count" or "inherit"; KAS-G's generated correction
  lowers mean regret by a small amount (< 0.5 bits/site) — likely INCONCLUSIVE at ~2,100 test sites.
* H3: PARTIAL or SUPPORTED — byte-derived heads no worse on rare buckets, worse on the most frequent.
* H4: NOT APPLICABLE if H2 is not supported; Δspell (KAS-G − KAS-G-shuf) negative if spelling is used.

LOCKED: 2026-10-01 (commit hash recorded in research-log)
