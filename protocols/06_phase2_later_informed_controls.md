# Protocol 06 — Phase 2: later-informed controls (written 2026-10-07, before any Phase 2 run)

Phase 1 (protocols 01–05; canonical set `m3`, 14 runs, 100M tokens) is the incumbent result and is not
changed by this protocol. Phase 2 adds five runs that test three specific claims of the reference report
that Phase 1 did not know about. Everything below was fixed before the first Phase 2 run started; the
commit hash is recorded in `research-log.md`.

## 1. What is new since Phase 1

Phase 1 was designed from a partial reading of the reference report (the planning notes summarised the bias floor,
the rank-k table, the 100M gap and the inconclusive spelling test). The full reconstruction of the report
(made 2026-10-04 from the lecture recording; its numbers were machine-checked against OCR; not redistributable) contains three items absent from every Phase 1 document
(`grep -i "dropout|2.0498|tanh|ADALM"` over RESEARCH_PLAN, README, research-log, protocols, audit: no hits):

* **(a) Vocabulary dropout (report §06, THIS WORK).** "Hiding 25% of the per-token rows at random each
  training step forces the model to route through the byte-derived logits, so a token with no learned row
  sits in a state it saw constantly during training." On the rank-16 head at 20M tokens: bpb 1.7089 → 1.7370
  (+0.028), ADALM rank at N=10 107 → 6, "converts the property from broken to working". Single run.
* **(b) Byte-derived per-token coordinates (report §09, THIS WORK).** The report's own generated-correction
  arm, `U_v = tanh(occ_v · W₁) · W₂` computed from the Kronecker occupancy, "landed at 2.0498 bpb — 0.24 worse
  than the identical head with no coordinates at all. The learned table encodes something genuinely not
  predictable from spelling; a function of the bytes can only give similar coordinates to similarly-spelled
  tokens, and the model then spends capacity suppressing a term that misleads it."
* **(c) Rank-64 (report §03/§09).** "Rank-64 coordinates fail to train even with variance-corrected
  initialisation — bpb flat around 2.33, top-1 stalled at 5.78% against rank-16's 22.4% ... whether that is
  fundamental is unresolved."

Phase 1 evidence that bears on each:

* (a) KAS-U16 is the arm that mints worst: net re-tokenised Δbpb +2.6 to +3.5 ×10⁻³ against +0.3 (KAS-P) and
  +0.4 (KAS-G); regret under the `zero`/`count` rules +1.66 bits/site against +0.25 for KAS-P
  (`results/final/tables_m3_test.md`). The body has learned to rely on rows a minted token does not have.
  Dropout is the report's own remedy for exactly this; Phase 1 never trained with it. If it works at a small
  quality cost, the stored-correction head could dominate the generated one on both axes, which would change
  the Phase 1 interpretation ("the heads lie on one trade-off; generation moves along it").
* (b) Phase 1's generated correction (ByteCNN v3, shift-aware) recovered 0.69 of the stored correction's gain
  (H1), and its shuffled-spelling control 0.61: a shared network helps a lot even when it cannot read real
  spelling. That contradicts the report's stated mechanism. The disagreement may come from the generator's
  form (a one-hidden-layer tanh MLP on the 8,192-dimensional occupancy, which cannot share a suffix across
  word lengths) or from something in the report's optimisation that we cannot inspect. The form can be
  tested directly.
* (c) Phase 1 found that the `c_zero` initialisation (C = 0, U ~ N(0,1)) freezes the stored table
  (corr(U, U₀) = 0.9997); under `u_zero` (U = 0, C ~ N(0, 1/d)) it learns. The report's rank-64 failure
  "even with variance-corrected initialisation" looks like the same class of defect. Prediction: rank 64
  trains under `u_zero`.

## 2. Arms (run ids `p2-*`)

Each arm is the canonical `m3` configuration of KAS-U16 (`m3-kasu0-s{1,2}`) or KAS-G (`m3-kasg-s{1,2}`)
from `configs/jobs/r3_canonical_g.json` with **exactly one change** (`scripts/modal_app.py::phase2_specs`).
Unchanged: data arrays and manifest, window order (data_seed 1234), 1,526 steps × 64 × 1024 =
100,007,936 tokens, warmup, cosine schedule, lr 1e-3, head lr ×0.1, generator lr ×1, `u_zero`
initialisation, fp16 autocast with fp32 loss, per-group clipping, evaluation windows, the dev/test analysis
rules (`retok_rules` of kernel G), torch 2.10.0+cu128.

| Arm | head | change | seeds |
|---|---|---|---|
| KAS-U16-drop25 | kasu | `vocab_dropout = 0.25`: each training step a random 25% of tokens have their learned prior β_v and stored correction u_v hidden (β_v → 0 = the training-vocabulary mean prior, u_v → 0; no rescaling). Evaluation and analysis use all rows. | 1, 2 |
| KAS-G-occ | kasg | generator = `OccMLP` (report §09 form): `g_v = W₂ tanh(W₁ κ_v + b₁) + b₂` on the exact Kronecker code κ_v (D = 8,192), hidden 92, 755,337 parameters (ByteCNN v3: 751,889), rank+1 outputs as KAS-G (16 correction coordinates + 1 prior residual), `u_zero` (W₂ = 0 at init). | 1, 2 |
| KAS-U64 | kasu | `rank = 64` | 1 (exploratory) |

Analyses after each run: dev and test minting under every rule, re-tokenised bpb, logit-term shares,
ablations (same code path as `m3`); vocabulary growth on seed 1 of each arm. Hidden rows are redrawn every
step with the run's torch seed; the evaluation code (`per_position_nll`, `analysis.load_run`) puts the model
in eval mode, so no dropout touches any reported number.

A run that fails for a code bug may be rerun at most twice; reruns are logged.

## 3. Hardware (user requirement)

A 32-step throughput probe of `p2-kasu-drop25-s1` and `p2-kasg-occ-s1` on B200, H100 and RTX PRO 6000
(`modal run scripts/modal_app.py::probe_phase2`) at the Modal list rates read on 2026-10-07
(B200 $6.25/h, H100 $3.95/h, RTX PRO 6000 $3.03/h). Choice rule, fixed now: the GPU with the lowest
projected cost of the five runs (100,007,936 tokens ÷ measured tok/s × rate, plus the same ratio for
analysis), except that RTX PRO 6000 — the hardware of the incumbent `m3` set — is chosen if it is within
10% of the cheapest, so that every Phase 2 arm is compared with its Phase 1 comparator on identical hardware.

## 4. Pre-registered questions and verdict rules

All on the **test** split at 100M tokens, seed means, two-level bootstrap exactly as Protocol 03 (windows
paired across arms, seeds within arms, B = 2,000, 95% percentile intervals); minting statistics resample
windows as clusters of sites. Dev selects rules; test is read once per Phase 2 run (the Phase 1 test arrays
are reused unchanged). Comparators are the `m3` arms. The analysis is `scripts/final_analysis.py --prefix m3 p2`
(adds the Phase 2 arms to the Phase 1 analysis; Phase 1 verdict code is untouched) plus
`scripts/phase2_analysis.py` for the quantities below that the Phase 1 script does not compute.

**Q1 — quality cost of vocabulary dropout.** Δ₁ = bpb(KAS-U16-drop25) − bpb(KAS-U16), with CI. Reported as
a number (the report's value is +0.028 at 20M, single run); no verdict word.

**H5 — vocabulary dropout restores minting to the stored-correction head.** Rules: each arm's rule with the
lowest mean regret on dev among {floor, zero, count, inherit, inherit_prior} (Protocol 03's H2 selection).
Δregret = mean over the 2,110 test sites of regret(KAS-U16-drop25) − regret(KAS-U16); Δleak likewise.
* SUPPORTED if Δregret's CI is entirely < 0 and Δleak's CI lower bound ≤ 0.
* REJECTED if Δregret's CI is entirely ≥ 0, or Δregret < 0 but Δleak's CI entirely > 0.
* INCONCLUSIVE otherwise.
Secondary (reported, not a verdict): net re-tokenised Δbpb under each arm's rule with the lowest net dev
Δbpb (criterion added after the Phase 1 analysis review), per seed and seed mean; regret under the `zero` rule specifically (the state
dropout trains for).

**H6 — does the dropout head dominate the generated-correction head?** Axes as in the Phase 1 quality-vs-
minting table: in-vocabulary test bpb and net re-tokenised test Δbpb under the rule chosen by net dev Δbpb.
* DOMINATES if bpb(KAS-U16-drop25) − bpb(KAS-G) has a CI entirely < 0 **and** net Δbpb is lower for
  KAS-U16-drop25 in both seeds and in the seed mean **and** Δregret (own rules, as H5) against KAS-G does not
  have a CI entirely > 0.
* TRADE-OFF if bpb is better (CI < 0) but net Δbpb is worse in the seed mean.
* DOMINATED if KAS-G is better on both axes.
* INCONCLUSIVE otherwise.
Also reported: the same two axes against Dense (does Dense still dominate the stored-correction head once
it is trained with dropout?).

**H7 — does the report's §09 negative result replicate?** d₇ = bpb(KAS-P) − bpb(KAS-G-occ) (positive =
coordinates help). The report's result is the report's own head without coordinates minus with
coordinates: −0.24.
* REPLICATES if d₇'s CI is entirely < 0 (the occupancy-MLP coordinates hurt).
* DOES NOT REPLICATE if d₇'s CI is entirely > 0 (they help).
* INCONCLUSIVE otherwise.
Secondary: ρ_occ = d₇ / [bpb(KAS-P) − bpb(KAS-U16)] with CI, on the H1 scale; bpb(KAS-G) − bpb(KAS-G-occ)
with CI (does the shift-aware generator beat the report's form at matched parameters?); KAS-G-occ's
minting under its generated rules; the term-share and ablation diagnostics.

**H8 — does rank 64 train under `u_zero`?** (single seed; the report's failure was ~0.6 bpb, near its
no-correction arm.)
* FAILS if bpb(KAS-U64) ≥ bpb(KAS-P) (no better than no correction).
* TRAINS if bpb(KAS-U64) − bpb(KAS-U16) lies within ±0.03 or is negative (CI reported; one seed, so the
  m3 KAS-U16 seed spread 0.0010 and the largest m3 spread 0.0086 are the noise references).
* PARTIAL otherwise.

**Falsifiers.** H5: Δregret CI ≥ 0 (possible: dropout may simply weaken the stored rows without teaching the
body to do without them). H6: either axis worse (possible: the quality cost may exceed 0.05 bpb). H7: CI of
d₇ below 0 (possible: the arm can be worse than KAS-P, as in the report). H8: KAS-U64 at KAS-P level or worse.

## 5. Predictions (2026-10-07, before any Phase 2 result)

* Q1: Δ₁ between +0.015 and +0.040 bpb (the report: +0.028 at 20M).
* H5: SUPPORTED. Regret under `zero`/`count` falls from +1.66 to below +0.5 bits/site; net Δbpb from
  +2.6 ×10⁻³ to below +1.0 ×10⁻³; leak falls.
* H6: genuinely uncertain. Point guess: bpb(KAS-U16-drop25) ≈ 1.48–1.50 (better than KAS-G's 1.5138); net
  minting roughly at KAS-G's level (+0.4 ×10⁻³). P(DOMINATES) ≈ 0.5, P(TRADE-OFF) ≈ 0.35.
* H7: DOES NOT REPLICATE (≈ 0.7): KAS-G-occ better than KAS-P by 0.05–0.12 bpb; the ByteCNN better than the
  occupancy MLP by 0.00–0.05. If it does replicate, the generator's form (shift-awareness) is what makes
  Phase 1's H1 result possible, which is itself a positive finding.
* H8: TRAINS (≈ 0.8), within 0.01 of KAS-U16.

## 6. Budget and stopping

Modal metered cost at the start of Phase 2: $16.71 (`modal billing summary`, 2026-10-07); user-stated
usable headroom ≈ $13.50. Estimated Phase 2 cost: probe ≈ $1, five runs with analyses ≈ $5–7. No new
Phase 2 run is started if the remaining headroom would fall below $2; the user is then asked.

LOCKED: 2026-10-07 (commit hash recorded in research-log before launch).

## Erratum (2026-10-07 21:50, after an analysis review; protocol text above left as written)
Item (c) in §1 was wrong about chronology: Phase 1's planning documents (not included in this repository)
already contained the report's rank table including "rank 64: 2.33, fails to train" and the conjecture that this
"looks like an initialisation artefact", listed as an unrun optional item. The grep in §1 did not search for
"rank 64" and so could not support item (c). Only the sentence "even with variance-corrected initialisation ...
whether that is fundamental is unresolved" was new. H8 executes a Phase 1 prediction. Verdict rules unaffected.
