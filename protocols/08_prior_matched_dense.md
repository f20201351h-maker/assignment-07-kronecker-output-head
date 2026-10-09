# Protocol 08 — Final control: prior-matched Dense against KAS-U64 (written 2026-10-08, before launch)

Independent-review-prompted control, after Phases 1–3 were complete. Not a new research phase, not an
architecture search. One arm, four runs, nothing else. Phase 1–3 result artefacts are not touched.

## 1. Why this control exists (decision record, written before any result)

An independent read-only review of the code (2026-10-08; its text is not in this repository) observed that
KAS-U64 receives a learned, unigram-initialised per-token output bias while the Dense comparator does not, so the
U64 − Dense gap may partly measure explicit frequency information rather than the structured head. The decision
gate applied before spending (all four had to hold):

1. **Technically real — YES.** `src/kq5/heads.py`: Dense materialises `(W, None)`, no bias of any kind
   (`Head.materialize`, kind "dense"); `model.py` has no biases anywhere and `ln_f` is `LayerNorm(bias=False)`.
   KAS-U (incl. rank 64) has `self.beta = nn.Parameter(prior)`: a per-token bias initialised to the centred
   log-unigram prior of the training split (`centred_log_unigram`), trainable at lr ×1 in the `per_token`
   group with no weight decay (1-D; `train.py::make_optimizer`), plus a 33-entry length bias. README §3's table
   says the same. Dense can only emulate a bias through its V × d rows along a near-constant direction of h,
   which it must learn and which weight decay 0.1 acts on.
2. **Relevant to the claim — YES.** The four-seed frequency-bucket decomposition of Dense − KAS-U64
   (`results/phase3/analysis_m3_p2_p3_test.json`, `H3.buckets["KAS-U64"]`, test split) puts almost the whole
   gap in tokens with training count < 100,000: +0.92 nats/target at [10,100), +0.32 at [100,1K), +0.19 at
   [1K,10K), +0.07 at [10K,100K), −0.01 at [100K,1M), +0.03 at ≥1M. Recomputed from those rows,
   ~92% of the summed NLL difference comes from the buckets below 100,000. That is the signature a per-token
   frequency prior would leave; it is also the signature of parameter sharing in the byte-code head. The
   existing data cannot tell them apart. KAS-U64's learned prior stays near its initialisation
   (`prior_drift.json`: mean drift −0.01 to −0.05 nats per count bin), so the prior it benefits from is
   essentially the counted one.
3. **Materially important — PLAUSIBLY.** The headline is a 0.025 bpb margin. If a bias closes half of it,
   "the rank-64 structured head beats Dense" becomes "a per-token bias explains much of it", a change of
   interpretation, not a footnote. If it closes little, the structural reading is strengthened. Neither can be
   ruled out from existing evidence. Against: the dev curve (`gap_curves["KAS-U64 - Dense"]`) shows the U64
   lead at 12M tokens (−0.027) vanishing by 24M–46M and re-emerging during the learning-rate decay
   (−0.017 at 68M, −0.023 at 100M), which does not look like an initialisation head-start alone.
4. **Currently unresolved — YES.** No Dense arm with a bias exists in Phases 1–3 (`grep` of configs and run
   records; README §3 table). KAS-0 vs KAS-P measures the prior's value for the byte-code head (0.31 bpb), not
   for Dense. The only other bias arm we knew of (Subramanya Naik's) is on a KAS head at 5M tokens. Nothing answers this question.

Decision: **RUN PRIOR-MATCHED DENSE CONTROL** (four runs). Recorded here and in `research-log.md` before launch.

## 2. Research question

Does KAS-U64 still outperform Dense when Dense is given the same unigram-prior / output-bias information,
without otherwise retuning or improving Dense? Distinguish "U64 beats the existing Dense configuration" from
"the advantage is not explained by U64's access to an explicit log-unigram per-token bias".

## 3. The arm: prior-matched Dense (`Dense-prior`, run ids `p4-densep-s{1,2,3,4}`)

Exactly the four existing Dense configurations (`m3-dense-s1`, `m3-dense-s2`, `p3-dense-s3`, `p3-dense-s4`) with
one change, `dense_bias = True` (`scripts/modal_app.py::phase4_specs` asserts that each config differs from its
source only in `run_id` and `dense_bias`):

    logit_v(h) = <W_v, h> + beta_v,   beta_v initialised to log((c_v+1)/(N+V)) − mean over the training vocabulary

* `beta` is an `nn.Parameter` of shape V, in the `per_token` optimiser group (lr ×1, weight decay 0: the same
  treatment as KAS-U64's `beta`); `W` keeps lr ×1 and weight decay 0.1 as in every Dense run. Head and body
  gradients are clipped separately as before (`beta` is in the head group, like KAS-U64's).
* The bias is created after `W` is drawn and from the prior table (no random numbers), so for a given seed `W`,
  `W_in`, the positions and every block are bit-identical to the Dense run of that seed
  (`tests/test_prior_matched.py::test_prior_matched_dense_is_dense_plus_one_bias`). Each `p4-densep-s{k}` is
  therefore the Dense run `s{k}` with a bias and nothing else: same initialisation, same data order
  (`data_seed` 1234), same 1,526 steps, warmup, cosine schedule, fp16 recipe, evaluation windows, dev+test
  analyses, torch build (2.10.0+cu128 in the Modal image) and hardware (RTX PRO 6000).
* **Not matched, on purpose:** KAS-U's 33-entry length bias (lr ×0.1, zero-initialised) — a per-token bias spans
  every per-length bias, so adding it would change nothing the model can express; it is noted, not copied.
  Nothing else about Dense is changed: no learning-rate, weight-decay or width tuning, no new initialisation.
* Minting rules for the biased dense head mirror KAS-U's (`minting.rules_for("dense", dense_bias=True)`): the
  `mean` rule gives a minted token the training-vocabulary mean prior (centred zero) and the mean row; `inherit`
  copies the prefix's row and bias plus log P(b | a). Minting is secondary here and is reported, not judged.
* KAS-U64 is not modified. No other arm is added.

## 4. Comparators (reused, not rerun)

Dense seeds 1–4 (`m3-dense-s{1,2}`, `p3-dense-s{3,4}`) and KAS-U64 seeds 1–4 (`p2-kasu64-s{1,2}`,
`p3-kasu64-s{3,4}`), test split, as analysed in `results/phase3/`. Four prior-matched runs give four paired
(same-initialisation) Dense/Dense-prior differences; two would leave a df-1 interval (the protocol 07 lesson).

## 5. Pre-registered analysis (`scripts/prior_analysis.py` → `results/phase4/prior_verdict.json`; also
`scripts/final_analysis.py --prefix m3 p2 p3 p4 --out results/phase4` for the joint tables and buckets)

All on the test split, bpb at 100,007,936 tokens, seed means over seeds 1–4.

* **P = bpb(Dense) − bpb(Dense-prior)** (positive = the prior helps Dense): paired-by-seed t interval
  (df 3; the pairs share initialisation) and the Protocol 03 two-level bootstrap (windows paired, seeds within
  arm, B = 2,000).
* **R = bpb(KAS-U64) − bpb(Dense-prior)** (negative = U64 still ahead): two-level bootstrap and Welch on the
  four seed means per arm.
* **G = bpb(Dense) − bpb(KAS-U64)** (the original gap, 0.0250 over four seeds) and the **share explained
  = P / G**, with a bootstrap interval from the joint draws.
* Per-seed table; seed standard deviations; the 16 (U64, Dense-prior) pairs; rank order of the twelve runs;
  frequency-bucket decomposition of Dense − Dense-prior beside Dense − KAS-U64; the `Dense-prior` ablation
  (`bpb_no_prior`: the trained model scored with its bias removed), term shares and prior drift, reported as
  diagnostics.

## 6. Interpretation rule (fixed now; not to be changed after results are seen)

The bootstrap interval of R governs whether U64 is "clearly ahead"; the paired t interval of P governs whether
the prior "helps Dense"; the share point estimate places the band. Welch on R is reported beside the bootstrap;
if the two disagree about zero, both are stated and the bootstrap decides the word, as written here.

* **D — PRIOR-MATCHED DENSE MATCHES OR BEATS U64:** point R ≥ 0.
* **E — AMBIGUOUS:** point R < 0 but the bootstrap interval of R includes 0; or share ≥ 0.5 while the paired
  interval of P includes 0 (large but noisy).
* Otherwise (U64 clearly ahead):
  * **A — CLOSES MOST:** share ≥ 0.5 and the paired interval of P lies above 0.
  * **B — PARTIAL:** 0.2 ≤ share < 0.5 and the paired interval of P lies above 0.
  * **C — LITTLE CHANGE:** share < 0.2, or the paired interval of P includes 0 (this includes a negative P: a
    bias that does not help, or hurts, this Dense configuration).

Reading of each band: A — a substantial part of the apparent U64 advantage is attributable to explicit
frequency-prior information, not evidence for the structured rank-64 design alone; B — the prior explains part,
not all; C — the simple unigram-prior explanation does not account for the U64 advantage under this controlled
setup; D — say so directly, the headline does not survive the control; E — say that, no certainty manufactured.
Whatever the band, no claim about other ranks, longer training, other data, tokenizers or tuned baselines.

The Phase 3 verdict word (WEAKENED) and every Phase 1–3 number stay as they are.

## 7. Seeds, runs, stopping

Four runs, seeds 1–4, ids above; launched together by the server-side orchestrator (tag `phase4`, no growth
task). Nothing else is launched whatever the outcome. A run that fails (crash, incomplete step count, skipped
fp16 steps > 0) is rerun once, only if the projected spend stays under the ceiling below; a second failure leaves
that seed out and the verdict is reported on the remaining seeds as UNCERTAIN-with-n. The run count is not
changed after any result is seen.

## 8. Compute ceiling and estimate (ESTIMATED before launch; actual cost reported afterwards)

* Authorised by the user for this pass: ≈ $5.60 of Modal credit, a hard ceiling. `modal billing summary`
  reading before launch is recorded in `research-log.md`; the charge of this pass is the delta from it.
* Measured inputs: Dense training wall 465–472 s per run (four runs, `summary.json`); dev and test analyses
  13–14 s each (`analysis_*/done.json`); RTX PRO 6000 list rate $3.03/h as read 2026-10-07 (not re-read today).
  Four runs ≈ 4 × 500 s ≈ 0.56 GPU-h ≈ $1.7 at list, plus container start-up and the CPU orchestrator.
* Cross-check: the protocol 07 pass (two Dense + two KAS-U64 runs ≈ 0.60 GPU-h ≈ $1.8 at list) moved the
  metered reading by $2.09 (22.48 → 24.57; the reading was unchanged this morning), ratio ≈ 1.14.
  Estimate for this pass: **≈ $2.0, upper guess $2.5**, inside the $5.60 ceiling with room for one rerun.
* If the reading after the runs implies more than $5.60, nothing further is started and the overrun is reported.

## 9. Predictions (before launch)

P between 0 and +0.012 bpb, central guess +0.005; share ≈ 0.2 (0 to 0.5); most likely band B or C
(probability ≈ 0.7 together), A ≈ 0.2, D ≈ 0.05, E ≈ 0.05. The re-emergence of the U64 lead late in training
argues against a pure head-start effect; the rare-token concentration of the gap argues for some bias effect.

LOCKED: 2026-10-08 (commit hash recorded in `research-log.md` before launch).
