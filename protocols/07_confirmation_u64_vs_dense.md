# Protocol 07 — Confirmation pass: KAS-U64 against Dense on two fresh seed pairs (written 2026-10-07, before launch)

Confirmation only. No architecture search, no retuning, no new ranks, no new initialisation, no new hyper-parameters.

## Existing evidence (Phase 2, protocol 06 / 06a; `results/phase2/analysis_m3_p2_test.json`)
Test bpb at 100,007,936 tokens, two seeds per arm, seeds 1 and 2 (seeds change initialisation only; the data
order is the same for every arm and seed, `data_seed` 1234):

| arm | seed 1 | seed 2 | mean |
|---|---:|---:|---:|
| Dense (`m3-dense-s{1,2}`) | 1.4452 | 1.4403 | 1.4427 |
| KAS-U64 (`p2-kasu64-s{1,2}`) | 1.4168 | 1.4124 | 1.4146 |

bpb(KAS-U64) − bpb(Dense) = −0.0281, paired two-level bootstrap 95% [−0.0337, −0.0229]; Welch (df ≈ 2)
[−0.0422, −0.0141]. Every KAS-U64 seed is below every Dense seed. The rank-64 arm was added during a broader
search (protocol 06), and its second seed after the first had been seen below Dense (amendment 06a); the result
has not yet been tested on seeds that were not part of that search.

## Question
Does the KAS-U64 advantage over Dense reproduce in two fresh matched seed pairs (seeds 3 and 4) trained after
the search that produced it?

## Arms (ids `p3-*`), frozen
* `p3-dense-s3`, `p3-dense-s4`: exactly the `m3-dense-s1` configuration (`configs/jobs/r3_canonical_x.json`
  via `scripts/modal_app.py::canonical_specs`) with `seed` 3 / 4. Dense untied head, head lr ×1.
* `p3-kasu64-s3`, `p3-kasu64-s4`: exactly the `p2-kasu64-s1` configuration (`phase2_specs`: kasu, rank 64,
  `u_zero`, head lr ×0.1) with `seed` 3 / 4.
`phase3_specs` asserts that each new config differs from its source only in `run_id` and `seed`. Same data
arrays and manifest, same window order, same 1,526-step budget, warmup, cosine schedule, fp16 recipe,
per-group clipping, same dev/test evaluation windows, same dev+test analyses (minting under every rule,
re-tokenised bpb, term shares, ablations), same torch build (2.10.0+cu128), same hardware (RTX PRO 6000, the GPU of
every m3 and p2 run; no new GPU survey — the Phase 2 probe is less than two hours old and nothing has changed).
Difference from Phase 2 that is documented here before running: no vocabulary-growth task for the new seeds
(growth was run on seed 1 only in Phases 1 and 2; it is not part of the quality comparison).

## Pre-registered analysis and interpretation rule (fixed now; not to be changed after seeing results)
All on the test split. Δ = bpb(KAS-U64) − bpb(Dense) (negative = KAS-U64 better).

1. **Fresh pairs alone** (seeds 3, 4 of each arm; `scripts/final_analysis.py --prefix p3` and
   `scripts/confirm_analysis.py`): per-seed bpb; Δ_fresh with the Protocol 03 two-level bootstrap (windows
   paired, seeds within arm, B = 2,000) and a Welch interval on the two seed means per arm.
2. **All seeds combined** (seeds 1–4 per arm; `--prefix m3 p2 p3`): the same two intervals with four seeds per
   arm; per-arm seed standard deviation; the number of the 16 (KAS-U64 seed, Dense seed) pairs in which KAS-U64
   is lower; the rank order of all eight runs.
3. **Verdict on the original advantage**, from the fresh pairs alone:
   * **CONFIRMED** if both fresh KAS-U64 seeds are below both fresh Dense seeds, and Δ_fresh's bootstrap CI and
     Welch CI both lie entirely below 0.
   * **WEAKENED** if Δ_fresh < 0 but any of the three conditions above fails, or if |Δ_fresh| is less than half
     of the original advantage (0.0140).
   * **REVERSED** if Δ_fresh ≥ 0.
   * **UNCERTAIN** if a run fails or finishes off its planned step count (it is then rerun once; a second
     failure leaves the verdict UNCERTAIN and is reported as such).
   The combined four-seed intervals are reported alongside but do not change the verdict word.
4. Nothing else is launched after the four runs, whatever the outcome.

## Predictions (before launch)
Δ_fresh between −0.020 and −0.035; CONFIRMED with probability ≈ 0.8 (the four existing runs are separated by
0.023 at the closest pair, five times Dense's seed spread, but the second KAS-U64 seed was added post hoc and
two seeds give a wide Welch interval).

## Budget
Four runs with dev+test analyses on RTX PRO 6000 ≈ $4.5 (Phase 2 averaged ≈ $1.1 per run with analyses; Dense
trains in ≈ 470 s, KAS-U64 in ≈ 545 s). Authorised: ≈ $7.50 above the Modal reading at launch. Nothing further
is started if the delta approaches that figure.

LOCKED: 2026-10-07 (commit hash recorded in research-log before launch).
