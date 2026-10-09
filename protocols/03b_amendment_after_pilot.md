# Protocol 03 — amendment after the pilot (2026-10-01 17:30, before any canonical KAS-U16, KAS-G or KAS-G-shuf result)

This is the pilot gate that Protocols 02 and 03 reserved ("if K2's ordering is not reproduced,
find out why before the canonical runs"). Hypotheses, metrics, budgets and verdict rules are
unchanged. Only the initialisation of the rank-16 correction changes, for both corrected arms.

## What the pilot showed (dev bpb at 20,971,520 tokens; results/r2/summary.json)
| Dense s1/s2 | KAS-0 | KAS-P s1/s2 | KAS-U16 s1/s2 | KAS-G v1/v2/v3 | KAS-G-shuf v1 |
|---|---|---|---|---|---|
| 1.8666 / 1.8707 | 2.3118 | 1.9343 / 1.9371 | 1.9255 / 1.9258 | 1.8898 / 1.9133 / 1.8794 | 1.8899 |

* K2's ordering is not reproduced: KAS-U16 gains 0.010 bpb over KAS-P (K2: 0.098) and Dense leads
  KAS-U16 by about 0.06 (K2: a tie).
* KAS-G beats KAS-U16, but the shuffled-spelling control matches KAS-G v1 (1.8899 vs 1.8898), so
  that in-vocabulary gain is not from spelling.

## Diagnosis (experiments/kq5-r2-pilot/runs/r2-kasu-s1/model.pt)
The stored table never learned. corr(U_trained, U_init) = 0.9997; mean relative movement of a row
is 0.018 for tokens seen < 1,000 times and ≤ 0.11 for the most frequent; ‖C‖_F = 0.16. Removing
the correction costs KAS-U16 only 0.021 bpb (KAS-G: 0.101). Cause: our initialisation C = 0,
U ~ N(0, 1). Adam moves each U entry by about lr = 1e-3 per step from a magnitude-1 start, and C
(lr × 0.1) grows slowly, so the stored correction stays a fixed random projection. This is a defect
of our re-implementation, not a property of stored corrections; the reference report's rank-16
correction did learn (0.098 bpb).

## Change (applies to the canonical KAS-U16, KAS-G and KAS-G-shuf runs)
`corr_init = "u_zero"`: per-token correction vectors start at exactly 0 — the stored U = 0 for
KAS-U16, the generator's u-output layer = 0 for KAS-G / KAS-G-shuf — and C ~ N(0, 1/d). This is
the LoRA convention, gives every stored row a gradient from step 0, keeps both arms function-
preserving at initialisation, and treats the stored and generated corrections identically.

## Consequences
* The canonical KAS-U16, KAS-G and KAS-G-shuf runs (two seeds each) run in kernel G with u_zero.
* Kernel X's KAS-U16 runs (already training with c_zero) are kept and reported as an
  initialisation ablation, "KAS-U16 (c_zero)". They are not the KAS-U16 of H1–H4.
* Generator: v3 by Protocol 02 rule 3 (lowest dev bpb: 1.8794 vs v1 1.8898; the gap exceeds the
  largest pilot seed spread, 0.0041).
* The new initialisation has not been checked at the pilot scale on GPU; a CPU smoke check
  confirms the stored rows move (research-log). If canonical KAS-U16 (u_zero) still gains little
  over KAS-P, that is reported as a result: with a standard initialisation, a stored rank-16
  correction is weak in this setting.
