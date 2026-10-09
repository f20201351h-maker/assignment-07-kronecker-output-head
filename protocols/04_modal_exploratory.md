# Protocol 04 — exploratory runs on Modal (written 2026-10-01 19:55, before the runs)

The user made Modal GPU credits available at 19:50. The pre-registered canonical runs stay on
Kaggle T4s (kernels X and G); nothing here enters the H1–H4 verdicts. Modal runs use the same
`kq5` code, the same sha256-verified token arrays and the same RunConfig, on a different GPU
(H100, fp16 autocast), and are labelled with their device.

## Check 1 — does the u_zero initialisation learn at pilot scale? (m2-*)
R2 configuration (8L×512, 20,971,520 tokens, head lr ×0.1): KAS-U16 u_zero s1, KAS-G v3 u_zero s1,
and KAS-P s1 as a same-hardware reference.
Prediction: KAS-U16 (u_zero) beats KAS-P by clearly more than the broken c_zero init did (0.010),
and its correction ablation is worth more than 0.021 bpb. If KAS-U16 (u_zero) is no better than
KAS-P, canonical G's KAS-U16 is suspect and the cause is investigated before G finishes.

## Use as fallback
If a Kaggle canonical kernel fails, the failed runs may be rerun here with the same configs; the
README would then state which runs came from which hardware.
