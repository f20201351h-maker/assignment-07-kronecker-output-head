# Protocol 05 — canonical runs move to Modal H100s; Kaggle runs become a replication
# (written 2026-10-01 20:20, before any canonical result exists)

No canonical result has been seen: Kaggle kernels X and G are still running and expose no output
until they finish. The only new information is exploratory (Protocol 04, 20M tokens, dev):
KAS-P 1.9348 on H100 vs 1.9343 / 1.9371 on the Kaggle T4 pilot (hardware agrees); KAS-U16 u_zero
1.8437; KAS-G v3 u_zero 1.8876; ~210K tok/s per H100 (KAS-G v3 ~140K).

## Change
The primary canonical set is now 14 runs on Modal (NVIDIA RTX PRO 6000 Blackwell Server Edition,
torch 2.10.0+cu128, fp16 autocast — the same torch build as Kaggle), run ids `m3-*`, with exactly the configurations of
kernels X and G (configs/jobs/r3_canonical_{x,g}.json: same data arrays, window order, seeds,
100,007,936-token budget, schedule, learning rates, u_zero init for the corrected arms, generator
v3), followed by the same dev and test analyses, vocabulary growth and the head-cost benchmark.
All arms of the primary set run on the same hardware. Hypotheses, metrics and verdict rules
(Protocol 03, errata 03a, amendment 03b) are unchanged and apply to the `m3` set.

## Why
Time (user deadline tonight). The Kaggle T4 path would finish ~23:05 plus re-analysis; the Modal
path finishes in under an hour of wall time for about $11 of the user's credits.

## Replication
Kaggle kernels X and G are left running and, when they finish, are analysed with the same rules as
an independent T4 replication (`r3` set). If the two sets disagree on a verdict, both are reported.

## GPU choice (20:25, by measured cost per token; data/logs/modal_probe.log)
Throughput probe, 32 steps each, KAS-P / KAS-G v3 (tok/s), Modal list price: RTX PRO 6000 190.5K / 119.6K
at $3.03/h; L40S 117.0K / 71.5K at $1.95/h; A100-40GB 117.5K / 79.8K at $2.10/h; H200 231.8K / 154.1K at
$4.54/h; A100-80GB 127.6K / 86.5K at $2.50/h; A10 54.9K / 37.0K at $1.10/h; L4 38.5K / 24.4K. RTX PRO 6000
gives the most tokens per dollar and near-H100 speed; every primary run uses it. Estimated cost ≈ $11.
