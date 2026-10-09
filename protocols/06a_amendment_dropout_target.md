# Amendment 06a — a second vocabulary-dropout variant: hide only the stored correction (2026-10-07 21:20)

## What was seen before this amendment
The two KAS-U16-drop25 runs of protocol 06 finished first (test bpb 1.6381 and 1.6313; seed mean 1.6347,
`results/modal/phase2/phase2_progress.json`). That is 0.173 bpb worse than KAS-U16 (1.4613) and at the level
of KAS-P (1.6308), against the report's +0.028 at 20M tokens and this protocol's prediction of +0.015 to
+0.040. The run's logit-term diagnostics (`experiments/modal-p2/runs/p2-kasu-drop25-s1/analysis_test/terms.json`)
show why: the correction term's variance doubled relative to Phase 1's KAS-U16 (10.6 against 3.7) while its
covariance share of the target logit turned negative (−0.23 against +0.11) and the byte-derived term grew
(19.1 against 14.9). Removing the correction at test time costs this arm 0.79 bpb (Phase 1's KAS-U16: 0.36).
The head did not fail to learn; it learned a different arrangement, in which the byte path carries more and
the per-token rows correct it back, and that arrangement is worse with every row present. No minting result
of this arm had been looked at when this amendment was written.

## Why a second variant
Protocol 06 implemented the report's "hiding 25% of the per-token rows" as hiding **both** the learned prior
β_v and the stored correction u_v, because the report calls the unigram bias "exactly the k = 0 case" of the
per-token parameters. Hiding the prior of a frequent token moves its logit by up to ~8 nats that step and
hiding a rare token's prior raises it by a similar amount, so a quarter of the vocabulary is mis-scaled at
random every step. The report's phrase can equally be read as the rows of the |V| × k table U only, with the
bias kept. The two readings make different experiments, and the first has now been run. The second is added
so that the report's claim is tested under both readings rather than rejected on one.

## The added arm (same everything as protocol 06 §2)
| Arm | head | change | seeds |
|---|---|---|---|
| KAS-U16-dropU25 | kasu | `vocab_dropout = 0.25`, `vocab_dropout_target = "u"`: each step a random 25% of tokens have u_v hidden; β_v is kept. | 1, 2 |

Run ids `p2-kasu-dropu25-s{1,2}`; growth on seed 1. Same hardware (RTX PRO 6000), same data, order, budget,
schedule, analysis rules.

## Also added: a second seed of rank 64
All five protocol 06 runs had finished when this was written (`phase2_progress.json`): KAS-U64 seed 1 reached
1.4168 test bpb, below Dense (1.4427, two seeds, spread 0.0048) and below KAS-U16 (1.4613). Protocol 06 made
rank 64 a single-seed exploratory arm with the question "does it train?"; a result below the dense head is a
stronger claim than that question needs, and one seed cannot carry it. Seed 2 (`p2-kasu64-s2`, identical
configuration) is added. Pre-registered reading: the claim "KAS-U64 is below Dense at 100M" is made only if
the two-seed paired CI of bpb(KAS-U64) − bpb(Dense) lies entirely below 0; otherwise it is reported as a
single-seed observation. Prediction: seed 2 within 0.01 of seed 1.

## Verdict rules
Q1, H5 and H6 of protocol 06 are evaluated for this arm with the same rules and thresholds, labelled Q1', H5',
H6'. The protocol 06 arm (both hidden) keeps its own verdicts; neither replaces the other. If the two variants
disagree, both are reported, and the report's claim is described as holding under one reading and not the
other. H8 is evaluated on the two rank-64 seeds.

## Predictions (before the run)
* Q1': +0.01 to +0.05 bpb (the report's +0.028 is now the central guess).
* H5': SUPPORTED — hidden-correction training should make the `zero`/`count` rules usable (regret well below
  KAS-U16's +1.66 bits/site).
* H6': P(DOMINATES KAS-G) ≈ 0.45; P(TRADE-OFF) ≈ 0.4.

## Cost
Two runs with analyses ≈ $2 at RTX PRO 6000 rates (protocol 06 §3 probe). Modal metered before the amendment:
$17.40 plus the five protocol 06 runs in progress.

LOCKED: 2026-10-07 21:22 (commit hash in research-log before launch).
