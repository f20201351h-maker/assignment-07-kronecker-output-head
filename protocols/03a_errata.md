# Protocol 03 — errata and clarifications (2026-10-01 16:10, before any R3 result was seen)

Found by an independent code review before any canonical result. Protocol 03 itself is not edited;
these notes say how its text maps to the code, and where the code is stricter.

1. **Sign slip in the K2 replication rule.** The text says "Dense − KAS-U16 dev bpb gap … is
   reproduced if the gap at 100M is > 0". With K2's numbers Dense is *better* (lower bpb), so the
   intended quantity is **bpb(KAS-U16) − bpb(Dense) > 0 at 100M, and larger than at 12M**. The
   analysis implements the intended sign.
2. **Dev has 1,196 windows**, not 1,197 (1,225,678 tokens / 1024). Test has 1,105.
3. **Scope of the ×0.1 head learning rate.** "KAS head-parameter lr ×0.1" covers W_out, C and the
   33-entry length bias. Per-token tables (KAS-U16's U and β; the Dense head's W) train at ×1. The
   generator trains at gen_lr_mult (1.0 for v1 and v3, 0.1 for v2).
4. **H1 precondition.** Implemented as CI(bpb(KAS-P) − bpb(KAS-U16)) entirely > 0 (the text says
   "excludes 0"). A negative denominator would make the ratio uninterpretable, so that case is also
   reported as UNDEFINED.
5. **H3 "no trend".** Implemented as Spearman ρ(bucket order, d) ≥ 0, computed with Spearman
   (an earlier draft used a Kendall-type statistic with a 0.2 cut-off, which was not registered).
   Only buckets with ≥ 1,000 test targets qualify.
6. **"zero" minting rule** means β = 0 on the centred scale, i.e. the mean log-prior of the
   training vocabulary — not the raw 0 of the reference report.
7. **Exploratory rule `count_cal`** (added after the review, before any result): count prior plus
   the mean drift of trained tokens' effective priors in the same count bin. Reported, but
   excluded from the confirmatory rule selection for H2/H4.
8. **Re-tokenised bpb** is compared with the ordinary bpb of exactly the same text (both streams
   fully tiled, final window padded and masked); the per-arm comparison is unchanged.
9. **Kernel X's analysis** (minting/retok/terms/ablations) ran with the pre-review code. It is
   recomputed from kernel X's checkpoints by `kq5-r4x-reanalysis` with the same code that kernel G
   uses, so all arms are analysed identically; the pre-review outputs are kept as `*_v1code`.
