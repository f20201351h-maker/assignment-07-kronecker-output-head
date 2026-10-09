# Protocols

Each protocol was written and committed before the runs it governs, with its hypotheses, verdict rules and
predictions fixed in advance. Amendments and errata are separate files; the protocol text they amend was left as
written. Amendment 06a was written after the first Phase 2 results had been seen, and it says so.

| Protocol | Written | Governs |
|---|---|---|
| [01](01_r1_smoke.md) | 2026-10-01 | smoke test: throughput, memory, learning-rate probe |
| [02](02_r2_pilot.md) | 2026-10-01 | 21M-token pilot: reproduce the reference ordering, choose the generator |
| [03](03_r3_canonical.md), [03a](03a_errata.md), [03b](03b_amendment_after_pilot.md) | 2026-10-01 | Phase 1 canonical runs, hypotheses H1 to H4; errata from the code review; initialisation fix after the pilot |
| [04](04_modal_exploratory.md), [05](05_amendment_modal_canonical.md) | 2026-10-01 | exploratory runs on Modal; move of the canonical set to Modal before any canonical result |
| [06](06_phase2_later_informed_controls.md), [06a](06a_amendment_dropout_target.md) | 2026-10-07 | Phase 2: vocabulary dropout, the reference report's generator form, rank 64 |
| [07](07_confirmation_u64_vs_dense.md) | 2026-10-07 | confirmation of KAS-U64 against Dense on fresh seeds |
| [08](08_prior_matched_dense.md) | 2026-10-08 | prior-matched Dense against KAS-U64 |

These are public copies of the dated records. A few phrases naming internal planning documents or tools were
reworded; hypotheses, rules, predictions and dates are unchanged. The protocols cite working files that are not
part of this repository: "README §n" means the study report as it stood at the time (its content is in
[docs/EXPERIMENTS.md](../docs/EXPERIMENTS.md)), and `research-log.md` and `audit/` were the working log and review
notes, including the commit hashes that time-stamp each protocol. "The reference report" is the unpublished
lecture report *The Vocabulary-Free Output Head* described in the [README](../README.md#prior-work-and-attribution).
