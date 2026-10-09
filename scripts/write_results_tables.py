"""Markdown tables for the README, generated from the analysis JSON (never typed by hand).

    python scripts/write_results_tables.py results/final/analysis_r3_test.json > results/final/tables.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ORDER = ["Dense", "KAS-0", "KAS-P", "KAS-U16", "KAS-G", "KAS-G-shuf", "KAS-U16 (c_zero)"]


def f(x, d=4):
    return "—" if x is None else f"{x:.{d}f}"


def ci(r, d=4):
    return f"{r['point']:+.{d}f} [{r['ci95'][0]:+.{d}f}, {r['ci95'][1]:+.{d}f}]"


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    a = json.load(open(sys.argv[1], encoding="utf-8"))
    out = []
    sp = a["split"]
    out.append(f"### Bits per byte ({sp}, end of training; seed mean, per-seed values, spread)\n")
    out.append("| Arm | bpb | seeds | spread | head params | V-dependent |")
    out.append("|---|---:|---|---:|---:|---:|")
    for arm in ORDER:
        v = a["arms"].get(arm)
        if not v:
            continue
        out.append(f"| {arm} | {v['bpb_mean']:.4f} | {', '.join(f'{x:.4f}' for x in v['bpb_per_seed'])} | "
                   f"{f(v['seed_spread'])} | {v['head_params']:,} | {v['head_V_dependent']:,} |")
    out.append(f"\nLargest seed spread: {f(a.get('max_seed_spread'))} bpb.\n")
    out.append("### Paired differences (bpb, 95% two-level bootstrap)\n")
    out.append("| Difference | estimate [95% CI] |")
    out.append("|---|---|")
    for k, r in a.get("bpb_differences", {}).items():
        if r:
            out.append(f"| {k} | {ci(r)} |")
    h1 = a.get("H1")
    if h1:
        out.append("\n### H1\n")
        out.append(f"* Denominator bpb(KAS-P) − bpb(KAS-U16): {ci(h1['denominator_P_minus_U'])}")
        out.append(f"* ρ = [bpb(KAS-P) − bpb(KAS-G)] / [bpb(KAS-P) − bpb(KAS-U16)]: {ci(h1['rho'], 3)}")
        out.append(f"* Verdict: **{h1['verdict']}** (band: {h1['outcome_band']})")
    h2 = a.get("H2")
    if h2:
        out.append("\n### H2\n")
        out.append(f"* Rules selected on dev: KAS-U16 `{h2['kasu_rule_selected_on_dev']}`, "
                   f"KAS-G `{h2['kasg_rule_selected_on_dev']}`")
        out.append(f"* Δregret (KAS-G − KAS-U16), bits/site: {ci(h2['delta_regret_bits'], 3)} "
                   f"over {h2['delta_regret_bits']['n_sites']:,} sites")
        out.append(f"* Δleak (KAS-G − KAS-U16), bpb: {ci(h2['delta_leak_bpb'], 6)}")
        out.append(f"* KAS-U16 'inherit' beats every KAS-G rule: {h2['kasu_inherit_beats_all_kasg_rules']}")
        out.append(f"* Verdict: **{h2['verdict']}**")
    h4 = a.get("H4")
    if h4:
        out.append("\n### H4\n")
        out.append(f"* Δspell (KAS-G − KAS-G-shuf), bits/site: {ci(h4['delta_spell_kasg_minus_shuf'], 3)}")
        out.append(f"* Δshuf (KAS-G-shuf − KAS-U16), bits/site: {ci(h4['delta_shuf_minus_kasu'], 3)}")
        out.append(f"* Verdict: **{h4['verdict']}**")
    h3 = a.get("H3")
    if h3:
        out.append(f"\n### H3 — NLL(Dense) − NLL(KAS), nats per target (primary: {h3['primary']})\n")
        arms = [k for k in ("KAS-P", "KAS-U16", "KAS-G") if k in h3["buckets"]]
        out.append("| training count | targets | " + " | ".join(arms) + " |")
        out.append("|---|---:|" + "---|" * len(arms))
        for i, r in enumerate(h3["buckets"][arms[0]]):
            if not r.get("n_targets"):
                continue
            cells = []
            for arm in arms:
                rr = h3["buckets"][arm][i]
                cells.append(f"{rr['d_nats']:+.3f} [{rr['ci95'][0]:+.3f}, {rr['ci95'][1]:+.3f}]")
            out.append(f"| {r['bucket']} | {r['n_targets']:,} | " + " | ".join(cells) + " |")
        out.append(f"\nSpearman ρ (bucket order vs d, {h3['primary']}): {h3['spearman_rho_bucket_order_vs_d']:+.3f}. "
                   f"Verdict: **{h3['verdict']}**")
    t = a.get("minting_table", {}).get(sp)
    if t:
        out.append(f"\n### Minting every held-out merge ({sp}; seed mean)\n")
        out.append("| Arm | rule | regret bits/site | beats prefix | median rank | leak bpb |")
        out.append("|---|---|---:|---:|---:|---:|")
        for arm in ORDER:
            if arm not in t:
                continue
            for rule, v in t[arm].items():
                if rule.startswith("_"):
                    continue
                out.append(f"| {arm} | {rule} | {v['mean_regret_bits']:+.3f} | {v['beats_prefix']:.3f} | "
                           f"{v['median_rank']:,.0f} | {v['leak_bpb']:.5f} |")
    ex = a.get("extras", {})
    rows = []
    for arm in ORDER:
        for rt in ex.get(arm, {}).get("retok", []):
            pass
        per = {}
        for rt in ex.get(arm, {}).get("retok", []):
            for rule, v in rt.items():
                if not rule.startswith("_") and "delta_bpb" in v:
                    per.setdefault(rule, []).append((v["delta_bpb"], v["bpb_standard_same_text"], v["n_tokens"],
                                                     v["n_tokens_standard"]))
        for rule, vals in per.items():
            d = np.mean([x[0] for x in vals])
            rows.append(f"| {arm} | {rule} | {np.mean([x[1] for x in vals]):.4f} | {d * 1e3:+.3f} | "
                        f"{100 * (1 - vals[0][2] / vals[0][3]):.2f}% |")
    if rows:
        out.append(f"\n### Net effect: bpb on {sp} text re-tokenised with the 1,000 held-out merges\n")
        out.append("| Arm | rule | standard bpb (same text) | Δ bpb ×10⁻³ | tokens saved |")
        out.append("|---|---|---:|---:|---:|")
        out += rows
    k2 = a.get("K2_pattern")
    if k2:
        out.append(f"\n### K2 pattern (dev): bpb(KAS-U16) − bpb(Dense) first {ci(k2['gap_first'])}, "
                   f"last {ci(k2['gap_last'])}; reproduced: {k2['reproduced']}")
    print("\n".join(out))


if __name__ == "__main__":
    main()
