"""All README figures from results files (PDF + 300-dpi PNG). Missing inputs are skipped.

    python scripts/make_figures.py --analysis results/final/analysis_r3_test.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
NL = chr(10)
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 9,
    "axes.titlesize": 10, "axes.titleweight": "bold", "axes.labelsize": 9, "legend.fontsize": 7.5,
    "legend.frameon": False, "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.2,
    "lines.linewidth": 1.6, "lines.markersize": 4,
})
# Okabe-Ito (colour-blind safe); KAS-G is the arm under test
COL = {"Dense": "#000000", "KAS-0": "#999999", "KAS-P": "#56B4E9", "KAS-U16": "#0072B2",
       "KAS-G": "#D55E00", "KAS-G-shuf": "#E69F00", "KAS-U16 (c_zero)": "#CC79A7"}
MARK = {"Dense": "s", "KAS-0": "x", "KAS-P": "o", "KAS-U16": "^", "KAS-G": "D", "KAS-G-shuf": "v",
        "KAS-U16 (c_zero)": "<"}
ORDER = ["Dense", "KAS-0", "KAS-P", "KAS-U16", "KAS-G", "KAS-G-shuf", "KAS-U16 (c_zero)"]
K2_GAP = {12e6: 0.0031, 24e6: -0.0048, 46e6: 0.0123, 68e6: 0.0229, 100e6: 0.0304}  # reference report (K2)


def save(fig, name):
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=300)
    plt.close(fig)
    print("wrote", name)


def fig_curves(a):
    if not a.get("dev_curves"):
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.7))
    for arm in ORDER:
        c = a["dev_curves"].get(arm)
        if not c:
            continue
        t = sorted(c, key=float)
        ax1.plot([float(x) / 1e6 for x in t], [c[x] for x in t], marker=MARK[arm], color=COL[arm], label=arm)
    ax1.set_xlabel("training tokens (M)"); ax1.set_ylabel("dev bpb"); ax1.set_title("dev bits per byte")
    ax1.legend(ncol=2)
    for key, arm in (("KAS-U16 - Dense", "KAS-U16"), ("KAS-P - Dense", "KAS-P"), ("KAS-G - Dense", "KAS-G")):
        rows = a.get("gap_curves", {}).get(key)
        if not rows:
            continue
        x = np.array([r["tokens"] for r in rows]) / 1e6
        y = np.array([r["b_minus_a"]["point"] for r in rows])
        lo = np.array([r["b_minus_a"]["ci95"][0] for r in rows]); hi = np.array([r["b_minus_a"]["ci95"][1] for r in rows])
        ax2.plot(x, y, marker=MARK[arm], color=COL[arm], label=f"{arm} − Dense (ours)")
        ax2.fill_between(x, lo, hi, color=COL[arm], alpha=0.15)
    ax2.plot([k / 1e6 for k in K2_GAP], list(K2_GAP.values()), ls="--", color="#777777", marker="*",
             label="rank-16 − dense (reference report)")
    ax2.axhline(0, color="#444444", lw=0.8)
    ax2.set_xlabel("training tokens (M)"); ax2.set_ylabel("bpb gap (positive = Dense better)")
    ax2.set_title("gap to the dense head"); ax2.legend()
    save(fig, "fig1_gap_vs_tokens")


def fig_params(a):
    arms = a.get("arms", {})
    if not arms:
        return
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for arm in ORDER:
        v = arms.get(arm)
        if not v:
            continue
        ax.scatter(v["head_params"], v["bpb_mean"], color=COL[arm], marker=MARK[arm], s=30, label=arm, zorder=3)
        if v.get("seed_spread"):
            ax.errorbar(v["head_params"], v["bpb_mean"], yerr=v["seed_spread"] / 2, color=COL[arm], lw=1)
    ax.set_xscale("log"); ax.set_xlabel("output-head parameters"); ax.set_ylabel(f"{a['split']} bpb")
    ax.set_title("head size vs quality"); ax.legend(ncol=2)
    save(fig, "fig2_head_params_vs_bpb")


def fig_buckets(a):
    h3 = a.get("H3", {}).get("buckets")
    if not h3:
        return
    fig, ax = plt.subplots(figsize=(6.0, 2.7))
    arms = [k for k in ("KAS-P", "KAS-U16", "KAS-G") if k in h3]
    labels = [r["bucket"] for r in h3[arms[0]]]
    x = np.arange(len(labels))
    wdt = 0.8 / len(arms)
    for i, arm in enumerate(arms):
        rows = h3[arm]
        y = np.array([r.get("d_nats", np.nan) for r in rows])
        lo = np.array([r.get("ci95", [np.nan, np.nan])[0] for r in rows]); hi = np.array([r.get("ci95", [np.nan, np.nan])[1] for r in rows])
        ax.bar(x + (i - len(arms) / 2 + 0.5) * wdt, y, wdt * 0.9, color=COL[arm], label=arm,
               yerr=[y - lo, hi - y], error_kw={"lw": 0.8, "capsize": 1.5})
    ax.axhline(0, color="#444444", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([f"{l}\nn={r.get('n_targets', 0):,}" for l, r in zip(labels, h3[arms[0]])], fontsize=6.5)
    ax.set_xlabel("target's training count"); ax.set_ylabel("NLL(Dense) − NLL(KAS) (nats)")
    ax.set_title("where the gap lives (positive = byte-derived head better)"); ax.legend()
    save(fig, "fig3_frequency_buckets")


def _retok_deltas(a):
    """Net bpb change from minting every held-out merge and re-tokenising (per arm, per rule,
    seed-mean). New-format entries carry delta_bpb on identical text; old (pilot) entries are
    rule bpb minus the run's _standard bpb."""
    out = {}
    for arm, ex in a.get("extras", {}).items():
        per_rule = {}
        for rt in ex.get("retok", []):
            std = rt.get("_standard", {}).get("bpb")
            for rule, v in rt.items():
                if rule.startswith("_"):
                    continue
                d = v.get("delta_bpb", v["bpb"] - std if std is not None else None)
                if d is not None:
                    per_rule.setdefault(rule, []).append(d)
        out[arm] = {r: float(np.mean(v)) for r, v in per_rule.items()}
    return out


def fig_regret(a):
    t = a.get("minting_table", {}).get(a["split"])
    if not t:
        return
    arms = [k for k in ORDER if k in t]
    rules = []
    for arm in arms:
        rules += [r for r in t[arm] if not r.startswith("_") and r not in rules]
    M = np.full((len(arms), len(rules)), np.nan)
    L = np.full((len(arms), len(rules)), np.nan)
    for i, arm in enumerate(arms):
        for j, r in enumerate(rules):
            if r in t[arm]:
                M[i, j] = t[arm][r]["mean_regret_bits"]
                L[i, j] = t[arm][r]["leak_bpb"]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.6, 2.9), gridspec_kw={"width_ratios": [1.6, 1]})
    im = ax.imshow(M, cmap="coolwarm", aspect="auto", vmin=-np.nanmax(np.abs(M)), vmax=np.nanmax(np.abs(M)))
    for i in range(len(arms)):
        for j in range(len(rules)):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.2f}" + NL + f"{L[i, j] * 1e3:.2f}", ha="center", va="center", fontsize=5.8)
    ax.set_xticks(range(len(rules))); ax.set_xticklabels(rules, rotation=35, ha="right", fontsize=6.5)
    ax.set_yticks(range(len(arms))); ax.set_yticklabels(arms, fontsize=7)
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.8, label="mean regret (bits/site)")
    ax.set_title("regret at sites (top) and leak ×10⁻³ bpb (bottom)", fontsize=8.5)
    d = _retok_deltas(a)
    names, vals, cols = [], [], []
    for arm in arms:
        if arm in d and d[arm]:
            best = min(d[arm], key=d[arm].get)
            names.append(f"{arm}" + NL + f"({best})"); vals.append(d[arm][best] * 1e3); cols.append(COL.get(arm, "#555555"))
    ax2.barh(range(len(vals)), vals, color=cols)
    ax2.set_yticks(range(len(vals))); ax2.set_yticklabels(names, fontsize=6.5); ax2.invert_yaxis()
    ax2.axvline(0, color="#444444", lw=0.8)
    ax2.set_xlabel("Δ bpb ×10⁻³ (negative = minting helps)")
    ax2.set_title("net effect on re-tokenised text", fontsize=8.5)
    save(fig, "fig4_minting_regret_leak_net")


def fig_growth(paths):
    files = sorted(glob.glob(paths))
    if not files:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.7))
    for f in files:
        rid = Path(f).stem
        rows = json.load(open(f, encoding="utf-8"))
        arm = {"dense": "Dense", "kasp": "KAS-P", "kasu": "KAS-U16 (c_zero)", "kasu0": "KAS-U16", "kasg": "KAS-G",
               "kasgshuf": "KAS-G-shuf", "kas0": "KAS-0"}[rid.split("-")[1]]
        if arm == "KAS-0":
            continue        # collapses (+0.46 bpb at 722K); plotted values in results/growth, omitted for scale
        for rule in sorted({r["rule"] for r in rows}):
            rr = [r for r in rows if r["rule"] == rule]
            v = [r["vocab_total"] for r in rr]
            ls = "-" if rule in ("gen_count", "inherit", "bytes") or arm in ("Dense", "KAS-P") else ":"
            rr = [r for r in rr if r["K"] > 0] if ax1 else rr
            base = [r for r in rows if r["rule"] == rule and r["K"] == 0]
            b0 = base[0]["retok_bpb"] if base else rr[0]["retok_bpb"]
            ax1.plot(v[-len(rr):], [r["leak_bpb"] for r in rr], ls=ls, marker=MARK[arm], color=COL[arm], label=f"{arm} ({rule})")
            ax2.plot([r["vocab_total"] for r in rr], [r["retok_bpb"] - b0 for r in rr], ls=ls, marker=MARK[arm], color=COL[arm])
    for ax in (ax1, ax2):
        ax.set_xscale("log"); ax.set_xlabel("vocabulary size after minting")
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax1.set_ylabel("leak (bpb)"); ax1.set_title("probability leaked to minted tokens"); ax1.legend(fontsize=6)
    ax2.set_ylabel("Δ bpb vs no minting (dev text)"); ax2.set_title("net cost of minting K merges"); ax2.axhline(0, color="#444444", lw=0.8)
    save(fig, "fig5_vocab_growth")


def fig_r0b(p):
    if not Path(p).exists():
        return
    r = json.load(open(p, encoding="utf-8"))
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.5))
    for ax, key, title in ((axes[0], "log_frequency", "log training frequency"),
                           (axes[1], "gpt2_wte", "GPT-2 output-embedding rows")):
        if key not in r:
            continue
        models = ["linear", "cnn", "linear_plus_cnn"]
        x = np.arange(len(models))
        for i, tag in enumerate(("true", "shuffled")):
            ax.bar(x + (i - 0.5) * 0.38, [r[key][f"{m}_{tag}"]["r2_test"] for m in models], 0.36,
                   color=["#0072B2", "#E69F00"][i], label=f"{tag} spelling")
        ax.set_xticks(x); ax.set_xticklabels(["linear code", "ByteCNN", "linear + CNN"], fontsize=7.5)
        ax.axhline(0, color="#444444", lw=0.8); ax.set_ylabel("held-out R²"); ax.set_title(title)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.06))
    save(fig, "fig6_r0b_predictability")


def fig_headcost(p):
    if not Path(p).exists():
        return
    r = json.load(open(p, encoding="utf-8"))["results"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.4, 2.5))
    for head, col, lab in (("dense", "#000000", "dense head"), ("kas", "#0072B2", "KAS head")):
        rr = [x for x in r if x.get("head") == head and "error" not in x]
        V = [x["V"] for x in rr]
        ax1.plot(V, [x["fwd_bwd_s_median"] for x in rr], marker="o", color=col, label=lab)
        ax2.plot(V, [x["train_state_bytes_adamw_fp32"] / 1e9 for x in rr], marker="o", color=col, label=f"{lab}: params+AdamW")
        ax2.plot(V, [x["peak_mem_gb"] for x in rr], marker="x", ls=":", color=col, label=f"{lab}: measured peak")
    for ax in (ax1, ax2):
        ax.set_xscale("log"); ax.set_xlabel("vocabulary size V")
    ax2.set_yscale("log")
    ax1.set_ylabel("head fwd+bwd (s, 8192 tokens)"); ax1.set_title("compute still scales with V"); ax1.legend()
    ax2.set_ylabel("GB"); ax2.set_title("stored state does not (KAS)"); ax2.legend(fontsize=6)
    save(fig, "fig7_head_cost_vs_V")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", default=str(ROOT / "results" / "final" / "analysis_m3_test.json"))
    ap.add_argument("--growth", default=str(ROOT / "results" / "growth" / "m3-*.json"))  # Phase 1 arms only
    ap.add_argument("--r0b", default=str(ROOT / "results" / "r0b" / "r0b.json"))
    ap.add_argument("--headcost", default=str(ROOT / "results" / "headcost" / "headcost.json"))
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()
    if args.outdir:
        globals()["FIG"] = Path(args.outdir)
    if Path(args.analysis).exists():
        a = json.load(open(args.analysis, encoding="utf-8"))
        fig_curves(a); fig_params(a); fig_buckets(a); fig_regret(a)
    fig_growth(args.growth); fig_r0b(args.r0b); fig_headcost(args.headcost)


if __name__ == "__main__":
    main()
