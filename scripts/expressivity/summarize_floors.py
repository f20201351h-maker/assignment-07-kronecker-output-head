"""Tables and one figure from the two floor measurements.

    python scripts/expressivity/summarize_floors.py
      -> results/expressivity/floor_summary.md
      -> results/expressivity/floor_curve.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from floor_lib import ROOT, LN2, NATS_PER_TARGET_TO_BPB  # noqa: E402

EV = ROOT / "results/expressivity"
FIG = ROOT / "results/expressivity"


def boot_se(x: np.ndarray, groups: np.ndarray | None, B: int = 2000, seed: int = 0) -> tuple[float, float]:
    """Naive SE of the mean and a cluster bootstrap SE (resampling groups)."""
    rng = np.random.default_rng(seed)
    naive = float(x.std(ddof=1) / np.sqrt(len(x)))
    if groups is None:
        return naive, float("nan")
    g = np.unique(groups)
    means = []
    for _ in range(B):
        pick = rng.choice(g, size=len(g), replace=True)
        sel = np.concatenate([np.flatnonzero(groups == k) for k in pick])
        means.append(x[sel].mean())
    return naive, float(np.std(means))


def main():
    st = json.loads((EV / "floor_static.json").read_text(encoding="utf-8"))
    cx = json.loads((EV / "floor_contextual.json").read_text(encoding="utf-8"))
    npz = np.load(EV / "floor_contextual.npz", allow_pickle=True)
    lines = ["# Floor measurements: summary tables", "",
             "All floors are `min_s CE(p, softmax(Phi s + b)) - H(p)` in nats per target; the bits-per-byte "
             f"equivalent divides by ln2 x {1/NATS_PER_TARGET_TO_BPB/LN2:.3f} bytes/target (test-split ratio).", ""]

    # ---- unigram
    lines += ["## 1. Context-free (unigram) floor by feature set", "",
              "| features | n_features | floor nats | bpb-equiv |", "|---|---:|---:|---:|"]
    for k, v in st["unigram_floor"].items():
        if "check" in k:
            continue
        lines.append(f"| {k} | {v['n_features']} | {v['floor_nats']:.3f} | {v['floor_bpb_equiv']:.3f} |")
    chk = st["unigram_floor"].get("kas_adam1500_check")
    if chk:
        lines.append(f"\nConvergence check (kas, Adam 1500 steps): {chk['floor_nats']:.4f} nats vs LBFGS "
                     f"{st['unigram_floor']['kas']['floor_nats']:.4f}.")
    r = st.get("unigram_additive_residual")
    if r:
        lines.append(f"\nResidual of log-unigram after the best additive fit: std {r['std_nats_token_weighted']:.2f} nats "
                     f"(token-weighted), {r['std_nats_count_weighted']:.2f} (count-weighted); 1–99% range "
                     f"[{r['pct_1_99_nats'][0]:.2f}, {r['pct_1_99_nats'][1]:.2f}] nats; R² {r['r2_token_weighted']:.3f}.")

    # ---- bigram
    bg = st["bigram"]
    lines += ["", f"## 2. Bigram-conditional floor (previous-token contexts: top {bg['n_contexts']}, covering "
              f"{100*bg['position_coverage']:.1f}% of training positions; H = {bg['H_nats_weighted']:.3f} nats)", "",
              "| features | bias | n_features | floor nats | bpb-equiv |", "|---|---|---:|---:|---:|"]
    for name, row in bg["floors"].items():
        for bias, v in row.items():
            lines.append(f"| {name} | {bias} | {v['n_features']} | {v['floor_nats']:.3f} | {v['floor_bpb_equiv']:.3f} |")
    cc = bg.get("convergence_check", {})
    if cc:
        lines.append("\nConvergence check (kas, unigram bias): " + ", ".join(f"{k} = {v:.4f}" for k, v in cc.items()))
    pc = bg.get("kas_unigram_bias_per_context")
    if pc:
        lines.append("\nPer-context floor quantiles (10/25/50/75/90%): " +
                     ", ".join(f"{q:.3f}" for q in pc["floor_quartiles"]) + " nats. Top-16 contexts: " +
                     ", ".join(f"`{c}`={f:.2f}" for c, f in zip(pc["contexts_top16"], pc["floor_top16"])))

    # ---- parallelograms / automaton
    pg, pd, au = st["parallelograms"], st["parallelogram_defect_bigram"], st["automaton"]
    lines += ["", "## 3. Parallelogram census and defect", "",
              f"Among the {pg['top_n_tokens']} most frequent tokens: {pg['n_single_substitution_pairs']} same-length "
              f"single-byte-substitution pairs in {pg['n_groups']} (length, position, byte→byte) groups; "
              f"{pg['n_groups_with_2plus_pairs']} groups hold ≥2 pairs ({pg['n_pairs_in_such_groups']} pairs), i.e. "
              "quadruples on which every additive head must satisfy a context-independent log-odds identity.", ""]
    for ex in pg["examples"][:6]:
        lines.append(f"- L={ex['L']}, p={ex['p']}, {ex['byte_from']!r}→{ex['byte_to']!r}: {ex['n_pairs']} pairs, e.g. "
                     + ", ".join(f"{a!r}/{b!r}" for a, b in ex["examples"][:4]))
    if pd.get("n_groups_measured"):
        lines.append(f"\nEmpirical defect under bigram conditionals (groups with ≥{pd['min_count']} counts for all four "
                     f"tokens): {pd['n_groups_measured']} groups measured; median std of "
                     f"[logodds(a:b) − logodds(c:d)] across contexts = {pd['median_delta_std_nats']:.2f} nats "
                     f"(mean {pd['mean_delta_std_nats']:.2f}). Worst: " +
                     "; ".join(f"{'/'.join(repr(t) for t in w['group'])} std {w['delta_std_nats']:.2f} over "
                               f"{w['n_contexts']} contexts" for w in pd["worst"][:3]))
    lines.append(f"\nAutomaton census: trie nodes {au['trie_nodes']}, minimal DAFSA states {au['dafsa_states']}, "
                 f"DAFSA edges {au['dafsa_edges']} for V = {au['V']}.")

    # ---- contextual
    lines += ["", f"## 4. Contextual floor against the Dense-prior model (`{cx['checkpoint']}`), dev split, "
              f"{cx['n_contexts']} contexts from {len(cx['windows'])} windows", "",
              f"Dense-prior NLL on these positions: {cx['dense_prior_nll_saved_gpu']:.4f} nats (saved GPU array) vs "
              f"{cx['dense_prior_nll_recomputed_cpu']:.4f} (CPU recompute). Reference entropy mean "
              f"{cx['reference_entropy_nats_mean']:.3f} nats.", "",
              "Realised NLL of the historical arms on the SAME positions (nats/target; gap to Dense-prior):", "",
              "| arm | NLL | gap |", "|---|---:|---:|"]
    arms = cx["arms_nll_nats_per_target_same_positions"]
    for k, v in sorted(arms.items(), key=lambda kv: kv[1]):
        lines.append(f"| {k} | {v:.4f} | {v - arms['Dense-prior']:+.4f} |")
    lines += ["", "Floors (KL of the best fixed-feature fit to the Dense-prior distribution) and the fitted mimic's NLL on the "
              "true targets:", "",
              "| features / bias | n_ctx | n_features | floor nats | base KAS floor, same contexts | bpb-equiv | mimic − Dense-prior (true targets) | KAS-P − Dense-prior (same subset) |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    tags = list(npz["tags"]) if "tags" in npz else []
    valid = npz["valid"]
    y = npz["y"]
    # window id per context for the cluster bootstrap
    n_ctx = len(y)
    per_win = cx["positions_per_window"]
    groups = np.repeat(np.arange(len(cx["windows"])), per_win)[:n_ctx]
    for i, (tag, v) in enumerate(cx["floors"].items()):
        kasp_gap = v["arms_nll_same_subset"]["KAS-P"] - v["dense_prior_nll_true_targets_same"]
        se_txt = ""
        if f"floor_{i}" in npz:
            fl = npz[f"floor_{i}"][:v["n_contexts"]]
            vv = valid[:v["n_contexts"]]
            naive, clus = boot_se(fl[vv], groups[:v["n_contexts"]][vv])
            se_txt = f" (SE {naive:.3f}, window-bootstrap SE {clus:.3f})"
        base_same = float("nan")
        if "floor_0" in npz:
            b0 = npz["floor_0"][:v["n_contexts"]]
            base_same = float(b0[valid[:v["n_contexts"]]].mean())
        lines.append(f"| {tag} | {v['n_contexts']} | {v['n_features']} | {v['floor_nats_per_target']:.3f}{se_txt} | {base_same:.3f} | "
                     f"{v['floor_bpb_equiv_global_ratio']:.3f} | {v['mimic_minus_dense_prior_nats']:+.3f} | {kasp_gap:+.3f} |")

    rows_path = EV / "floor_rows.json"
    if rows_path.exists():
        rw = json.loads(rows_path.read_text(encoding="utf-8"))
        lines += ["", f"## 5. Floors of dense row sets (learned or body-generated rows as fixed features; s free in R^d), "
                  f"{rw['n_contexts']} contexts, same windows", "",
                  "| row set | d | floor nats | bpb-equiv | mimic − Dense-prior (true targets) |", "|---|---:|---:|---:|---:|"]
        for k, v in rw["rows"].items():
            lines.append(f"| {k} | {v['d']} | {v['floor_nats_per_target']:.3f} | {v['floor_bpb_equiv']:.3f} | {v['mimic_minus_dense_prior_nats']:+.3f} |")
    (EV / "floor_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    # ---- figure: floor by feature set, three targets
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        FIG.mkdir(exist_ok=True)
        names = [n for n in st["bigram"]["floors"]]
        uni = st["unigram_floor"]
        bgf = {n: st["bigram"]["floors"][n].get("unigram_bias", {}).get("floor_nats") for n in names}
        ctxf = {}
        for tag, v in cx["floors"].items():
            key = tag.split(" | ")[0]
            if "beta (fixed)" in tag and key not in ctxf:
                ctxf[key] = v["floor_nats_per_target"]
        fig, ax = plt.subplots(figsize=(9, 4.2))
        xs = np.arange(len(names))
        w = 0.27
        ax.bar(xs - w, [uni.get(n, {}).get("floor_nats", np.nan) for n in names], w, label="unigram target (context-free)")
        ax.bar(xs, [bgf[n] if bgf[n] is not None else np.nan for n in names], w, label=f"bigram conditionals (top {bg['n_contexts']} contexts)")
        ax.bar(xs + w, [ctxf.get(n, np.nan) for n in names], w, label="Dense-prior model, dev contexts")
        ax.set_xticks(xs)
        ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel("floor, nats per target")
        ax.set_title("Representational floor of fixed-feature log-linear heads (lower = more expressive)")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIG / "floor_curve.png", dpi=150)
        print("wrote", FIG / "floor_curve.png")
    except Exception as e:  # figure is optional
        print("figure skipped:", e)


if __name__ == "__main__":
    main()
