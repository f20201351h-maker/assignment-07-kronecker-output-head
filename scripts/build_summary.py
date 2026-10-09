"""Summary tables and figures 9-10, generated from saved artefacts (CPU only; no training, GPU or network).

Every number in the generated blocks is read from a results file; nothing is typed by hand. Sources:

  results/phase4/analysis_m3_p2_p3_p4_test.json   all trained arms (joint analysis, every run present)
  results/phase4/prior_verdict.json                protocol 08 (prior-matched Dense)
  results/final/analysis_m3_test.json              protocol 03 verdicts (H1-H4, K2 pattern)
  results/phase2/phase2_verdicts.json              protocol 06/06a verdicts (Q1, H5-H8)
  results/phase3/confirm_verdict.json              protocol 07 verdict
  results/phase4/independent_diagnostics.json     independent recomputation of protocol 08 from NLL arrays
  results/expressivity/evidence_audit.json         non-EOS teacher-projection numbers, codec mapping,
                                                   same-length census, head accounting
  results/expressivity/fit_checks.json             four-context float64 convergence check
  results/expressivity/*.json                      model-free targets, 512-context convergence, row sets

Writes results/summary/summary_values.json and results/summary/tables.md, figures fig9 and fig10,
and refreshes the blocks between `<!-- table:NAME -->` and `<!-- /table:NAME -->` in README.md,
docs/EXPERIMENTS.md and docs/EXPRESSIVITY.md.

    python scripts/build_summary.py           # write everything
    python scripts/build_summary.py --check   # exit 1 if any document block differs from the artefacts
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import make_figures as mf  # noqa: E402  (shared figure style, colours and save())

plt = mf.plt
EV = ROOT / "results/expressivity"
OUT = ROOT / "results/summary"
DOCS = [ROOT / "README.md", ROOT / "docs/EXPERIMENTS.md", ROOT / "docs/EXPRESSIVITY.md"]
MIB = 2 ** 20

# Descriptive metadata only (no numbers): where each arm entered the study and what it changes.
ARM_META = {
    "Dense-prior": ("P08", "Dense + trainable per-token bias initialised to log-unigram counts"),
    "KAS-U64": ("P06a, P07", "KAS + learned prior + stored rank-64 correction"),
    "Dense": ("P03, P07", "untied d x V head on the same Kronecker input; no output bias"),
    "KAS-U16": ("P03", "KAS + learned prior + stored rank-16 correction (K2 re-implemented)"),
    "KAS-U16-dropU25": ("P06a", "KAS-U16; each step 25% of tokens lose u_v (vocabulary dropout)"),
    "KAS-G": ("P03", "KAS + prior + correction generated from bytes by a shared ByteCNN"),
    "KAS-G-shuf": ("P03", "KAS-G, generator reads a fixed other token's bytes"),
    "KAS-G-occ": ("P06", "KAS-G with the reference report's occupancy-MLP generator"),
    "KAS-U16 (c_zero)": ("P03 (ablation)", "KAS-U16 with the broken C = 0 initialisation"),
    "KAS-P": ("P03", "KAS + fixed log-unigram prior; no per-token trainable term"),
    "KAS-U16-drop25": ("P06", "KAS-U16; each step 25% of tokens lose beta_v and u_v"),
    "KAS-0": ("P03", "Kronecker additive head (KAS) alone"),
}
REFS = [("floor_gpu_s1_dev", "seed 1 / dev"), ("floor_gpu_s2_dev", "seed 2 / dev"),
        ("floor_gpu_s1_test", "seed 1 / test"), ("floor_gpu_s2_test", "seed 2 / test")]
KAS_FIXED = "kas | bias = log-unigram prior (fixed; KAS-P family)"
FAMILIES = [  # (fit tag, label, extra features, bias treatment)
    (KAS_FIXED, "KAS (exact codec family)", "0", "fixed log-unigram (= KAS-P's prior)"),
    ("kas | bias = dense-prior's learned beta (fixed)", "KAS (exact codec family)", "0", "teacher's learned bias, fixed"),
    ("kas | bias = free per-token, init beta (looser lower bound)", "KAS (exact codec family)", "0",
     "free per-token bias, refit per 512-context chunk (relaxation)"),
    ("kas+suffix | bias = beta (fixed)", "KAS + end-anchored occupancy cells", "1,773", "teacher's learned bias, fixed"),
    ("kas+private4000 | bias = beta (fixed)", "KAS + private cell for 4,000 most frequent tokens", "4,000",
     "teacher's learned bias, fixed"),
    ("kas+trigram_free | bias = beta (fixed)", "KAS + 8,192 hashed byte-trigram cells", "8,192",
     "teacher's learned bias, fixed"),
    ("kas+hash8192 | bias = beta (fixed)", "KAS + 8,192 signed hashed identity cells (8 per token)", "8,192",
     "teacher's learned bias, fixed"),
]


def j(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def f(x, d=4, sign=False):
    return (f"{x:+.{d}f}" if sign else f"{x:.{d}f}").replace("-", "−")


def ci(c, d=4):
    return f"[{f(c[0], d, True)}, {f(c[1], d, True)}]"


def table(header, rows, align):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---:" if a == "r" else "---" for a in align) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def build():
    V = {}  # flat numeric values, saved so scripts/check_readme_numbers.py can trace every quoted number
    T = {}  # name -> markdown block
    joint = j(ROOT / "results/phase4/analysis_m3_p2_p3_p4_test.json")
    pv = j(ROOT / "results/phase4/prior_verdict.json")
    p1 = j(ROOT / "results/final/analysis_m3_test.json")
    p2 = j(ROOT / "results/phase2/phase2_verdicts.json")
    p3 = j(ROOT / "results/phase3/confirm_verdict.json")
    ea = j(EV / "evidence_audit.json")
    fc = j(EV / "fit_checks.json")
    st = j(EV / "floor_static.json")
    cv = j(EV / "floor_convergence.json")
    rw = j(EV / "floor_rows.json")

    # ---- all trained arms -------------------------------------------------------------------------------
    arms = joint["arms"]
    ref = arms["Dense-prior"]["bpb_mean"]
    rows = []
    for name in sorted(arms, key=lambda a: arms[a]["bpb_mean"]):
        a = arms[name]
        V[f"arm/{name}/bpb_mean"] = a["bpb_mean"]
        V[f"arm/{name}/minus_dense_prior"] = a["bpb_mean"] - ref
        where, what = ARM_META[name]
        rows.append([f"**{name}**" if name in ("Dense-prior", "KAS-U64", "Dense", "KAS-P") else name, where, what,
                     len(a["runs"]), f(a["bpb_mean"]), "—" if name == "Dense-prior" else f(a["bpb_mean"] - ref, 4, True),
                     f"{a['head_params']:,}", f"{a['head_V_dependent']:,}"])
    T["arms"] = table(["Arm", "Added in", "What differs", "Seeds", "Test BPB", "Δ vs Dense-prior", "Head params",
                       "V-dependent head params"], rows, "lllrrrrr")

    # ---- pre-registered questions and verdicts -----------------------------------------------------------
    H1, H2, H3, H4, K2 = (p1[k] for k in ("H1", "H2", "H3", "H4", "K2_pattern"))
    q1, h7, h8 = p2["Q1_dropout_cost"], p2["H7_report_generator_form"], p2["H8_rank64"]
    h5 = p2["H5_dropout_minting"]
    fresh = p3["fresh_pairs_seeds34"]
    R = pv["R_u64_minus_prior"]
    d1 = q1["bpb(KAS-U16-drop25) - bpb(KAS-U16)"]
    d2 = q1["bpb(KAS-U16-dropU25) - bpb(KAS-U16)"]
    d7 = h7["d7 = bpb(KAS-P) - bpb(KAS-G-occ)"]
    d8 = h8["bpb(KAS-U64) - bpb(Dense)"]
    rows = [
        ["H1", "P03", "Does a correction generated from bytes recover ≥ 0.5 of a stored rank-16 correction's gain?",
         f"ρ = {f(H1['rho']['point'], 3)} {ci(H1['rho']['ci95'], 3)}", H1["verdict"]],
        ["H2", "P03", "Do generated parameters mint held-out merges with lower regret than stored ones?",
         f"Δregret {f(H2['delta_regret_bits']['point'], 3, True)} {ci(H2['delta_regret_bits']['ci95'], 3)} bits/site",
         H2["verdict"] + " (relative cost only)"],
        ["H3", "P03", "Does the byte head's deficit change sign across frequency buckets?",
         f"Spearman ρ = {f(H3['spearman_rho_bucket_order_vs_d'], 3)}", H3["verdict"]],
        ["H4", "P03", "Does the minting advantage need real spelling (selected rule)?",
         f"{f(H4['delta_spell_kasg_minus_shuf']['point'], 3, True)} {ci(H4['delta_spell_kasg_minus_shuf']['ci95'], 3)} bits/site",
         H4["verdict"]],
        ["K2", "P03", "Does Dense's lead over KAS-U16 grow from 12M to 100M tokens (dev)?",
         f"{f(K2['gap_first']['point'], 4, True)} → {f(K2['gap_last']['point'], 4, True)} BPB",
         "REPRODUCED" if K2["reproduced"] else "NOT REPRODUCED"],
        ["Q1/H5", "P06", "Vocabulary dropout of prior and correction: quality cost; minting",
         f"{f(d1['point'], 4, True)} {ci(d1['ci95'])} BPB", f"costly; H5 {h5['KAS-U16-drop25']['verdict']}"],
        ["Q1/H5′", "P06a", "Vocabulary dropout of the correction only (post-hoc reading)",
         f"{f(d2['point'], 4, True)} {ci(d2['ci95'])} BPB", f"H5′ {h5['KAS-U16-dropU25']['verdict']}"],
        ["H7", "P06", "Does the reference report's negative result for byte-computed coordinates replicate?",
         f"KAS-P − KAS-G-occ {f(d7['point'], 4, True)} {ci(d7['ci95'])} BPB (report: {f(h7['report_value'], 2)})",
         h7["verdict"]],
        ["H8", "P06/06a", "Does a rank-64 stored correction train (and how does it compare with bias-free Dense)?",
         f"KAS-U64 − Dense {f(d8['point'], 4, True)} {ci(d8['ci95'])} BPB (2 seeds)", h8["verdict"]],
        ["P07", "P07", "Does KAS-U64 < bias-free Dense hold on fresh seeds 3 and 4?",
         f"{f(fresh['delta_point'], 4, True)} BPB; Welch {ci(fresh['welch']['ci95'])}", p3["verdict"]["word"]],
        ["P08", "P08", "Does KAS-U64 still beat Dense once Dense gets the same log-unigram bias?",
         f"KAS-U64 − Dense-prior {f(R['point'], 4, True)} {ci(R['bootstrap']['ci95'])} BPB",
         "band D: prior-matched Dense beats KAS-U64"],
    ]
    T["verdicts"] = table(["ID", "Protocol", "Question", "Key number (test, 95% interval)", "Pre-registered verdict"],
                          rows, "lllll")

    # ---- protocol 08 --------------------------------------------------------------------------------------
    rows = []
    for s, r in pv["per_seed"].items():
        rows.append([s, f(r["Dense"]), f(r["Dense-prior"]), f(r["KAS-U64"]), f(r["P_dense_minus_prior"], 4, True),
                     f(r["R_u64_minus_prior"], 4, True)])
    m = pv["means"]
    P = pv["P_dense_minus_prior"]
    rows.append(["**mean**", f"**{f(m['Dense'])}**", f"**{f(m['Dense-prior'])}**", f"**{f(m['KAS-U64'])}**",
                 f"**{f(P['point'], 4, True)}**", f"**{f(R['point'], 4, True)}**"])
    T["protocol08"] = table(["Seed", "Dense", "Dense-prior", "KAS-U64", "Dense − Dense-prior", "KAS-U64 − Dense-prior"],
                            rows, "lrrrrr") + (
        f"\n\nDense − Dense-prior: paired t (df 3) {ci(P['paired_t']['ci95'])}. KAS-U64 − Dense-prior: two-level "
        f"bootstrap {ci(R['bootstrap']['ci95'])}, Welch (df ≈ {R['welch']['df']:.1f}) {ci(R['welch']['ci95'])}. "
        f"KAS-U64 is lower in {pv['pairs_u64_below_prior']} of {pv['n_pairs']} (seed, seed) pairs. Share of the "
        f"earlier KAS-U64-vs-Dense gap removed by giving Dense the bias: {pv['share_explained']['point']:.2f} "
        f"{ci(pv['share_explained']['ci95'], 2).replace('+', '')}. Source: `results/phase4/prior_verdict.json`.")
    V["p08/share_explained"] = pv["share_explained"]["point"]

    # ---- teacher-projection table (non-EOS outer average) --------------------------
    rows = []
    for key, label in REFS:
        mx = ea["matrix"][key]
        fit = mx["fits"][KAS_FIXED]
        n = mx["valid"] - mx["eos_targets_in_saved_valid"]
        vals = [fit["canonical_non_eos_floor"], fit["canonical_non_eos_mimic_gap"], mx["canonical_non_eos_kasp_gap"],
                fit["canonical_non_eos_kasp_minus_mimic"]]
        rows.append([label, f"{n:,}"] + [f(v, 6) for v in vals])
        for nm, v in zip(("fitted_kl", "mimic_gap", "kasp_gap", "kasp_minus_mimic"), vals):
            V[f"projection/{key}/{nm}"] = v
        V[f"projection/{key}/n_targets"] = n
        V[f"projection/{key}/eos_in_original_grid"] = mx["eos_targets_in_saved_valid"]
        V[f"projection/{key}/window_cluster_se"] = fit["window_cluster_se_kasp_minus_mimic"]
    T["projection"] = table(["Teacher / split", "Non-EOS targets", "Fitted teacher KL (fixed log-unigram bias)",
                             "Mimic target-NLL gap", "Realised KAS-P gap", "KAS-P minus mimic"], rows, "lrrrrr")

    # ---- feature-family achieved objectives (range over the four teacher/split grids) --------------------
    rows = []
    for tag, label, extra, bias in FAMILIES:
        kl = [ea["matrix"][k]["fits"][tag]["canonical_non_eos_floor"] for k, _ in REFS]
        mg = [ea["matrix"][k]["fits"][tag]["canonical_non_eos_mimic_gap"] for k, _ in REFS]
        V[f"family/{tag}/kl_min"], V[f"family/{tag}/kl_max"] = min(kl), max(kl)
        V[f"family/{tag}/mimic_min"], V[f"family/{tag}/mimic_max"] = min(mg), max(mg)
        rows.append([label, extra, bias, f"{f(min(kl), 3)}–{f(max(kl), 3)}", f"{f(min(mg), 3)}–{f(max(mg), 3)}"])
    T["families"] = table(["Fixed feature family", "Extra features", "Bias in the fit",
                           "Achieved teacher KL, nats (4 grids)", "Mimic target-NLL gap, nats (4 grids)"],
                          rows, "lrlrr")

    # ---- output-head parameter / optimiser-state accounting ---------------------------------------------
    rows = []
    for name, a in ea["output_head_accounting"].items():
        label = "KAS + 8,192 hashed trigrams (hypothetical, not trained)" if name == "KAS+8192_trigrams" else name
        rows.append([label, f"{a['trainable']:,}", f(a["fp32_adam_moments_bytes"] / MIB, 3),
                     f(a["fp32_weight_grad_adam_bytes"] / MIB, 3)])
        V[f"heads/{name}/adam_mib"] = a["fp32_adam_moments_bytes"] / MIB
        V[f"heads/{name}/all_mib"] = a["fp32_weight_grad_adam_bytes"] / MIB
    T["params"] = table(["Output head (d = 512, V = 57,243)", "Trainable parameters", "FP32 Adam moments, MiB",
                         "FP32 weights + grads + moments, MiB"], rows, "lrrr")

    # ---- convergence evidence ----------------------------------------------------------------------------
    r512 = cv["runs"]
    rows = [["KAS, fixed log-unigram bias; first 512 dev contexts (original fitter)", "LBFGS 200 / 400 / 800",
             " / ".join(f(r512[k]["floor_nats"], 6) for k in ("200", "400", "800"))]]
    for name, label in (("kas_fixed_unigram", "KAS, fixed log-unigram bias"),
                        ("kas_trigram_fixed_teacher_beta", "KAS + 8,192 trigrams, teacher bias")):
        x = fc["fits"][name]
        a0 = float(np.mean(x["original_float32_200_KL"]))
        a1 = float(np.mean(x["float64_stages"][-1]["KL"]))
        red = np.array(x["original_float32_200_KL"]) - np.array(x["float64_stages"][-1]["KL"])
        V[f"conv4/{name}/saved200"], V[f"conv4/{name}/indep800"], V[f"conv4/{name}/reduction"] = a0, a1, a0 - a1
        V[f"conv4/{name}/red_min"], V[f"conv4/{name}/red_max"] = float(red.min()), float(red.max())
        rows.append([f"{label}; 4 preselected dev contexts (independent float64 fitter)",
                     "saved 200 (float32) → independent 800 (float64)",
                     f"{f(a0, 6)} → {f(a1, 6)} (−{f(a0 - a1, 6)}; per context {f(red.min(), 4)}–{f(red.max(), 4)})"])
    T["convergence"] = table(["Fit", "Iterations", "Mean achieved KL, nats"], rows, "lll")

    # ---- model-free targets and row sets -----------------------------------------------------------------
    uni, big = st["unigram_floor"], st["bigram"]
    rows = [["training unigram (context-free)", "1", f(uni["kas"]["floor_nats"], 4) + " (LBFGS); "
             + f(uni["kas_adam1500_check"]["floor_nats"], 4) + " (Adam)", "none"],
            [f"bigram conditionals of the 128 most frequent previous tokens ({100 * big['position_coverage']:.2f}% of "
             "training positions)", "128", f(big["floors"]["kas"]["unigram_bias"]["floor_nats"], 4),
             "fixed log-unigram"],
            ["same", "128", f(big["floors"]["kas"]["free_bias"]["floor_nats"], 4), "free per-token, joint over contexts"]]
    T["static"] = table(["Target distribution", "Contexts", "Achieved KL of the KAS family, nats", "Bias"], rows, "lrrl")
    rr = rw["rows"]
    rows = [["Dense-prior's own rows and bias (sanity: exact optimum is 0)", f(rr["dense_prior_W (sanity)"]["floor_nats_per_target"], 7)],
            ["bias-free Dense rows, teacher's bias supplied", f(rr["dense_nobias_W (beta fixed)"]["floor_nats_per_target"], 6)],
            ["trained KAS-U64 rows, own bias", f(rr["kasu64_rows (own bias)"]["floor_nats_per_target"], 6)],
            ["trained KAS-U16 rows, own bias", f(rr["kasu16_rows (own bias)"]["floor_nats_per_target"], 6)],
            ["tied input projection P·κ_v (a KAS subfamily)", f(rr["read_to_write_k0"]["floor_nats_per_target"], 6)]]
    T["rowsets"] = table(["Fixed rows (d = 512, free context vector)", "Achieved teacher KL, nats (1,024 dev contexts)"],
                         rows, "lr")

    # ---- small derived values quoted in prose ------------------------------------------------------------
    ce = ea["codec_equivalence"]
    V.update({"codec/rank": ce["numerical_rank_at_original_tolerance"], "codec/occupied": ce["occupied_cells"],
              "codec/features": ce["features"], "codec/fwd_err": ce["forward_max_abs_error_float32_feature_weights"],
              "codec/rev_err": ce["reverse_max_abs_error_float32_feature_weights"],
              "trigram/occupied": ea["trigram"]["occupied_buckets"], "trigram/occurrences": ea["trigram"]["occurrences"],
              "trigram/extra_trainable": ea["output_head_accounting"]["KAS+8192_trigrams"]["trainable"]
              - ea["output_head_accounting"]["KAS-P"]["trainable"],
              "trigram/over_u16": ea["output_head_accounting"]["KAS+8192_trigrams"]["trainable"]
              / ea["output_head_accounting"]["KAS-U16"]["trainable"] - 1,
              "u64/head_ratio_vs_dense_prior": arms["KAS-U64"]["head_params"] / arms["Dense-prior"]["head_params"],
              # materialised FP32 V x d row matrix (d = 512): here, and at V = 1,000,000
              "rows_fp32_mib_here": ce["V"] * 512 * 4 / MIB, "rows_fp32_gib_1M": 1_000_000 * 512 * 4 / 2 ** 30})
    idg = j(ROOT / "results/phase4/independent_diagnostics.json")  # independent recomputation of protocol 08
    tail = idg["tail_below_1000"]["contribution_bpb"]
    V.update({"tail/below1000_bpb": tail, "tail/rest_bpb": idg["U64_minus_Dense_prior"] - tail,
              "tail/below1000_fraction": idg["tail_below_1000"]["fraction"],
              "tail/10_100_nats": [b for b in idg["frequency_buckets"] if b["range"] == "[10,100)"][0]["u64_minus_prior_nats"]})
    return V, T, pv, ea


def figures(pv, ea):
    # fig9: protocol 08, every run as a point, seeds joined
    fig, ax = plt.subplots(figsize=(4.8, 2.9))
    order = [("Dense", "Dense\n(no output bias)", "#000000", "s"),
             ("KAS-U64", "KAS-U64\n(learned prior + rank 64)", "#882255", "D"),
             ("Dense-prior", "Dense-prior\n(Dense + the same prior)", "#009E73", "o")]
    seeds = sorted(pv["per_seed"], key=int)
    for s in seeds:
        ax.plot(range(3), [pv["per_seed"][s][a] for a, *_ in order], color="#bbbbbb", lw=0.8, zorder=1)
    for i, (a, lab, c, mk) in enumerate(order):
        ys = [pv["per_seed"][s][a] for s in seeds]
        ax.scatter([i] * len(ys), ys, color=c, marker=mk, s=40, zorder=3, edgecolor="white", linewidth=0.6)
        ax.hlines(pv["means"][a], i - 0.22, i + 0.22, color=c, lw=2, zorder=2)
        ax.annotate(f"mean {pv['means'][a]:.4f}", (i + 0.25, pv["means"][a]), fontsize=7.5, va="center", color="#333333")
    ax.set_xticks(range(3), [o[1] for o in order], fontsize=7.5)
    ax.set_xlim(-0.4, 2.9)
    ax.set_ylabel("test bits per byte (lower is better)")
    ax.set_title("Protocol 08: matching the output prior reverses the ranking")
    ax.grid(axis="x", visible=False)
    mf.save(fig, "fig9_protocol08_prior_matched")

    # fig10: teacher-referenced projection audit (corrected non-EOS values)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.6, 3.0), gridspec_kw={"width_ratios": [1.05, 1.25]})
    qs = [("canonical_non_eos_floor", "fitted teacher KL (KAS family)", "#0072B2"),
          ("canonical_non_eos_mimic_gap", "teacher-fitted mimic: target-NLL gap", "#56B4E9"),
          ("kasp", "trained KAS-P: target-NLL gap", "#E69F00")]
    x = np.arange(len(REFS))
    for q, (key, lab, c) in enumerate(qs):
        vals = [ea["matrix"][r]["canonical_non_eos_kasp_gap"] if key == "kasp" else
                ea["matrix"][r]["fits"][KAS_FIXED][key] for r, _ in REFS]
        a1.bar(x + (q - 1) * 0.27, vals, 0.25, color=c, label=lab, edgecolor="white", linewidth=0.6)
    a1.set_xticks(x, [r[1].replace(" / ", "\n") for r in REFS], fontsize=7.5)
    a1.set_ylabel("nats per target (non-EOS)")
    a1.set_ylim(0, 1.0)
    a1.set_title("Same positions, two teachers, two splits")
    a1.legend(loc="upper center", ncol=1, fontsize=7, bbox_to_anchor=(0.5, -0.2))
    labels, lo, hi = [], [], []
    for tag, lab, extra, bias in FAMILIES:
        kl = [ea["matrix"][k]["fits"][tag]["canonical_non_eos_floor"] for k, _ in REFS]
        short = lab.replace(" (exact codec family)", "").replace("signed hashed identity cells (8 per token)",
                                                                 "hashed identity cells").replace(
            "private cell for 4,000 most frequent tokens", "4,000 private cells")
        tag_bias = bias.split(",")[0].replace(" (= KAS-P's prior)", "").replace("free per-token bias", "free bias per chunk")
        labels.append(f"{short}\n[{tag_bias}]")
        lo.append(min(kl)); hi.append(max(kl))
    y = np.arange(len(labels))[::-1]
    a2.hlines(y, lo, hi, color="#0072B2", lw=3)
    a2.scatter([(l + h) / 2 for l, h in zip(lo, hi)], y, color="#0072B2", s=14, zorder=3)
    for yy, h in zip(y, hi):
        a2.annotate(f"{h:.3f}", (h + 0.015, yy), va="center", fontsize=7, color="#333333")
    a2.set_yticks(y, labels, fontsize=6.5)
    a2.set_xlim(0, 0.9)
    a2.set_xlabel("achieved teacher KL, nats (range over 4 grids)")
    a2.set_title("Fixed feature families (oracle fits, not trained)")
    fig.text(0.01, -0.04, "Achieved objectives of numerical fits: estimates of the infimum, not certified bounds. "
             "Enriched families use the teacher's learned bias. Source: results/expressivity/evidence_audit.json",
             fontsize=6.5, color="#555555")
    fig.tight_layout()
    mf.save(fig, "fig10_teacher_projection_audit")


def inject(text: str, blocks: dict) -> str:
    def rep(m):
        name = m.group(1)
        if name not in blocks:
            raise KeyError(f"unknown table block {name!r}")
        return f"<!-- table:{name} -->\n{blocks[name]}\n<!-- /table:{name} -->"
    return re.sub(r"<!-- table:([\w-]+) -->.*?<!-- /table:\1 -->", rep, text, flags=re.S)


def main() -> int:
    check = "--check" in sys.argv
    V, T, pv, ea = build()
    stale = []
    for doc in DOCS:
        if not doc.exists():
            continue
        old = doc.read_text(encoding="utf-8")
        new = inject(old, T)
        if new != old:
            stale.append(str(doc.relative_to(ROOT)))
            if not check:
                doc.write_text(new, encoding="utf-8", newline="\n")
    if check:
        print("stale generated blocks:", stale or "none")
        return 1 if stale else 0
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary_values.json").write_text(json.dumps(V, indent=1, sort_keys=True), encoding="utf-8",
                                                newline="\n")
    (OUT / "tables.md").write_text("# Generated summary tables\n\nWritten by `scripts/build_summary.py`; "
                                   "do not edit by hand.\n\n" + "\n\n".join(f"## {k}\n\n{v}" for k, v in T.items())
                                   + "\n", encoding="utf-8", newline="\n")
    figures(pv, ea)
    print(f"{len(V)} values, {len(T)} tables; refreshed: {stale or 'nothing (already current)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
