"""Exploratory sensitivity analyses of the minting results (added after an analysis review; not pre-registered).

1. Absolute minting cost: -log2 p'(m) at each site (= regret + the arm's own two-token cost), per arm.
2. H2/H4 under generating rules (KAS-G gen_count vs KAS-U16 count; KAS-G vs KAS-G-shuf under gen_count).
3. Rule chosen by net dev bpb on re-tokenised text (includes leak) instead of regret.
4. Quality-vs-minting table: in-vocabulary test bpb against net minting cost (best net rule on dev).
5. Generator compute: per-step overhead measured in R1b at V = 57,243, scaled linearly to V = 1,048,576.

    python scripts/sensitivity_minting.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from kq5.stats import cluster_bootstrap_mean  # noqa: E402

RUNS = ROOT / "experiments" / "modal-m3" / "runs"
LN2 = math.log(2)
SEEDS = {"KAS-U16": ["m3-kasu0-s1", "m3-kasu0-s2"], "KAS-G": ["m3-kasg-s1", "m3-kasg-s2"],
         "KAS-G-shuf": ["m3-kasgshuf-s1", "m3-kasgshuf-s2"], "KAS-P": ["m3-kasp-s1", "m3-kasp-s2"],
         "Dense": ["m3-dense-s1", "m3-dense-s2"], "KAS-0": ["m3-kas0-s1", "m3-kas0-s2"],
         "KAS-U16 (c_zero)": ["m3-kasu-s1", "m3-kasu-s2"]}


def site_arrays(run, split, rule):
    z = np.load(RUNS / run / f"analysis_{split}" / "minting.npz")
    nll = np.load(RUNS / run / f"nll_{split}.npy")
    w, j = z["site_w"], z["site_j"]
    two = (nll[w, j].astype(np.float64) + nll[w, j + 1]) / LN2
    reg = z[f"{rule}__regret"].astype(np.float64)
    return reg, reg + two, two, w


def seed_mean(arm, split, rule):
    regs, abss, twos = [], [], []
    for r in SEEDS[arm]:
        reg, ab, two, w = site_arrays(r, split, rule)
        regs.append(reg); abss.append(ab); twos.append(two)
    return np.mean(regs, 0), np.mean(abss, 0), np.mean(twos, 0), w


def retok_delta(arm, split, rule):
    vals = []
    for r in SEEDS[arm]:
        rt = json.loads((RUNS / r / f"analysis_{split}" / "retok.json").read_text(encoding="utf-8"))
        if rule in rt and "delta_bpb" in rt[rule]:
            vals.append(rt[rule]["delta_bpb"])
    return float(np.mean(vals)) if vals else None


def main() -> None:
    out = {"note": "exploratory, after an analysis review; not pre-registered"}
    # 1. absolute cost under each arm's dev-selected (pre-registered) rule
    sel = {"KAS-U16": "inherit", "KAS-G": "inherit", "KAS-G-shuf": "inherit"}
    absolute = {}
    for arm, rule in sel.items():
        reg, ab, two, w = seed_mean(arm, "test", rule)
        absolute[arm] = {"rule": rule, "mean_regret_bits": float(reg.mean()),
                         "mean_minted_token_bits": float(ab.mean()), "mean_two_token_bits": float(two.mean())}
    _, abU, _, w = seed_mean("KAS-U16", "test", "inherit")
    _, abG, _, _ = seed_mean("KAS-G", "test", "inherit")
    out["absolute_minted_bits"] = absolute
    out["absolute_KASG_minus_KASU16_bits"] = cluster_bootstrap_mean(abG - abU, w)
    # 2. generating rules
    regU_c, _, _, w = seed_mean("KAS-U16", "test", "count")
    regG_g, abG_g, _, _ = seed_mean("KAS-G", "test", "gen_count")
    regS_g, _, _, _ = seed_mean("KAS-G-shuf", "test", "gen_count")
    out["H2_gen_count_vs_kasu_count"] = cluster_bootstrap_mean(regG_g - regU_c, w)
    out["H4_gen_count_kasg_minus_shuf"] = cluster_bootstrap_mean(regG_g - regS_g, w)
    # 3. rule chosen by net dev bpb (re-tokenised, includes leak), confirmed on test
    rules = {"KAS-U16": ["floor", "zero", "count", "inherit", "inherit_prior"],
             "KAS-G": ["count", "inherit", "gen_count", "gen_inherit"],
             "KAS-G-shuf": ["count", "inherit", "gen_count", "gen_inherit"],
             "KAS-P": ["count", "inherit"], "Dense": ["mean", "inherit"], "KAS-0": ["bytes"],
             "KAS-U16 (c_zero)": ["count", "inherit", "inherit_prior"]}
    table = {}
    A = json.loads((ROOT / "results/final/analysis_m3_test.json").read_text(encoding="utf-8"))
    for arm, rs in rules.items():
        dev = {r: retok_delta(arm, "dev", r) for r in rs}
        dev = {r: v for r, v in dev.items() if v is not None}
        if not dev:
            continue
        best = min(dev, key=dev.get)
        table[arm] = {"rule_by_net_dev": best, "net_minting_test_bpb": retok_delta(arm, "test", best),
                      "in_vocab_test_bpb": A["arms"][arm]["bpb_mean"]}
    out["quality_vs_minting"] = table
    # dominance: is any arm better on both axes than another?
    dom = []
    for a, va in table.items():
        for b, vb in table.items():
            if a != b and va["in_vocab_test_bpb"] < vb["in_vocab_test_bpb"] and \
                    va["net_minting_test_bpb"] < vb["net_minting_test_bpb"]:
                dom.append(f"{a} dominates {b}")
    out["dominance"] = dom
    # 5. generator compute from R1b (T4): per-step seconds at 65,536 tokens/step
    def tps(run):
        lines = [json.loads(l) for l in open(ROOT / "experiments/kq5-r1b-kasg/runs" / run / "metrics.jsonl")]
        v = sorted(l["tok_per_s"] for l in lines if l.get("tok_per_s") and l["step"] > 10)
        return v[len(v) // 2]
    p, g = tps("r1b-tput-8L512-kasp"), tps("r1b-tput-8L512-kasg-v3")
    over = 65536 / g - 65536 / p
    out["generator_cost_T4"] = {"kasp_tok_per_s": p, "kasg_v3_tok_per_s": g, "overhead_s_per_step_V57243": over,
                                "overhead_fraction_of_kasg_step": over / (65536 / g),
                                "overhead_s_per_step_scaled_to_V1048576": over * 1048576 / 57243}
    (ROOT / "results/final/minting_sensitivity.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
