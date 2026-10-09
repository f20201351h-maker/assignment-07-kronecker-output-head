"""Phase 2 (protocol 06) verdicts: H5–H8 and Q1, from the joint Phase 1 + Phase 2 analysis.

    python scripts/final_analysis.py --runs experiments/modal-m3/runs experiments/modal-p2/runs \
        --prefix m3 p2 --split test --out results/phase2
    python scripts/phase2_analysis.py            # -> results/phase2/phase2_verdicts.json

Comparators are the Phase 1 canonical arms (m3). Rules are selected on dev; test is read once.
Bootstraps follow Protocol 03 (windows paired across arms, seeds within arms, B = 2,000).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from kq5.data import TokenData, eval_windows  # noqa: E402
from kq5.stats import LN2, bpb_of, two_level_bootstrap, window_sums  # noqa: E402
from kq5.vocab import WorkingVocab  # noqa: E402
from final_analysis import site_bootstrap  # noqa: E402

B = 2000
RUNS = {"KAS-U16": ("experiments/modal-m3/runs", ["m3-kasu0-s1", "m3-kasu0-s2"]),
        "KAS-G": ("experiments/modal-m3/runs", ["m3-kasg-s1", "m3-kasg-s2"]),
        "KAS-P": ("experiments/modal-m3/runs", ["m3-kasp-s1", "m3-kasp-s2"]),
        "Dense": ("experiments/modal-m3/runs", ["m3-dense-s1", "m3-dense-s2"]),
        "KAS-G-shuf": ("experiments/modal-m3/runs", ["m3-kasgshuf-s1", "m3-kasgshuf-s2"]),
        "KAS-U16-drop25": ("experiments/modal-p2/runs", ["p2-kasu-drop25-s1", "p2-kasu-drop25-s2"]),
        "KAS-U16-dropU25": ("experiments/modal-p2/runs", ["p2-kasu-dropu25-s1", "p2-kasu-dropu25-s2"]),
        "KAS-G-occ": ("experiments/modal-p2/runs", ["p2-kasg-occ-s1", "p2-kasg-occ-s2"]),
        "KAS-U64": ("experiments/modal-p2/runs", ["p2-kasu64-s1", "p2-kasu64-s2"])}
DROP_ARMS = ("KAS-U16-drop25", "KAS-U16-dropU25")       # protocol 06 (prior + correction hidden), amendment 06a (u only)
KASU_RULES = ["floor", "zero", "count", "inherit", "inherit_prior"]        # Protocol 03 H2 candidates
NET_RULES = {"KAS-U16": ["floor", "zero", "count", "inherit", "inherit_prior"],
             "KAS-U16-drop25": ["floor", "zero", "count", "inherit", "inherit_prior"],
             "KAS-U16-dropU25": ["floor", "zero", "count", "inherit", "inherit_prior"],
             "KAS-U64": ["floor", "zero", "count", "inherit", "inherit_prior"],
             "KAS-G": ["count", "inherit", "gen_count", "gen_inherit"],
             "KAS-G-occ": ["count", "inherit", "gen_count", "gen_inherit"],
             "KAS-P": ["count", "inherit"], "Dense": ["mean", "inherit"]}


def run_dirs(arm):
    """Seeds of an arm that have finished (complete at the planned step count, test NLL saved)."""
    root, ids = RUNS[arm]
    out = []
    for r in ids:
        d = ROOT / root / r
        if (d / "summary.json").exists() and (d / "nll_test.npy").exists():
            s = json.loads((d / "summary.json").read_text(encoding="utf-8"))
            rec = json.loads((d / "run_record.json").read_text(encoding="utf-8"))
            if s.get("status") == "complete" and s.get("steps") == rec["total_steps"]:
                out.append(d)
    return out


def present(arm):
    return len(run_dirs(arm)) > 0


def mint(arm, split):
    out = []
    for d in run_dirs(arm):
        z = np.load(d / f"analysis_{split}" / "minting.npz")
        j = json.loads((d / f"analysis_{split}" / "minting.json").read_text(encoding="utf-8"))
        out.append((dict(z), j))
    return out


def mean_regret(arm, split, rule):
    return float(np.mean([j["rules"][rule]["mean_regret_bits"] for _, j in mint(arm, split)]))


def best_rule_by_regret(arm, cands):
    dev = {r: mean_regret(arm, "dev", r) for r in cands}
    return min(dev, key=dev.get), dev


def retok_delta(arm, split, rule):
    vals = []
    for d in run_dirs(arm):
        rt = json.loads((d / f"analysis_{split}" / "retok.json").read_text(encoding="utf-8"))
        if rule in rt and "delta_bpb" in rt[rule]:
            vals.append(rt[rule]["delta_bpb"])
    return vals


def best_rule_by_net(arm):
    dev = {r: float(np.mean(v)) for r in NET_RULES[arm] if (v := retok_delta(arm, "dev", r))}
    return min(dev, key=dev.get), dev


def site_vals(arm, rule, split):
    per = mint(arm, split)
    reg = np.stack([z[f"{rule}__regret"] for z, _ in per]).astype(np.float64)
    leak = [z[f"{rule}__leak_per_window"].astype(np.float64) for z, _ in per]
    return reg, leak, per[0][0]["site_w"]


def paired_regret_leak(a, ra, b, rb, bw, seed):
    """Δregret (a − b) by site bootstrap; Δleak (a − b) in bpb by two-level bootstrap."""
    A, Al, sw = site_vals(a, ra, "test")
    Bv, Bl, sw2 = site_vals(b, rb, "test")
    assert np.array_equal(sw, sw2)
    dreg = site_bootstrap({"a": A, "b": Bv}, sw, lambda v: v["a"] - v["b"], seed=seed)
    dleak = two_level_bootstrap({"a": Al, "b": Bl}, bw, lambda v: v["a"] - v["b"], n_boot=B, seed=seed)
    dleak.pop("draws")
    return dreg, dleak


def main() -> None:
    tok = ROOT / "data" / "tokens"
    data = TokenData.load(tok, verify=False)
    vocab = WorkingVocab.load(tok / "vocab.json")
    tb = vocab.target_bytes()
    win = eval_windows(data.test, 1024)
    nbytes = tb[win[:, 1:]]
    _, bw = window_sums(np.zeros_like(nbytes, dtype=np.float64), nbytes)
    wsum = {}
    for arm in RUNS:
        if present(arm):
            wsum[arm] = [window_sums(np.load(d / "nll_test.npy"), nbytes)[0] for d in run_dirs(arm)]
    bpb = {a: float(np.mean([bpb_of(s, bw) for s in v])) for a, v in wsum.items()}
    out = {"protocol": "06", "arms_present": sorted(wsum), "bpb_test": bpb}

    def diff(a, b, seed=0):
        if a not in wsum or b not in wsum:
            return None
        r = two_level_bootstrap({a: wsum[a], b: wsum[b]}, bw, lambda v: v[a] - v[b], n_boot=B, seed=seed)
        r.pop("draws")
        return r

    # ---------- Q1 / H5 / H6 for each dropout variant (protocol 06: both hidden; 06a: u only) ----------
    out["Q1_dropout_cost"] = {f"bpb({arm}) - bpb(KAS-U16)": diff(arm, "KAS-U16") for arm in DROP_ARMS if arm in wsum}
    out["Q1_dropout_cost"]["report_value_20M_single_run"] = 0.028
    out["H5_dropout_minting"], out["H6_frontier"] = {}, {}
    for k, arm in enumerate(DROP_ARMS):
        if arm not in wsum:
            continue
        rd, dev_d = best_rule_by_regret(arm, KASU_RULES)
        ru, dev_u = best_rule_by_regret("KAS-U16", KASU_RULES)
        dreg, dleak = paired_regret_leak(arm, rd, "KAS-U16", ru, bw, seed=11 + 10 * k)
        if dreg["ci95"][1] < 0 and dleak["ci95"][0] <= 0:
            v5 = "SUPPORTED"
        elif dreg["ci95"][0] >= 0 or (dreg["ci95"][1] < 0 and dleak["ci95"][0] > 0):
            v5 = "REJECTED"
        else:
            v5 = "INCONCLUSIVE"
        zero = paired_regret_leak(arm, "zero", "KAS-U16", "zero", bw, seed=12 + 10 * k)
        nd, ndev = best_rule_by_net(arm)
        nu, _ = best_rule_by_net("KAS-U16")
        out["H5_dropout_minting"][arm] = {
            "rule_drop_by_dev_regret": rd, "rule_kasu_by_dev_regret": ru,
            "dev_regret_candidates": {arm: dev_d, "KAS-U16": dev_u},
            "delta_regret_bits": dreg, "delta_leak_bpb": dleak, "verdict": v5,
            "secondary_zero_rule": {"delta_regret_bits": zero[0], "delta_leak_bpb": zero[1],
                                    "regret_drop": mean_regret(arm, "test", "zero"),
                                    "regret_kasu": mean_regret("KAS-U16", "test", "zero")},
            "net_retok": {"rule_drop_by_net_dev": nd, "rule_kasu_by_net_dev": nu,
                          "test_delta_drop_per_seed": retok_delta(arm, "test", nd),
                          "test_delta_kasu_per_seed": retok_delta("KAS-U16", "test", nu),
                          "dev_candidates_drop": ndev}}

        def frontier(other, seed, arm=arm, rd=rd, nd=nd):
            ro, _ = best_rule_by_net(other)
            dq = diff(arm, other, seed=seed)
            net_d = retok_delta(arm, "test", nd)
            net_o = retok_delta(other, "test", ro)
            rO, _ = (best_rule_by_regret(other, KASU_RULES) if other.startswith("KAS-U")
                     else best_rule_by_regret(other, NET_RULES[other]))
            dreg_o, dleak_o = paired_regret_leak(arm, rd, other, rO, bw, seed=seed)
            better_q = dq["ci95"][1] < 0
            worse_q = dq["ci95"][0] > 0
            net_better = all(x < y for x, y in zip(net_d, net_o)) and np.mean(net_d) < np.mean(net_o)
            net_worse = np.mean(net_d) > np.mean(net_o)
            if better_q and net_better and not dreg_o["ci95"][0] > 0:
                v = "DOMINATES"
            elif better_q and net_worse:
                v = "TRADE-OFF"
            elif worse_q and net_worse:
                v = "DOMINATED"
            else:
                v = "INCONCLUSIVE"
            return {"other": other, "rule_other_by_net_dev": ro, "bpb_diff_drop_minus_other": dq,
                    "net_test_delta_drop_per_seed": net_d, "net_test_delta_other_per_seed": net_o,
                    "delta_regret_bits_own_rules": dreg_o, "delta_leak_bpb_own_rules": dleak_o, "verdict": v}
        out["H6_frontier"][arm] = {"vs_KAS-G": frontier("KAS-G", 21 + 10 * k), "vs_Dense": frontier("Dense", 22 + 10 * k)}

    # ---------- H7: the report's §09 negative result ----------
    if "KAS-G-occ" in wsum:
        d7 = diff("KAS-P", "KAS-G-occ", seed=31)
        v7 = ("REPLICATES" if d7["ci95"][1] < 0 else "DOES NOT REPLICATE" if d7["ci95"][0] > 0
              else "INCONCLUSIVE")
        rho = two_level_bootstrap({"P": wsum["KAS-P"], "U": wsum["KAS-U16"], "G": wsum["KAS-G-occ"]}, bw,
                                  lambda v: (v["P"] - v["G"]) / (v["P"] - v["U"]), n_boot=B, seed=32)
        rho.pop("draws")
        out["H7_report_generator_form"] = {
            "d7 = bpb(KAS-P) - bpb(KAS-G-occ)": d7, "report_value": -0.24, "verdict": v7,
            "rho_occ (H1 scale)": rho, "bpb(KAS-G) - bpb(KAS-G-occ)": diff("KAS-G", "KAS-G-occ", seed=33),
            "bpb(KAS-G-shuf) - bpb(KAS-G-occ)": diff("KAS-G-shuf", "KAS-G-occ", seed=34),
            "bpb(KAS-G-occ) - bpb(KAS-U16)": diff("KAS-G-occ", "KAS-U16", seed=35)}
        try:
            rg, _ = best_rule_by_regret("KAS-G-occ", NET_RULES["KAS-G-occ"] + ["gen_zero"])
            rG, _ = best_rule_by_regret("KAS-G", NET_RULES["KAS-G"] + ["gen_zero"])
            dreg, dleak = paired_regret_leak("KAS-G-occ", rg, "KAS-G", rG, bw, seed=36)
            out["H7_report_generator_form"]["minting_vs_KAS-G"] = {
                "rule_occ": rg, "rule_kasg": rG, "delta_regret_bits": dreg, "delta_leak_bpb": dleak,
                "gen_count_regret_occ": mean_regret("KAS-G-occ", "test", "gen_count"),
                "gen_count_regret_kasg": mean_regret("KAS-G", "test", "gen_count")}
        except (FileNotFoundError, KeyError) as e:  # noqa: BLE001
            out["H7_report_generator_form"]["minting_vs_KAS-G"] = {"error": repr(e)}

    # ---------- H8: rank 64 under u_zero ----------
    if "KAS-U64" in wsum:
        d8 = diff("KAS-U64", "KAS-U16", seed=41)
        dp = diff("KAS-U64", "KAS-P", seed=42)
        if dp["point"] >= 0:
            v8 = "FAILS"
        elif d8["point"] <= 0.03:
            v8 = "TRAINS"
        else:
            v8 = "PARTIAL"
        out["H8_rank64"] = {"bpb(KAS-U64) - bpb(KAS-U16)": d8, "bpb(KAS-U64) - bpb(KAS-P)": dp,
                            "bpb(KAS-U64) - bpb(Dense)": diff("KAS-U64", "Dense", seed=43),
                            "n_seeds": len(wsum["KAS-U64"]),
                            "note": "paired two-level bootstrap (seeds resampled with replacement, no small-sample "
                                    "multiplier); a Welch interval is in readme_derived_phase2.json",
                            "verdict": v8}
        try:
            nr, ndev = best_rule_by_net("KAS-U64")
            out["H8_rank64"]["net_retok"] = {"rule_by_net_dev": nr, "test_delta_per_seed": retok_delta("KAS-U64", "test", nr)}
        except (FileNotFoundError, ValueError) as e:  # noqa: BLE001
            out["H8_rank64"]["net_retok"] = {"error": repr(e)}

    # ---------- quality-vs-minting table for every arm present (rule by net dev Δbpb) ----------
    table = {}
    for arm in NET_RULES:
        if arm not in wsum:
            continue
        try:
            r, dev = best_rule_by_net(arm)
        except ValueError:
            continue
        t = retok_delta(arm, "test", r)
        table[arm] = {"rule_by_net_dev": r, "in_vocab_test_bpb": bpb[arm], "net_minting_test_bpb": float(np.mean(t)),
                      "net_minting_test_bpb_per_seed": t, "dev_candidates": dev}
    dom = [f"{a} dominates {b}" for a, va in table.items() for b, vb in table.items()
           if a != b and va["in_vocab_test_bpb"] < vb["in_vocab_test_bpb"]
           and va["net_minting_test_bpb"] < vb["net_minting_test_bpb"]]
    nondom = [a for a in table if not any(d.endswith(f"dominates {a}") for d in dom)]
    out["quality_vs_minting"] = {"table": table, "dominance": dom, "non_dominated": nondom}

    out_dir = ROOT / "results" / "phase2"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "phase2_verdicts.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    brief = {k: v.get("verdict") for k, v in out.items() if isinstance(v, dict) and "verdict" in v}
    brief["H5"] = {a: v["verdict"] for a, v in out["H5_dropout_minting"].items()}
    brief["H6"] = {a: {o: v["verdict"] for o, v in d.items()} for a, d in out["H6_frontier"].items()}
    print(json.dumps({"bpb": bpb, "verdicts": brief}, indent=1))


if __name__ == "__main__":
    main()
