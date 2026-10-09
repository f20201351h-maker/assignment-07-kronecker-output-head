"""Verdicts and tables from downloaded runs (no GPU). Implements Protocol 03.

    python scripts/final_analysis.py --runs experiments/kq5-r3-canonical-x/runs experiments/kq5-r3-canonical-g/runs \
        --prefix r3 --split test --out results/final
For the pilot: --prefix r2 --split dev (exploratory; no verdicts are claimed from it).
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kq5.data import TokenData, eval_windows, sha256_array  # noqa: E402
from kq5.stats import (LN2, bpb_of, bucket_nll, frequency_buckets, two_level_bootstrap,  # noqa: E402
                       verdict_interval, window_sums)
from kq5.vocab import WorkingVocab  # noqa: E402

PROBLEMS: list[str] = []
SEEN: set[str] = set()
ARM_OF = {"dense": "Dense", "kas0": "KAS-0", "kasp": "KAS-P", "kasu": "KAS-U16", "kasg": "KAS-G",
          "kasg_shuf": "KAS-G-shuf"}
B = 2000


def arm_name(head: str, cfg: dict) -> str:
    """Arm label from a run's config. Phase 2 (protocol 06) arms are the canonical kasu / kasg
    configurations with one change each, named by that change."""
    arm = ARM_OF[head]
    gen_kind = (cfg.get("generator") or {}).get("kind", "bytecnn")
    if head == "dense" and cfg.get("dense_bias", False):
        return "Dense-prior"            # protocol 08: Dense + trainable log-unigram bias (prior-matched Dense)
    if head == "kasu" and cfg.get("vocab_dropout", 0) > 0:
        tgt = "U" if cfg.get("vocab_dropout_target", "both") == "u" else ""
        return f"KAS-U{cfg.get('rank', 16)}-drop{tgt}{int(round(100 * cfg['vocab_dropout']))}"
    if head == "kasu" and cfg.get("rank", 16) != 16:
        return f"KAS-U{cfg['rank']}"
    if head in ("kasg", "kasg_shuf") and gen_kind == "occmlp":
        return arm + "-occ"
    return arm


def collect(run_dirs: list[str], prefixes: list[str]) -> dict:
    """arm -> list of run dicts (one per seed), only runs whose id starts with one of the prefixes."""
    arms: dict[str, list] = {}
    for root in run_dirs:
      for prefix in prefixes:
        for d in sorted(glob.glob(os.path.join(root, f"{prefix}-*"))):
            rec_p = os.path.join(d, "run_record.json")
            if not os.path.exists(rec_p) or not os.path.exists(os.path.join(d, "summary.json")):
                continue
            rec = json.load(open(rec_p, encoding="utf-8"))
            summ = json.load(open(os.path.join(d, "summary.json"), encoding="utf-8"))
            head = rec["config"]["head"]
            rid = rec["config"]["run_id"]
            # review fix: only complete runs at the planned step count count as seeds
            if summ.get("status") != "complete" or summ.get("steps") != rec["total_steps"]:
                PROBLEMS.append(f"{rid}: status={summ.get('status')} steps={summ.get('steps')} "
                                f"planned={rec['total_steps']} -> excluded")
                continue
            if rid in SEEN:
                PROBLEMS.append(f"{rid}: duplicate run id under {d} -> excluded")
                continue
            SEEN.add(rid)
            arm = arm_name(head, rec["config"])
            # canonical corrected arms use the u_zero initialisation (protocol 03b); kernel X's
            # c_zero KAS-U16 runs are a separate initialisation-ablation arm
            if prefix in ("r3", "m3", "p2", "p3") and head in ("kasu", "kasg", "kasg_shuf") and                     rec["config"].get("corr_init", "c_zero") != "u_zero":
                arm = f"{arm} (c_zero)"
            m = re.search(r"-(v\d)-", rid)
            if m and prefix == "r2":
                arm = f"{arm}-{m.group(1)}"
            arms.setdefault(arm, []).append({"dir": d, "id": rid, "record": rec, "summary": summ})
    return arms


def site_bootstrap(per_arm_site_vals: dict[str, np.ndarray], clusters: np.ndarray, stat, seed: int = 0,
                   n_boot: int = B) -> dict:
    """per_arm_site_vals: arm -> (n_seeds, n_sites). Windows (clusters) resampled jointly,
    seeds resampled within arm."""
    rng = np.random.default_rng(seed)
    uc, inv = np.unique(clusters, return_inverse=True)
    sums = {k: np.stack([np.bincount(inv, weights=v[s], minlength=len(uc)) for s in range(v.shape[0])])
            for k, v in per_arm_site_vals.items()}
    cnt = np.bincount(inv, minlength=len(uc)).astype(np.float64)
    point = stat({k: float(v.mean()) for k, v in per_arm_site_vals.items()})
    draws = np.empty(n_boot)
    for i in range(n_boot):
        w = rng.integers(0, len(uc), len(uc))
        c = cnt[w].sum()
        vals = {}
        for k, s in sums.items():
            pick = rng.integers(0, s.shape[0], s.shape[0])
            vals[k] = float(s[pick][:, w].sum(1).mean() / c)
        draws[i] = stat(vals)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"point": point, "ci95": [float(lo), float(hi)], "n_sites": int(len(clusters)),
            "n_windows": int(len(uc))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--prefix", nargs="+", default=["r3"],
                    help="run-id prefixes analysed together (e.g. m3 p2: Phase 1 canonical + Phase 2 arms)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--select-split", default="dev")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = ROOT / "data" / "tokens"
    data = TokenData.load(tok, verify=False)
    vocab = WorkingVocab.load(tok / "vocab.json")
    unigram = np.load(tok / "unigram_train.npy")
    tb = vocab.target_bytes()
    arms = collect(args.runs, args.prefix)
    split = args.split
    stream = {"dev": data.dev, "test": data.test}[split]
    win = eval_windows(stream, 1024)
    targets = win[:, 1:]
    nbytes = tb[targets]
    tag = "_".join(args.prefix)
    res = {"split": split, "prefix": tag, "arms": {}, "fairness": {}}

    # ---------- fairness ----------
    orders = {r["record"]["train_order_sha256"] for rs in arms.values() for r in rs}
    steps = {r["record"]["total_steps"] for rs in arms.values() for r in rs}
    wsha = {r["summary"]["final"][split]["windows_sha256"] for rs in arms.values() for r in rs
            if split in r["summary"]["final"]}
    manifests = {json.dumps(r["record"]["data_manifest_files"], sort_keys=True) for rs in arms.values() for r in rs}
    seeds = {arm: sorted(r["record"]["config"]["seed"] for r in rs) for arm, rs in arms.items()}
    for arm, rs in arms.items():
        for r in rs:
            for sp in {split, args.select_split}:
                if not os.path.exists(os.path.join(r["dir"], f"analysis_{sp}", "minting.npz")):
                    PROBLEMS.append(f"{r['id']}: no {sp} minting analysis")
    res["fairness"] = {"distinct_train_orders": len(orders), "distinct_total_steps": sorted(steps),
                       "distinct_data_manifests": len(manifests), "seeds_per_arm": seeds,
                       "distinct_eval_windows": len(wsha), "local_windows_sha256": sha256_array(win),
                       "eval_windows_match_local": wsha == {sha256_array(win)},
                       "skipped_fp16_steps": {r["id"]: r["summary"].get("skipped_steps") for rs in arms.values() for r in rs},
                       "problems": PROBLEMS}

    # ---------- per-arm bpb ----------
    wsum = {}
    for arm, rs in sorted(arms.items()):
        sums, bpbs = [], []
        for r in rs:
            p = os.path.join(r["dir"], f"nll_{split}.npy")
            if not os.path.exists(p):
                continue
            s, b = window_sums(np.load(p), nbytes)
            sums.append(s); bpbs.append(bpb_of(s, b))
        if not sums:
            continue
        wsum[arm] = sums
        params = rs[0]["record"]["params"]
        res["arms"][arm] = {"runs": [r["id"] for r in rs], "bpb_per_seed": bpbs, "bpb_mean": float(np.mean(bpbs)),
                            "seed_spread": float(np.max(bpbs) - np.min(bpbs)) if len(bpbs) > 1 else None,
                            "params_total": params["total"], "head_params": params["head"],
                            "head_V_dependent": params["head_V_dependent"]}
    _, bw = window_sums(np.zeros_like(nbytes, dtype=np.float64), nbytes)
    spreads = [a["seed_spread"] for a in res["arms"].values() if a["seed_spread"] is not None]
    res["max_seed_spread"] = max(spreads) if spreads else None

    def pairdiff(a, b):
        if a in wsum and b in wsum:
            r = two_level_bootstrap({a: wsum[a], b: wsum[b]}, bw, lambda v: v[a] - v[b], n_boot=B)
            r.pop("draws")
            return r
        return None

    res["bpb_differences"] = {f"{a} - {b}": pairdiff(a, b) for a, b in
                              [("KAS-0", "KAS-P"), ("KAS-P", "KAS-U16"), ("KAS-U16", "Dense"), ("KAS-P", "Dense"),
                               ("KAS-G", "KAS-U16"), ("KAS-P", "KAS-G"), ("KAS-G-shuf", "KAS-G"),
                               ("KAS-P", "KAS-G-shuf"),
                               # Phase 2 (protocol 06) pairs; None unless both arms are present
                               ("KAS-U16-drop25", "KAS-U16"), ("KAS-U16-drop25", "KAS-G"),
                               ("KAS-U16-drop25", "Dense"), ("KAS-P", "KAS-U16-drop25"),
                               ("KAS-P", "KAS-G-occ"), ("KAS-G", "KAS-G-occ"), ("KAS-G-occ", "KAS-U16"),
                               ("KAS-G-shuf", "KAS-G-occ"), ("KAS-U64", "KAS-U16"), ("KAS-U64", "Dense"),
                               # amendment 06a
                               ("KAS-U16-dropU25", "KAS-U16"), ("KAS-U16-dropU25", "KAS-G"),
                               ("KAS-U16-dropU25", "Dense"), ("KAS-U16-drop25", "KAS-U16-dropU25"),
                               # protocol 08: prior-matched Dense
                               ("Dense-prior", "Dense"), ("KAS-U64", "Dense-prior")]}

    # ---------- H1 ----------
    gen_arm = "KAS-G" if "KAS-G" in wsum else next((k for k in wsum if k.startswith("KAS-G-v")), None)
    if {"KAS-P", "KAS-U16"} <= set(wsum) and gen_arm:
        den = two_level_bootstrap({"P": wsum["KAS-P"], "U": wsum["KAS-U16"]}, bw, lambda v: v["P"] - v["U"], n_boot=B)
        den.pop("draws")
        rho = two_level_bootstrap({"P": wsum["KAS-P"], "U": wsum["KAS-U16"], "G": wsum[gen_arm]}, bw,
                                  lambda v: (v["P"] - v["G"]) / (v["P"] - v["U"]), n_boot=B)
        rho.pop("draws")
        defined = den["ci95"][0] > 0
        band = None
        if defined:
            band = "success" if rho["point"] >= 0.5 else ("partial" if rho["point"] >= 0.2 else "informative failure")
        res["H1"] = {"note": "precondition implemented as CI(P - U16) entirely > 0 (protocol text: 'excludes 0'); "
                             "a negative denominator makes the ratio uninterpretable",
                     "generator_arm": gen_arm, "denominator_P_minus_U": den, "rho": rho,
                     "verdict": verdict_interval(rho["ci95"], 0.5) if defined else "UNDEFINED",
                     "outcome_band": band}

    # ---------- H3: frequency buckets ----------
    bidx, labels = frequency_buckets(unigram)
    nb_k = len(labels)
    per_arm_b = {}
    for arm in wsum:
        per_seed = []
        for r in arms[arm]:
            p = os.path.join(r["dir"], f"nll_{split}.npy")
            if os.path.exists(p):
                per_seed.append(bucket_nll(np.load(p), targets, nbytes, bidx, nb_k))
        per_arm_b[arm] = per_seed
    h3 = {}
    if "Dense" in per_arm_b:
        rng = np.random.default_rng(1)
        counts = per_arm_b["Dense"][0]["count"]                 # (buckets, windows), same for every arm
        for kas in ("KAS-U16", "KAS-P", "KAS-G", "KAS-0", "KAS-G-shuf", "KAS-U16-drop25", "KAS-U16-dropU25",
                    "KAS-G-occ", "KAS-U64", "Dense-prior"):
            if kas not in per_arm_b:
                continue
            rows = []
            dn = np.stack([s["nll"] for s in per_arm_b["Dense"]])      # (seeds, buckets, windows)
            kn = np.stack([s["nll"] for s in per_arm_b[kas]])
            for k in range(nb_k):
                n_t = counts[k].sum()
                if n_t == 0:
                    rows.append({"bucket": labels[k], "n_targets": 0})
                    continue
                point = (dn[:, k].sum(1).mean() - kn[:, k].sum(1).mean()) / n_t
                draws = np.empty(B)
                nw = counts.shape[1]
                for i in range(B):
                    w = rng.integers(0, nw, nw)
                    c = counts[k, w].sum()
                    if c == 0:
                        draws[i] = np.nan
                        continue
                    ds = dn[rng.integers(0, dn.shape[0], dn.shape[0])][:, k][:, w].sum(1).mean()
                    ks = kn[rng.integers(0, kn.shape[0], kn.shape[0])][:, k][:, w].sum(1).mean()
                    draws[i] = (ds - ks) / c
                lo, hi = np.nanpercentile(draws, [2.5, 97.5])
                rows.append({"bucket": labels[k], "n_targets": int(n_t), "d_nats": float(point),
                             "ci95": [float(lo), float(hi)],
                             "nll_dense": float(dn[:, k].sum(1).mean() / n_t), "nll_kas": float(kn[:, k].sum(1).mean() / n_t)})
            h3[kas] = rows
        prim = h3.get("KAS-U16") or h3.get("KAS-P")
        if prim:
            q = [r for r in prim if r.get("n_targets", 0) >= 1000]
            better = [i for i, r in enumerate(q) if r["ci95"][0] > 0]
            worse = [i for i, r in enumerate(q) if r["ci95"][1] < 0]
            ds = np.array([r["d_nats"] for r in q])
            # Spearman rho between bucket order (rare -> frequent) and d, as pre-registered
            rank_d = np.argsort(np.argsort(ds)).astype(float)
            order = np.arange(len(ds), dtype=float)
            trend = float(np.corrcoef(order, rank_d)[0, 1]) if len(ds) > 2 else 0.0
            if better and worse and max(better) < min(worse):
                verdict = "SUPPORTED"
            elif better and worse and min(better) > max(worse):
                verdict = "REJECTED"
            elif not better and worse and trend < 0 and q[-1]["ci95"][1] < 0:
                verdict = "PARTIAL"
            elif (len(worse) == len(q) or len(better) == len(q)) and trend >= 0:
                verdict = "REJECTED"
            else:
                verdict = "INCONCLUSIVE"
            res["H3"] = {"primary": "KAS-U16" if h3.get("KAS-U16") else "KAS-P", "buckets": h3,
                         "spearman_rho_bucket_order_vs_d": trend,
                         "operationalization": "qualifying buckets have >= 1000 targets; 'no trend' for REJECTED "
                                               "is implemented as Spearman rho >= 0 (not pre-specified numerically)",
                         "verdict": verdict}

    # ---------- H2 / H4 and minting tables ----------
    def minting(run, sp):
        p = os.path.join(run["dir"], f"analysis_{sp}", "minting.npz")
        j = os.path.join(run["dir"], f"analysis_{sp}", "minting.json")
        if not os.path.exists(p):
            return None, None
        return dict(np.load(p)), json.load(open(j, encoding="utf-8"))

    mint = {}
    for arm, rs in arms.items():
        for sp in {split, args.select_split}:
            per = [minting(r, sp) for r in rs]
            per = [x for x in per if x[0] is not None]
            if per:
                mint[(arm, sp)] = per
    table = {}
    for (arm, sp), per in mint.items():
        rules = per[0][1]["rules"]
        table.setdefault(sp, {})[arm] = {
            name: {k: float(np.mean([pp[1]["rules"][name][k] for pp in per]))
                   for k in ("mean_regret_bits", "beats_prefix", "median_rank", "leak_bpb")}
            for name in rules}
        table[sp][arm]["_n_sites"] = per[0][1]["n_sites"]
    res["minting_table"] = table

    def best_rule(arm, allowed=None):
        t = table.get(args.select_split, {}).get(arm)
        if not t:
            return None
        cands = {k: v for k, v in t.items() if not k.startswith("_") and not k.endswith("_cal")
                 and (allowed is None or k in allowed)}
        return min(cands, key=lambda k: cands[k]["mean_regret_bits"]) if cands else None

    def site_vals(arm, rule, sp):
        per = mint.get((arm, sp))
        if not per:
            return None, None, None
        arr = np.stack([pp[0][f"{rule}__regret"] for pp in per]).astype(np.float64)
        leak = np.stack([pp[0][f"{rule}__leak_per_window"] for pp in per]).astype(np.float64)
        return arr, leak, per[0][0]["site_w"]

    gen_m = "KAS-G" if any(a == "KAS-G" for a, _ in mint) else gen_arm
    if gen_m and ("KAS-U16", split) in mint and (gen_m, split) in mint:
        ru = best_rule("KAS-U16", {"floor", "zero", "count", "inherit", "inherit_prior"})
        rg = best_rule(gen_m)
        U, Ul, sw = site_vals("KAS-U16", ru, split)
        G, Gl, sw2 = site_vals(gen_m, rg, split)
        assert np.array_equal(sw, sw2), "sites differ between arms"
        dreg = site_bootstrap({"G": G, "U": U}, sw, lambda v: v["G"] - v["U"])
        tot_bytes = float(nbytes.sum())
        dleak = two_level_bootstrap({"G": list(Gl), "U": list(Ul)}, bw, lambda v: (v["G"] - v["U"]),
                                    n_boot=B)
        # two_level_bootstrap divides by ln2 * bytes -> already in bits per byte of the eval text
        dleak.pop("draws")
        if dreg["ci95"][1] < 0 and dleak["ci95"][0] <= 0:
            v2 = "SUPPORTED"
        elif dreg["ci95"][0] >= 0 or (dreg["ci95"][1] < 0 and dleak["ci95"][0] > 0):
            v2 = "REJECTED"
        else:
            v2 = "INCONCLUSIVE"
        res["H2"] = {"kasu_rule_selected_on_dev": ru, "kasg_rule_selected_on_dev": rg, "generator_arm": gen_m,
                     "delta_regret_bits": dreg, "delta_leak_bpb": dleak, "verdict": v2,
                     "kasu_inherit_beats_all_kasg_rules": None}
        tt = table.get(split, {})
        if "KAS-U16" in tt and gen_m in tt and "inherit" in tt["KAS-U16"]:
            g_rules = [v["mean_regret_bits"] for k, v in tt[gen_m].items()
                       if not k.startswith("_") and not k.endswith("_cal")]
            res["H2"]["kasu_inherit_beats_all_kasg_rules"] = bool(tt["KAS-U16"]["inherit"]["mean_regret_bits"] < min(g_rules))
        shuf = "KAS-G-shuf"
        if (shuf, split) in mint:
            S, Sl, sw3 = site_vals(shuf, rg, split)
            assert np.array_equal(sw, sw3)
            dshuf = site_bootstrap({"S": S, "U": U}, sw, lambda v: v["S"] - v["U"], seed=2)
            dspell = site_bootstrap({"G": G, "S": S}, sw, lambda v: v["G"] - v["S"], seed=3)
            if v2 == "SUPPORTED":
                v4 = ("SUPPORTED" if dspell["ci95"][1] < 0 and dshuf["ci95"][1] >= 0 else
                      "REJECTED" if dspell["ci95"][0] <= 0 <= dspell["ci95"][1] else "INCONCLUSIVE")
            else:
                v4 = "NOT APPLICABLE"
            res["H4"] = {"rule": rg, "delta_shuf_minus_kasu": dshuf, "delta_spell_kasg_minus_shuf": dspell,
                         "verdict": v4}

    # ---------- K2 gap curve (dev) ----------
    curve = {}
    dev_win = eval_windows(data.dev, 1024)
    dev_b = tb[dev_win[:, 1:]].sum(1).astype(np.float64)
    for arm, rs in arms.items():
        pts = {}
        for r in rs:
            for f in sorted(glob.glob(os.path.join(r["dir"], "evalcurve", "step*.npz"))):
                z = np.load(f)
                pts.setdefault(int(z["tokens"]), []).append(z["nll_sum"])
            p = os.path.join(r["dir"], "nll_dev.npy")
            if os.path.exists(p):
                s, _ = window_sums(np.load(p), tb[dev_win[:, 1:]])
                pts.setdefault(int(r["summary"]["tokens"]), []).append(s)
        curve[arm] = pts
    gaps = {}
    for a, b in (("Dense", "KAS-U16"), ("Dense", "KAS-P"), ("Dense", "KAS-G"), ("KAS-P", "KAS-U16"),
                 ("Dense", "KAS-U16-drop25"), ("KAS-U16", "KAS-U16-drop25"), ("Dense", "KAS-G-occ"),
                 ("Dense", "KAS-U64"), ("Dense", "KAS-U16-dropU25"), ("KAS-U16", "KAS-U16-dropU25"),
                 ("Dense", "Dense-prior"), ("Dense-prior", "KAS-U64")):
        if a in curve and b in curve:
            rows = []
            for t in sorted(set(curve[a]) & set(curve[b])):
                nwin = min(len(x) for x in curve[a][t] + curve[b][t])
                ca = [x[:nwin] for x in curve[a][t]]; cb = [x[:nwin] for x in curve[b][t]]
                r = two_level_bootstrap({"a": ca, "b": cb}, dev_b[:nwin], lambda v: v["b"] - v["a"], n_boot=1000)
                r.pop("draws")
                rows.append({"tokens": t, "n_windows": nwin, "bpb_a": float(np.mean([bpb_of(x, dev_b[:nwin]) for x in ca])),
                             "bpb_b": float(np.mean([bpb_of(x, dev_b[:nwin]) for x in cb])),
                             "b_minus_a": r})
            gaps[f"{b} - {a}"] = rows
    res["dev_curves"] = {arm: {str(t): float(np.mean([bpb_of(x, dev_b[:len(x)]) for x in v]))
                               for t, v in sorted(pts.items())} for arm, pts in curve.items()}
    res["gap_curves"] = gaps
    if gaps.get("KAS-U16 - Dense"):
        g = gaps["KAS-U16 - Dense"]
        res["K2_pattern"] = {"gap_first": g[0]["b_minus_a"], "gap_last": g[-1]["b_minus_a"],
                             "reproduced": bool(g[-1]["b_minus_a"]["ci95"][0] > 0 and
                                                g[-1]["b_minus_a"]["point"] > g[0]["b_minus_a"]["point"])}

    # ---------- extras ----------
    extras = {}
    for arm, rs in arms.items():
        for r in rs:
            for name in ("terms", "ablations", "counterfactual", "retok"):
                p = os.path.join(r["dir"], f"analysis_{split}", f"{name}.json")
                if os.path.exists(p):
                    extras.setdefault(arm, {}).setdefault(name, []).append(json.load(open(p, encoding="utf-8")))
    res["extras"] = extras
    (out / f"analysis_{tag}_{split}.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    brief = {k: res.get(k, {}).get("verdict") if isinstance(res.get(k), dict) else None for k in ("H1", "H2", "H3", "H4")}
    print(json.dumps({"arms": {a: (v["bpb_mean"], v["seed_spread"]) for a, v in res["arms"].items()},
                      "verdicts": brief, "fairness": res["fairness"],
                      "H1": res.get("H1", {}).get("rho"), "K2": res.get("K2_pattern")}, indent=1, default=float))


if __name__ == "__main__":
    main()
