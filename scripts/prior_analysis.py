"""Protocol 08 analysis: prior-matched Dense (Dense + trainable log-unigram bias) against the existing Dense and
KAS-U64 runs, seeds 1-4 per arm. Per-seed table; P = bpb(Dense) - bpb(Dense-prior) with a paired-by-seed t
interval (the two arms share initialisation seed by seed) and the Protocol 03 two-level bootstrap;
R = bpb(KAS-U64) - bpb(Dense-prior) with the bootstrap and a Welch interval; the share of the original
Dense - KAS-U64 gap that prior matching explains, with a bootstrap interval; frequency-bucket decomposition when
the joint final_analysis output exists; and the pre-registered outcome word (protocol 08 §6).

    python scripts/prior_analysis.py      # -> results/phase4/prior_verdict.json
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
from confirm_analysis import t_ppf, welch  # noqa: E402
from kq5.data import TokenData, eval_windows  # noqa: E402
from kq5.stats import bpb_of, two_level_bootstrap, window_sums  # noqa: E402
from kq5.vocab import WorkingVocab  # noqa: E402

B = 2000
RUNS = {"Dense": [("experiments/modal-m3/runs", "m3-dense-s1", 1), ("experiments/modal-m3/runs", "m3-dense-s2", 2),
                  ("experiments/modal-p3/runs", "p3-dense-s3", 3), ("experiments/modal-p3/runs", "p3-dense-s4", 4)],
        "Dense-prior": [("experiments/modal-p4/runs", f"p4-densep-s{s}", s) for s in (1, 2, 3, 4)],
        "KAS-U64": [("experiments/modal-p2/runs", "p2-kasu64-s1", 1), ("experiments/modal-p2/runs", "p2-kasu64-s2", 2),
                    ("experiments/modal-p3/runs", "p3-kasu64-s3", 3), ("experiments/modal-p3/runs", "p3-kasu64-s4", 4)]}


def load(arm):
    out = []
    for root, rid, seed in RUNS[arm]:
        d = ROOT / root / rid
        if not (d / "nll_test.npy").exists():
            continue
        s = json.loads((d / "summary.json").read_text(encoding="utf-8"))
        rec = json.loads((d / "run_record.json").read_text(encoding="utf-8"))
        out.append({"id": rid, "seed": seed,
                    "complete": s.get("status") == "complete" and s.get("steps") == rec["total_steps"],
                    "skipped_fp16_steps": s.get("skipped_steps"), "wall_s": s.get("wall_s"),
                    "dense_bias": rec["config"].get("dense_bias", False), "head": rec["config"]["head"],
                    "nll": np.load(d / "nll_test.npy"),
                    "device": json.loads((d / "device.json").read_text(encoding="utf-8")).get("device")
                    if (d / "device.json").exists() else None})
    return out


def paired_t(diffs):
    d = np.asarray(diffs, float)
    n = len(d)
    m, sd = float(d.mean()), float(d.std(ddof=1))
    t = t_ppf(0.975, n - 1)
    se = sd / math.sqrt(n)
    return {"point": m, "sd": sd, "se": se, "df": n - 1, "t": t, "ci95": [m - t * se, m + t * se]}


def main() -> None:
    tok = ROOT / "data" / "tokens"
    data = TokenData.load(tok, verify=False)
    vocab = WorkingVocab.load(tok / "vocab.json")
    tb = vocab.target_bytes()
    win = eval_windows(data.test, 1024)
    nbytes = tb[win[:, 1:]]
    _, bw = window_sums(np.zeros_like(nbytes, dtype=np.float64), nbytes)
    arms = {a: load(a) for a in RUNS}
    out = {"protocol": "08", "runs": {}, "per_seed": {}}
    sums, bpb = {}, {}
    for a, rs in arms.items():
        sums[a] = [window_sums(r["nll"], nbytes)[0] for r in rs]
        bpb[a] = {r["seed"]: bpb_of(s, bw) for r, s in zip(rs, sums[a])}
        out["runs"][a] = [{k: v for k, v in r.items() if k != "nll"} | {"test_bpb": bpb[a][r["seed"]]} for r in rs]
    seeds = sorted(set(bpb["Dense"]) & set(bpb["Dense-prior"]) & set(bpb["KAS-U64"]))
    for s in seeds:
        out["per_seed"][str(s)] = {"Dense": bpb["Dense"][s], "Dense-prior": bpb["Dense-prior"][s], "KAS-U64": bpb["KAS-U64"][s],
                                   "P_dense_minus_prior": bpb["Dense"][s] - bpb["Dense-prior"][s],
                                   "R_u64_minus_prior": bpb["KAS-U64"][s] - bpb["Dense-prior"][s],
                                   "G_dense_minus_u64": bpb["Dense"][s] - bpb["KAS-U64"][s]}
    incomplete = [r["id"] for rs in arms.values() for r in rs if not r["complete"]]
    out["means"] = {a: float(np.mean(list(v.values()))) for a, v in bpb.items() if v}
    if len(seeds) < 2 or incomplete or len(arms["Dense-prior"]) < 4:
        out["verdict"] = {"word": "UNCERTAIN", "reason": "prior-matched runs missing or incomplete",
                          "available": [r["id"] for r in arms["Dense-prior"]], "incomplete": incomplete}
    else:
        D, Pm, U = [bpb["Dense"][s] for s in seeds], [bpb["Dense-prior"][s] for s in seeds], [bpb["KAS-U64"][s] for s in seeds]
        sD = [sums["Dense"][[r["seed"] for r in arms["Dense"]].index(s)] for s in seeds]
        sP = [sums["Dense-prior"][[r["seed"] for r in arms["Dense-prior"]].index(s)] for s in seeds]
        sU = [sums["KAS-U64"][[r["seed"] for r in arms["KAS-U64"]].index(s)] for s in seeds]
        G = float(np.mean(D) - np.mean(U))
        P_boot = two_level_bootstrap({"D": sD, "P": sP}, bw, lambda v: v["D"] - v["P"], n_boot=B, seed=81); P_boot.pop("draws")
        R_boot = two_level_bootstrap({"U": sU, "P": sP}, bw, lambda v: v["U"] - v["P"], n_boot=B, seed=82); R_boot.pop("draws")
        share = two_level_bootstrap({"D": sD, "P": sP, "U": sU}, bw,
                                    lambda v: (v["D"] - v["P"]) / (v["D"] - v["U"]) if v["D"] - v["U"] > 0 else float("nan"),
                                    n_boot=B, seed=83)
        draws = share.pop("draws")
        share["n_nan_draws"] = int(np.isnan(draws).sum())
        share["ci95"] = [float(x) for x in np.nanpercentile(draws, [2.5, 97.5])]
        res = {"seeds": seeds, "G_dense_minus_u64": G,
               "P_dense_minus_prior": {"point": float(np.mean(D) - np.mean(Pm)), "paired_t": paired_t(np.array(D) - np.array(Pm)),
                                       "bootstrap": P_boot},
               "R_u64_minus_prior": {"point": float(np.mean(U) - np.mean(Pm)), "bootstrap": R_boot, "welch": welch(U, Pm)},
               "share_explained": share,
               "seed_sd": {"Dense": float(np.std(D, ddof=1)), "Dense-prior": float(np.std(Pm, ddof=1)), "KAS-U64": float(np.std(U, ddof=1))},
               "pairs_u64_below_prior": int(sum(u < p for u in U for p in Pm)), "n_pairs": len(U) * len(Pm),
               "all_u64_below_all_prior": bool(max(U) < min(Pm)), "all_prior_below_all_dense": bool(max(Pm) < min(D)),
               "rank_order_all_runs": [{"id": r["id"], "test_bpb": r["test_bpb"]} for r in
                                       sorted((r for rs in out["runs"].values() for r in rs), key=lambda r: r["test_bpb"])]}
        # pre-registered outcome word (protocol 08 §6)
        Rp, Rci = res["R_u64_minus_prior"]["point"], R_boot["ci95"]
        Pci, sh = res["P_dense_minus_prior"]["paired_t"]["ci95"], share["point"]
        if Rp >= 0:
            word = "D_PRIOR_MATCHED_DENSE_MATCHES_OR_BEATS_U64"
        elif Rci[1] >= 0:
            word = "E_AMBIGUOUS"
        elif sh >= 0.5 and Pci[0] > 0:
            word = "A_CLOSES_MOST"
        elif sh >= 0.2 and Pci[0] > 0:
            word = "B_PARTIAL"
        elif sh >= 0.5:
            word = "E_AMBIGUOUS"
        else:
            word = "C_LITTLE_CHANGE"
        res["verdict"] = {"word": word, "conditions": {"R_point_lt_0": Rp < 0, "R_bootstrap_ci_below_0": Rci[1] < 0,
                                                       "R_welch_ci_below_0": res["R_u64_minus_prior"]["welch"]["ci95"][1] < 0,
                                                       "P_paired_ci_above_0": Pci[0] > 0, "share_point": sh}}
        out.update(res)
        out["verdict"] = res["verdict"]
    # frequency-bucket decomposition from the joint final_analysis output, if present
    joint = ROOT / "results" / "phase4" / "analysis_m3_p2_p3_p4_test.json"
    if joint.exists():
        j = json.loads(joint.read_text(encoding="utf-8"))
        h3 = j.get("H3", {}).get("buckets", {})
        if "KAS-U64" in h3 and "Dense-prior" in h3:
            rows = []
            for u, p in zip(h3["KAS-U64"], h3["Dense-prior"]):
                if u.get("n_targets", 0) == 0:
                    continue
                rows.append({"bucket": u["bucket"], "n_targets": u["n_targets"],
                             "dense_minus_u64_nats": u["d_nats"], "dense_minus_u64_ci95": u["ci95"],
                             "dense_minus_prior_nats": p["d_nats"], "dense_minus_prior_ci95": p["ci95"],
                             "contribution_nats_u64": u["d_nats"] * u["n_targets"],
                             "contribution_nats_prior": p["d_nats"] * p["n_targets"]})
            out["buckets"] = {"rows": rows, "note": "d = NLL(Dense) - NLL(arm) per target, positive = arm better; "
                                                    "contribution = d x n_targets (nats over the test split)"}
        # derived README numbers from the joint analysis (seed means over the runs present)
        arms_j, ex = j.get("arms", {}), j.get("extras", {})
        der = {"mean_bpb": {a: v["bpb_mean"] for a, v in arms_j.items()},
               "mean_diff_vs_dense_prior": {a: v["bpb_mean"] - arms_j["Dense-prior"]["bpb_mean"] for a, v in arms_j.items()
                                            if "Dense-prior" in arms_j},
               "net_retok_delta_bpb_x1e3": {a: {rule: float(np.mean([r[rule]["delta_bpb"] for r in ex[a]["retok"]])) * 1e3
                                                for rule in ex[a]["retok"][0]} for a in ex if "retok" in ex[a]},
               "dense_prior_ablation_mean": {k: float(np.mean([r[k] for r in ex.get("Dense-prior", {}).get("ablations", [])]))
                                             for k in ("bpb_full", "bpb_no_prior")} if ex.get("Dense-prior", {}).get("ablations") else None,
               "dense_prior_term_share_prior_mean": float(np.mean([r["share_prior"] for r in ex.get("Dense-prior", {}).get("terms", [])]))
               if ex.get("Dense-prior", {}).get("terms") else None,
               "kas_u64_term_share_prior_mean": float(np.mean([r["share_prior"] for r in ex.get("KAS-U64", {}).get("terms", [])]))
               if ex.get("KAS-U64", {}).get("terms") else None}
        if "buckets" in out:
            tot_u = sum(r["contribution_nats_u64"] for r in out["buckets"]["rows"])
            tot_p = sum(r["contribution_nats_prior"] for r in out["buckets"]["rows"])
            der["bucket_share_below_100k"] = {
                "u64": sum(r["contribution_nats_u64"] for r in out["buckets"]["rows"] if r["bucket"] not in ("[100000,1000000)", "[1000000,inf)")) / tot_u,
                "prior": sum(r["contribution_nats_prior"] for r in out["buckets"]["rows"] if r["bucket"] not in ("[100000,1000000)", "[1000000,inf)")) / tot_p}
            der["bucket_total_nats"] = {"u64": tot_u, "prior": tot_p}
        out["derived"] = der
    out_dir = ROOT / "results" / "phase4"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "prior_verdict.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    brief = {"verdict": out["verdict"], "means": out["means"], "per_seed": out["per_seed"],
             "P": out.get("P_dense_minus_prior"), "R": out.get("R_u64_minus_prior"), "share": out.get("share_explained")}
    print(json.dumps(brief, indent=1, default=float))


if __name__ == "__main__":
    main()
