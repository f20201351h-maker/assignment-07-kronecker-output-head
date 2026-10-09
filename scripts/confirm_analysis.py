"""Protocol 07 confirmation analysis: KAS-U64 against Dense on the fresh seed pairs (3, 4) alone and on all
seeds combined. Per-seed values, two intervals (Protocol 03 two-level bootstrap; Welch t on seed means), seed
standard deviations, all-pairs count, rank order, and the pre-registered verdict word.

    python scripts/confirm_analysis.py      # -> results/phase3/confirm_verdict.json
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from kq5.data import TokenData, eval_windows  # noqa: E402
from kq5.stats import bpb_of, two_level_bootstrap, window_sums  # noqa: E402
from kq5.vocab import WorkingVocab  # noqa: E402

B = 2000
RUNS = {"Dense": [("experiments/modal-m3/runs", "m3-dense-s1", 1), ("experiments/modal-m3/runs", "m3-dense-s2", 2),
                  ("experiments/modal-p3/runs", "p3-dense-s3", 3), ("experiments/modal-p3/runs", "p3-dense-s4", 4)],
        "KAS-U64": [("experiments/modal-p2/runs", "p2-kasu64-s1", 1), ("experiments/modal-p2/runs", "p2-kasu64-s2", 2),
                    ("experiments/modal-p3/runs", "p3-kasu64-s3", 3), ("experiments/modal-p3/runs", "p3-kasu64-s4", 4)]}
ORIGINAL_ADVANTAGE = 0.0281          # Phase 2, seeds 1-2 (protocol 07 text)


def t_ppf(q, df, lo=0.0, hi=60.0):
    """Student t quantile by numeric integration (no scipy on this machine)."""
    c = math.gamma((df + 1) / 2) / (math.sqrt(df * math.pi) * math.gamma(df / 2))

    def cdf(x, n=20000):
        h = x / n
        tot = 0.0
        for i in range(n + 1):
            xi = i * h
            w = 1 if i in (0, n) else (4 if i % 2 else 2)
            tot += w * c * (1 + xi * xi / df) ** (-(df + 1) / 2)
        return 0.5 + tot * h / 3
    for _ in range(60):
        mid = (lo + hi) / 2
        if cdf(mid) < q:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def welch(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = a.mean() - b.mean()
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = math.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    t = t_ppf(0.975, df)
    return {"point": float(m), "se": se, "df": df, "t": t, "ci95": [float(m - t * se), float(m + t * se)]}


def load(arm, seeds):
    out = []
    for root, rid, seed in RUNS[arm]:
        if seed not in seeds:
            continue
        d = ROOT / root / rid
        if not (d / "nll_test.npy").exists():
            continue
        s = json.loads((d / "summary.json").read_text(encoding="utf-8"))
        rec = json.loads((d / "run_record.json").read_text(encoding="utf-8"))
        ok = s.get("status") == "complete" and s.get("steps") == rec["total_steps"]
        out.append({"id": rid, "seed": seed, "complete": ok, "skipped_fp16_steps": s.get("skipped_steps"),
                    "wall_s": s.get("wall_s"), "nll": np.load(d / "nll_test.npy"),
                    "device": json.loads((d / "device.json").read_text(encoding="utf-8")).get("device") if (d / "device.json").exists() else None})
    return out


def analyse(dense, u64, bw, nbytes, seed):
    """Both intervals for bpb(U64) - bpb(Dense) given lists of run dicts."""
    sd = [window_sums(r["nll"], nbytes)[0] for r in dense]
    su = [window_sums(r["nll"], nbytes)[0] for r in u64]
    bd = [bpb_of(s, bw) for s in sd]
    bu = [bpb_of(s, bw) for s in su]
    boot = two_level_bootstrap({"U": su, "D": sd}, bw, lambda v: v["U"] - v["D"], n_boot=B, seed=seed)
    boot.pop("draws")
    return {"dense_bpb": bd, "u64_bpb": bu, "dense_mean": float(np.mean(bd)), "u64_mean": float(np.mean(bu)),
            "delta_point": float(np.mean(bu) - np.mean(bd)), "bootstrap": boot,
            "welch": welch(bu, bd) if len(bu) > 1 and len(bd) > 1 else None,
            "dense_seed_sd": float(np.std(bd, ddof=1)) if len(bd) > 1 else None,
            "u64_seed_sd": float(np.std(bu, ddof=1)) if len(bu) > 1 else None,
            "pairs_u64_lower": int(sum(u < d for u in bu for d in bd)), "n_pairs": len(bu) * len(bd),
            "all_u64_below_all_dense": bool(max(bu) < min(bd)) if bu and bd else None}


def main() -> None:
    tok = ROOT / "data" / "tokens"
    data = TokenData.load(tok, verify=False)
    vocab = WorkingVocab.load(tok / "vocab.json")
    tb = vocab.target_bytes()
    win = eval_windows(data.test, 1024)
    nbytes = tb[win[:, 1:]]
    _, bw = window_sums(np.zeros_like(nbytes, dtype=np.float64), nbytes)
    out = {"protocol": "07", "original_advantage_seeds12": ORIGINAL_ADVANTAGE}
    fresh_d, fresh_u = load("Dense", {3, 4}), load("KAS-U64", {3, 4})
    all_d, all_u = load("Dense", {1, 2, 3, 4}), load("KAS-U64", {1, 2, 3, 4})
    out["runs"] = {arm: [{k: v for k, v in r.items() if k != "nll"} | {"test_bpb": bpb_of(window_sums(r["nll"], nbytes)[0], bw)}
                         for r in rs] for arm, rs in (("Dense", all_d), ("KAS-U64", all_u))}
    incomplete = [r["id"] for r in fresh_d + fresh_u if not r["complete"]]
    if len(fresh_d) == 2 and len(fresh_u) == 2 and not incomplete:
        f = analyse(fresh_d, fresh_u, bw, nbytes, seed=71)
        out["fresh_pairs_seeds34"] = f
        conds = {"both_fresh_u64_below_both_fresh_dense": f["all_u64_below_all_dense"],
                 "bootstrap_ci_below_0": f["bootstrap"]["ci95"][1] < 0,
                 "welch_ci_below_0": f["welch"]["ci95"][1] < 0,
                 "at_least_half_original": abs(f["delta_point"]) >= ORIGINAL_ADVANTAGE / 2 and f["delta_point"] < 0}
        if f["delta_point"] >= 0:
            v = "REVERSED"
        elif all(conds.values()):
            v = "CONFIRMED"
        else:
            v = "WEAKENED"
        out["verdict"] = {"word": v, "conditions": conds}
    else:
        out["fresh_pairs_seeds34"] = {"available": [r["id"] for r in fresh_d + fresh_u], "incomplete": incomplete}
        out["verdict"] = {"word": "UNCERTAIN", "reason": "fresh runs missing or incomplete"}
    by = {arm: {r["seed"]: r["test_bpb"] for r in rs} for arm, rs in out["runs"].items()}
    out["per_seed_diff_u64_minus_dense"] = {str(s): by["KAS-U64"][s] - by["Dense"][s]
                                            for s in sorted(set(by["Dense"]) & set(by["KAS-U64"]))}
    if len(all_d) >= 2 and len(all_u) >= 2:
        c = analyse(all_d, all_u, bw, nbytes, seed=72)
        c["u64_sd_over_advantage"] = c["u64_seed_sd"] / abs(c["delta_point"]) if c["u64_seed_sd"] else None
        c["original_minus_fresh_advantage"] = ORIGINAL_ADVANTAGE - abs(out["fresh_pairs_seeds34"].get("delta_point", 0.0))
        order = sorted([(r["test_bpb"], r["id"]) for arm in out["runs"].values() for r in arm])
        c["rank_order_all_runs"] = [{"id": i, "test_bpb": b} for b, i in order]
        out["combined_all_seeds"] = c
    out_dir = ROOT / "results" / "phase3"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "confirm_verdict.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    brief = {"verdict": out["verdict"], "fresh": {k: out["fresh_pairs_seeds34"].get(k) for k in ("dense_bpb", "u64_bpb", "delta_point", "bootstrap", "welch")},
             "combined": {k: out.get("combined_all_seeds", {}).get(k) for k in ("delta_point", "bootstrap", "welch", "dense_seed_sd", "u64_seed_sd", "pairs_u64_lower")}}
    print(json.dumps(brief, indent=1, default=float))


if __name__ == "__main__":
    main()
