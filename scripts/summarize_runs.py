"""Summarise downloaded Kaggle runs: final bpb, eval curves, throughput, minting tables.

    python scripts/summarize_runs.py experiments/kq5-r2-pilot [more dirs] --out results/r2/summary.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_run(d: str) -> dict:
    rid = os.path.basename(d.rstrip("/\\"))
    out = {"run_id": rid}
    mpath = os.path.join(d, "metrics.jsonl")
    if not os.path.exists(mpath):
        out["status"] = "no metrics"
        return out
    lines = [json.loads(l) for l in open(mpath, encoding="utf-8")]
    rec = json.load(open(os.path.join(d, "run_record.json"), encoding="utf-8"))
    cfg = rec["config"]
    tr = [l for l in lines if l.get("tok_per_s")]
    fin = [l for l in lines if "final" in l]
    out.update({
        "head": cfg["head"], "seed": cfg["seed"], "generator": cfg.get("generator"),
        "gen_lr_mult": cfg.get("gen_lr_mult"), "head_lr_mult": cfg.get("head_lr_mult"),
        "backbone": f"{cfg['n_layer']}L{cfg['d_model']}",
        "params": rec["params"], "train_order_sha256": rec["train_order_sha256"],
        "dev_mid_sha256": rec["dev_mid_sha256"], "total_steps": rec["total_steps"],
        "tok_per_s_median": float(np.median([l["tok_per_s"] for l in tr if l["step"] > 10])) if tr else None,
        "peak_mem_gb": max((l.get("peak_mem_gb", 0) for l in tr), default=None),
        "eval_curve": [(l["tokens"], l["bpb"]) for l in lines if l.get("eval") == "dev_mid"],
        "final_loss": tr[-1]["loss"] if tr else None,
    })
    if fin:
        f = fin[-1]
        out["status"] = f["status"]
        out["steps"] = f["step"]
        out["tokens"] = f["tokens"]
        out["dev_bpb"] = f["final"]["dev"]["bpb"]
        out["dev_windows_sha256"] = f["final"]["dev"]["windows_sha256"]
        if "test" in f["final"]:
            out["test_bpb"] = f["final"]["test"]["bpb"]
            out["test_windows_sha256"] = f["final"]["test"]["windows_sha256"]
    for split in ("dev", "test"):
        a = os.path.join(d, f"analysis_{split}")
        if os.path.exists(os.path.join(a, "minting.json")):
            m = json.load(open(os.path.join(a, "minting.json"), encoding="utf-8"))
            out[f"minting_{split}"] = {"n_sites": m["n_sites"], "rules": m["rules"]}
            for name in ("terms", "ablations", "counterfactual", "retok"):
                p = os.path.join(a, f"{name}.json")
                if os.path.exists(p):
                    out[f"{name}_{split}"] = json.load(open(p, encoding="utf-8"))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    runs = []
    for d in args.dirs:
        for rd in sorted(glob.glob(os.path.join(d, "runs", "*"))):
            if os.path.isdir(rd):
                runs.append(load_run(rd))
    orders = {r.get("train_order_sha256") for r in runs if "train_order_sha256" in r}
    devs = {r.get("dev_windows_sha256") for r in runs if "dev_windows_sha256" in r}
    summary = {"runs": runs, "fairness": {"distinct_train_orders": len(orders), "distinct_dev_windows": len(devs)}}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"{'run':28s} {'head':10s} {'tok/s':>7s} {'dev bpb':>8s} {'test bpb':>8s}  minting(best rule: mean regret bits, leak bpb, beats)")
    for r in runs:
        best = ""
        m = r.get("minting_dev") or r.get("minting_test")
        if m:
            k = min(m["rules"], key=lambda k: m["rules"][k]["mean_regret_bits"])
            v = m["rules"][k]
            best = f"{k}: {v['mean_regret_bits']:.3f}, {v['leak_bpb']:.4f}, {v['beats_prefix']:.2f} (n={m['n_sites']})"
        tps = r.get("tok_per_s_median")
        print(f"{r['run_id']:28s} {r.get('head', '?'):10s} {tps or 0:7.0f} {r.get('dev_bpb', float('nan')):8.4f} "
              f"{r.get('test_bpb', float('nan')):8.4f}  {best}")
    print("fairness:", summary["fairness"])


if __name__ == "__main__":
    main()
