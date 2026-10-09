"""Reference-stability table: floors against Dense-prior seed 1 / seed 2 on dev / test, CPU and GPU runs.

Reads evidence/gpu/floor_gpu_s{1,2}_{dev,test}.json (Modal L4, 4,096 contexts) and the CPU runs
evidence/floor_contextual.json (s1 dev, 1,024), floor_contextual_seed2.json (s2 dev), floor_contextual_test.json (s1 test).

    python scripts/expressivity/summarize_stability.py
      -> results/expressivity/stability_summary.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from floor_lib import ROOT  # noqa: E402

EV = ROOT / "results/expressivity"
RUNS = [("s1 dev (CPU)", EV / "floor_contextual.json"), ("s2 dev (CPU)", EV / "floor_contextual_seed2.json"),
        ("s1 test (CPU)", EV / "floor_contextual_test.json"),
        ("s1 dev (GPU)", EV / "gpu/floor_gpu_s1_dev.json"), ("s2 dev (GPU)", EV / "gpu/floor_gpu_s2_dev.json"),
        ("s1 test (GPU)", EV / "gpu/floor_gpu_s1_test.json"), ("s2 test (GPU)", EV / "gpu/floor_gpu_s2_test.json")]


def short(tag: str) -> str:
    return (tag.replace(" | bias = dense-prior's learned beta (fixed)", " / beta")
               .replace(" | bias = log-unigram prior (fixed; KAS-P family)", " / log-unigram")
               .replace(" | bias = free per-token, init beta (looser lower bound)", " / free bias")
               .replace(" | bias = beta (fixed)", " / beta"))


def main():
    rows = {}
    arms = {}
    meta = {}
    for name, path in RUNS:
        if not path.exists():
            continue
        d = json.loads(path.read_text(encoding="utf-8"))
        meta[name] = (d["n_contexts"], d.get("dense_prior_nll_saved_gpu"), d.get("dense_prior_nll_recomputed_cpu"))
        for tag, v in d["floors"].items():
            rows.setdefault(short(tag), {})[name] = (v["floor_nats_per_target"], v["mimic_minus_dense_prior_nats"],
                                                     v["arms_nll_same_subset"]["KAS-P"] - v["dense_prior_nll_true_targets_same"],
                                                     v["n_contexts"])
        arms[name] = {k: v - d["arms_nll_nats_per_target_same_positions"]["Dense-prior"]
                      for k, v in d["arms_nll_nats_per_target_same_positions"].items()}
    names = [n for n, _ in RUNS if n in meta]
    lines = ["# Reference stability of the floors", "",
             "Floor (nats/target) of each feature family against the Dense-prior distribution; in brackets: fitted mimic's "
             "true-target gap to Dense-prior / realised KAS-P gap on the same contexts / number of contexts.", "",
             "| family / bias | " + " | ".join(names) + " |", "|---|" + "---:|" * len(names)]
    for fam, per in rows.items():
        cells = []
        for n in names:
            if n in per:
                f, mg, kg, nc = per[n]
                cells.append(f"{f:.3f} [{mg:+.3f} / {kg:+.3f} / {nc}]")
            else:
                cells.append("")
        lines.append(f"| {fam} | " + " | ".join(cells) + " |")
    lines += ["", "Realised gaps of the historical arms to Dense-prior on each run's positions (nats/target):", "",
              "| arm | " + " | ".join(names) + " |", "|---|" + "---:|" * len(names)]
    for arm in ["KAS-U64", "Dense", "KAS-U16", "KAS-G", "KAS-P", "KAS-0"]:
        lines.append(f"| {arm} | " + " | ".join(f"{arms[n].get(arm, float('nan')):+.3f}" for n in names) + " |")
    lines += ["", "Reference NLL on the sampled positions (saved GPU array vs recomputed here): " +
              "; ".join(f"{n}: {m[1]:.4f} vs {m[2]:.4f} ({m[0]} contexts)" for n, m in meta.items())]
    (EV / "stability_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
