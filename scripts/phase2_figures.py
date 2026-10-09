"""Phase 2 figure (protocol 06): gap-vs-tokens curves with the Phase 2 arms, and the
quality-versus-minting frontier with every arm (Phase 1 + Phase 2).

    python scripts/phase2_figures.py      # reads results/phase2/{analysis_m3_p2_test.json, phase2_verdicts.json}
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from make_figures import COL, K2_GAP, MARK, save  # noqa: E402

COL.update({"KAS-U16-drop25": "#009E73", "KAS-G-occ": "#F0E442", "KAS-U64": "#882255"})
MARK.update({"KAS-U16-drop25": "P", "KAS-G-occ": "X", "KAS-U64": "*"})
P2 = ("KAS-U16-drop25", "KAS-G-occ", "KAS-U64")


def main() -> None:
    A = json.loads((ROOT / "results/phase2/analysis_m3_p2_test.json").read_text(encoding="utf-8"))
    V = json.loads((ROOT / "results/phase2/phase2_verdicts.json").read_text(encoding="utf-8"))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.8))
    # left: dev gap to Dense over training, Phase 1 comparators + Phase 2 arms
    for key, arm in (("KAS-U16 - Dense", "KAS-U16"), ("KAS-G - Dense", "KAS-G"),
                     ("KAS-U16-drop25 - Dense", "KAS-U16-drop25"), ("KAS-G-occ - Dense", "KAS-G-occ"),
                     ("KAS-U64 - Dense", "KAS-U64")):
        rows = A.get("gap_curves", {}).get(key)
        if not rows:
            continue
        t = [r["tokens"] / 1e6 for r in rows]
        y = [r["b_minus_a"]["point"] for r in rows]
        lo = [r["b_minus_a"]["ci95"][0] for r in rows]
        hi = [r["b_minus_a"]["ci95"][1] for r in rows]
        ax1.plot(t, y, marker=MARK[arm], color=COL[arm], label=arm, ls="--" if arm in P2 else "-")
        ax1.fill_between(t, lo, hi, color=COL[arm], alpha=0.12, lw=0)
    ax1.plot([k / 1e6 for k in K2_GAP], list(K2_GAP.values()), ls=":", color="#888888", marker=".",
             label="reference report (k16 − dense)")
    ax1.axhline(0, color="black", lw=0.6)
    ax1.set_xlabel("training tokens (M)"); ax1.set_ylabel("dev bpb − Dense"); ax1.set_title("gap to the dense head")
    ax1.legend(ncol=1)
    # right: frontier
    T = V["quality_vs_minting"]["table"]
    for arm, row in T.items():
        ax2.scatter(row["in_vocab_test_bpb"], row["net_minting_test_bpb"] * 1e3, marker=MARK.get(arm, "o"),
                    color=COL.get(arm, "#333333"), s=46 if arm in P2 else 30, zorder=3,
                    edgecolor="black" if arm in P2 else "none", lw=0.6, label=f"{arm} ({row['rule_by_net_dev']})")
    nd = sorted(((T[a]["in_vocab_test_bpb"], T[a]["net_minting_test_bpb"] * 1e3) for a in V["quality_vs_minting"]["non_dominated"]))
    if len(nd) > 1:
        ax2.plot([x for x, _ in nd], [y for _, y in nd], color="#bbbbbb", lw=0.8, zorder=1)
    ax2.set_xlabel("in-vocabulary test bpb (lower is better)")
    ax2.set_ylabel("net minting Δbpb ×10⁻³ (lower is better)")
    ax2.set_title("quality versus minting cost")
    ax2.legend(fontsize=6.2, ncol=1, loc="upper right")
    save(fig, "fig8_phase2_gap_and_frontier")


if __name__ == "__main__":
    main()
