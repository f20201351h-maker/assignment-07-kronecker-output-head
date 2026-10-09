"""Build (and optionally execute) the Phase 2 notebook from saved results (no GPU, no training).

    python scripts/build_phase2_notebook.py [--execute]
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]


def md(s):
    return nbf.v4.new_markdown_cell(s.strip())


def code(s):
    return nbf.v4.new_code_cell(s.strip())


def notebook() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md("""
# Phase 2 (protocol 06 / 06a) — verdicts from saved results

Recomputes the Phase 2 verdicts (Q1, H5–H8), the quality-versus-minting table and figure 8 from the
downloaded Phase 1 (`experiments/modal-m3/runs`) and Phase 2 (`experiments/modal-p2/runs`) run directories.
No GPU and no training. The Phase 1 verdicts are recomputed with the Phase 2 arms present and must be unchanged.

Later correction: every "Dense" here is the dense head without an output bias. The prior-matched control
(protocol 08, `results/phase4/prior_verdict.json`) gave Dense the same log-unigram bias that the KAS-U heads have; it
then beat KAS-U64 on every seed, so the KAS-U64 lead shown below does not survive information parity.
"""),
        code("""
import json, subprocess, sys
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
OUT = ROOT / "results" / "phase2"
RUNS = [str(ROOT / "experiments/modal-m3/runs"), str(ROOT / "experiments/modal-p2/runs")]
"""),
        md("## 1. Joint analysis of Phase 1 + Phase 2 arms (same code and bootstraps as Phase 1)"),
        code("""
r = subprocess.run([sys.executable, str(ROOT / "scripts" / "final_analysis.py"), "--runs", *RUNS,
                    "--prefix", "m3", "p2", "--split", "test", "--out", str(OUT)], capture_output=True, text=True, cwd=ROOT)
print(r.stdout[-2500:] or r.stderr[-2500:])
A = json.load(open(OUT / "analysis_m3_p2_test.json"))
P1 = json.load(open(ROOT / "results/final/analysis_m3_test.json"))
for h in ("H1", "H2", "H3", "H4"):
    assert A[h]["verdict"] == P1[h]["verdict"], h
print("Phase 1 verdicts unchanged:", {h: A[h]["verdict"] for h in ("H1", "H2", "H3", "H4")})
for arm, v in A["arms"].items():
    print(f"{arm:18s} {v['bpb_mean']:.4f}  seeds {[round(x, 4) for x in v['bpb_per_seed']]}  head {v['head_params']:,}  V-dep {v['head_V_dependent']:,}")
"""),
        md("## 2. Phase 2 verdicts"),
        code("""
r = subprocess.run([sys.executable, str(ROOT / "scripts" / "phase2_analysis.py")], capture_output=True, text=True, cwd=ROOT)
print(r.stdout[-3000:] or r.stderr[-3000:])
V = json.load(open(OUT / "phase2_verdicts.json"))
def ci(x): return f"{x['point']:+.4f} [{x['ci95'][0]:+.4f}, {x['ci95'][1]:+.4f}]"
for k, v in A["bpb_differences"].items():
    if v and ("drop" in k or "occ" in k or "U64" in k):
        print(f"{k:34s} {ci(v)}")
"""),
        md("## 3. Quality versus minting cost, every arm"),
        code("""
T = V["quality_vs_minting"]["table"]
print(f"{'arm':16s} {'rule':14s} {'bpb':>8s} {'net x1e3':>9s}")
for arm, row in sorted(T.items(), key=lambda kv: kv[1]["in_vocab_test_bpb"]):
    print(f"{arm:16s} {row['rule_by_net_dev']:14s} {row['in_vocab_test_bpb']:8.4f} {row['net_minting_test_bpb']*1e3:+9.3f}")
print("non-dominated:", V["quality_vs_minting"]["non_dominated"])
print("dominance:", V["quality_vs_minting"]["dominance"])
"""),
        md("## 4. Figure 8"),
        code("""
subprocess.run([sys.executable, str(ROOT / "scripts" / "phase2_figures.py")], cwd=ROOT, check=True)
from IPython.display import Image, display
display(Image(filename=str(ROOT / "figures" / "fig8_phase2_gap_and_frontier.png"), width=800))
"""),
    ]
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    return nb


def main() -> None:
    out = ROOT / "notebooks"
    nb = notebook()
    path = out / "02_phase2_from_results.ipynb"
    nbf.write(nb, path)
    print("wrote", path)
    if "--execute" in sys.argv:
        from nbclient import NotebookClient
        client = NotebookClient(nb, timeout=1800, kernel_name="python3", resources={"metadata": {"path": str(out)}})
        client.execute()
        nbf.write(nb, path)
        print("executed", path)


if __name__ == "__main__":
    main()
