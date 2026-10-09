"""Build (and optionally execute) the analysis notebook from saved results.

    python scripts/build_notebooks.py [--execute]
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


def analysis_notebook() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md("""
# Phase 1 analysis from saved results

Recomputes every verdict, table and figure in the README from the files under `experiments/`
(run outputs downloaded from the training volume: per-position NLL, minting records, evaluation curves) and
`results/`. No GPU and no training: this notebook only reads saved results.

Inputs: `experiments/modal-m3/runs/*` (primary canonical set, Protocol 05), `results/r0b/`, `results/growth/`,
`results/headcost/`, `results/probe/`, `data/tokens/` (token arrays for byte counts).
"""),
        code("""
import json, subprocess, sys
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))
RUNS = [str(ROOT / "experiments" / "modal-m3" / "runs")]
OUT = ROOT / "results" / "final"
"""),
        md("## 1. Recompute the verdicts (two-level paired bootstrap, Protocol 03)"),
        code("""
r = subprocess.run([sys.executable, str(ROOT / "scripts" / "final_analysis.py"), "--runs", *RUNS,
                    "--prefix", "m3", "--split", "test", "--out", str(OUT)], capture_output=True, text=True, cwd=ROOT)
print(r.stdout[-3000:] or r.stderr[-3000:])
A = json.load(open(OUT / "analysis_m3_test.json"))
"""),
        md("## 2. Bits per byte on the test split (seed means and spread)"),
        code("""
for arm, v in sorted(A["arms"].items(), key=lambda kv: kv[1]["bpb_mean"]):
    print(f"{arm:12s} bpb {v['bpb_mean']:.4f}  seeds {['%.4f' % x for x in v['bpb_per_seed']]}  "
          f"spread {v['seed_spread'] if v['seed_spread'] is None else round(v['seed_spread'], 4)}  "
          f"head params {v['head_params']:,}  V-dependent {v['head_V_dependent']:,}")
print("fairness:", A["fairness"])
"""),
        md("## 3. Hypotheses"),
        code("""
for h in ("H1", "H2", "H3", "H4"):
    v = A.get(h, {})
    print(h, "->", v.get("verdict"))
print(json.dumps({k: A.get("H1", {}).get(k) for k in ("rho", "denominator_P_minus_U", "outcome_band")}, indent=1))
print(json.dumps({k: A.get("H2", {}).get(k) for k in ("kasu_rule_selected_on_dev", "kasg_rule_selected_on_dev",
                                                       "delta_regret_bits", "delta_leak_bpb")}, indent=1))
print(json.dumps({k: A.get("H4", {}).get(k) for k in ("delta_shuf_minus_kasu", "delta_spell_kasg_minus_shuf")}, indent=1))
"""),
        md("## 4. Minting table (test; mean regret in bits per site, beats-prefix, median rank, leak)"),
        code("""
t = A["minting_table"]["test"]
for arm, rules in t.items():
    print(arm, "sites:", rules.get("_n_sites"))
    for name, v in rules.items():
        if name.startswith("_"):
            continue
        print(f"   {name:14s} regret {v['mean_regret_bits']:8.3f}  beats {v['beats_prefix']:.3f}  "
              f"median rank {v['median_rank']:8.0f}  leak {v['leak_bpb']:.5f}")
"""),
        md("## 5. Where the gap lives: NLL(Dense) − NLL(KAS) by training frequency"),
        code("""
for arm, rows in A.get("H3", {}).get("buckets", {}).items():
    print(arm)
    for r in rows:
        if r.get("n_targets"):
            print(f"   {r['bucket']:16s} n={r['n_targets']:8d}  d={r['d_nats']:+.4f}  CI {r['ci95'][0]:+.4f} .. {r['ci95'][1]:+.4f}")
"""),
        md("## 6. Figures"),
        code("""
subprocess.run([sys.executable, str(ROOT / "scripts" / "make_figures.py"), "--analysis", str(OUT / "analysis_m3_test.json")], cwd=ROOT, check=True)
from IPython.display import Image, display
for f in sorted((ROOT / "figures").glob("fig*.png")):
    print(f.name); display(Image(filename=str(f), width=700))
"""),
    ]
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    return nb


def main() -> None:
    out = ROOT / "notebooks"
    out.mkdir(exist_ok=True)
    nb = analysis_notebook()
    path = out / "01_analysis_from_results.ipynb"
    nbf.write(nb, path)
    print("wrote", path)
    if "--execute" in sys.argv:
        from nbclient import NotebookClient
        client = NotebookClient(nb, timeout=1800, kernel_name="python3",
                                resources={"metadata": {"path": str(out)}})
        client.execute()
        nbf.write(nb, path)
        print("executed", path)


if __name__ == "__main__":
    main()
