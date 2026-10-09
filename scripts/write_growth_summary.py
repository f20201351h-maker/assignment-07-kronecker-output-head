"""Summarise results/growth/*.json (vocabulary growth toward 1M on dev, seed 1, 256 windows): per arm and rule,
bpb at K = 0 and at the largest K, their difference, and the leak at the largest K.

    python scripts/write_growth_summary.py     # -> results/phase2/growth_summary.json
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def rows_of(d):
    if isinstance(d, list):
        return d
    for v in d.values():
        if isinstance(v, list) and v and isinstance(v[0], dict) and "K" in v[0]:
            return v
    raise ValueError("no growth rows")


def main() -> None:
    out = {}
    for f in sorted(glob.glob(str(ROOT / "results/growth/*.json"))):
        name = Path(f).stem
        by = {}
        for r in rows_of(json.loads(Path(f).read_text(encoding="utf-8"))):
            by.setdefault(r["rule"], []).append(r)
        out[name] = {}
        for rule, rs in by.items():
            rs = sorted(rs, key=lambda r: r["K"])
            k0, km = rs[0], rs[-1]
            out[name][rule] = {"K_max": km["K"], "vocab_total": km["vocab_total"], "bpb_K0": k0["retok_bpb"],
                               "bpb_Kmax": km["retok_bpb"], "delta_bpb": km["retok_bpb"] - k0["retok_bpb"],
                               "leak_bpb_Kmax": km["leak_bpb"]}
    (ROOT / "results/phase2/growth_summary.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    for n, v in out.items():
        print(n, {r: (round(x["delta_bpb"], 4), round(x["leak_bpb_Kmax"], 4)) for r, x in v.items()})


if __name__ == "__main__":
    main()
