"""Gather per-run diagnostics (correction ablations, logit-term shares, counterfactual spelling,
prior drift) into one results file with seed means, so README numbers trace to results/.

    python scripts/collect_diagnostics.py experiments/modal-m3/runs test results/final/diagnostics_m3_test.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    root, split, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    res = {"split": split, "runs": {}, "arm_means": {}}
    arm_of = {"kasu0": "KAS-U16", "kasg": "KAS-G", "kasgshuf": "KAS-G-shuf", "kasu": "KAS-U16 (c_zero)",
              "kasp": "KAS-P", "kas0": "KAS-0", "dense": "Dense"}
    per_arm: dict[str, list] = {}
    for d in sorted(root.glob("*")):
        a = d / f"analysis_{split}"
        if not a.exists():
            continue
        r = {}
        for name in ("ablations", "terms", "counterfactual", "prior_drift"):
            p = a / f"{name}.json"
            if p.exists():
                r[name] = json.loads(p.read_text(encoding="utf-8"))
        if "ablations" in r and r["ablations"]:
            r["correction_value_bpb"] = r["ablations"]["bpb_no_correction"] - r["ablations"]["bpb_full"]
        res["runs"][d.name] = r
        per_arm.setdefault(arm_of[d.name.split("-")[1]], []).append(r)
    for arm, rs in per_arm.items():
        m = {}
        vals = [r.get("correction_value_bpb") for r in rs if r.get("correction_value_bpb") is not None]
        if vals:
            m["correction_value_bpb"] = float(np.mean(vals))
        shares = [r["terms"].get("share_correction") for r in rs if r.get("terms")]
        shares = [x for x in shares if x is not None]
        if shares:
            m["share_correction"] = float(np.mean(shares))
        cf = [r["counterfactual"] for r in rs if r.get("counterfactual")]
        if cf:
            for rule in cf[0]:
                m[f"cf_{rule}_true"] = float(np.mean([c[rule]["mean_regret_bits_true"] for c in cf]))
                m[f"cf_{rule}_counterfactual"] = float(np.mean([c[rule]["mean_regret_bits_counterfactual"] for c in cf]))
        dr = [r["prior_drift"]["overall_mean_drift"] for r in rs if r.get("prior_drift")]
        if dr:
            m["prior_drift_overall"] = float(np.mean(dr))
        res["arm_means"][arm] = m
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res["arm_means"], indent=1))


if __name__ == "__main__":
    main()
