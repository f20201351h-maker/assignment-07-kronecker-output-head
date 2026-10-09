"""Derived numbers quoted in README §11 (Phase 2) that are not stored verbatim in a results file:
differences of stored values and ×10³ scalings. Written to results/phase2/readme_derived_phase2.json so
scripts/check_readme_numbers.py can trace them (same role as results/final/readme_derived.json for §5–6).

    python scripts/write_phase2_derived.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    A = json.loads((ROOT / "results/phase2/analysis_m3_p2_test.json").read_text(encoding="utf-8"))
    V = json.loads((ROOT / "results/phase2/phase2_verdicts.json").read_text(encoding="utf-8"))
    out = {"note": "derived from results/phase2/analysis_m3_p2_test.json and phase2_verdicts.json"}
    # ablation deltas (bpb without the correction / generated terms minus full)
    for arm, ab in A["extras"].items():
        for d in ab.get("ablations", []):
            for k, v in d.items():
                if k != "bpb_full":
                    out.setdefault("ablation_delta", {}).setdefault(arm, {}).setdefault(k, []).append(v - d["bpb_full"])
    for arm, dd in out.get("ablation_delta", {}).items():
        for k, vs in dd.items():
            dd[k] = {"per_seed": vs, "mean": sum(vs) / len(vs)}
    # seed-mean logit-term shares (Cov(term, z)/Var(z)) per arm
    for arm, ex in A["extras"].items():
        ts = ex.get("terms", [])
        if ts:
            out.setdefault("term_share_mean", {})[arm] = {k: sum(t[k] for t in ts) / len(ts) for k in ts[0] if k.startswith("share")}
    # leak ratios quoted in prose (KAS-U16 inherit over the dropout arm's inherit)
    mt = A["minting_table"]["test"]
    for arm in ("KAS-U16-dropU25", "KAS-U16-drop25"):
        if arm in mt:
            out.setdefault("leak_ratio_kasu_inherit_over", {})[arm] = mt["KAS-U16"]["inherit"]["leak_bpb"] / mt[arm]["inherit"]["leak_bpb"]
    # net minting ×10^3 for the frontier table and H5/H6 text
    q = V["quality_vs_minting"]["table"]
    out["net_minting_x1e3"] = {arm: {"mean": r["net_minting_test_bpb"] * 1e3,
                                     "per_seed": [x * 1e3 for x in r["net_minting_test_bpb_per_seed"]]}
                               for arm, r in q.items()}
    for arm, h in V["H5_dropout_minting"].items():
        n = h["net_retok"]
        out["net_minting_x1e3"].setdefault(f"{arm}__H5_rule_{n['rule_drop_by_net_dev']}", {})["per_seed"] = \
            [x * 1e3 for x in n["test_delta_drop_per_seed"]]
        out["net_minting_x1e3"].setdefault(f"KAS-U16__H5_rule_{n['rule_kasu_by_net_dev']}", {})["per_seed"] = \
            [x * 1e3 for x in n["test_delta_kasu_per_seed"]]
    # generator parameter counts
    g = A["arms"]
    if "KAS-G" in g and "KAS-G-occ" in g:
        out["generator_params"] = {"bytecnn_v3": 751_889, "occmlp_h92": 755_337,
                                   "head_param_difference": g["KAS-G-occ"]["head_params"] - g["KAS-G"]["head_params"]}
    # KAS-U64 head parameter ratio against Dense
    if "KAS-U64" in g:
        out["dense_over_kasu64_head_params"] = g["Dense"]["head_params"] / g["KAS-U64"]["head_params"]
    # H1-scale ratios already stored; sign-flipped differences used in prose
    for k, v in A["bpb_differences"].items():
        if v:
            out.setdefault("abs_bpb_differences", {})[k] = {"point": abs(v["point"]), "ci95": sorted(abs(x) for x in v["ci95"])}
    (ROOT / "results/phase2/readme_derived_phase2.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: (v if not isinstance(v, dict) else list(v)[:6]) for k, v in out.items()}, indent=1, default=str)[:1500])


if __name__ == "__main__":
    main()
