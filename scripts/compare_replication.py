"""Light replication check: Kaggle T4 (prefix r3) against the primary Modal set (prefix m3).

Same configurations and seeds on different hardware, so runs are not bitwise identical. Compares test
bpb run by run, the arm ordering, the key paired differences and the H1 point ratio (seed means only;
the full bootstrap analysis is on the primary set). Missing runs are listed, not imputed.

    python scripts/compare_replication.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
M3 = ROOT / "experiments" / "modal-m3" / "runs"
R3 = [ROOT / "experiments" / "kq5-r3-canonical-x" / "runs", ROOT / "experiments" / "kq5-r3-canonical-g" / "runs"]
ARMS = {"Dense": "dense", "KAS-0": "kas0", "KAS-P": "kasp", "KAS-U16": "kasu0", "KAS-G": "kasg",
        "KAS-G-shuf": "kasgshuf", "KAS-U16 (c_zero)": "kasu"}


def load(run_dir: Path):
    f = run_dir / "summary.json"
    if not f.exists():
        return None
    s = json.loads(f.read_text(encoding="utf-8"))
    if s.get("status") != "complete":
        return None
    return s


def main() -> None:
    rows, means = [], {"m3": {}, "r3": {}}
    for arm, key in ARMS.items():
        for prefix in ("m3", "r3"):
            vals = []
            for seed in (1, 2):
                rid = f"{prefix}-{key}-s{seed}"
                dirs = [M3 / rid] if prefix == "m3" else [d / rid for d in R3]
                s = next((x for x in map(load, dirs) if x is not None), None)
                if s is not None:
                    vals.append(s["final"]["test"]["bpb"])
                    if prefix == "r3":
                        m = load(M3 / f"m3-{key}-s{seed}")
                        rows.append({"arm": arm, "seed": seed, "t4_bpb": s["final"]["test"]["bpb"],
                                     "modal_bpb": m["final"]["test"]["bpb"],
                                     "diff": s["final"]["test"]["bpb"] - m["final"]["test"]["bpb"],
                                     "same_windows": s["final"]["test"]["windows_sha256"] == m["final"]["test"]["windows_sha256"],
                                     "steps": s["steps"], "tokens": s["tokens"]})
            if len(vals) == 2:
                means[prefix][arm] = sum(vals) / 2
    out = {"note": "light replication check, seed means only; primary analysis is on m3. Only training-side "
                   "test bpb is compared: the Kaggle X kernel ran its minting/growth analyses with code from before "
                   "the pre-canonical code review (protocols/03a_errata.md), so those outputs are not used.",
           "runs": rows, "seed_means": means,
           "missing_r3": [a for a in ARMS if a not in means["r3"]]}
    common = [a for a in ARMS if a in means["r3"]]
    out["ordering_m3"] = sorted(common, key=lambda a: means["m3"][a])
    out["ordering_r3"] = sorted(common, key=lambda a: means["r3"][a])
    out["ordering_agrees"] = out["ordering_m3"] == out["ordering_r3"]
    pairs = [("KAS-P", "KAS-U16"), ("KAS-P", "KAS-G"), ("KAS-G-shuf", "KAS-G"), ("KAS-U16", "Dense"),
             ("KAS-0", "KAS-P"), ("KAS-U16 (c_zero)", "KAS-U16"), ("KAS-P", "KAS-U16 (c_zero)"),
             ("KAS-U16 (c_zero)", "Dense"), ("KAS-P", "Dense")]
    out["paired"] = {f"{a} - {b}": {p: means[p][a] - means[p][b] for p in ("m3", "r3")}
                     for a, b in pairs if a in common and b in common}
    if all(a in common for a in ("KAS-P", "KAS-U16", "KAS-G")):
        out["H1_ratio"] = {p: (means[p]["KAS-P"] - means[p]["KAS-G"]) / (means[p]["KAS-P"] - means[p]["KAS-U16"])
                           for p in ("m3", "r3")}
    if rows:
        out["max_abs_run_diff"] = max(abs(r["diff"]) for r in rows)
    dst = ROOT / "results" / "replication" / "kaggle_vs_modal.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
