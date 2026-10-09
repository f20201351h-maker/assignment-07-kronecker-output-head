"""Stratify the per-context KAS floors and the arms' realised gaps by the target token's training count.

Uses the per-context arrays saved by floor_contextual.py (exact KAS family, bias = Dense-prior's beta, tag index 0).

    python scripts/expressivity/floor_by_frequency.py
      -> results/expressivity/floor_by_frequency.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from floor_lib import ROOT  # noqa: E402

EV = ROOT / "results/expressivity"
EDGES = [0, 10, 100, 1000, 10_000, 100_000, 1_000_000, 10 ** 12]


def main():
    npz = np.load(EV / "floor_contextual.npz", allow_pickle=True)
    uni = np.load(ROOT / "data/tokens/unigram_train.npy")
    y, valid = npz["y"], npz["valid"]
    fl = npz["floor_0"]
    mimic = npz["mimic_0"]
    ref = npz["ref_nll"]
    arms = {k[4:]: npz[k] for k in npz.files if k.startswith("arm_")}
    cnt = uni[y]
    out = {"provenance": "RECOMPUTED from floor_contextual.npz (tag 0: KAS features, bias = Dense-prior beta) and "
                         "data/tokens/unigram_train.npy; buckets by training count of the TARGET token",
           "tag": str(npz["tags"][0]), "buckets": []}
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        m = valid & (cnt >= lo) & (cnt < hi)
        if m.sum() == 0:
            continue
        row = {"range": f"[{lo},{hi})" if hi < 10 ** 12 else f"[{lo},inf)", "n_targets": int(m.sum()),
               "share_of_targets": float(m.mean() / valid.mean()),
               "floor_nats_mean": float(fl[m].mean()), "floor_nats_median": float(np.median(fl[m])),
               "mimic_minus_dense_prior": float((mimic[m] - ref[m]).mean()),
               "arms_gap_to_dense_prior": {k: float((v[m] - arms["Dense-prior"][m]).mean()) for k, v in arms.items()
                                           if k != "Dense-prior"}}
        out["buckets"].append(row)
    out["overall"] = {"floor_nats_mean": float(fl[valid].mean()), "n_targets": int(valid.sum())}
    (EV / "floor_by_frequency.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"{'count bucket':>18} {'n':>5} {'floor':>7} {'mimic-DP':>9} {'KAS-P':>7} {'KAS-U16':>8} {'KAS-U64':>8} {'Dense':>7}")
    for r in out["buckets"]:
        g = r["arms_gap_to_dense_prior"]
        print(f"{r['range']:>18} {r['n_targets']:>5} {r['floor_nats_mean']:>7.3f} {r['mimic_minus_dense_prior']:>+9.3f} "
              f"{g['KAS-P']:>+7.3f} {g['KAS-U16']:>+8.3f} {g['KAS-U64']:>+8.3f} {g['Dense']:>+7.3f}")
    print("wrote", EV / "floor_by_frequency.json")


if __name__ == "__main__":
    main()
