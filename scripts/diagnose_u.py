"""Diagnosis of a stored-correction checkpoint (amendment 03b evidence): how far U moved from its
seeded initialisation, by training-count bin, plus ‖C‖ and the correction's ablation value.

    python scripts/diagnose_u.py experiments/kq5-r2-pilot/runs/r2-kasu-s1 results/r2/u_diagnosis_r2-kasu-s1.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kq5.train import RunConfig, build_model  # noqa: E402
from kq5.vocab import WorkingVocab  # noqa: E402


def main() -> None:
    run, out = Path(sys.argv[1]), Path(sys.argv[2])
    ck = torch.load(run / "model.pt", map_location="cpu", weights_only=False)
    cfg = RunConfig(**ck["config"])
    vocab = WorkingVocab.load(ROOT / "data" / "tokens" / "vocab.json")
    unigram = np.load(ROOT / "data" / "tokens" / "unigram_train.npy")
    init, _ = build_model(cfg, vocab, unigram)
    U, U0 = ck["model"]["head.U"].double(), init.state_dict()["head.U"].double()
    C = ck["model"]["head.C"].double()
    c = unigram[: vocab.n_train_vocab]
    edges = [0, 1, 10, 100, 1_000, 10_000, 100_000, 1_000_000, np.inf]
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (c >= lo) & (c < hi)
        if m.sum():
            rel = ((U - U0)[m].norm(dim=1) / U0[m].norm(dim=1).clamp_min(1e-12)).mean().item()
            rows.append({"count_bin": [lo, None if np.isinf(hi) else hi], "n": int(m.sum()), "mean_rel_move": rel})
    res = {"run": cfg.run_id, "corr_init": cfg.corr_init,
           "corr_U_U0": float(np.corrcoef(U.numpy().ravel(), U0.numpy().ravel())[0, 1]),
           "C_frobenius": float(C.norm()), "movement_by_count_bin": rows}
    abl = run / "analysis_dev" / "ablations.json"
    if abl.exists():
        a = json.loads(abl.read_text(encoding="utf-8"))
        res["ablation_dev"] = a
        res["correction_value_bpb"] = a["bpb_no_correction"] - a["bpb_full"]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "movement_by_count_bin"}, indent=1))


if __name__ == "__main__":
    main()
