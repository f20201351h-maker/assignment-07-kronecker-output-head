"""Convergence check for the sparse KAS-family fit on the contextual target.

The dense row-set fits (floor_rows.py) converge fully in 200 LBFGS iterations (sanity row set reaches 0), and the
tied-projection family (a strict subfamily of KAS) reached 0.779 nats, below the 0.796 of the 200-iteration KAS fit:
the sparse 1,787-feature fit is therefore not fully converged at 200 iterations and its floors are upper bounds.
This script refits the first chunk (windows 0-3, 512 contexts) with 200 / 400 / 800 iterations.

    python scripts/expressivity/floor_convergence.py
      -> results/expressivity/floor_convergence.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from floor_lib import ROOT, NATS_PER_TARGET_TO_BPB, FeatureSet, fam_kas, fit_floor, load_vocab  # noqa: E402
from floor_contextual import hidden_and_logits, load_dense_prior  # noqa: E402
from kq5.data import eval_windows  # noqa: E402

OUT = ROOT / "results/expressivity/floor_convergence.json"


def main():
    torch.set_num_threads(8)
    vocab, table, unigram = load_vocab()
    V = vocab.n_train_vocab
    unigram_full = np.load(ROOT / "data/tokens/unigram_train.npy")
    dev = np.load(ROOT / "data/tokens/dev.npy", mmap_mode="r")
    windows_all = eval_windows(dev, 1024)
    win_idx = np.linspace(0, windows_all.shape[0] - 1, 8).round().astype(int)[:4]     # chunk 1 of floor_contextual
    positions = np.linspace(0, 1023, 128).round().astype(int)
    model, _ = load_dense_prior(vocab, unigram_full)
    _, Z, Y = hidden_and_logits(model, windows_all[win_idx], positions)
    P = F.softmax(Z, dim=1)
    nb = np.array([len(t) for t in vocab.tokens])[Y.numpy()]
    valid = nb > 0
    beta = model.head.beta.detach().float()
    fs = FeatureSet(V)
    fs.add_family("kas", *fam_kas(table))
    res = {"provenance": "RECOMPUTED; KAS-family floor on floor_contextual chunk 1 (512 contexts) vs LBFGS iterations",
           "n_contexts": int(len(Y)), "runs": {}}
    for iters in (200, 400, 800):
        t0 = time.time()
        r = fit_floor(fs, P, torch.full((len(Y),), 1.0 / len(Y)), beta, steps=iters, lbfgs=True)
        fl = r["floor_per_ctx"][valid].mean()
        res["runs"][str(iters)] = {"floor_nats": float(fl), "floor_bpb_equiv": float(fl * NATS_PER_TARGET_TO_BPB),
                                   "seconds": round(time.time() - t0, 1)}
        print(f"  iters {iters}: floor {fl:.4f} nats ({time.time()-t0:.0f}s)", flush=True)
        OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
