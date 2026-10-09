"""Static check (Phase 2, following Anusha Raju's output-rank observation): the rank of the Kronecker code matrix
K (V × 8192) for the working vocabulary. The KAS logit matrix K·W_out has rank ≤ min(rank K, d), so the
code's rank is a width-independent ceiling on what any byte-derived head of this family can express.

    python scripts/code_rank.py      # -> results/setup/code_rank_gpt2bpe.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from kq5.codec import CodecTable  # noqa: E402
from kq5.vocab import WorkingVocab  # noqa: E402


def main() -> None:
    t0 = time.time()
    vocab = WorkingVocab.load(ROOT / "data/tokens/vocab.json")
    V = vocab.n_train_vocab
    table = CodecTable.from_bytes(vocab.tokens[:V])
    occ = np.unique(table.flat_index[table.valid])             # (byte, position) cells used by some token
    # K = A_v·1[S_v] + B_v·1: every unoccupied column equals the vector B, so rank K = rank [K_occupied | B]
    K = table.dense(np.float64)[:, occ]
    M = np.concatenate([K, table.B[:, None]], 1)
    s = np.linalg.svd(M, compute_uv=False)
    tol = s.max() * max(M.shape) * np.finfo(float).eps
    rank = int((s > tol).sum())
    out = {"V": V, "D": table.D, "pos_dim": table.pos_dim, "occupied_cells": int(len(occ)),
           "rank_K_exact_svd": rank, "sigma_at_cutoff": [float(s[rank - 1]), float(s[rank])],
           "binds_at_d512": bool(rank < 512), "binds_at_d768": bool(rank < 768), "binds_at_d1536": bool(rank < 1536),
           "seconds": round(time.time() - t0, 1),
           "note": "rank of the full V x 8192 code matrix K (GPT-2 BPE working vocabulary, pos_dim 32) by SVD of "
                   "[K_occupied | B]; the KAS logit matrix K W_out has rank <= min(rank K, d). Anusha Raju's "
                   "Brahmic-only 40,000-word slice gave 603 at pos_dim 32."}
    (ROOT / "results/setup/code_rank_gpt2bpe.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
