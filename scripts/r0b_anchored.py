"""Optional R0b extension: a linear code anchored at BOTH ends of the token.

The Kronecker code places byte b at its distance from the token's start, so a suffix such as
"ived" lands in different columns in "Archived" and "derived". Adding a second one-hot block
indexed by distance from the END lets a purely linear readout share suffix features. If this
closes most of the gap between the start-anchored linear code and the ByteCNN in R0b, the
generator's advantage is "seeing suffixes", not nonlinearity.

Same targets, split and ridge protocol as scripts/r0b_predictability.py.
    python scripts/r0b_anchored.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from kq5.codec import CodecTable  # noqa: E402
from kq5.vocab import WorkingVocab  # noqa: E402
from r0b_predictability import load_wte, r2  # noqa: E402


def sparse_design(table: CodecTable, rows: np.ndarray, both: bool) -> torch.Tensor:
    """Rows x features (float64 sparse): start one-hot * A, [end one-hot * A], B, 1."""
    D = table.D
    P = table.pos_dim
    r_idx, c_idx, vals = [], [], []
    for k, v in enumerate(rows):
        L = int(table.lengths[v])
        bts = table.byte_buffer[v, :L].astype(np.int64)
        a = table.A[v]
        cols = bts * P + np.arange(L)
        r_idx += [k] * L; c_idx += cols.tolist(); vals += [a] * L
        if both:
            cols_e = D + bts * P + (L - 1 - np.arange(L))
            r_idx += [k] * L; c_idx += cols_e.tolist(); vals += [a] * L
    F = 2 * D if both else D
    r_idx += list(range(len(rows))) * 2
    c_idx += [F] * len(rows) + [F + 1] * len(rows)
    vals += table.B[rows].tolist() + [1.0] * len(rows)
    return torch.sparse_coo_tensor(torch.tensor([r_idx, c_idx]), torch.tensor(vals, dtype=torch.float64),
                                   (len(rows), F + 2)).coalesce()


def ridge_sparse(table, tr, va, te, Ytr, Yva, Yte, both, lambdas=(1, 10, 100, 1000)):
    Xtr, Xva, Xte = (sparse_design(table, r, both) for r in (tr, va, te))
    XtX = torch.sparse.mm(Xtr.t().coalesce(), Xtr).to_dense().double()   # sparse x sparse, no dense X
    XtY = torch.sparse.mm(Xtr.t(), Ytr)
    I = torch.eye(XtX.shape[0], dtype=torch.float64)
    best = None
    for lam in lambdas:
        W = torch.linalg.solve(XtX + lam * I, XtY)
        s = r2(torch.sparse.mm(Xva, W).double(), Yva)
        if best is None or s > best[0]:
            best = (s, lam, W)
    _, lam, W = best
    return {"lambda": lam, "r2_test": r2(torch.sparse.mm(Xte, W).double(), Yte),
            "r2_train": r2(torch.sparse.mm(Xtr, W).double(), Ytr), "features": int(XtX.shape[0])}


def run(name, table, ids, Y, seed=0):
    rng = np.random.default_rng(seed)
    n = len(ids)
    perm = rng.permutation(n)
    n_te = n // 10
    te, rest = perm[:n_te], perm[n_te:]
    va, tr = rest[: len(rest) // 10], rest[len(rest) // 10:]     # identical split to r0b_predictability
    Yt = torch.as_tensor(Y, dtype=torch.float64)
    mu, sd = Yt[tr].mean(0), Yt[tr].std(0) + 1e-8
    Ys = (Yt - mu) / sd
    out = {"target": name}
    for both in (False, True):
        t0 = time.time()
        out["linear_start_only" if not both else "linear_both_ends"] = ridge_sparse(
            table, ids[tr], ids[va], ids[te], Ys[tr], Ys[va], Ys[te], both)
        print(name, both, out, f"{time.time() - t0:.0f}s", flush=True)
    return out


def main() -> None:
    torch.set_num_threads(8)
    tok = ROOT / "data" / "tokens"
    vocab = WorkingVocab.load(tok / "vocab.json")
    unigram = np.load(tok / "unigram_train.npy")
    V = vocab.n_train_vocab
    table = CodecTable.from_bytes(vocab.tokens[:V], pos_dim=vocab.pos_dim)
    res = {}
    ids_a = np.array([i for i in range(V) if i != vocab.eos_id])
    res["log_frequency"] = run("log_frequency", table, ids_a, np.log1p(unigram[ids_a].astype(np.float64))[:, None])
    wte = load_wte(ROOT / "data" / "gpt2" / "model.safetensors")
    gpt2_of = {}
    for g, e in enumerate(vocab.gpt2_to_work):
        if len(e) == 1:
            gpt2_of.setdefault(e[0], g)
    ids_b = np.array([i for i in range(vocab.n_base) if i != vocab.eos_id and i in gpt2_of])
    res["gpt2_wte"] = run("gpt2_wte", table, ids_b, wte[[gpt2_of[i] for i in ids_b]].astype(np.float64))
    out = ROOT / "results" / "r0b" / "r0b_anchored.json"
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
