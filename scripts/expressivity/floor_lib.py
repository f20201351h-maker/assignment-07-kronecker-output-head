"""Shared code for the teacher-projection (fixed-feature family) measurements (CPU only).

A *fixed-feature log-linear head* assigns token v the logit

    z_v(x) = sum_f phi_v[f] * s_f(x) + b_v

where phi_v is a FIXED feature vector of the token (computed from its bytes, or
from its identity), s(x) is any context-dependent parameter vector produced by
the body, and b_v is an optional per-token base measure (bias). The KAS /
additive-softmax head is the case phi_v = kappa_v (Kronecker occupancy cells,
plus the codec's length-dependent constants and a length bias).

For a target distribution p_x over the vocabulary, the best this family can do
is  min_s  CE(p_x, softmax(Phi s + b)).  Because s is free, the minimum is a
lower bound on the cross-entropy of ANY model of this form -- any width, any
body, any nonlinearity between the hidden state and s -- if p_x were the truth.
We call  min_s CE(p_x, softmax(Phi s + b)) - H(p_x)  the *floor* (nats/target).

Everything here is sparse: a token's features are a padded list of
(feature id, weight) pairs, and logits for all V tokens and N contexts are one
embedding_bag call with the parameter matrix S (F x N) as the "embedding".
"""

from __future__ import annotations

import hashlib
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from kq5.codec import CodecTable  # noqa: E402
from kq5.heads import centred_log_unigram  # noqa: E402
from kq5.vocab import WorkingVocab  # noqa: E402

LN2 = math.log(2.0)
# OBSERVED (results/phase4/independent_diagnostics.json): test split has 1,130,301
# positive-byte targets and 5,310,205 bytes. nats/target -> bits/byte divides by this.
BYTES_PER_TARGET = 5310205 / 1130301
NATS_PER_TARGET_TO_BPB = 1.0 / (LN2 * BYTES_PER_TARGET)


def load_vocab():
    vocab = WorkingVocab.load(ROOT / "data/tokens/vocab.json")
    V = vocab.n_train_vocab
    table = CodecTable.from_bytes(vocab.tokens[:V], pos_dim=vocab.pos_dim)
    unigram = np.load(ROOT / "data/tokens/unigram_train.npy")[:V].astype(np.float64)
    return vocab, table, unigram


def stable_hash(s: bytes, mod: int) -> int:
    return int.from_bytes(hashlib.blake2b(s, digest_size=8).digest(), "little") % mod


class FeatureSet:
    """Sparse fixed features for every token: lists of (feature id, weight)."""

    def __init__(self, V: int):
        self.V = V
        self.feats: list[list[tuple[int, float]]] = [[] for _ in range(V)]
        self.n_features = 1          # feature 0 is the padding dummy (weight 0)
        self.families: dict[str, tuple[int, int]] = {}

    def add_family(self, name: str, per_token: list[list[tuple[int, float]]], n_local: int) -> None:
        """per_token[v] = list of (local id in [0, n_local), weight)."""
        off = self.n_features
        for v in range(self.V):
            self.feats[v].extend((off + i, w) for i, w in per_token[v])
        self.families[name] = (off, n_local)
        self.n_features += n_local

    def tensors(self):
        maxf = max(len(f) for f in self.feats)
        idx = np.zeros((self.V, maxf), dtype=np.int64)
        w = np.zeros((self.V, maxf), dtype=np.float32)
        for v, f in enumerate(self.feats):
            if f:
                idx[v, :len(f)] = [i for i, _ in f]
                w[v, :len(f)] = [x for _, x in f]
        return torch.as_tensor(idx), torch.as_tensor(w)

    def rank(self) -> int:
        """Exact rank of the V x F feature matrix (Gram eigenvalues)."""
        idx, w = self.tensors()
        Fn = self.n_features
        M = torch.zeros(self.V, Fn, dtype=torch.float64)
        M.scatter_add_(1, idx, w.double())
        M = M[:, 1:]
        G = M.t() @ M
        ev = torch.linalg.eigvalsh(G)
        tol = ev.max() * max(G.shape) * 1e-15
        return int((ev > tol).sum())


# ----------------------------------------------------------------------------- feature families
def fam_kas(table: CodecTable) -> tuple[list, int]:
    """Kronecker occupancy cells + the codec's rank-1 B term + length one-hot (len_bias): exactly the KAS family.

    kappa_v = A_v 1[S_v] + B_v 1, so the KAS logit is A_v * sum_{cells in v} s[cell] + B_v * s_B + len_bias[L_v].
    The cell weight A_v and the B weight B_v are kept up to one global constant each (absorbed by s); dropping the
    length-dependent gain A_v (as a first version of this file did) changes the family: the tied-projection rows
    P kappa_v, a strict subfamily of KAS, then reached a lower floor than the "KAS" fit."""
    occ = np.unique(table.flat_index[table.valid])
    cell_id = {int(c): i for i, c in enumerate(occ)}
    n_cells = len(occ)
    B_ID, L0 = n_cells, n_cells + 1
    A_ref = float(np.mean(table.A[table.lengths > 0]))
    B_ref = float(np.mean(np.abs(table.B[table.lengths > 0]))) or 1.0
    out = []
    for v in range(table.V):
        f = [(cell_id[int(c)], float(table.A[v] / A_ref)) for c in table.flat_index[v][table.valid[v]]]
        f.append((B_ID, float(table.B[v] / B_ref)))
        f.append((L0 + int(table.lengths[v]), 1.0))
        out.append(f)
    return out, L0 + table.pos_dim + 1


def fam_suffix_cells(table: CodecTable, tokens: list[bytes]) -> tuple[list, int]:
    """End-anchored Kronecker grid: (byte, position from the end) cells, same A_v scaling."""
    seen: dict[tuple[int, int], int] = {}
    A_ref = float(np.mean(table.A[table.lengths > 0]))
    out = []
    for v in range(table.V):
        b = tokens[v]
        f = []
        L = len(b)
        for j, byte in enumerate(b):
            key = (byte, L - 1 - j)
            if key not in seen:
                seen[key] = len(seen)
            f.append((seen[key], float(table.A[v] / A_ref)))
        out.append(f)
    return out, len(seen)


def fam_ngrams(table: CodecTable, tokens: list[bytes], n: int, positional: bool, m: int) -> tuple[list, int]:
    """Hashed byte n-grams, optionally position-indexed, m buckets, A_v weight."""
    A_ref = float(np.mean(table.A[table.lengths > 0]))
    out = []
    for v in range(table.V):
        b = tokens[v]
        f = []
        for j in range(len(b) - n + 1):
            key = b[j:j + n] + (bytes([j]) if positional else b"")
            f.append((stable_hash(key, m), float(table.A[v] / A_ref)))
        out.append(f)
    return out, m


def fam_prefix(tokens: list[bytes], V: int, min_len: int, max_len: int, min_share: int) -> tuple[list, int]:
    """Indicator cells for prefixes (length min_len..max_len) shared by >= min_share tokens."""
    from collections import Counter
    cnt = Counter()
    for v in range(V):
        b = tokens[v]
        for k in range(min_len, min(max_len, len(b)) + 1):
            cnt[b[:k]] += 1
    keep = {p: i for i, p in enumerate(sorted(p for p, c in cnt.items() if c >= min_share))}
    out = []
    for v in range(V):
        b = tokens[v]
        f = []
        for k in range(min_len, min(max_len, len(b)) + 1):
            if b[:k] in keep:
                f.append((keep[b[:k]], 1.0))
        out.append(f)
    return out, len(keep)


def fam_private(unigram: np.ndarray, V: int, top_n: int) -> tuple[list, int]:
    """One private indicator cell for each of the top_n most frequent tokens (adaptive-softmax head part)."""
    order = np.argsort(-unigram)[:top_n]
    slot = {int(v): i for i, v in enumerate(order)}
    out = [([(slot[v], 1.0)] if v in slot else []) for v in range(V)]
    return out, top_n


def fam_hash_signed(tokens: list[bytes], V: int, m: int, k: int) -> tuple[list, int]:
    """Bloom-style signed sparse hash code of the token's identity: k buckets in m, +-1/sqrt(k)."""
    out = []
    for v in range(V):
        f = []
        for r in range(k):
            h = stable_hash(tokens[v] + bytes([r, 0xFF]), 2 * m)
            f.append((h // 2, (1.0 if h % 2 == 0 else -1.0) / math.sqrt(k)))
        out.append(f)
    return out, m


# ----------------------------------------------------------------------------- fitting
def fit_floor(fs: FeatureSet, P: torch.Tensor, ctx_w: torch.Tensor, bias: torch.Tensor | None,
              free_bias: bool = False, steps: int = 400, lr: float = 0.05, seed: int = 0,
              log_every: int = 0, device="cpu", lbfgs: bool = False) -> dict:
    """min over S (F x N) [and optional global bias b (V)] of sum_x ctx_w[x] * CE(P[x], softmax(Phi S[:,x] + b)).

    P: (N, V) target distributions (rows sum to 1). ctx_w: (N,) weights summing to 1.
    Returns the floor = weighted CE* - weighted H(P), in nats per target, plus diagnostics.
    """
    torch.manual_seed(seed)
    idx, w = fs.tensors()
    idx, w = idx.to(device), w.to(device)
    N, V = P.shape
    P = P.to(device)
    ctx_w = ctx_w.to(device)
    H = -(P * torch.log(P.clamp_min(1e-30))).sum(1)                    # (N,)
    S = torch.zeros(fs.n_features, N, device=device, requires_grad=True)
    params = [S]
    b = None
    if free_bias:   # a learned per-token bias, initialised at the supplied prior (or zero); replaces the fixed base
        b = (bias.to(device).clone() if bias is not None else torch.zeros(V, device=device)).requires_grad_(True)
        params.append(b)
        bias = None
    opt = torch.optim.Adam(params, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr * 0.02)
    Pt = P.t().contiguous()                                              # (V, N)
    base = bias.to(device).unsqueeze(1) if bias is not None else None

    def logits():
        z = F.embedding_bag(idx, S, per_sample_weights=w, mode="sum")   # (V, N)
        if base is not None:
            z = z + base
        if b is not None:
            z = z + b.unsqueeze(1)
        return z

    t0 = time.time()
    hist = []
    if lbfgs:
        opt = torch.optim.LBFGS(params, lr=1.0, max_iter=steps, history_size=50, line_search_fn="strong_wolfe",
                                tolerance_grad=1e-9, tolerance_change=1e-12)

        def closure():
            opt.zero_grad(set_to_none=True)
            z = logits()
            ce = -(Pt * F.log_softmax(z, dim=0)).sum(0)
            loss = (ce * ctx_w).sum()
            loss.backward()
            return loss
        opt.step(closure)
        steps_run = opt.state[opt._params[0]].get("n_iter", steps)
    for it in range(0 if lbfgs else steps):
        opt.zero_grad(set_to_none=True)
        z = logits()
        ce = -(Pt * F.log_softmax(z, dim=0)).sum(0)                     # (N,)
        loss = (ce * ctx_w).sum()
        loss.backward()
        opt.step()
        sched.step()
        if log_every and (it % log_every == 0 or it == steps - 1):
            fl = float(((ce - H) * ctx_w).sum())
            hist.append((it, fl))
            print(f"    step {it:4d} floor {fl:.5f} nats  ({time.time()-t0:.0f}s)", flush=True)
    with torch.no_grad():
        z = logits()
        ce = -(Pt * F.log_softmax(z, dim=0)).sum(0)
        floor_per_ctx = (ce - H)
        floor = float((floor_per_ctx * ctx_w).sum())
    return {"floor_nats": floor, "floor_bpb_equiv": floor * NATS_PER_TARGET_TO_BPB,
            "H_nats": float((H * ctx_w).sum()), "ce_nats": float((ce * ctx_w).sum()),
            "n_features": fs.n_features - 1, "steps": steps, "lr": lr, "optimizer": "lbfgs" if lbfgs else "adam",
            "seconds": round(time.time() - t0, 1),
            "history": hist, "floor_per_ctx": floor_per_ctx.cpu().numpy(),
            "S": S.detach().cpu(), "b": (b.detach().cpu() if b is not None else None)}


def unigram_bias(unigram: np.ndarray, V: int) -> torch.Tensor:
    prior, _, _ = centred_log_unigram(unigram, V)
    return torch.as_tensor(prior, dtype=torch.float32)
