"""R0b - no LM training. How well do a token's bytes predict, on held-out tokens,
(a) its log training frequency and (b) its row of public GPT-2's (tied) output
embedding? Linear-in-the-code ridge vs the ByteCNN generator vs shuffled spellings.

The linear model is exactly what a KAS head can express (E_v = kappa_v^T W).
"lin+cnn" fits the CNN to the ridge residual: the part a generated correction
would have to supply on top of KAS, which is what KAS-G does.

    python scripts/r0b_predictability.py
"""

from __future__ import annotations

import json
import math
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kq5.codec import CodecTable  # noqa: E402
from kq5.generator import ByteCNN, GeneratorConfig  # noqa: E402
from kq5.heads import derangement  # noqa: E402
from kq5.vocab import WorkingVocab  # noqa: E402

OUT = ROOT / "results" / "r0b"


def load_wte(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        meta = header["wte.weight"]
        start, end = meta["data_offsets"]
        f.seek(8 + n + start)
        buf = f.read(end - start)
    assert meta["dtype"] == "F32"
    return np.frombuffer(buf, dtype=np.float32).reshape(meta["shape"]).copy()


def features_gram(table: CodecTable, rows: np.ndarray):
    """X^T X and a function computing X^T Y for X = [A*onehot(S), B, 1] restricted to rows."""
    D = table.D
    P = D + 2
    G = torch.zeros(P, P, dtype=torch.float64)
    idx = table.flat_index[rows]
    val = np.where(table.valid[rows], table.A[rows, None], 0.0)
    B = table.B[rows]
    for r in range(len(rows)):
        m = table.valid[rows[r]]
        ii = idx[r][m]
        a = val[r][m]
        full_i = np.concatenate([ii, [D, D + 1]])
        full_v = np.concatenate([a, [B[r], 1.0]])
        G[np.ix_(full_i, full_i)] += torch.as_tensor(np.outer(full_v, full_v))
    return G


def design_apply(table: CodecTable, rows: np.ndarray, W: torch.Tensor, chunk: int = 1024) -> torch.Tensor:
    """X[rows] @ W for W of shape (D+2, k), in row chunks (memory O(chunk * 32 * k))."""
    D = table.D
    outs = []
    for s in range(0, len(rows), chunk):
        r = rows[s:s + chunk]
        idx = torch.as_tensor(table.flat_index[r])
        val = torch.as_tensor(np.where(table.valid[r], table.A[r, None], 0.0))
        o = torch.einsum("npk,np->nk", W[:D][idx], val)
        outs.append(o + torch.as_tensor(table.B[r])[:, None] * W[D] + W[D + 1])
    return torch.cat(outs)


def xty(table: CodecTable, rows: np.ndarray, Y: torch.Tensor) -> torch.Tensor:
    D = table.D
    out = torch.zeros(D + 2, Y.shape[1], dtype=torch.float64)
    for s in range(0, len(rows), 1024):
        r = rows[s:s + 1024]
        idx = torch.as_tensor(table.flat_index[r])
        val = torch.as_tensor(np.where(table.valid[r], table.A[r, None], 0.0))
        contrib = val[..., None] * Y[s:s + 1024, None, :]
        out.index_add_(0, idx.reshape(-1), contrib.reshape(-1, Y.shape[1]))
    out[D] = (torch.as_tensor(table.B[rows])[:, None] * Y).sum(0)
    out[D + 1] = Y.sum(0)
    return out


def r2(pred: torch.Tensor, y: torch.Tensor, w: np.ndarray | None = None) -> float:
    if w is None:
        w = np.ones(len(y))
    w = torch.as_tensor(w, dtype=torch.float64)[:, None]
    mu = (w * y).sum(0) / w.sum()
    sse = (w * (y - pred) ** 2).sum()
    sst = (w * (y - mu) ** 2).sum()
    return float(1 - sse / sst)


def ridge(table, tr, va, Ytr, Yva, lambdas=(1e-2, 1e-1, 1, 10, 100, 1000)):
    G = features_gram(table, tr)
    b = xty(table, tr, Ytr)
    best = None
    I = torch.eye(G.shape[0], dtype=torch.float64)
    for lam in lambdas:
        W = torch.linalg.solve(G + lam * I, b)
        s = r2(design_apply(table, va, W), Yva)
        if best is None or s > best[0]:
            best = (s, lam, W)
    return best


def train_cnn(byte_src: CodecTable, tr, va, Ytr, Yva, epochs=25, seed=0):
    torch.manual_seed(seed)
    cfg = GeneratorConfig()
    net = ByteCNN(Ytr.shape[1], cfg)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=0.0)
    bb = torch.as_tensor(byte_src.byte_buffer).long()
    ll = torch.as_tensor(byte_src.lengths).long()
    Ytr_f, Yva_f = Ytr.float(), Yva.float()
    best = (-1e9, None)
    rng = np.random.default_rng(seed)
    for ep in range(epochs):
        net.train()
        perm = rng.permutation(len(tr))
        for s in range(0, len(tr), 512):
            i = perm[s:s + 512]
            rows = torch.as_tensor(tr[i])
            loss = ((net(bb[rows], ll[rows]) - Ytr_f[i]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            pv = torch.cat([net(bb[torch.as_tensor(va[s:s + 4096])], ll[torch.as_tensor(va[s:s + 4096])])
                            for s in range(0, len(va), 4096)])
        sc = r2(pv.double(), Yva)
        if sc > best[0]:
            best = (sc, {k: v.clone() for k, v in net.state_dict().items()})
    net.load_state_dict(best[1])
    net.eval()
    return net, sum(p.numel() for p in net.parameters())


def predict(net, byte_src: CodecTable, rows):
    bb = torch.as_tensor(byte_src.byte_buffer).long()
    ll = torch.as_tensor(byte_src.lengths).long()
    with torch.no_grad():
        return torch.cat([net(bb[torch.as_tensor(rows[s:s + 4096])], ll[torch.as_tensor(rows[s:s + 4096])])
                          for s in range(0, len(rows), 4096)]).double()


def study(name, table, tokens_ids, Y, weights, rng_seed=0, epochs=25):
    rng = np.random.default_rng(rng_seed)
    n = len(tokens_ids)
    perm = rng.permutation(n)
    n_te = n // 10
    te, rest = perm[:n_te], perm[n_te:]
    va, tr = rest[: len(rest) // 10], rest[len(rest) // 10:]
    Yt = torch.as_tensor(Y, dtype=torch.float64)
    mu, sd = Yt[tr].mean(0), Yt[tr].std(0) + 1e-8
    Ys = (Yt - mu) / sd
    shuf = derangement(len(table.lengths), 4242)
    shuf_table = CodecTable(table.byte_buffer[shuf], table.lengths[shuf], pos_dim=table.pos_dim)
    res = {"target": name, "n_tokens": n, "n_train": len(tr), "n_val": len(va), "n_test": len(te)}
    ids = np.asarray(tokens_ids)
    for tag, tab in (("true", table), ("shuffled", shuf_table)):
        t0 = time.time()
        s_va, lam, W = ridge(tab, ids[tr], ids[va], Ys[tr], Ys[va])
        lin_te = design_apply(tab, ids[te], W)
        lin_tr = design_apply(tab, ids[tr], W)
        lin_va = design_apply(tab, ids[va], W)
        res[f"linear_{tag}"] = {"lambda": lam, "r2_test": r2(lin_te, Ys[te]),
                                "r2_test_weighted": r2(lin_te, Ys[te], weights[te]),
                                "r2_train": r2(lin_tr, Ys[tr]), "params": int(W.numel())}
        net, npar = train_cnn(tab, ids[tr], ids[va], Ys[tr], Ys[va], epochs=epochs)
        cnn_te = predict(net, tab, ids[te])
        res[f"cnn_{tag}"] = {"r2_test": r2(cnn_te, Ys[te]), "r2_test_weighted": r2(cnn_te, Ys[te], weights[te]),
                             "r2_train": r2(predict(net, tab, ids[tr]), Ys[tr]), "params": npar}
        rnet, _ = train_cnn(tab, ids[tr], ids[va], Ys[tr] - lin_tr, Ys[va] - lin_va, epochs=epochs)
        comb_te = lin_te + predict(rnet, tab, ids[te])
        comb_tr = lin_tr + predict(rnet, tab, ids[tr])
        res[f"linear_plus_cnn_{tag}"] = {"r2_test": r2(comb_te, Ys[te]),
                                         "r2_test_weighted": r2(comb_te, Ys[te], weights[te]),
                                         "r2_train": r2(comb_tr, Ys[tr])}
        print(name, tag, json.dumps({k: v for k, v in res.items() if k.endswith(tag)}), f"{time.time()-t0:.0f}s",
              flush=True)
    return res


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    tok_dir = ROOT / "data" / "tokens"
    vocab = WorkingVocab.load(tok_dir / "vocab.json")
    unigram = np.load(tok_dir / "unigram_train.npy")
    V = vocab.n_train_vocab
    table = CodecTable.from_bytes(vocab.tokens[:V], pos_dim=vocab.pos_dim)
    results = {"created": time.strftime("%Y-%m-%dT%H:%M:%S")}
    # (a) log frequency over the training vocabulary (base + trained merges), EOS excluded
    part_a = OUT / "part_log_frequency.json"
    if part_a.exists():
        results["log_frequency"] = json.loads(part_a.read_text(encoding="utf-8"))
    else:
        ids_a = np.array([i for i in range(V) if i != vocab.eos_id])
        y_a = np.log1p(unigram[ids_a].astype(np.float64))[:, None]
        results["log_frequency"] = study("log_frequency", table, ids_a, y_a, unigram[ids_a] + 1.0)
        part_a.write_text(json.dumps(results["log_frequency"], indent=2), encoding="utf-8")
    # (b) public GPT-2 output-embedding rows, base tokens only (merges have no GPT-2 row)
    wte = load_wte(ROOT / "data" / "gpt2" / "model.safetensors")
    gpt2_of = {}
    for g, e in enumerate(vocab.gpt2_to_work):
        if len(e) == 1 and len(vocab.tokens[e[0]]) == len(vocab.tokens[e[0]]):
            gpt2_of.setdefault(e[0], g)
    ids_b = np.array([i for i in range(vocab.n_base) if i != vocab.eos_id and i in gpt2_of])
    y_b = wte[[gpt2_of[i] for i in ids_b]].astype(np.float64)
    results["gpt2_wte"] = study("gpt2_wte", table, ids_b, y_b, unigram[ids_b] + 1.0)
    (OUT / "part_gpt2_wte.json").write_text(json.dumps(results["gpt2_wte"], indent=2), encoding="utf-8")
    (OUT / "r0b.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
