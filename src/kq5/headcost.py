"""Head cost versus vocabulary size (R4, should-have).

Forward + backward of the output head alone (hidden states in, loss and gradients out) at
V in {50K, 131K, 262K, 1M}, with a chunked-vocabulary cross-entropy that recomputes logits
in the backward pass, so neither head materialises the (BT x V) logits. The dense head
stores W (V x d); the KAS head stores only W_out (D x d) and builds each vocabulary chunk's
rows K_c W_out on the fly, so its parameter and optimiser memory do not depend on V. The
logit compute O(BT * V * d) is the same for both and is reported, not hidden.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from .codec import CodecTable, TorchCodec


def chunked_ce_step(h: torch.Tensor, y: torch.Tensor, rows, n_vocab: int, chunk: int) -> torch.Tensor:
    """Mean CE loss; accumulates grads into whatever `rows(lo, hi)` depends on and returns
    dL/dh. rows(lo, hi) must return E[lo:hi] (with autograd if the head has parameters)."""
    N = h.shape[0]
    hf = h.detach()
    acc = torch.float64 if h.dtype == torch.float64 else torch.float32
    m = torch.full((N,), -float("inf"), device=h.device, dtype=acc)
    s = torch.zeros(N, device=h.device, dtype=acc)
    zy = torch.zeros(N, device=h.device, dtype=acc)
    with torch.no_grad():
        for lo in range(0, n_vocab, chunk):
            hi = min(n_vocab, lo + chunk)
            with torch.autocast("cuda", dtype=torch.float16, enabled=h.is_cuda):
                z = (hf @ rows(lo, hi).t()).to(acc)
            mc = z.max(1).values
            new_m = torch.maximum(m, mc)
            s = s * torch.exp(m - new_m) + torch.exp(z - new_m[:, None]).sum(1)
            m = new_m
            inside = (y >= lo) & (y < hi)
            zy[inside] = z[inside, (y[inside] - lo)]
    lse = m + torch.log(s)
    loss = (lse - zy).mean()
    grad_h = torch.zeros_like(hf, dtype=acc)
    for lo in range(0, n_vocab, chunk):
        hi = min(n_vocab, lo + chunk)
        E = rows(lo, hi)
        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.float16, enabled=h.is_cuda):
                z = (hf @ E.detach().t()).to(acc)
            g = torch.exp(z - lse[:, None])
            inside = (y >= lo) & (y < hi)
            g[inside, (y[inside] - lo)] -= 1.0
            g /= N
            grad_h += g @ E.detach().to(acc)
            gE = g.t() @ hf.to(acc)
        if E.requires_grad:
            E.backward(gE.to(E.dtype))
    return loss.detach(), grad_h


def synthetic_table(V: int, lengths_pool: np.ndarray, seed: int = 0) -> CodecTable:
    rng = np.random.default_rng(seed)
    L = rng.choice(lengths_pool, size=V)
    buf = rng.integers(1, 256, size=(V, 32), dtype=np.uint8)
    buf[np.arange(32)[None, :] >= L[:, None]] = 0
    return CodecTable(buf, L)


def bench_one(kind: str, V: int, d: int, BT: int, chunk: int, lengths_pool: np.ndarray, device, reps: int = 3) -> dict:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    h = torch.randn(BT, d, device=device)
    y = torch.randint(0, V, (BT,), device=device)
    if kind == "dense":
        W = torch.nn.Parameter(torch.randn(V, d, device=device) * 0.02)
        params = W.numel()
        rows = lambda lo, hi: W[lo:hi]  # noqa: E731
    else:
        table = synthetic_table(V, lengths_pool)
        tc = TorchCodec(table, device)
        W = torch.nn.Parameter(torch.randn(table.D, d, device=device) * 0.02 / math.sqrt(table.D))
        params = W.numel()
        rows = lambda lo, hi: tc.matmul(W, torch.arange(lo, hi, device=device))  # noqa: E731
    base_mem = torch.cuda.memory_allocated()
    times = []
    for r in range(reps + 1):
        W.grad = None
        torch.cuda.synchronize()
        t = time.time()
        chunked_ce_step(h, y, rows, V, chunk)
        torch.cuda.synchronize()
        if r > 0:
            times.append(time.time() - t)
    peak = torch.cuda.max_memory_allocated()
    flops = 2 * BT * V * d * 4   # forward, recompute, grad_h, grad_E matmuls
    return {"head": kind, "V": V, "d": d, "BT": BT, "chunk": chunk, "head_params": int(params),
            "train_state_bytes_adamw_fp32": int(params) * 16, "fwd_bwd_s_median": float(np.median(times)),
            "peak_mem_gb": peak / 1e9, "static_mem_gb": base_mem / 1e9, "logit_matmul_tflop": flops / 1e12}


def main(data_dir: Path, work: Path, d_model: int = 768, BT: int = 8192, chunk: int = 32768,
         Vs=(50_257, 131_072, 262_144, 1_048_576), **_) -> dict:
    from .vocab import WorkingVocab
    device = torch.device("cuda")
    vocab = WorkingVocab.load(Path(data_dir) / "vocab.json")
    pool = vocab.token_lengths()[: vocab.n_train_vocab]
    out = []
    for V in Vs:
        for kind in ("dense", "kas"):
            try:
                out.append(bench_one(kind, V, d_model, BT, chunk, pool, device))
            except torch.cuda.OutOfMemoryError as e:
                out.append({"head": kind, "V": V, "error": "OOM", "detail": str(e)[:300]})
            print(json.dumps(out[-1]), flush=True)
    res = {"device": torch.cuda.get_device_name(0), "results": out}
    p = Path(work) / "runs" / "headcost"
    p.mkdir(parents=True, exist_ok=True)
    (p / "headcost.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    return {"n": len(out)}
