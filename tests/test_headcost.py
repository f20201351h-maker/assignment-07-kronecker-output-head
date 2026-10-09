"""R0: the chunked-vocabulary CE used for the head-cost benchmark is exact."""

import math

import numpy as np
import torch

from kq5.codec import CodecTable, TorchCodec
from kq5.headcost import chunked_ce_step, synthetic_table


def test_chunked_ce_matches_full_softmax_dense_and_kas():
    torch.manual_seed(0)
    N, d, V = 13, 6, 50
    h = torch.randn(N, d, dtype=torch.float64)
    y = torch.randint(0, V, (N,))
    # dense
    W = torch.nn.Parameter(torch.randn(V, d, dtype=torch.float64))
    loss, gh = chunked_ce_step(h, y, lambda lo, hi: W[lo:hi], V, chunk=7)
    W2 = W.detach().clone().requires_grad_(True); h2 = h.clone().requires_grad_(True)
    ref = torch.nn.functional.cross_entropy(h2 @ W2.t(), y)
    ref.backward()
    assert abs(float(loss) - float(ref)) < 1e-9
    assert torch.allclose(gh.double(), h2.grad, atol=1e-6) and torch.allclose(W.grad, W2.grad, atol=1e-9)
    # kas
    table = synthetic_table(V, np.array([1, 3, 5, 9]))
    tc = TorchCodec(table)
    Wk = torch.nn.Parameter(torch.randn(table.D, d, dtype=torch.float64) * 0.01)
    loss, gh = chunked_ce_step(h, y, lambda lo, hi: tc.matmul(Wk, torch.arange(lo, hi)), V, chunk=11)
    Wk2 = Wk.detach().clone().requires_grad_(True); h3 = h.clone().requires_grad_(True)
    ref = torch.nn.functional.cross_entropy(h3 @ tc.matmul(Wk2).t(), y)
    ref.backward()
    assert abs(float(loss) - float(ref)) < 1e-9
    assert torch.allclose(gh.double(), h3.grad, atol=1e-6) and torch.allclose(Wk.grad, Wk2.grad, atol=1e-9)
