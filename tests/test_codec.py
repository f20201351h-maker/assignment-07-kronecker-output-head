"""R0: codec round trip, reference agreement, collisions, KAS == dense(K W)."""

import numpy as np
import pytest
import torch

from kq5.codec import CodecTable, codec_coefficients, code_norm_sq, reference_codec

SAMPLES = [
    b"the", b" Archived", "भारत".encode(), "తెలుగు".encode(), "राष्ट्र".encode(),
    b"", b"x" * 32, bytes(range(32)), b"\xe0\xa4", b"\x00", b"\xff" * 5, b"ZQVBX", b"###777",
]


def decode_code(kappa: np.ndarray, pos_dim: int = 32) -> bytes:
    """Invert one code vector: the support is where kappa exceeds its off-support value."""
    if np.allclose(kappa, 0):
        return b""
    lo, hi = kappa.min(), kappa.max()
    support = np.flatnonzero(kappa > (lo + hi) / 2)
    byte_pos = sorted((j % pos_dim, j // pos_dim) for j in support)
    assert [p for p, _ in byte_pos] == list(range(len(byte_pos))), "support is not a prefix"
    return bytes(b for _, b in byte_pos)


def test_round_trip_including_indic_empty_and_max_length():
    t = CodecTable.from_bytes(SAMPLES)
    K = t.dense()
    for i, s in enumerate(SAMPLES):
        assert decode_code(K[i]) == s


def test_too_long_is_rejected_not_truncated():
    with pytest.raises(ValueError):
        CodecTable.from_bytes([b"y" * 33])


def test_matches_reference_package():
    t = CodecTable.from_bytes(SAMPLES)
    ref = reference_codec(t)
    if ref is None:
        pytest.skip("kronecker-embeddings not installed")
    assert np.max(np.abs(ref - t.dense())) < 2e-4


def test_matches_reference_on_gpt2_sample(gpt2_bytes):
    toks, _ = gpt2_bytes
    rng = np.random.default_rng(0)
    pick = [t for t in (toks[i] for i in rng.choice(len(toks), 3000, replace=False)) if len(t) <= 32]
    t = CodecTable.from_bytes(pick)
    ref = reference_codec(t)
    if ref is None:
        pytest.skip("kronecker-embeddings not installed")
    assert np.max(np.abs(ref - t.dense())) < 2e-4


def test_coefficients_give_unit_unbiased_std():
    for L in (1, 2, 5, 17, 32):
        A, B = codec_coefficients(np.array([L]))
        row = np.full(8192, B[0]); row[:L] += A[0]
        assert abs(row.mean()) < 1e-12
        assert abs(row.std(ddof=1) - 1.0) < 1e-4
        assert abs(code_norm_sq(L) - (row ** 2).sum()) < 1e-6


def test_no_collisions_distinct_strings_distinct_codes():
    strings = [bytes([a, b]) for a in range(0, 256, 7) for b in range(0, 256, 11)] + SAMPLES[:6]
    strings = list(dict.fromkeys(strings))
    K = CodecTable.from_bytes(strings).dense()
    rounded = {tuple(np.round(r, 6)) for r in K}
    assert len(rounded) == len(strings)


@pytest.mark.parametrize("d", [3, 16])
def test_codec_matmul_equals_dense_float64(d):
    t = CodecTable.from_bytes(SAMPLES)
    W = torch.randn(t.D, d, dtype=torch.float64)
    fast = t.torch_buffers().matmul(W)
    dense = torch.as_tensor(t.dense()) @ W
    assert torch.max(torch.abs(fast - dense)) < 1e-10
    ids = torch.tensor([[0, 3], [5, 5]])
    assert torch.max(torch.abs(t.torch_buffers().matmul(W, ids) - dense[ids])) < 1e-10


def test_kas_logit_is_sum_of_score_table_lookups():
    """logit_v = kappa_v . (W h) = A_v * sum_{(byte,pos) in v} s[byte,pos] + B_v * sum(s)."""
    t = CodecTable.from_bytes(SAMPLES)
    d = 8
    W = torch.randn(t.D, d, dtype=torch.float64)
    h = torch.randn(d, dtype=torch.float64)
    s = (W @ h).numpy()                       # the 256 x 32 score table, flattened
    via_dense = torch.as_tensor(t.dense()) @ (W @ h)
    via_rows = t.torch_buffers().matmul(W) @ h
    lookups = np.array([t.A[v] * s[t.flat_index[v][t.valid[v]]].sum() + t.B[v] * s.sum()
                        for v in range(t.V)])
    assert np.max(np.abs(via_dense.numpy() - lookups)) < 1e-10
    assert np.max(np.abs(via_rows.numpy() - lookups)) < 1e-10


def test_codec_gradient_matches_dense():
    t = CodecTable.from_bytes(SAMPLES)
    W1 = torch.randn(t.D, 4, dtype=torch.float64, requires_grad=True)
    W2 = W1.detach().clone().requires_grad_(True)
    G = torch.randn(t.V, 4, dtype=torch.float64)
    (t.torch_buffers().matmul(W1) * G).sum().backward()
    ((torch.as_tensor(t.dense()) @ W2) * G).sum().backward()
    assert torch.max(torch.abs(W1.grad - W2.grad)) < 1e-10
