"""R0: every head - shapes, gradient flow, parameter counts by hand, exactness,
unseen-string scoring, and the per-step materialisation trick."""

import math

import numpy as np
import pytest
import torch

from kq5.codec import CodecTable
from kq5.generator import GeneratorConfig, count_params
from kq5.heads import HEADS, Head, HeadConfig, centred_log_unigram, derangement
from kq5.model import GPT, ModelConfig, param_report

TOKS = [b"a", b"b", b"ab", b"the", b" the", b"ing", b" Arch", b"ived", b"\n", b"<|endoftext|>",
        b"x", b"yz", "भा".encode(), b"0", b"12", b"!!"]
V = len(TOKS)
GEN = GeneratorConfig(emb=8, channels=12, layers=2, kernel=3, hidden=16, pos_dim=32)


def tiny(kind, d=12, rank=4, seed=0):
    torch.manual_seed(seed)
    table = CodecTable.from_bytes(TOKS)
    counts = np.arange(1, V + 1) * 10
    prior, _, _ = centred_log_unigram(counts, V)
    cfg = ModelConfig(n_layer=1, n_head=2, d_model=d, seq_len=8,
                      head=HeadConfig(kind=kind, rank=rank, generator=GEN))
    return GPT(cfg, table, V, None if kind in ("dense", "kas0") else prior)


@pytest.mark.parametrize("kind", HEADS)
def test_shapes_and_gradient_flow(kind):
    m = tiny(kind)
    E_in = m.input_table_rows()
    E, b = m.head.materialize()
    assert E.shape == (V, 12)
    assert b is None if kind == "dense" else b.shape == (V,)
    x = torch.randint(0, V, (3, 8)); y = torch.randint(0, V, (3, 8))
    nll = m.nll(x, y, E_in, E, b)
    assert nll.shape == (3, 8) and torch.isfinite(nll).all()
    nll.mean().backward()
    for n, p in m.named_parameters():
        assert p.grad is not None, n
    # corrected arms start exactly at the uncorrected head (C = 0)
    if kind in ("kasu", "kasg", "kasg_shuf"):
        assert torch.count_nonzero(m.head.C) == 0
        assert m.head.C.grad.abs().sum() > 0


@pytest.mark.parametrize("kind", HEADS)
def test_parameter_counts_by_hand(kind):
    d, r = 12, 4
    m = tiny(kind, d=d, rank=r)
    D = 256 * 32
    expect = {"dense": V * d,
              "kas0": D * d + 33,
              "kasp": D * d + 33,
              "kasu": D * d + 33 + V + r * d + V * r,
              "kasg": D * d + 33 + r * d,
              "kasg_shuf": D * d + 33 + r * d}[kind]
    if kind in ("kasg", "kasg_shuf"):
        g = GEN
        gen = (256 * g.emb + 2 * 32 * g.emb + 33 * 2 * g.channels
               + g.kernel * g.emb * g.channels + g.channels
               + (g.layers - 1) * (g.kernel * g.channels * g.channels + g.channels)
               + 2 * g.channels * g.hidden + g.hidden + g.hidden * (r + 1) + (r + 1))
        assert count_params(m.head.gen) == gen
        expect += gen
    assert sum(p.numel() for p in m.head.parameters()) == expect
    rep = param_report(m)
    assert rep["input_projection"] == D * d
    assert rep["head_V_dependent"] == {"dense": V * d, "kasu": V + V * r}.get(kind, 0)


def test_kas_equals_dense_head_with_KW_float64():
    m = tiny("kasp").double()
    E, b = m.head.materialize()
    K = torch.as_tensor(CodecTable.from_bytes(TOKS).dense())
    W_eff = K @ m.head.W_out
    h = torch.randn(5, 12, dtype=torch.float64)
    lens = torch.as_tensor([len(t) for t in TOKS])
    dense_logits = h @ W_eff.t() + m.head.beta + m.head.len_bias[lens]
    assert torch.max(torch.abs(h @ E.t() + b - dense_logits)) < 1e-10


@pytest.mark.parametrize("kind", ["kas0", "kasp", "kasu", "kasg", "kasg_shuf"])
def test_unseen_string_scored_from_bytes(kind):
    m = tiny(kind).double()
    new = CodecTable.from_bytes([b"ZQVBX", b" Archived"])
    prior = torch.tensor([0.5, -1.0])
    corr = "generate" if kind.startswith("kasg") else "none"
    E_new, b_new = m.head.rows_for(new, prior, torch.tensor([0, 6]), corr,
                                   gen_bytes_src=new)
    K = torch.as_tensor(new.dense())
    expect_E = K @ m.head.W_out
    if corr == "generate":
        g = m.head.generate(torch.as_tensor(new.byte_buffer).long(), torch.as_tensor(new.lengths))
        expect_E = expect_E + g[:, :4].double() @ m.head.C
    assert torch.allclose(E_new, expect_E, atol=1e-10)
    lens = torch.as_tensor(new.lengths)
    base_b = m.head.len_bias[lens] + prior.double()
    if corr == "generate":
        base_b = base_b + g[:, 4].double()
    assert torch.allclose(b_new, base_b, atol=1e-10)


def test_inherit_rows_copy_prefix_parameters():
    m = tiny("kasu").double()
    with torch.no_grad():
        m.head.C.normal_()
    new = CodecTable.from_bytes([b" Archived"])
    E_new, _ = m.head.rows_for(new, None, torch.tensor([6]), "inherit")
    expect = torch.as_tensor(new.dense()) @ m.head.W_out + m.head.U[6:7] @ m.head.C
    assert torch.allclose(E_new, expect, atol=1e-10)


def test_shuffled_generator_reads_other_tokens():
    p = derangement(1000, 3)
    assert sorted(p.tolist()) == list(range(1000)) and not np.any(p == np.arange(1000))
    m = tiny("kasg_shuf")
    assert not torch.equal(m.head.gen_src, torch.arange(V))


@pytest.mark.parametrize("kind", HEADS)
def test_materialisation_trick_gives_same_gradients(kind):
    """Detached leaves + one backward through the tables == ordinary backprop."""
    torch.manual_seed(1)
    m1 = tiny(kind, seed=3)
    m2 = tiny(kind, seed=3)
    if kind in ("kasu", "kasg", "kasg_shuf"):
        with torch.no_grad():
            m1.head.C.normal_(); m2.head.C.copy_(m1.head.C)
    x = torch.randint(0, V, (4, 8)); y = torch.randint(0, V, (4, 8))
    # ordinary
    E_in = m1.input_table_rows(); E, b = m1.head.materialize()
    m1.nll(x, y, E_in, E, b).mean().backward()
    # trick, two micro-batches
    E_in = m2.input_table_rows(); E, b = m2.head.materialize()
    E_in_l = E_in.detach().requires_grad_(True); E_l = E.detach().requires_grad_(True)
    b_l = b.detach().requires_grad_(True) if b is not None else None
    for s in (slice(0, 2), slice(2, 4)):
        (m2.nll(x[s], y[s], E_in_l, E_l, b_l).mean() / 2).backward()
    roots, grads = [E_in, E], [E_in_l.grad, E_l.grad]
    if b is not None and b.requires_grad:
        roots.append(b); grads.append(b_l.grad)
    torch.autograd.backward(roots, grads)
    for (n, p1), (_, p2) in zip(m1.named_parameters(), m2.named_parameters()):
        assert torch.allclose(p1.grad, p2.grad, atol=1e-5, rtol=1e-4), n


def test_chunked_generator_matches_unchunked_values_and_grads():
    m = tiny("kasg").double()
    m.head.GEN_CHUNK = 5
    src = m.head.gen_src
    g_chunk = m.head.generate(m.head.gen_bytes[src], m.head.gen_lens[src])
    g_full = m.head.gen(m.head.gen_bytes[src], m.head.gen_lens[src])
    assert torch.allclose(g_chunk, g_full, atol=1e-12)
    w = torch.randn_like(g_full)
    (g_chunk * w).sum().backward()
    grads = [p.grad.clone() for p in m.head.gen.parameters()]
    for p in m.head.gen.parameters():
        p.grad = None
    (g_full * w).sum().backward()
    for a, p in zip(grads, m.head.gen.parameters()):
        assert torch.allclose(a, p.grad, atol=1e-10)


def test_kasg_inherit_includes_prefix_generated_prior():
    """Review fix: 'inherit' on a generator head copies the prefix's generated correction AND
    its generated prior residual."""
    m = tiny("kasg").double()
    with torch.no_grad():
        m.head.C.normal_()
        m.head.gen.fc2.weight.normal_()
        m.head.gen.fc2.bias.normal_()
    new = CodecTable.from_bytes([b" Archived"])
    _, b_new = m.head.rows_for(new, None, torch.tensor([6]), "inherit")
    g = m.head.generate(m.head.gen_bytes[6:7], m.head.gen_lens[6:7])
    expect = m.head.len_bias[torch.as_tensor(new.lengths)] + g[:, 4]
    assert torch.allclose(b_new, expect, atol=1e-10)


@pytest.mark.parametrize("kind", ["kasu", "kasg", "kasg_shuf"])
def test_u_zero_init_is_function_preserving_and_trainable(kind):
    torch.manual_seed(0)
    table = CodecTable.from_bytes(TOKS)
    prior, _, _ = centred_log_unigram(np.arange(1, V + 1) * 10, V)
    cfg = ModelConfig(n_layer=1, n_head=2, d_model=12, seq_len=8,
                      head=HeadConfig(kind=kind, rank=4, generator=GEN, corr_init="u_zero"))
    m = GPT(cfg, table, V, prior)
    assert torch.count_nonzero(m.head.C) > 0
    E, b = m.head.materialize()
    E_kas = m.head.codec.matmul(m.head.W_out)
    assert torch.allclose(E, E_kas)                       # correction is exactly zero at init
    x = torch.randint(0, V, (2, 8))
    m.nll(x, x, m.input_table_rows(), E, b).mean().backward()
    if kind == "kasu":
        assert m.head.U.grad.abs().sum() > 0              # stored rows receive gradient at step 0
    else:
        assert m.head.gen.fc2.weight.grad.abs().sum() > 0
