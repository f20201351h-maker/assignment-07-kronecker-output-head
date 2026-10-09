"""Phase 2 (protocol 06): vocabulary dropout on the stored-correction head, and the reference
report's generator form (OccMLP) — exactness of its code input and its parameter count."""

import numpy as np
import torch

from kq5.codec import CodecTable
from kq5.generator import GeneratorConfig, OccMLP, count_params
from kq5.heads import Head, HeadConfig, centred_log_unigram

TOKS = [b"a", b"b", b"ab", b"the", b" the", b"ing", b" Arch", b"ived", b"\n", b"<|endoftext|>",
        b"x", b"yz", "भा".encode(), b"0", b"12", b"!!", b"", b"z" * 32]
V = len(TOKS)


def kasu(p, d=12, rank=4, seed=0):
    torch.manual_seed(seed)
    table = CodecTable.from_bytes(TOKS)
    prior, _, _ = centred_log_unigram(np.arange(1, V + 1) * 10, V)
    h = Head(HeadConfig(kind="kasu", rank=rank, corr_init="u_zero", vocab_dropout=p), d, table, V, prior)
    with torch.no_grad():                    # give the stored table something to hide
        h.U.normal_()
    return h


def test_vocab_dropout_hides_rows_in_training_only():
    torch.manual_seed(1)
    h = kasu(0.5)
    h.eval()
    E0, b0 = h.materialize()
    h.train()
    E1, b1 = h.materialize()
    E2, b2 = h.materialize()
    # eval mode: full rows, deterministic; train mode: a random subset of rows hidden, redrawn per call
    assert not torch.equal(E1, E0) and not torch.equal(E1, E2)
    hidden = (b1 != b0)
    assert 0 < int(hidden.sum()) < V
    base_E = h.codec.matmul(h.W_out)         # byte-derived rows only
    base_b = h.len_bias[h.lengths]
    assert torch.allclose(E1[hidden], base_E[hidden])          # hidden rows: no correction
    assert torch.allclose(b1[hidden], base_b[hidden])          # hidden rows: prior = mean (0 centred)
    assert torch.allclose(E1[~hidden], E0[~hidden]) and torch.allclose(b1[~hidden], b0[~hidden])


def test_vocab_dropout_u_only_keeps_prior():
    torch.manual_seed(3)
    table = CodecTable.from_bytes(TOKS)
    prior, _, _ = centred_log_unigram(np.arange(1, V + 1) * 10, V)
    h = Head(HeadConfig(kind="kasu", rank=4, corr_init="u_zero", vocab_dropout=0.5, vocab_dropout_target="u"),
             12, table, V, prior)
    with torch.no_grad():
        h.U.normal_()
    h.eval()
    E0, b0 = h.materialize()
    h.train()
    E1, b1 = h.materialize()
    assert torch.equal(b1, b0)                                   # prior never hidden
    hidden = ~torch.isclose(E1, E0).all(1)
    assert 0 < int(hidden.sum()) < V
    assert torch.allclose(E1[hidden], h.codec.matmul(h.W_out)[hidden])


def test_vocab_dropout_off_is_phase1_head():
    torch.manual_seed(2)
    h = kasu(0.0)
    h.train()
    E1, b1 = h.materialize()
    h.eval()
    E0, b0 = h.materialize()
    assert torch.equal(E1, E0) and torch.equal(b1, b0)


def test_occmlp_code_matches_codec_exactly():
    table = CodecTable.from_bytes(TOKS)
    g = OccMLP(5, GeneratorConfig(kind="occmlp", hidden=7))
    code = g.code(torch.as_tensor(table.byte_buffer).long(), torch.as_tensor(table.lengths)).double().numpy()
    assert np.allclose(code, table.dense(), atol=1e-6)
    out = g(torch.as_tensor(table.byte_buffer).long(), torch.as_tensor(table.lengths))
    assert out.shape == (V, 5) and torch.isfinite(out).all()


def test_occmlp_parameter_count_matches_v3_budget():
    D = 256 * 32
    g = OccMLP(17, GeneratorConfig(kind="occmlp", hidden=92))
    assert count_params(g) == D * 92 + 92 + 92 * 17 + 17 == 755_337
    v3 = GeneratorConfig(emb=64, channels=256, layers=3, kernel=3, hidden=512)
    from kq5.generator import ByteCNN
    n3 = count_params(ByteCNN(17, v3))
    assert n3 == 751_889 and abs(count_params(g) / n3 - 1) < 0.01
