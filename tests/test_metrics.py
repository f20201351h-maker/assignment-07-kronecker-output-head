"""R0: metrics on tiny hand-checkable cases - bpb, regret, leak, beats-prefix,
rank, retokenised bpb."""

import math

import numpy as np
import pytest
import torch

from kq5.codec import CodecTable
from kq5.evaluate import bpb_from, per_position_nll
from kq5.generator import GeneratorConfig
from kq5.heads import HeadConfig, centred_log_unigram
from kq5.minting import find_sites, heldout_info, minting_eval, mint_rows, rules_for
from kq5.model import GPT, ModelConfig
from kq5.vocab import WorkingVocab


def test_bpb_hand_case():
    nll = np.array([[math.log(2), math.log(4), 5.0]])      # 1 bit, 2 bits, EOS
    nb = np.array([[1, 2, 0]])
    assert abs(bpb_from(nll, nb) - 3 / 3) < 1e-12


def small_vocab():
    base = [b"a", b"b", b"c", b" Arch", b"ived", b"x", b"<|endoftext|>"]
    v = WorkingVocab(tokens=list(base), n_base=len(base), n_trained=0, n_heldout=0, eos_id=6,
                     pos_dim=32, gpt2_to_work=[[i] for i in range(len(base))])
    v.add_merges([(b"ab", [(0, 1)], 50)], [(b"bc", [(1, 2)], 7), (b" Archived", [(3, 4)], 3)])
    return v


def small_model(v, kind):
    torch.manual_seed(0)
    table = CodecTable.from_bytes(v.tokens)
    counts = np.array([30, 20, 10, 5, 5, 40, 3, 50, 0, 0])
    prior, _, _ = centred_log_unigram(counts, v.n_train_vocab)
    cfg = ModelConfig(n_layer=1, n_head=2, d_model=8, seq_len=6,
                      head=HeadConfig(kind=kind, rank=3, generator=GeneratorConfig(
                          emb=4, channels=6, layers=1, hidden=8)))
    m = GPT(cfg, table, v.n_train_vocab, None if kind in ("dense", "kas0") else prior)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(torch.randn_like(p) * (0.3 if p.dim() < 2 or p.shape[0] < 1000 else 0.003))
    return m, counts


@pytest.mark.parametrize("kind", ["dense", "kas0", "kasp", "kasu", "kasg", "kasg_shuf"])
def test_regret_leak_rank_bruteforce(kind):
    v = small_vocab()
    m, counts = small_model(v, kind)
    # sites: (b, c) at targets 1,2 of window 0 ; (" Arch","ived") at targets 3,4 of window 1
    windows = np.array([[5, 1, 2, 0, 5, 6, 7],
                        [0, 5, 3, 3, 4, 1, 0]])
    sites = find_sites(windows, v)
    assert sorted(zip(sites["w"].tolist(), sites["j"].tolist())) == [(0, 0), (1, 2)]
    pair_count = np.array([7.0, 3.0])
    info = heldout_info(v, counts, pair_count)
    nb = v.target_bytes()
    res = minting_eval(m, windows, v, info, torch.device("cpu"), nb)
    E_in = m.input_table_rows(torch.arange(v.n_total))
    with torch.no_grad():
        E, b = m.head.materialize()
        b = torch.zeros(E.shape[0]) if b is None else b
        x = torch.as_tensor(windows[:, :-1]); y = torch.as_tensor(windows[:, 1:])
        h = m.body(x, E_in[: v.n_train_vocab])
        z0 = h @ E.t() + b
        lp0 = torch.log_softmax(z0.double(), -1)
        for name, prior_rule, corr in rules_for(kind):
            E_new, b_new = mint_rows(m, info, prior_rule, corr)
            z1 = torch.cat([z0, h @ E_new.t() + b_new], -1)
            lp1 = torch.log_softmax(z1.double(), -1)
            r = res["rules"][name]
            for k, (w, j) in enumerate(zip(sites["w"], sites["j"])):
                mid = sites["m"][k]
                expect = (-lp1[w, j, mid] + lp0[w, j, y[w, j]] + lp0[w, j + 1, y[w, j + 1]]) / math.log(2)
                assert abs(float(expect) - float(r["regret_bits"][k])) < 2e-3, (name, k)
                zm = z1[w, j, mid]
                assert bool(r["beats"][k]) == bool(zm > z1[w, j, sites["a"][k]])
                assert int(r["rank"][k]) == 1 + int((z1[w, j] > zm).sum())
            site = np.zeros(y.shape, bool); site[sites["w"], sites["j"]] = True
            nbt = nb[windows[:, 1:]]
            mask = (~site) & (nbt > 0)
            leak = sum(float(lp0[w, j, y[w, j]] - lp1[w, j, y[w, j]])
                       for w, j in zip(*np.nonzero(mask)))
            assert abs(leak / (math.log(2) * nbt.sum()) - r["leak_bpb"]) < 2e-4, name


def test_prior_rules_values():
    v = small_vocab()
    counts = np.array([30, 20, 10, 5, 5, 40, 3, 50, 0, 0])
    info = heldout_info(v, counts, np.array([7.0, 3.0]))
    from kq5.minting import prior_for
    N, V = counts[:8].sum(), 8
    mu = np.log((counts[:8] + 1) / (N + V)).mean()
    assert np.allclose(prior_for("floor", info, None).numpy(), math.log(1 / (N + V)) - mu, atol=1e-5)
    assert np.allclose(prior_for("count", info, None).numpy(),
                       np.log((np.array([7, 3]) + 1) / (N + V)) - mu, atol=1e-5)
    beta = torch.arange(8, dtype=torch.float32)
    inh = prior_for("inherit", info, beta).numpy()
    assert np.allclose(inh, [1 + math.log(8 / 21), 3 + math.log(4 / 6)], atol=1e-5)


def test_full_tiling_scores_every_target_once():
    from kq5.data import eval_windows_full
    s = np.arange(1, 24)
    w, v = eval_windows_full(s, 5)
    targets = w[:, 1:][v]
    assert targets.tolist() == list(range(2, 24))


def test_retok_reports_same_text_standard_bpb():
    v = small_vocab()
    m, counts = small_model(v, "kasp")
    info = heldout_info(v, counts, np.array([7.0, 3.0]))
    from kq5.minting import retokenized_bpb
    from kq5.data import eval_windows_full
    stream = np.array([5, 1, 2, 0, 5, 6, 7, 0, 5, 3, 4, 1, 0, 1, 2, 5, 3, 4], dtype=np.int64)
    r = retokenized_bpb(m, v, info, stream, "count", "none", torch.device("cpu"), 6, v.target_bytes())
    # standard bpb recomputed independently over the full tiling
    w, valid = eval_windows_full(stream, 6)
    tb = v.target_bytes()
    with torch.no_grad():
        E_in = m.input_table_rows(torch.arange(v.n_total)); E, b = m.head.materialize()
        h = m.body(torch.as_tensor(w[:, :-1]), E_in)
        lp = torch.log_softmax((h @ E.t() + b).double(), -1)
    y = w[:, 1:]
    nll = -lp.gather(-1, torch.as_tensor(y)[..., None]).squeeze(-1).numpy()
    nb = tb[y] * valid
    assert abs(r["bpb_standard_same_text"] - (nll * (nb > 0)).sum() / (math.log(2) * nb.sum())) < 1e-5
    assert r["n_tokens"] < r["n_tokens_standard"]          # held-out merges shortened the text
    assert abs(r["bytes_retok"] - r["bytes_standard"]) <= max(len(t) for t in v.tokens)
