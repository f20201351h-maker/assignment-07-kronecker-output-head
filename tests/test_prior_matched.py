"""Protocol 08 (prior-matched Dense): the dense head with `dense_bias=True` differs from the plain dense
head in exactly one thing — a trainable per-token bias initialised to the centred log-unigram prior — and
the analysis code paths that touch it (optimiser grouping, minting rules, term shares, ablation) run."""

import numpy as np
import torch

from kq5.codec import CodecTable
from kq5.heads import HeadConfig, centred_log_unigram, head_param_groups
from kq5.minting import HeldoutInfo, prior_for, rules_for
from kq5.model import GPT, ModelConfig

TOKS = [b"a", b"b", b"ab", b"the", b" the", b"ing", b" Arch", b"ived", b"\n", b"<|endoftext|>",
        b"x", b"yz", "भा".encode(), b"0", b"12", b"!!"]
V = len(TOKS)
COUNTS = np.arange(1, V + 1) * 10


def build(dense_bias: bool, seed: int = 0, d: int = 12) -> GPT:
    torch.manual_seed(seed)
    table = CodecTable.from_bytes(TOKS)
    prior, _, _ = centred_log_unigram(COUNTS, V)
    cfg = ModelConfig(n_layer=1, n_head=2, d_model=d, seq_len=8,
                      head=HeadConfig(kind="dense", dense_bias=dense_bias))
    return GPT(cfg, table, V, prior if dense_bias else None)


def test_prior_matched_dense_is_dense_plus_one_bias():
    plain, pm = build(False, seed=3), build(True, seed=3)
    # every shared parameter is bit-identical for the same seed; the bias is the only addition
    sd0, sd1 = plain.state_dict(), pm.state_dict()
    assert set(sd1) - set(sd0) == {"head.beta"}
    for k in sd0:
        assert torch.equal(sd0[k], sd1[k]), k
    prior, _, _ = centred_log_unigram(COUNTS, V)
    assert torch.allclose(pm.head.beta.detach(), torch.as_tensor(prior, dtype=torch.float32))
    E0, b0 = plain.head.materialize()
    E1, b1 = pm.head.materialize()
    assert b0 is None and torch.equal(E0, E1) and b1 is pm.head.beta


def test_bias_trains_in_the_per_token_group_without_decay():
    pm = build(True)
    groups = head_param_groups(pm.head)
    assert any(p is pm.head.beta for p in groups["per_token"])      # same group as KAS-U's beta
    assert all(p is not pm.head.beta for p in groups["head_matrix"] + groups["head_small"])
    from kq5.train import RunConfig, make_optimizer
    opt = make_optimizer(pm, RunConfig(run_id="t", head="dense", dense_bias=True, weight_decay=0.1))
    g = next(g for g in opt.param_groups if any(p is pm.head.beta for p in g["params"]))
    assert g["weight_decay"] == 0.0 and g["lr_mult"] == 1.0
    gw = next(g for g in opt.param_groups if any(p is pm.head.W for p in g["params"]))
    assert gw["weight_decay"] == 0.1 and gw["lr_mult"] == 1.0
    # gradient reaches the bias through the materialised (E, b) path used in training
    E_in = pm.input_table_rows()
    E, b = pm.head.materialize()
    x = torch.randint(0, V, (2, 8)); y = torch.randint(0, V, (2, 8))
    pm.nll(x, y, E_in, E, b).mean().backward()
    assert pm.head.beta.grad is not None and pm.head.beta.grad.abs().sum() > 0


def test_minting_rules_mirror_kas_u_for_the_biased_dense_head():
    assert rules_for("dense") == [("mean", "none", "mean"), ("inherit", "logpba", "inherit")]
    assert rules_for("dense", True) == [("mean", "zero", "mean"), ("inherit", "inherit", "inherit")]
    pm = build(True).double()
    new = CodecTable.from_bytes([b" Archived", b"xyz"])
    info = HeldoutInfo(ids=np.array([V, V + 1]), a=np.array([6, 10]), b=np.array([7, 11]),
                       count_m=np.array([5.0, 1.0]), count_a=COUNTS[[6, 10]].astype(float), table=new, standin=new,
                       n_train_tokens=float(COUNTS.sum()), n_vocab=V, mean_log_prior=0.0, train_counts=COUNTS)
    prior = prior_for("inherit", info, pm.head.beta)
    expect = pm.head.beta.detach()[[6, 10]].double().numpy() + np.log(np.minimum((info.count_m + 1) / (info.count_a + 1), 1))
    assert np.allclose(prior.numpy(), expect, atol=1e-6)
    E_new, b_new = pm.head.rows_for(new, prior, torch.tensor([6, 10]), "inherit")
    assert torch.equal(E_new, pm.head.W[[6, 10]]) and torch.allclose(b_new.double(), prior.double())
    assert torch.equal(prior_for("zero", info, pm.head.beta), torch.zeros(2))


def test_analysis_paths_for_the_biased_dense_head():
    from kq5.analysis import ablation_bpb, term_shares
    pm = build(True)
    windows = torch.randint(0, V, (60, 9)).numpy()
    H = torch.randn(60, 8, 12)
    t = term_shares(pm, H, windows, torch.device("cpu"))
    assert set(t) == {"var_total", "share_dense", "var_dense", "share_prior", "var_prior"}
    # shares use a population covariance over an unbiased variance (as the KAS branch does): sum to 1 up to 1/n
    assert abs(t["share_dense"] + t["share_prior"] - 1.0) < 0.01
    tb = np.array([len(t) for t in TOKS])
    a = ablation_bpb(pm, windows, tb, torch.device("cpu"))
    assert set(a) == {"bpb_full", "bpb_no_prior"} and np.isfinite(a["bpb_full"]) and np.isfinite(a["bpb_no_prior"])
    assert term_shares(build(False), H, windows, torch.device("cpu")) == {}
    assert ablation_bpb(build(False), windows, tb, torch.device("cpu")) == {}
