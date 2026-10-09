"""Minting held-out merges at inference: regret, leak, beats-prefix, rank.

For a held-out merge m = a + b at a position whose text continues with a b:

    regret = -log2 p'(m | ctx) - [ -log2 p(a | ctx) - log2 p(b | ctx, a) ]

p' uses the same hidden state with the vocabulary grown by every held-out merge
at once; p is the ordinary forward pass. Leak is the extra NLL that growing the
vocabulary adds at every position where no minted token is correct, expressed
in bits per byte of the whole evaluation text. Since a target y in the training
vocabulary keeps its logit, its NLL rises by exactly lse'(h) - lse(h) >= 0.

Rules give a minted token the parameters it never learned:
  prior:      floor | zero | count | inherit | none
  correction: none | inherit | generate | mean (dense)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from .codec import CodecTable
from .evaluate import LN2, autocast_for
from .heads import derangement
from .vocab import WorkingVocab

PRIOR_RULES = ("floor", "zero", "count", "inherit")


@dataclass
class HeldoutInfo:
    ids: np.ndarray          # working ids of held-out merges
    a: np.ndarray            # canonical left piece
    b: np.ndarray            # canonical right piece
    count_m: np.ndarray      # adjacent occurrences of the pair(s) in the training stream
    count_a: np.ndarray      # unigram count of a in the training stream
    table: CodecTable        # codec rows for the held-out strings
    standin: CodecTable      # bytes a shuffled generator reads instead
    n_train_tokens: float
    n_vocab: int
    mean_log_prior: float
    train_counts: np.ndarray | None = None   # unigram counts of the training vocabulary


def heldout_info(vocab: WorkingVocab, unigram: np.ndarray, pair_count: np.ndarray,
                 shuffle_seed: int = 4242) -> HeldoutInfo:
    V = vocab.n_train_vocab
    ks = range(vocab.n_trained, vocab.n_trained + vocab.n_heldout)
    ids = np.array([vocab.merge_id(k) for k in ks], dtype=np.int64)
    a = np.array([vocab.merge_pairs[k][0][0] for k in ks], dtype=np.int64)
    b = np.array([vocab.merge_pairs[k][0][1] for k in ks], dtype=np.int64)
    c = np.asarray(unigram[:V], dtype=np.float64)
    N = float(c.sum())
    mean_lp = float(np.log((c + 1.0) / (N + V)).mean())
    table = CodecTable.from_bytes([vocab.tokens[i] for i in ids], pos_dim=vocab.pos_dim)
    rng = np.random.default_rng(shuffle_seed + 1)
    stand = rng.integers(0, V, size=len(ids))
    standin = CodecTable.from_bytes([vocab.tokens[i] for i in stand], pos_dim=vocab.pos_dim)
    return HeldoutInfo(ids=ids, a=a, b=b, count_m=np.asarray(pair_count, dtype=np.float64),
                       count_a=c[a], table=table, standin=standin, n_train_tokens=N,
                       n_vocab=V, mean_log_prior=mean_lp, train_counts=c)


def prior_for(rule: str, info: HeldoutInfo, beta_train: torch.Tensor | None) -> torch.Tensor | None:
    """Centred log-prior for each minted token under a rule (same scale as beta)."""
    N, V, mu = info.n_train_tokens, info.n_vocab, info.mean_log_prior
    if rule == "none":
        return None
    if rule == "floor":
        v = np.full(len(info.ids), math.log(1.0 / (N + V)) - mu)
    elif rule == "zero":
        v = np.zeros(len(info.ids))
    elif rule == "count":
        v = np.log((info.count_m + 1.0) / (N + V)) - mu
    elif rule == "inherit":
        p_b_given_a = (info.count_m + 1.0) / (info.count_a + 1.0)
        base = (beta_train.detach()[torch.as_tensor(info.a, device=beta_train.device)].double().cpu().numpy()
                if beta_train is not None else 0.0)
        v = base + np.log(np.minimum(p_b_given_a, 1.0))
    elif rule == "logpba":   # dense head: no prior table, add log P(b|a) to the inherited row
        v = np.log(np.minimum((info.count_m + 1.0) / (info.count_a + 1.0), 1.0))
    else:
        raise ValueError(rule)
    return torch.as_tensor(v, dtype=torch.float32)


DRIFT_EDGES = np.array([0, 1, 10, 100, 1_000, 10_000, 100_000, 1_000_000, np.inf])


@torch.no_grad()
def prior_drift(model, info: HeldoutInfo) -> dict:
    """How far each trained token's effective prior (beta, + generated residual for KAS-G)
    moved from its counted initial value, averaged by training-count bin. Minted tokens get
    priors on the initial scale, so a systematic drift is a bias against minting (review S1)."""
    head = model.head
    c = info.train_counts
    init = np.log((c + 1.0) / (info.n_train_tokens + info.n_vocab)) - info.mean_log_prior
    eff = head.beta.detach().double().cpu().numpy().copy()
    if head.kind in ("kasg", "kasg_shuf"):
        src = head.gen_src
        eff += head.generate(head.gen_bytes[src], head.gen_lens[src])[:, head.cfg.rank].double().cpu().numpy()
    drift = eff - init
    b = np.clip(np.searchsorted(DRIFT_EDGES, c, side="right") - 1, 0, len(DRIFT_EDGES) - 2)
    means = np.array([drift[b == k].mean() if np.any(b == k) else 0.0 for k in range(len(DRIFT_EDGES) - 1)])
    counts = np.array([(b == k).sum() for k in range(len(DRIFT_EDGES) - 1)])
    return {"edges": DRIFT_EDGES.tolist(), "mean_drift": means.tolist(), "n_tokens": counts.tolist(),
            "overall_mean_drift": float(drift.mean())}


def rules_for(head_kind: str, dense_bias: bool = False) -> list[tuple[str, str, str]]:
    """(name, prior rule, correction rule) evaluated for each head."""
    if head_kind == "dense":
        if dense_bias:
            # prior-matched Dense (protocol 08) has a prior table, so its rules mirror KAS-U's: a minted
            # token's bias is the training-vocabulary mean ("zero") or the prefix's bias + log P(b|a)
            return [("mean", "zero", "mean"), ("inherit", "inherit", "inherit")]
        return [("mean", "none", "mean"), ("inherit", "logpba", "inherit")]
    if head_kind == "kas0":
        return [("bytes", "none", "none")]
    rules = [(p, p, "none") for p in PRIOR_RULES]
    if head_kind == "kasp":
        return rules
    # "count_cal" (exploratory, not pre-registered): count prior shifted by the mean drift of
    # trained tokens' effective priors in the same count bin; excluded from confirmatory selection
    if head_kind == "kasu":
        return rules[:3] + [("inherit", "inherit", "inherit")] + [("inherit_prior", "inherit", "none")] + \
            [("count_cal", "count_cal", "none")]
    # generator heads: every prior with the generated correction, plus the training-free ones
    return rules[:3] + [("inherit", "inherit", "inherit")] + \
        [(f"gen_{p}", p, "generate") for p in PRIOR_RULES] + [("count_cal", "count_cal", "none")]


@torch.no_grad()
def mint_rows(model, info: HeldoutInfo, prior_rule: str, correction: str):
    head = model.head
    beta = getattr(head, "beta", None)
    if prior_rule == "count_cal":
        d = prior_drift(model, info)
        bins = np.clip(np.searchsorted(DRIFT_EDGES, info.count_m, side="right") - 1, 0, len(DRIFT_EDGES) - 2)
        prior = prior_for("count", info, beta) + torch.as_tensor(np.array(d["mean_drift"])[bins], dtype=torch.float32)
    else:
        prior = prior_for(prior_rule, info, beta)
    standin = info.standin if head.kind == "kasg_shuf" else None
    inherit_from = torch.as_tensor(info.a)
    return head.rows_for(info.table, prior, inherit_from, correction, gen_bytes_src=standin)


def find_sites(windows: np.ndarray, vocab: WorkingVocab) -> dict[str, np.ndarray]:
    """Positions (window w, target index j) whose targets j, j+1 are a held-out
    decomposition (a, b). Target j of window w is windows[w, j+1]."""
    keys, vals = vocab.pair_table("heldout")
    y = windows[:, 1:]
    pk = y[:, :-1].astype(np.int64) * vocab.n_total + y[:, 1:]
    pos = np.minimum(np.searchsorted(keys, pk), len(keys) - 1)
    hit = keys[pos] == pk
    w, j = np.nonzero(hit)
    return {"w": w, "j": j, "a": y[w, j], "b": y[w, j + 1], "m": vals[pos[w, j]]}


@torch.no_grad()
def hidden_states(model, windows: np.ndarray, device, batch: int = 8):
    """Final hidden states (n, T, d) fp32 on device, plus fp32 lse and target logits."""
    from .evaluate import materialize
    model.eval()
    E_in, E, b = materialize(model, model.n_vocab)
    H, LSE, ZT = [], [], []
    for s in range(0, len(windows), batch):
        w = torch.as_tensor(windows[s:s + batch], dtype=torch.long, device=device)
        x, y = w[:, :-1], w[:, 1:]
        with autocast_for(device):
            h = model.body(x, E_in)
        h = h.float()                       # head logits in fp32 so minted and ordinary terms match
        z = h @ E.float().t()
        if b is not None:
            z = z + b.float()
        LSE.append(torch.logsumexp(z, -1))
        ZT.append(z.gather(-1, y.unsqueeze(-1)).squeeze(-1))
        H.append(h)
    return torch.cat(H), torch.cat(LSE), torch.cat(ZT), (E_in, E, b)


@torch.no_grad()
def minting_eval(model, windows: np.ndarray, vocab: WorkingVocab, info: HeldoutInfo, device,
                 target_bytes: np.ndarray, rules=None, states=None, batch: int = 8) -> dict:
    """Per-rule regret/leak/beats-prefix/rank on fixed windows."""
    H, LSE, ZT, (E_in, E, b) = states if states is not None else hidden_states(model, windows, device, batch)
    sites = find_sites(windows, vocab)
    y = torch.as_tensor(windows[:, 1:], device=device)
    nb = torch.as_tensor(target_bytes[windows[:, 1:]], device=device, dtype=torch.float64)
    total_bytes = float(nb.sum())
    nll0 = (LSE - ZT).float()                                   # ordinary NLL, nats
    sw = torch.as_tensor(sites["w"], device=device)
    sj = torch.as_tensor(sites["j"], device=device)
    is_site = torch.zeros_like(nll0, dtype=torch.bool)
    is_site[sw, sj] = True
    hs = H[sw, sj].float()                                       # (S, d)
    a_site = torch.as_tensor(sites["a"], device=device)
    m_index = torch.as_tensor(sites["m"] - vocab.n_train_vocab, device=device)
    Ef = E.float()
    out = {"n_sites": int(len(sites["w"])), "n_positions": int(nll0.numel()),
           "total_bytes": total_bytes, "sites": sites, "rules": {}}
    CH = 2048
    for name, prior_rule, corr in (rules or rules_for(model.head.kind, getattr(model.head.cfg, "dense_bias", False))):
        E_new, b_new = mint_rows(model, info, prior_rule, corr)
        E_new = E_new.float().to(device)
        b_new = b_new.float().to(device)
        # grown-vocabulary log partition at every position
        lse_new = torch.empty_like(LSE)
        flatH = H.view(-1, H.shape[-1])
        flat = lse_new.view(-1)
        for s in range(0, flatH.shape[0], 16384):
            zn = flatH[s:s + 16384].float() @ E_new.t() + b_new
            flat[s:s + 16384] = torch.logsumexp(zn, -1)
        lse_g = torch.logaddexp(LSE, lse_new)
        delta = (lse_g - LSE).double()                           # NLL increase for in-vocab targets
        leak_mask = (~is_site) & (nb > 0)
        leak_bpb = float((delta * leak_mask).sum() / (LN2 * total_bytes))
        zm_site = torch.empty(len(hs), device=device)
        rank = torch.empty(len(hs), device=device, dtype=torch.long)
        beats = torch.empty(len(hs), device=device, dtype=torch.bool)
        for s in range(0, len(hs), CH):
            hc = hs[s:s + CH]
            z_tr = hc @ Ef.t() + (b.float() if b is not None else 0.0)
            z_m = hc @ E_new.t() + b_new
            zm = z_m.gather(1, m_index[s:s + CH, None]).squeeze(1)
            za = z_tr.gather(1, a_site[s:s + CH, None]).squeeze(1)
            zm_site[s:s + CH] = zm
            beats[s:s + CH] = zm > za
            rank[s:s + CH] = 1 + (z_tr > zm[:, None]).sum(1) + (z_m > zm[:, None]).sum(1)
        nll_m = (lse_g[sw, sj] - zm_site).double()
        nll_a = nll0[sw, sj].double()
        nll_b = nll0[sw, sj + 1].double()
        regret = (nll_m - nll_a - nll_b) / LN2
        out["rules"][name] = {
            "prior": prior_rule, "correction": corr,
            "regret_bits": regret.cpu().numpy().astype(np.float32),
            "beats": beats.cpu().numpy(),
            "rank": rank.cpu().numpy(),
            "mean_regret_bits": float(regret.mean()),
            "beats_prefix": float(beats.double().mean()),
            "median_rank": float(rank.double().median()),
            "leak_bpb": leak_bpb,
            "leak_mean_nats_per_pos": float(delta[leak_mask].mean()),
            "leak_per_window_nats": (delta * leak_mask).sum(1).cpu().numpy(),
        }
    return out


def _stream_bpb(model, win, valid, E_in, E_all, b_all, target_bytes_all, device, batch):
    tot_nll, tot_bytes = 0.0, 0.0
    for s in range(0, len(win), batch):
        w = torch.as_tensor(win[s:s + batch], dtype=torch.long, device=device)
        x, yy = w[:, :-1], w[:, 1:]
        with autocast_for(device):
            h = model.body(x, E_in)
        z = h.float() @ E_all.t() + b_all
        nll = torch.nn.functional.cross_entropy(z.view(-1, z.shape[-1]), yy.reshape(-1),
                                                reduction="none").view(yy.shape)
        nbt = torch.as_tensor(target_bytes_all[win[s:s + batch, 1:]] * valid[s:s + batch], device=device,
                              dtype=torch.float64)
        tot_nll += float((nll.double() * (nbt > 0)).sum())
        tot_bytes += float(nbt.sum())
    return tot_nll / (LN2 * tot_bytes), tot_bytes


def retokenized_bpb(model, vocab: WorkingVocab, info: HeldoutInfo, base_merged_windows_stream: np.ndarray,
                    prior_rule: str, corr: str, device, seq_len: int, target_bytes_all: np.ndarray,
                    max_windows: int | None = None, batch: int = 8) -> dict:
    """bpb when the evaluation text is re-tokenised with the held-out merges and the model scores
    the grown vocabulary (input rows come from the codec), and the ordinary bpb of the SAME text:
    both streams are tiled completely (final partial window padded and masked), fp32 logits."""
    from .data import eval_windows_full
    stream0 = np.asarray(base_merged_windows_stream, dtype=np.int64)
    if max_windows is not None:
        stream0 = stream0[: max_windows * seq_len + 1]
    stream1 = vocab.apply_merges(stream0, "heldout")
    model.eval()
    E_new, b_new = mint_rows(model, info, prior_rule, corr)
    with torch.no_grad():
        E_in = model.input_table_rows(torch.arange(vocab.n_total, device=device))
        E, b = model.head.materialize()
        b0 = b.float() if b is not None else torch.zeros(E.shape[0], device=device)
        E_all = torch.cat([E.float(), E_new.float().to(device)])
        b_all = torch.cat([b0, b_new.float().to(device)])
        w0, v0 = eval_windows_full(stream0, seq_len)
        w1, v1 = eval_windows_full(stream1, seq_len)
        tb = np.concatenate([target_bytes_all[:E.shape[0]], info.table.lengths])             if len(target_bytes_all) < E_all.shape[0] else target_bytes_all
        bpb0, by0 = _stream_bpb(model, w0, v0, E_in, E.float(), b0, tb, device, batch)
        bpb1, by1 = _stream_bpb(model, w1, v1, E_in, E_all, b_all, tb, device, batch)
    model.train()
    return {"bpb": bpb1, "bpb_standard_same_text": bpb0, "delta_bpb": bpb1 - bpb0,
            "bytes_retok": by1, "bytes_standard": by0, "n_tokens": int(len(stream1)),
            "n_tokens_standard": int(len(stream0))}


__all__ = ["HeldoutInfo", "heldout_info", "prior_for", "rules_for", "mint_rows", "find_sites",
           "hidden_states", "minting_eval", "retokenized_bpb", "derangement"]
