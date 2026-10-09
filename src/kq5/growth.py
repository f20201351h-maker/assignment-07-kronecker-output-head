"""Vocabulary growth toward 1M candidates (R4, should-have).

Mint K new tokens at inference: the 1,000 held-out merges first, then the most frequent
further word-internal pair strings from the training split (never trained). For each K:
  leak    extra NLL at positions where no minted token is correct, in bits per byte
  regret  mean minting regret over every site of every minted pair
  retok   bpb and token count when the text is re-tokenised with all K merges
Rules: each head's best training-free rule family (count prior; inherit; generated).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from .codec import CodecTable, TorchCodec
from .data import TokenData, eval_windows, eval_windows_full
from .evaluate import LN2, autocast_for
from .minting import HeldoutInfo, hidden_states, prior_for
from .vocab import apply_pair_merges


def candidate_info(vocab, unigram, heldout_count, growth, K: int, shuffle_seed: int = 4242):
    """HeldoutInfo-like bundle for the first K minted candidates, plus their pair keys."""
    V = vocab.n_train_vocab
    H = vocab.n_heldout
    ks = range(vocab.n_trained, vocab.n_trained + H)
    a = [vocab.merge_pairs[k][0][0] for k in ks]
    b = [vocab.merge_pairs[k][0][1] for k in ks]
    cnt = list(np.asarray(heldout_count, dtype=np.float64))
    keys = [p[0] * vocab.n_total + p[1] for k in ks for p in vocab.merge_pairs[k]]
    owner = [i for i, k in enumerate(ks) for _ in vocab.merge_pairs[k]]
    n_g = max(0, min(K - H, len(growth["a"])))
    a = np.array(a + growth["a"][:n_g].tolist(), dtype=np.int64)
    b = np.array(b + growth["b"][:n_g].tolist(), dtype=np.int64)
    cnt = np.array(cnt + growth["count_merged_train"][:n_g].astype(np.float64).tolist())
    # growth keys were built with key = a * n_total + b in prepare_data
    gk, gown = growth["keys"], growth["key_to_cand"]
    sel = gown < n_g
    keys = np.array(keys + gk[sel].tolist(), dtype=np.int64)
    owner = np.array(owner + (gown[sel] + H).tolist(), dtype=np.int64)
    order = np.argsort(keys, kind="stable")
    keys, owner = keys[order], owner[order]
    c = np.asarray(unigram[:V], dtype=np.float64)
    N = float(c.sum())
    strings = [vocab.tokens[x] + vocab.tokens[y] for x, y in zip(a, b)]
    table = CodecTable.from_bytes(strings, pos_dim=vocab.pos_dim)
    rng = np.random.default_rng(shuffle_seed + 1)
    stand = rng.integers(0, V, size=len(strings))
    standin = CodecTable(table.byte_buffer[0:0], table.lengths[0:0], pos_dim=vocab.pos_dim) if False else \
        CodecTable.from_bytes([vocab.tokens[i] for i in stand], pos_dim=vocab.pos_dim)
    info = HeldoutInfo(ids=np.arange(V, V + len(strings)), a=a, b=b, count_m=cnt, count_a=c[a], table=table,
                       standin=standin, n_train_tokens=N, n_vocab=V,
                       mean_log_prior=float(np.log((c + 1.0) / (N + V)).mean()), train_counts=c)
    return info, keys, owner


@torch.no_grad()
def rows_chunked(model, info: HeldoutInfo, prior_rule: str, corr: str, chunk: int = 65536):
    """Rows for many minted tokens, built in chunks to bound memory."""
    head = model.head
    beta = getattr(head, "beta", None)
    prior = prior_for(prior_rule, info, beta)
    Es, bs = [], []
    n = len(info.ids)
    for s in range(0, n, chunk):
        sl = slice(s, min(n, s + chunk))
        sub = CodecTable(info.table.byte_buffer[sl], info.table.lengths[sl], pos_dim=info.table.pos_dim)
        stand = CodecTable(info.standin.byte_buffer[sl], info.standin.lengths[sl], pos_dim=info.table.pos_dim)
        E, b = head.rows_for(sub, None if prior is None else prior[sl], torch.as_tensor(info.a[sl]), corr,
                             gen_bytes_src=stand if head.kind == "kasg_shuf" else None)
        Es.append(E.float()); bs.append(b.float())
    return torch.cat(Es), torch.cat(bs)


@torch.no_grad()
def growth_eval(model, vocab, windows, stream, info_full, keys_full, owner_full, Ks, rule, device,
                target_bytes_all, seq_len) -> list[dict]:
    name, prior_rule, corr = rule
    H, LSE, ZT, (E_in, E, b) = hidden_states(model, windows, device)
    nb = torch.as_tensor(target_bytes_all[windows[:, 1:]], device=device, dtype=torch.float64)
    total_bytes = float(nb.sum())
    nll0 = (LSE - ZT).double()
    E_all, b_all = rows_chunked(model, info_full, prior_rule, corr)
    E_all, b_all = E_all.to(device), b_all.to(device)
    y = windows[:, 1:]
    pk = y[:, :-1].astype(np.int64) * vocab.n_total + y[:, 1:]
    pos = np.minimum(np.searchsorted(keys_full, pk), len(keys_full) - 1)
    hit = keys_full[pos] == pk
    site_owner = np.where(hit, owner_full[pos], -1)            # candidate index per (w, j)
    flatH = H.view(-1, H.shape[-1])
    out = []
    w0, v0 = eval_windows_full(np.asarray(stream, dtype=np.int64), seq_len)
    tot_nll, tot_b = 0.0, 0.0
    for s in range(0, len(w0), 4):
        w = torch.as_tensor(w0[s:s + 4], dtype=torch.long, device=device)
        with autocast_for(device):
            h = model.body(w[:, :-1], E_in)
        z0 = h.float() @ E.float().t() + (b.float() if b is not None else 0.0)
        nll = torch.nn.functional.cross_entropy(z0.view(-1, z0.shape[-1]), w[:, 1:].reshape(-1), reduction="none")
        nbt = torch.as_tensor((target_bytes_all[w0[s:s + 4, 1:]] * v0[s:s + 4]).reshape(-1), device=device,
                              dtype=torch.float64)
        tot_nll += float((nll.double() * (nbt > 0)).sum()); tot_b += float(nbt.sum())
    out.append({"rule": name, "K": 0, "vocab_total": int(vocab.n_train_vocab), "leak_bpb": 0.0, "n_sites": 0,
                "mean_regret_bits": None, "frac_regret_below_zero": None, "retok_bpb": tot_nll / (LN2 * tot_b),
                "retok_bytes": tot_b, "retok_tokens": int(len(stream)), "base_tokens": int(len(stream)), "wall_s": 0.0})
    for K in Ks:
        t0 = time.time()
        En, bn = E_all[:K], b_all[:K]
        lse_new = torch.empty(flatH.shape[0], device=device)
        for s in range(0, flatH.shape[0], 4096):
            acc = None
            for c in range(0, K, 65536):   # fp32 throughout, like the minted logits below
                z = flatH[s:s + 4096] @ En[c:c + 65536].t() + bn[c:c + 65536]
                lz = torch.logsumexp(z, -1)
                acc = lz if acc is None else torch.logaddexp(acc, lz)
            lse_new[s:s + 4096] = acc
        lse_g = torch.logaddexp(LSE.view(-1), lse_new).view_as(LSE)
        delta = (lse_g - LSE).double()
        own = torch.as_tensor(site_owner, device=device)
        is_site = torch.zeros_like(LSE, dtype=torch.bool)
        is_site[:, :-1] = (own >= 0) & (own < K)
        leak = float((delta * ((~is_site) & (nb > 0))).sum() / (LN2 * total_bytes))
        sw, sj = torch.nonzero(is_site, as_tuple=True)
        mids = own[sw, sj]
        hs = H[sw, sj].float()
        zm = (hs * En[mids]).sum(-1) + bn[mids]
        regret = ((lse_g[sw, sj] - zm) - nll0[sw, sj] - nll0[sw, sj + 1]) / LN2
        # re-tokenise the SAME text with all K merges (full tiling, last window padded) and
        # score the grown vocabulary in fp32
        sel = owner_full < K
        st = apply_pair_merges(np.asarray(stream, dtype=np.int64), keys_full[sel],
                               owner_full[sel] + vocab.n_train_vocab, vocab.n_total)
        win, valid = eval_windows_full(st, seq_len)
        lens_new = np.asarray(info_full.table.lengths[:K])
        tb_all = np.concatenate([target_bytes_all[:vocab.n_train_vocab], lens_new])
        new_in = TorchCodec(info_full.table, device).matmul(model.W_in, torch.arange(K, device=device))
        E_in_all = torch.cat([E_in[:vocab.n_train_vocab], new_in])
        tot_nll, tot_b = 0.0, 0.0
        for s in range(0, len(win), 4):
            w = torch.as_tensor(win[s:s + 4], dtype=torch.long, device=device)
            x, yy = w[:, :-1], w[:, 1:]
            with autocast_for(device):
                h = model.body(x, E_in_all)
            h = h.float().reshape(-1, h.shape[-1])
            z0 = h @ E.float().t() + (b.float() if b is not None else 0.0)
            l0 = torch.logsumexp(z0, -1)
            acc = None
            for c in range(0, K, 65536):
                lz = torch.logsumexp(h @ En[c:c + 65536].t() + bn[c:c + 65536], -1)
                acc = lz if acc is None else torch.logaddexp(acc, lz)
            lse = torch.logaddexp(l0, acc)
            yf = yy.reshape(-1)
            zt = torch.where(yf < vocab.n_train_vocab, z0.gather(1, yf.clamp(max=vocab.n_train_vocab - 1)[:, None]).squeeze(1),
                             (h * En[(yf - vocab.n_train_vocab).clamp(min=0)]).sum(-1) + bn[(yf - vocab.n_train_vocab).clamp(min=0)])
            nbt = torch.as_tensor((tb_all[win[s:s + 4, 1:]] * valid[s:s + 4]).reshape(-1), device=device,
                                  dtype=torch.float64)
            tot_nll += float(((lse - zt).double() * (nbt > 0)).sum())
            tot_b += float(nbt.sum())
        out.append({"rule": name, "K": int(K), "vocab_total": int(vocab.n_train_vocab + K),
                    "leak_bpb": leak, "n_sites": int(len(sw)),
                    "mean_regret_bits": float(regret.mean()) if len(sw) else None,
                    "frac_regret_below_zero": float((regret < 0).double().mean()) if len(sw) else None,
                    "retok_bpb": tot_nll / (LN2 * tot_b), "retok_bytes": tot_b,
                    "retok_tokens": int(len(st)), "base_tokens": int(len(stream)), "wall_s": time.time() - t0})
        print(json.dumps(out[-1]), flush=True)
    return out


GROWTH_RULES = {
    "dense": [("inherit", "logpba", "inherit")],
    "kasp": [("count", "count", "none")],
    "kasu": [("count", "count", "none"), ("inherit", "inherit", "inherit")],
    "kasg": [("count", "count", "none"), ("gen_count", "count", "generate")],
    "kasg_shuf": [("gen_count", "count", "generate")],
    "kas0": [("bytes", "none", "none")],
}


def main(data_dir: Path, work: Path, runs: list[str], input_runs_root: str | None = None,
         Ks=(1000, 10_000, 100_000, 300_000, 721_938), n_windows: int = 256, **_) -> dict:
    from .analysis import load_run
    from .train import attention_context
    device = torch.device("cuda")
    data = TokenData.load(Path(data_dir), verify=False)
    growth = dict(np.load(Path(data_dir) / "growth_candidates.npz"))
    pair_count = np.load(Path(data_dir) / "heldout_pair_count_train.npy")
    res = {}
    root = Path(input_runs_root) if input_runs_root else Path(work) / "runs"
    for r in runs:
        rd = root / r
        if not (rd / "model.pt").exists():
            matches = list(Path("/kaggle/input").rglob(f"{r}/model.pt"))
            if not matches:
                res[r] = "missing"
                continue
            rd = matches[0].parent
        model, cfg, vocab, unigram = load_run(rd, Path(data_dir), device)
        Kmax = max(Ks)
        info, keys, owner = candidate_info(vocab, unigram, pair_count, growth, Kmax)
        Ks_eff = sorted({min(k, len(info.ids)) for k in Ks})
        windows = eval_windows(data.dev, cfg.seq_len, n_windows)
        stream = np.asarray(data.dev[: n_windows * cfg.seq_len + 1])
        ctx = attention_context(device); ctx.__enter__()
        try:
            res[r] = []
            for rule in GROWTH_RULES[cfg.head]:
                res[r] += growth_eval(model, vocab, windows, stream, info, keys, owner, Ks_eff, rule, device,
                                      vocab.target_bytes(), cfg.seq_len)
        finally:
            ctx.__exit__(None, None, None)
        out = Path(work) / "runs" / "growth"
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{r}.json").write_text(json.dumps(res[r], indent=2), encoding="utf-8")
        del model
        torch.cuda.empty_cache()
    return {k: (len(v) if isinstance(v, list) else v) for k, v in res.items()}
