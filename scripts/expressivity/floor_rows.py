"""Floors for DENSE per-token row sets against the Dense-prior model on the same dev contexts.

Each row set E (V x d) is treated as a fixed feature matrix: the floor is  min_s KL(p_x || softmax(E s + b))
per context, s free in R^d. Row sets:

  dense_prior_W       the reference model's own rows (sanity: floor must be ~0)
  read_to_write_k{K}  candidate B: the Dense-prior body's first K blocks applied to the isolated token
                      (K = 0 is the tied input projection P kappa_v, linear in the code -> >= KAS floor)
  kasu64_rows         materialised rows of the trained KAS-U64 head (p2-kasu64-s1): K W_out + U C, with its bias
  kasu16_rows         same for KAS-U16 (m3-kasu0-s1)
  dense_nobias_W      rows of the bias-free Dense model (m3-dense-s1)

    python scripts/expressivity/floor_rows.py --windows 8 --per-window 128 --iters 200
      -> results/expressivity/floor_rows.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from floor_lib import ROOT, LN2, NATS_PER_TARGET_TO_BPB, load_vocab, unigram_bias  # noqa: E402
from floor_contextual import CK, hidden_and_logits, load_dense_prior  # noqa: E402
from kq5.data import eval_windows  # noqa: E402
from kq5.train import RunConfig, build_model  # noqa: E402

OUT = ROOT / "results/expressivity/floor_rows.json"
CK_U64 = ROOT / "checkpoints/p2-kasu64-s1/model.pt"
CK_U16 = ROOT / "checkpoints/m3-kasu0-s1/model.pt"
CK_DENSE = ROOT / "checkpoints/m3-dense-s1/model.pt"


def load_any(ck_path, vocab, unigram_full):
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    cfg = RunConfig(**ck["config"])
    model, _ = build_model(cfg, vocab, unigram_full)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, cfg


@torch.no_grad()
def read_rows(model, k: int) -> torch.Tensor:
    """Depth-k reading features of every isolated token: blocks 1..k on a length-1 sequence."""
    E_in = model.input_table_rows(None)[:model.n_vocab]                 # (V, d) = P kappa_v
    x = E_in + model.pos[:1]
    out = []
    for s in range(0, x.shape[0], 4096):
        h = x[s:s + 4096].unsqueeze(1)                                   # (n, 1, d)
        for blk in model.blocks[:k]:
            h = blk(h)
        if k > 0:
            h = model.ln_f(h)
        out.append(h[:, 0].float())
    return torch.cat(out)


def fit_dense_floor(E: torch.Tensor, b: torch.Tensor | None, P: torch.Tensor, Y: torch.Tensor, iters: int, chunk: int):
    """min_s KL(P_x || softmax(E s_x + b)) per context, LBFGS; returns per-context floor and mimic NLL."""
    N, V = P.shape
    floors = np.zeros(N); mimic = np.zeros(N)
    H = -(P * torch.log(P.clamp_min(1e-30))).sum(1)
    for s0 in range(0, N, chunk):
        Pc = P[s0:s0 + chunk]
        n = Pc.shape[0]
        S = torch.zeros(E.shape[1], n, requires_grad=True)
        opt = torch.optim.LBFGS([S], lr=1.0, max_iter=iters, history_size=50, line_search_fn="strong_wolfe",
                                tolerance_grad=1e-9, tolerance_change=1e-12)
        Pt = Pc.t().contiguous()

        def logits():
            z = E @ S                                                    # (V, n)
            return z + b.unsqueeze(1) if b is not None else z

        def closure():
            opt.zero_grad(set_to_none=True)
            ce = -(Pt * F.log_softmax(logits(), dim=0)).sum(0)
            loss = ce.mean()
            loss.backward()
            return loss
        opt.step(closure)
        with torch.no_grad():
            lsm = F.log_softmax(logits(), dim=0)
            ce = -(Pt * lsm).sum(0)
            floors[s0:s0 + n] = (ce - H[s0:s0 + n]).numpy()
            mimic[s0:s0 + n] = -lsm[Y[s0:s0 + n], torch.arange(n)].numpy()
    return floors, mimic


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", type=int, default=8)
    ap.add_argument("--per-window", type=int, default=128)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--ks", nargs="*", type=int, default=[0, 1, 2, 4, 8])
    args = ap.parse_args()
    torch.set_num_threads(8)
    t0 = time.time()
    vocab, table, unigram = load_vocab()
    V = vocab.n_train_vocab
    unigram_full = np.load(ROOT / "data/tokens/unigram_train.npy")
    dev = np.load(ROOT / "data/tokens/dev.npy", mmap_mode="r")
    windows_all = eval_windows(dev, 1024)
    win_idx = np.linspace(0, windows_all.shape[0] - 1, args.windows).round().astype(int)
    positions = np.linspace(0, 1023, args.per_window).round().astype(int)
    windows = windows_all[win_idx]
    nbytes_all = np.array([len(t) for t in vocab.tokens], dtype=np.int64)

    print("reference forward passes ...", flush=True)
    ref, _ = load_dense_prior(vocab, unigram_full)
    _, Z, Y = hidden_and_logits(ref, windows, positions)
    P = F.softmax(Z, dim=1)
    nll_ref = F.cross_entropy(Z, Y, reduction="none").numpy()
    nb = nbytes_all[Y.numpy()]
    valid = nb > 0
    beta = ref.head.beta.detach().float()
    res = {"provenance": "RECOMPUTED on CPU; floors of dense row sets against the Dense-prior (p4-densep-s1) distribution "
                         "on the same dev contexts as floor_contextual.json", "n_contexts": int(len(Y)),
           "windows": win_idx.tolist(), "dense_prior_nll": float(nll_ref[valid].mean()), "rows": {}}

    row_sets = {}
    row_sets["dense_prior_W (sanity)"] = (ref.head.W.detach().float(), beta)
    for k in args.ks:
        row_sets[f"read_to_write_k{k}"] = (read_rows(ref, k), beta)
    if CK_U64.exists():
        m, _ = load_any(CK_U64, vocab, unigram_full)
        E, b = m.head.materialize()
        row_sets["kasu64_rows (own bias)"] = (E.detach().float(), b.detach().float())
        del m
    if CK_U16.exists():
        m, _ = load_any(CK_U16, vocab, unigram_full)
        E, b = m.head.materialize()
        row_sets["kasu16_rows (own bias)"] = (E.detach().float(), b.detach().float())
        del m
    if CK_DENSE.exists():
        m, _ = load_any(CK_DENSE, vocab, unigram_full)
        row_sets["dense_nobias_W (beta fixed)"] = (m.head.W.detach().float(), beta)
        del m

    for name, (E, b) in row_sets.items():
        t1 = time.time()
        fl, mi = fit_dense_floor(E, b, P, Y, args.iters, args.chunk)
        out = {"d": int(E.shape[1]), "floor_nats_per_target": float(fl[valid].mean()),
               "floor_bpb_equiv": float(fl[valid].mean() * NATS_PER_TARGET_TO_BPB),
               "mimic_minus_dense_prior_nats": float(mi[valid].mean() - nll_ref[valid].mean()),
               "floor_quantiles": [float(np.percentile(fl[valid], q)) for q in (10, 25, 50, 75, 90)],
               "seconds": round(time.time() - t1, 1)}
        res["rows"][name] = out
        print(f"  {name}: floor {out['floor_nats_per_target']:.4f} nats ({out['floor_bpb_equiv']:.4f} bpb-equiv); "
              f"mimic-dense {out['mimic_minus_dense_prior_nats']:+.4f} ({out['seconds']}s)", flush=True)
        OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    res["seconds_total"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
