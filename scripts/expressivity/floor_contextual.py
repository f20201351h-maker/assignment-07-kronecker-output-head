"""Contextual floor of fixed-feature log-linear heads, measured against a trained dense head.

Reference distribution: the prior-matched Dense model of protocol 08 (`p4-densep-s1`, test bpb 1.3986),
evaluated on dev-split contexts. For every sampled context x we take the model's full next-token
distribution p_x and fit the best additive (KAS-family) logit vector  Phi s_x + b  to it. The weighted
average of  KL(p_x || q*_x)  is the representational floor of that family *if p_x were the truth*:
no body, width, or nonlinearity feeding s can get a KAS-type head closer to p_x than this.

Alongside, on the SAME positions, we report the realised dev NLL of the historical arms
(Dense-prior, Dense, KAS-P, KAS-U16, KAS-U64, KAS-G, KAS-0) from their saved per-position arrays, so the
representational floor can be compared with the gaps that training actually produced.

    python scripts/expressivity/floor_contextual.py --windows 16 --per-window 128
      -> results/expressivity/floor_contextual.json (+ .npz with per-context arrays)
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
from floor_lib import (ROOT, LN2, NATS_PER_TARGET_TO_BPB, FeatureSet, fam_hash_signed, fam_kas, fam_ngrams,  # noqa: E402
                       fam_prefix, fam_private, fam_suffix_cells, fit_floor, load_vocab, unigram_bias)
from kq5.data import eval_windows  # noqa: E402
from kq5.train import RunConfig, build_model  # noqa: E402

CK = ROOT / "checkpoints/p4-densep-s1/model.pt"
OUT = ROOT / "results/expressivity/floor_contextual.json"
ARMS = {"Dense-prior": "experiments/modal-p4/runs/p4-densep-s1/nll_dev.npy",
        "Dense": "experiments/modal-m3/runs/m3-dense-s1/nll_dev.npy",
        "KAS-U64": "experiments/modal-p2/runs/p2-kasu64-s1/nll_dev.npy",
        "KAS-U16": "experiments/modal-m3/runs/m3-kasu0-s1/nll_dev.npy",
        "KAS-G": "experiments/modal-m3/runs/m3-kasg-s1/nll_dev.npy",
        "KAS-P": "experiments/modal-m3/runs/m3-kasp-s1/nll_dev.npy",
        "KAS-0": "experiments/modal-m3/runs/m3-kas0-s1/nll_dev.npy"}


def load_dense_prior(vocab, unigram_full):
    ck = torch.load(CK, map_location="cpu", weights_only=False)
    cfg = RunConfig(**ck["config"])
    assert cfg.head == "dense" and cfg.dense_bias, cfg
    model, prior = build_model(cfg, vocab, unigram_full)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, cfg


@torch.no_grad()
def hidden_and_logits(model, windows: np.ndarray, positions: np.ndarray, device="cpu"):
    """Final hidden states and dense-prior logits at the chosen positions of each window."""
    E_in = model.input_table_rows(None)
    W, beta = model.head.W, model.head.beta
    H, Z, Y = [], [], []
    for i in range(windows.shape[0]):
        w = torch.as_tensor(windows[i:i + 1], dtype=torch.long, device=device)
        x, y = w[:, :-1], w[:, 1:]
        h = model.body(x, E_in)[0, positions].float()                 # (P, d)
        z = h @ W.float().t() + beta.float()                            # (P, V)
        H.append(h)
        Z.append(z)
        Y.append(y[0, positions])
        print(f"    window {i+1}/{windows.shape[0]} done", flush=True)
    return torch.cat(H), torch.cat(Z), torch.cat(Y)


CONFIGS = {
    "kas": ["kas"],
    "kas+suffix": ["kas", "suffix_cells"],
    "kas+trigram_free": ["kas", "trigram_free_h8192"],
    "kas+prefix": ["kas", "prefix_ge5"],
    "kas+private4000": ["kas", "private_top4000"],
    "kas+hash8192": ["kas", "hash_m8192_k8"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", type=int, default=16)
    ap.add_argument("--per-window", type=int, default=128)
    ap.add_argument("--iters", type=int, default=250)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--configs", nargs="*", default=["kas", "kas+suffix", "kas+trigram_free", "kas+private4000", "kas+hash8192"])
    ap.add_argument("--enrich-contexts", type=int, default=1024, help="contexts used for the non-base configs")
    ap.add_argument("--ckpt", default=None, help="reference checkpoint (default: p4-densep-s1)")
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--out", default=None, help="output JSON name (default floor_contextual.json)")
    ap.add_argument("--base-only", action="store_true", help="only the beta-fixed and log-unigram base fits")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    device = args.device
    global CK, OUT, ARMS
    if args.ckpt:
        CK = Path(args.ckpt).resolve()
        run_name = CK.parent.name                       # e.g. p4-densep-s2: use that run's own saved NLL array
        if (ROOT / "experiments/modal-p4/runs" / run_name).exists():
            ARMS = dict(ARMS, **{"Dense-prior": f"experiments/modal-p4/runs/{run_name}/nll_dev.npy"})
    if args.out:
        OUT = ROOT / "results/expressivity" / args.out
    if args.split == "test":
        ARMS = {k: v.replace("nll_dev.npy", "nll_test.npy") for k, v in ARMS.items()}
    torch.set_num_threads(8)
    t0 = time.time()
    vocab, table, unigram = load_vocab()
    V = vocab.n_train_vocab
    tokens = vocab.tokens[:V]
    unigram_full = np.load(ROOT / "data/tokens/unigram_train.npy")
    dev = np.load(ROOT / f"data/tokens/{args.split}.npy", mmap_mode="r")
    windows_all = eval_windows(dev, 1024)
    n_win = windows_all.shape[0]
    win_idx = np.linspace(0, n_win - 1, args.windows).round().astype(int)
    positions = np.linspace(0, 1023, args.per_window).round().astype(int)
    windows = windows_all[win_idx]
    nbytes_all = np.array([len(t) for t in vocab.tokens], dtype=np.int64)

    print("loading Dense-prior checkpoint and running forward passes ...", flush=True)
    model, cfg = load_dense_prior(vocab, unigram_full)
    model.to(device)
    H, Z, Y = hidden_and_logits(model, windows, positions, device)
    N = Z.shape[0]
    Y_cpu = Y.cpu()
    y_np = Y_cpu.numpy()
    nb = nbytes_all[y_np]
    valid = nb > 0                                                        # EOS targets carry no bytes
    P_all = F.softmax(Z, dim=1)
    H_all = -(P_all * F.log_softmax(Z, dim=1)).sum(1).cpu().numpy()     # entropy of the reference per context
    nll_ref_here = F.cross_entropy(Z, Y, reduction="none").cpu().numpy() # dense-prior NLL recomputed here
    del H, Z

    # realised NLL of every historical arm at these exact positions
    arms = {}
    for name, rel in ARMS.items():
        a = np.load(ROOT / rel, mmap_mode="r")
        arms[name] = np.asarray(a[win_idx][:, positions]).reshape(-1).astype(np.float64)
    res = {"provenance": f"RECOMPUTED on CPU from checkpoint {CK.parent.name} on {args.split}-split "
                         "contexts; floors are KL(p_dense_prior || best additive fit), i.e. representational floors "
                         "conditional on the dense-prior distribution being the truth",
           "checkpoint": str(CK.relative_to(ROOT)) if CK.is_relative_to(ROOT) else str(CK), "n_contexts": int(N),
           "windows": win_idx.tolist(),
           "positions_per_window": int(args.per_window), "n_valid_byte_targets": int(valid.sum()),
           "bytes_per_target_here": float(nb[valid].mean()),
           "dense_prior_nll_recomputed_cpu": float(nll_ref_here[valid].mean()),
           "dense_prior_nll_saved_gpu": float(arms["Dense-prior"][valid].mean()),
           "reference_entropy_nats_mean": float(H_all[valid].mean()),
           "arms_nll_nats_per_target_same_positions": {k: float(v[valid].mean()) for k, v in arms.items()},
           "arms_bpb_same_positions": {k: float(v[valid].sum() / (LN2 * nb[valid].sum())) for k, v in arms.items()},
           "floors": {}}
    print(json.dumps({k: res[k] for k in ("dense_prior_nll_recomputed_cpu", "dense_prior_nll_saved_gpu",
                                          "arms_nll_nats_per_target_same_positions")}, indent=1), flush=True)

    print("building feature families ...", flush=True)
    fams = {"kas": fam_kas(table), "suffix_cells": fam_suffix_cells(table, tokens),
            "trigram_free_h8192": fam_ngrams(table, tokens, 3, False, 8192), "prefix_ge5": fam_prefix(tokens, V, 2, 8, 5),
            "private_top4000": fam_private(unigram, V, 4000), "hash_m8192_k8": fam_hash_signed(tokens, V, 8192, 8)}
    beta = model.head.beta.detach().float().cpu()                         # dense-prior's learned per-token bias
    b_uni = unigram_bias(unigram, V)
    per_ctx = {}

    def run(name, bias, free, n_ctx, tag):
        fs = FeatureSet(V)
        for fam in CONFIGS[name]:
            fs.add_family(fam, *fams[fam])
        idx_ctx = np.arange(min(n_ctx, N))
        floors = np.zeros(len(idx_ctx)); mimic_nll = np.zeros(len(idx_ctx))
        t1 = time.time()
        for s in range(0, len(idx_ctx), args.chunk):
            sel = idx_ctx[s:s + args.chunk]
            P = P_all[torch.as_tensor(sel, device=P_all.device)]
            r = fit_floor(fs, P, torch.full((len(sel),), 1.0 / len(sel)), bias, free_bias=free, steps=args.iters,
                          lbfgs=True, device=device)
            floors[s:s + len(sel)] = r["floor_per_ctx"]
            # the fitted mimic's NLL at the TRUE targets of these contexts
            with torch.no_grad():
                idx, w = fs.tensors()
                z = F.embedding_bag(idx, r["S"], per_sample_weights=w, mode="sum")      # (V, n)
                z = z + (r["b"].unsqueeze(1) if r["b"] is not None else (bias.unsqueeze(1) if bias is not None else 0))
                lsm = F.log_softmax(z, dim=0)
                mimic_nll[s:s + len(sel)] = -lsm[Y_cpu[sel], torch.arange(len(sel))].numpy()
            print(f"    [{tag}] chunk {s // args.chunk + 1}: floor {r['floor_per_ctx'].mean():.4f} nats "
                  f"({time.time() - t1:.0f}s)", flush=True)
        v = valid[idx_ctx]
        out = {"n_contexts": int(len(idx_ctx)), "n_features": fs.n_features - 1,
               "floor_nats_per_target": float(floors[v].mean()),
               "floor_bpb_equiv_global_ratio": float(floors[v].mean() * NATS_PER_TARGET_TO_BPB),
               "floor_bpb_byte_weighted_here": float(floors[v].sum() / (LN2 * nb[idx_ctx][v].sum())),
               "mimic_nll_true_targets": float(mimic_nll[v].mean()),
               "dense_prior_nll_true_targets_same": float(nll_ref_here[idx_ctx][v].mean()),
               "mimic_minus_dense_prior_nats": float(mimic_nll[v].mean() - nll_ref_here[idx_ctx][v].mean()),
               "arms_nll_same_subset": {k: float(a[idx_ctx][v].mean()) for k, a in arms.items()},
               "floor_quantiles_nats": [float(np.percentile(floors[v], q)) for q in (10, 25, 50, 75, 90)],
               "seconds": round(time.time() - t1, 1)}
        per_ctx[tag] = {"floors": floors, "mimic_nll": mimic_nll}
        res["floors"][tag] = out
        print(f"  {tag}: floor {out['floor_nats_per_target']:.4f} nats/target "
              f"({out['floor_bpb_equiv_global_ratio']:.4f} bpb-equiv); mimic-dense {out['mimic_minus_dense_prior_nats']:.4f}", flush=True)
        OUT.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")

    # base family, three bias treatments
    run("kas", beta, False, N, "kas | bias = dense-prior's learned beta (fixed)")
    run("kas", b_uni, False, N, "kas | bias = log-unigram prior (fixed; KAS-P family)")
    if not args.base_only:
        run("kas", beta, True, N, "kas | bias = free per-token, init beta (looser lower bound)")
    for name in ([] if args.base_only else args.configs):
        if name == "kas":
            continue
        run(name, beta, False, args.enrich_contexts, f"{name} | bias = beta (fixed)")
    res["seconds_total"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    np.savez_compressed(OUT.with_suffix(".npz"), y=y_np, nbytes=nb, valid=valid, ref_entropy=H_all, ref_nll=nll_ref_here,
                        **{f"arm_{k}": v for k, v in arms.items()},
                        **{f"floor_{i}": d["floors"] for i, d in enumerate(per_ctx.values())},
                        **{f"mimic_{i}": d["mimic_nll"] for i, d in enumerate(per_ctx.values())},
                        tags=np.array(list(per_ctx.keys())))
    print("wrote", OUT, f"({res['seconds_total']}s)")


if __name__ == "__main__":
    main()
