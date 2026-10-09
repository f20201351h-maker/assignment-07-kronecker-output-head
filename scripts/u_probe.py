"""Probe: is what KAS-U16 stores per token a function of the token's spelling?

Targets (per token, training vocabulary, EOS excluded):
  dU_eff = C^T (U_v - U_v,init)    the learned correction in output space (d = 512)
  dbeta  = beta_v - beta_v,init     the learned change of the prior
and, for comparison, the Dense head's learned rows dW_v = W_v - W_v,init.
Initial values are rebuilt exactly from the run's config and seed. Tokens are split 90/10
at random; models: ridge on the Kronecker code (what a KAS head can express), the ByteCNN
generator, linear + CNN residual, and the same with shuffled spellings. Held-out R2,
unweighted and weighted by training count, plus R2 restricted to tokens seen >= 100 times.

    python scripts/u_probe.py --kasu experiments/.../r3-kasu-s1 --dense experiments/.../r3-dense-s1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from kq5.codec import CodecTable  # noqa: E402
from kq5.train import RunConfig, build_model  # noqa: E402
from kq5.vocab import WorkingVocab  # noqa: E402
from r0b_predictability import study  # noqa: E402


def learned_delta(run_dir: Path, vocab, unigram):
    ck = torch.load(run_dir / "model.pt", map_location="cpu", weights_only=False)
    cfg = RunConfig(**ck["config"])
    init, _ = build_model(cfg, vocab, unigram)         # same seed -> same initial tensors
    sd0 = init.state_dict()
    sd = ck["model"]
    return cfg, sd, sd0


EDGES = np.array([0, 1, 10, 100, 1_000, 10_000, 100_000, 1_000_000, np.inf])


def movement_by_bin(delta: np.ndarray, init: np.ndarray, counts: np.ndarray) -> list[dict]:
    """Mean row movement ||delta_v|| / ||init_v|| by training-count bin (review S2: are rare-token
    rows of a stored table damped?)."""
    rel = np.linalg.norm(delta, axis=1) / np.maximum(np.linalg.norm(init, axis=1), 1e-12)
    b = np.clip(np.searchsorted(EDGES, counts, side="right") - 1, 0, len(EDGES) - 2)
    return [{"bin": f"[{int(EDGES[k])},{'inf' if np.isinf(EDGES[k + 1]) else int(EDGES[k + 1])})",
             "n": int((b == k).sum()), "mean_rel_move": float(rel[b == k].mean()) if np.any(b == k) else None}
            for k in range(len(EDGES) - 1)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kasu", nargs="*", default=[])
    ap.add_argument("--dense", nargs="*", default=[])
    ap.add_argument("--out", default=str(ROOT / "results" / "probe" / "u_probe.json"))
    ap.add_argument("--epochs", type=int, default=20)
    args = ap.parse_args()
    torch.set_num_threads(8)
    tok = ROOT / "data" / "tokens"
    vocab = WorkingVocab.load(tok / "vocab.json")
    unigram = np.load(tok / "unigram_train.npy")
    V = vocab.n_train_vocab
    table = CodecTable.from_bytes(vocab.tokens[:V], pos_dim=vocab.pos_dim)
    ids = np.array([i for i in range(V) if i != vocab.eos_id])
    w = unigram[ids] + 1.0
    res = {}
    for rd in args.kasu:
        cfg, sd, sd0 = learned_delta(Path(rd), vocab, unigram)
        C = sd["head.C"].double()
        dU = (sd["head.U"] - sd0["head.U"]).double()
        eff = (dU @ C).numpy()[ids]
        db = (sd["head.beta"] - sd0["head.beta"]).double().numpy()[ids][:, None]
        r = {"run": cfg.run_id,
             "norm_ratio_dU_over_U0": float(dU.norm() / sd0["head.U"].double().norm()),
             "dbeta_abs_mean": float(np.abs(db).mean()),
             # sanity: if the initial tensors were rebuilt correctly, U stays correlated with U0
             "corr_U_U0": float(np.corrcoef(sd["head.U"].double().numpy().ravel(),
                                            sd0["head.U"].double().numpy().ravel())[0, 1])}
        r["U_movement_by_count_bin"] = movement_by_bin(dU.numpy(), sd0["head.U"].double().numpy(), unigram[:V])
        if float(sd0["head.U"].norm()) == 0.0:      # u_zero init: ratios to the initial U are undefined
            r.update(norm_ratio_dU_over_U0=None, corr_U_U0=None, U_movement_by_count_bin=None)
        r["dU_eff"] = study("kasu_dU_eff", table, ids, eff, w, epochs=args.epochs)
        r["dbeta"] = study("kasu_dbeta", table, ids, db, w, epochs=args.epochs)
        res[cfg.run_id] = r
        print(json.dumps({k: v for k, v in r.items() if not isinstance(v, dict)}), flush=True)
    for rd in args.dense:
        cfg, sd, sd0 = learned_delta(Path(rd), vocab, unigram)
        dW_all = (sd["head.W"] - sd0["head.W"]).double().numpy()
        dW = dW_all[ids]
        res[cfg.run_id] = {"run": cfg.run_id,
                           "W_movement_by_count_bin": movement_by_bin(dW_all, sd0["head.W"].double().numpy(), unigram[:V]),
                           "dW": study("dense_dW", table, ids, dW, w, epochs=args.epochs)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
