"""Corpus, vocabulary, merge and parameter statistics quoted in the README (CPU, no training).

    python scripts/setup_stats.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from kq5.data import TokenData, eval_windows  # noqa: E402
from kq5.minting import find_sites  # noqa: E402
from kq5.model import param_report  # noqa: E402
from kq5.train import RunConfig, build_model  # noqa: E402
from kq5.vocab import WorkingVocab  # noqa: E402
from make_jobs import BACKBONES, GEN_VARIANTS  # noqa: E402


def main() -> None:
    tok = ROOT / "data" / "tokens"
    data = TokenData.load(tok, verify=True)
    vocab = WorkingVocab.load(tok / "vocab.json")
    unigram = np.load(tok / "unigram_train.npy")
    m = data.manifest
    meta = json.loads((ROOT / "data" / "raw" / "download_meta.json").read_text(encoding="utf-8"))
    g = np.load(tok / "growth_candidates.npz")
    out = {"corpus": {"dataset": meta["dataset"], "docs": meta["docs"], "gpt2_tokens": meta["tokens"],
                      "base_tokens": m["base_tokens"], "merged_tokens": m["tokens"],
                      "train_compression_pct": 100 * (1 - m["tokens"]["train"] / m["base_tokens"]["train"])},
           "vocab": {"gpt2": 50257, "dropped_long": len(m["dropped_gpt2_long_tokens"]), "n_base": vocab.n_base,
                     "n_trained_merges": vocab.n_trained, "n_heldout_merges": vocab.n_heldout,
                     "n_train_vocab": vocab.n_train_vocab, "n_total_with_heldout": vocab.n_total,
                     "growth_candidates": int(len(g["a"])),
                     "vocab_after_full_growth": int(vocab.n_total + len(g["a"])),
                     "gpt2_decode_equals_fffd": m["gpt2_decode_fffd"]["decode_equals_fffd"],
                     "train_tokens_never_seen_in_vocab": int((unigram[:vocab.n_train_vocab] == 0).sum()),
                     "kasu_stored_budget": int(vocab.n_train_vocab * 17)},
           "sites": {}, "params": {}}
    for split, stream in (("dev", data.dev), ("test", data.test)):
        w = eval_windows(stream, 1024)
        s = find_sites(w, vocab)
        out["sites"][split] = {"windows": int(len(w)), "targets": int(w[:, 1:].size), "heldout_sites": int(len(s["w"])),
                               "distinct_heldout_merges_present": int(len(np.unique(s["m"]))),
                               "windows_with_sites": int(len(np.unique(s["w"])))}
    for head in ("dense", "kas0", "kasp", "kasu", "kasg", "kasg_shuf"):
        variants = ["v1", "v3"] if head.startswith("kasg") else [None]
        for gv in variants:
            cfg = RunConfig(run_id="x", head=head, **BACKBONES["8L512"], generator=GEN_VARIANTS[gv][0] if gv else {})
            model, _ = build_model(cfg, vocab, unigram)
            out["params"][head + (f"-{gv}" if gv else "")] = param_report(model)
    (ROOT / "results" / "setup").mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / "setup" / "setup_stats.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "params"}, indent=1))
    for k, v in out["params"].items():
        print(k, v)


if __name__ == "__main__":
    main()
