"""Build the token arrays every arm trains and evaluates on.

Stage 1 (download): stream FineWeb-Edu sample-10BT in its published order,
split documents by a hash of their id into train / dev / test, tokenise with
GPT-2 BPE and store GPT-2 id streams (EOS after each document).

Stage 2 (build): working vocabulary (GPT-2 tokens <= 32 bytes), candidate
merges counted on the TRAIN split only, a frequency-stratified random split of
merges into trained / held-out, merge application, unigram counts, and a
manifest with sha256 of every file.

    python scripts/prepare_data.py download --train-tokens 250000000
    python scripts/prepare_data.py build
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kq5.gpt2bytes import load_gpt2_raw_bytes, decode_replacement_collisions  # noqa: E402
from kq5.vocab import WorkingVocab, count_pairs, count_keys, is_word_continuation  # noqa: E402

DATA = ROOT / "data"
RAW = DATA / "raw"
OUT = DATA / "tokens"
TOKENIZER_URL = "https://huggingface.co/openai-community/gpt2/resolve/main/tokenizer.json"
DATASET = ("HuggingFaceFW/fineweb-edu", "sample-10BT")


CHUNK = 16_000_000


def doc_chunks(stream: np.ndarray, eos: int | None, size: int = CHUNK):
    """[lo, hi) slices of about `size` tokens. With eos given, every slice ends just
    after an EOS (merges never span EOS), so chunked merging is exact."""
    n = len(stream)
    lo = 0
    while lo < n:
        hi = min(n, lo + size)
        if eos is not None and hi < n:
            seg = np.flatnonzero(stream[lo + size // 2:hi] == eos)
            if len(seg):
                hi = lo + size // 2 + int(seg[-1]) + 1
        yield lo, hi
        lo = hi


def chunked(stream: np.ndarray, fn, vocab_eos: int | None) -> np.ndarray:
    parts = [fn(np.asarray(stream[lo:hi])).astype(np.uint16) for lo, hi in doc_chunks(stream, vocab_eos)]
    return np.concatenate(parts)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def split_of(doc_id: str) -> str:
    h = int(hashlib.sha256(doc_id.encode("utf-8")).hexdigest()[:8], 16) % 1000
    if h < 5:
        return "test"
    if h < 10:
        return "dev"
    return "train"


def download(args) -> None:
    from datasets import load_dataset
    from tokenizers import Tokenizer

    RAW.mkdir(parents=True, exist_ok=True)
    tok_path = DATA / "gpt2" / "tokenizer.json"
    tok_path.parent.mkdir(parents=True, exist_ok=True)
    if not tok_path.exists():
        urllib.request.urlretrieve(TOKENIZER_URL, tok_path)
    tok = Tokenizer.from_file(str(tok_path))
    eos = tok.token_to_id("<|endoftext|>")
    ds = load_dataset(*DATASET, split="train", streaming=True)
    streams = {s: [] for s in ("train", "dev", "test")}
    counts = {s: 0 for s in streams}
    docs = {s: 0 for s in streams}
    doc_ids = {s: [] for s in streams}
    batch_text, batch_split, batch_id = [], [], []
    t0 = time.time()
    n_seen = 0

    def flush():
        enc = tok.encode_batch(batch_text, add_special_tokens=False)
        for e, s, i in zip(enc, batch_split, batch_id):
            ids = np.array(e.ids + [eos], dtype=np.uint16)
            streams[s].append(ids)
            counts[s] += len(ids)
            docs[s] += 1
            doc_ids[s].append(i)
        batch_text.clear(); batch_split.clear(); batch_id.clear()

    for row in ds:
        n_seen += 1
        s = split_of(row["id"])
        batch_text.append(row["text"]); batch_split.append(s); batch_id.append(row["id"])
        if len(batch_text) >= 2000:
            flush()
            if n_seen % 20000 == 0:
                rate = counts["train"] / (time.time() - t0)
                print(f"docs={n_seen} train={counts['train']:,} dev={counts['dev']:,} "
                      f"test={counts['test']:,} {rate:,.0f} tok/s", flush=True)
            if counts["train"] >= args.train_tokens:
                break
    if batch_text:
        flush()
    meta = {"dataset": list(DATASET), "docs_seen": n_seen, "tokens": counts, "docs": docs,
            "split_rule": "sha256(doc id)[:8] mod 1000: <5 test, <10 dev, else train",
            "tokenizer_url": TOKENIZER_URL, "tokenizer_sha256": sha256_file(tok_path),
            "eos_gpt2_id": eos}
    for s, parts in streams.items():
        arr = np.concatenate(parts)
        np.save(RAW / f"{s}_gpt2.npy", arr)
        meta[f"{s}_gpt2_sha256"] = sha256_file(RAW / f"{s}_gpt2.npy")
        (RAW / f"{s}_doc_ids.txt").write_text("\n".join(doc_ids[s]), encoding="utf-8")
    (RAW / "download_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


def build(args) -> None:
    rng = np.random.default_rng(args.seed)
    OUT.mkdir(parents=True, exist_ok=True)
    tok_path = DATA / "gpt2" / "tokenizer.json"
    gpt2_bytes, eos_g = load_gpt2_raw_bytes(tok_path)
    vocab = WorkingVocab.from_gpt2(gpt2_bytes, eos_g, args.pos_dim)
    long_ids = [g for g, b in enumerate(gpt2_bytes) if len(b) > args.pos_dim]
    print(f"base vocab {vocab.n_base} (dropped {len(long_ids)} GPT-2 tokens > {args.pos_dim} bytes)")

    base = {}
    for s in ("train", "dev", "test"):
        g = np.load(RAW / f"{s}_gpt2.npy")
        base[s] = chunked(g, lambda c: vocab.gpt2_stream_to_base(c.astype(np.int64)), vocab_eos=None)
        print(f"{s}: gpt2 {len(g):,} -> base {len(base[s]):,}", flush=True)
        del g

    # ---- candidate merges, TRAIN split only ----
    nb = vocab.n_base
    toks = vocab.tokens
    left_ok = np.array([i != vocab.eos_id and toks[i].strip(b" ") != b"" for i in range(nb)])
    right_ok = np.array([i != vocab.eos_id and len(toks[i]) > 0 and toks[i][0] not in b" \t\n\r\x0b\x0c"
                         for i in range(nb)])
    keys, cnts = count_pairs(base["train"], nb, left_ok, right_ok)
    order = np.argsort(-cnts, kind="stable")
    existing = set(toks)
    by_string: dict[bytes, list] = {}
    for idx in order:
        if cnts[idx] < args.min_count:
            break
        a, b = divmod(int(keys[idx]), nb)
        if not is_word_continuation(toks[a], toks[b], args.pos_dim):
            continue
        s = toks[a] + toks[b]
        if s in existing:
            continue
        entry = by_string.setdefault(s, [0, []])
        entry[0] += int(cnts[idx])
        entry[1].append((a, b, int(cnts[idx])))
    ranked = sorted(by_string.items(), key=lambda kv: (-kv[1][0], kv[0]))
    n_cand = args.n_trained + args.n_heldout
    if len(ranked) < n_cand:
        raise RuntimeError(f"only {len(ranked)} candidates")
    cand = ranked[:n_cand]
    # Frequency-stratified split: within every block of `block` consecutive ranks,
    # hold out n_heldout/n_cand of them at random.
    block = n_cand // args.n_heldout
    held = np.zeros(n_cand, dtype=bool)
    for s0 in range(0, n_cand, block):
        held[s0 + rng.integers(0, min(block, n_cand - s0))] = True
    assert held.sum() == args.n_heldout, held.sum()

    def pack(items):
        out = []
        for s, (tot, pairs) in items:
            pairs = sorted(pairs, key=lambda p: -p[2])
            out.append((s, [(a, b) for a, b, _ in pairs], tot))
        return out

    trained = pack([c for c, h in zip(cand, held) if not h])
    heldout = pack([c for c, h in zip(cand, held) if h])
    vocab.add_merges(trained, heldout)
    print(f"merges: trained {vocab.n_trained}, held-out {vocab.n_heldout}; "
          f"rank-1 count {cand[0][1][0]:,}, last {cand[-1][1][0]:,}")

    merged = {}
    for s in ("train", "dev", "test"):
        merged[s] = chunked(base[s], lambda c: vocab.apply_merges(c.astype(np.int64), "trained"),
                            vocab_eos=vocab.eos_id)
        print(f"{s}: base {len(base[s]):,} -> merged {len(merged[s]):,} "
              f"({100 * (1 - len(merged[s]) / len(base[s])):.2f}% shorter)")

    # ---- leakage and round-trip assertions ----
    V_train = vocab.n_train_vocab
    for s in ("train", "dev", "test"):
        assert merged[s].max() < V_train, f"held-out merge id in {s}"
    # exact byte round trip on dev and test (full) and on a train prefix
    for s in ("dev", "test"):
        assert vocab.stream_bytes(base[s]) == vocab.stream_bytes(merged[s]), f"round trip {s}"
    pre_b = base["train"][:3_000_000]
    pre_m = vocab.apply_merges(pre_b, "trained")
    assert vocab.stream_bytes(pre_b) == vocab.stream_bytes(pre_m)
    # greedy merging is causal: a prefix merges identically except possibly its last token
    assert np.array_equal(pre_m[:-1], merged["train"][:len(pre_m) - 1])

    # ---- statistics from TRAIN only ----
    V_all = vocab.n_total
    unigram = np.bincount(merged["train"], minlength=V_all).astype(np.int64)
    # held-out pair adjacency counted chunk-wise at document boundaries
    assert unigram[V_train:].sum() == 0
    hk, hv = vocab.pair_table("heldout")
    pair_counts_merged = np.zeros(len(hk), dtype=np.int64)  # held-out pairs adjacent in train
    for lo, hi in doc_chunks(merged["train"], vocab.eos_id):
        pair_counts_merged += count_keys(merged["train"][lo:hi], hk, V_all)
    heldout_string_count = np.zeros(vocab.n_heldout, dtype=np.int64)
    for key_i, (key, mid) in enumerate(zip(hk, hv)):
        heldout_string_count[mid - V_train] += pair_counts_merged[key_i]

    for s in ("train", "dev", "test"):
        np.save(OUT / f"{s}.npy", merged[s].astype(np.uint16))
    for s in ("dev", "test"):
        np.save(OUT / f"{s}_base.npy", base[s].astype(np.uint16))
    np.save(OUT / "unigram_train.npy", unigram)
    # growth candidates: every further word-internal pair string, ranked by train count,
    # never trained; used only for the vocabulary-growth analysis (R4)
    grow = ranked[n_cand:n_cand + args.n_growth]
    ga = np.array([sorted(p, key=lambda q: -q[2])[0][0] for _, (_, p) in grow], dtype=np.int64)
    gb = np.array([sorted(p, key=lambda q: -q[2])[0][1] for _, (_, p) in grow], dtype=np.int64)
    gkeys_all, gid_all = [], []
    for gi, (_, (_, pairs)) in enumerate(grow):
        for a_, b_, _c in pairs:
            gkeys_all.append(a_ * V_all + b_); gid_all.append(gi)
    gkeys_all = np.array(gkeys_all, dtype=np.int64); gid_all = np.array(gid_all, dtype=np.int64)
    order_g = np.argsort(gkeys_all, kind="stable")
    gkeys_all, gid_all = gkeys_all[order_g], gid_all[order_g]
    gcount_keys = np.zeros(len(gkeys_all), dtype=np.int64)
    for lo, hi in doc_chunks(merged["train"], vocab.eos_id):
        gcount_keys += count_keys(merged["train"][lo:hi], gkeys_all, V_all)
    gcount = np.bincount(gid_all, weights=gcount_keys, minlength=len(grow)).astype(np.int64)
    np.savez(OUT / "growth_candidates.npz", a=ga, b=gb, keys=gkeys_all, key_to_cand=gid_all,
             count_merged_train=gcount,
             count_base_train=np.array([c for _, (c, _) in grow], dtype=np.int64))
    print(f"growth candidates: {len(grow):,} (count range {grow[0][1][0] if grow else 0}"
          f" .. {grow[-1][1][0] if grow else 0})", flush=True)
    np.save(OUT / "heldout_pair_count_train.npy", heldout_string_count)
    vocab.save(OUT / "vocab.json")

    fffd = decode_replacement_collisions(tok_path)
    manifest = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "pos_dim": args.pos_dim, "seed": args.seed,
        "n_base": vocab.n_base, "n_trained": vocab.n_trained, "n_heldout": vocab.n_heldout,
        "n_train_vocab": V_train, "n_total": V_all, "eos_id": vocab.eos_id,
        "dropped_gpt2_long_tokens": long_ids,
        "gpt2_decode_fffd": fffd,
        "merge_rule": "adjacent base pair, word-internal (^ ?[^\\W_]+$ after UTF-8 decode), "
                      "right piece not starting with whitespace, <= pos_dim bytes, string not "
                      "already a token, ranked by train-split count summed over decompositions; "
                      f"top {n_cand}; within each block of {block} ranks one held out at random",
        "tokens": {s: int(len(merged[s])) for s in merged},
        "base_tokens": {s: int(len(base[s])) for s in base},
        "files": {},
    }
    for f in sorted(OUT.iterdir()):
        if f.suffix in (".npy", ".json", ".npz") and f.name != "manifest.json":
            manifest["files"][f.name] = sha256_file(f)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k != "files"}, indent=2)[:3000])


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download")
    d.add_argument("--train-tokens", type=int, default=250_000_000)
    b = sub.add_parser("build")
    b.add_argument("--pos-dim", type=int, default=32)
    b.add_argument("--n-trained", type=int, default=7000)
    b.add_argument("--n-heldout", type=int, default=1000)
    b.add_argument("--min-count", type=int, default=2)
    b.add_argument("--n-growth", type=int, default=1_200_000)
    b.add_argument("--seed", type=int, default=20261001)
    args = p.parse_args()
    {"download": download, "build": build}[args.cmd](args)


if __name__ == "__main__":
    main()
