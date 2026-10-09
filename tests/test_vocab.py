"""R0: raw-byte extraction, working-vocabulary collisions, merge tokenizer round trip."""

import numpy as np
import pytest

from kq5.codec import CodecTable
from kq5.vocab import WorkingVocab, apply_pair_merges, count_pairs, is_word_continuation

TEXTS = ["Hello world, the quick brown fox.", "भारत एक देश है। తెలుగు భాష",
         "Archived records of 1999 — naïve café 😀 \n\n\tend", "  multiple   spaces\r\n",
         "emoji 👩‍👩‍👧 and CJK 漢字かな"]


def test_raw_bytes_reconstruct_text(gpt2_bytes, tokenizer):
    toks, _ = gpt2_bytes
    for text in TEXTS:
        ids = tokenizer.encode(text, add_special_tokens=False).ids
        assert b"".join(toks[i] for i in ids) == text.encode("utf-8")


def test_raw_bytes_are_unique(gpt2_bytes):
    toks, eos = gpt2_bytes
    assert len(set(toks)) == len(toks)
    assert toks[eos] == b"<|endoftext|>"


def test_decode_collapses_but_raw_bytes_do_not(gpt2_bytes, tokenizer):
    toks, _ = gpt2_bytes
    fffd = [i for i in range(len(toks)) if tokenizer.decode([i]) == "�"]
    assert len(fffd) > 100          # many fragments collapse under decode()
    assert len({toks[i] for i in fffd}) == len(fffd)


def test_working_vocab_has_no_codec_collisions(gpt2_bytes):
    toks, eos = gpt2_bytes
    v = WorkingVocab.from_gpt2(toks, eos, 32)
    assert all(len(t) <= 32 for t in v.tokens)
    assert len(set(v.tokens)) == len(v.tokens)
    t = CodecTable.from_bytes(v.tokens)
    # the codec is injective on byte strings of length <= pos_dim: support (byte,pos) set -> string
    supports = {tuple(t.flat_index[i][t.valid[i]]) for i in range(t.V)}
    assert len(supports) == t.V
    dropped = [i for i, b in enumerate(toks) if len(b) > 32]
    assert v.n_base == len(toks) - len(dropped)


def test_long_tokens_reencode_losslessly(gpt2_bytes):
    toks, eos = gpt2_bytes
    v = WorkingVocab.from_gpt2(toks, eos, 32)
    long_ids = [i for i, b in enumerate(toks) if len(b) > 32]
    assert long_ids, "GPT-2 has tokens longer than 32 bytes"
    stream = np.array(long_ids + [0, 1, eos], dtype=np.int64)
    out = v.gpt2_stream_to_base(stream)
    assert v.stream_bytes(out) == b"".join(toks[i] for i in stream)


def naive_merge(stream, table):
    out, i = [], 0
    while i < len(stream):
        if i + 1 < len(stream) and (stream[i], stream[i + 1]) in table:
            out.append(table[(stream[i], stream[i + 1])]); i += 2
        else:
            out.append(stream[i]); i += 1
    return out


def test_vectorised_merge_matches_naive_greedy():
    rng = np.random.default_rng(1)
    V = 6
    table = {(0, 1): 10, (1, 2): 11, (2, 2): 12, (3, 0): 13}
    keys = np.array(sorted(a * 100 + b for a, b in table), dtype=np.int64)
    vals = np.array([table[divmod(int(k), 100)] for k in keys])
    for _ in range(200):
        s = rng.integers(0, V, size=rng.integers(0, 40))
        got = apply_pair_merges(s.astype(np.int64), keys, vals, 100)
        assert got.tolist() == naive_merge(s.tolist(), table)


def test_merge_round_trip_and_heldout_never_in_trained_stream(gpt2_bytes, tokenizer):
    toks, eos = gpt2_bytes
    v = WorkingVocab.from_gpt2(toks, eos, 32)
    text = " ".join(TEXTS) * 3 + " Archived archived Archiving"
    g = np.array(tokenizer.encode(text, add_special_tokens=False).ids, dtype=np.int64)
    base = v.gpt2_stream_to_base(g)
    lookup = {b: i for i, b in enumerate(v.tokens)}
    pairs = [(base[i], base[i + 1]) for i in range(len(base) - 1)
             if is_word_continuation(v.tokens[base[i]], v.tokens[base[i + 1]], 32)
             and v.tokens[base[i]] + v.tokens[base[i + 1]] not in lookup]
    pairs = list(dict.fromkeys((int(a), int(b)) for a, b in pairs))
    assert len(pairs) >= 4
    tr, ho = pairs[:-2], pairs[-2:]
    v.add_merges([(v.tokens[a] + v.tokens[b], [(a, b)], 5) for a, b in tr],
                 [(v.tokens[a] + v.tokens[b], [(a, b)], 3) for a, b in ho])
    merged = v.apply_merges(base, "trained")
    assert v.stream_bytes(merged) == v.stream_bytes(base)
    assert merged.max() < v.n_train_vocab
    assert len(merged) < len(base)
    both = v.apply_merges(merged, "heldout")
    assert v.stream_bytes(both) == v.stream_bytes(base)
    assert both.max() >= v.n_train_vocab


def test_count_pairs_matches_bruteforce():
    rng = np.random.default_rng(2)
    s = rng.integers(0, 9, size=5000)
    ok = np.ones(9, bool); ok[3] = False
    k, c = count_pairs(s, 9, ok, ok, chunk=700)
    brute = {}
    for a, b in zip(s[:-1], s[1:]):
        if ok[a] and ok[b]:
            brute[a * 9 + b] = brute.get(a * 9 + b, 0) + 1
    assert dict(zip(k.tolist(), c.tolist())) == brute


def test_word_continuation_rule():
    assert is_word_continuation(b" Arch", b"ived", 32)
    assert not is_word_continuation(b" Arch", b" ived", 32)
    assert not is_word_continuation(b"word", b".", 32)
    assert not is_word_continuation(b" ", b"word", 32)
    assert not is_word_continuation(b"\xe0\xa4", b"x", 32)
