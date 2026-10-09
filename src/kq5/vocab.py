"""Working vocabulary: GPT-2 base tokens that fit the byte window, plus merges.

Id space (contiguous):
  [0, n_base)                         GPT-2 tokens with <= pos_dim raw bytes
  [n_base, n_base + n_trained)        trained merges (appear in training text)
  [n_base + n_trained, n_total)       held-out merges (never in training text)

GPT-2 tokens longer than pos_dim bytes would collide under the codec (the
Kronecker code only sees the first pos_dim bytes). They are removed from the
vocabulary and every occurrence is re-encoded, deterministically and
losslessly, as the greedy longest-prefix match over live base tokens.

A merge is a byte string m = a + b over two base tokens. It is applied
single-level, greedily left to right, to a base-token stream.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

WHITESPACE = set(b" \t\n\r\x0b\x0c")
WORD_RE = re.compile(r"^ ?[^\W_]+$")


def greedy_split(data: bytes, lookup: dict[bytes, int], max_len: int) -> list[int]:
    out, i = [], 0
    while i < len(data):
        for n in range(min(max_len, len(data) - i), 0, -1):
            j = lookup.get(data[i:i + n])
            if j is not None:
                out.append(j)
                i += n
                break
        else:  # pragma: no cover - every single byte is a GPT-2 token
            raise ValueError("byte not covered by base vocabulary")
    return out


@dataclass
class WorkingVocab:
    tokens: list[bytes]                 # raw bytes by working id
    n_base: int
    n_trained: int
    n_heldout: int
    eos_id: int
    pos_dim: int
    gpt2_to_work: list[list[int]]       # expansion of every GPT-2 id
    merge_pairs: list[list[tuple[int, int]]] = field(default_factory=list)  # per merge, decompositions
    merge_counts: list[int] = field(default_factory=list)  # base-stream pair count used to rank

    @property
    def n_total(self) -> int:
        return len(self.tokens)

    @property
    def n_train_vocab(self) -> int:
        return self.n_base + self.n_trained

    def merge_id(self, k: int) -> int:
        return self.n_base + k

    def is_heldout(self, tok: int) -> bool:
        return tok >= self.n_train_vocab

    # ---------- construction ----------
    @classmethod
    def from_gpt2(cls, gpt2_bytes: list[bytes], eos_gpt2: int, pos_dim: int) -> "WorkingVocab":
        live = [i for i, b in enumerate(gpt2_bytes) if len(b) <= pos_dim]
        new_of = {g: k for k, g in enumerate(live)}
        tokens = [gpt2_bytes[g] for g in live]
        lookup = {b: k for k, b in enumerate(tokens) if live[k] != eos_gpt2}
        expansion = []
        for g, b in enumerate(gpt2_bytes):
            if g in new_of:
                expansion.append([new_of[g]])
            else:
                expansion.append(greedy_split(b, lookup, pos_dim))
        return cls(tokens=tokens, n_base=len(tokens), n_trained=0, n_heldout=0,
                   eos_id=new_of[eos_gpt2], pos_dim=pos_dim, gpt2_to_work=expansion)

    def gpt2_stream_to_base(self, ids: np.ndarray) -> np.ndarray:
        """Map a GPT-2 id stream to working base ids, expanding long tokens."""
        ids = np.asarray(ids, dtype=np.int64)
        lens = np.array([len(e) for e in self.gpt2_to_work], dtype=np.int64)
        offsets = np.concatenate([[0], np.cumsum(lens)[:-1]])
        flat = np.array([t for e in self.gpt2_to_work for t in e], dtype=np.int64)
        counts = lens[ids]
        total = int(counts.sum())
        starts = np.repeat(np.cumsum(counts) - counts, counts)
        within = np.arange(total, dtype=np.int64) - starts
        return flat[np.repeat(offsets[ids], counts) + within]

    def add_merges(self, trained: list[tuple[bytes, list[tuple[int, int]], int]],
                   heldout: list[tuple[bytes, list[tuple[int, int]], int]]) -> None:
        if self.n_trained or self.n_heldout:
            raise ValueError("merges already added")
        existing = set(self.tokens)
        for s, pairs, count in trained + heldout:
            if s in existing or len(s) > self.pos_dim:
                raise ValueError(f"bad merge {s!r}")
            for a, b in pairs:
                if self.tokens[a] + self.tokens[b] != s or a >= self.n_base or b >= self.n_base:
                    raise ValueError(f"decomposition mismatch for {s!r}")
            existing.add(s)
            self.tokens.append(s)
            self.merge_pairs.append([tuple(p) for p in pairs])
            self.merge_counts.append(int(count))
        self.n_trained = len(trained)
        self.n_heldout = len(heldout)

    # ---------- merge tables ----------
    def pair_table(self, which: str) -> tuple[np.ndarray, np.ndarray]:
        """Sorted pair keys (a * n_total + b) and the merge id each maps to."""
        if which == "trained":
            ks = range(self.n_trained)
        elif which == "heldout":
            ks = range(self.n_trained, self.n_trained + self.n_heldout)
        else:
            raise ValueError(which)
        keys, vals = [], []
        for k in ks:
            for a, b in self.merge_pairs[k]:
                keys.append(a * self.n_total + b)
                vals.append(self.merge_id(k))
        keys_a = np.array(keys, dtype=np.int64)
        vals_a = np.array(vals, dtype=np.int64)
        order = np.argsort(keys_a, kind="stable")
        keys_a, vals_a = keys_a[order], vals_a[order]
        if len(keys_a) and np.any(np.diff(keys_a) == 0):
            raise ValueError("a base pair maps to two merges")
        return keys_a, vals_a

    def apply_merges(self, stream: np.ndarray, which: str) -> np.ndarray:
        keys, vals = self.pair_table(which)
        return apply_pair_merges(np.asarray(stream, dtype=np.int64), keys, vals, self.n_total)

    def stream_bytes(self, stream: np.ndarray) -> bytes:
        return b"".join(self.tokens[int(t)] for t in stream)

    def token_lengths(self) -> np.ndarray:
        return np.array([len(t) for t in self.tokens], dtype=np.int64)

    def target_bytes(self) -> np.ndarray:
        """Bytes credited to each target token for bpb. EOS is a document marker,
        not text, so it is credited 0 bytes and excluded from bpb."""
        n = self.token_lengths()
        n[self.eos_id] = 0
        return n

    # ---------- persistence ----------
    def to_json(self) -> dict:
        return {
            "tokens_hex": [t.hex() for t in self.tokens],
            "n_base": self.n_base, "n_trained": self.n_trained, "n_heldout": self.n_heldout,
            "eos_id": self.eos_id, "pos_dim": self.pos_dim,
            "gpt2_to_work": self.gpt2_to_work,
            "merge_pairs": [[list(p) for p in ps] for ps in self.merge_pairs],
            "merge_counts": self.merge_counts,
        }

    @classmethod
    def from_json(cls, d: dict) -> "WorkingVocab":
        return cls(tokens=[bytes.fromhex(h) for h in d["tokens_hex"]], n_base=d["n_base"],
                   n_trained=d["n_trained"], n_heldout=d["n_heldout"], eos_id=d["eos_id"],
                   pos_dim=d["pos_dim"], gpt2_to_work=d["gpt2_to_work"],
                   merge_pairs=[[tuple(p) for p in ps] for ps in d["merge_pairs"]],
                   merge_counts=list(d["merge_counts"]))

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.to_json()), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "WorkingVocab":
        return cls.from_json(json.loads(Path(path).read_text(encoding="utf-8")))


def apply_pair_merges(stream: np.ndarray, keys: np.ndarray, vals: np.ndarray,
                      key_base: int) -> np.ndarray:
    """Single-level greedy left-to-right merge of adjacent pairs found in keys."""
    t = stream
    if len(t) < 2 or len(keys) == 0:
        return t.copy()
    pk = t[:-1] * key_base + t[1:]
    pos = np.searchsorted(keys, pk)
    pos_c = np.minimum(pos, len(keys) - 1)
    hit = keys[pos_c] == pk
    # In a run of consecutive hits, greedy left-to-right takes offsets 0, 2, 4, ...
    prev = np.concatenate([[False], hit[:-1]])
    run_start = hit & ~prev
    run_id = np.cumsum(run_start)
    starts = np.flatnonzero(run_start)
    idx = np.arange(len(hit))
    offset = np.where(hit, idx - starts[np.maximum(run_id - 1, 0)] if len(starts) else 0, 1)
    take = hit & (offset % 2 == 0)
    out = t.copy()
    out[:-1][take] = vals[pos_c[take]]
    keep = np.ones(len(t), dtype=bool)
    keep[1:][take] = False
    return out[keep]


def is_word_continuation(a: bytes, b: bytes, pos_dim: int) -> bool:
    """Candidate merge rule: the pair stays inside one word (letters/digits,
    optional leading space on the first piece) and fits the byte window."""
    if not a or not b or b[0] in WHITESPACE or len(a) + len(b) > pos_dim:
        return False
    try:
        s = (a + b).decode("utf-8")
    except UnicodeDecodeError:
        return False
    if not WORD_RE.match(s):
        return False
    return a.strip(b" ") != b"" and True


def count_pairs(stream: np.ndarray, key_base: int, allowed_left: np.ndarray,
                allowed_right: np.ndarray, chunk: int = 20_000_000) -> tuple[np.ndarray, np.ndarray]:
    """Counts of adjacent pairs (a, b) with allowed_left[a] & allowed_right[b].
    Returns (sorted keys, counts)."""
    acc_k = np.zeros(0, dtype=np.int64)
    acc_c = np.zeros(0, dtype=np.int64)
    n = len(stream)
    for s in range(0, n - 1, chunk):
        e = min(n, s + chunk + 1)
        a = stream[s:e - 1].astype(np.int64)
        b = stream[s + 1:e].astype(np.int64)
        m = allowed_left[a] & allowed_right[b]
        k, c = np.unique(a[m] * key_base + b[m], return_counts=True)
        acc_k = np.concatenate([acc_k, k])
        acc_c = np.concatenate([acc_c, c])
        order = np.argsort(acc_k, kind="stable")
        acc_k, acc_c = acc_k[order], acc_c[order]
        uk, first = np.unique(acc_k, return_index=True)
        acc_c = np.add.reduceat(acc_c, first)
        acc_k = uk
    return acc_k, acc_c


def count_keys(stream: np.ndarray, keys: np.ndarray, key_base: int) -> np.ndarray:
    """Occurrences of each sorted pair key in a stream (overlapping occurrences counted)."""
    t = np.asarray(stream, dtype=np.int64)
    pk = t[:-1] * key_base + t[1:]
    pos = np.searchsorted(keys, pk)
    pos_c = np.minimum(pos, max(len(keys) - 1, 0))
    hit = (keys[pos_c] == pk) if len(keys) else np.zeros(len(pk), bool)
    return np.bincount(pos_c[hit], minlength=len(keys))
