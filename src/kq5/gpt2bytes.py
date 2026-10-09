"""Raw UTF-8 byte strings of GPT-2 BPE tokens.

GPT-2's vocabulary stores each token as a string over a printable alias of the
256 byte values (``bytes_to_unicode``). Inverting that alias gives the exact
bytes a token contributes to the text. ``tokenizer.decode([id])`` must not be
used: a token that is a fragment of a multi-byte character decodes to U+FFFD
and many distinct tokens then share one byte string (an observation credited to
Mukund Singh in the reference report; see README, Prior work).
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

EOS_TEXT = "<|endoftext|>"


@lru_cache(maxsize=1)
def bytes_to_unicode() -> dict[int, str]:
    """The GPT-2 byte -> printable character table (openai/gpt-2 encoder.py)."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("\xa1"), ord("\xac") + 1)) \
        + list(range(ord("\xae"), ord("\xff") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


@lru_cache(maxsize=1)
def unicode_to_bytes() -> dict[str, int]:
    return {c: b for b, c in bytes_to_unicode().items()}


def piece_to_bytes(piece: str) -> bytes:
    table = unicode_to_bytes()
    return bytes(table[ch] for ch in piece)


def load_gpt2_raw_bytes(tokenizer_json: Path) -> tuple[list[bytes], int]:
    """Return (raw bytes indexed by GPT-2 id, eos_id). EOS maps to its literal text,
    as the reference package does for special tokens."""
    spec = json.loads(Path(tokenizer_json).read_text(encoding="utf-8"))
    vocab: dict[str, int] = spec["model"]["vocab"]
    added = {t["content"]: t["id"] for t in spec.get("added_tokens", [])}
    size = max(max(vocab.values()), max(added.values(), default=-1)) + 1
    out: list[bytes | None] = [None] * size
    for piece, idx in vocab.items():
        if piece in added:
            continue
        out[idx] = piece_to_bytes(piece)
    for piece, idx in added.items():
        out[idx] = piece.encode("utf-8")
    if any(b is None for b in out):
        raise ValueError("GPT-2 vocabulary has holes")
    eos = added.get(EOS_TEXT, vocab.get(EOS_TEXT))
    return out, int(eos)  # type: ignore[return-value]


def decode_replacement_collisions(tokenizer_json: Path) -> dict[str, int]:
    """Count GPT-2 ids whose ``decode([id])`` yields text containing U+FFFD, and the
    number of distinct ids that collapse onto the single string U+FFFD."""
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(tokenizer_json))
    n = tok.get_vocab_size()
    with_fffd = 0
    exactly_fffd = 0
    for i in range(n):
        s = tok.decode([i], skip_special_tokens=False)
        if "�" in s:
            with_fffd += 1
            if s == "�":
                exactly_fffd += 1
    return {"vocab": n, "decode_contains_fffd": with_fffd, "decode_equals_fffd": exactly_fffd}
