"""Token arrays, windowing and batch order shared by every arm.

Training windows are consecutive, non-overlapping slices of T+1 tokens of the
train stream. A single permutation (data_seed) fixes their order; step s reads
windows order[s*B:(s+1)*B]. Every arm and every init seed reads the same
windows in the same order. Evaluation windows tile the dev or test stream the
same way and are never shuffled.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


def find_data_dir(start: Path) -> Path:
    """Locate the token directory by its manifest (never a hard-coded mount)."""
    hits = [p.parent for p in Path(start).rglob("manifest.json")
            if (p.parent / "vocab.json").exists() and (p.parent / "train.npy").exists()]
    if len(hits) != 1:
        raise RuntimeError(f"expected one token directory under {start}, found {hits}")
    return hits[0]


def sha256_array(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


@dataclass
class TokenData:
    root: Path
    manifest: dict
    train: np.ndarray
    dev: np.ndarray
    test: np.ndarray

    @classmethod
    def load(cls, root: Path, verify: bool = True) -> "TokenData":
        root = Path(root)
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        if verify:
            for name, digest in manifest["files"].items():
                h = hashlib.sha256()
                with open(root / name, "rb") as f:
                    for block in iter(lambda: f.read(1 << 22), b""):
                        h.update(block)
                if h.hexdigest() != digest:
                    raise RuntimeError(f"hash mismatch for {name}")
        arrays = {s: np.load(root / f"{s}.npy", mmap_mode="r") for s in ("train", "dev", "test")}
        return cls(root, manifest, arrays["train"], arrays["dev"], arrays["test"])


def train_order(n_tokens: int, seq_len: int, data_seed: int) -> np.ndarray:
    n_windows = (n_tokens - 1) // seq_len
    return np.random.default_rng(data_seed).permutation(n_windows)


class TrainBatches:
    def __init__(self, stream: np.ndarray, seq_len: int, batch_seqs: int, data_seed: int):
        self.stream = stream
        self.T = seq_len
        self.B = batch_seqs
        self.order = train_order(len(stream), seq_len, data_seed)
        self.order_sha256 = sha256_array(self.order.astype(np.int64))

    @property
    def max_steps(self) -> int:
        return len(self.order) // self.B

    def windows(self, step: int) -> np.ndarray:
        return self.order[step * self.B:(step + 1) * self.B]

    def batch(self, step: int) -> np.ndarray:
        w = self.windows(step)
        starts = w * self.T
        idx = starts[:, None] + np.arange(self.T + 1)[None, :]
        return np.asarray(self.stream[idx.reshape(-1)], dtype=np.int64).reshape(len(w), self.T + 1)


def eval_windows(stream: np.ndarray, seq_len: int, max_windows: int | None = None) -> np.ndarray:
    n = (len(stream) - 1) // seq_len
    if max_windows is not None:
        n = min(n, max_windows)
    idx = np.arange(n)[:, None] * seq_len + np.arange(seq_len + 1)[None, :]
    return np.asarray(stream[idx.reshape(-1)], dtype=np.int64).reshape(n, seq_len + 1)


def eval_windows_full(stream: np.ndarray, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    """Tile the WHOLE stream: like eval_windows plus a final partial window padded with token 0.
    Returns (windows (n, T+1), valid (n, T) mask of real targets). Every token after the first is
    a target exactly once, so two tokenisations of the same text are scored on the same bytes
    (up to the first token)."""
    s = np.asarray(stream, dtype=np.int64)
    n = -(-(len(s) - 1) // seq_len)
    pad = n * seq_len + 1 - len(s)
    full = np.concatenate([s, np.zeros(pad, dtype=np.int64)])
    idx = np.arange(n)[:, None] * seq_len + np.arange(seq_len + 1)[None, :]
    valid = (idx[:, 1:] < len(s))
    return full[idx], valid


def to_device(batch: np.ndarray, device) -> tuple[torch.Tensor, torch.Tensor]:
    # torch.tensor copies; never alias a numpy buffer across an async copy
    t = torch.tensor(batch, dtype=torch.long)
    if device is not None and str(device).startswith("cuda"):
        t = t.pin_memory().to(device, non_blocking=True)
    return t[:, :-1], t[:, 1:]
