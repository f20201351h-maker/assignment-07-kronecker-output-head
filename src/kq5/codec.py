"""Kronecker byte codec, written so that K @ W never materialises K.

Reference (pip ``kronecker-embeddings`` 0.1.1, ``codec.kronecker_codec``):

    x[b * pos_dim + p] = 1/sqrt(L)   for every byte b at position p < L
    kappa = (x - mean(x)) / (std_unbiased(x) + eps)

For a token with L > 0 bytes every entry of kappa takes one of two values, so

    kappa_v = A_v * 1[S_v] + B_v * 1,      |S_v| = L_v

with A_v = (1/sqrt(L)) / (std + eps) and B_v = -mean / (std + eps). Hence

    (K @ W)[v] = A_v * sum_{j in S_v} W[j] + B_v * sum_j W[j]

which is one ``embedding_bag`` plus a rank-1 term. This is exact (not an
approximation) and is the identity behind the KAS head: a token's logit is a
sum of scalar lookups in the shared score table s = W h, plus a length term.
For L = 0 the reference returns the zero vector, so A = B = 0.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

CHAR_DIM = 256
POS_DIM = 32
EPS = 1e-6


def code_dim(char_dim: int = CHAR_DIM, pos_dim: int = POS_DIM) -> int:
    return char_dim * pos_dim


def codec_coefficients(lengths: np.ndarray, char_dim: int = CHAR_DIM,
                       pos_dim: int = POS_DIM, eps: float = EPS) -> tuple[np.ndarray, np.ndarray]:
    """Return float64 (A, B) per token so that kappa = A*1[S] + B*1 exactly."""
    L = np.asarray(lengths, dtype=np.float64)
    if np.any(L < 0) or np.any(L > pos_dim):
        raise ValueError("lengths must lie in [0, pos_dim]")
    D = float(char_dim * pos_dim)
    s = np.where(L > 0, 1.0 / np.sqrt(np.maximum(L, 1.0)), 0.0)
    mean = L * s / D
    var = (L * (s - mean) ** 2 + (D - L) * mean ** 2) / (D - 1.0)
    denom = np.sqrt(var) + eps
    A = np.where(L > 0, s / denom, 0.0)
    B = np.where(L > 0, -mean / denom, 0.0)
    return A, B


@dataclass
class CodecTable:
    """Byte buffer for a vocabulary, plus the gather indices the codec needs.

    byte_buffer: uint8 (V, pos_dim), zero padded
    lengths:     int64 (V,), 0 <= L <= pos_dim
    """

    byte_buffer: np.ndarray
    lengths: np.ndarray
    char_dim: int = CHAR_DIM
    pos_dim: int = POS_DIM

    def __post_init__(self) -> None:
        self.byte_buffer = np.ascontiguousarray(self.byte_buffer, dtype=np.uint8)
        self.lengths = np.ascontiguousarray(self.lengths, dtype=np.int64)
        V, P = self.byte_buffer.shape
        if P != self.pos_dim or self.lengths.shape != (V,):
            raise ValueError("byte_buffer must be (V, pos_dim) and lengths (V,)")
        pos = np.arange(P)[None, :]
        valid = pos < self.lengths[:, None]
        if np.any(self.byte_buffer[~valid] != 0):
            raise ValueError("byte_buffer padding must be zero")
        self.A, self.B = codec_coefficients(self.lengths, self.char_dim, self.pos_dim)
        self.flat_index = (self.byte_buffer.astype(np.int64) * P + pos) * valid
        self.valid = valid

    @classmethod
    def from_bytes(cls, tokens: list[bytes], pos_dim: int = POS_DIM) -> "CodecTable":
        V = len(tokens)
        buf = np.zeros((V, pos_dim), dtype=np.uint8)
        lens = np.zeros(V, dtype=np.int64)
        for i, b in enumerate(tokens):
            if len(b) > pos_dim:
                raise ValueError(f"token {i} has {len(b)} bytes > pos_dim={pos_dim}")
            buf[i, : len(b)] = np.frombuffer(b, dtype=np.uint8) if b else []
            lens[i] = len(b)
        return cls(buf, lens, pos_dim=pos_dim)

    @property
    def V(self) -> int:
        return int(self.lengths.shape[0])

    @property
    def D(self) -> int:
        return self.char_dim * self.pos_dim

    def dense(self, dtype=np.float64) -> np.ndarray:
        """Explicit (V, D) code matrix. Tests only: memory is V*D."""
        K = np.zeros((self.V, self.D), dtype=dtype)
        K += self.B[:, None]
        rows = np.repeat(np.arange(self.V), self.pos_dim)[self.valid.reshape(-1)]
        cols = self.flat_index[self.valid]
        K[rows, cols] += self.A[rows]
        return K

    def torch_buffers(self, device=None) -> "TorchCodec":
        return TorchCodec(self, device)


class TorchCodec:
    """Device-side copy of the gather indices; computes K[ids] @ W with autograd."""

    def __init__(self, table: CodecTable, device=None):
        w = np.where(table.valid, table.A[:, None], 0.0)
        self.index = torch.as_tensor(table.flat_index, dtype=torch.long, device=device)
        self.weight64 = torch.as_tensor(w, dtype=torch.float64, device=device)
        self.B64 = torch.as_tensor(table.B, dtype=torch.float64, device=device)
        self.lengths = torch.as_tensor(table.lengths, dtype=torch.long, device=device)
        self.D = table.D
        self.V = table.V

    def to(self, device) -> "TorchCodec":
        for name in ("index", "weight64", "B64", "lengths"):
            setattr(self, name, getattr(self, name).to(device))
        return self

    def matmul(self, W: torch.Tensor, ids: torch.Tensor | None = None) -> torch.Tensor:
        """Return K[ids] @ W for W of shape (D, d). ids=None means all rows."""
        if W.shape[0] != self.D:
            raise ValueError(f"W has {W.shape[0]} rows, codec dim is {self.D}")
        if ids is None:
            index, weight, B = self.index, self.weight64, self.B64
        else:
            index, weight, B = self.index[ids], self.weight64[ids], self.B64[ids]
        shape = index.shape[:-1]
        index = index.reshape(-1, index.shape[-1])
        weight = weight.reshape(-1, weight.shape[-1]).to(W.dtype)
        out = F.embedding_bag(index, W, per_sample_weights=weight, mode="sum")
        out = out + B.reshape(-1, 1).to(W.dtype) * W.sum(0, keepdim=True)
        return out.reshape(*shape, W.shape[1])


def reference_codec(table: CodecTable) -> np.ndarray | None:
    """Codes from the published package, or None if it is not installed."""
    try:
        from kronecker_embeddings import kronecker_codec  # type: ignore
    except ImportError:
        return None
    out = kronecker_codec(torch.as_tensor(table.byte_buffer), torch.as_tensor(table.lengths),
                          char_dim=table.char_dim, pos_dim=table.pos_dim)
    return out.numpy().astype(np.float64)


def code_norm_sq(length: int, char_dim: int = CHAR_DIM, pos_dim: int = POS_DIM) -> float:
    """||kappa||^2; equals (D-1) * (std/(std+eps))^2 for L>0, i.e. about D-1."""
    A, B = codec_coefficients(np.array([length]), char_dim, pos_dim)
    D = char_dim * pos_dim
    return float(length * (A[0] + B[0]) ** 2 + (D - length) * B[0] ** 2)


__all__ = ["CHAR_DIM", "POS_DIM", "EPS", "CodecTable", "TorchCodec", "codec_coefficients",
           "code_dim", "reference_codec", "code_norm_sq", "math"]
