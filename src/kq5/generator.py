"""Byte -> per-token correction generators (KAS-G).

A generator reads a token's bytes and returns rank+1 numbers: a correction
vector u_v (rank) and a scalar prior residual delta_v. It is shared by all
tokens, so its parameter count does not depend on the vocabulary size.

ByteCNN is shift-aware twice over: convolutions are translation-equivariant
along the byte axis, and each byte also sees its distance from the token's
start and from its end, so a suffix such as "ived" gets the same features in
"Archived" and in "derived" (the linear Kronecker code cannot share it: the
same suffix lands in different position columns when word length differs).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GeneratorConfig:
    kind: str = "bytecnn"
    emb: int = 64
    channels: int = 128
    layers: int = 2
    kernel: int = 3
    hidden: int = 256
    pos_dim: int = 32


class ByteCNN(nn.Module):
    def __init__(self, out_dim: int, cfg: GeneratorConfig):
        super().__init__()
        self.cfg = cfg
        P = cfg.pos_dim
        self.byte_emb = nn.Embedding(256, cfg.emb)
        self.pos_start = nn.Embedding(P, cfg.emb)
        self.pos_end = nn.Embedding(P, cfg.emb)
        self.len_emb = nn.Embedding(P + 1, 2 * cfg.channels)
        convs = []
        c_in = cfg.emb
        for _ in range(cfg.layers):
            convs.append(nn.Conv1d(c_in, cfg.channels, cfg.kernel, padding=cfg.kernel // 2))
            c_in = cfg.channels
        self.convs = nn.ModuleList(convs)
        self.fc1 = nn.Linear(2 * cfg.channels, cfg.hidden)
        self.fc2 = nn.Linear(cfg.hidden, out_dim)
        for emb in (self.byte_emb, self.pos_start, self.pos_end):
            nn.init.normal_(emb.weight, std=0.5)
        nn.init.zeros_(self.len_emb.weight)

    def forward(self, byte_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        N, P = byte_ids.shape
        pos = torch.arange(P, device=byte_ids.device)
        mask = (pos[None, :] < lengths[:, None])                       # (N, P)
        from_end = (lengths[:, None] - 1 - pos[None, :]).clamp(0, P - 1)
        x = self.byte_emb(byte_ids.long()) + self.pos_start(pos)[None] + self.pos_end(from_end)
        m = mask.unsqueeze(1).to(x.dtype)                               # (N, 1, P)
        x = x.transpose(1, 2) * m                                       # (N, C, P)
        for conv in self.convs:
            x = F.gelu(conv(x)) * m
        denom = m.sum(-1).clamp_min(1.0)
        mean = x.sum(-1) / denom
        mx = x.masked_fill(m == 0, float("-inf")).amax(-1)
        mx = torch.where(torch.isfinite(mx), mx, torch.zeros_like(mx))
        h = torch.cat([mean, mx], dim=-1) + self.len_emb(lengths.clamp(0, P))
        return self.fc2(F.gelu(self.fc1(h)))


class OccMLP(nn.Module):
    """The reference report's own generator form ("Byte-derived per-token coordinates", its §09):
    U_v = tanh(occ_v · W1) · W2 where occ_v is the token's Kronecker code (the z-normalised
    byte-position occupancy, D = 256 × pos_dim; the same vector the KAS head reads). One hidden
    layer, no shift-awareness: a suffix lands in different input columns for different word
    lengths. Phase 2 (protocol 06) runs it parameter-matched to ByteCNN v3 (hidden = 92:
    755,337 vs 751,889 parameters). The report found this form 0.24 bpb WORSE than no coordinates.
    """

    def __init__(self, out_dim: int, cfg: GeneratorConfig):
        super().__init__()
        from .codec import codec_coefficients
        self.cfg = cfg
        P = cfg.pos_dim
        D = 256 * P
        self.fc1 = nn.Linear(D, cfg.hidden)
        self.fc2 = nn.Linear(cfg.hidden, out_dim)
        nn.init.normal_(self.fc1.weight, std=1.0 / math.sqrt(D))   # code entries have unit variance
        nn.init.zeros_(self.fc1.bias)
        A, B = codec_coefficients(np.arange(P + 1), pos_dim=P)       # exact codec scalars per length
        self.register_buffer("A_tab", torch.as_tensor(A, dtype=torch.float32))
        self.register_buffer("B_tab", torch.as_tensor(B, dtype=torch.float32))

    def code(self, byte_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Dense Kronecker codes (N, D) == CodecTable.dense() rows (fp32)."""
        N, P = byte_ids.shape
        pos = torch.arange(P, device=byte_ids.device)
        valid = pos[None, :] < lengths[:, None]
        idx = (byte_ids.long() * P + pos[None, :]) * valid                    # CodecTable.flat_index layout
        A, B = self.A_tab[lengths], self.B_tab[lengths]
        occ = torch.zeros(N, 256 * P, device=byte_ids.device, dtype=torch.float32)
        occ.scatter_add_(1, idx, (A[:, None] * valid).float())
        return occ + B[:, None]

    def forward(self, byte_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.tanh(self.fc1(self.code(byte_ids, lengths))))


def build_generator(out_dim: int, cfg: GeneratorConfig) -> nn.Module:
    if cfg.kind == "bytecnn":
        return ByteCNN(out_dim, cfg)
    if cfg.kind == "occmlp":
        return OccMLP(out_dim, cfg)
    raise ValueError(f"unknown generator kind {cfg.kind}")


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())
