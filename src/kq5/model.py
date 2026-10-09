"""GPT backbone with a Kronecker input path; identical in every arm.

Input:  e_t = kappa(token_t)^T W_in + p_t      (reference KroneckerEmbedding,
        W_in ~ N(0, 1/D), the only learned input parameter)
Body:   pre-LayerNorm blocks, causal SDPA attention, GELU MLP, no biases.
Output: final LayerNorm, then logits = h @ E.T + b from a Head.

The input table e = K W_in and the head (E, b) are materialised once per
optimiser step over the whole vocabulary; micro-batches index into them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .codec import CodecTable, TorchCodec
from .heads import Head, HeadConfig


@dataclass
class ModelConfig:
    n_layer: int = 8
    n_head: int = 8
    d_model: int = 512
    seq_len: int = 1024
    mlp_ratio: int = 4
    head: HeadConfig = field(default_factory=HeadConfig)


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d = cfg.d_model
        self.n_head = cfg.n_head
        self.ln1 = nn.LayerNorm(d, bias=False)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.ln2 = nn.LayerNorm(d, bias=False)
        self.fc = nn.Linear(d, cfg.mlp_ratio * d, bias=False)
        self.fc_out = nn.Linear(cfg.mlp_ratio * d, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(d, dim=2)
        q = q.view(B, T, self.n_head, d // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, d // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, d // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(y.transpose(1, 2).reshape(B, T, d))
        return x + self.fc_out(F.gelu(self.fc(self.ln2(x))))


class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig, input_codec: CodecTable, n_vocab: int,
                 prior: np.ndarray | None):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.input_table = input_codec          # covers every token id that can appear as input
        self.in_codec = TorchCodec(input_codec)
        D = input_codec.D
        self.W_in = nn.Parameter(torch.randn(D, d) / math.sqrt(D))
        self.pos = nn.Parameter(torch.randn(cfg.seq_len, d) * 0.02)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(d, bias=False)
        for name, p in self.blocks.named_parameters():
            if p.dim() == 2:
                std = 0.02 / math.sqrt(2 * cfg.n_layer) if name.endswith(("proj.weight", "fc_out.weight")) else 0.02
                nn.init.normal_(p, std=std)
        head_codec = CodecTable(input_codec.byte_buffer[:n_vocab], input_codec.lengths[:n_vocab],
                                pos_dim=input_codec.pos_dim)
        self.head = Head(cfg.head, d, head_codec, n_vocab, prior)
        self.n_vocab = n_vocab

    def _sync_codec(self) -> None:
        if self.in_codec.index.device != self.W_in.device:
            self.in_codec.to(self.W_in.device)

    def input_table_rows(self, ids: torch.Tensor | None = None) -> torch.Tensor:
        """K[ids] @ W_in (all input ids if None)."""
        self._sync_codec()
        return self.in_codec.matmul(self.W_in, ids)

    def body(self, idx: torch.Tensor, E_in: torch.Tensor) -> torch.Tensor:
        T = idx.shape[1]
        x = F.embedding(idx, E_in) + self.pos[:T]
        for blk in self.blocks:
            x = blk(x)
        return self.ln_f(x)

    @staticmethod
    def logits(h: torch.Tensor, E: torch.Tensor, b: torch.Tensor | None) -> torch.Tensor:
        z = h @ E.to(h.dtype).t()
        if b is not None:
            z = z.float() + b
        return z.float()

    def nll(self, idx: torch.Tensor, targets: torch.Tensor, E_in: torch.Tensor,
            E: torch.Tensor, b: torch.Tensor | None) -> torch.Tensor:
        """Per-position NLL in nats, shape (B, T), computed in float32."""
        h = self.body(idx, E_in)
        z = self.logits(h, E, b)
        return F.cross_entropy(z.view(-1, z.shape[-1]), targets.reshape(-1),
                               reduction="none").view(targets.shape)


def param_report(model: GPT) -> dict[str, int]:
    body = sum(p.numel() for n, p in model.named_parameters() if n.startswith(("blocks", "ln_f", "pos")))
    inp = model.W_in.numel()
    head = sum(p.numel() for p in model.head.parameters())
    v_dep = sum(p.numel() for n, p in model.head.named_parameters() if n in ("W", "U", "beta"))
    return {"body": body, "input_projection": inp, "head": head, "head_V_dependent": v_dep,
            "head_V_independent": head - v_dep, "total": body + inp + head,
            "total_V_independent": body + inp + head - v_dep}
