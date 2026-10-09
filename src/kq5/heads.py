"""Output heads. Every head materialises (E, b) with logits = h @ E.T + b.

    logit_v(h) = KAS_v(h) + beta_v + <u_v, C h>,   KAS_v(h) = kappa_v^T W_out h

  dense      E = W (V x d, stored)                         b = 0
  dense + dense_bias (protocol 08, prior-matched Dense)   b = beta (learned, log-unigram init)
  kas0       E = K W_out                                   b = len_bias[L_v]
  kasp       as kas0                                       b += beta (log-unigram, fixed)
  kasu       E = K W_out + U C  (U stored V x r)           b += beta (learned, log-unigram init)
  kasg       E = K W_out + G(bytes_v)[:r] C                b += beta (fixed) + G(bytes_v)[r]
  kasg_shuf  as kasg, but G reads the bytes of a fixed random other token

K is the fixed Kronecker code (CodecTable); W_out is (D x d) with D = 256*32,
independent of V. C is zero-initialised so every corrected arm starts as the
uncorrected head (function-preserving at init).

Minting: ``rows_for`` returns (E, b) rows for tokens that were never part of
training, under a named rule for the parameters those tokens lack.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

from .codec import CodecTable, TorchCodec
from .generator import GeneratorConfig, build_generator

HEADS = ("dense", "kas0", "kasp", "kasu", "kasg", "kasg_shuf")


@dataclass
class HeadConfig:
    kind: str = "kasp"
    rank: int = 16
    init_std: float = 0.02          # std of the materialised rows E at init
    len_bias: bool = True
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    shuffle_seed: int = 4242
    # Initialisation of the rank-r correction <u_v, C h>:
    #  "c_zero": C = 0, stored U ~ N(0,1), generator u-rows scaled to unit std (pilot R2, kernel X).
    #            Under Adam the stored table barely leaves its random init (pilot: corr(U, U0) = 0.9997).
    #  "u_zero": per-token u starts at 0 (U = 0; generator u-rows = 0), C ~ N(0, 1/d) (LoRA convention;
    #            canonical KAS-U16 / KAS-G from protocol amendment 03b).
    corr_init: str = "c_zero"
    # Vocabulary dropout (reference report, "Vocabulary dropout is what makes it work"; Phase 2,
    # protocol 06): fraction of tokens whose per-token row (learned prior beta_v and stored
    # correction u_v) is hidden at random each training step. kasu only; 0 = off (Phase 1).
    vocab_dropout: float = 0.0
    # Amendment 06a: which per-token parameters a hidden row loses. "both" (protocol 06 as written) hides the
    # learned prior beta_v and u_v; "u" hides only the stored correction u_v and keeps the prior.
    vocab_dropout_target: str = "both"
    # Protocol 08 (prior-matched Dense): give the dense head the same per-token bias the KAS-U heads have —
    # a trainable beta_v initialised to the centred log-unigram prior (same parameter group, lr and no weight
    # decay as KAS-U's beta). dense only; False = Phase 1-3 Dense (no bias).
    dense_bias: bool = False


def centred_log_unigram(counts: np.ndarray, n_vocab: int) -> np.ndarray:
    """log((c+1)/(N+V)) minus its mean over the training vocabulary (reference-report
    formula; centring is a softmax no-op and fixes what 'zero' means)."""
    c = np.asarray(counts[:n_vocab], dtype=np.float64)
    lp = np.log((c + 1.0) / (c.sum() + n_vocab))
    return lp - lp.mean(), float(lp.mean()), float(c.sum())


def derangement(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    fixed = np.flatnonzero(perm == np.arange(n))
    for i in fixed:  # swap each fixed point with a random other position
        j = (i + 1 + rng.integers(0, n - 1)) % n
        perm[i], perm[j] = perm[j], perm[i]
    assert not np.any(perm == np.arange(n))
    return perm


class Head(nn.Module):
    def __init__(self, cfg: HeadConfig, d_model: int, codec: CodecTable, n_vocab: int,
                 prior: np.ndarray | None):
        super().__init__()
        if cfg.kind not in HEADS:
            raise ValueError(cfg.kind)
        self.cfg = cfg
        self.kind = cfg.kind
        self.n_vocab = n_vocab
        self.d_model = d_model
        self.pos_dim = codec.pos_dim
        if codec.V != n_vocab:
            raise ValueError("head codec must cover exactly the training vocabulary")
        self.table = codec
        self.codec = TorchCodec(codec)
        r = cfg.rank
        if self.kind == "dense":
            self.W = nn.Parameter(torch.randn(n_vocab, d_model) * cfg.init_std)
            if cfg.dense_bias:
                # drawn after W, from the prior (no RNG): W and the body are bit-identical to the plain
                # Dense run with the same seed; the bias is the only difference
                if prior is None:
                    raise ValueError("dense_bias needs a prior table")
                self.beta = nn.Parameter(torch.as_tensor(prior[:n_vocab], dtype=torch.float32).clone())
            return
        D = codec.D
        self.W_out = nn.Parameter(torch.randn(D, d_model) * (cfg.init_std / math.sqrt(D - 1)))
        if cfg.len_bias:
            self.len_bias = nn.Parameter(torch.zeros(codec.pos_dim + 1))
        if self.kind in ("kasp", "kasu", "kasg", "kasg_shuf"):
            if prior is None:
                raise ValueError(f"{self.kind} needs a prior table")
            beta = torch.as_tensor(prior[:n_vocab], dtype=torch.float32)
            if self.kind == "kasu":
                self.beta = nn.Parameter(beta.clone())
            else:
                self.register_buffer("beta", beta.clone())
        if self.kind in ("kasu", "kasg", "kasg_shuf"):
            if cfg.corr_init == "u_zero":
                self.C = nn.Parameter(torch.randn(r, d_model) / math.sqrt(d_model))
            elif cfg.corr_init == "c_zero":
                self.C = nn.Parameter(torch.zeros(r, d_model))
            else:
                raise ValueError(cfg.corr_init)
        if self.kind == "kasu":
            self.U = nn.Parameter(torch.zeros(n_vocab, r) if cfg.corr_init == "u_zero"
                                  else torch.randn(n_vocab, r))
        if self.kind in ("kasg", "kasg_shuf"):
            self.gen = build_generator(r + 1, cfg.generator)
            src = np.arange(n_vocab)
            if self.kind == "kasg_shuf":
                src = derangement(n_vocab, cfg.shuffle_seed)
            self.register_buffer("gen_src", torch.as_tensor(src, dtype=torch.long))
            self.register_buffer("gen_bytes", torch.as_tensor(codec.byte_buffer, dtype=torch.long))
            self.register_buffer("gen_lens", torch.as_tensor(codec.lengths, dtype=torch.long))
            self._calibrate_generator()
        self.register_buffer("lengths", torch.as_tensor(codec.lengths, dtype=torch.long))

    # ----- generator plumbing -----
    @torch.no_grad()
    def _calibrate_generator(self) -> None:
        """Scale the u-rows of the output layer so u has unit std at init; the
        prior-residual row starts at exactly zero."""
        r = self.cfg.rank
        if self.cfg.corr_init == "u_zero":      # u starts at exactly zero, like a zero-initialised U
            self.gen.fc2.weight.zero_()
            self.gen.fc2.bias.zero_()
            return
        sub = torch.as_tensor(np.random.default_rng(0).choice(
            self.n_vocab, size=min(8192, self.n_vocab), replace=False), dtype=torch.long)
        src = self.gen_src[sub]
        g = self.gen(self.gen_bytes[src], self.gen_lens[src])
        s = g[:, :r].std().item()
        self.gen.fc2.weight[:r] /= max(s, 1e-6)
        self.gen.fc2.bias[:r] /= max(s, 1e-6)
        self.gen.fc2.weight[r:].zero_()
        self.gen.fc2.bias[r:].zero_()

    GEN_CHUNK = 8192

    def _gen_chunk(self, byte_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        with torch.autocast("cuda", dtype=torch.float16, enabled=byte_ids.is_cuda):
            out = self.gen(byte_ids, lengths)
        return out.to(self.W_out.dtype)

    def generate(self, byte_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Generator outputs for a batch of tokens, in chunks; with grad enabled each chunk is
        checkpointed (recomputed in backward) so memory does not scale with V."""
        from torch.utils.checkpoint import checkpoint
        outs = []
        for s in range(0, byte_ids.shape[0], self.GEN_CHUNK):
            b, l = byte_ids[s:s + self.GEN_CHUNK], lengths[s:s + self.GEN_CHUNK]
            if torch.is_grad_enabled():
                outs.append(checkpoint(self._gen_chunk, b, l, use_reentrant=False))
            else:
                outs.append(self._gen_chunk(b, l))
        return torch.cat(outs) if outs else torch.zeros(0, self.cfg.rank + 1, device=byte_ids.device)

    def _to_device(self, device) -> None:
        if self.codec.index.device != device:
            self.codec.to(device)

    # ----- training-time materialisation -----
    def materialize(self) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.kind == "dense":
            return self.W, (self.beta if self.cfg.dense_bias else None)
        self._to_device(self.W_out.device)
        E = self.codec.matmul(self.W_out)
        b = self.len_bias[self.lengths] if self.cfg.len_bias else torch.zeros(
            self.n_vocab, device=E.device)
        r = self.cfg.rank
        if self.kind == "kasu":
            U, beta = self.U, self.beta
            if self.training and self.cfg.vocab_dropout > 0:
                # vocabulary dropout: hide the per-token row (prior and correction) of a random
                # fraction of tokens this step, so the model learns to score a token through its
                # byte-derived logit alone. No rescaling: a hidden row is exactly the state of a
                # minted token under the "zero" rule (prior = training-vocabulary mean, u = 0).
                keep = (torch.rand(self.n_vocab, device=E.device) >= self.cfg.vocab_dropout).to(E.dtype)
                U = U * keep[:, None]
                if self.cfg.vocab_dropout_target == "both":
                    beta = beta * keep
                elif self.cfg.vocab_dropout_target != "u":
                    raise ValueError(self.cfg.vocab_dropout_target)
            b = b + beta
            E = E + U @ self.C
        elif self.kind in ("kasg", "kasg_shuf"):
            b = b + self.beta
            g = self.generate(self.gen_bytes[self.gen_src], self.gen_lens[self.gen_src])
            E = E + g[:, :r] @ self.C
            b = b + g[:, r]
        elif self.kind == "kasp":
            b = b + self.beta
        return E, b

    # ----- minting -----
    @torch.no_grad()
    def rows_for(self, new: CodecTable, prior: torch.Tensor | None, inherit_from: torch.Tensor | None,
                 correction: str, gen_bytes_src: CodecTable | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Rows (E, b) for new tokens.

        prior:        prior value per new token (already chosen by the rule), or None
        inherit_from: training-vocab id per new token whose stored parameters are copied
        correction:   'none' | 'inherit' | 'generate'
        gen_bytes_src: bytes the generator reads (the true bytes, or a shuffled
                       stand-in for kasg_shuf)
        For the dense head the rule is expressed by the caller through prior and
        inherit_from ('mean' rows are handled by passing inherit_from=None and
        correction='mean').
        """
        dev = (self.W if self.kind == "dense" else self.W_out).device
        n = new.V
        if self.kind == "dense":
            if correction == "mean":
                E = self.W.mean(0, keepdim=True).expand(n, -1).clone()
            elif correction == "inherit":
                E = self.W[inherit_from.to(dev)].clone()
            else:
                raise ValueError(correction)
            b = torch.zeros(n, device=dev) if prior is None else prior.to(dev).float()
            return E, b
        tc = TorchCodec(new, dev)
        E = tc.matmul(self.W_out)
        lens = torch.as_tensor(new.lengths, device=dev)
        b = self.len_bias[lens].clone() if self.cfg.len_bias else torch.zeros(n, device=dev)
        if prior is not None:
            b = b + prior.to(dev).float()
        r = self.cfg.rank
        if correction == "none":
            pass
        elif correction == "inherit":
            if self.kind == "kasu":
                E = E + self.U[inherit_from.to(dev)] @ self.C
            elif self.kind in ("kasg", "kasg_shuf"):
                # the prefix's generated parameters: correction AND prior residual (review fix)
                src = self.gen_src[inherit_from.to(dev)]
                g = self.generate(self.gen_bytes[src], self.gen_lens[src])
                E = E + g[:, :r] @ self.C
                b = b + g[:, r]
        elif correction == "generate":
            if self.kind not in ("kasg", "kasg_shuf"):
                raise ValueError("only generator heads can generate")
            gt = gen_bytes_src if gen_bytes_src is not None else new
            g = self.generate(torch.as_tensor(gt.byte_buffer, device=dev).long(),
                              torch.as_tensor(gt.lengths, device=dev).long())
            E = E + g[:, :r] @ self.C
            b = b + g[:, r]
        else:
            raise ValueError(correction)
        return E, b


def head_param_groups(head: Head) -> dict[str, list[nn.Parameter]]:
    """Parameter groups by role, for optimiser settings and accounting."""
    groups: dict[str, list[nn.Parameter]] = {"head_matrix": [], "per_token": [], "generator": [],
                                            "head_small": []}
    for name, p in head.named_parameters():
        if name in ("W", "U", "beta"):
            groups["per_token"].append(p)
        elif name.startswith("gen."):
            groups["generator"].append(p)
        elif name in ("W_out", "C"):
            groups["head_matrix"].append(p)
        else:
            groups["head_small"].append(p)
    return groups


def head_param_counts(head: Head) -> dict[str, int]:
    out = {}
    for name, p in head.named_parameters():
        out[name.split(".")[0] if name.startswith("gen.") else name] = \
            out.get(name.split(".")[0] if name.startswith("gen.") else name, 0) + p.numel()
    out["total"] = sum(p.numel() for p in head.parameters())
    out["V_dependent"] = sum(p.numel() for n, p in head.named_parameters() if n in ("W", "U", "beta"))
    return out
