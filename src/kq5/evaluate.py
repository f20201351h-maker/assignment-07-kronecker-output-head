"""Bits per byte and per-position NLL on fixed evaluation windows.

bpb = sum of NLL (nats) over targets with >0 bytes / (ln 2 * their byte count).
EOS targets carry 0 bytes and are excluded from both sums.
"""

from __future__ import annotations

import math
from contextlib import nullcontext

import numpy as np
import torch

LN2 = math.log(2.0)


def bpb_from(nll: np.ndarray, nbytes: np.ndarray) -> float:
    m = nbytes > 0
    return float(nll[m].sum(dtype=np.float64) / (LN2 * nbytes[m].sum(dtype=np.float64)))


def autocast_for(device) -> object:
    if str(device).startswith("cuda"):
        return torch.autocast("cuda", dtype=torch.float16)
    return nullcontext()


@torch.no_grad()
def materialize(model, n_input: int | None = None):
    E_in = model.input_table_rows(None if n_input is None else torch.arange(
        n_input, device=model.W_in.device))
    E, b = model.head.materialize()
    return E_in, E, b


@torch.no_grad()
def per_position_nll(model, windows: np.ndarray, device, batch: int = 8,
                     tables=None, return_hidden: bool = False):
    """NLL (nats) of every target in every window, shape (n, T), float32."""
    model.eval()
    E_in, E, b = tables if tables is not None else materialize(model, model.n_vocab)
    out = np.zeros((windows.shape[0], windows.shape[1] - 1), dtype=np.float32)
    hidden = [] if return_hidden else None
    for s in range(0, windows.shape[0], batch):
        w = torch.as_tensor(windows[s:s + batch], dtype=torch.long, device=device)
        x, y = w[:, :-1], w[:, 1:]
        with autocast_for(device):
            h = model.body(x, E_in)
            z = model.logits(h, E, b)
            nll = torch.nn.functional.cross_entropy(z.view(-1, z.shape[-1]), y.reshape(-1),
                                                    reduction="none").view(y.shape)
        out[s:s + batch] = nll.float().cpu().numpy()
        if return_hidden:
            hidden.append(h.float().cpu())
    model.train()
    if return_hidden:
        return out, torch.cat(hidden)
    return out


def evaluate_bpb(model, windows: np.ndarray, target_bytes: np.ndarray, device, batch: int = 8,
                 tables=None) -> dict:
    nll = per_position_nll(model, windows, device, batch, tables)
    nb = target_bytes[windows[:, 1:]]
    return {"bpb": bpb_from(nll, nb), "nll_nats_per_token": float(nll[nb > 0].mean()),
            "n_targets": int((nb > 0).sum()), "n_bytes": int(nb.sum()), "nll": nll}
