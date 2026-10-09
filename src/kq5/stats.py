"""Uncertainty and verdicts. All comparisons are paired: every arm is scored on the same
evaluation windows, so we resample windows (clusters of 1024 targets) jointly across arms.
Seed noise is added by resampling seeds within each arm (two-level bootstrap).
"""

from __future__ import annotations

import math

import numpy as np

LN2 = math.log(2.0)


def window_sums(nll: np.ndarray, nbytes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-window NLL sum (nats, targets with >0 bytes) and byte count."""
    m = nbytes > 0
    return (nll * m).sum(1, dtype=np.float64), nbytes.sum(1, dtype=np.float64)


def bpb_of(nll_w: np.ndarray, bytes_w: np.ndarray) -> float:
    return float(nll_w.sum() / (LN2 * bytes_w.sum()))


def two_level_bootstrap(arms: dict[str, list[np.ndarray]], bytes_w: np.ndarray, stat, n_boot: int = 2000,
                        seed: int = 0) -> dict:
    """arms: name -> list over seeds of per-window NLL sums (same windows for every arm).
    stat: f(dict name -> bpb) -> float. Returns point estimate (seed means), percentile CI,
    and the bootstrap draws. Windows are resampled jointly; seeds independently per arm."""
    rng = np.random.default_rng(seed)
    n_w = len(bytes_w)
    point = stat({k: float(np.mean([bpb_of(v, bytes_w) for v in vs])) for k, vs in arms.items()})
    draws = np.empty(n_boot)
    stacked = {k: np.stack(vs) for k, vs in arms.items()}
    for i in range(n_boot):
        w = rng.integers(0, n_w, n_w)
        bsum = bytes_w[w].sum()
        vals = {}
        for k, s in stacked.items():
            pick = rng.integers(0, s.shape[0], s.shape[0])
            vals[k] = float(s[pick][:, w].sum(1).mean() / (LN2 * bsum))
        draws[i] = stat(vals)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"point": point, "ci95": [float(lo), float(hi)], "n_boot": n_boot, "draws": draws}


def cluster_bootstrap_mean(values: np.ndarray, clusters: np.ndarray, n_boot: int = 2000, seed: int = 0) -> dict:
    """Mean of per-site values with a CI that resamples clusters (evaluation windows)."""
    rng = np.random.default_rng(seed)
    uc, inv = np.unique(clusters, return_inverse=True)
    sums = np.bincount(inv, weights=values, minlength=len(uc))
    cnts = np.bincount(inv, minlength=len(uc)).astype(np.float64)
    draws = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, len(uc), len(uc))
        draws[i] = sums[pick].sum() / max(cnts[pick].sum(), 1.0)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"mean": float(values.mean()), "ci95": [float(lo), float(hi)], "n": int(len(values)),
            "n_clusters": int(len(uc))}


def frequency_buckets(train_counts: np.ndarray, edges=(0, 1, 10, 100, 1_000, 10_000, 100_000, 1_000_000, np.inf)):
    """Bucket index for every token id by its training-target count."""
    e = np.asarray(edges, dtype=np.float64)
    idx = np.searchsorted(e, np.asarray(train_counts, dtype=np.float64), side="right") - 1
    labels = []
    for i in range(len(e) - 1):
        lo, hi = e[i], e[i + 1]
        labels.append(f"[{int(lo)},{'inf' if math.isinf(hi) else int(hi)})")
    return np.clip(idx, 0, len(e) - 2), labels


def bucket_nll(nll: np.ndarray, targets: np.ndarray, nbytes: np.ndarray, bucket_of_token: np.ndarray,
               n_buckets: int) -> dict:
    """Per-bucket NLL sums, target counts and byte counts, per window (for clustering)."""
    m = nbytes > 0
    b = bucket_of_token[targets]
    n_w = nll.shape[0]
    out = {"nll": np.zeros((n_buckets, n_w)), "count": np.zeros((n_buckets, n_w)),
           "bytes": np.zeros((n_buckets, n_w))}
    for k in range(n_buckets):
        sel = (b == k) & m
        out["nll"][k] = (nll * sel).sum(1)
        out["count"][k] = sel.sum(1)
        out["bytes"][k] = (nbytes * sel).sum(1)
    return out


def verdict_interval(ci: list[float], threshold: float, higher_is_support: bool = True) -> str:
    lo, hi = ci
    if higher_is_support:
        if lo >= threshold:
            return "SUPPORTED"
        if hi < threshold:
            return "REJECTED"
    else:
        if hi <= threshold:
            return "SUPPORTED"
        if lo > threshold:
            return "REJECTED"
    return "INCONCLUSIVE"
