"""Inference-only analyses of a trained checkpoint (R4), run on the GPU worker.

For one run and one split it writes, under <run>/analysis_<split>/:
  minting.json / minting.npz   regret, beats-prefix, rank per site and leak, per rule
  retok.json                   bpb on text re-tokenised with the held-out merges
  terms.json                   share of target-logit variance from KAS / prior / correction
  ablations.json               bpb with the correction or prior residual zeroed (KAS-G, KAS-U16)
  counterfactual.json          KAS-G minting regret when the generator reads another
                               held-out merge's bytes (spelling counterfactual)
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from .codec import CodecTable
from .data import TokenData, eval_windows
from .evaluate import LN2, bpb_from, per_position_nll
from .heads import derangement
from .minting import HeldoutInfo, heldout_info, hidden_states, minting_eval, retokenized_bpb, rules_for
from .train import RunConfig, build_model
from .vocab import WorkingVocab


def load_run(run_dir: Path, data_root: Path, device):
    ck = torch.load(Path(run_dir) / "model.pt", map_location="cpu", weights_only=False)
    cfg = RunConfig(**ck["config"])
    vocab = WorkingVocab.load(Path(data_root) / "vocab.json")
    unigram = np.load(Path(data_root) / "unigram_train.npy")
    model, _ = build_model(cfg, vocab, unigram)
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    return model, cfg, vocab, unigram


@torch.no_grad()
def term_shares(model, H: torch.Tensor, windows: np.ndarray, device) -> dict:
    """Decompose the target logit z_y = KAS_y + prior_y + corr_y (+ length bias) and report
    each term's share Cov(term, z)/Var(z) over positions, plus raw variances."""
    head = model.head
    if head.kind == "dense" and getattr(head, "beta", None) is None:
        return {}
    y = torch.as_tensor(windows[:, 1:], device=device).reshape(-1)
    h = H.reshape(-1, H.shape[-1])
    if head.kind == "dense":
        # prior-matched Dense (protocol 08): z_y = <h, W_y> + beta_y
        parts = {"dense": (h * head.W[y].float()).sum(-1), "prior": head.beta[y].float()}
    else:
        head._to_device(device)
        E_kas = head.codec.matmul(head.W_out)
        kas = (h * E_kas[y].float()).sum(-1)
        lens = head.lengths[y]
        parts = {"kas": kas, "length_bias": head.len_bias[lens].float()}
        if head.kind != "kas0":
            parts["prior"] = head.beta[y].float()
        r = head.cfg.rank
        if head.kind == "kasu":
            parts["correction"] = (h @ head.C.t().float() * head.U[y].float()).sum(-1)
        elif head.kind in ("kasg", "kasg_shuf"):
            src = head.gen_src
            g = head.generate(head.gen_bytes[src], head.gen_lens[src]).float()
            parts["correction"] = (h @ head.C.t().float() * g[y, :r]).sum(-1)
            parts["prior_residual"] = g[y, r]
    total = sum(parts.values())
    vz = float(total.var())
    out = {"var_total": vz}
    for k, v in parts.items():
        out[f"share_{k}"] = float(((v - v.mean()) * (total - total.mean())).mean() / vz)
        out[f"var_{k}"] = float(v.var())
    return out


@torch.no_grad()
def ablation_bpb(model, windows, target_bytes, device) -> dict:
    """bpb with the correction term (and, for KAS-G, the prior residual) removed."""
    head = model.head
    if head.kind == "dense" and getattr(head, "beta", None) is not None:
        # prior-matched Dense (protocol 08): bpb with the learned bias removed at test time
        E_in = model.input_table_rows(torch.arange(model.n_vocab, device=device))
        E, b = head.materialize()
        nb = target_bytes[windows[:, 1:]]
        return {"bpb_full": bpb_from(per_position_nll(model, windows, device, tables=(E_in, E, b)), nb),
                "bpb_no_prior": bpb_from(per_position_nll(model, windows, device, tables=(E_in, E, None)), nb)}
    if head.kind not in ("kasu", "kasg", "kasg_shuf"):
        return {}
    out = {}
    head._to_device(device)
    E_in = model.input_table_rows(torch.arange(model.n_vocab, device=device))
    E, b = head.materialize()
    E_kas = head.codec.matmul(head.W_out)
    nb = target_bytes[windows[:, 1:]]
    base = per_position_nll(model, windows, device, tables=(E_in, E, b))
    out["bpb_full"] = bpb_from(base, nb)
    out["bpb_no_correction"] = bpb_from(per_position_nll(model, windows, device, tables=(E_in, E_kas, b)), nb)
    if head.kind in ("kasg", "kasg_shuf"):
        g = head.generate(head.gen_bytes[head.gen_src], head.gen_lens[head.gen_src]).float()
        b2 = b - g[:, head.cfg.rank]
        out["bpb_no_prior_residual"] = bpb_from(per_position_nll(model, windows, device, tables=(E_in, E, b2)), nb)
        out["bpb_no_generated_terms"] = bpb_from(per_position_nll(model, windows, device,
                                                                  tables=(E_in, E_kas, b2)), nb)
    return out


def _np(obj):
    if isinstance(obj, dict):
        return {k: _np(v) for k, v in obj.items()}
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def analyze_run(run_dir: Path, data_root: Path, split: str = "dev", device=None,
                retok_rules: tuple[str, ...] | None = None, max_windows: int | None = None,
                out_dir: Path | None = None) -> dict:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    run_dir = Path(run_dir)
    out = Path(out_dir) if out_dir is not None else run_dir / f"analysis_{split}"
    out.mkdir(parents=True, exist_ok=True)
    model, cfg, vocab, unigram = load_run(run_dir, data_root, device)
    data = TokenData.load(Path(data_root), verify=False)
    stream = {"dev": data.dev, "test": data.test}[split]
    windows = eval_windows(stream, cfg.seq_len, max_windows)
    if max_windows is not None:
        stream = stream[: max_windows * cfg.seq_len + 1]
    target_bytes = vocab.target_bytes()
    pair_count = np.load(Path(data_root) / "heldout_pair_count_train.npy")
    info = heldout_info(vocab, unigram, pair_count)
    from .train import attention_context
    ctx = attention_context(device)
    ctx.__enter__()
    try:
        states = hidden_states(model, windows, device)
        res = minting_eval(model, windows, vocab, info, device, target_bytes, states=states)
        arrays = {"site_w": res["sites"]["w"], "site_j": res["sites"]["j"], "site_m": res["sites"]["m"],
                  "site_a": res["sites"]["a"], "site_b": res["sites"]["b"]}
        summary = {"run_id": cfg.run_id, "head": cfg.head, "split": split, "n_sites": res["n_sites"],
                   "n_positions": res["n_positions"], "total_bytes": res["total_bytes"], "rules": {}}
        for name, r in res["rules"].items():
            arrays[f"{name}__regret"] = r["regret_bits"]
            arrays[f"{name}__beats"] = r["beats"]
            arrays[f"{name}__rank"] = r["rank"]
            arrays[f"{name}__leak_per_window"] = r["leak_per_window_nats"]
            summary["rules"][name] = {k: r[k] for k in ("prior", "correction", "mean_regret_bits",
                                                        "beats_prefix", "median_rank", "leak_bpb",
                                                        "leak_mean_nats_per_pos")}
        np.savez_compressed(out / "minting.npz", **arrays)
        (out / "minting.json").write_text(json.dumps(_np(summary), indent=2), encoding="utf-8")

        if model.head.kind in ("kasp", "kasu", "kasg", "kasg_shuf") or getattr(model.head, "beta", None) is not None:
            from .minting import prior_drift
            (out / "prior_drift.json").write_text(json.dumps(prior_drift(model, info), indent=2), encoding="utf-8")
        terms = term_shares(model, states[0], windows, device)
        (out / "terms.json").write_text(json.dumps(terms, indent=2), encoding="utf-8")
        H = None
        del states
        torch.cuda.empty_cache() if device.type == "cuda" else None

        abl = ablation_bpb(model, windows, target_bytes, device)
        (out / "ablations.json").write_text(json.dumps(abl, indent=2), encoding="utf-8")

        cf = {}
        if model.head.kind == "kasg":
            perm = derangement(len(info.ids), 777)
            cf_table = CodecTable(info.table.byte_buffer[perm], info.table.lengths[perm],
                                  pos_dim=info.table.pos_dim)
            info_cf = HeldoutInfo(**{**info.__dict__, "standin": cf_table})
            # feed the generator another held-out merge's bytes (the KAS term keeps the true bytes)
            saved_kind = model.head.kind
            model.head.kind = "kasg_shuf"
            try:
                rules = [(n, p, c) for n, p, c in rules_for("kasg") if c == "generate"]
                res_cf = minting_eval(model, windows, vocab, info_cf, device, target_bytes, rules=rules)
            finally:
                model.head.kind = saved_kind
            for n, r in res_cf["rules"].items():
                cf[n] = {"mean_regret_bits_counterfactual": r["mean_regret_bits"],
                         "mean_regret_bits_true": summary["rules"][n]["mean_regret_bits"],
                         "beats_prefix_counterfactual": r["beats_prefix"]}
            np.savez_compressed(out / "counterfactual.npz",
                                **{f"{n}__regret": r["regret_bits"] for n, r in res_cf["rules"].items()})
        (out / "counterfactual.json").write_text(json.dumps(cf, indent=2), encoding="utf-8")

        retok = {}
        base_stream = np.asarray(stream, dtype=np.int64)
        names = retok_rules or tuple(n for n, _, _ in rules_for(cfg.head, cfg.dense_bias))
        for name, p, c in rules_for(cfg.head, cfg.dense_bias):
            if name in names:
                retok[name] = retokenized_bpb(model, vocab, info, base_stream, p, c, device, cfg.seq_len,
                                              target_bytes)
        # each rule's entry carries the ordinary bpb of the same text ("bpb_standard_same_text")
        (out / "retok.json").write_text(json.dumps(retok, indent=2), encoding="utf-8")
    finally:
        ctx.__exit__(None, None, None)
    rec = {"run_id": cfg.run_id, "split": split, "wall_s": time.time() - t0,
           "best_rule_by_regret": min(summary["rules"], key=lambda k: summary["rules"][k]["mean_regret_bits"])}
    (out / "done.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
    return rec


def main(data_dir: Path, work: Path, runs: list[str], split: str = "dev", splits: list[str] | None = None,
         **kw) -> dict:
    done = {}
    for r in runs:
        rd = Path(work) / "runs" / r
        out_root = rd
        if not (rd / "model.pt").exists():
            # re-analysis: checkpoint from a previous kernel's output (read-only input mount)
            hits = list(Path("/kaggle/input").rglob(f"{r}/model.pt")) if Path("/kaggle/input").exists() else []
            if not hits:
                done[r] = "missing checkpoint"
                continue
            rd = hits[0].parent
        for sp in (splits or [split]):
            try:
                done[f"{r}:{sp}"] = analyze_run(rd, Path(data_dir), sp, out_dir=out_root / f"analysis_{sp}", **kw)
            except Exception as e:  # noqa: BLE001
                import traceback
                done[f"{r}:{sp}"] = {"error": repr(e), "traceback": traceback.format_exc()[-3000:]}
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return done
