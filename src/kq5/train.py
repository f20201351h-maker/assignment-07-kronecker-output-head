"""One arm, one GPU. Same data, order, steps and schedule for every arm.

Each optimiser step materialises the input table K W_in and the head (E, b)
once, runs the micro-batches against detached copies, then pushes the
accumulated table gradients back through the codec / generator in one call.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import platform
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from .codec import CodecTable
from .data import TokenData, TrainBatches, eval_windows, sha256_array, to_device
from .evaluate import autocast_for, evaluate_bpb, materialize
from .generator import GeneratorConfig
from .heads import HeadConfig, centred_log_unigram, head_param_counts, head_param_groups
from .model import GPT, ModelConfig, param_report
from .vocab import WorkingVocab


@dataclass
class RunConfig:
    run_id: str
    head: str
    seed: int = 1
    data_seed: int = 1234
    n_layer: int = 8
    n_head: int = 8
    d_model: int = 512
    seq_len: int = 1024
    rank: int = 16
    corr_init: str = "c_zero"               # see heads.HeadConfig; canonical KAS-U16/KAS-G use "u_zero"
    vocab_dropout: float = 0.0              # kasu only; Phase 2 (protocol 06) KAS-U16-drop25 uses 0.25
    vocab_dropout_target: str = "both"      # "both" (prior + correction, protocol 06) or "u" (amendment 06a)
    dense_bias: bool = False                # dense only; protocol 08 prior-matched Dense (trainable log-unigram bias)
    generator: dict = field(default_factory=dict)
    lr: float = 1e-3
    min_lr_frac: float = 0.1
    warmup_tokens: int = 2_000_000
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)
    grad_clip: float = 1.0
    head_lr_mult: float = 1.0
    gen_lr_mult: float = 1.0
    batch_seqs: int = 64
    micro_seqs: int = 8
    total_tokens: int = 20_000_000
    stop_tokens: int | None = None          # stop early (smoke tests); schedule still uses total_tokens
    eval_at_tokens: list = field(default_factory=list)
    eval_windows_mid: int = 256
    final_dev_windows: int | None = None    # None = all dev windows
    final_test: bool = False
    amp: bool = True
    require_cuda: bool = True
    time_limit_s: float = 11 * 3600
    log_every: int = 10
    save_checkpoint: bool = True
    out_dir: str = "runs/x"
    data_dir: str = "data/tokens"


def build_model(cfg: RunConfig, vocab: WorkingVocab, unigram: np.ndarray) -> tuple[GPT, np.ndarray]:
    table = CodecTable.from_bytes(vocab.tokens, pos_dim=vocab.pos_dim)
    V = vocab.n_train_vocab
    prior, mean_lp, n_train = centred_log_unigram(unigram, V)
    hc = HeadConfig(kind=cfg.head, rank=cfg.rank, generator=GeneratorConfig(**cfg.generator),
                    corr_init=cfg.corr_init, vocab_dropout=cfg.vocab_dropout,
                    vocab_dropout_target=cfg.vocab_dropout_target, dense_bias=cfg.dense_bias)
    mc = ModelConfig(n_layer=cfg.n_layer, n_head=cfg.n_head, d_model=cfg.d_model,
                     seq_len=cfg.seq_len, head=hc)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    needs_prior = cfg.head not in ("dense", "kas0") or (cfg.head == "dense" and cfg.dense_bias)
    model = GPT(mc, table, V, prior if needs_prior else None)
    return model, prior


def make_optimizer(model: GPT, cfg: RunConfig) -> torch.optim.Optimizer:
    groups = head_param_groups(model.head)
    mult = {}
    for p in groups["head_matrix"] + groups["head_small"]:
        mult[id(p)] = cfg.head_lr_mult
    for p in groups["generator"]:
        mult[id(p)] = cfg.gen_lr_mult
    buckets: dict[tuple, list] = {}
    for p in model.parameters():
        if not p.requires_grad:
            continue
        key = (mult.get(id(p), 1.0), cfg.weight_decay if p.dim() >= 2 else 0.0)
        buckets.setdefault(key, []).append(p)
    param_groups = [{"params": ps, "lr_mult": m, "weight_decay": wd, "lr": cfg.lr * m}
                    for (m, wd), ps in sorted(buckets.items())]
    fused = next(model.parameters()).is_cuda
    return torch.optim.AdamW(param_groups, lr=cfg.lr, betas=tuple(cfg.betas), eps=1e-8, fused=fused)


def lr_at(step: int, total_steps: int, warmup_steps: int, cfg: RunConfig) -> float:
    if step < warmup_steps:
        return cfg.lr * (step + 1) / warmup_steps
    t = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
    return cfg.lr * (cfg.min_lr_frac + (1 - cfg.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * t)))


def environment() -> dict:
    env = {"python": platform.python_version(), "torch": torch.__version__,
           "numpy": np.__version__, "platform": platform.platform(),
           "cuda_available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        env["cuda"] = torch.version.cuda
        env["device_name"] = torch.cuda.get_device_name(0)
        env["device_count_visible"] = torch.cuda.device_count()
        env["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    return env


def attention_context(device):
    if str(device).startswith("cuda"):
        from torch.nn.attention import SDPBackend, sdpa_kernel
        return sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION])
    from torch.nn.attention import SDPBackend, sdpa_kernel
    return sdpa_kernel([SDPBackend.MATH])


def train(cfg: RunConfig, data: TokenData | None = None) -> dict:
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if cfg.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA required but not available; refusing to train on CPU")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t_start = time.time()
    data = data or TokenData.load(Path(cfg.data_dir))
    vocab = WorkingVocab.load(Path(data.root) / "vocab.json")
    unigram = np.load(Path(data.root) / "unigram_train.npy")
    model, prior = build_model(cfg, vocab, unigram)
    model.to(device)
    opt = make_optimizer(model, cfg)
    head_ids = {id(p) for p in model.head.parameters()}
    head_params = [p for p in model.parameters() if id(p) in head_ids]
    body_params = [p for p in model.parameters() if id(p) not in head_ids]
    use_amp = cfg.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    T = cfg.seq_len
    tokens_per_step = cfg.batch_seqs * T
    total_steps = cfg.total_tokens // tokens_per_step
    stop_steps = total_steps if cfg.stop_tokens is None else min(total_steps, cfg.stop_tokens // tokens_per_step)
    warmup_steps = max(1, cfg.warmup_tokens // tokens_per_step)
    batches = TrainBatches(data.train, T, cfg.batch_seqs, cfg.data_seed)
    if total_steps > batches.max_steps:
        raise ValueError(f"budget {total_steps} steps exceeds one epoch ({batches.max_steps})")
    n_micro = cfg.batch_seqs // cfg.micro_seqs
    assert n_micro * cfg.micro_seqs == cfg.batch_seqs
    eval_steps = sorted({min(total_steps, math.ceil(t / tokens_per_step)) for t in cfg.eval_at_tokens})
    target_bytes = vocab.target_bytes()
    dev_mid = eval_windows(data.dev, T, cfg.eval_windows_mid)
    held_lo = vocab.n_train_vocab

    record = {
        "config": dataclasses.asdict(cfg), "environment": environment(),
        "params": param_report(model), "head_params": head_param_counts(model.head),
        "data_manifest_files": data.manifest["files"],
        "train_order_sha256": batches.order_sha256,
        "first_step_windows_sha256": sha256_array(batches.windows(0).astype(np.int64)),
        "dev_mid_sha256": sha256_array(dev_mid),
        "total_steps": total_steps, "tokens_per_step": tokens_per_step,
        "warmup_steps": warmup_steps, "eval_steps": eval_steps, "n_vocab": vocab.n_train_vocab,
    }
    (out / "run_record.json").write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    log = open(out / "metrics.jsonl", "a", encoding="utf-8")

    def write(rec):
        log.write(json.dumps(rec) + "\n")
        log.flush()

    status = "running"
    step = 0
    skipped_steps = 0
    tok_t0 = time.time()
    running_loss = 0.0
    attn = attention_context(device)
    attn.__enter__()
    try:
        while step < stop_steps:
            lr = lr_at(step, total_steps, warmup_steps, cfg)
            for g in opt.param_groups:
                g["lr"] = lr * g["lr_mult"]
            batch = batches.batch(step)
            if batch.max() >= held_lo:
                raise RuntimeError("held-out merge id in a training batch")
            # materialise tables once per step
            E_in = model.input_table_rows(torch.arange(vocab.n_train_vocab, device=device))
            E, b = model.head.materialize()
            E_in_l = E_in.detach().requires_grad_(True)
            E_l = E.detach().requires_grad_(True)
            b_l = b.detach().requires_grad_(True) if b is not None else None
            loss_sum = 0.0
            for m in range(n_micro):
                x, y = to_device(batch[m * cfg.micro_seqs:(m + 1) * cfg.micro_seqs], device)
                with autocast_for(device) if use_amp else nullcontext():
                    nll = model.nll(x, y, E_in_l, E_l, b_l)
                    loss = nll.mean() / n_micro
                scaler.scale(loss).backward()
                loss_sum += float(loss.detach())
            roots, grads = [E_in, E], [E_in_l.grad, E_l.grad]
            if b is not None and b.requires_grad:
                roots.append(b); grads.append(b_l.grad)
            torch.autograd.backward(roots, grads)
            scaler.unscale_(opt)
            # clip the head and the rest separately: a byte-derived head's gradient norm is
            # ~100x a dense head's, and a global clip would shrink the body's gradients only
            # in KAS arms (Adam's eps then damps the body); per-group clipping keeps the body
            # treatment identical across arms
            gnorm = torch.nn.utils.clip_grad_norm_(body_params, cfg.grad_clip)
            gnorm_head = torch.nn.utils.clip_grad_norm_(head_params, cfg.grad_clip)
            scale_before = scaler.get_scale() if use_amp else 1.0
            scaler.step(opt)
            scaler.update()
            if use_amp and scaler.get_scale() < scale_before:
                skipped_steps += 1        # fp16 overflow: GradScaler skipped this update
            opt.zero_grad(set_to_none=True)
            step += 1
            running_loss += loss_sum
            if not math.isfinite(loss_sum):
                raise RuntimeError(f"non-finite loss at step {step}")
            if step % cfg.log_every == 0 or step == stop_steps:
                dt = time.time() - tok_t0
                rec = {"step": step, "tokens": step * tokens_per_step, "loss": running_loss / cfg.log_every
                       if step % cfg.log_every == 0 else loss_sum, "lr": lr, "grad_norm": float(gnorm), "grad_norm_head": float(gnorm_head),
                       "scale": float(scaler.get_scale()) if use_amp else 1.0, "skipped_steps": skipped_steps,
                       "tok_per_s": cfg.log_every * tokens_per_step / dt if step % cfg.log_every == 0 else None,
                       "elapsed_s": time.time() - t_start}
                if device.type == "cuda":
                    rec["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
                write(rec)
                running_loss = 0.0
                tok_t0 = time.time()
            if step in eval_steps:
                ev = evaluate_bpb(model, dev_mid, target_bytes, device, batch=cfg.micro_seqs)
                nbw = target_bytes[dev_mid[:, 1:]]
                curve = out / "evalcurve"
                curve.mkdir(exist_ok=True)
                np.savez_compressed(curve / f"step{step:06d}.npz",
                                    nll_sum=(ev["nll"] * (nbw > 0)).sum(1, dtype=np.float64),
                                    bytes=nbw.sum(1, dtype=np.float64), tokens=step * tokens_per_step)
                write({"eval": "dev_mid", "step": step, "tokens": step * tokens_per_step,
                       "bpb": ev["bpb"], "nll": ev["nll_nats_per_token"], "n_targets": ev["n_targets"]})
            if time.time() - t_start > cfg.time_limit_s:
                status = "time_limit"
                break
        else:
            status = "complete"
        final = {}
        dev_all = eval_windows(data.dev, T, cfg.final_dev_windows)
        ev = evaluate_bpb(model, dev_all, target_bytes, device, batch=cfg.micro_seqs)
        np.save(out / "nll_dev.npy", ev.pop("nll"))
        final["dev"] = {**ev, "windows_sha256": sha256_array(dev_all)}
        if cfg.final_test:
            test_all = eval_windows(data.test, T, None)
            ev = evaluate_bpb(model, test_all, target_bytes, device, batch=cfg.micro_seqs)
            np.save(out / "nll_test.npy", ev.pop("nll"))
            final["test"] = {**ev, "windows_sha256": sha256_array(test_all)}
        write({"final": final, "step": step, "tokens": step * tokens_per_step, "status": status})
        if cfg.save_checkpoint:
            torch.save({"model": model.state_dict(), "config": dataclasses.asdict(cfg), "step": step},
                       out / "model.pt")
    finally:
        attn.__exit__(None, None, None)
        log.close()
    summary = {"run_id": cfg.run_id, "status": status, "steps": step, "skipped_steps": skipped_steps,
               "tokens": step * tokens_per_step, "final": final,
               "wall_s": time.time() - t_start, "params": record["params"]}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
