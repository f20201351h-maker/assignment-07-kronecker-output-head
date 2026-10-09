# Protocol 01 — R1 smoke, throughput and head learning rate (written before the run)

Not evidence for H1–H4. Decides model size, micro-batch and the KAS head learning-rate
multiplier. Kernel `kq5-r1-smoke`, T4×2, two independent workers.

## Runs
* GPU0 — throughput: backbones 8L×512 (8 heads) and 6L×768 (12 heads) × heads
  {dense, kasp, kasu, kasg}; 40 optimiser steps of 64 sequences × 1024 tokens (65,536 tokens),
  micro-batch 8 sequences, fp16 autocast + GradScaler, SDPA efficient-attention backend.
  Measure tokens/s (median of logged intervals after step 10) and peak allocated memory.
* GPU1 — head LR: 8L×512, KAS-P with head_lr_mult ∈ {1, 0.1, 0.01} and Dense with 1.0;
  base LR 1e-3, AdamW(0.9, 0.95), wd 0.1 on ≥2-D tensors, warmup 1M tokens, cosine to 10%,
  8,388,608 tokens each; dev_mid bpb (first 256 dev windows) at 4M and 8M tokens.

## Predictions
* Throughput is nearly head-independent because every head materialises E (V×d) once per
  step and pays the same h·Eᵀ matmul; differences < 15% between dense and KAS. KAS-G adds the
  generator over 57K tokens per step: < 10% overhead.
* 8L×512: 22–32K tok/s; 6L×768: 13–20K tok/s.
* KAS head at head_lr_mult = 1 trains but is the noisiest (logit changes scale with
  ‖κ‖₁·‖h‖₁); 0.1 is best or tied; 0.01 is under-trained at 8M tokens.

## Decision rules (fixed now)
1. Backbone: the larger (6L×768) if its slowest arm reaches ≥ 15K tok/s on one T4 (so a 100M-
   token run takes ≤ 1.9 h and ten canonical runs fit in ≈ 10 quota-hours on T4×2); otherwise
   8L×512. Both put KAS arms in the tens of millions of parameters.
2. Micro-batch: largest of {8, 16} sequences whose peak memory stays below 13 GB on every arm.
3. KAS head_lr_mult: the value with the lowest dev_mid bpb at 8M tokens; a run that diverges
   or whose loss is non-finite is excluded. The same multiplier is used for every KAS arm.
   The dense head keeps 1.0 (standard); this asymmetry is disclosed.
4. Base LR stays 1e-3 unless the dense run fails to train; then the plan is revised in writing.
