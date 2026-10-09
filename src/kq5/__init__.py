"""Kronecker byte-derived output heads: where do a new token's output parameters come from?

Package layout:
  codec      Kronecker byte codec, exact K @ W products via gather
  gpt2bytes  raw GPT-2 token bytes (never tokenizer.decode)
  vocab      working vocabulary, merge tokenisation
  model      GPT backbone with a Kronecker input path
  heads      Dense / KAS-0 / KAS-P / KAS-U16 / KAS-G / KAS-G-shuf output heads
  generator  byte -> per-token correction networks
  data       token-stream batching shared by every arm
  train      training loop (one arm, one GPU)
  evaluate   bpb and per-position NLL
  minting    vocabulary growth, regret, leak, beats-prefix
  stats      paired bootstrap and hypothesis verdicts
"""

__version__ = "0.1.0"
