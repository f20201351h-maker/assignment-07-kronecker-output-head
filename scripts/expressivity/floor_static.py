"""Static (checkpoint-free) floor measurements for fixed-feature log-linear heads.

Targets:
  * the training unigram distribution (context-free floor);
  * bigram conditionals p(next | previous token) for the most frequent previous tokens,
    estimated from the full 247M-token training stream (a model-free "true" conditional).

For each target and feature family we report  min_s CE(p, softmax(Phi s + b)) - H(p)  in nats per
target (and the bits-per-byte equivalent at the test split's bytes/target ratio).

Also: a census of byte-parallelograms (quadruples a,b,c,d with kappa_a - kappa_b = kappa_c - kappa_d),
their empirical log-odds defect under the bigram conditionals, the residual of the best additive fit to
log-unigram (frequency non-factorisability), and trie / minimal-automaton sizes of the vocabulary.

    python scripts/expressivity/floor_static.py [--contexts 512] [--steps 400]
      -> results/expressivity/floor_static.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from floor_lib import (ROOT, NATS_PER_TARGET_TO_BPB, FeatureSet, fam_hash_signed, fam_kas, fam_ngrams,  # noqa: E402
                       fam_prefix, fam_private, fam_suffix_cells, fit_floor, load_vocab, unigram_bias)

OUT = ROOT / "results/expressivity/floor_static.json"


def bigram_counts(train: np.ndarray, contexts: np.ndarray, V: int, chunk: int = 20_000_000) -> np.ndarray:
    """counts[c, v] for previous tokens in `contexts` (ranks) over the whole training stream."""
    rank = np.full(65536, -1, dtype=np.int64)
    rank[contexts] = np.arange(len(contexts))
    C = len(contexts)
    counts = np.zeros(C * V, dtype=np.int64)
    n = len(train)
    for s in range(0, n - 1, chunk):
        prev = np.asarray(train[s:min(s + chunk, n - 1)], dtype=np.int64)
        nxt = np.asarray(train[s + 1:min(s + chunk, n - 1) + 1], dtype=np.int64)
        r = rank[prev]
        m = (r >= 0) & (nxt < V)
        counts += np.bincount(r[m] * V + nxt[m], minlength=C * V)
    return counts.reshape(C, V)


def build_families(table, tokens, unigram, V):
    fams = {}
    fams["kas"] = fam_kas(table)
    fams["suffix_cells"] = fam_suffix_cells(table, tokens)
    fams["bigram_pos_h4096"] = fam_ngrams(table, tokens, 2, True, 4096)
    fams["bigram_free_h4096"] = fam_ngrams(table, tokens, 2, False, 4096)
    fams["trigram_free_h8192"] = fam_ngrams(table, tokens, 3, False, 8192)
    fams["prefix_ge5"] = fam_prefix(tokens, V, 2, 8, 5)
    fams["private_top1000"] = fam_private(unigram, V, 1000)
    fams["private_top4000"] = fam_private(unigram, V, 4000)
    fams["hash_m2048_k8"] = fam_hash_signed(tokens, V, 2048, 8)
    fams["hash_m8192_k8"] = fam_hash_signed(tokens, V, 8192, 8)
    return fams


def make_fs(V, fams, names) -> FeatureSet:
    fs = FeatureSet(V)
    for n in names:
        per_token, n_local = fams[n]
        assert all(0 <= i < n_local for f in per_token for i, _ in f), f"feature id out of range in {n}"
        fs.add_family(n, per_token, n_local)
    return fs


# feature-set configurations to measure (name -> list of families)
CONFIGS = {
    "kas": ["kas"],
    "kas+suffix": ["kas", "suffix_cells"],
    "kas+bigram_pos": ["kas", "bigram_pos_h4096"],
    "kas+bigram_free": ["kas", "bigram_free_h4096"],
    "kas+trigram_free": ["kas", "trigram_free_h8192"],
    "kas+prefix": ["kas", "prefix_ge5"],
    "kas+suffix+bigram_free+trigram_free": ["kas", "suffix_cells", "bigram_free_h4096", "trigram_free_h8192"],
    "kas+private1000": ["kas", "private_top1000"],
    "kas+private4000": ["kas", "private_top4000"],
    "kas+hash2048": ["kas", "hash_m2048_k8"],
    "kas+hash8192": ["kas", "hash_m8192_k8"],
}


def parallelogram_census(tokens, unigram, V, top_n=8000):
    """Pairs of same-length tokens differing in exactly one byte position, grouped by (L, p, b, b')."""
    order = np.argsort(-unigram)[:top_n]
    by_len = defaultdict(list)
    for v in order:
        by_len[len(tokens[v])].append(int(v))
    groups = defaultdict(list)
    for L, vs in by_len.items():
        if L < 2:
            continue
        # bucket by token with one position wildcarded
        buckets = defaultdict(list)
        for v in vs:
            b = tokens[v]
            for p in range(L):
                buckets[(p, b[:p], b[p + 1:])].append(v)
        for (p, pre, suf), members in buckets.items():
            if len(members) < 2:
                continue
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    a, c = members[i], members[j]
                    ba, bc = tokens[a][p], tokens[c][p]
                    if ba > bc:
                        a, c, ba, bc = c, a, bc, ba
                    groups[(L, p, ba, bc)].append((a, c))
    multi = {k: v for k, v in groups.items() if len(v) >= 2}
    n_pairs = sum(len(v) for v in groups.values())
    n_pairs_in_multi = sum(len(v) for v in multi.values())
    # rank groups by total frequency of their members
    scored = sorted(multi.items(), key=lambda kv: -sum(unigram[a] + unigram[c] for a, c in kv[1]))
    examples = []
    for (L, p, ba, bc), pairs in scored[:12]:
        ex = [(tokens[a].decode("utf-8", "replace"), tokens[c].decode("utf-8", "replace")) for a, c in pairs[:6]]
        examples.append({"L": L, "p": p, "byte_from": chr(ba) if 32 <= ba < 127 else ba,
                         "byte_to": chr(bc) if 32 <= bc < 127 else bc, "n_pairs": len(pairs), "examples": ex})
    return {"top_n_tokens": top_n, "n_single_substitution_pairs": n_pairs, "n_groups": len(groups),
            "n_groups_with_2plus_pairs": len(multi), "n_pairs_in_such_groups": n_pairs_in_multi,
            "examples": examples}, scored


def parallelogram_defect(scored, counts, contexts, unigram, tokens, min_count=20, top_groups=200):
    """For pairs (a,b),(c,d) in one group: under any additive head, logodds(a:b) - logodds(c:d) is
    context-independent. Measure its spread across previous-token contexts in the bigram conditionals."""
    tot = counts.sum(1, keepdims=True).astype(np.float64)
    out = []
    for (L, p, ba, bc), pairs in scored[:top_groups]:
        if len(pairs) < 2:
            continue
        (a, b), (c, d) = pairs[0], pairs[1]
        cnt = counts[:, [a, b, c, d]].astype(np.float64)
        ok = (cnt >= min_count).all(1)
        if ok.sum() < 8:
            continue
        lp = np.log(cnt[ok] / tot[ok])
        delta = (lp[:, 0] - lp[:, 1]) - (lp[:, 2] - lp[:, 3])
        out.append({"group": [tokens[a].decode("utf-8", "replace"), tokens[b].decode("utf-8", "replace"),
                              tokens[c].decode("utf-8", "replace"), tokens[d].decode("utf-8", "replace")],
                    "n_contexts": int(ok.sum()), "delta_std_nats": float(delta.std()),
                    "delta_range_nats": [float(delta.min()), float(delta.max())]})
    stds = [o["delta_std_nats"] for o in out]
    return {"n_groups_measured": len(out), "median_delta_std_nats": float(np.median(stds)) if stds else None,
            "mean_delta_std_nats": float(np.mean(stds)) if stds else None, "min_count": min_count,
            "worst": sorted(out, key=lambda o: -o["delta_std_nats"])[:10]}


def automaton_census(tokens, V):
    """Trie nodes/edges and minimal-DAFSA states/edges for the vocabulary (suffix-sharing)."""
    trie = {}
    n_nodes = 1
    for v in range(V):
        node = trie
        for byte in tokens[v]:
            if byte not in node:
                node[byte] = {}
                n_nodes += 1
            node = node[byte]
        node["$"] = True
    # minimal DAFSA: merge nodes with identical right languages (bottom-up signature hashing)
    sig_id = {}

    def minimize(node):
        items = []
        for k in sorted(k for k in node if k != "$"):
            items.append((k, minimize(node[k])))
        sig = (("$" in node), tuple(items))
        if sig not in sig_id:
            sig_id[sig] = len(sig_id)
        return sig_id[sig]
    minimize(trie)
    n_states = len(sig_id)
    n_edges = sum(len(sig[1]) for sig in sig_id)
    return {"trie_nodes": n_nodes, "trie_edges": n_nodes - 1, "dafsa_states": n_states, "dafsa_edges": n_edges,
            "V": V, "note": "a per-node (trie) or per-state (DAFSA) parameter vector costs that many rows; "
                            "KAS uses 256*32 cells regardless"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", type=int, default=512)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--configs", nargs="*", default=None)
    ap.add_argument("--merge", action="store_true", help="reuse floor_static.json, compute only missing configs")
    args = ap.parse_args()
    torch.set_num_threads(8)
    t0 = time.time()
    vocab, table, unigram = load_vocab()
    V = vocab.n_train_vocab
    tokens = vocab.tokens[:V]
    res = {"provenance": "RECOMPUTED, CPU only, from data/tokens (training split) and the fixed codec; "
                         "no model checkpoint used", "V": V, "pos_dim": vocab.pos_dim}
    if args.merge and OUT.exists():
        prev = json.loads(OUT.read_text(encoding="utf-8"))
        if prev.get("bigram", {}).get("n_contexts") == args.contexts:
            res.update({k: prev[k] for k in ("unigram_floor", "unigram_additive_residual", "bigram") if k in prev})
            print("merging into existing results:", sorted(res.get("bigram", {}).get("floors", {})), flush=True)

    print("building feature families ...", flush=True)
    fams = build_families(table, tokens, unigram, V)
    res["family_sizes"] = {k: v[1] for k, v in fams.items()}
    fs_kas = make_fs(V, fams, ["kas"])
    res["rank_kas_features"] = fs_kas.rank()
    print("  KAS feature rank", res["rank_kas_features"], flush=True)

    # ---------------- unigram (context-free) floor
    p_uni = torch.as_tensor(unigram / unigram.sum(), dtype=torch.float32).unsqueeze(0)
    one = torch.ones(1)
    res.setdefault("unigram_floor", {})
    configs = args.configs or list(CONFIGS)
    for name in ["kas", "kas+suffix", "kas+bigram_free", "kas+trigram_free", "kas+private1000", "kas+hash8192"]:
        if name not in configs or name in res["unigram_floor"]:
            continue
        fs = make_fs(V, fams, CONFIGS[name])
        r = fit_floor(fs, p_uni, one, None, steps=args.steps, lr=args.lr, lbfgs=True)
        res["unigram_floor"][name] = {k: r[k] for k in ("floor_nats", "floor_bpb_equiv", "H_nats", "n_features")}
        print(f"  unigram floor [{name}]: {r['floor_nats']:.4f} nats ({r['floor_bpb_equiv']:.4f} bpb-equiv)", flush=True)
        OUT.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
        if name == "kas":
            r_lb = fit_floor(fs, p_uni, one, None, steps=1500, lr=0.05)
            res["unigram_floor"]["kas_adam1500_check"] = {k: r_lb[k] for k in ("floor_nats", "floor_bpb_equiv", "seconds")}
            print(f"  unigram floor [kas, Adam 1500 check]: {r_lb['floor_nats']:.4f} nats", flush=True)
            with torch.no_grad():
                idx, w = fs.tensors()
                z = torch.nn.functional.embedding_bag(idx, r["S"], per_sample_weights=w, mode="sum")[:, 0]
                fitted = torch.log_softmax(z, 0).numpy()
            lp = np.log(unigram / unigram.sum() + 1e-12)
            m = unigram > 0
            resid = lp[m] - fitted[m]
            wts = unigram[m] / unigram[m].sum()
            res["unigram_additive_residual"] = {
                "note": "log p_unigram minus best additive-in-bytes fit; token-weighted and count-weighted",
                "std_nats_token_weighted": float(resid.std()),
                "std_nats_count_weighted": float(np.sqrt(np.average((resid - np.average(resid, weights=wts)) ** 2, weights=wts))),
                "pct_1_99_nats": [float(np.percentile(resid, 1)), float(np.percentile(resid, 99))],
                "r2_token_weighted": float(1 - resid.var() / lp[m].var())}

    # ---------------- bigram conditionals
    C = args.contexts
    order = np.argsort(-unigram)
    contexts = order[:C].astype(np.int64)
    train = np.load(ROOT / "data/tokens/train.npy", mmap_mode="r")
    print(f"counting bigrams for the {C} most frequent previous tokens ...", flush=True)
    counts = bigram_counts(train, contexts, V)
    tot = counts.sum(1)
    coverage = float(tot.sum() / (len(train) - 1))
    P = torch.as_tensor(counts / tot[:, None], dtype=torch.float32)
    ctx_w = torch.as_tensor(tot / tot.sum(), dtype=torch.float32)
    if not (args.merge and res.get("bigram")):
        res["bigram"] = {"n_contexts": C, "position_coverage": coverage, "min_context_count": int(tot.min()),
                         "H_nats_weighted": None, "floors": {}}
    b_uni = unigram_bias(unigram, V)
    for name in configs:
        if name in res["bigram"]["floors"]:
            continue
        fs = make_fs(V, fams, CONFIGS[name])
        row = {}
        for bias_name, bias, free in (("no_bias", None, False), ("unigram_bias", b_uni, False), ("free_bias", b_uni, True)):
            if name != "kas" and bias_name != "unigram_bias":
                continue
            r = fit_floor(fs, P, ctx_w, bias, free_bias=free, steps=args.steps, lr=args.lr, lbfgs=True)
            row[bias_name] = {k: r[k] for k in ("floor_nats", "floor_bpb_equiv", "ce_nats", "H_nats", "n_features", "seconds")}
            res["bigram"]["H_nats_weighted"] = r["H_nats"]
            print(f"  bigram floor [{name} / {bias_name}]: {r['floor_nats']:.4f} nats "
                  f"({r['floor_bpb_equiv']:.4f} bpb-equiv), H={r['H_nats']:.4f}, {r['seconds']}s", flush=True)
            if name == "kas" and bias_name == "unigram_bias":
                # per-context floors vs context frequency, for the report
                fl = r["floor_per_ctx"]
                res["bigram"]["kas_unigram_bias_per_context"] = {
                    "contexts_top16": [tokens[int(c)].decode("utf-8", "replace") for c in contexts[:16]],
                    "floor_top16": [float(x) for x in fl[:16]],
                    "floor_quartiles": [float(np.percentile(fl, q)) for q in (10, 25, 50, 75, 90)]}
        res["bigram"]["floors"][name] = row
        OUT.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")

    # convergence check: LBFGS with twice the iterations, and Adam, on the base configuration
    fs = make_fs(V, fams, ["kas"])
    r_long = fit_floor(fs, P, ctx_w, b_uni, steps=args.steps * 2, lbfgs=True, seed=1)
    r_adam = fit_floor(fs, P, ctx_w, b_uni, steps=600, lr=0.05, seed=1)
    res["bigram"]["convergence_check"] = {"kas_unigram_bias_lbfgs_x2_floor_nats": r_long["floor_nats"],
                                          "kas_unigram_bias_adam600_floor_nats": r_adam["floor_nats"],
                                          "kas_unigram_bias_steps_x1_floor_nats": res["bigram"]["floors"]["kas"]["unigram_bias"]["floor_nats"]}
    print(f"  convergence: x1 {res['bigram']['convergence_check']['kas_unigram_bias_steps_x1_floor_nats']:.5f} "
          f"vs x2 {r_long['floor_nats']:.5f}", flush=True)

    # ---------------- parallelograms, automaton
    census, scored = parallelogram_census(tokens, unigram, V)
    res["parallelograms"] = census
    res["parallelogram_defect_bigram"] = parallelogram_defect(scored, counts, contexts, unigram, tokens)
    res["automaton"] = automaton_census(tokens, V)
    res["seconds_total"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print("wrote", OUT, f"({res['seconds_total']}s)")


if __name__ == "__main__":
    main()
