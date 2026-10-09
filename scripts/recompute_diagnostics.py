"""CPU-only independent recomputation of the prior-matched comparison (protocol 08) from the saved
per-position test NLL arrays, with a frequency-bucket decomposition of KAS-U64 minus Dense-prior.
Writes results/phase4/independent_diagnostics.json."""
import hashlib
import json
import math
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results' / 'phase4'
RUNS = {
    'Dense': [('modal-m3', 'm3-dense-s1'), ('modal-m3', 'm3-dense-s2'),
              ('modal-p3', 'p3-dense-s3'), ('modal-p3', 'p3-dense-s4')],
    'Dense-prior': [('modal-p4', f'p4-densep-s{s}') for s in range(1, 5)],
    'KAS-U64': [('modal-p2', 'p2-kasu64-s1'), ('modal-p2', 'p2-kasu64-s2'),
                ('modal-p3', 'p3-kasu64-s3'), ('modal-p3', 'p3-kasu64-s4')],
}


def git_head():
    try:
        return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], cwd=ROOT, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main():
    vocab = json.loads((ROOT / 'data/tokens/vocab.json').read_text())
    lengths = np.array([len(bytes.fromhex(t)) for t in vocab['tokens_hex']])
    lengths[vocab['eos_id']] = 0
    stream = np.load(ROOT / 'data/tokens/test.npy', mmap_mode='r')
    T = 1024
    starts = np.arange((len(stream) - 1) // T) * T
    targets = stream[starts[:, None] + np.arange(1, T + 1)]
    nb = lengths[targets]
    valid = nb > 0
    counts = np.load(ROOT / 'data/tokens/unigram_train.npy')
    freq = counts[targets]
    denom = float(nb.sum()) * math.log(2)
    raw, arms, hashes = {}, {}, {}
    for arm, specs in RUNS.items():
        arrays, bpbs = [], []
        for folder, rid in specs:
            p = ROOT / 'experiments' / folder / 'runs' / rid / 'nll_test.npy'
            a = np.load(p).astype(np.float64)
            assert a.shape == nb.shape
            summary = json.loads((p.parent / 'summary.json').read_text())
            bpb = float(a[valid].sum() / denom)
            assert abs(bpb - summary['final']['test']['bpb']) < 1e-10
            hashes[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
            arrays.append(a)
            bpbs.append(bpb)
        raw[arm] = np.stack(arrays)
        arms[arm] = {'values': bpbs, 'mean': float(np.mean(bpbs))}
    diff = raw['KAS-U64'] - raw['Dense-prior']
    bins = [0, 1, 10, 100, 1000, 10000, 100000, 1000000, np.inf]
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = valid & (freq >= lo) & (freq < hi)
        n = int(m.sum())
        rows.append({'range': f'[{lo},{hi})', 'n_targets': n,
                     'token_fraction': n / int(valid.sum()),
                     'u64_minus_prior_nats': float(diff[:, m].mean()) if n else None,
                     'contribution_to_global_bpb': float(diff[:, m].sum(axis=1).mean() / denom),
                     'per_seed_nats': diff[:, m].mean(axis=1).tolist() if n else []})
    tail = valid & (freq < 1000)
    out = {'provenance': 'RECOMPUTED; exploratory decomposition of previously inspected test data',
           'base_commit': git_head(),
           'arms': arms, 'n_positive_byte_targets': int(valid.sum()), 'bytes': int(nb.sum()),
           'U64_minus_Dense_prior': float(diff[:, valid].sum(axis=1).mean() / denom),
           'tail_below_1000': {'targets': int(tail.sum()), 'fraction': float(tail.sum()/valid.sum()),
                              'contribution_bpb': float(diff[:, tail].sum(axis=1).mean()/denom)},
           'frequency_buckets': rows, 'input_nll_sha256': hashes}
    (OUT / 'independent_diagnostics.json').write_text(json.dumps(out, indent=2), encoding='utf-8')

    print(json.dumps({k: v for k, v in out.items() if k not in ['input_nll_sha256', 'frequency_buckets']}, indent=2))
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
