"""Independent read-only checks of the saved teacher-projection fits (array audit, exact codec-to-feature
mapping, rank, same-length census, head accounting, non-EOS re-aggregation).
Writes results/expressivity/evidence_audit.json.

Run: python scripts/expressivity/audit_evidence.py
No network, Modal, training, or modifications to historical evidence.
"""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
EV = ROOT / 'results/expressivity'
sys.path.insert(0, str(ROOT / 'scripts/expressivity'))
from floor_lib import load_vocab, fam_kas, FeatureSet, fam_ngrams
from floor_contextual import ARMS
from floor_static import parallelogram_census
from kq5.codec import TorchCodec

torch.set_num_threads(4)
out = {'scope': 'Independent CPU array, construction, and arithmetic audit; not a rerun of floor matrix', 'inputs': {}}
def sha(p):
    p = Path(p)
    key = str(p.relative_to(ROOT))
    if key not in out['inputs']:
        with p.open('rb') as f:
            out['inputs'][key] = hashlib.file_digest(f, 'sha256').hexdigest()

vocab, table, counts = load_vocab()
V = table.V
lengths = np.array([len(t) for t in vocab.tokens])
for p in [ROOT/'data/tokens/vocab.json', ROOT/'data/tokens/unigram_train.npy', ROOT/'src/kq5/codec.py', ROOT/'src/kq5/heads.py', ROOT/'scripts/expressivity/floor_lib.py', ROOT/'scripts/expressivity/floor_contextual.py']:
    sha(p)
out['matrix'] = {}
for p in sorted((EV/'gpu').glob('*.npz')):
    sha(p); sha(p.with_suffix('.json'))
    a = np.load(p)
    j = json.loads(p.with_suffix('.json').read_text())
    split = 'test' if '_test' in p.name else 'dev'
    seed = 2 if '_s2_' in p.name else 1
    stream = np.load(ROOT/f'data/tokens/{split}.npy', mmap_mode='r')
    pos = np.linspace(0, 1023, 128).round().astype(int)
    wi = np.array(j['windows'])
    yy = np.asarray(stream[wi[:,None]*1024+pos[None,:]+1]).reshape(-1)
    assert np.array_equal(yy, a['y'])
    assert np.array_equal(lengths[yy], a['nbytes'])
    valid = a['valid']; assert np.array_equal(valid, lengths[yy]>0)
    arm_errors = {}
    for arm, rel in ARMS.items():
        if arm == 'Dense-prior':
            rel = f'experiments/modal-p4/runs/p4-densep-s{seed}/nll_dev.npy'
        rel = rel.replace('nll_dev.npy',f'nll_{split}.npy')
        pp = ROOT/rel; sha(pp)
        raw = np.load(pp, mmap_mode='r')[wi][:,pos].reshape(-1).astype(np.float64)
        arm_errors[arm] = float(np.max(np.abs(raw-a[f'arm_{arm}'])))
        assert arm_errors[arm] == 0
    rows = {}
    canonical_valid = yy != vocab.eos_id
    for i,tag in enumerate(a['tags']):
        fl = a[f'floor_{i}']; mi = a[f'mimic_{i}']; r = j['floors'][str(tag)]
        assert abs(fl[valid].mean()-r['floor_nats_per_target'])<1e-10
        assert abs(mi[valid].mean()-r['mimic_nll_true_targets'])<1e-10
        gap = a['arm_KAS-P']-mi
        # Descriptive window-cluster SE; deterministic grid is not a probability sample.
        means = gap.reshape(32,128).mean(1)
        rows[str(tag)] = {'floor':float(fl[valid].mean()), 'mimic_gap':float((mi-a['ref_nll'])[valid].mean()),
                         'kasp_minus_mimic':float(gap[valid].mean()),
                         'window_cluster_se_kasp_minus_mimic':float(means.std(ddof=1)/np.sqrt(32)),
                         'minimum_context_KL':float(fl.min()),
                         'canonical_non_eos_floor':float(fl[canonical_valid].mean()),
                         'canonical_non_eos_mimic_gap':float((mi-a['ref_nll'])[canonical_valid].mean()),
                         'canonical_non_eos_kasp_minus_mimic':float(gap[canonical_valid].mean())}
    out['matrix'][p.stem] = {'valid':int(valid.sum()), 'same_position_arrays_max_errors':arm_errors,
        'kasp_gap':float((a['arm_KAS-P']-a['arm_Dense-prior'])[valid].mean()),
        'ref_saved_vs_recomputed_mean':float((a['ref_nll']-a['arm_Dense-prior'])[valid].mean()),
        'eos_targets_in_saved_valid':int((yy==vocab.eos_id).sum()),
        'canonical_non_eos_kasp_gap':float((a['arm_KAS-P']-a['arm_Dense-prior'])[canonical_valid].mean()),
        'canonical_non_eos_bytes':int(lengths[yy][canonical_valid].sum()), 'fits':rows}

# Compare sparse features against exact codec logits over all 57,243 tokens.
fs = FeatureSet(V); fs.add_family('kas',*fam_kas(table))
idx,w = fs.tensors()
occ = np.unique(table.flat_index[table.valid]); nc=len(occ)
ar = table.A[table.lengths>0].mean(); br=np.abs(table.B[table.lengths>0]).mean()
torch.manual_seed(173)
score = torch.randn(table.D,3,dtype=torch.float64)
lb = torch.randn(table.pos_dim+1,3,dtype=torch.float64)
s = torch.zeros(fs.n_features,3,dtype=torch.float64)
s[1:1+nc] = ar*score[occ]
s[1+nc] = br*score.sum(0)
s[2+nc:] = lb
direct = TorchCodec(table).matmul(score)+lb[table.lengths]
sparse = F.embedding_bag(idx,s,per_sample_weights=w.double(),mode='sum')
# Reverse representation: arbitrary feature logits are exact-codec logits plus length offsets.
t = torch.randn_like(s); rev=torch.zeros_like(score); rev[occ]=t[1:1+nc]/ar
rev_lb=t[2+nc:].clone()
AA,BB=__import__('kq5.codec',fromlist=['codec_coefficients']).codec_coefficients(np.arange(33))
rev_lb += torch.tensor(BB[:,None])*(t[1+nc]/br-rev.sum(0))[None,:]
reverse_error=(F.embedding_bag(idx,t,per_sample_weights=w.double(),mode='sum')-
               (TorchCodec(table).matmul(rev)+rev_lb[table.lengths])).abs().max().item()
out['codec_equivalence']={'V':V,'D':table.D,'occupied_cells':nc,'features':fs.n_features-1,
    'forward_max_abs_error_float32_feature_weights':float((direct-sparse).abs().max()),
    'reverse_max_abs_error_float32_feature_weights':reverse_error,
    'length_zero_tokens':int((table.lengths==0).sum()),'unused_codec_cells':int(table.D-nc)}
assert (direct-sparse).abs().max()<3e-5
assert reverse_error<3e-5

# Independent all-vocabulary Gram rank with chunked memory, at the original tolerance.
gram=torch.zeros(fs.n_features-1,fs.n_features-1,dtype=torch.float64)
for start in range(0,V,2048):
    m=torch.zeros(min(2048,V-start),fs.n_features,dtype=torch.float64)
    m.scatter_add_(1,idx[start:start+2048],w[start:start+2048].double())
    m=m[:,1:]; gram+=m.T@m
ev=torch.linalg.eigvalsh(gram)
tol=ev.max()*len(ev)*1e-15
out['codec_equivalence']['numerical_rank_at_original_tolerance']=int((ev>tol).sum())
census,_=parallelogram_census(vocab.tokens,counts,V)
old=json.loads((EV/'floor_static.json').read_text())['parallelograms']
assert json.loads(json.dumps(census))==old
out['parallelogram_census']=census
tri,n= fam_ngrams(table,vocab.tokens[:V],3,False,8192)
used=len({i for row in tri for i,_ in row})
out['trigram']={'allocated_buckets':n,'occupied_buckets':used,'occurrences':sum(len(r) for r in tri)}
d=512
params={'KAS-P':8192*d+33,'KAS+8192_trigrams':(8192+8192)*d+33,
        'KAS-U16':8192*d+33+V*16+16*d+V,'KAS-U64':8192*d+33+V*64+64*d+V,
        'Dense-prior':V*d+V}
out['output_head_accounting']={k:{'trainable':v,'fp32_adam_moments_bytes':v*8,
     'fp32_weight_grad_adam_bytes':v*16} for k,v in params.items()}
dest=EV/'evidence_audit.json'
dest.write_text(json.dumps(out,indent=2))
print(json.dumps({'codec':out['codec_equivalence'],'trigram':out['trigram'],'params':params,'checked_matrix_runs':len(out['matrix']),'output':str(dest)},indent=2))
