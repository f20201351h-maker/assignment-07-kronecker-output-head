"""Tiny independent CPU float64 check, four preselected original dev contexts.

Uses COO matrix multiplication and analytic moment gradients, not the original
embedding_bag/autograd fitter. No model training. No cloud access.
"""
import sys,json,time,hashlib
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
EV=ROOT/'results/expressivity'
sys.path.insert(0,str(ROOT/'scripts/expressivity'))
from floor_lib import load_vocab,FeatureSet,fam_kas,fam_ngrams,unigram_bias
from floor_contextual import load_dense_prior
torch.set_num_threads(4)
torch.manual_seed(0)
vocab,table,counts=load_vocab()
old=np.load(EV/'gpu/floor_gpu_s1_dev.npz')
meta=json.loads((EV/'gpu/floor_gpu_s1_dev.json').read_text())
# Fixed before inspecting their results: spread over the original window/position grid.
picks=[(0,17),(10,43),(20,79),(31,111)]
pos=np.linspace(0,1023,128).round().astype(int)
stream=np.load(ROOT/'data/tokens/dev.npy',mmap_mode='r')
model,cfg=load_dense_prior(vocab,np.load(ROOT/'data/tokens/unigram_train.npy'))
res={'scope':'Four-context numerical spot check only; not representative or a full-matrix certificate',
     'picks':picks,'torch':torch.__version__,'config':vars(cfg),'fits':{}}
Z=[]
with torch.no_grad():
    E=model.input_table_rows(None)
    for iw,ip in picks:
        start=meta['windows'][iw]*1024
        x=torch.tensor(np.array(stream[start:start+1024]),dtype=torch.long)[None,:]
        h=model.body(x,E)[0,pos[ip]].float()
        Z.append(h@model.head.W.float().T+model.head.beta.float())
    Z=torch.stack(Z).double()
    logp=Z.log_softmax(1); P=logp.exp()
    beta=model.head.beta.detach().double()
    res['probability_sum_error']=float((P.sum(1)-1).abs().max())
    print('reference ready',flush=True)
del model,E

def solve(name,families,bias,tag):
    fs=FeatureSet(table.V)
    for nm,args in families:fs.add_family(nm,*args)
    rr=[]; cc=[]; vv=[]
    for v,features in enumerate(fs.feats):
        for f,w in features:rr.append(v);cc.append(f);vv.append(w)
    M=torch.sparse_coo_tensor(torch.tensor([rr,cc]),torch.tensor(vv,dtype=torch.float64),(table.V,fs.n_features)).coalesce()
    Mt=M.T.coalesce()
    S=torch.zeros(fs.n_features,4,dtype=torch.float64,requires_grad=True)
    target=P.T.contiguous()
    opt=torch.optim.LBFGS([S],lr=1.,max_iter=200,history_size=50,line_search_fn='strong_wolfe',tolerance_grad=1e-11,tolerance_change=1e-14)
    run=[];t=time.time()
    def closure():
        with torch.no_grad():
            lq=(torch.sparse.mm(M,S)+bias[:,None]).log_softmax(0)
            S.grad=torch.sparse.mm(Mt,lq.exp()-target)/4
            return -(target*lq).sum()/4
    for block in range(4):
        opt.step(closure)
        with torch.no_grad():
            lq=(torch.sparse.mm(M,S)+bias[:,None]).log_softmax(0)
            kl=(target*(logp.T-lq)).sum(0)
            moments=torch.sparse.mm(Mt,lq.exp()-target)
        run.append({'iterations':int(opt.state[S]['n_iter']),'KL':kl.tolist(),'max_abs_moment_residual':float(moments.abs().max())})
        print(name,run[-1],round(time.time()-t,1),flush=True)
    ids=[iw*128+ip for iw,ip in picks]
    i=list(old['tags']).index(tag)
    res['fits'][name]={'original_float32_200_KL':old[f'floor_{i}'][ids].tolist(),'float64_stages':run,'seconds':time.time()-t}
    (EV/'fit_checks.json').write_text(json.dumps(res,indent=2))

base=[('kas',fam_kas(table))]
solve('kas_fixed_unigram',base,unigram_bias(counts,table.V).double(),'kas | bias = log-unigram prior (fixed; KAS-P family)')
solve('kas_trigram_fixed_teacher_beta',base+[('trigram',fam_ngrams(table,vocab.tokens[:table.V],3,False,8192))],beta,'kas+trigram_free | bias = beta (fixed)')
