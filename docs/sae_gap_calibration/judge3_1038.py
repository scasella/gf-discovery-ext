import numpy as np, glob
from safetensors.numpy import load_file
p=glob.glob('/Users/scasella/Downloads/gf-discovery-ext/hf_home/hub/models--google--gemma-scope-2-1b-pt/snapshots/*/resid_post_all/layer_12_width_16k_l0_big/params.safetensors')[0]
w=load_file(p); print({k:v.shape for k,v in w.items()})
X=np.load('/Users/scasella/Downloads/gf-discovery-ext/results/e67_raw_L12_rows.npy',mmap_mode='r')
print(X.shape,X.dtype)
rng=np.random.default_rng(0); idx=np.sort(rng.choice(X.shape[0],4000,replace=False))
h=np.asarray(X[idx],dtype=np.float32)
We=w['w_enc'] if 'w_enc' in w else w['W_enc']; be=w.get('b_enc'); th=w.get('threshold'); Wd=w['w_dec'] if 'w_dec' in w else w['W_dec']; bd=w['b_dec']
def enc(h):
    pre=h@We+be; return pre*(pre>th)
z=enc(h); h0=h.copy(); h0[:,1038]=0; z0=enc(h0)
print('L0 full',(z>0).sum(1).mean(),'L0 1038-zeroed',(z0>0).sum(1).mean())
act=(z>0); act0=(z0>0); print('jaccard active sets', (act&act0).sum()/ (act|act0).sum())
rec=z@Wd+bd; rec0=z0@Wd+bd
print('FVU full', ((h-rec)**2).sum()/((h-h.mean(0))**2).sum())
m=np.ones(1152,bool); m[1038]=False
print('FVU excl1038 full-encode', ((h-rec)[:,m]**2).sum()/((h[:,m]-h[:,m].mean(0))**2).sum(), 'zeroed-encode', ((h-rec0)[:,m]**2).sum()/((h[:,m]-h[:,m].mean(0))**2).sum())
