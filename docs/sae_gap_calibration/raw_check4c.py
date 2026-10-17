import json, time, sys
from pathlib import Path
import numpy as np, scipy.sparse as sp
from safetensors.numpy import load_file
from tokenizers import Tokenizer
from sklearn.decomposition import FastICA
EXT = Path("/Users/scasella/Downloads/gf-discovery-ext"); t0=time.time()
K=int(sys.argv[1])
rows = np.load(EXT/"results/occ_units_v2_L12.npz")["row"].astype(np.int64)
meta = np.load(EXT/"cache/occ/occD2_meta.npz"); full = meta["tokens"].astype(np.int64)
tok, seq, pos = full[rows], meta["seq"][rows], meta["pos"][rows]
H = np.asarray(np.load(EXT/"results/e67_raw_L12_rows.npy", mmap_mode="r"), dtype=np.float32)
sae = load_file(str(next((EXT/"hf_home/hub/models--google--gemma-scope-2-1b-pt/snapshots").glob("*"))/"resid_post_all/layer_12_width_16k_l0_big/params.safetensors"))
We,be,th,Wd,bd = (sae[k].astype(np.float32) for k in ("w_enc","b_enc","threshold","w_dec","b_dec"))
keep = np.ones(1152,bool); keep[[1038,1125]] = False
D = np.empty((len(H), keep.sum()), np.float32); L0=np.empty(len(H),np.float32)
for lo in range(0,len(H),8192):
    pre = H[lo:lo+8192]@We+be; z = pre*(pre>th); L0[lo:lo+8192]=(z>0).sum(1); D[lo:lo+8192]=(z@Wd+bd)[:,keep]
R = np.ascontiguousarray(H[:,keep]); del H
def demean(V,g,block=128):
    n,G=len(g),int(g.max())+1; M=sp.csr_matrix((np.ones(n),(g,np.arange(n))),shape=(G,n)); cnt=np.maximum(np.asarray(M.sum(1)).ravel(),1)
    for j in range(0,V.shape[1],block):
        b=V[:,j:j+block].astype(np.float64); V[:,j:j+block]=(b-(M@b/cnt[:,None])[g]).astype(np.float32)
groups=[np.unique(a,return_inverse=True)[1] for a in (tok,seq,pos)]
for _ in range(3):
    for g in groups: demean(R,g); demean(D,g)
en=np.sqrt(((R-D)**2).sum(1))
s=R[::3].astype(np.float64); C=s.T@s/len(s); del s; w,U=np.linalg.eigh(C); o=np.argsort(w)[::-1][:K]; w,U=w[o],U[:,o]
P = U/np.sqrt(w)                      # raw -> whitened
P32=P.astype(np.float32); Wr = R@P32; del R; Wdd = D@P32; del D
H2 = np.asarray(np.load(EXT/"results/e67_raw_L12_rows.npy", mmap_mode="r")[::3], dtype=np.float32)[:, keep]
nv = (H2.astype(np.float64)**2).mean(0) * (2.0**-14) / 12      # bf16 rounding variance per coord (rel. spacing 2^-7)
del H2
floor = (U**2).T @ nv / w
print("bf16 noise/eigenvalue at PC", {k: float(floor[k-1]) for k in (64,256,512,768,1024,K) if k<=K}, flush=True)
Gr=np.random.default_rng(1).standard_normal((K,1000)).astype(np.float32); Yr_=Wr[::3]@Gr; Yd_=Wdd[::3]@Gr
esr=np.array([1-np.corrcoef(Yr_[:,j],Yd_[:,j])[0,1]**2 for j in range(1000)]); del Yr_,Yd_
print("random ES p50/p99/p99.9", np.percentile(esr,[50,99,99.9]), flush=True)
print("rep", time.time()-t0, flush=True)
ica = FastICA(n_components=K, whiten="unit-variance", random_state=0, max_iter=200, tol=1e-4).fit(Wr[::3])
print("ica", time.time()-t0, "n_iter", ica.n_iter_, flush=True)
Y = ica.transform(Wr); Yd = ica.transform(Wdd) - ica.mean_ @ ica.components_.T * 0  # transform subtracts mean_ (constant)
ES = np.array([1-np.corrcoef(Y[:,j],Yd[:,j])[0,1]**2 for j in range(K)])
tk = Tokenizer.from_file(str(next((EXT/"hf_home/hub/models--google--gemma-3-1b-pt/snapshots").glob("*"))/"tokenizer.json"))
strs={}; s_of=lambda t: strs.setdefault(t, tk.decode([int(t)]))
row_set={int(r):i for i,r in enumerate(rows)}; par=np.zeros(len(rows),bool); quo=np.zeros(len(rows),bool)
for s0 in range(0,len(full),255):
    depth=0;q=0
    for j,t in enumerate(full[s0:s0+255]):
        st=s_of(int(t)); i=row_set.get(s0+j)
        if i is not None: par[i]=depth>0; quo[i]=q%2==1
        depth=max(0,depth+st.count("(")-st.count(")")); q+=st.count('"')+st.count("“")+st.count("”")
from scipy.stats import rankdata
def auc(score,y):
    r=rankdata(score); n1=y.sum(); n0=len(y)-n1; a=(r[y].sum()-n1*(n1+1)/2)/(n1*n0); return max(a,1-a)
ap=np.array([auc(Y[:,j],par) for j in range(K)]); aq=np.array([auc(Y[:,j],quo) for j in range(K)])
cnt=np.bincount(np.unique(tok,return_inverse=True)[1])
tinv=np.unique(tok,return_inverse=True)[1]
def eta2(v):
    mt=np.bincount(tinv,weights=v)/cnt; return float(((mt[tinv]-v.mean())**2).sum()/((v-v.mean())**2).sum())
e2=np.array([eta2(Y[:,j]**2) for j in range(K)])
rL0=np.array([abs(np.corrcoef(Y[:,j],L0)[0,1]) for j in range(K)]); rL0a=np.array([abs(np.corrcoef(np.abs(Y[:,j]),L0)[0,1]) for j in range(K)])
ren=np.array([abs(np.corrcoef(np.abs(Y[:,j]),en)[0,1]) for j in range(K)])
kurt=np.array([((Y[:,j]-Y[:,j].mean())**4).mean()/Y[:,j].var()**2-3 for j in range(K)])
thr=max(0.5,float(np.percentile(esr,99.9)))
out=dict(K=K, rand_es_p999=float(np.percentile(esr,99.9)), n_ES_ge_thr=int((ES>=thr).sum()), n_ES_ge_thr_and_eta=int(((ES>=thr)&(e2<=0.3)&(rL0a<=0.3)&(ren<=0.3)).sum()), n_iter=int(ica.n_iter_), ES_quantiles={q:float(np.percentile(ES,q)) for q in (10,50,90,99)}, n_ES_ge_0p5=int((ES>=0.5).sum()),
 paren_best=dict(auroc=float(ap.max()), ES=float(ES[ap.argmax()])), quote_best=dict(auroc=float(aq.max()), ES=float(ES[aq.argmax()])),
 eta2sq_max=float(e2.max()), n_eta2sq_gt_0p3=int((e2>0.3).sum()),
 among_ES_ge_0p5=dict(n=int((ES>=0.5).sum()), n_absy_L0_r_gt_0p3=int(((ES>=0.5)&(rL0a>0.3)).sum()), n_absy_errnorm_r_gt_0p3=int(((ES>=0.5)&(ren>0.3)).sum()),
   median_exkurt=float(np.median(kurt[ES>=0.5])) if (ES>=0.5).any() else None, max_paren=float(ap[ES>=0.5].max()) if (ES>=0.5).any() else None, max_quote=float(aq[ES>=0.5].max()) if (ES>=0.5).any() else None),
 all_median_exkurt=float(np.median(kurt)), seconds=round(time.time()-t0))
Path(f"raw_check4b_K{K}.json").write_text(json.dumps(out,indent=1)); print(json.dumps(out,indent=1))
