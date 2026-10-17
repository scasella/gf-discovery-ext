import json, time
from pathlib import Path
import numpy as np, scipy.sparse as sp, scipy.linalg as sl
from safetensors.numpy import load_file
EXT = Path("/Users/scasella/Downloads/gf-discovery-ext"); t0=time.time()
rows = np.load(EXT/"results/occ_units_v2_L12.npz")["row"].astype(np.int64)
meta = np.load(EXT/"cache/occ/occD2_meta.npz"); full = meta["tokens"].astype(np.int64)
tok, seq, pos = full[rows], meta["seq"][rows], meta["pos"][rows]
H = np.asarray(np.load(EXT/"results/e67_raw_L12_rows.npy", mmap_mode="r"), dtype=np.float32)
sae = load_file(str(next((EXT/"hf_home/hub/models--google--gemma-scope-2-1b-pt/snapshots").glob("*"))/"resid_post_all/layer_12_width_16k_l0_big/params.safetensors"))
We,be,th,Wd,bd = (sae[k].astype(np.float32) for k in ("w_enc","b_enc","threshold","w_dec","b_dec"))
keep = np.ones(1152,bool); keep[[1038,1125]] = False     # variant: drop 1125 too
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
E = R - D
s = slice(None, None, 3)
C = R[s].astype(np.float64).T@R[s].astype(np.float64)/len(R[s])
CEE = E[s].astype(np.float64).T@E[s].astype(np.float64)/len(R[s])
CDD = D[s].astype(np.float64).T@D[s].astype(np.float64)/len(R[s])
CDR = D[s].astype(np.float64).T@R[s].astype(np.float64)/len(R[s])
w,U = np.linalg.eigh(C); o=np.argsort(w)[::-1]; w,U=w[o],U[:,o]
out={}
cv=np.diag(C); out["idrem_coordvar_top3_share_after_drop"]=sorted(((float(cv[i]/cv.sum()),int(np.flatnonzero(keep)[i])) for i in range(len(cv))),reverse=True)[:3]
out["raw_pca_varfrac_cum"]={k: float(w[:k].sum()/w.sum()) for k in (64,128,256,512)}
def es_corr(V):
    num=np.einsum("ij,ik,kj->j",V,CDR,V); vr=np.einsum("ij,ik,kj->j",V,C,V); vd=np.einsum("ij,ik,kj->j",V,CDD,V)
    return 1-num**2/(vr*vd)
for k in (64,128,256,512,len(w)):
    Uk=U[:,:k]; A=Uk.T@CEE@Uk; B=np.diag(w[:k])
    gw,gv=sl.eigh(A,B); o=np.argsort(gw)[::-1]; gw,gv=gw[o],gv[:,o]
    V=Uk@gv[:,:50]; ec=es_corr(V)
    # random directions in whitened k-subspace
    G=np.random.default_rng(1).standard_normal((k,500)); Vr=Uk@(G/np.sqrt(w[:k])[:,None]); er=es_corr(Vr)
    out[f"k{k}"]=dict(max_var_share=float(gw[0]), n_var_share_ge_0p5=int((gw>=0.5).sum()), n_ge_0p4=int((gw>=0.4).sum()),
                     top5_var_share=[round(float(x),3) for x in gw[:5]], es_corr_of_top5=[round(float(x),3) for x in ec[:5]],
                     n_escorr_ge_0p5_among_top50=int((ec>=0.5).sum()),
                     random_whitened_escorr=dict(p50=float(np.median(er)),p99=float(np.percentile(er,99)),p999=float(np.percentile(er,99.9))))
    print(k, out[f"k{k}"], time.time()-t0, flush=True)
en=np.linalg.norm(E,axis=1)
out["r_errnorm_L0_idrem"]=float(np.corrcoef(en,L0)[0,1])
Path("raw_check2_out.json").write_text(json.dumps(out,indent=1)); print(json.dumps(out,indent=1))
