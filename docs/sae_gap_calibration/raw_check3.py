import json, time
from pathlib import Path
import numpy as np, scipy.sparse as sp
from safetensors.numpy import load_file
from tokenizers import Tokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
EXT = Path("/Users/scasella/Downloads/gf-discovery-ext"); t0=time.time()
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
R = np.ascontiguousarray(H[:,keep]); hn=np.linalg.norm(R,axis=1); del H
def demean(V,g,block=128):
    n,G=len(g),int(g.max())+1; M=sp.csr_matrix((np.ones(n),(g,np.arange(n))),shape=(G,n)); cnt=np.maximum(np.asarray(M.sum(1)).ravel(),1)
    for j in range(0,V.shape[1],block):
        b=V[:,j:j+block].astype(np.float64); V[:,j:j+block]=(b-(M@b/cnt[:,None])[g]).astype(np.float32)
groups=[np.unique(a,return_inverse=True)[1] for a in (tok,seq,pos)]
for _ in range(3):
    for g in groups: demean(R,g); demean(D,g)
E = R - D
en = np.linalg.norm(E,axis=1)
tk = Tokenizer.from_file(str(next((EXT/"hf_home/hub/models--google--gemma-3-1b-pt/snapshots").glob("*"))/"tokenizer.json"))
strs={}; s_of=lambda t: strs.setdefault(t, tk.decode([int(t)]))
row_set={int(r):i for i,r in enumerate(rows)}
names=("paren","quote","line_pos","sent_pos","after_newline","next_newline","word_continuation","next_punct","prev_punct","in_caps_run","after_digit","since_punct")
F={k:np.zeros(len(rows)) for k in names}
for s0 in range(0,len(full),255):
    toks=full[s0:s0+255]; depth=0;q=0;lp=0;sp_=0;prev="";caps=0;sincep=0
    for j,t in enumerate(toks):
        st=s_of(int(t)); i=row_set.get(s0+j)
        if i is not None:
            nx=s_of(int(toks[j+1])) if j+1<len(toks) else ""
            ps=prev.strip()
            F["paren"][i]=depth>0; F["quote"][i]=q%2==1; F["line_pos"][i]=lp; F["sent_pos"][i]=sp_
            F["after_newline"][i]="\n" in prev; F["next_newline"][i]="\n" in nx
            F["word_continuation"][i]=bool(st) and st[0].isalpha() and j>0 and not prev.endswith((" ","\n"))
            F["next_punct"][i]=bool(nx.strip()) and not any(c.isalnum() for c in nx.strip())
            F["prev_punct"][i]=bool(ps) and not any(c.isalnum() for c in ps)
            F["in_caps_run"][i]=caps>=2; F["after_digit"][i]=ps[-1:].isdigit(); F["since_punct"][i]=sincep
        depth=max(0,depth+st.count("(")-st.count(")")); q+=st.count('"')+st.count("“")+st.count("”")
        lp=0 if "\n" in st else lp+1
        sp_=0 if st.strip()[-1:] in (".","!","?") else sp_+1
        sincep=0 if (st.strip() and not any(c.isalnum() for c in st.strip())) else sincep+1
        al=[c for c in st if c.isalpha()]; caps=caps+1 if (al and all(c.isupper() for c in al)) else 0
        prev=st
F["L0"]=L0.astype(float); F["hnorm"]=hn.astype(float)
ev = seq%2==0; tr = ev & (np.arange(len(rows))%2==0)
mu=R[tr].mean(0); sd=R[tr].std(0)+1e-6
out={}
for k,v in F.items():
    y = v>np.median(v) if k in ("line_pos","sent_pos","since_punct","L0","hnorm") else v>0.5
    rec=dict(base=float(y.mean()))
    for name,V in (("raw",R),("dec",D),("err",E)):
        m=V[tr].mean(0); s=V[tr].std(0)+1e-6
        clf=LogisticRegression(C=0.1,max_iter=1000).fit((V[tr]-m)/s, y[tr])
        rec[name]=round(float(roc_auc_score(y[~ev], clf.decision_function((V[~ev]-m)/s))),3)
        if name=="raw":
            f=clf.coef_.ravel()/s
            yr=R[~ev]@f; yd=D[~ev]@f
            rec["ES_corr_rawprobe"]=round(float(1-np.corrcoef(yr,yd)[0,1]**2),3)
    out[k]=rec; print(k,rec,round(time.time()-t0),flush=True)
out["r_errnorm_L0"]=float(np.corrcoef(en,L0)[0,1]); out["r_errnorm_hnorm_kept"]=float(np.corrcoef(en,hn)[0,1])
Path("raw_check3_out.json").write_text(json.dumps(out,indent=1))
