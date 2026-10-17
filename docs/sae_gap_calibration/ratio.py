import json, numpy as np, collections
EXT='/Users/scasella/Downloads/gf-discovery-ext/'
P=json.load(open(EXT+'results/e64c_d2_pieces.json'))
flat=np.array([p for s in P for p in s], dtype=object)
N=len(flat); allr=np.arange(N); allr=allr[(allr%255)!=254]
# type-conditional next distribution over all non-final D2 rows (pieces as types)
cur=flat[allr]; nx=flat[allr+1]
cnt=collections.defaultdict(collections.Counter)
for c,n in zip(cur,nx): cnt[c][n]+=1
tot={c:sum(v.values()) for c,v in cnt.items()}
fz=json.load(open(EXT+'results/e65_frozen.json')); ver=json.load(open(EXT+'results/e64_verdict.json'))
for L in (7,12):
    z=np.load(EXT+f'results/occ_units_v2_L{L}.npz'); rows=z['row'].astype(np.int64)
    m=np.load(EXT+f'results/e64_maps_L{L}.npz'); valid=(rows%255)!=254
    U={(u['kind'],u['id']):u for u in ver['layers'][str(L)]['units']}
    res=[]
    for key,s in fz['layers'][str(L)]['signatures'].items():
        k,j=key.split('/'); j=int(j); u=U.get((k,j))
        if not (u and u['reading']=='none'): continue
        T=set(s['T_text'])
        y=z[f'Y_{k}'][:,j].astype(np.float64)*m[f'sign_{k}'][j]
        o=np.argsort(-y); top=[rows[i] for i in o if valid[i]][:200]
        sh=np.mean([flat[r+1] in T for r in top])
        ex=np.mean([sum(cnt[flat[r]][t] for t in T)/tot[flat[r]] for r in top])
        res.append((key,u['tier'],sh,ex,sh/max(ex,1e-9)))
    a=np.array([r[4] for r in res]); b=np.array([r[2] for r in res]); e=np.array([r[3] for r in res])
    print(f'L{L} cand n={len(res)} own200 share med {np.median(b):.3f}; type-cond expected med {np.median(e):.3f}; ratio med {np.median(a):.2f} min {a.min():.2f}; ratio<3 n={int((a<3).sum())}; ratio<2 n={int((a<2).sum())}')
