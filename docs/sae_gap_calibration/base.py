import json, numpy as np, sys
EXT='/Users/scasella/Downloads/gf-discovery-ext/'
P=json.load(open(EXT+'results/e64c_d2_pieces.json'))
flat=np.array([p for s in P for p in s], dtype=object)
fz=json.load(open(EXT+'results/e65_frozen.json'))
ver=json.load(open(EXT+'results/e64_verdict.json'))
out={}
for L in (7,12):
    z=np.load(EXT+f'results/occ_units_v2_L{L}.npz'); rows=z['row'].astype(np.int64)
    m=np.load(EXT+f'results/e64_maps_L{L}.npz')
    if L==7: print('row range', rows.min(), rows.max(), len(flat))
    valid=(rows%255)!=254
    nxt=np.empty(len(rows),dtype=object); nxt[valid]=flat[rows[valid]+1]
    U={(u['kind'],u['id']):u for u in ver['layers'][str(L)]['units']}
    for key,s in fz['layers'][str(L)]['signatures'].items():
        k,j=key.split('/'); j=int(j)
        T=set(s['T_text'])
        inT=np.array([x in T for x in nxt]) & valid
        base=inT[valid].mean()
        y=z[f'Y_{k}'][:,j].astype(np.float64)*m[f'sign_{k}'][j]
        o=np.argsort(-y)
        top200=[i for i in o if valid[i]][:200]
        k1=int(np.ceil(0.01*len(y))); top1=[i for i in o[:k1] if valid[i]]
        # bottom tail too
        bot200=[i for i in o[::-1] if valid[i]][:200]
        u=U.get((k,j))
        status = 'cand' if (u and u['reading']=='none') else ('ctx' if u else 'other')
        out[(L,key)]=dict(base=base, thr=max(3*base,0.10), own200=inT[top200].mean(), own1pct=inT[top1].mean(), bot200=inT[bot200].mean(), status=status, tier=(u['tier'] if u else None))
import collections
for L in (7,12):
    for st in ('cand','ctx','other'):
        R=[v for (l,k),v in out.items() if l==L and v['status']==st]
        if not R: continue
        b=np.array([r['base'] for r in R]); o2=np.array([r['own200'] for r in R]); o1=np.array([r['own1pct'] for r in R]); th=np.array([r['thr'] for r in R])
        print(f'L{L} {st} n={len(R)} base med {np.median(b):.3f} [{b.min():.3f},{b.max():.3f}] thr>0.10 n={int((3*b>0.10).sum())} own200 med {np.median(o2):.3f} min {o2.min():.3f}; own200<thr n={int((o2<th).sum())}; own1%<thr n={int((o1<th).sum())}')
for (L,k),v in sorted(out.items()):
    if v['status']!='other' and (v['own200']<v['thr'] or v['own1pct']<v['thr']):
        print(L,k,v['status'],v['tier'],{a:round(float(b),3) for a,b in v.items() if a not in('status','tier')})
print('---calib')
for key in ('logcosh/13','skew/16','logcosh/33','skew/33','logcosh/11','skew/23'):
    v=out.get((12,key)); print(12,key,{a:(round(float(b),3) if isinstance(b,(float,np.floating)) else b) for a,b in v.items()})
for key in ('logcosh/12','skew/62'):
    v=out.get((7,key)); print(7,key,{a:(round(float(b),3) if isinstance(b,(float,np.floating)) else b) for a,b in v.items()})
# duplicates in T_text
for L in (7,12):
    d=0
    for key,s in fz['layers'][str(L)]['signatures'].items():
        if len(set(s['T_text']))<len(s['T_text']): d+=1
    print('L',L,'signatures with duplicate decoded strings', d)
# replicated candidates failing own match
for L in (7,12):
    R=[(k,v) for (l,k),v in out.items() if l==L and v['status']=='cand']
    rep=[(k,v) for k,v in R if v['tier']=='replicated geometric']
    print(L,'replicated cand units',len(rep),'own200<thr',sum(v['own200']<v['thr'] for k,v in rep))
