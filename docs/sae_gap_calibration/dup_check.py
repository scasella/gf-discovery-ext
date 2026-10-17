import numpy as np
EXT="/Users/scasella/Downloads/gf-discovery-ext/results/"
a=np.fromfile(EXT+"corpus_tokens_pass1.i32",dtype=np.int32).reshape(-1,255)
r=np.fromfile(EXT+"reserve_tokens.i32",dtype=np.int32).reshape(-1,256)[:,1:]
key=lambda x: x[:32].tobytes()
used={}
for name,(lo,hi,src) in dict(D2=(1000,2000,a),P2=(13000,14000,a),v1=(14000,15000,a),c0=(0,1000,a),res=(0,8000,r)).items():
    for i in range(lo,hi): used.setdefault(key(src[i]),set()).add(name)
for name,(lo,hi) in dict(C3=(12000,13000),D3=(2000,3000),R3=(3000,4000),B5=(4000,5000),B6=(5000,6000),B7=(6000,7000)).items():
    ks=[key(a[i]) for i in range(lo,hi)]
    print(name,"dup vs used:",sum(k in used for k in ks),"within-range dups:",len(ks)-len(set(ks)))
allk=[key(a[i]) for i in list(range(2000,7000))+list(range(12000,13000))]
print("dups across proposed sets:",len(allk)-len(set(allk)))
