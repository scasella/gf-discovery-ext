import numpy as np
from scipy.optimize import nnls
def omp(A, d, k_max=48, nonneg=False):
    d = d / np.linalg.norm(d); r = d.copy(); sel, curve, coef = [], [], None
    for _ in range(k_max):
        c = A @ r.astype(np.float32)
        if nonneg:
            c[sel] = -np.inf; j = int(np.argmax(c))
            if c[j] <= 0:
                break
        else:
            c = np.abs(c); c[sel] = -1; j = int(np.argmax(c))
        sel.append(j)
        B = A[sel].astype(np.float64).T
        coef = nnls(B, d)[0] if nonneg else np.linalg.lstsq(B, d, rcond=None)[0]
        dh = B @ coef; r = d - dh
        curve.append(float(dh @ d / max(np.linalg.norm(dh), 1e-12)))
    return curve, sel, coef
def k_at(curve, t):
    return next((i + 1 for i, c in enumerate(curve) if c >= t), None)
rng = np.random.default_rng(0)
for dim, na in ((64, 4000), (96, 8000)):
    A = rng.standard_normal((na, dim)).astype(np.float32); A /= np.linalg.norm(A, axis=1, keepdims=True)
    ks, kn = [], []
    for i in range(100):
        d = rng.standard_normal(dim)
        ks.append(k_at(omp(A, d)[0], .8) or 49); kn.append(k_at(omp(A, d, nonneg=True)[0], .8) or 49)
    ks, kn = np.array(ks), np.array(kn)
    print(dim, na, 'signed p1/p50', np.quantile(ks, .01), np.median(ks), 'nonneg p1/p50', np.quantile(kn, .01), np.median(kn), 'frac nonneg>signed', (kn > ks).mean(), 'nonneg<signed', (kn < ks).mean())
# k_at None-or behaviour, and quantile on censored
print(np.quantile([49]*99 + [30], .01), np.quantile([49]*98 + [30, 31], .01), np.quantile([49]*30, .01))
