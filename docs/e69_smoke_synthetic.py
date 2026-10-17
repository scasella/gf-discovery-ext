"""Synthetic smoke test of e69's pilot functions (fake tokens, fake SAE). No real pilot number."""
import importlib.util, sys, time, json
import numpy as np
EXT = "/Users/scasella/Downloads/gf-discovery-ext"
spec = importlib.util.spec_from_file_location("e69", f"{EXT}/scripts/e69_sae_gap_pilot.py")
m = importlib.util.module_from_spec(spec); sys.modules["e69"] = m; spec.loader.exec_module(m)
m.T0 = time.time()
rng = np.random.default_rng(1)
nseq = 60; n = nseq * 255
toks = rng.integers(0, 400, size=n); meta = dict(tokens=toks, seq=np.repeat(np.arange(nseq), 255), pos=np.tile(np.arange(255), nseq))
S = m.RowSet(meta); print("rows", len(S.rows), "pairs", {L: len(S.pairs[L][0]) for L in m.LAGS})
# plant a slow AR(1) source along one direction in a fake residual
H = rng.standard_normal((len(S.rows), 1152)).astype(np.float32)
slow = np.zeros(len(S.rows)); 
for i in range(1, len(S.rows)):
    slow[i] = 0.95 * slow[i - 1] * (S.seq[i] == S.seq[i - 1]) + rng.standard_normal()
u = rng.standard_normal(1152); u /= np.linalg.norm(u); H += (3 * slow[:, None] * u[None, :]).astype(np.float32)
nl = 300
P = dict(w_enc=(rng.standard_normal((1152, nl)) / 30).astype(np.float32), b_enc=np.zeros(nl, np.float32), threshold=np.full(nl, 2.0, np.float32),
         w_dec=(rng.standard_normal((nl, 1152)) / 30).astype(np.float32), b_dec=np.zeros(1152, np.float32))
m.sae_params = lambda d: P
E, st = m.error_rows(H.astype(np.float16), P)
m.K_PCA = 64
F = m.fit_instrument(S, E, 16)
units = m.screens(S, F, st, m=16)
print("eig top3", np.round(F["eig"][:3], 3), "unit0", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in units[0].items() if k != "rho_state"})
pat0 = F["pat"][:, 0]; uk = u[m.KEEP] / np.linalg.norm(u[m.KEEP]); print("planted |cos| with unit0 pattern", round(float(abs(pat0 @ uk)), 3))
y = F["Y"][:, 0]; print("acf1", round(S.acf(y, 1), 3), "acov1", round(S.acov(y, 1), 3))
x = rng.standard_normal(n).astype(np.float32); x[254::255] = np.nan
cm = m.causal_mean(x, S.rows, 4, True); print("causal_mean finite", bool(np.isfinite(cm).all()))
X = m.causal_mean(rng.standard_normal((n, 8)), S.rows, 16, False)
print("ridge R2", round(m.heldout_r2(X, y.astype(np.float64), S.even, "ridge"), 3), "mlp R2", round(m.heldout_r2(X, y.astype(np.float64), S.even, "mlp"), 3))
m.N_SUBLAT = 16
sub = m.subthreshold(H.astype(np.float16), {"unit/0": y.astype(np.float64), "rande/0": rng.standard_normal(len(y))}, S.even, ("fake",))
print("subthreshold", sub)
Dn = m.RowSet(meta, S.rows[(S.rows % 255) != 254]); print("Dn rows", len(Dn.rows), "demean3 1-D ok", Dn.demean3(rng.standard_normal(len(Dn.rows)).astype(np.float32)).shape)
json.dumps(units); print("json ok")
