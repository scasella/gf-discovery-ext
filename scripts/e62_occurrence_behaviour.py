"""e62 -- Stage 4 behavioural tier for occurrence units (PREREG_occurrence.md section 9.2).

For a frozen unit u at layer L (cluster or ICA component from e61):
  direction d_u   cluster: its raw identity-removed centroid; ICA: its mixing column mapped back
                  to raw space (U @ (m * sd)), signed toward the tail its members were read from;
                  re-embedded into 1152 dims with 0.0 at the dropped coordinates, unit-normalised.
  signature T_u   the 20 next tokens most over-represented after u's D members relative to
                  token-matched expectation (each member contributes its own type's next-token
                  distribution over all D rows), count >= 5. Derived blind (no label read).
  random set R_u  20 tokens frequency-matched (corpus count within 10%) to T_u, fixed seed.
  occurrences     N_OCC random (sequence, position) sites from reserve sequences 1000-1999,
                  position >= 4; prefix = BOS + up to 64 preceding tokens + the site token.
  intervention    h <- h + eps * ||h|| * d_u at resid_post L, at the site (last position).
  readout         Delta = [lse(logits[T_u]) - lse(logits[R_u])]_steered - the same unsteered.
Nulls at the SAME occurrences: (i) 10 random directions orthogonal to d_u (own signature);
(iv) up to 30 sibling units (other screened units of the same kind, fixed-seed sample), each
with ITS OWN direction and signature. PASS: sign consistency (over the 20 commonest site token
types, per-type mean Delta) >= 0.8 at some eps AND mean Delta@0.02 > sibling p99 AND > max(i).
The positive-control unit runs first; if it does not PASS no verdict is issued on any other.
"""
import argparse, json, os, sys, time
from pathlib import Path

import numpy as np
import torch

EXT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXT / "src"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import gfdisc.model as gm                                          # noqa: E402
gm.MODEL_NAME = str(sorted((EXT / "hf_home/hub/models--google--gemma-3-1b-pt/snapshots").glob("*"))[0])
from gfdisc.env import DEVICE                                      # noqa: E402
from gfdisc.model import load_model                                # noqa: E402

EPS_GRID = (0.005, 0.01, 0.02)
EPS_NULL = 0.02
N_OCC = 200
N_RAND = 10
N_SIB = 30
CTX = 64
BATCH = 32


def drop_for(L):
    if L <= 10:
        return [1038] if L >= 3 else [1125]
    p = EXT / f"results/lag_deltas_L{L}.npz"
    if p.exists():
        return [int(x) for x in np.load(p)["dropped_coords"]]
    # declared fallback (PREREG_occurrence.md section 4; same as e46): Addendum-H rule on the
    # split-half raw type means of pass 1
    tm = np.load(EXT / "results/typemeans_pass1.npz")
    A, B = tm[f"raw_A_L{L}"].astype(np.float64), tm[f"raw_B_L{L}"].astype(np.float64)
    sd = np.sqrt(((A - B) ** 2).mean(0)); sh = sd ** 2 / (sd ** 2).sum()
    return [int(x) for x in np.flatnonzero(sh > 0.25)]


def embed(v, drop):
    keep = np.ones(1152, bool); keep[drop] = False
    out = np.zeros(1152); out[keep] = v
    return out / max(np.linalg.norm(out), 1e-12)


def unit_members(kind, uid, z):
    if kind == "cluster":
        return np.flatnonzero(z["lab"] == uid), 1.0
    y = z["ica_Y"][:, uid]
    q99, q01 = np.quantile(y, 0.99), np.quantile(y, 0.01)
    if abs(q99) >= abs(q01):
        return np.flatnonzero(y >= q99), 1.0
    return np.flatnonzero(y <= q01), -1.0


def unit_direction(kind, uid, z, sign, drop):
    if kind == "cluster":
        return embed(z["centroids_raw"][uid], drop)
    m = z["mix"][:, uid] * z["pca_sd"]
    return embed(sign * (z["pca_U"] @ m), drop)


def signature(rows_D, mem_idx, tokD, nxt_by_type, vocab_n):
    r = rows_D[mem_idx]
    ok = (r % 255) != 254
    r = r[ok]
    nxt = tokD[r + 1]
    obs = np.bincount(nxt, minlength=vocab_n).astype(float)
    exp = np.zeros(vocab_n)
    types, cnt = np.unique(tokD[r], return_counts=True)
    for t, c in zip(types, cnt):
        d = nxt_by_type.get(int(t))
        if d is not None:
            exp[d[0]] += c * d[1]
    score = np.log((obs + 1) / (exp + 1))
    score[obs < 5] = -np.inf
    top = np.argsort(-score)[:20]
    return [int(x) for x in top if np.isfinite(score[x])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--anchor", required=True, help="kind:id of the positive-control unit")
    ap.add_argument("--units", default="", help="comma list kind:id of candidate units")
    ap.add_argument("--D", default="occD")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-occ", type=int, default=N_OCC)
    a = ap.parse_args()
    L = a.layer
    t0 = time.time()
    z = np.load(EXT / f"results/occ_units_L{L}.npz")
    frozen = json.loads((EXT / "results/e61_occurrence.json").read_text())["layers"][str(L)]["frozen"]
    drop = drop_for(L)
    metaD = np.load(EXT / f"cache/occ/{a.D}_meta.npz")
    tokD = metaD["tokens"].astype(np.int64)
    rows_D = z["row"].astype(np.int64)                         # kept D rows, aligned with lab / ica_Y
    vocab_n = 262_144 + 16
    # next-token distribution per type over all D rows
    nxt_by_type = {}
    r_all = np.arange(len(tokD)); r_all = r_all[(r_all % 255) != 254]
    order = np.argsort(tokD[r_all], kind="stable")
    tt = tokD[r_all][order]; nn = tokD[r_all + 1][order]
    for b in np.split(np.arange(len(tt)), np.flatnonzero(np.diff(tt)) + 1):
        u_, c_ = np.unique(nn[b], return_counts=True)
        nxt_by_type[int(tt[b[0]])] = (u_, c_ / c_.sum())
    corpus = np.fromfile(EXT / "results/corpus_tokens_pass1.i32", dtype=np.int32)
    ccount = np.bincount(corpus, minlength=vocab_n)

    def spec(sv):
        k, i = sv.split(":")
        return k, int(i)

    rng = np.random.default_rng(6200 + L)

    def build(kind, uid):
        mem, sign = unit_members(kind, uid, z)
        d = unit_direction(kind, uid, z, sign, drop)
        T = signature(rows_D, mem, tokD, nxt_by_type, vocab_n)
        Rset = []
        for t in T:
            c = ccount[t]
            cand = np.flatnonzero((ccount >= 0.9 * c) & (ccount <= 1.1 * c) & (ccount > 0))
            cand = cand[~np.isin(cand, T)]
            Rset.append(int(rng.choice(cand)) if len(cand) else int(rng.integers(vocab_n)))
        return dict(kind=kind, id=uid, dir=d, T=T, R=Rset, n_members=int(len(mem)))

    model, tok = load_model(token=None)
    res = dict(script="scripts/e62_occurrence_behaviour.py", layer=L, rules=__doc__,
               started=time.strftime("%Y-%m-%dT%H:%M:%S"))
    # occurrence sites (reserve 1000-1999), fixed
    rv = np.fromfile(EXT / "results/reserve_tokens.i32", dtype=np.int32).reshape(-1, 256)[1000:2000]
    srng = np.random.default_rng(6262 + L)
    sites = [(int(srng.integers(len(rv))), int(srng.integers(5, 256))) for _ in range(a.n_occ)]
    prompts = [np.concatenate([[tok.bos_token_id], rv[s, max(1, p - CTX):p + 1]]) for s, p in sites]
    site_tok = np.asarray([int(rv[s, p]) for s, p in sites])

    def forward(directions, eps):
        """logits at the last position for every prompt; directions: (n_prompts, 1152) or None."""
        outs = []
        for lo in range(0, len(prompts), BATCH):
            blk = prompts[lo:lo + BATCH]
            Lm = max(len(x) for x in blk)
            ids = torch.full((len(blk), Lm), tok.pad_token_id, dtype=torch.long)
            att = torch.zeros_like(ids)
            for i, x in enumerate(blk):
                ids[i, :len(x)] = torch.from_numpy(x.astype(np.int64)); att[i, :len(x)] = 1
            last = torch.tensor([len(x) - 1 for x in blk])
            hook = None
            if directions is not None:
                D = torch.tensor(directions[lo:lo + len(blk)], dtype=torch.float32, device=DEVICE)

                def fn(_m, _i, out, D=D, last=last):
                    t = out[0] if isinstance(out, tuple) else out
                    new = t.float()
                    ib = torch.arange(len(last), device=DEVICE); lp = last.to(DEVICE)
                    h = new[ib, lp]
                    new[ib, lp] = h + eps * h.norm(dim=-1, keepdim=True) * D
                    new = new.to(t.dtype)
                    return (new,) + out[1:] if isinstance(out, tuple) else new
                hook = model.model.layers[L].register_forward_hook(fn)
            try:
                with torch.no_grad():
                    lg = model(input_ids=ids.to(DEVICE), attention_mask=att.to(DEVICE),
                               use_cache=False).logits
                outs.append(lg[torch.arange(len(blk)), last.to(DEVICE)].float().cpu())
            finally:
                if hook is not None:
                    hook.remove()
        return torch.cat(outs, 0)

    base = forward(None, 0.0)

    def delta(lg, U):
        T = torch.tensor(U["T"]); R = torch.tensor(U["R"])
        f = lambda x: torch.logsumexp(x[:, T], -1) - torch.logsumexp(x[:, R], -1)
        return (f(lg) - f(base)).numpy()

    def sign_consistency(dl):
        types, cnt = np.unique(site_tok, return_counts=True)
        top = types[np.argsort(-cnt)[:20]]
        means = [dl[site_tok == t].mean() for t in top]
        return float(np.mean(np.asarray(means) > 0))

    def evaluate(U, sibs):
        rec = dict(kind=U["kind"], id=U["id"], n_members=U["n_members"],
                   signature=[tok.decode([t]) for t in U["T"]], random_set=[tok.decode([t]) for t in U["R"]])
        dirs = np.tile(U["dir"], (len(prompts), 1))
        for e in EPS_GRID:
            dl = delta(forward(dirs, e), U)
            rec[f"eps={e}"] = dict(mean_delta=float(dl.mean()), sign_consistency=sign_consistency(dl))
        rr = []
        for j in range(N_RAND):
            g = rng.standard_normal(1152); g -= (g @ U["dir"]) * U["dir"]; g /= np.linalg.norm(g)
            rr.append(float(delta(forward(np.tile(g, (len(prompts), 1)), EPS_NULL), U).mean()))
        sv = []
        for S in sibs:
            sv.append(float(delta(forward(np.tile(S["dir"], (len(prompts), 1)), EPS_NULL), S).mean()))
        d = rec[f"eps={EPS_NULL}"]["mean_delta"]
        rec.update(null_random_max=max(rr), null_random=rr, sibling_means=sv,
                   sibling_p99=float(np.percentile(sv, 99, method="higher")) if sv else None,
                   sc_ok=any(rec[f"eps={e}"]["sign_consistency"] >= 0.8 for e in EPS_GRID))
        rec["PASS"] = bool(rec["sc_ok"] and d > rec["null_random_max"]
                           and rec["sibling_p99"] is not None and d > rec["sibling_p99"])
        return rec

    kind_a, id_a = spec(a.anchor)
    screened = [(u["kind"], u["id"]) for u in frozen["units"] if u["screened"]]
    cands = [spec(x) for x in a.units.split(",") if x]
    out_units = []
    for kind, uid in [(kind_a, id_a)] + cands:
        pool_s = [x for x in screened if x[0] == kind and x != (kind, uid)]
        pick = [pool_s[i] for i in np.random.default_rng(6300 + L + uid).permutation(len(pool_s))[:N_SIB]]
        U = build(kind, uid)
        sibs = [build(k2, i2) for k2, i2 in pick]
        sibs = [s_ for s_ in sibs if s_["T"]]
        rec = evaluate(U, sibs)
        rec["role"] = "anchor" if (kind, uid) == (kind_a, id_a) else "candidate"
        out_units.append(rec)
        res["units"] = out_units
        Path(a.out).write_text(json.dumps(res, indent=1, default=float))
        print(f"[{time.time()-t0:.0f}s] {rec['role']} {kind}:{uid} PASS={rec['PASS']} "
              f"d@0.02={rec[f'eps={EPS_NULL}']['mean_delta']:+.4f} sib_p99={rec['sibling_p99']} "
              f"rand_max={rec['null_random_max']:+.4f} sig={rec['signature'][:8]}", flush=True)
        if rec["role"] == "anchor" and not rec["PASS"]:
            res["verdict"] = "ANCHOR FAILED -- no behavioural verdict issued on any unit"
            break
    Path(a.out).write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
