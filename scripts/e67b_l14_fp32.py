"""e67b -- supplement to e67: the L14 activation correlations, recomputed with the residual kept in float32.

e67's acts stage stored raw residuals as float16; coordinate 1038 overflows float16 at L14, so every L14
latent correlation came out NaN (reported as 0, "void"). This recomputes, for the two L14 dictionaries
only, exactly what e67's acts stage computes -- Pearson r of every latent's activation with the unit's
frozen D2 score (Y) and with the raw projection h.d over D2's 203,328 kept rows, and the candidate
latents' behavioural match and contexts -- with the L14 residual held in float32 throughout. e67's own
hashed outputs are not modified; this writes results/e67b_acts_L14_fp32.json.
"""
import hashlib, importlib.util, json, os, sys, time
from pathlib import Path

import numpy as np

EXT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXT / "src"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
OUT = "results/e67b_acts_L14_fp32.json"
LAYER = 14


def main():
    print("e67b sha256", hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), flush=True)
    spec = importlib.util.spec_from_file_location("e67", EXT / "scripts/e67_prior_feature_check.py")
    e67 = importlib.util.module_from_spec(spec); sys.modules["e67"] = e67; spec.loader.exec_module(e67)
    e66, e65, e64 = e67.mods()
    e64.check_swap()
    assert not (EXT / OUT).exists(), "e67b runs once"
    import torch
    dirs, keep, _ = e67.directions(e64)
    fnpz = np.load(EXT / "results/occ_units_v2_L12.npz")
    rows = fnpz["row"].astype(np.int64)
    Y = {f"{k}/{j}": fnpz[f"Y_{k}"][:, j].astype(np.float64) for k, j in e67.UNITS}
    toks = np.load(EXT / "cache/occ/occD2_meta.npz")["tokens"].astype(np.int64)
    import gfdisc.model as gm
    gm.MODEL_NAME = str(sorted((EXT / "hf_home/hub/models--google--gemma-3-1b-pt/snapshots").glob("*"))[0])
    from gfdisc.env import DEVICE
    from gfdisc.model import capture
    model, tok = gm.load_model(token=None)
    T = toks.reshape(-1, 255)
    H = np.zeros((len(toks), 1152), np.float32)
    t0 = time.time()
    with torch.inference_mode():
        for lo in range(0, len(T), 16):
            blk = T[lo:lo + 16]
            ids = np.concatenate([np.full((len(blk), 1), tok.bos_token_id), blk], 1)
            with capture(model, [LAYER]) as st:
                model(torch.from_numpy(ids).to(DEVICE), use_cache=False)
            H[lo * 255:(lo + len(blk)) * 255] = st[LAYER][:, 1:, :].float().reshape(-1, 1152).cpu().numpy()
    Hl = H[rows]; del H
    assert np.isfinite(Hl).all(), "non-finite L14 residual"
    print(f"[{time.time() - t0:.0f}s] L14 residual captured (float32); max |x| {float(np.abs(Hl).max()):.0f} at coord "
          f"{int(np.abs(Hl).max(0).argmax())}", flush=True)
    # targets, as e67: Y (unit score) and h.d -- h.d uses the L12 residual in e67; recompute it here from
    # e67's saved L12 kept rows so the targets are identical
    H12 = np.load(EXT / e67.OUT["raw"]).astype(np.float32)
    proj = {n: (H12 @ v.astype(np.float32)).astype(np.float64) for n, v in dirs.items()}
    del H12
    targets = {**{f"Y:{n}": Y[n] for n in Y}, **{f"hd:{n}": proj[n] for n in proj}}
    tnames = list(targets)
    Tm = np.stack([targets[n] for n in tnames], 1)
    Tz = (Tm - Tm.mean(0)) / Tm.std(0)
    valid_next = (rows % 255) != 254
    cats_next = np.full(len(rows), "", dtype=object); cache = {}
    for i in np.flatnonzero(valid_next):
        t = int(toks[rows[i] + 1]); cats_next[i] = cache.setdefault(t, e66.category(tok.decode([t])))
    is_match = np.isin(cats_next, e67.MATCH_CATS)
    dec_rows = json.loads((EXT / e67.OUT["decoder"]).read_text())
    dec = lambda ids: tok.decode([int(t) for t in ids])
    out = dict(layer=LAYER, rows=int(len(rows)), dtype="float32", base_match_share=float(is_match[valid_next].mean()),
               max_abs_residual=float(np.abs(Hl).max()), dicts={})
    for d in [d for d in e67.DICTS if d[1] == LAYER]:
        t1 = time.time()
        P = e67.sae_params(d)
        W = torch.tensor(P["w_enc"], device=DEVICE); b = torch.tensor(P["b_enc"], device=DEVICE)
        th = torch.tensor(P["threshold"], device=DEVICE)
        n, nl = len(rows), W.shape[1]
        s1 = torch.zeros(nl, dtype=torch.float64); s2 = torch.zeros(nl, dtype=torch.float64)
        sy = torch.zeros((nl, Tz.shape[1]), dtype=torch.float64); nz = torch.zeros(nl, dtype=torch.float64)
        Tz_t = torch.tensor(Tz, dtype=torch.float32)
        with torch.no_grad():
            for lo in range(0, n, 4096):
                x = torch.tensor(Hl[lo:lo + 4096], device=DEVICE)
                pre = x @ W + b; a = pre * (pre > th)
                assert bool(torch.isfinite(a).all()), "non-finite activation"
                s1 += a.sum(0).cpu().double(); s2 += (a * a).sum(0).cpu().double(); nz += (a > 0).sum(0).cpu().double()
                sy += (a.T @ Tz_t[lo:lo + 4096].to(DEVICE)).cpu().double()
        mean = s1 / n; var = s2 / n - mean ** 2
        r = ((sy / n) / torch.sqrt(torch.clamp(var, min=1e-12))[:, None]).numpy()
        n_nonfinite = int((~np.isfinite(r)).sum()); r[~np.isfinite(r)] = 0.0
        rec = dict(n_latents=int(nl), mean_l0=float(nz.sum() / n), n_nonfinite_r=n_nonfinite, corr={})
        cand = set()
        for ti, tn in enumerate(tnames):
            o = np.argsort(-np.abs(r[:, ti]))[:e67.TOP]
            rec["corr"][tn] = [dict(latent=int(i), r=float(r[i, ti]), freq=float(nz[i] / n)) for i in o]
            if tn.startswith("Y:"):
                cand.update(int(i) for i in o[:5])
        for name in dirs:
            cand.update(int(t["latent"]) for t in dec_rows["dicts"][e67.dname(d)][name]["top"][:5])
            for mode in ("signed", "nonneg_pos", "nonneg_neg"):
                cand.update(int(a) for a in dec_rows["dicts"][e67.dname(d)]["recon"][name][mode]["first5"])
        ci = sorted(cand); Wc, bc, thc = W[:, ci], b[ci], th[ci]
        acts = np.zeros((n, len(ci)), np.float32)
        with torch.no_grad():
            for lo in range(0, n, 8192):
                x = torch.tensor(Hl[lo:lo + 8192], device=DEVICE)
                pre = x @ Wc + bc; acts[lo:lo + 8192] = (pre * (pre > thc)).cpu().numpy()
        rec["latents"] = {}
        for j, li in enumerate(ci):
            a = acts[:, j]; order = np.argsort(-a)
            top = [i for i in order if valid_next[i] and a[i] > 0][:e67.MATCH_TOP]
            share = float(is_match[top].mean()) if top else None
            ctx = []
            for i in order[:12]:
                if a[i] <= 0:
                    break
                rr = int(rows[i]); s0 = (rr // 255) * 255
                ctx.append(dec(toks[max(s0, rr - 15):rr]) + "[[" + dec([toks[rr]]) + "]]" + dec(toks[rr + 1:min(s0 + 255, rr + 4)]))
            rec["latents"][str(li)] = dict(n_active=int((a > 0).sum()), match_share=share, n_top=len(top),
                                           behavioural_match=bool(share is not None and share >= e67.MATCH_SHARE and len(top) >= 50),
                                           contexts=ctx)
        # the rubric's two COVERED routes, for this dictionary only
        hits = []
        for name in dirs:
            for t in dec_rows["dicts"][e67.dname(d)][name]["top"]:
                if abs(t["cos"]) >= e67.COS_COVER and rec["latents"].get(str(t["latent"]), {}).get("behavioural_match"):
                    hits.append(dict(unit=name, latent=t["latent"], via="decoder_cos", value=t["cos"]))
            for t in rec["corr"][f"Y:{name}"]:
                if abs(t["r"]) >= e67.R_COVER and rec["latents"].get(str(t["latent"]), {}).get("behavioural_match"):
                    hits.append(dict(unit=name, latent=t["latent"], via="activation_r", value=t["r"]))
        rec["covered_hits"] = hits
        rec["seconds"] = round(time.time() - t1)
        out["dicts"][e67.dname(d)] = rec
        del W, b, th, Wc
        if DEVICE == "mps":
            torch.mps.empty_cache()
        best = {tn: (rec["corr"][tn][0]["latent"], round(rec["corr"][tn][0]["r"], 3)) for tn in tnames}
        print(f"[{time.time() - t0:.0f}s] {e67.dname(d)} L0 {rec['mean_l0']:.0f} non-finite r {n_nonfinite} best r {best} "
              f"covered hits {hits}", flush=True)
    out["swap_gb_at_end"] = e64.swap_used_gb()
    e64.dump(out, OUT)
    e64.append_hash(OUT, "e67b supplement: L14 activation correlations recomputed in float32 (e67's L14 cells were void)")


if __name__ == "__main__":
    main()
