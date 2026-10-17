"""e01 -- one corpus pass: 06's sparse CSR cache for the chosen layers, plus split-half
per-token-type accumulators, plus a held-out token reserve.

Stage 0/1 of the gf-discovery-ext campaign (BUDGET.md).

THE CACHE PART IS 06, NOT A REWRITE. The encode -> CSR write path is the exact sequence of
calls in scripts/06_cache_sparse.py (gfdisc.model.capture, gfdisc.saes.load_sae in fp32,
SparseWriter.add_batch(z.cpu().numpy(), tok_ids), BOS dropped). The document stream is
06's stream() body (FineWeb sample-10BT, streaming, truncation to 256, skip docs < 256
tokens) with the n_tokens stop replaced by an exact sequence count (15,696 = 981 x 16, which
is where 06's --n-tokens 4000000 --bs 16 stopped) so the corpus can then run on into the
reserve.

ADDED, AND NOT IN 06:
  * fp64 per-pool-type sums, split by SEQUENCE parity (R9 halves), of
      raw residual  x[pos 1..255]                       (35b part-A construction), and
      SAE-decoded   fp16(code) @ W_dec  (no b_dec)       (scripts/22 representation; linear,
                                                          so the mean of decodes is the decode
                                                          of the mean code)
    for the --raw-acc and --sae-acc layers. Accumulation runs over the FROZEN pool read from
    the stored artifact; the end-of-pass check asserts that the rebuilt counts reproduce that
    pool exactly.
  * corpus-identity anchors asserted EARLY (after --anchor-check-seq sequences) and at the end
    against month rows stored by scripts 39 and 37b; pass 2 additionally asserts byte equality
    with pass 1's corpus tokens batch by batch.
  * isfinite check on every captured residual batch.
  * disk guards that ABORT (never truncate silently): cache bytes, project-folder size,
    system free space.
"""
import argparse, json, os, shutil, sys, time
from pathlib import Path

import numpy as np
import torch

EXT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXT / "src"))
from gfdisc.env import DEVICE, seed_all                     # noqa: E402
from gfdisc.model import load_model, capture                # noqa: E402
from gfdisc.saes import load_sae                            # noqa: E402
from gfdisc.cache import SparseWriter                       # noqa: E402
import gfdisc.model as _gm                                  # noqa: E402

# Load the gated model from the local snapshot by PATH, so dataset streaming can stay online
# without any token (BUDGET.md section 7). Same revision, sha256-verified.
_SNAP = sorted((EXT / "hf_home/hub/models--google--gemma-3-1b-pt/snapshots").glob("*"))
assert len(_SNAP) == 1, _SNAP
_gm.MODEL_NAME = str(_SNAP[0])

OLD = Path.home() / "Downloads/gf-discovery/results"
SEQ_LEN = 256
ROWS_PER_SEQ = SEQ_LEN - 1
N_SEQ_CORPUS = 15_696
D_MODEL = 1152


def qualifying_docs(tok):
    """06's stream() body, one sequence at a time and without the n_tokens stop."""
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True)
    for row in ds:
        ids = tok(row["text"], truncation=True, max_length=SEQ_LEN)["input_ids"]
        if len(ids) < SEQ_LEN:
            continue
        yield ids[:SEQ_LEN]


def load_anchors():
    """row -> expected token id, from two independent stored artifacts."""
    z = np.load(OLD / "phase3_per_occurrence_coords.npz")
    rows = [z["month_row_index"].astype(np.int64)]
    toks = [z["month_token_id"].astype(np.int64)]
    z2 = np.load(OLD / "phase3_context_ladder_natural_means.npz")
    for k in z2.files:
        if k.startswith("occ_rows_"):
            r = z2[k].astype(np.int64)
            rows.append(r)
            toks.append(np.full(len(r), int(k.split("_")[-1]), dtype=np.int64))
    return np.concatenate(rows), np.concatenate(toks)


def dir_bytes(p):
    tot = 0
    for root, _, files in os.walk(p):
        for f in files:
            try:
                tot += os.lstat(os.path.join(root, f)).st_size
            except OSError:
                pass
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--cache-layers", default="12,14,16,18")
    ap.add_argument("--sae-acc", default="")
    ap.add_argument("--raw-acc", default="")
    ap.add_argument("--root", default=str(EXT / "cache/sparse_deep"))
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--reserve", type=int, default=0)
    ap.add_argument("--max-cache-gb", type=float, required=True)
    ap.add_argument("--max-project-gb", type=float, default=19.5)
    ap.add_argument("--min-free-gb", type=float, default=8.0)
    ap.add_argument("--anchor-check-seq", type=int, default=1000)
    ap.add_argument("--reference-tokens", default=None,
                    help="pass-1 corpus tokens; assert byte equality batch by batch")
    ap.add_argument("--n-seq", type=int, default=N_SEQ_CORPUS)
    ap.add_argument("--out-dir", default=str(EXT / "results"))
    ap.add_argument("--skip-seq", type=int, default=0,
                    help="skip this many qualifying docs first (reserve pass: 15696)")
    ap.add_argument("--no-anchor", action="store_true",
                    help="reserve pass: the corpus anchors do not apply")
    a = ap.parse_args()

    seed_all(0)
    t_start = time.time()
    cache_layers = [int(x) for x in a.cache_layers.split(",") if x]
    sae_acc = [int(x) for x in a.sae_acc.split(",") if x]
    raw_acc = [int(x) for x in a.raw_acc.split(",") if x]
    sae_layers = sorted(set(cache_layers) | set(sae_acc))
    cap_layers = sorted(set(sae_layers) | set(raw_acc))
    out_res = Path(a.out_dir)
    out_res.mkdir(parents=True, exist_ok=True)
    log = dict(tag=a.tag, started=time.strftime("%Y-%m-%dT%H:%M:%S"), args=vars(a),
               cache_layers=cache_layers, sae_acc=sae_acc, raw_acc=raw_acc)
    print(json.dumps(log, indent=1), flush=True)

    # ---- frozen pool (verified at the end against the rebuilt counts) ----
    pool_ids = np.load(OLD / "phase3_lag_action_robust_deltas.npz")["pool_ids"].astype(np.int64)
    n_pool = len(pool_ids)
    anchor_rows, anchor_toks = load_anchors()
    ref = None
    if a.reference_tokens:
        ref = np.fromfile(a.reference_tokens, dtype=np.int32)
        assert len(ref) == a.n_seq * ROWS_PER_SEQ

    model, tok = load_model(token=None)
    saes = {L: load_sae(L, "big", device=DEVICE, token=None) for L in sae_layers}
    for L in sae_layers:
        print(f"  layer {L}: nominal l0={saes[L].cfg['l0']}", flush=True)
    wdec = {L: saes[L].w_dec for L in sae_acc}
    writers = {L: SparseWriter(a.root, L, saes[L].d_sae,
                               meta={"suite": "big", "model": "google/gemma-3-1b-pt",
                                     "nominal_l0": saes[L].cfg["l0"],
                                     "writer": "scripts/e01_pass.py (06 path)"})
               for L in cache_layers}
    lut = torch.full((262_144 + 16,), -1, dtype=torch.int64)
    lut[torch.from_numpy(pool_ids)] = torch.arange(n_pool)
    acc_raw = {L: torch.zeros(2 * n_pool, D_MODEL, dtype=torch.float64) for L in raw_acc}
    acc_sae = {L: torch.zeros(2 * n_pool, D_MODEL, dtype=torch.float64) for L in sae_acc}
    cnt_half = np.zeros(2 * n_pool, dtype=np.int64)

    corpus = np.empty(a.n_seq * ROWS_PER_SEQ, dtype=np.int32)
    gen = qualifying_docs(tok)
    for _ in range(a.skip_seq):
        next(gen)
    n_done_seq, nb = 0, 0
    anchors_early_ok = None
    nnz_first = None
    prof = dict(data=0.0, forward=0.0, acc_raw=0.0, csr=0.0, acc_sae=0.0)
    t0 = time.time()
    with torch.inference_mode():
        while n_done_seq < a.n_seq:
            buf = []
            _t = time.time()
            while len(buf) < a.bs and n_done_seq + len(buf) < a.n_seq:
                buf.append(next(gen))
            prof["data"] += time.time() - _t; _t = time.time()
            ids = torch.tensor(buf).to(DEVICE)
            B = ids.shape[0]
            with capture(model, cap_layers) as store:
                model(ids, use_cache=False)
            tok_ids = ids[:, 1:].reshape(-1).cpu().numpy()          # BOS dropped
            prof["forward"] += time.time() - _t; _t = time.time()
            r0 = n_done_seq * ROWS_PER_SEQ
            corpus[r0:r0 + len(tok_ids)] = tok_ids
            if ref is not None and not np.array_equal(ref[r0:r0 + len(tok_ids)], tok_ids):
                raise SystemExit(f"ABORT: corpus differs from reference at seq {n_done_seq}")
            tix = lut[torch.from_numpy(tok_ids.astype(np.int64))]
            par = torch.from_numpy(np.repeat((np.arange(n_done_seq, n_done_seq + B) % 2),
                                             ROWS_PER_SEQ)).to(torch.int64)
            m = tix >= 0
            sel = (tix + par * n_pool)[m]
            np.add.at(cnt_half, sel.numpy(), 1)
            for L in cap_layers:
                x = store[L][:, 1:, :].float().reshape(-1, D_MODEL)       # drop BOS
                if not bool(torch.isfinite(x).all()):
                    raise SystemExit(f"ABORT: non-finite residual at layer {L}, seq {n_done_seq}")
                if L in acc_raw:
                    acc_raw[L].index_add_(0, sel, x.cpu()[m].to(torch.float64))
                    prof["acc_raw"] += time.time() - _t; _t = time.time()
                if L in saes:
                    z = saes[L].encode(x)
                    if L in writers:
                        writers[L].add_batch(z.cpu().numpy(), tok_ids)
                        prof["csr"] += time.time() - _t; _t = time.time()
                    if L in acc_sae:
                        dec = z.half().float() @ wdec[L]
                        acc_sae[L].index_add_(0, sel, dec.cpu()[m].to(torch.float64))
                        prof["acc_sae"] += time.time() - _t; _t = time.time()
                    del z
            del store
            n_done_seq += B
            nb += 1

            if not a.no_anchor and anchors_early_ok is None and n_done_seq >= a.anchor_check_seq:
                lim = n_done_seq * ROWS_PER_SEQ
                mk = anchor_rows < lim
                ok = bool(np.array_equal(corpus[anchor_rows[mk]], anchor_toks[mk]))
                anchors_early_ok = dict(n_checked=int(mk.sum()), ok=ok, at_seq=n_done_seq)
                print(f"  [ANCHOR EARLY] {anchors_early_ok}", flush=True)
                if not ok:
                    raise SystemExit("ABORT: corpus identity anchors fail -- dataset order changed")
            if nnz_first is None and nb == 50 and writers:
                nnz_first = {L: writers[L].n_nz / (n_done_seq * ROWS_PER_SEQ) for L in writers}
                print(f"  [NNZ/ROW first 50 batches] {nnz_first}", flush=True)
            if nb % 20 == 0 or n_done_seq >= a.n_seq:
                el = time.time() - t0
                rows = n_done_seq * ROWS_PER_SEQ
                cgb = sum(w.n_nz * 4 + len(w.indptr) * 12 for w in writers.values()) / 1e9
                pgb = dir_bytes(EXT) / 1e9
                free = shutil.disk_usage(EXT).free / 1e9
                print(f"  seq {n_done_seq}/{a.n_seq} | {rows/el:,.0f} tok/s | cache {cgb:.2f} GB"
                      f" | project {pgb:.2f} GB | free {free:.1f} GB | eta "
                      f"{(a.n_seq-n_done_seq)*ROWS_PER_SEQ/max(rows/el,1)/60:.0f} min | prof "
                      f"{ {k: round(v / nb, 2) for k, v in prof.items()} } s/batch", flush=True)
                if cgb > a.max_cache_gb or pgb > a.max_project_gb or free < a.min_free_gb:
                    raise SystemExit(f"ABORT: disk guard (cache {cgb:.2f}/{a.max_cache_gb}, "
                                     f"project {pgb:.2f}/{a.max_project_gb}, "
                                     f"free {free:.1f}/{a.min_free_gb})")

    secs_corpus = time.time() - t0
    meta = {str(L): writers[L].close() for L in cache_layers}

    # ---- end-of-pass identity checks ----
    mk_all = anchor_rows < len(corpus)
    if a.no_anchor:
        mk_all = np.zeros(len(anchor_rows), dtype=bool)
    ok_all = bool(np.array_equal(corpus[anchor_rows[mk_all]], anchor_toks[mk_all])
                  and (a.n_seq < N_SEQ_CORPUS or mk_all.all()))
    cnt = np.bincount(corpus, minlength=max(int(corpus.max()), int(pool_ids.max())) + 1)
    pool_rebuilt = np.flatnonzero(cnt >= 100)
    pool_same = bool(np.array_equal(pool_rebuilt, pool_ids)) or a.no_anchor
    counts_pool = cnt[pool_ids]
    half_counts_ok = bool(np.array_equal(cnt_half[:n_pool] + cnt_half[n_pool:], counts_pool))
    ident = dict(anchors_early=anchors_early_ok, anchors_all=dict(n=int(mk_all.sum()),
                 ok=ok_all), pool_reproduced=pool_same, n_pool_rebuilt=int(len(pool_rebuilt)),
                 n_distinct_types=int((cnt > 0).sum()),
                 n_rows_covered_by_pool=int(counts_pool.sum()), half_counts_ok=half_counts_ok)
    print(f"  [IDENTITY] {ident}", flush=True)

    corpus.tofile(out_res / f"corpus_tokens_{a.tag}.i32")

    # ---- held-out reserve (tokens only, BOS included) ----
    if a.reserve:
        res = np.empty((a.reserve, SEQ_LEN), dtype=np.int32)
        for i in range(a.reserve):
            res[i] = next(gen)
        res.tofile(out_res / "reserve_tokens.i32")
        print(f"  reserve: {a.reserve} sequences x {SEQ_LEN} written", flush=True)

    # ---- means ----
    nA, nB = cnt_half[:n_pool], cnt_half[n_pool:]
    arrs = dict(pool_ids=pool_ids, counts=counts_pool, counts_A=nA, counts_B=nB,
                raw_layers=np.asarray(raw_acc, dtype=np.int64),
                sae_layers=np.asarray(sae_acc, dtype=np.int64))
    for name, acc in (("raw", acc_raw), ("sae", acc_sae)):
        for L, A in acc.items():
            sA, sB = A[:n_pool].numpy(), A[n_pool:].numpy()
            arrs[f"{name}_A_L{L}"] = (sA / np.maximum(nA, 1)[:, None]).astype(np.float32)
            arrs[f"{name}_B_L{L}"] = (sB / np.maximum(nB, 1)[:, None]).astype(np.float32)
            arrs[f"{name}_L{L}"] = ((sA + sB) / np.maximum(nA + nB, 1)[:, None]).astype(np.float32)
            assert np.isfinite(arrs[f"{name}_L{L}"]).all()
    if acc_raw or acc_sae:
        np.savez(out_res / f"typemeans_{a.tag}.npz", **arrs)
    if acc_raw and not a.no_anchor:   # 35b part-A format, so 42/47/47b/49 load it unchanged
        np.savez(out_res / "phase3_token_type_mean_resid_fulln.npz",
                 pool_ids=pool_ids, counts=counts_pool.astype(np.int64),
                 n_seq=np.int64(a.n_seq), layers=np.asarray(raw_acc, dtype=np.int64),
                 **{f"R_L{L}": arrs[f"raw_L{L}"] for L in raw_acc})

    log.update(finished=time.strftime("%Y-%m-%dT%H:%M:%S"), seconds_corpus=round(secs_corpus),
               seconds_total=round(time.time() - t_start), cache_meta=meta,
               nnz_per_row={L: m_["n_nonzero"] / m_["n_rows"] for L, m_ in meta.items()},
               nnz_per_row_first50=nnz_first, identity=ident,
               project_gb_end=dir_bytes(EXT) / 1e9,
               free_gb_end=shutil.disk_usage(EXT).free / 1e9)
    (out_res / f"pass_{a.tag}.json").write_text(json.dumps(log, indent=2))
    print(json.dumps({k: log[k] for k in ("nnz_per_row", "identity", "project_gb_end",
                                           "seconds_corpus")}, indent=1), flush=True)
    if not (ok_all and pool_same and half_counts_ok):
        raise SystemExit("IDENTITY CHECK FAILED -- see pass json")


if __name__ == "__main__":
    # pyarrow's thread pool can hang interpreter exit while a streaming generator is still
    # open (seen in the smoke run), so leave hard -- after printing any traceback.
    import traceback
    rc = 0
    try:
        main()
    except SystemExit as e:
        print(f"SystemExit: {e}", flush=True)
        rc = 0 if e.code in (None, 0) else 1
    except BaseException:
        traceback.print_exc()
        rc = 1
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(rc)
