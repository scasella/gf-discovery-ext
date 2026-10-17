"""48b -- Phase 5 Addendum P (P1/P2/P3): re-adjudicate the stored proposals of scripts 48
and 49 under the duplicate-referent guard (P1) and the saturated-cell channel-3 adjudicator
(P2).

Nothing about the proposers changes and nothing is re-proposed (P3: addendum-only). What
changes is (a) every proposal is passed through the P1 duplicate-referent guard before the
tier sees it, and (b) proposals that the tier left NOT ADJUDICABLE for want of an
unsaturated cell are adjudicated by the R16 channel-3 conjunct instead (P2).

The tier is imported by path from scripts/48_agreement_graph_screen.py (which itself imports
42's closure / cell_stats through 43). Both scripts' proposals are run through the SAME
imported tier so that 49's rows and 48's rows are adjudicated identically.

R17 (CPU only) is in force: no MPS tensor is created anywhere in this script.
"""
import argparse, importlib.util, json, os, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(REPO / "cache/hf"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(REPO / "src"))

SRC48 = "results/phase5_agreement_screen.json"
SRC49 = "results/phase5_occurrence_plane_screen.json"
LAYERS = (6, 7)
PREFIX_MIN = 3


def _load(name, rel):
    sp = importlib.util.spec_from_file_location(name, str(REPO / rel))
    m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    return m


# =====================================================================================
# declaration (written before any artifact is opened)
# =====================================================================================
def declared_rules():
    return dict(
        prereg="docs/phase5_prereg_2026-08-22.md (R13-R18) + docs/phase5_addendum_P_2026-08-22.md (P1-P3)",
        scope=("re-adjudication only. No proposer is re-run for the purpose of proposing: "
               "48's screens are re-executed ONLY to recover proposal MEMBERSHIPS, which "
               "48 did not store for non-candidate rows, and the recovered population is "
               "verified identical to 48's before anything is guarded (see "
               "verification.regeneration). 49's 130 proposals are read from its JSON."),
        P1_guard=(
            "duplicate-referent guard, label-free and distinct from R15. Normalisation: "
            "strip ONE leading space if present, then lowercase. Two members are duplicates "
            "iff their normalised strings are (i) equal, or (ii) one is a prefix of the "
            f"other and the shorter has length >= {PREFIX_MIN}, or (iii) Levenshtein "
            "distance <= 1. "
            "DROP RULE (operationalised here; P1 says only 'the lower-count member is "
            "dropped', which is not order-invariant for chains): within a proposal, define "
            "each member's key as (corpus count, -pool index); while any duplicate pair "
            "remains among the surviving members, drop the member with the globally "
            "smallest key among all members that are the lower-key half of at least one "
            "live duplicate pair; repeat. This is deterministic and independent of member "
            "order. "
            "NOTE (declared, not softened): rule (iii) makes every pair of single-character "
            "normalised strings duplicates, so any proposal built from single letters "
            "collapses to k = 1. That is what the frozen rule says and it is applied as "
            "written."),
        P2_adjudicator=(
            "saturated-cell adjudication by the R16 channel-3 conjunct. A proposal is "
            "P2-ELIGIBLE iff, after the P1 guard: k <= 10 (P2's bound, which is also R16's "
            "declared-conjunct bound); closure gap_ratio < 2.5; rho == 1.0; all k "
            "leave-one-out subsets close (gap_ratio < 2.5); and the tier's verdict is NOT "
            "ADJUDICABLE BECAUSE a cell is saturated (p99 == 1.0) -- either the k cell or "
            "the (k-1) LOO cell. PASS_P2 := P2-eligible AND channel 3 (OV retention) > the "
            "99th percentile of 500 random-k planes at that layer AND the frequency-matched "
            "arm passes AND the stored permuted-search mean < 0.5. The cell's conditional "
            "exact-1.0 rate is reported for every P2 row. PASS_P2 is reported SEPARATELY "
            "from R14 PASS and is never merged into it. k >= 11 saturated rows stay NOT "
            "ADJUDICABLE: P2's k <= 10 bound governs."),
        P2_multiplicity=("expected false PASS_P2 = 0.01 x (number of P2-adjudicated "
                         "proposals), reported per source."),
        tier=("imported by path from scripts/48_agreement_graph_screen.py: closure first "
              "(gap_ratio < 2.5), then the R14 order conjunct against the stricter of the "
              "ch1-drawn and ch2-drawn exact cluster-like cells with the per-k screen tie "
              "class (48's D5), then LOO closure and LOO order at the (k-1) ch1 cell "
              "(48's D1), then the frequency-matched arm (200 count-matched k-sets, +-0.25 "
              "log10), then the stored permuted-search mean. Channel 3 recomputed here with "
              "48's Channel3 for BOTH sources; 49's stored channel3_ov_retention is not "
              "used."),
        tie_class=("For SOURCE 48, R14's tie class is recomputed on the GUARDED population "
                   "per (layer, proposer, k): guarding moves k downward, so 48's stored "
                   "screen_max_rho_by_k no longer describes the population being "
                   "adjudicated. For SOURCE 49 the tie-class clause is DISABLED and the "
                   "order conjunct is decided by the cell p99 alone. Reason (declared "
                   "here): 49's screen population per (layer, tau) is 2-33 proposals, so "
                   "'rho >= the screen's per-k maximum' is satisfied BY CONSTRUCTION by "
                   "every proposal that is the only one at its k -- the clause would be "
                   "vacuous and strictly permissive. 49's own tier likewise had no "
                   "tie-class clause. The per-k maxima are still reported."),
        permuted_search=("reported AS STORED from 48. Not re-run: the guard was NOT applied "
                         "inside 48's permuted searches. For proposer A this cannot matter "
                         "-- the permuted searches produced 0 PASSers (and, at L7, 0 "
                         "proposals at all), and a guard can only shrink proposals. For "
                         "proposer B, 48 stored only passers[:10] (empty), not the permuted "
                         "proposal memberships, so guard+P2 cannot be applied to them; this "
                         "is stated as a limitation, not run."),
        deviations=[
            "E1 (memberships not stored). 48 stored memberships only for PASS/MARGINAL "
            "candidates (2 rows, both the strict-12). Its screens are deterministic "
            "functions of the frozen channels, so the full proposal population is "
            "REGENERATED here and verified against 48's stored counts, strict-12 hashes, "
            "sources and per-(layer, proposer) tallies before any guard is applied. A "
            "mismatch aborts the run.",
            "E2 (nulls and plants not guarded). P1 says the guard applies identically to "
            "proposals, nulls and plants. Here it is applied to PROPOSALS only: the exact "
            "cluster-like cells, the frequency-matched arm draws, the channel-3 random-k "
            "planes, the permuted searches and the plants are unguarded. Reason: a guarded "
            "null draw has variable k, so a fixed-k cell would no longer exist. Direction "
            "of bias: duplicate members inflate a set's rho, so an unguarded null is if "
            "anything HARDER to beat than a guarded one -- the deviation is conservative "
            "against PASS, not permissive.",
            "E3 (L5 skipped). 49's L5 proposals (64 rows at tau 30/40/45) are NOT "
            "re-adjudicated. Channel 2 at L5 is not on disk (results/phase3_lag_action_"
            "robust_deltas.npz holds L6/L7 only; results/phase4_lag1_deltas_L*.npz hold "
            "L0-L4 and L10) and recomputing it needs 42's cache stream, which R12/R17 "
            "forbid while script 47 holds MPS. All 64 stored L5 verdicts were FAIL "
            "(closure), and the guard can move closure in either direction, so they are "
            "reported as NOT RE-CHECKED.",
            "E4 (49's ring conjunct carried forward). Prereg 2.2's extra conjunct -- the "
            "cluster's shared plane > 60 deg from its own type-mean ring plane -- is a "
            "function of the occurrence-plane rows, which are not re-streamed here (same "
            "R12/R17 reason as E3). Stored ring_conjunct_pass values are carried forward "
            "and flagged: for a guarded subset the shared plane would have to be refitted.",
            "E5 (49 tier re-implementation). 49 adjudicated with its own null "
            "construction (sampled cluster-like cells, 3 seeds, 6000 draws, plus a "
            "count-matched-random cell). Here 49's proposals are run through 48's imported "
            "tier (exact enumerated cells). To separate the guard's effect from the tier "
            "change, 49's proposals are adjudicated BOTH unguarded and guarded under 48's "
            "tier, and both are reported.",
        ],
    )


# =====================================================================================
# P1 duplicate-referent guard
# =====================================================================================
def normalise(s):
    return (s[1:] if s.startswith(" ") else s).lower()


def lev_le1(a, b):
    """True iff Levenshtein(a, b) <= 1."""
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False
    if la == lb:
        diff = 0
        for x, y in zip(a, b):
            if x != y:
                diff += 1
                if diff > 1:
                    return False
        return True
    if la > lb:
        a, b, la, lb = b, a, lb, la
    i = 0
    while i < la and a[i] == b[i]:
        i += 1
    return a[i:] == b[i + 1:]


def build_dupmap(strings):
    """i -> set of j != i that are P1 duplicates of i. Candidate pairs are generated by
    equality buckets, a prefix dictionary and deletion-neighbourhood hashing (which is
    complete for Levenshtein <= 1); every candidate pair is then verified exactly."""
    norm = [normalise(s) for s in strings]
    n = len(norm)
    cand = defaultdict(set)

    def add(i, j):
        if i != j:
            cand[i].add(j)
            cand[j].add(i)

    eq = defaultdict(list)
    for i, a in enumerate(norm):
        eq[a].append(i)
    for a, idxs in eq.items():
        for x in range(len(idxs)):
            for y in range(x + 1, len(idxs)):
                add(idxs[x], idxs[y])

    # prefix rule: shorter must have length >= PREFIX_MIN
    for i, a in enumerate(norm):
        for L in range(PREFIX_MIN, len(a)):
            for j in eq.get(a[:L], ()):
                add(i, j)

    # Levenshtein <= 1 via deletion neighbourhoods
    delmap = defaultdict(list)
    for i, a in enumerate(norm):
        keys = {a}
        for p in range(len(a)):
            keys.add(a[:p] + a[p + 1:])
        for kk in keys:
            delmap[kk].append(i)
    for kk, idxs in delmap.items():
        if len(idxs) < 2:
            continue
        for x in range(len(idxs)):
            for y in range(x + 1, len(idxs)):
                add(idxs[x], idxs[y])

    dup = {}
    for i, js in cand.items():
        a = norm[i]
        keep = set()
        for j in js:
            b = norm[j]
            if a == b:
                keep.add(j); continue
            if min(len(a), len(b)) >= PREFIX_MIN and (a.startswith(b) or b.startswith(a)):
                keep.add(j); continue
            if lev_le1(a, b):
                keep.add(j)
        if keep:
            dup[i] = keep
    return dup, norm


def guard(members, dup, key):
    """P1. key[i] = (count, -pool index); the globally smallest-key lower-half member of a
    live duplicate pair is dropped, repeatedly."""
    S = sorted(int(x) for x in members)
    if not any(i in dup for i in S):
        return S, []
    dropped = []
    while True:
        alive = set(S)
        losers = []
        for i in S:
            js = dup.get(i)
            if not js:
                continue
            for j in js:
                if j in alive and j != i and key[i] < key[j]:
                    losers.append(i)
                    break
        if not losers:
            break
        drop = min(losers, key=lambda i: key[i])
        S = [x for x in S if x != drop]
        dropped.append(int(drop))
        if len(S) < 2:
            break
    return sorted(S), dropped


# =====================================================================================
# adjudication of a whole population through 48's tier
# =====================================================================================
def adjudicate_population(s48, S42, props, C1, C2, cells, counts, label, log,
                          want_records=True, use_tie_class=True):
    """props: hash -> dict(members=..., k=...). Returns (verdicts by hash, records by hash
    for gate rows, per-k tie floors, tally)."""
    t0 = time.time()
    GAP, KMIN = s48.GAP, s48.KMIN
    quick = {}
    for h, p in props.items():
        mem = np.asarray(p["members"])
        if len(mem) < KMIN:
            quick[h] = dict(gap=None, rho=None)
            continue
        g = S42.gap_ratio_of(C1[mem])
        if g is None or g >= GAP:
            quick[h] = dict(gap=None if g is None else float(g), rho=None)
            continue
        _, r = S42.cell_stats(C1[mem], C2[mem])
        quick[h] = dict(gap=float(g), rho=None if r is None else float(r))
    rho_max_k = {}
    for h, q in quick.items():
        if q["rho"] is None or props[h]["k"] < KMIN:
            continue
        k = props[h]["k"]
        if k not in rho_max_k or q["rho"] > rho_max_k[k]:
            rho_max_k[k] = q["rho"]
    tie_size = defaultdict(int)
    for h, q in quick.items():
        if q["rho"] is None or props[h]["k"] < KMIN:
            continue
        if q["rho"] >= rho_max_k[props[h]["k"]] - 1e-12:
            tie_size[props[h]["k"]] += 1

    gate = []
    for h, q in quick.items():
        if q["rho"] is None or props[h]["k"] < KMIN:
            continue
        kk = props[h]["k"]
        cell = s48.cell_view(cells.get(kk), "strict")
        lo = cell["p99_min_strict"]
        tf = rho_max_k.get(kk) if use_tie_class else None
        if (tf is not None and q["rho"] >= tf - 1e-12) or \
                (lo is not None and q["rho"] >= lo) or cell["saturated"]:
            gate.append(h)

    verdicts, recs = {}, {}
    n_small = 0
    for h, p in props.items():
        if p["k"] < KMIN:
            verdicts[h] = "NOT ADJUDICABLE"
            n_small += 1
    for h in gate:
        r = s48.adjudicate(S42, props[h]["members"], C1, C2, cells,
                           rho_max_k.get(props[h]["k"]) if use_tie_class else None)
        r["member_hash"] = h
        verdicts[h] = r["verdict"]
        if want_records:
            recs[h] = r
    for h in props:
        if h not in verdicts:
            verdicts[h] = "FAIL"
    tally = defaultdict(int)
    for v in verdicts.values():
        tally[v] += 1
    log(f"  [{label}] {len(props)} proposals, "
        f"{sum(1 for q in quick.values() if q['rho'] is not None)} adjudicable closers, "
        f"{len(gate)} at the order gate, tally {dict(tally)} ({time.time()-t0:.0f} s)")
    return dict(verdicts=verdicts, records=recs, rho_max_k=rho_max_k,
                tie_size={int(k): int(v) for k, v in tie_size.items()},
                tally=dict(tally), n_gate=len(gate), quick=quick,
                n_closers=sum(1 for q in quick.values() if q["rho"] is not None),
                seconds=round(time.time() - t0, 1))


def p2_eligible(rec, members, cells, s48, S42, C1):
    """P2 eligibility. Returns (bool, reason, which cell was saturated, exact1 rate)."""
    k = rec["k"]
    if rec["verdict"] != "NOT ADJUDICABLE":
        return False, "verdict is not NOT ADJUDICABLE", None, None
    if k > 10:
        return False, "k > 10 (P2 bound)", None, None
    if not rec["closure_pass"]:
        return False, "closure fail", None, None
    rho = rec["two_channel_rho"]
    if rho is None or rho < 1.0 - 1e-12:
        return False, f"rho {rho} != 1.0", None, None
    ck = cells.get(k)
    ckl = cells.get(k - 1)
    sat_k = bool(ck["saturated"])
    sat_loo = bool(ckl["ch1"]["saturated"])
    if not (sat_k or sat_loo):
        return False, "NOT ADJUDICABLE for a reason other than cell saturation", None, None
    loo_ok = rec.get("loo_closure_pass")
    if loo_ok is None:
        # adjudicate() returned at the saturated k cell before running LOO; P2 still
        # requires LOO closure, so compute it here.
        mem = np.asarray(sorted(int(x) for x in members))
        gaps = [S42.gap_ratio_of(C1[np.delete(mem, i)]) for i in range(k)]
        loo_ok = all(g is not None and g < s48.GAP for g in gaps)
        rec["loo_closure_pass"] = bool(loo_ok)
        rec["loo_gap_max"] = max([g for g in gaps if g is not None], default=None)
    if not loo_ok:
        return False, "LOO closure fail", None, None
    return True, "eligible", ("k" if sat_k else "loo"), (
        ck["exact1_rate"] if sat_k else ckl["ch1"]["exact1_rate"])


# =====================================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/phase5_readjudication_P.json")
    ap.add_argument("--layers", default="6,7")
    ap.add_argument("--skip-verify", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    layers = [int(x) for x in args.layers.split(",")]

    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    doc = dict(status="RUNNING -- declaration stub written before any artifact was opened",
               script="scripts/48b_readjudicate_P.py",
               declared_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
               sources=[SRC48, SRC49],
               config=dict(layers=layers, prefix_min=PREFIX_MIN, smoke=bool(args.smoke)),
               declared_rules=declared_rules())
    out.write_text(json.dumps(doc, indent=1))

    logf = open(REPO / "logs/48b_readj.log", "a")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n"); logf.flush()

    log(f"=== 48b start {doc['declared_at']} layers={layers}")
    res = dict(verification={}, guard={}, source48={}, source49={}, channel3_nulls={},
               permuted_search_as_stored={})

    def save(status="RUNNING"):
        out.write_text(json.dumps({**doc, **res, "status": status}, indent=1, default=float))

    s48 = _load("s48mod", "scripts/48_agreement_graph_screen.py")
    S43 = _load("s43mod", "scripts/43_neighbour_assembly_screen.py")
    S42 = S43.S42
    log("tier imported: scripts/48 (screens, cells, adjudicate, Channel3) over scripts/42")

    stored48 = json.loads((REPO / SRC48).read_text())
    stored49 = json.loads((REPO / SRC49).read_text())
    res["permuted_search_as_stored"] = {
        k: dict(n_permutations=v["n_permutations"], mean_pass=v["mean_pass"],
                pass_counts=v["pass_counts"], proposal_counts=v["proposal_counts"])
        for k, v in stored48["permuted_search"].items()}

    d1 = np.load(REPO / s48.FULLN_RAW)
    d2 = np.load(REPO / s48.ROBUST_DELTAS)
    pool = d1["pool_ids"].astype(int)
    assert (pool == d2["pool_ids"].astype(int)).all()
    counts = d1["counts"].astype(np.int64)
    n_types = len(pool)
    pos = {int(t): i for i, t in enumerate(pool)}
    months = np.asarray(sorted(pos[t] for t in s48.STRICT12_IDS))
    assert s48.mhash(months) == s48.STRICT12_HASH
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("google/gemma-3-1b-pt")
    strings = [tok.decode([int(i)]) for i in pool]
    log(f"pool {n_types} types; strict-12 hash verified")

    C1, C2, Z1, Z2, keepc = {}, {}, {}, {}, {}
    for L in layers:
        keep = d2[f"L{L}_lag1_kept_coords"]
        keepc[L] = np.asarray(keep)
        C1[L] = d1[f"R_L{L}"][:, keep].astype(np.float64)
        C2[L] = d2[f"L{L}_lag1"].astype(np.float32).astype(np.float64)
        Z1[L] = s48.centred_unit(C1[L]).astype(np.float32)
        Z2[L] = s48.centred_unit(C2[L]).astype(np.float32)
    c3 = s48.Channel3(layers, keepc)
    cellsL = {L: s48.Cells(S42, C1[L], C2[L], Z1[L], Z2[L], L) for L in layers}
    log("channels + channel 3 + cells ready")

    # channel-3 null helper (48's ov_null, 48's seeding), with reproduction check
    def ch3_null(L, k):
        key = f"L{L}_k{k}"
        if key not in res["channel3_nulls"]:
            res["channel3_nulls"][key] = s48.ov_null(c3, L, C1[L], k)
        return res["channel3_nulls"][key]

    chk = ch3_null(6, 12)
    stored_p99 = stored48["channel3_nulls"]["L6_k12"]["p99"]
    res["verification"]["channel3_null_L6_k12"] = dict(
        recomputed=chk["p99"], stored=stored_p99,
        agrees=bool(abs(chk["p99"] - stored_p99) < 1e-12))
    log(f"channel-3 null reproduction L6_k12: {chk['p99']} vs stored {stored_p99} "
        f"({res['verification']['channel3_null_L6_k12']['agrees']})")

    # ---------------- P1 guard map ----------------
    t0 = time.time()
    dup, norm = build_dupmap(strings)
    key = {i: (int(counts[i]), -i) for i in range(n_types)}
    n_pairs = sum(len(v) for v in dup.values()) // 2
    months_g, months_drop = guard(months, dup, key)
    res["guard"] = dict(
        n_types_with_at_least_one_duplicate=len(dup), n_duplicate_pairs=int(n_pairs),
        seconds=round(time.time() - t0, 1),
        strict12_unchanged=bool(list(months_g) == list(months)),
        strict12_dropped=[strings[i] for i in months_drop],
        examples=[[strings[i], strings[j]] for i in list(dup)[:400]
                  for j in list(dup[i])[:1]][:25])
    log(f"P1 guard map: {len(dup)} types in >=1 duplicate pair, {n_pairs} pairs "
        f"({time.time()-t0:.0f} s); strict-12 unchanged = "
        f"{res['guard']['strict12_unchanged']}")
    save()

    def row(L, mem, rec, extra=None):
        mem = list(int(x) for x in mem)
        k = len(mem)
        ret, per_head = c3.retention(L, C1[L][np.asarray(mem)])
        nl = ch3_null(L, k)
        d = dict(member_hash=s48.mhash(mem), layer=L, k=k,
                 tokens=[strings[i] for i in mem],
                 token_ids=[int(pool[i]) for i in mem],
                 counts=[int(counts[i]) for i in mem],
                 gap_ratio=rec.get("gap_ratio"), rho=rec.get("two_channel_rho"),
                 order_state=rec.get("order_state"), order_detail=rec.get("order_detail"),
                 loo_order_state=rec.get("loo_order_state"),
                 loo_closure_pass=rec.get("loo_closure_pass"),
                 loo_rho_min=rec.get("loo_rho_min"), loo_gap_max=rec.get("loo_gap_max"),
                 cell_p99_band=rec.get("cell_p99_band"),
                 cell_saturated=rec.get("cell_saturated"),
                 verdict=rec.get("verdict"),
                 channel3=dict(ov_retention=ret, per_head=per_head, null_p99=nl["p99"],
                               null_mean=nl["mean"], above_p99=bool(ret > nl["p99"]),
                               declared_conjunct=bool(k <= 10)),
                 is_strict12=bool(s48.mhash(mem) == s48.STRICT12_HASH))
        if extra:
            d.update(extra)
        return d

    def finish_R14(L, mem, rec, perm_mean):
        """frequency arm + R14 conjunction for a PASS/MARGINAL row."""
        fa = s48.frequency_arm(S42, np.asarray(mem), C1[L], C2[L], counts,
                               rec["two_channel_rho"])
        r = row(L, mem, rec, dict(frequency_arm=fa, permuted_search_mean=perm_mean))
        r["R14_PASS"] = bool(rec["verdict"] == "PASS" and fa["pass_"]
                             and perm_mean is not None and perm_mean < 0.5
                             and (r["channel3"]["above_p99"] if r["k"] <= 10 else True))
        return r

    def finish_P2(L, mem, rec, perm_mean, why, sat_cell, exact1):
        """Channel 3 is evaluated FIRST (it is P2's adjudicator and is cheap); the
        frequency-matched arm is run only for rows that clear it."""
        r = row(L, mem, rec, dict(frequency_arm=None, permuted_search_mean=perm_mean,
                                  p2_saturated_cell=sat_cell, p2_exact1_rate=exact1,
                                  p2_reason=why))
        if not r["channel3"]["above_p99"]:
            r["PASS_P2"] = False
            r["p2_fail_reason"] = "channel 3 <= 99th percentile of 500 random-k planes"
            r["frequency_arm_note"] = "not run (channel-3 conjunct already fails)"
            r["R14_PASS"] = False
            return r
        fa = s48.frequency_arm(S42, np.asarray(mem), C1[L], C2[L], counts,
                               rec["two_channel_rho"])
        r["frequency_arm"] = fa
        r["PASS_P2"] = bool(fa["pass_"] and perm_mean is not None and perm_mean < 0.5)
        if not r["PASS_P2"]:
            r["p2_fail_reason"] = ("frequency-matched arm" if not fa["pass_"]
                                   else "permuted-search mean >= 0.5")
        r["R14_PASS"] = False
        return r

    # =================================================================================
    # SOURCE 48
    # =================================================================================
    for L in layers:
        t0 = time.time()
        pA, sA = s48.screen_A(S42, C1[L], C2[L], Z1[L], Z2[L], Z1[L], counts, cellsL[L], L)
        pB, sB = s48.screen_B(S42, C1[L], C2[L], Z2[L], Z1[L], counts, L)
        log(f"L{L} screens regenerated: A {len(pA)}, B {len(pB)} ({time.time()-t0:.0f} s)")
        for nm, props, st in (("A", pA, sA), ("B", pB, sB)):
            stored_blk = stored48["layers"][f"L{L}"][nm]
            v = dict(n_unique_regenerated=len(props), n_unique_stored=stored_blk["n_unique"],
                     counts_agree=bool(len(props) == stored_blk["n_unique"]),
                     strict12_present=bool(s48.STRICT12_HASH in props),
                     strict12_present_stored=stored48["positive_control"][f"L{L}_{nm}"][
                         "strict12_proposed_blind"],
                     proposer_stats_agree=bool(st == stored_blk["proposer_stats"]))
            if s48.STRICT12_HASH in props:
                v["strict12_sources"] = props[s48.STRICT12_HASH]["source"]
                v["strict12_sources_stored"] = \
                    stored48["positive_control"][f"L{L}_{nm}"]["sources"]
                v["strict12_sources_agree"] = bool(
                    v["strict12_sources"] == v["strict12_sources_stored"])
            res["verification"][f"48_L{L}_{nm}_regeneration"] = v
            log(f"  verify L{L}/{nm}: counts {v['counts_agree']}, stats "
                f"{v['proposer_stats_agree']}, strict12 {v['strict12_present']}")
            if not (v["counts_agree"] and v["proposer_stats_agree"]):
                save("ABORTED -- regenerated population does not match 48")
                raise SystemExit(f"regeneration mismatch at L{L}/{nm}")
        save()

        for nm, props in (("A", pA), ("B", pB)):
            perm_mean = stored48["permuted_search"][f"L{L}_{nm}"]["mean_pass"]
            blk = {}
            un = None
            # -------- unguarded re-adjudication (verification of the tier path) --------
            if not args.skip_verify:
                un = adjudicate_population(s48, S42, props, C1[L], C2[L], cellsL[L], counts,
                                           f"48/L{L}/{nm} unguarded", log)
                stored_tally = stored48["layers"][f"L{L}"][nm]["tally"]
                agree = all(un["tally"].get(kk, 0) == stored_tally.get(kk, 0)
                            for kk in set(un["tally"]) | set(stored_tally))
                res["verification"][f"48_L{L}_{nm}_tally"] = dict(
                    regenerated=un["tally"], stored=stored_tally, agrees=bool(agree))
                log(f"  verify L{L}/{nm} tally agrees = {agree}")
                if not agree:
                    save("ABORTED -- unguarded tally does not reproduce 48")
                    raise SystemExit(f"tally mismatch at L{L}/{nm}")
                unverdicts = un["verdicts"]
            else:
                unverdicts = {}

            # -------- guard --------
            t1 = time.time()
            gprops, parent = {}, {}
            n_changed = n_below = 0
            for h, p in props.items():
                g, dropped = guard(p["members"], dup, key)
                if dropped:
                    n_changed += 1
                gh = s48.mhash(g) if len(g) >= 2 else f"small:{h}"
                if len(g) < s48.KMIN:
                    n_below += 1
                if gh not in gprops:
                    gprops[gh] = dict(members=g, k=len(g))
                parent[h] = gh
            log(f"  [48/L{L}/{nm}] guard: {n_changed}/{len(props)} proposals changed, "
                f"{n_below} fell below k=9, {len(gprops)} unique guarded sets "
                f"({time.time()-t1:.0f} s)")
            gsources = defaultdict(set)
            for h, gh in parent.items():
                gsources[gh].update(props[h]["source"])
            ga = adjudicate_population(s48, S42, gprops, C1[L], C2[L], cellsL[L], counts,
                                       f"48/L{L}/{nm} guarded", log)

            # -------- verdict changes at the parent-proposal level --------
            changes = defaultdict(int)
            for h in props:
                a = unverdicts.get(h, "n/a")
                b = ga["verdicts"][parent[h]]
                changes[f"{a} -> {b}"] += 1
            changed_rows = []
            for h in props:
                a = unverdicts.get(h, "n/a")
                b = ga["verdicts"][parent[h]]
                if a != b and (a in ("PASS", "MARGINAL") or b in ("PASS", "MARGINAL")):
                    changed_rows.append(dict(
                        original_hash=h, original_verdict=a, guarded_verdict=b,
                        original_tokens=[strings[i] for i in props[h]["members"]],
                        guarded_tokens=[strings[i] for i in gprops[parent[h]]["members"]],
                        source=props[h]["source"]))

            # -------- R14 rows and P2 rows on the guarded population --------
            passes, p2rows = [], []
            n_p2 = 0
            for gh, r in ga["records"].items():
                mem = gprops[gh]["members"]
                if r["verdict"] in ("PASS", "MARGINAL"):
                    passes.append(finish_R14(L, mem, r, perm_mean))
                else:
                    ok, why, sat_cell, ex1 = p2_eligible(r, mem, cellsL[L], s48, S42,
                                                         C1[L])
                    if ok:
                        n_p2 += 1
                        pr = finish_P2(L, mem, r, perm_mean, why, sat_cell, ex1)
                        pr["sources"] = sorted(gsources.get(gh, set()))[:8]
                        p2rows.append(pr)
            blk = dict(
                n_proposals=len(props), n_unique_guarded=len(gprops),
                n_proposals_changed_by_guard=n_changed,
                n_guarded_below_kmin=n_below,
                guarded_tally=ga["tally"],
                unguarded_tally=un["tally"] if un else None,
                guarded_n_gate=ga["n_gate"], guarded_n_closers=ga["n_closers"],
                guarded_screen_max_rho_by_k={str(k): v for k, v in
                                             sorted(ga["rho_max_k"].items())},
                guarded_tie_size_by_k={str(k): v for k, v in sorted(ga["tie_size"].items())},
                verdict_transitions=dict(changes),
                verdict_change_rows=changed_rows[:50],
                n_verdict_change_rows=len(changed_rows),
                permuted_search_mean_as_stored=perm_mean,
                R14_rows=passes, n_R14_PASS=sum(1 for r in passes if r["R14_PASS"]),
                n_P2_adjudicated=n_p2,
                expected_false_P2=0.01 * n_p2,
                P2_rows=[r for r in p2rows if r["PASS_P2"]],
                n_PASS_P2=sum(1 for r in p2rows if r["PASS_P2"]),
                P2_rows_failed_summary=[dict(k=r["k"], rho=r["rho"], tokens=r["tokens"],
                                             ch3=r["channel3"]["ov_retention"],
                                             ch3_p99=r["channel3"]["null_p99"],
                                             fail=r.get("p2_fail_reason"))
                                        for r in p2rows if not r["PASS_P2"]][:40])
            res["source48"][f"L{L}_{nm}"] = blk
            log(f"  [48/L{L}/{nm}] guarded: {blk['n_R14_PASS']} R14 PASS, "
                f"{n_p2} P2-adjudicated, {blk['n_PASS_P2']} PASS_P2, "
                f"{len(changed_rows)} PASS/MARGINAL-touching verdict changes")
            save()
        del pA, pB

    # =================================================================================
    # SOURCE 49
    # =================================================================================
    res["source49"] = dict(L5_note=(
        "NOT RE-CHECKED. Channel 2 at L5 is not on disk and 42's cache stream may not run "
        "while 47 holds MPS (R12/R17). 64 stored L5 proposals (tau 30/40/45), all stored "
        "verdict FAIL."))
    for L in layers:
        perm_mean = 0.0  # 49's own permuted search: 0/20 at every rung, all layers
        pstored = stored49["layers"][str(L)]["permuted_search"]
        res["source49"][f"L{L}_permuted_search_as_stored"] = pstored
        for tau, blkt in stored49["layers"][str(L)]["taus"].items():
            props_raw, gprops, parent, stored_verdict, stored_ring = {}, {}, {}, {}, {}
            skipped = []
            for p in blkt["proposals"]:
                if "token_ids" not in p:
                    # 49 stored no ids for rows that fell below k = 9 under R15 dedup.
                    # They are already NOT ADJUDICABLE for k < 9 and the P1 guard can only
                    # shrink a proposal further, so their verdict cannot change.
                    skipped.append(dict(tokens=p["tokens"], k_raw=p["k_raw"],
                                        k_dedup=p["k_dedup"], verdict=p["verdict"],
                                        reason=p["reason"],
                                        note=("membership not stored as token ids; already "
                                              "NOT ADJUDICABLE at k < 9 after R15, and P1 "
                                              "can only shrink it further -- verdict cannot "
                                              "change")))
                    continue
                mem = sorted(pos[int(t)] for t in p["token_ids"])
                h = s48.mhash(mem)
                props_raw[h] = dict(members=mem, k=len(mem), source=[f"C/L{L}/tau{tau}"])
                stored_verdict[h] = p["verdict"]
                stored_ring[h] = p.get("ring_conjunct_pass")
                g, dropped = guard(mem, dup, key)
                gh = s48.mhash(g) if len(g) >= 2 else f"small:{h}"
                if gh not in gprops:
                    gprops[gh] = dict(members=g, k=len(g))
                parent[h] = gh
            if not props_raw:
                res["source49"][f"L{L}_tau{tau}"] = dict(
                    n_proposals=0, skipped_no_ids=skipped)
                continue
            un = adjudicate_population(s48, S42, props_raw, C1[L], C2[L], cellsL[L], counts,
                                       f"49/L{L}/tau{tau} unguarded", log,
                                       use_tie_class=False)
            ga = adjudicate_population(s48, S42, gprops, C1[L], C2[L], cellsL[L], counts,
                                       f"49/L{L}/tau{tau} guarded", log,
                                       use_tie_class=False)
            passes, p2rows, n_p2 = [], [], 0
            for gh, r in ga["records"].items():
                mem = gprops[gh]["members"]
                if r["verdict"] in ("PASS", "MARGINAL"):
                    passes.append(finish_R14(L, mem, r, perm_mean))
                else:
                    ok, why, sat_cell, ex1 = p2_eligible(r, mem, cellsL[L], s48, S42,
                                                         C1[L])
                    if ok:
                        n_p2 += 1
                        p2rows.append(finish_P2(L, mem, r, perm_mean, why, sat_cell, ex1))
            rows = []
            for h, p in props_raw.items():
                gh = parent[h]
                rows.append(dict(
                    k_raw=p["k"], k_guarded=gprops[gh]["k"],
                    tokens_raw=[strings[i] for i in p["members"]],
                    tokens_guarded=[strings[i] for i in gprops[gh]["members"]],
                    dropped=[strings[i] for i in p["members"]
                             if i not in set(gprops[gh]["members"])],
                    verdict_49_stored=stored_verdict[h],
                    verdict_48tier_unguarded=un["verdicts"][h],
                    verdict_48tier_guarded=ga["verdicts"][gh],
                    ring_conjunct_pass_stored=stored_ring[h]))
            res["source49"][f"L{L}_tau{tau}"] = dict(
                n_proposals=len(props_raw), n_unique_guarded=len(gprops),
                n_changed_by_guard=sum(1 for r in rows if r["dropped"]),
                n_below_kmin=sum(1 for r in rows if r["k_guarded"] < s48.KMIN),
                stored_verdicts_49={k: sum(1 for r in rows if r["verdict_49_stored"] == k)
                                    for k in set(stored_verdict.values())},
                tally_48tier_unguarded=un["tally"], tally_48tier_guarded=ga["tally"],
                guarded_screen_max_rho_by_k={str(k): v for k, v in
                                             sorted(ga["rho_max_k"].items())},
                proposals=rows, skipped_no_ids=skipped,
                R14_rows=passes, n_R14_PASS=sum(1 for r in passes if r["R14_PASS"]),
                n_P2_adjudicated=n_p2, expected_false_P2=0.01 * n_p2,
                P2_rows=p2rows, n_PASS_P2=sum(1 for r in p2rows if r["PASS_P2"]))
            log(f"  [49/L{L}/tau{tau}] {len(props_raw)} proposals, "
                f"{res['source49'][f'L{L}_tau{tau}']['n_changed_by_guard']} guarded, "
                f"{res['source49'][f'L{L}_tau{tau}']['n_R14_PASS']} R14 PASS, "
                f"{n_p2} P2-adjudicated, "
                f"{res['source49'][f'L{L}_tau{tau}']['n_PASS_P2']} PASS_P2")
            save()

    # =================================================================================
    summary = dict(
        strict12_guard_unchanged=res["guard"]["strict12_unchanged"],
        all_R14_PASS=[r for k, b in res["source48"].items() for r in b["R14_rows"]
                      if r["R14_PASS"]]
        + [r for k, b in res["source49"].items() if isinstance(b, dict)
           for r in b.get("R14_rows", []) if r["R14_PASS"]],
        all_PASS_P2=[r for k, b in res["source48"].items() for r in b["P2_rows"]]
        + [r for k, b in res["source49"].items() if isinstance(b, dict)
           for r in b.get("P2_rows", []) if r.get("PASS_P2")],
        n_P2_adjudicated_total=sum(b["n_P2_adjudicated"] for b in res["source48"].values())
        + sum(b.get("n_P2_adjudicated", 0) for b in res["source49"].values()
              if isinstance(b, dict)),
        deviations=doc["declared_rules"]["deviations"])
    summary["expected_false_P2_total"] = 0.01 * summary["n_P2_adjudicated_total"]
    res["summary"] = summary
    doc["status"] = "COMPLETE"
    doc["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    doc["runtime_seconds"] = round(time.time() - t_start, 1)
    save("COMPLETE")
    log(f"=== 48b complete in {doc['runtime_seconds']} s -> {out}")
    logf.close()


if __name__ == "__main__":
    main()
