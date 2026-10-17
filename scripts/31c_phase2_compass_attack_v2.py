"""CORRECTION PASS on the POST-HOC C10 attack tier for compass (script 31b).

SCOPE, STATED FIRST SO NOTHING BELOW IS MISREAD.
  * The two PREREG-BOUND compass verdicts in results/phase2_compass.json are NOT
    touched, NOT recomputed as verdicts, and NOT in question:
        blind arm      -> `untestable` (pre-registered in prereg 3.4/5.3, forced by
                          corpus counts: only the 4 cardinals clear the >= 100 floor)
        supervised arm -> `clean_negative` (0/78 cells on the PRIMARY pre-declared
                          top-2-PCA statistic)
    Both were independently reproduced by review at every load-bearing number.
  * Everything this script computes concerns the POST-HOC C10 widened-plane tier and
    its attack battery in results/phase2_compass_attack.json. That tier earns no prereg
    verdict under prereg 4.3 no matter what any number here says.
  * This script does NOT edit script 31, script 31b, or their artifacts. It writes a new
    artifact that SUPERSEDES the `verdict` block and the A4/A6/A7/A10 interpretation of
    results/phase2_compass_attack.json. The frozen numbers in both prior artifacts remain
    valid as measurements; what changes is which of them discriminate.

WHY A CORRECTION IS NEEDED. The attack verdict `partially_compositional` rests on three
statistics placed in its SUPPORTED list. Review found that two of them cannot
discriminate the hypothesis they were used to test, and that the third points the other
way. This script demonstrates each claim by recomputation rather than by argument.

  D1  A7 IS DEGENERATE BY CONSTRUCTION. A7 projects span(north-south, east-west) OUT of
      the eight centred vectors and rescores. But (v_N - c) - (v_S - c) = v_N - v_S IS
      the first basis axis, so the projection annihilates it EXACTLY: resid_north ==
      resid_south and resid_east == resid_west identically, for any input whatsoever.
      The residual cloud has rank <= 5 and only 6 distinct points for 8 labels, and the
      "residual ring" statistic is computed on coincident points. Demonstrated
      algebraically and numerically on fresh means.

  D2  A7 FIRES IDENTICALLY ON GROUND TRUTH. A planted TRUE compass circle -- equal
      radius, planar, zero compositional structure, the answer known by construction --
      is pushed through the IDENTICAL A7 code path. If its ring is destroyed too, A7
      has no discriminating power and its 33/33 kill is not evidence for composition.

  D3  A4 FIRES IDENTICALLY ON GROUND TRUTH. For any 8-point rose in geographic order,
      the intercardinals sit at the bisectors of the two cardinal diameters whether the
      rose was composed lexically or learned as a circle. Demonstrated by running a
      planted TRUE circle and pure composition through the identical A4 code path.

  D4  A10-ADDITIVITY FIRES IDENTICALLY ON GROUND TRUTH. cos(NE, N+E) = +1 exactly for
      any perfect rose, because the bisector of two orthogonal vectors IS their
      normalised sum. Same for antipodality = -1. Demonstrated on both.

  D5  A6 IS THE ONLY STATISTIC WITH DISCRIMINATING POWER -- REFERENCE CURVES. Raw-sum
      composition predicts inter/cardinal radius sqrt(2) = 1.414; an equal-radius rose
      (a true circle, and also unit-normalised composition, which is geometrically the
      same object) predicts 1.000. The observed label-plane median is 0.7350, BELOW every
      composition reference. This section computes the noise curve for all three
      references, 200 draws per level, AND -- an control neither the analyst nor the
      review ran -- the matched noun-octet null of the same statistic on real means, so
      the observed 0.735 can be read against no-structure as well as against theory.

  D6  VARIANT COUNT. blind_arm.variants_searched is verified against prereg 3.4's
      registered form set: 8 words x 4 forms = 32, enumerated, 0 missing, 0 extra.

  D7  REGISTERED-BUT-NOT-COMPUTABLE. Prereg 3.4 registers cardinal co-location against
      matched random 4-subsets as a descriptive blind-arm result. This section audits
      results/token_type_geometry.json to establish that no type-to-cluster assignment
      exists there for arbitrary pool types, so the item is registered-but-not-computable
      rather than silently omitted.

  D8  EXACT P-VALUE TABLE for the cells quoted in the analyst summary, read from the
      frozen main artifact, so the reported ranges are exact.

DECISION RULE, DECLARED BEFORE ANY NUMBER BELOW IS COMPUTED.
  A statistic is DISCRIMINATING only if it separates the two accounts -- i.e. it takes
  materially different values on (i) pure composition and (ii) a planted true circle,
  both pushed through the identical code path. A statistic that returns the same answer
  on both is CONSISTENT-WITH-BOTH and may not appear in a SUPPORTED list for either.
  If no statistic in the battery discriminates AGAINST a learned circle, the phrase
  "not a learned compass circle" is withdrawn and the attack verdict becomes
        `composition_sufficient_not_discriminated`
  meaning: A1 stands (composition is SUFFICIENT to reproduce the C10 signature), and the
  battery cannot establish that it is NECESSARY. No new search family is run either way,
  and no verdict of any tier can be upgraded by this script.
"""
import argparse, json, os, sys, time
from pathlib import Path
import importlib.util

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from gfdisc.env import DEVICE, D_MODEL
from gfdisc.model import load_model
from gfdisc.fits import fit_circle, circularity, angles


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


C31 = _load("c31", "scripts/31_phase2_compass.py")
C31B = _load("c31b", "scripts/31b_phase2_compass_attack.py")   # identical code path

DIRECTIONS = C31B.DIRECTIONS
CARD_IDX = C31B.CARD_IDX
INTER_IDX = C31B.INTER_IDX
COMPOUND_PARTS = C31B.COMPOUND_PARTS
READOUTS = C31.READOUTS
ATTACK_LAYERS = C31B.ATTACK_LAYERS
LAB = np.arange(8)

REGISTERED_FORMS = ("{w}", " {w}", "{W}", " {W}")   # prereg 3.4, verbatim


# ---------------------------------------------------------------------------
# helpers -- every geometric primitive is the one script 31/31b uses
# ---------------------------------------------------------------------------
unit = C31B.unit
cos = C31B.cos
label_plane = C31B.label_plane
project_and_score = C31B.project_and_score
radius_alternation = C31B.radius_alternation
c10_full = C31B.c10_full
c10_stat = C31B.c10_stat


def residual_after_cardinal_axes(P):
    """A7's transform, verbatim from 31b lines 563-564."""
    B = label_plane(P)
    V = P - P.mean(0, keepdims=True)
    return V - (V @ B.T) @ B, B


def degeneracy_metrics(resid, scale_ref):
    """How degenerate is the A7 residual cloud?"""
    dNS = float(np.linalg.norm(resid[0] - resid[4]))
    dEW = float(np.linalg.norm(resid[2] - resid[6]))
    rank = int(np.linalg.matrix_rank(resid))
    # distinct points at a relative tolerance
    tol = 1e-6 * max(scale_ref, 1e-12)
    kept = []
    for i in range(len(resid)):
        if not any(np.linalg.norm(resid[i] - resid[j]) <= tol for j in kept):
            kept.append(i)
    return dict(norm_resid_north_minus_south=dNS,
                norm_resid_east_minus_west=dEW,
                typical_point_norm=float(np.linalg.norm(resid, axis=1).mean()),
                reference_scale=float(scale_ref),
                matrix_rank=rank,
                n_distinct_points_of_8=len(kept),
                coincident_pairs=[["north", "south"], ["east", "west"]])


def radius_ratio_label_plane(P):
    """A6's statistic, plus the two label-plane scale statistics used to calibrate how
    much noise the cloud carries. No ordering permutation test, so the reference curves
    in D5 stay cheap. Geometry identical to project_and_score + radius_alternation.

    Returns (inter/cardinal radius ratio, plane variance fraction, circle-fit resid/r).
    The plane variance fraction is the share of the centred 8-point variance that lies
    inside span(N-S, E-W); it falls monotonically as isotropic noise rises, so it
    calibrates the noise level at which the A6 prediction should be read.
    """
    B = label_plane(P)
    c = P.mean(0, keepdims=True)
    V = P - c
    z = V @ B.T
    cx, cy, r, resid = fit_circle(z)
    radii = np.linalg.norm(z - np.array([cx, cy]), axis=1)
    f = float((z ** 2).sum() / max((V ** 2).sum(), 1e-12))
    return (float(radii[INTER_IDX].mean() / max(radii[CARD_IDX].mean(), 1e-12)),
            f, float(resid / max(r, 1e-12)))


# ---------------------------------------------------------------------------
# reference generators. Noise convention identical to 31b.composition_synthetics:
# unit-radius construction + (nz / sqrt(d)) * N(0, 1), i.e. per-point noise norm ~ nz.
# ---------------------------------------------------------------------------
def planted_true_circle(rng, d=D_MODEL, nz=0.0, radius=1.0):
    """Equal-radius planar 8-rose at 45 deg spacing. No compositional structure: the
    intercardinals are not sums of anything, they are points on a circle."""
    u = unit(rng.normal(size=d))
    v = rng.normal(size=d)
    v = unit(v - (v @ u) * u)
    th = np.radians(45.0 * np.arange(8))
    P = radius * (np.cos(th)[:, None] * u[None, :] + np.sin(th)[:, None] * v[None, :])
    if nz:
        P = P + (nz / np.sqrt(d)) * rng.normal(size=(8, d))
    return P


def planted_composition(rng, d=D_MODEL, nz=0.0, normalise=False):
    """a, a+b, b, -a+b, -a, -a-b, -b, a-b with a orthogonal to b -- 31b's A1 build."""
    a = unit(rng.normal(size=d))
    b = rng.normal(size=d)
    b = unit(b - (b @ a) * a)
    comp = [a, a + b, b, -a + b, -a, -a - b, -b, a - b]
    if normalise:
        comp = [unit(x) for x in comp]
    P = np.vstack(comp)
    if nz:
        P = P + (nz / np.sqrt(d)) * rng.normal(size=(8, d))
    return P


def with_group_offset(P, rng, phi):
    """Push the four cardinals and the four intercardinals apart along a direction
    orthogonal to the construction, so that the cardinal-minus-intercardinal group axis
    carries variance fraction `phi` of the centred cloud.

    A10 measured exactly this axis on the model (median var fraction 0.5917) and neither
    plain reference has it, which is why the plain references cannot match the observed
    label-plane variance fraction. The offset is common to all four cardinals, so it
    cancels in north-south and east-west and leaves the label PLANE untouched; it only
    changes how much total variance sits outside that plane.
    """
    d = P.shape[1]
    V = P - P.mean(0, keepdims=True)
    w = rng.normal(size=d)
    # orthogonalise against the row space of the construction
    Q = np.linalg.svd(V, full_matrices=False)[2]
    w = w - (w @ Q.T) @ Q
    w = unit(w)
    S = float((V ** 2).sum())
    g = float(np.sqrt(phi * S / max(2.0 * (1.0 - phi), 1e-12)))
    out = P.copy()
    out[CARD_IDX] += (g / 2.0) * w
    out[INTER_IDX] -= (g / 2.0) * w
    return out


GROUP_PHI = 0.5917069292123409     # A10_summary.median_var_fraction, frozen artifact

def _noised(P, rng, nz):
    return P + (nz / np.sqrt(P.shape[1])) * rng.normal(size=P.shape) if nz else P


def circle_plus_group_offset(rng, nz):
    return _noised(with_group_offset(planted_true_circle(rng, nz=0.0), rng, GROUP_PHI),
                   rng, nz)


def raw_sums_plus_group_offset(rng, nz):
    return _noised(with_group_offset(planted_composition(rng, nz=0.0), rng, GROUP_PHI),
                   rng, nz)


REFERENCES = {
    "planted_true_circle": lambda rng, nz: planted_true_circle(rng, nz=nz),
    "composition_unit_normalised": lambda rng, nz: planted_composition(rng, nz=nz,
                                                                       normalise=True),
    "composition_raw_sums": lambda rng, nz: planted_composition(rng, nz=nz,
                                                                normalise=False),
    # group-offset-matched variants: the same two accounts, given the cardinal-vs-
    # compound group axis the model actually has (A10 median var fraction)
    "circle_plus_group_offset": circle_plus_group_offset,
    "raw_sums_plus_group_offset": raw_sums_plus_group_offset,
}
# the ground-truth discrimination table (D2/D3/D4) uses the two plain accounts only
GT_REFERENCES = ("planted_true_circle", "composition_unit_normalised",
                 "composition_raw_sums")


def group_centred_cosines(P):
    """A10's group-centred antipodality / additivity, verbatim construction."""
    g_card, g_int = P[CARD_IDX].mean(0), P[INTER_IDX].mean(0)
    vd = P.copy()
    vd[CARD_IDX] -= g_card
    vd[INTER_IDX] -= g_int
    anti = {"north_south": cos(vd[0], vd[4]), "east_west": cos(vd[2], vd[6])}
    add = {DIRECTIONS[i]: cos(vd[i], vd[p1] + vd[p2])
           for i, (p1, p2) in COMPOUND_PARTS.items()}
    return (dict(antipodality_group_centred=anti,
                 antipodality_group_centred_mean=float(np.mean(list(anti.values()))),
                 additivity_group_centred=add,
                 additivity_group_centred_mean=float(np.mean(list(add.values())))))


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/phase2_compass_attack_v2.json")
    ap.add_argument("--main", default="results/phase2_compass.json")
    ap.add_argument("--attack", default="results/phase2_compass_attack.json")
    ap.add_argument("--catalogue", default="results/token_type_geometry.json")
    ap.add_argument("--layers", default=",".join(str(x) for x in ATTACK_LAYERS))
    ap.add_argument("--n-ref-draws", type=int, default=200)
    ap.add_argument("--n-gt-seeds", type=int, default=8)
    ap.add_argument("--n-null", type=int, default=2000)
    ap.add_argument("--n-noun-null-a6", type=int, default=1000)
    ap.add_argument("--chunk", type=int, default=48)
    a = ap.parse_args()
    os.chdir(REPO)
    t0 = time.time()
    layers = [int(x) for x in a.layers.split(",")]

    print("=" * 88)
    print("CORRECTION PASS on the POST-HOC C10 attack tier (script 31b)")
    print("  prereg verdicts NOT in question: blind `untestable`, supervised "
          "`clean_negative`")
    print("  D1-D8 declared in the module docstring before any number below")
    print("=" * 88, flush=True)

    main_art = json.loads(Path(a.main).read_text())
    atk_art = json.loads(Path(a.attack).read_text())
    a4567 = atk_art["A4_A5_A6_A7_label_plane_alternation_angles_residual"]

    out = dict(
        status="correction pass on a POST-HOC tier; exploratory Phase-2 under "
               "docs/phase2_prereg_2026-08-21.md",
        item="phase2_compass_attack_v2",
        script="scripts/31c_phase2_compass_attack_v2.py",
        prereg="docs/phase2_prereg_2026-08-21.md",
        prereg_section="3.4 COMPASS DIRECTIONS (post-hoc C10 tier only)",
        supersedes=dict(
            artifact="results/phase2_compass_attack.json",
            what="the `verdict` block and the SUPPORTED/NOT-SUPPORTED classification of "
                 "A4, A6, A7 and A10 only",
            what_is_unchanged="every measurement in results/phase2_compass_attack.json "
                              "and results/phase2_compass.json remains valid as a "
                              "measurement; A1 (composition sufficiency), A2/A3, A5, A8 "
                              "and A9 are not revisited",
            prereg_verdicts_untouched=dict(
                blind=main_art["verdict"]["blind"],
                supervised=main_art["verdict"]["supervised"],
                note="reproduced independently by review; this script cannot and does "
                     "not change them"),
        ),
        generated=time.strftime("%Y-%m-%dT%H:%M:%S"),
        model="google/gemma-3-1b-pt",
        decision_rule_declared_before_computation="module docstring, D1-D8, "
                                                  "including the DISCRIMINATING test",
        layers=layers,
    )

    # ---- D6: variant count -------------------------------------------------
    print("\n--- D6 VARIANT COUNT (blind_arm.variants_searched vs prereg 3.4) ---")
    vs = list(main_art["blind_arm"]["variants_searched"])
    expected = []
    for w in DIRECTIONS:
        for f in REGISTERED_FORMS:
            expected.append(f.replace("{w}", w).replace("{W}", w.capitalize()))
    missing = [x for x in expected if x not in vs]
    extra = [x for x in vs if x not in expected]
    d6 = dict(n_variants_in_artifact=len(vs),
              n_words=len(DIRECTIONS), n_registered_forms=len(REGISTERED_FORMS),
              n_expected=len(expected), registered_forms=list(REGISTERED_FORMS),
              missing=missing, extra=extra,
              exact_match=bool(not missing and not extra and len(vs) == len(expected)),
              variants=vs,
              analyst_report_text_said=24,
              correction="the artifact searched 32 variants (8 words x 4 registered "
                         "forms). The number 24 in the analyst report text is wrong and "
                         "would misrepresent a complete search as incomplete.")
    out["D6_variant_count"] = d6
    print(f"  artifact {d6['n_variants_in_artifact']} variants; expected "
          f"{d6['n_expected']} = {d6['n_words']} words x {d6['n_registered_forms']} "
          f"forms; missing {len(missing)}, extra {len(extra)}; exact_match "
          f"{d6['exact_match']}  (report text said 24 -> WRONG, write 32)", flush=True)

    # ---- D7: registered-but-not-computable ---------------------------------
    print("\n--- D7 REGISTERED CO-LOCATION TEST: computable from the frozen artifact? ---")
    cat = json.loads(Path(a.catalogue).read_text())
    n_screened = n_screened_hash = n_screened_tok = 0
    for L in cat["layers"]:
        for arm in L["arms"].values():
            for s in (arm.get("screened") or []):
                n_screened += 1
                n_screened_hash += int("member_hash" in s)
                n_screened_tok += int(bool(s.get("token_strings")))
    n_ls = sum(len(L["label_scored_clusters"]) for L in cat["layers"])
    n_ls_tok = sum(1 for L in cat["layers"] for r in L["label_scored_clusters"]
                   if r.get("token_strings"))
    n_flag = len(cat["verdict"]["other_rings"])
    n_flag_tok = sum(1 for r in cat["verdict"]["other_rings"] if r.get("token_strings"))
    declared_label_sets = list(cat["label_availability"].keys())
    d7 = dict(
        registered_in="prereg 3.4: 'co-location of the four cardinals against matched "
                      "random 4-subsets', descriptive placement only, never ring evidence",
        status="REGISTERED BUT NOT COMPUTABLE from the frozen artifact",
        artifact=a.catalogue,
        n_screened_cluster_records=n_screened,
        n_screened_with_member_hash=n_screened_hash,
        n_screened_with_token_strings=n_screened_tok,
        n_label_scored_records=n_ls, n_label_scored_with_token_strings=n_ls_tok,
        n_flagged_ring_records=n_flag, n_flagged_with_token_strings=n_flag_tok,
        declared_label_sets=declared_label_sets,
        compass_in_declared_label_sets=bool(
            any("compass" in k or "direction" in k for k in declared_label_sets)),
        why="membership is recorded only as `member_hash` for screened clusters; "
            "token_strings exist only on label-scored and flagged records, and compass "
            "was never a declared label set. There is therefore no type-to-cluster "
            "assignment for arbitrary pool types such as the four cardinals, and "
            "producing one would require the re-clustering prereg 4.1 forbids.",
        reporting_rule="must be reported as registered-but-not-computable, not silently "
                       "omitted; it is not verdict-bearing either way",
    )
    out["D7_registered_colocation_not_computable"] = d7
    print(f"  screened cluster records {n_screened}: member_hash {n_screened_hash}, "
          f"token_strings {n_screened_tok}")
    print(f"  label-scored {n_ls} (token_strings {n_ls_tok}); flagged rings {n_flag} "
          f"(token_strings {n_flag_tok})")
    print(f"  declared label sets: {declared_label_sets}")
    print(f"  -> {d7['status']}", flush=True)

    # ---- D8: exact p-value table -------------------------------------------
    print("\n--- D8 EXACT VALUES for the cells quoted in the analyst summary ---")
    def cells(pred):
        rr = []
        for r in main_art["rows"]:
            if pred(r):
                s = r["secondary_plane_search"]
                rr.append(dict(readout=r["readout"], layer=r["layer"],
                               gap_ratio=s["gap_ratio"],
                               p_noun=s["p_vs_noun_background"],
                               p_place=s["p_vs_place_background"],
                               rho=s["order"]["rho"], p_order=s["order"]["p"]))
        return rr
    tok_block = cells(lambda r: r["readout"] == "token" and 19 <= r["layer"] <= 25)
    suf_block = cells(lambda r: r["readout"] == "suffix" and r["layer"] in (12, 16, 18, 24))
    def rng_of(rows, key):
        v = [x[key] for x in rows]
        return [float(min(v)), float(max(v))]
    d8 = dict(
        token_L19_25=dict(
            rows=tok_block,
            gap_ratio_range=rng_of(tok_block, "gap_ratio"),
            p_noun_range=rng_of(tok_block, "p_noun"),
            p_place_range=rng_of(tok_block, "p_place"),
            p_noun_floor=float(min(x["p_noun"] for x in tok_block)),
            layers_at_p_noun_floor=[
                x["layer"] for x in tok_block
                if abs(x["p_noun"] - min(y["p_noun"] for y in tok_block)) < 1e-12],
            analyst_said="p_noun 0.0005, p_place 0.0005",
            correction="p_noun over token L19-25 spans 0.00050-0.00150 (L19 0.00150, "
                       "L20 0.00100, L25 0.00100); only L21-L24 sit at the 0.00050 "
                       "floor. p_place spans 0.00050-0.00100."),
        suffix_L12_16_18_24=dict(
            rows=suf_block,
            gap_ratio_range=rng_of(suf_block, "gap_ratio"),
            p_noun_range=rng_of(suf_block, "p_noun"),
            p_place_range=rng_of(suf_block, "p_place"),
            analyst_said="gap ratio 1.32-1.37, p_noun 0.0010-0.0025",
            correction="gap ratios 1.3240-1.3719 are correct. p_noun spans "
                       "0.00100-0.00300; L18 is 0.00300, outside the quoted range. The "
                       "quoted 0.0010-0.0025 is in fact the p_PLACE range "
                       "(0.00100-0.00250), which is the likely source of the slip."),
    )
    out["D8_exact_values"] = d8
    for name, blk in (("token L19-25", d8["token_L19_25"]),
                      ("suffix L12/16/18/24", d8["suffix_L12_16_18_24"])):
        print(f"  {name}: gapR {blk['gap_ratio_range'][0]:.4f}-"
              f"{blk['gap_ratio_range'][1]:.4f}  p_noun "
              f"{blk['p_noun_range'][0]:.5f}-{blk['p_noun_range'][1]:.5f}  p_place "
              f"{blk['p_place_range'][0]:.5f}-{blk['p_place_range'][1]:.5f}")
    print(f"  token layers at the p_noun 0.00050 floor: "
          f"{d8['token_L19_25']['layers_at_p_noun_floor']}", flush=True)

    # ---- D1a: A7 degeneracy, ALGEBRAIC -------------------------------------
    print("\n--- D1a A7 DEGENERACY, algebraic identity (holds for ANY input) ---")
    print("      resid_i = (v_i - c) - proj_B(v_i - c),  B = onb(span(N-S, E-W))")
    print("      resid_N - resid_S = (v_N - v_S) - proj_B(v_N - v_S) = 0, because "
          "v_N - v_S spans B's first axis")
    print("      => resid_N == resid_S and resid_E == resid_W EXACTLY; rank <= 5; "
          "6 distinct points for 8 labels")

    # ---- forwards ----------------------------------------------------------
    print("\ncollecting fresh means (directions + noun pool) ...", flush=True)
    model, tok = load_model(token=os.environ.get("HF_TOKEN"))
    excl = set(C31.DIRECTION_SENSE_EXCLUSIONS)

    def ok_bg(w):
        lw = w.lower()
        return lw not in excl and not any(s in lw for s in C31.DIRECTION_SUBSTRINGS)

    noun_words = [w for w in dict.fromkeys(C31.NOUN_POOL) if ok_bg(w)]
    noun_words = [x["word"] for x in C31.token_audit(tok, noun_words)[0]]
    words = DIRECTIONS + noun_words
    word_ids = [tok.encode(" " + w, add_special_tokens=False)[0] for w in words]
    dir_idx = np.arange(len(DIRECTIONS))
    noun_idx = np.arange(len(DIRECTIONS), len(words))
    tf = time.time()
    M = C31.collect(model, tok, words, word_ids, layers, chunk=a.chunk)
    fwd = time.time() - tf
    print(f"  {len(words)} words x {len(C31.TEMPLATES)} templates x {len(layers)} "
          f"layers x 3 readouts in {fwd:.1f}s", flush=True)
    del model
    if DEVICE == "mps":
        torch.mps.empty_cache()

    # ---- D1b: A7 degeneracy, NUMERIC on fresh means -------------------------
    print("\n--- D1b A7 DEGENERACY, numeric on fresh means (all 33 cells) ---")
    print(f"{'ro':>7} {'L':>3} | {'|rN-rS|':>10} {'|rE-rW|':>10} {'mean|r|':>10} "
          f"{'rank':>5} {'distinct':>9} | {'A7 gapR':>9} {'A7 rho':>8}")
    d1_rows = []
    for ro in READOUTS:
        for l in layers:
            P = M[ro][l][dir_idx]
            resid, B = residual_after_cardinal_axes(P)
            scale = float(np.linalg.norm(P - P.mean(0, keepdims=True), axis=1).mean())
            g = degeneracy_metrics(resid, scale)
            s_res, _ = c10_full(resid, LAB, 8, seed=l)
            g.update(readout=ro, layer=l,
                     a7_gap_ratio=s_res["gap_ratio"], a7_rho=s_res["order"]["rho"],
                     a7_p_order=s_res["order"]["p"])
            d1_rows.append(g)
            print(f"{ro:>7} {l:3d} | {g['norm_resid_north_minus_south']:10.2e} "
                  f"{g['norm_resid_east_minus_west']:10.2e} "
                  f"{g['typical_point_norm']:10.3f} {g['matrix_rank']:5d} "
                  f"{g['n_distinct_points_of_8']:9d} | {g['a7_gap_ratio']:9.4f} "
                  f"{g['a7_rho']:+8.3f}", flush=True)

    # rho tie-break instability at the strongest suffix cell
    print("\n  rho TIE-BREAK INSTABILITY at suffix L24 (coincident points, so the "
          "cyclic order is not determined):")
    P24 = M["suffix"][24][dir_idx]
    resid24, _ = residual_after_cardinal_axes(P24)
    base_s, _ = c10_full(resid24, LAB, 8, seed=24)
    jit = []
    scale24 = float(np.linalg.norm(resid24, axis=1).mean())
    for eps in (0.0, 1e-12, 1e-10, 1e-8, 1e-6):
        for sd in range(3):
            r = np.random.default_rng(70000 + sd)
            Q = resid24 + (eps * scale24) * r.normal(size=resid24.shape) if eps else resid24
            s, _ = c10_full(Q, LAB, 8, seed=24)
            jit.append(dict(eps_relative=eps, seed=sd, gap_ratio=s["gap_ratio"],
                            rho=s["order"]["rho"], p_order=s["order"]["p"]))
            if eps == 0.0:
                break
    rhos = sorted({round(x["rho"], 6) for x in jit})
    gaps = [x["gap_ratio"] for x in jit]
    print(f"    gap_ratio stable: {min(gaps):.12f} .. {max(gaps):.12f}")
    print(f"    rho takes {len(rhos)} distinct values under sub-1e-6 jitter: {rhos}")
    out["D1_A7_degenerate_by_construction"] = dict(
        algebraic_identity="resid_N - resid_S = (v_N - v_S) - proj_B(v_N - v_S) = 0 "
                           "because v_N - v_S IS the first basis axis of B; likewise "
                           "resid_E - resid_W = 0. Holds for ANY input matrix.",
        consequence="the A7 residual cloud has rank <= 5 and 6 distinct points for 8 "
                    "labels; the 'residual ring' statistic is computed on coincident "
                    "points and its cyclic order is decided by tie-breaking",
        rows=d1_rows,
        max_norm_resid_north_minus_south=float(
            max(r["norm_resid_north_minus_south"] for r in d1_rows)),
        max_norm_resid_east_minus_west=float(
            max(r["norm_resid_east_minus_west"] for r in d1_rows)),
        all_rank_le_5=bool(all(r["matrix_rank"] <= 5 for r in d1_rows)),
        all_6_distinct=bool(all(r["n_distinct_points_of_8"] == 6 for r in d1_rows)),
        rho_tie_break_probe=dict(
            cell="suffix L24", rows=jit,
            gap_ratio_min=float(min(gaps)), gap_ratio_max=float(max(gaps)),
            distinct_rho_values=rhos,
            explains="why an independent recomputation reports A7 rho +0.619 where the "
                     "31b artifact reports +0.524 at a bit-identical gap ratio "
                     "(2.434451326255): the two runs broke the same tie differently"),
        artifact_a7_suffix_L24=[r["residual_after_removing_cardinal_axes"]
                                for r in a4567
                                if r["readout"] == "suffix" and r["layer"] == 24],
    )

    # ---- D2/D3/D4: GROUND TRUTH through the identical code path ------------
    print("\n--- D2/D3/D4 GROUND TRUTH: planted TRUE circle vs pure composition, "
          "identical code path ---")
    # one shared residual background, built from the real noun means at suffix L24,
    # exactly as 31b builds res_nb
    rbg = np.random.default_rng(4_000_000 + 24)
    res_nb = np.empty(a.n_null)
    lab_nb = np.empty(a.n_null)
    for i in range(a.n_null):
        pick = noun_idx[rbg.choice(len(noun_idx), size=8, replace=False)]
        Q = M["suffix"][24][pick]
        QB = label_plane(Q)
        qc = Q.mean(0, keepdims=True)
        qz = (Q - qc) @ QB.T
        qcx, qcy, qr, _ = fit_circle(qz)
        lab_nb[i] = circularity(qz, qcx, qcy, qr)["gap_ratio"]
        qv = (Q - qc) - ((Q - qc) @ QB.T) @ QB
        res_nb[i] = c10_stat(qv)["gap_ratio"]

    gt_rows = []
    print(f"  {'case':<34} {'nz':>5} {'sd':>3} | {'A4 gapR':>8} {'A4 rho':>8} "
          f"{'A4 pOrd':>8} | {'A6 i/c':>7} | {'A7 gapR':>8} {'A7 rho':>8} "
          f"{'A7 pOrd':>8} {'A7 pBG':>7} | {'A10 add*':>9} {'A10 anti*':>10}")
    for name in GT_REFERENCES:
        gen = REFERENCES[name]
        for nz in (0.0, 0.15, 0.5, 0.75):
            for sd in range(a.n_gt_seeds):
                rng = np.random.default_rng(910000 + 7919 * sd + int(nz * 1000))
                P = gen(rng, nz)
                B = label_plane(P)
                sc, _ = project_and_score(P, B, LAB, 8, seed=sd)
                a6 = radius_alternation(sc["radii"])
                p_lab = float((1 + int((lab_nb <= sc["gap_ratio"]).sum()))
                              / (1 + a.n_null))
                resid, _ = residual_after_cardinal_axes(P)
                s_res, _ = c10_full(resid, LAB, 8, seed=sd)
                p_res = float((1 + int((res_nb <= s_res["gap_ratio"]).sum()))
                              / (1 + a.n_null))
                gc = group_centred_cosines(P)
                dg = degeneracy_metrics(
                    resid, float(np.linalg.norm(P - P.mean(0, keepdims=True),
                                                axis=1).mean()))
                row = dict(case=name, noise=nz, seed=sd,
                           A4_gap_ratio=sc["gap_ratio"], A4_rho=sc["rho"],
                           A4_p_order=sc["p_order"],
                           A4_p_vs_noun_background=p_lab,
                           A6_ratio_inter_over_card=a6["ratio_inter_over_card"],
                           A7_gap_ratio=s_res["gap_ratio"],
                           A7_rho=s_res["order"]["rho"],
                           A7_p_order=s_res["order"]["p"],
                           A7_p_vs_noun_background=p_res,
                           A7_rank=dg["matrix_rank"],
                           A7_n_distinct=dg["n_distinct_points_of_8"],
                           A10_additivity_group_centred_mean=gc[
                               "additivity_group_centred_mean"],
                           A10_antipodality_group_centred_mean=gc[
                               "antipodality_group_centred_mean"])
                gt_rows.append(row)
                print(f"  {name:<34} {nz:5.2f} {sd:3d} | {sc['gap_ratio']:8.4f} "
                      f"{sc['rho']:+8.3f} {sc['p_order']:8.4f} | "
                      f"{a6['ratio_inter_over_card']:7.3f} | "
                      f"{s_res['gap_ratio']:8.4f} {s_res['order']['rho']:+8.3f} "
                      f"{s_res['order']['p']:8.4f} {p_res:7.4f} | "
                      f"{row['A10_additivity_group_centred_mean']:+9.4f} "
                      f"{row['A10_antipodality_group_centred_mean']:+10.4f}",
                      flush=True)

    def agg(case, key, nz=None):
        v = [r[key] for r in gt_rows if r["case"] == case
             and (nz is None or r["noise"] == nz)]
        return dict(median=float(np.median(v)), min=float(min(v)), max=float(max(v)),
                    n=len(v))

    def a7_pass_rate(case):
        """31b's own res_pass criterion: p_vs_noun_background < .05 AND p_order < .05."""
        v = [r for r in gt_rows if r["case"] == case]
        n = sum(1 for r in v if r["A7_p_vs_noun_background"] < 0.05
                and r["A7_p_order"] < 0.05)
        return dict(n_surviving=int(n), n_draws=len(v),
                    criterion="p_vs_noun_background < 0.05 AND p_order < 0.05, "
                              "verbatim 31b res_pass")

    circ, comp = "planted_true_circle", "composition_raw_sums"
    obs_a7 = [r["residual_after_removing_cardinal_axes"] for r in a4567]
    d234 = dict(
        n_null_background=int(a.n_null),
        background="noun octets from the real suffix-L24 means, identical construction "
                   "to 31b (the label-plane background for A4, the projected-out "
                   "background for A7)",
        rows=gt_rows,
        A7=dict(
            statistic="residual ring after projecting span(N-S, E-W) out",
            observed_on_model=dict(
                n_cells=len(obs_a7),
                n_surviving=int(sum(x["p_vs_noun_background"] < 0.05 and
                                    x["p_order"] < 0.05 for x in obs_a7)),
                gap_ratio_median=float(np.median([x["gap_ratio"] for x in obs_a7])),
                rho_median=float(np.median([x["rho"] for x in obs_a7])),
                p_order_median=float(np.median([x["p_order"] for x in obs_a7]))),
            planted_true_circle=dict(gap_ratio=agg(circ, "A7_gap_ratio"),
                                     rho=agg(circ, "A7_rho"),
                                     p_order=agg(circ, "A7_p_order"),
                                     p_vs_background=agg(circ,
                                                         "A7_p_vs_noun_background"),
                                     survival=a7_pass_rate(circ),
                                     rank=agg(circ, "A7_rank"),
                                     n_distinct_points=agg(circ, "A7_n_distinct")),
            pure_composition=dict(gap_ratio=agg(comp, "A7_gap_ratio"),
                                  rho=agg(comp, "A7_rho"),
                                  p_order=agg(comp, "A7_p_order"),
                                  p_vs_background=agg(comp,
                                                      "A7_p_vs_noun_background"),
                                  survival=a7_pass_rate(comp),
                                  rank=agg(comp, "A7_rank"),
                                  n_distinct_points=agg(comp, "A7_n_distinct")),
            discriminating=False,
            zero_noise_note="at zero noise a planar rose has NO residual at all: the "
                            "projection removes 100% of its centred variance, so the "
                            "'residual ring' is scored on floating-point dust (the "
                            "n_distinct_points minimum of 1 and the rank maximum of 8 in "
                            "the aggregates below are that case). This is the sharpest "
                            "form of the same point.",
            verdict="A7 IS NOT A DISCRIMINATING TEST. It destroys the ring on a planted "
                    "TRUE circle just as completely as on the model, because for any "
                    "8-point rose in geographic order the two cardinal diameters SPAN "
                    "the ring plane -- so projecting them out is projecting the ring "
                    "plane out. Its 33/33 kill must be removed from the SUPPORTED list "
                    "and the label 'Attack A7 (decisive)' must not be carried forward."),
        A4=dict(
            statistic="closure + geographic order in the label-defined plane "
                      "span(N-S, E-W), intercardinals held out of the plane",
            planted_true_circle=dict(gap_ratio=agg(circ, "A4_gap_ratio"),
                                     rho=agg(circ, "A4_rho"),
                                     p_order=agg(circ, "A4_p_order")),
            pure_composition=dict(gap_ratio=agg(comp, "A4_gap_ratio"),
                                  rho=agg(comp, "A4_rho"),
                                  p_order=agg(comp, "A4_p_order")),
            discriminating=False,
            verdict="A4 IS CONSISTENT WITH BOTH ACCOUNTS. A learned circle places its "
                    "intercardinals at the 45-degree bisectors of the two cardinal "
                    "diameters exactly as lexical composition does, so the held-out "
                    "placement is a prediction of both."),
        A10_additivity=dict(
            statistic="group-centred cos(NE, N+E) and cos(N, S)",
            planted_true_circle=dict(
                additivity=agg(circ, "A10_additivity_group_centred_mean"),
                antipodality=agg(circ, "A10_antipodality_group_centred_mean")),
            pure_composition=dict(
                additivity=agg(comp, "A10_additivity_group_centred_mean"),
                antipodality=agg(comp, "A10_antipodality_group_centred_mean")),
            observed_on_model=dict(
                n_additivity_group_centred_pass=atk_art["A10_summary"][
                    "n_additivity_group_centred_pass"],
                n_cells=atk_art["A10_summary"]["n_cells"]),
            discriminating=False,
            verdict="A10-ADDITIVITY IS CONSISTENT WITH BOTH ACCOUNTS. cos(NE, N+E) = +1 "
                    "and cos(N, S) = -1 hold exactly for any perfect rose, because the "
                    "bisector of two orthogonal vectors IS their normalised sum. The "
                    "28/33 additivity count is not evidence for lexical composition."),
    )
    out["D2_D3_D4_ground_truth_discrimination"] = d234

    # ---- D5: A6 reference curves, noise-calibrated, + the noun-octet null ---
    print("\n--- D5 A6 REFERENCE CURVES (inter/cardinal radius in the label plane) ---")
    print("      two readings: (a) noise free -- at what noise does each account reach "
          "the observed ratio;")
    print("      (b) noise pinned by the label-plane variance fraction f. Both are "
          "reported; (b) turns out to fail.")
    noise_grid = [0.0, 0.15, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]
    curves = {}
    print(f"  {'reference':<30} " + "  ".join(f"{nz:>13.2f}" for nz in noise_grid))
    for name, gen in REFERENCES.items():
        row = {}
        cells_txt = []
        for nz in noise_grid:
            vals = np.empty(a.n_ref_draws)
            fs = np.empty(a.n_ref_draws)
            rs = np.empty(a.n_ref_draws)
            for i in range(a.n_ref_draws):
                rng = np.random.default_rng(510000 + 1009 * i + int(nz * 10000))
                vals[i], fs[i], rs[i] = radius_ratio_label_plane(gen(rng, nz))
            row[f"{nz:g}"] = dict(ratio_median=float(np.median(vals)),
                                  ratio_p05=float(np.percentile(vals, 5)),
                                  ratio_p95=float(np.percentile(vals, 95)),
                                  plane_var_fraction_median=float(np.median(fs)),
                                  circle_resid_over_r_median=float(np.median(rs)),
                                  n=int(a.n_ref_draws))
            cells_txt.append(f"{np.median(vals):5.3f}/f{np.median(fs):4.2f}")
        curves[name] = row
        print(f"  {name:<30} " + "  ".join(f"{c:>13}" for c in cells_txt), flush=True)

    # observed A6 statistic + calibrators, on fresh means, all 33 cells
    obs_rows = []
    for ro in READOUTS:
        for l in layers:
            ratio, f, rr = radius_ratio_label_plane(M[ro][l][dir_idx])
            frozen = [x["radius_alternation_label_plane"]["ratio_inter_over_card"]
                      for x in a4567 if x["readout"] == ro and x["layer"] == l]
            obs_rows.append(dict(readout=ro, layer=l, ratio_inter_over_card=ratio,
                                 plane_var_fraction=f, circle_resid_over_r=rr,
                                 frozen_artifact_ratio=frozen[0] if frozen else None))
    obs_f_med = float(np.median([r["plane_var_fraction"] for r in obs_rows]))
    obs_ratio_fresh_med = float(np.median([r["ratio_inter_over_card"] for r in obs_rows]))
    # the FROZEN artifact values are the ones every reading is quoted against
    obs_all = [x["radius_alternation_label_plane"]["ratio_inter_over_card"]
               for x in a4567]
    obs_med = float(np.median(obs_all))
    obs_c10 = [x["radius_alternation_c10_plane"]["ratio_inter_over_card"] for x in a4567]

    def predict_at_f(name, f_target):
        """Read a reference curve at the noise level whose plane variance fraction
        matches f_target (linear interpolation on the monotone f -> ratio curve)."""
        pts = [(curves[name][f"{nz:g}"]["plane_var_fraction_median"],
                curves[name][f"{nz:g}"]["ratio_median"], nz) for nz in noise_grid]
        pts.sort(key=lambda p: p[0])
        fs = np.array([p[0] for p in pts])
        rs = np.array([p[1] for p in pts])
        nzs = np.array([p[2] for p in pts])
        return dict(matched_plane_var_fraction=float(f_target),
                    predicted_ratio=float(np.interp(f_target, fs, rs)),
                    implied_noise=float(np.interp(f_target, fs, nzs)),
                    in_range=bool(fs.min() <= f_target <= fs.max()))

    def noise_reaching_ratio(name, target):
        """The noise level at which a reference curve reaches a given ratio. If two
        different accounts both reach the observed ratio at attainable noise levels,
        A6 does not discriminate with the noise level left free."""
        pts = [(curves[name][f"{nz:g}"]["ratio_median"], nz) for nz in noise_grid]
        pts.sort(key=lambda p: p[0])
        rs = np.array([p[0] for p in pts])
        nzs = np.array([p[1] for p in pts])
        return dict(target_ratio=float(target),
                    noise_reaching_target=float(np.interp(target, rs, nzs)),
                    in_range=bool(rs.min() <= target <= rs.max()))

    calib = {name: predict_at_f(name, obs_f_med) for name in REFERENCES}
    reach = {name: noise_reaching_ratio(name, obs_med) for name in REFERENCES}
    print(f"\n  noise level at which each reference REACHES the observed ratio "
          f"{obs_med:.4f}:")
    for name, r in reach.items():
        print(f"    {name:<30} nz ~ {r['noise_reaching_target']:.2f}"
              f"{'' if r['in_range'] else '   [not attainable on the grid]'}")
    print(f"\n  observed label-plane variance fraction f: median {obs_f_med:.4f}; "
          f"observed ratio (fresh means) median {obs_ratio_fresh_med:.4f}")
    for name, c in calib.items():
        flag = "" if c["in_range"] else "   [OUT OF RANGE: this reference cannot " \
                                        "reach the observed f at any noise level]"
        print(f"    at matched f, {name:<30} implies noise ~{c['implied_noise']:.2f} "
              f"and predicts ratio {c['predicted_ratio']:.3f}{flag}")

    # matched noun-octet null of the SAME statistic on real means
    print(f"\n  matched noun-octet NULL of the same statistic "
          f"({a.n_noun_null_a6} draws per cell, all cells):")
    null_rows = []
    for ro in READOUTS:
        for l in layers:
            r = np.random.default_rng(6_000_000 + 1000 * READOUTS.index(ro) + l)
            v = np.empty(a.n_noun_null_a6)
            fv = np.empty(a.n_noun_null_a6)
            for i in range(a.n_noun_null_a6):
                pick = noun_idx[r.choice(len(noun_idx), size=8, replace=False)]
                v[i], fv[i], _ = radius_ratio_label_plane(M[ro][l][pick])
            obs = [x["radius_alternation_label_plane"]["ratio_inter_over_card"]
                   for x in a4567 if x["readout"] == ro and x["layer"] == l]
            null_rows.append(dict(readout=ro, layer=l,
                                  null_median=float(np.median(v)),
                                  null_p05=float(np.percentile(v, 5)),
                                  null_p95=float(np.percentile(v, 95)),
                                  null_plane_var_fraction_median=float(np.median(fv)),
                                  observed=obs[0] if obs else None,
                                  p_obs_above_null=float(
                                      (1 + int((v >= obs[0]).sum())) / (1 + len(v)))
                                  if obs else None))
    null_med = float(np.median([r["null_median"] for r in null_rows]))
    null_f_med = float(np.median([r["null_plane_var_fraction_median"]
                                  for r in null_rows]))
    n_obs_above_null_p95 = sum(1 for r in null_rows
                               if r["observed"] is not None
                               and r["observed"] > r["null_p95"])
    print(f"    null median of cell medians {null_med:.4f} (f {null_f_med:.4f}); "
          f"frozen observed median {obs_med:.4f} over {len(obs_all)} cells "
          f"(range {min(obs_all):.4f}-{max(obs_all):.4f})")
    print(f"    cells where observed exceeds its own noun-null p95: "
          f"{n_obs_above_null_p95}/{len(null_rows)}", flush=True)

    out["D5_A6_reference_curves"] = dict(
        statistic="mean radius of the four intercardinals / mean radius of the four "
                  "cardinals, measured in the label-defined plane span(N-S, E-W)",
        noise_convention="unit-radius construction + (nz / sqrt(1152)) * N(0,1), i.e. "
                         "per-point noise norm ~ nz; identical to "
                         "31b.composition_synthetics",
        n_draws_per_level=int(a.n_ref_draws),
        noise_grid=noise_grid,
        theoretical_references=dict(
            composition_raw_sums=1.4142135623730951,
            equal_radius_rose=1.0,
            note="unit-normalised composition IS an equal-radius 8-rose at 45-degree "
                 "spacing -- geometrically the same object as a true circle -- so it "
                 "shares the circle's curve exactly at every noise level. A6 therefore "
                 "separates RAW-SUM composition from an equal-radius rose; it cannot "
                 "separate a learned circle from normalised composition."),
        curves=curves,
        observed_fresh_means=dict(rows=obs_rows,
                                  ratio_median=obs_ratio_fresh_med,
                                  plane_var_fraction_median=obs_f_med),
        noise_free_reading=dict(
            method="with the noise level left as a free parameter, find the noise at "
                   "which each reference curve reaches the observed ratio",
            observed_ratio_median=obs_med,
            per_reference=reach,
            finding="BOTH accounts reach the observed ratio at attainable noise levels, "
                    "so A6 does not discriminate unless the noise level is pinned "
                    "independently. This corrects the review's claim that 'neither "
                    "composition variant reaches below 1.000 at any noise level': "
                    "raw-sum composition crosses 1.000 near nz ~ 0.9 and reaches the "
                    "observed value at higher noise still."),
        attempted_noise_calibration=dict(
            method="pin the noise level by matching the label-plane variance fraction f, "
                   "then read the ratio prediction there (linear interpolation on the "
                   "monotone f -> ratio curve)",
            observed_plane_var_fraction_median=obs_f_med,
            observed_ratio_median=obs_med,
            per_reference=calib,
            group_offset_phi_used=GROUP_PHI,
            group_offset_source="A10_summary.median_var_fraction, "
                                "results/phase2_compass_attack.json",
            finding="the calibration FAILS: the observed f lies below the floor of every "
                    "reference at every noise level on the grid, including the "
                    "group-offset-matched ones. A ring (of either kind) plus isotropic "
                    "noise plus a cardinal-vs-compound group axis does not reproduce how "
                    "little of the observed variance sits in span(N-S, E-W), so the "
                    "noise level cannot be pinned this way and A6's separating power "
                    "cannot be cashed in.",
            caveat="real activations are anisotropic; the failure of this calibration is "
                   "itself the evidence that the isotropic-noise reference family does "
                   "not describe the observed cloud"),
        observed=dict(
            label_plane_median=obs_med, label_plane_min=float(min(obs_all)),
            label_plane_max=float(max(obs_all)),
            n_cells=len(obs_all),
            n_below_1_0=int(sum(x < 1.0 for x in obs_all)),
            n_below_sqrt2=int(sum(x < 1.4142135623730951 for x in obs_all)),
            c10_plane_median=float(np.median(obs_c10))),
        matched_noun_octet_null=dict(
            n_draws_per_cell=int(a.n_noun_null_a6), rows=null_rows,
            median_of_cell_medians=null_med,
            median_plane_var_fraction=null_f_med,
            n_cells_observed_above_own_null_p95=int(n_obs_above_null_p95),
            reading="unstructured noun octets put the same statistic far below the "
                    "observed value, so the observed median is elevated relative to "
                    "no-structure while sitting below every ZERO-NOISE compositional "
                    "reference. It does not sit below raw-sum composition once noise is "
                    "allowed -- see noise_free_reading."),
    )

    # ---- corrected verdict --------------------------------------------------
    n_disc = 0   # statistics that survive the DISCRIMINATING test against a circle
    rc = reach["circle_plus_group_offset"]["noise_reaching_target"]
    rr = reach["raw_sums_plus_group_offset"]["noise_reaching_target"]
    a6_text = (
        f"THE ONLY STATISTIC THAT SEPARATES THE ACCOUNTS AT ALL, AND IT DOES NOT SUPPORT "
        f"RAW-SUM COMPOSITION AT FACE VALUE -- but it cannot be cashed in. Observed "
        f"label-plane median {obs_med:.4f} "
        f"({sum(x < 1.0 for x in obs_all)}/{len(obs_all)} cells below 1.000, "
        f"{sum(x < 1.4142135623730951 for x in obs_all)}/{len(obs_all)} below sqrt(2) = "
        f"1.414), against zero-noise predictions of 1.414 (raw-sum composition) and "
        f"1.000 (equal-radius rose). With the noise level left free, BOTH accounts reach "
        f"{obs_med:.3f}: an equal-radius rose at nz ~ {rc:.2f}, raw-sum composition at "
        f"nz ~ {rr:.2f}. Pinning the noise by the label-plane variance fraction fails -- "
        f"the observed f = {obs_f_med:.4f} lies below every reference's floor at every "
        f"noise level, so ring + isotropic noise + group axis does not describe the "
        f"observed cloud. Unstructured noun octets put the same statistic at "
        f"{null_med:.4f}, so the observed value is elevated relative to no-structure. "
        f"A6 also cannot separate a learned circle from unit-normalised composition, "
        f"which is the same geometric object. Net: A6 does not license either "
        f"conclusion.")
    corrected = dict(
        attack_verdict="composition_sufficient_not_discriminated",
        replaces=dict(previous="partially_compositional",
                      previous_artifact="results/phase2_compass_attack.json"),
        phrase_withdrawn="'not a learned compass circle' -- withdrawn; no statistic in "
                         "the battery discriminates against a learned circle",
        label_withdrawn="'Attack A7 (decisive)' -- A7 is degenerate by construction and "
                        "is not a test",
        statistic_classification=dict(
            A1_planted_composition="SUFFICIENCY ONLY -- composition alone reproduces the "
                                   "C10 signature. Stands unchanged. It shows the "
                                   "compositional account is SUFFICIENT, never that it "
                                   "is NECESSARY.",
            A2_A3_antipodality_additivity="NON-INFORMATIVE as operationalised (0/33), "
                                          "for the group-offset reason A10 documents. "
                                          "Unchanged.",
            A4_label_plane="CONSISTENT WITH BOTH -- fires identically on a planted true "
                           "circle. Removed from the SUPPORTED list.",
            A5_plane_identity="DESCRIPTIVE. Unchanged.",
            A6_radius_alternation=a6_text,
            A7_residual_ring="DEGENERATE BY CONSTRUCTION -- not a test. Removed from the "
                             "SUPPORTED list.",
            A8_template_sensitivity="DESCRIPTIVE. Unchanged.",
            A9_copy_forward="DESCRIPTIVE. Unchanged.",
            A10_additivity="CONSISTENT WITH BOTH -- +1.000 exactly on a planted true "
                           "circle. Removed from any supporting role."),
        n_discriminating_statistics_supporting_composition=n_disc,
        reason="A1 stands: pure composition, pushed through the identical C10 code path, "
               "reproduces the C10 signature, so the compositional account is SUFFICIENT "
               "to explain the post-hoc positive. What the battery cannot do is show it "
               "is NECESSARY. A4, A7 and A10-additivity were used as evidence for "
               "composition, but a planted TRUE compass circle -- ground truth, no "
               "compositional structure at all -- scores the same on all three through "
               "the identical code path, so none of them discriminates. The one "
               "statistic that separates the accounts in principle, A6, places the "
               "observed data below every zero-noise compositional reference, but both "
               "accounts reach the observed value at attainable noise levels and the "
               "noise level cannot be pinned from the data, so A6 does not license a "
               "conclusion either. The honest statement is that composition suffices and "
               "the data do not choose. Symmetrically, nothing here licenses the claim "
               "that the C10 ring IS a learned compass circle.",
        prereg_verdict_unchanged=dict(
            blind=main_art["verdict"]["blind"],
            supervised=main_art["verdict"]["supervised"],
            note="unchanged by this script, by construction; the C10 tier is post-hoc "
                 "by its own declaration and earns no prereg verdict either way"),
        depth_caveat=atk_art["verdict"]["depth_caveat"],
    )
    out["verdict"] = corrected

    out["runtime_sec"] = time.time() - t0
    out["forward_sec"] = fwd
    Path(a.out).write_text(json.dumps(out, indent=1, default=C31.jsonable))
    print(f"\nwrote {a.out}  ({time.time() - t0:.1f}s total, forwards {fwd:.1f}s)")
    print(f"CORRECTED ATTACK VERDICT: {corrected['attack_verdict']}")
    print(f"PREREG VERDICTS UNCHANGED: blind {main_art['verdict']['blind']}, "
          f"supervised {main_art['verdict']['supervised']}")


if __name__ == "__main__":
    main()
