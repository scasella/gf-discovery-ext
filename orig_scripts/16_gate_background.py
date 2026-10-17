"""Background acceptance rate measured INSIDE the gate's own pipeline.

`scripts/11_months_and_power.py --stratified` cannot supply the Amendment-1
background. Its draw is index-band matched over all 16,384 latents and is then
intersected with the latents that fire on its own 240 prompt-final positions, so
at layer 9 / `l0_big` a template of 44 latents realises as 4.9 and only 65 of 200
draws are scorable. A 5-latent subset is not a background for a 6-103-latent
cluster. Nor can the draw be repaired in place: restricting it to the
prompt-active latents at full template size gives background subsets whose
members all fire on those 240 positions, while a real cluster's members mostly do
not -- mismatched in the opposite direction. That script's representation, 12 mean
vectors over 20 prompts, structurally cannot produce a large-k matched background.

So the background is measured where the claim is made. Templates are the members
of exactly the clusters the gate scored, read from the gate's own output; draws
are index-band matched at full template size with no activity mask; and each draw
is scored by `labelled()` imported from `scripts/13_rerun_gate.py` itself -- the
same function, the same `SparseReader`, the same rank-even half-B label positions
and the same ceil(2k/3) support rule -- so the background and the observation
differ only in how the latent set was chosen.

Draws that `labelled()` rejects for support are unscorable and leave the
denominator, exactly as the gate drops them. The headline rate counts raw
ordering p <= 0.05, which is what the earlier 0.295 / 0.44 rates counted; the gate
requires ordering after BH-FDR, so this errs towards a background that is too
high. Closure and both-tier rates are reported alongside.

Decoder-direction ranking is not applied to the draws, matching the earlier
measurement: the gate uses it to pick which 20 clusters to score, not as a tier.
"""
import argparse, importlib.util, json, os, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gfdisc.cache import SparseReader
from gfdisc.saes import load_sae, get_decoder
from gfdisc.nulls import stratified_resample

_spec = importlib.util.spec_from_file_location(
    "rerun_gate", Path(__file__).resolve().parent / "13_rerun_gate.py")
gate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gate)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", required=True, help="gate output JSON supplying the templates")
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--clusters", nargs="+", required=True)
    # `labelled()` runs its own 20,000-permutation ordering null and takes no n_perm
    # argument, so there is nothing to expose: the background is calibrated at exactly
    # the resolution the gate used, which is finer than the 4,000 the earlier
    # (script-11) background used. Draws rejected for support cost nothing -- they
    # return before any permutation runs -- so n is set by how many SCORABLE draws are
    # needed, not by the nominal draw count.
    ap.add_argument("--n-random", type=int, default=2000)
    ap.add_argument("--l0", default="big")
    ap.add_argument("--root", default="cache/sparse_big")
    ap.add_argument("--half", default="rank-even", choices=["full", "rank-even"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    token = os.environ["HF_TOKEN"]

    g = json.loads(Path(a.gate).read_text())
    templates = [np.asarray(r["members"]) for r in g["rows"] if r["layer"] == a.layer]
    if not templates:
        raise SystemExit(f"{a.gate} has no scored clusters at layer {a.layer}: nothing to match")
    names, label_ids = gate.label_ids_for(g["ontology"], json.load(open("results/token_map.json")))
    full_k = len(names)

    sae = load_sae(a.layer, a.l0, token=token); W = get_decoder(sae)
    reader = SparseReader(a.root, a.layer)
    label_pos = gate.half_filter([reader.positions_of_tokens(ids) for ids in label_ids], a.half)

    print(f"layer {a.layer}, {g['ontology']}: {len(templates)} scored clusters as templates, "
          f"sizes {sorted(len(t) for t in templates)}", flush=True)
    rng = np.random.default_rng(0)
    rows = []
    for i in range(a.n_random):
        if i and i % 200 == 0:
            n_sc = sum(r["scorable"] for r in rows)
            print(f"  {i}/{a.n_random} draws, {n_sc} scorable", flush=True)
        t = templates[i % len(templates)]
        f = stratified_resample(t, sae.d_sae, rng)
        lab = gate.labelled(reader, [int(x) for x in f], W, label_pos, seed=i, full_k=full_k)
        if lab is None:
            rows.append(dict(i=i, size=int(len(f)), template_size=int(len(t)), scorable=False))
            continue
        rows.append(dict(i=i, size=int(len(f)), template_size=int(len(t)), scorable=True,
                         rho=lab["rho"], p_order=lab["p_order"], p_closed=lab["p_closed"],
                         gap_ratio=lab["gap_ratio"], n_labels=lab["n_labels"]))

    sc = [r for r in rows if r["scorable"]]
    n = len(sc)
    n_ord = sum(r["p_order"] <= 0.05 for r in sc)
    n_clo = sum(r["p_closed"] < 0.05 for r in sc)
    n_both = sum(r["p_order"] <= 0.05 and r["p_closed"] < 0.05 for r in sc)
    summary = dict(layer=a.layer, ontology=g["ontology"], l0=a.l0, half=a.half,
                   gate=a.gate, n_draws=len(rows), n_scorable=n,
                   n_templates=len(templates),
                   template_size_mean=float(np.mean([len(t) for t in templates])),
                   draw_size_mean=float(np.mean([r["size"] for r in rows])),
                   n_sig_order=n_ord, rate_order=n_ord / max(n, 1),
                   n_sig_closure=n_clo, rate_closure=n_clo / max(n, 1),
                   n_sig_both=n_both, rate_both=n_both / max(n, 1))
    print(json.dumps(summary, indent=2))
    Path(a.out).write_text(json.dumps(dict(summary=summary, draws=rows), indent=2))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
