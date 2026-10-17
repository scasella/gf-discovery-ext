"""e71 page formatter: renders results/e71_readout.json (+ e71_calib.json) as the REPORT.md family pages.
Numbers are copied mechanically; the analyst's readings are added by hand in REPORT.md. Prints markdown to stdout."""
import json, sys
from pathlib import Path

EXT = Path(__file__).resolve().parents[1]
r = json.loads((EXT / "results/e71_readout.json").read_text())
SRC = {7: "resid_post_all/L7_16k_big", 12: "resid_post_all/L12_16k_big"}


def sd(d):
    return d.replace("resid_post_all/", "").replace("resid_post/", "rp ").replace("_16k", " 16k").replace("_65k", " 65k").replace("_262k", " 262k").replace("_", " ")


def un(nm):
    k, j = nm.split("/")
    return f"{k} #{j}"


def mcell(t):
    if not t:
        return "—"
    s = f"{t['side']}; sig {t['sig_share']:.2f} (≥{t['sig_threshold']:.2f}, p99 {t['sig_null_p99']:.2f}); ovl {t['overlap']:.2f} (≥{t['overlap_threshold']:.2f}, p99 {t['overlap_null_p99']:.2f})"
    return s + ("; **match**" if t["match"] else "; no match")


def page(p):
    L = p["layer"]; f = p["census"]; mem = list(p["members"])
    out = [f"#### L{L} {' / '.join(un(m) for m in mem)}", ""]
    e65 = {x["unit"]: x for x in f["e65"]}
    rows = ["| member | e68 verdict (with / without source) | best \\|cos\\| (dictionary) | max \\|r\\| | e64c best rule, \\|AUROC\\| | e65 B3 / B4 | signature testable (s_self vs base) |", "|---|---|---|---|---|---|---|"]
    for nm in mem:
        cu = p["members"][nm]["census_unit"]; pr = p["members"][nm]["prep"]; ec = f["e64c"][nm]
        b3b4 = "not run (L7)" if L == 7 else f"{e65[nm]['b3']} / {e65[nm]['b4']}"
        rows.append(f"| {un(nm)} | {cu['verdict']} / {cu['verdict_without_source']} | {cu['best_cos']['value']:.2f} ({sd(cu['best_cos']['dict'])}) | {cu['max_abs_r']:.2f} | {ec['rule']}, {ec['abs_auroc']:.2f} | {b3b4} | {pr['testable']} ({pr['s_self']:.3f} vs {pr['base']:.3f}) |")
    out += rows + [""]
    out.append(f"Census record: R2-replicated **{f['r2_replicated']}**; e64c-nameable {f['e64c_nameable']}; named twin {bool(f['twin_named'])}; "
               f"identity/generic {f['identity_or_generic']}; near a named/covered unit {bool(f['proximity'])}; census tier {f['tier']}.")
    for nm in mem:
        out.append(f"Signature T of {un(nm)}: " + " ".join(f"`{t.strip() or repr(t)}`" for t in p["members"][nm]["prep"]["T"]))
    out.append("")
    # decoder / acts
    for nm in mem:
        m = p["members"][nm]
        nmatch = len(m["matches"]); npos = sum(t["side"] == "pos" for t in m["matches"])
        nsrc = sum(t["dict"] != SRC[L] for t in m["matches"])
        mo = m["max_overlap"]
        out.append(f"**{un(nm)}: nearest latents (pooled over the {len(m['max_cos_by_dict'])} primary dictionaries).** "
                   f"max \\|cos\\| above the subspace-null p99.9 in {m['n_above_p999']} of {len(m['max_cos_by_dict'])} dictionaries "
                   f"({m['n_above_p999_without_source']} without source). Tested latents {m['n_tested']}; **{nmatch} match** "
                   f"({npos} on the heavy side, {nmatch - npos} on the opposite side; {nsrc} outside the source dictionary). "
                   f"Largest tail overlap: {sd(mo['dict'])} #{mo['latent']}, {mo['overlap']:.3f} (threshold {mo['threshold']:.3f}, null p99 {mo['null_p99']:.3f}).")
        out.append("")
        out.append("| | top \\|cos\\| latent | cos | match test | top \\|r\\| latent | r (within type) | match test |")
        out.append("|---|---|---|---|---|---|---|")
        for i in range(5):
            c = m["decoder_top5"][i]; a = m["acts_top5"][i]
            rw = "—" if a["r_within_type"] is None else f"{a['r_within_type']:.2f}"
            out.append(f"| {i + 1} | {sd(c['dict'])} #{c['latent']} | {c['cos']:.3f} | {mcell(c['tested'])} | {sd(a['dict'])} #{a['latent']} | {a['r']:.3f} ({rw}) | {mcell(a['tested'])} |")
        out.append("")
    # OMP table
    dicts = [c["dict"] for c in p["members"][mem[0]]["omp"] if c["mode"] == "signed"]
    hdr = "| dictionary | " + " | ".join(f"{un(nm)} signed / nonneg k08" for nm in mem) + " | null k08 p1 / p5 / p50, signed | nonneg |"
    out += ["**OMP atoms to cos 0.8** (band vs that dictionary's per-mode null; `>` = unreached by k_max, banded as k_max + 1):", "", hdr,
            "|---|" + "---|" * len(mem) + "---|---|"]
    for dn in dicts:
        cells = []
        for nm in mem:
            cs = {c["mode"]: c for c in p["members"][nm]["omp"] if c["dict"] == dn}
            fmt = lambda c: (f">{c['k08'] - 1}" if c["unreached"] else str(c["k08"])) + f" [{c['band']}]"
            cells.append(f"{fmt(cs['signed'])} / {fmt(cs['nonneg'])}")
        cs = {c["mode"]: c for c in p["members"][mem[0]]["omp"] if c["dict"] == dn}
        q = lambda c: f"{c['q1']:g} / {c['q5']:g} / {c['q50']:g}"
        out.append(f"| {sd(dn)} | " + " | ".join(cells) + f" | {q(cs['signed'])} | {q(cs['nonneg'])} |")
    out.append("")
    if L == 12:
        out.append("**(b) sparsity and (c) pre-check.**")
        out.append("")
        out.append("| member | (b) class, with source (S1 / S5 / H; per mode s, nn) | without source | (c) held-out R² at K = 4 / 8 / 16 / 32 / 64 | K\\* |")
        out.append("|---|---|---|---|---|")
        for nm in mem:
            m = p["members"][nm]; s = m["sparsity"]["with_source"]; w = m["sparsity"]["without_source"]
            pm = lambda st: f"{st['S1']} / {st['S5']} / {st['H']}; s {st['per_mode']['signed']['S1']}/{st['per_mode']['signed']['S5']}/{st['per_mode']['signed']['H']}, nn {st['per_mode']['nonneg']['S1']}/{st['per_mode']['nonneg']['S5']}/{st['per_mode']['nonneg']['H']}"
            r2 = m["precheck"]["r2"]
            out.append(f"| {un(nm)} | **{s['cls']}** ({pm(s['stats'])}) | {w['cls']} ({w['stats']['S1']} / {w['stats']['S5']} / {w['stats']['H']}) | "
                       + " / ".join(f"{r2[str(k)] if str(k) in r2 else r2[k]:.3f}" for k in (4, 8, 16, 32, 64)) + f" | {m['precheck']['kstar'] or '>64'} |")
        out.append("")
        for nm in mem:
            lc = p["members"][nm]["small16k_latent_contexts"]
            out.append(f"**{un(nm)}: L12 16k-small latents** (top 3 by \\|cos\\| and by \\|r\\|; their 12 strongest D2 contexts; share of those 12 rows in the member's 1% heavy tail):")
            out.append("")
            out.append("| latent | cos | r | e68 match | top-12 in tail | strongest contexts (first 3) |")
            out.append("|---|---|---|---|---|---|")
            for li, c in lc.items():
                ctx = " ⏐ ".join("`" + x.replace("`", "'").replace("\n", "⏎").replace("|", "¦") + "`" for x in c["contexts"][:3])
                cs = "—" if c["cos"] is None else f"{c['cos']:.3f}"; rr = "—" if c["r"] is None else f"{c['r']:.3f}"
                tt = "—" if c["top12_in_unit_tail"] is None else f"{round(12 * c['top12_in_unit_tail'])}/12"
                out.append(f"| #{li} | {cs} | {rr} | {c['match']} | {tt} | {ctx} |")
            out.append("")
    for nm in mem:
        e67 = [t for t in p["members"][nm]["decoder_top5"] + p["members"][nm]["acts_top5"] if "e67_contexts" in t]
        if e67:
            out.append(f"**{un(nm)}: e67's stored contexts for its pooled top latents** (cited from `results/e67_acts.json`; not recomputed):")
            out.append("")
            out.append("| latent | strongest contexts (first 3) |")
            out.append("|---|---|")
            seen = set()
            for t in e67:
                if (t["dict"], t["latent"]) in seen:
                    continue
                seen.add((t["dict"], t["latent"]))
                ctx = " ⏐ ".join("`" + x.replace("`", "'").replace("\n", "⏎").replace("|", "¦") + "`" for x in t["e67_contexts"][:3])
                out.append(f"| {sd(t['dict'])} #{t['latent']} | {ctx} |")
            out.append("")
    for nm in mem:
        out.append(f"<details><summary>{un(nm)}: 12 strongest D2 contexts (oriented score)</summary>")
        out.append("")
        for x in p["members"][nm]["contexts_strongest"]:
            out.append("- `" + x.replace("`", "'").replace("\n", "⏎") + "`")
        out.append("")
        out.append("</details>")
        out.append("")
    return "\n".join(out)


def precheck_table():
    import numpy as np
    c = json.loads((EXT / "results/e71_calib.json").read_text()); cen = json.loads((EXT / "results/e68_census.json").read_text())["groups"]["12"]["units"]
    pc = r["precheck"]; nul = c["precheck"]["nulls"]
    anchors = list(c["precheck"]["anchors"]); targets = ["logcosh/45", "skew/5", "logcosh/54", "skew/54"]
    groups = [("anchors", anchors), ("targets (#45/#5, #54/#54)", targets),
              ("all candidates", [n for n, u in cen.items() if u["role"] == "candidate"]),
              ("context-named", [n for n, u in cen.items() if u["role"] == "context_named"]),
              ("controls", [n for n, u in cen.items() if u["role"] == "control"])]
    out = ["| group (n) | R² at K = 32: median (min–max) | R² at K = 64: median (min–max) | reaching 0.5 at any K ≤ 64 |", "|---|---|---|---|"]
    for g, ns in groups:
        v32 = [pc[n]["r2"]["32"] for n in ns]; v64 = [pc[n]["r2"]["64"] for n in ns]
        out.append(f"| {g} ({len(ns)}) | {np.median(v32):.3f} ({min(v32):.3f}–{max(v32):.3f}) | {np.median(v64):.3f} ({min(v64):.3f}–{max(v64):.3f}) | {sum(pc[n]['kstar'] is not None for n in ns)} |")
    v32 = [x["r2"]["32"] for x in nul.values()]; v64 = [x["r2"]["64"] for x in nul.values()]
    out.append(f"| joint nulls (40) | {np.median(v32):.3f} ({min(v32):.3f}–{max(v32):.3f}) | {np.median(v64):.3f} ({min(v64):.3f}–{max(v64):.3f}) | {sum(x['kstar'] is not None for x in nul.values())} |")
    return "\n".join(out)


def summary_table():
    mn = json.loads((EXT / "results/e71_match_null.json").read_text())
    out = ["| family | e68 (with source) | best \\|cos\\| / max \\|r\\| | e68 matches: all (heavy side) | fixed-set match rate (nulls ≥, of 40) | (b) sparsity | (c) R² K = 32 / 64 |",
           "|---|---|---|---|---|---|---|"]
    for p in r["families"]:
        L = p["layer"]; cells = [[] for _ in range(6)]
        for nm, m in p["members"].items():
            cu = m["census_unit"]; u = mn[str(L)]["units"][nm]
            cells[0].append(cu["verdict"]); cells[1].append(f"{cu['best_cos']['value']:.2f} / {cu['max_abs_r']:.2f}")
            cells[2].append(f"{u['matches']} ({u['heavy_side']})"); cells[3].append(f"{u['fixed_rate']:.2f} ({u['nulls_ge_fixed_rate']})")
            if L == 12:
                cells[4].append(m["sparsity"]["with_source"]["cls"]); r2 = m["precheck"]["r2"]; cells[5].append(f"{r2['32']:.2f} / {r2['64']:.2f}")
            else:
                cells[4].append("— (L7)"); cells[5].append("— (L7)")
        out.append(f"| L{L} {' / '.join(un(x) for x in p['members'])} | " + " | ".join("; ".join(c) for c in cells) + " |")
    nl = {L: mn[str(L)]["null"] for L in (7, 12)}
    out.append("| **joint nulls** L7 / L12, median (max) | — | — | " + " / ".join(f"{nl[L]['matches']['p50']:g} ({nl[L]['matches']['max']:g})" for L in (7, 12))
               + " | " + " / ".join(f"{nl[L]['fixed_rate']['p50']:.2f} ({nl[L]['fixed_rate']['max']:.2f})" for L in (7, 12)) + " | L12: 38 of 40 chance-like | L12 K = 64: 0.16 (0.20) |")
    return "\n".join(out)


if __name__ == "__main__":
    if sys.argv[1:] == ["summary"]:
        print(summary_table()); sys.exit()
    if sys.argv[1:] == ["precheck"]:
        print(precheck_table()); sys.exit()
    for p in r["families"]:
        print(page(p))
        print()
