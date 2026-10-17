"""e64c -- post-verdict, descriptive: the 14 constructions of the e64 context reading as mechanical rules.

The analyst's context reading (results/e64_context_reading.json, sealed) named 26 battery-UNNAMED
units with 14 constructions. Here each construction becomes a fixed rule over D2's tokens (a spaCy
parse where syntax is needed), and every rule is scored against the frozen D2 unit scores. Nothing
here changes any e64 tier or the verdict. Rules were written and this file hashed before any score.

Stages (run in this order)
  decode  (.venv)        D2 token pieces per sequence, decoded one token at a time as e61 does
  parse   (.venv-parse)  spaCy parse of each sequence's text; one 0/1 feature per rule per D2 token
                         (--model en_core_web_trf, the primary; en_core_web_sm, the robustness parse)
  score   (.venv)        per named unit: its own rule, oriented to the read tail; every rule against
                         every screened unit at L7 and L12 (specificity and possible misses)

Mapping. A sequence's text is the concatenation of its pieces. A D2 token takes the spaCy token that
holds its first non-space character; a piece that is all whitespace maps to no spaCy token (0 on
every parse rule). Every piece of a multi-piece word gets that word's value.

Rules (S = a set of spaCy tokens; the D2 feature is 1 if the token's spaCy token is in S)
  A after_cc         previous piece, stripped and lower-cased, is one of: and, or, but, nor, &
  B temporal_clause  after a temporal subordinator: for t in {when, whenever, while, before, after,
                     until, till, once} with dep mark/advmod and an advcl head, the head's subtree
                     after t; for t in {before, after, until, during} with dep prep, t's subtree
                     after t
  C after_prep       the first piece of a word whose previous non-space spaCy token is ADP
                     (C_inat: previous piece is 'in' or 'at')
  D partitive_superl partitive: a quantifier or number (one, some, many, most, all, each, any,
                     none, several, few, both, a NUM, %) with a child prep 'of', from the
                     quantifier to the end of its subtree; superlative: a noun chunk with a JJS/RBS
                     token or first/last/only/next, from the chunk start to the end of its root's
                     subtree
  E second_conjunct  a coordination group of exactly two items (the first conjunct plus the conj
                     tokens reached through conj edges; spaCy chains them) with 'and'/'&' between
                     them: the second item's subtree
  F subject_np       the subtree of an nsubj/nsubjpass/csubj/csubjpass token, before its head verb
                     (F_multi: subjects of two or more tokens only)
  G fronted_adverbial  per sentence: a dependent of the root with dep advcl/prep/advmod/npadvmod,
                     whose subtree starts within the sentence's first three tokens and ends before
                     the root's subject (or before the root if it has none); that subtree
  H coord_item_onset the left edge of every conj token, and every cc token whose head has a conj
                     child after it
  I line_initial     at most 5 tokens after the most recent newline piece, provided a newline has
                     occurred earlier in the sequence (line position as in e61)
  J negation_scope   after a negator, inside its head's subtree. Negators: dep neg (not, n't,
                     never); det 'no'; nothing, nobody, none, nowhere, neither, nor, never. A
                     negative subject ('no one', 'nothing') takes its verb's subtree. For prep
                     'without', its own subtree after it
  K copular_pred     for c with dep attr/acomp/oprd whose head lemma is be, become, seem, remain,
                     appear or stay: the tokens from just after the head to c's right edge
  L enumeration      a coordination group of 3 or more items, from the first item to the last
                     item's right edge; and NUM (-|–|to|through) NUM ranges, all three tokens
  M temporal_expr    DATE/TIME entity tokens and the token right after each; for t in {before,
                     after, during, until, since} with dep prep, t's subtree after t
  N after_the        previous piece, stripped and lower-cased, is 'the'
"""
import argparse, hashlib, json, sys, time
from pathlib import Path

import numpy as np

EXT = Path(__file__).resolve().parents[1]
N_TOK = 255
RULES = ("A", "B", "C", "C_inat", "D", "E", "F", "F_multi", "G", "H", "I", "J", "K", "L", "M", "N")
CONSTRUCTION_RULE = {        # reading text in results/e64_context_reading.json -> rule
    "token right after a coordinating conjunction (and/or/but)": "A",
    "inside a temporal subordinate clause (when/while/before/after/during)": "B",
    "token right after a preposition (mostly 'in', also 'at')": "C",
    "inside a partitive/superlative NP ('one of the ...', 'the best/first/only ...', 'X% of all ...')": "D",
    "inside the second conjunct of an 'X and Y' coordination": "E",
    "inside the subject noun phrase, before its main verb": "F",
    "inside a fronted (sentence-initial) adverbial phrase or clause": "G",
    "onset of the next item in a coordinated series/list": "H",
    "first clause of a line (within about five tokens after a newline)": "I",
    "inside the scope of a negation (not/no/never/don't/cannot)": "J",
    "inside the predicate complement of a copula (is/are/was/become ...)": "K",
    "inside an enumeration/list of items or a numeric range": "L",
    "inside or right after a temporal expression (dates, durations, before/after)": "M",
    "token right after the definite article 'the'": "N",
}
# naming thresholds of prereg section 6.1, reused descriptively
NAME_AUROC, NAME_ENRICH, NAME_MIN_RATE, TAIL = 0.75, 3.0, 0.10, 0.01


# ---------------------------------------------------------------------------- decode (.venv)
def decode():
    sys.path.insert(0, str(EXT / "scripts"))
    import importlib.util
    spec = importlib.util.spec_from_file_location("e61", EXT / "scripts/e61_occurrence_screen.py")
    e61 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e61)
    tk = e61.token_strings()
    toks = np.load(EXT / "cache/occ/occD2_meta.npz")["tokens"].astype(np.int64)
    cache, seqs = {}, []
    for s0 in range(0, len(toks), N_TOK):
        row = []
        for t in toks[s0:s0 + N_TOK]:
            t = int(t)
            if t not in cache:
                cache[t] = tk.decode([t])
            row.append(cache[t])
        seqs.append(row)
    (EXT / "results/e64c_d2_pieces.json").write_text(json.dumps(seqs))
    print(f"decoded {len(seqs)} sequences", flush=True)


# ---------------------------------------------------------------------------- parse (.venv-parse)
def lexical(pieces):
    n = len(pieces)
    f = {k: np.zeros(n, np.uint8) for k in ("A", "C_inat", "I", "N")}
    seen_nl, lp = False, 0
    for j in range(n):
        prev = pieces[j - 1].strip().lower() if j > 0 else ""
        f["A"][j] = prev in ("and", "or", "but", "nor", "&")
        f["C_inat"][j] = prev in ("in", "at")
        f["N"][j] = prev == "the"
        f["I"][j] = seen_nl and lp <= 5
        if "\n" in pieces[j]:
            seen_nl, lp = True, 0
        else:
            lp += 1
    return f


def span_after(tok, after_i):
    return [t.i for t in tok.subtree if t.i > after_i]


def coord_group(t):
    """t followed by every conj reachable from t through conj edges, in order."""
    out, stack = [t], [t]
    while stack:
        for c in stack.pop().children:
            if c.dep_ == "conj":
                out.append(c); stack.append(c)
    return sorted(out, key=lambda x: x.i)


def parse_rules(doc):
    S = {k: set() for k in ("B", "D", "E", "F", "F_multi", "G", "H", "J", "K", "L", "M")}
    for t in doc:
        lo = t.lower_
        # B temporal clause
        if lo in ("when", "whenever", "while", "before", "after", "until", "till", "once") \
                and t.dep_ in ("mark", "advmod") and t.head.dep_ == "advcl":
            S["B"].update(span_after(t.head, t.i))
        if lo in ("before", "after", "until", "during") and t.dep_ == "prep":
            S["B"].update(span_after(t, t.i))
        # D partitive / superlative
        if (lo in ("one", "some", "many", "most", "all", "each", "any", "none", "several", "few", "both")
                or t.pos_ == "NUM" or t.text == "%") and any(c.dep_ == "prep" and c.lower_ == "of" for c in t.children):
            S["D"].update(x.i for x in t.subtree if x.i >= t.i)
        # E second conjunct of a two-item 'and' coordination (spaCy chains conjuncts: the group is the
        # first conjunct plus every conj reached through conj edges)
        group = coord_group(t) if t.dep_ != "conj" else []
        if len(group) == 2:
            c = group[1]
            if any(x.dep_ == "cc" and x.lower_ in ("and", "&") and t.i < x.i < c.i for x in list(t.children) + list(c.children)):
                S["E"].update(x.i for x in c.subtree)
        # F subject NP before its verb
        if t.dep_ in ("nsubj", "nsubjpass", "csubj", "csubjpass"):
            sub = [x.i for x in t.subtree if x.i < t.head.i]
            S["F"].update(sub)
            if len(sub) >= 2:
                S["F_multi"].update(sub)
        # H onset of the next coordinated item
        if t.dep_ == "conj":
            S["H"].add(t.left_edge.i)
        if t.dep_ == "cc" and any(c.dep_ == "conj" and c.i > t.i for c in t.head.children):
            S["H"].add(t.i)
        # J negation scope (a negative subject -- 'no one', 'nothing' -- scopes over its clause)
        if t.dep_ == "neg" or (lo == "no" and t.dep_ == "det") or lo in (
                "nothing", "nobody", "none", "nowhere", "neither", "nor", "never"):
            h = t.head if t.dep_ == "det" else t
            if h.dep_ in ("nsubj", "nsubjpass"):
                S["J"].update(span_after(h.head, t.i))
            else:
                S["J"].update(span_after(t.head if t.dep_ != "ROOT" else t, t.i))
        if lo == "without" and t.dep_ == "prep":
            S["J"].update(span_after(t, t.i))
        # K copular predicate
        if t.dep_ in ("attr", "acomp", "oprd") and t.head.lemma_ in ("be", "become", "seem", "remain", "appear", "stay"):
            S["K"].update(range(t.head.i + 1, t.right_edge.i + 1))
        # L enumeration (3+ items in one coordination group)
        if len(group) >= 3:
            S["L"].update(range(t.i, max(c.right_edge.i for c in group[1:]) + 1))
        # M temporal prepositions
        if lo in ("before", "after", "during", "until", "since") and t.dep_ == "prep":
            S["M"].update(span_after(t, t.i))
    # L numeric ranges
    for i in range(1, len(doc) - 1):
        if doc[i].text in ("-", "–", "to", "through") and doc[i - 1].pos_ == "NUM" and doc[i + 1].pos_ == "NUM":
            S["L"].update((i - 1, i, i + 1))
    # D superlative noun chunks
    for ch in doc.noun_chunks:
        if any(x.tag_ in ("JJS", "RBS") or x.lower_ in ("first", "last", "only", "next") for x in ch):
            S["D"].update(x.i for x in ch.root.subtree if x.i >= ch.start)
    # G fronted adverbial
    for sent in doc.sents:
        r = sent.root
        subj = [c.i for c in r.children if c.dep_ in ("nsubj", "nsubjpass", "csubj", "csubjpass", "expl")]
        limit = min(subj) if subj else r.i
        for d in r.children:
            if d.dep_ in ("advcl", "prep", "advmod", "npadvmod") and d.left_edge.i <= sent.start + 2 \
                    and d.right_edge.i < limit:
                S["G"].update(x.i for x in d.subtree)
    # M DATE/TIME entities and the token after each
    for e in doc.ents:
        if e.label_ in ("DATE", "TIME"):
            S["M"].update(range(e.start, min(e.end + 1, len(doc))))
    return S


def parse(model):
    import spacy
    nlp = spacy.load(model)
    seqs = json.loads((EXT / "results/e64c_d2_pieces.json").read_text())
    n = len(seqs) * N_TOK
    out = {k: np.zeros(n, np.uint8) for k in RULES}
    texts, starts = [], []
    for pieces in seqs:
        texts.append("".join(pieces))
        starts.append(np.cumsum([0] + [len(p) for p in pieces[:-1]]))
    t0 = time.time()
    for s, doc in enumerate(nlp.pipe(texts, batch_size=32 if "trf" in model else 128)):
        pieces, base = seqs[s], s * N_TOK
        for k, v in lexical(pieces).items():
            out[k][base:base + N_TOK] = v
        char2tok = np.full(len(doc.text) + 1, -1)
        for t in doc:
            char2tok[t.idx:t.idx + len(t.text)] = t.i
        tok_of = np.full(N_TOK, -1)
        for j, p in enumerate(pieces):
            k = next((starts[s][j] + q for q, ch in enumerate(p) if not ch.isspace()), None)
            if k is not None:
                tok_of[j] = char2tok[k]
        S = parse_rules(doc)
        for j in range(N_TOK):
            i = tok_of[j]
            if i < 0:
                continue
            for k in S:
                if i in S[k]:
                    out[k][base + j] = 1
            # C: first piece of a word whose previous non-space spaCy token is ADP
            if j == 0 or tok_of[j - 1] != i:
                q = i - 1
                while q >= 0 and doc[q].is_space:
                    q -= 1
                out["C"][base + j] = q >= 0 and doc[q].pos_ == "ADP"
        if s % 100 == 0:
            print(f"  {model}: {s}/{len(seqs)} sequences, {time.time() - t0:.0f}s", flush=True)
    np.savez_compressed(EXT / f"results/e64c_rules_{model}.npz", **out)
    print(f"parsed with {model} in {time.time() - t0:.0f}s; base rates "
          + json.dumps({k: round(float(v.mean()), 3) for k, v in out.items()}), flush=True)


# ---------------------------------------------------------------------------- score (.venv)
def assoc(R, o, f):
    """AUROC toward the high tail (rank-sum; R = ranks of y, o = argsort of y) and both 1% tails' rates."""
    n = len(R); k = int(np.ceil(TAIL * n)); pos = f > 0
    n1 = int(pos.sum())
    if n1 < 50 or n - n1 < 50:
        return None
    a = (R[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * (n - n1))
    return dict(auroc_hi=float(a), base=float(pos.mean()), rate_hi=float(f[o[-k:]].mean()),
                rate_lo=float(f[o[:k]].mean()))


def would_name(r):
    if r is None:
        return False
    tails = (r["rate_hi"], r["rate_lo"])
    return bool(max(r["auroc_hi"], 1 - r["auroc_hi"]) >= NAME_AUROC
                or any(t >= NAME_ENRICH * r["base"] and t >= NAME_MIN_RATE for t in tails))


def score(models):
    from scipy.stats import rankdata
    labs = json.loads((EXT / "results/e64_labels.json").read_text())
    rd = json.loads((EXT / "results/e64_context_reading.json").read_text())["units"]
    res = dict(models=models, layers={}, named_units=[])
    feats = {m: dict(np.load(EXT / f"results/e64c_rules_{m}.npz")) for m in models}
    for L in ("7", "12"):
        z = np.load(EXT / f"results/occ_units_v2_L{L}.npz")
        rows = z["row"]
        units = labs["layers"][L]["units"]
        table = {}
        for m in models:
            F = {k: v[rows] for k, v in feats[m].items()}
            for u in units:
                y = z[f"Y_{u['kind']}"][:, u["id"]].astype(np.float64)
                R, o = rankdata(y), np.argsort(y)
                key = f"{u['kind']}/{u['id']}"
                table.setdefault(key, {})[m] = {k: assoc(R, o, F[k]) for k in RULES}
        status = {}
        for u in units:
            key = f"{u['kind']}/{u['id']}"
            r = rd.get(f"L{L}/{key}")
            status[key] = ("battery-named" if u["named"] else
                           ("context:" + CONSTRUCTION_RULE[r["reading"]] if r["reading"] != "none" else "none"))
        # per rule: every unit's best-oriented |AUROC|, the units that would be named by it
        per_rule = {}
        for m in models:
            for k in RULES:
                vals = []
                for key, t in table.items():
                    r = t[m][k]
                    if r is None:
                        continue
                    vals.append((max(r["auroc_hi"], 1 - r["auroc_hi"]), key, would_name(r)))
                vals.sort(reverse=True)
                per_rule.setdefault(k, {})[m] = dict(
                    median_abs_auroc=float(np.median([v[0] for v in vals])) if vals else None,
                    top5=[dict(unit=key, abs_auroc=round(a, 3), status=status[key], would_name=w) for a, key, w in vals[:5]],
                    would_name=[dict(unit=key, abs_auroc=round(a, 3), status=status[key]) for a, key, w in vals if w])
        res["layers"][L] = dict(per_rule=per_rule)
        for key, st in status.items():
            if not st.startswith("context:"):
                continue
            k = st.split(":")[1]
            r = rd[f"L{L}/{key}"]
            sgn = r["tail"]
            rec = dict(layer=int(L), unit=key, reading=r["reading"], rule=k, tail=sgn, evidence=r["evidence"])
            for m in models:
                a = table[key][m][k]
                if a is None:
                    rec[m] = None
                    continue
                auc_or = a["auroc_hi"] if sgn == "hi" else 1 - a["auroc_hi"]
                tail_rate = a["rate_hi"] if sgn == "hi" else a["rate_lo"]
                other = a["rate_lo"] if sgn == "hi" else a["rate_hi"]
                ranks = sorted(((max(t[m][k]["auroc_hi"], 1 - t[m][k]["auroc_hi"]) if t[m][k] else 0.0), kk)
                               for kk, t in table.items())[::-1]
                best_rule = max(RULES, key=lambda q: (max(table[key][m][q]["auroc_hi"], 1 - table[key][m][q]["auroc_hi"])
                                                      if table[key][m][q] else 0.0))
                rec[m] = dict(auroc_oriented=round(auc_or, 3), base=round(a["base"], 4), read_tail_rate=round(tail_rate, 3),
                              enrichment=round(tail_rate / max(a["base"], 1e-9), 2), other_tail_rate=round(other, 3),
                              would_name=would_name(a), rank_in_layer=1 + [kk for _, kk in ranks].index(key),
                              n_units=len(ranks), best_rule=best_rule,
                              best_rule_abs_auroc=round(max(table[key][m][best_rule]["auroc_hi"],
                                                            1 - table[key][m][best_rule]["auroc_hi"]), 3))
            res["named_units"].append(rec)
    (EXT / "results/e64c_context_rules.json").write_text(json.dumps(res, indent=1))
    print("scored", len(res["named_units"]), "context-named units", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["decode", "parse", "score"])
    ap.add_argument("--model", default="en_core_web_trf")
    ap.add_argument("--models", default="en_core_web_trf,en_core_web_sm")
    a = ap.parse_args()
    print("e64c sha256", hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), flush=True)
    if a.stage == "decode":
        decode()
    elif a.stage == "parse":
        parse(a.model)
    else:
        score(a.models.split(","))


if __name__ == "__main__":
    main()
