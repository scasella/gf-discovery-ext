"""Tokenizer audit for the post-hoc label sets.

Weekday and month names appear in Gemma's 262k vocabulary in several surface
forms -- with and without a leading space, capitalised or not, sometimes
abbreviated. The Phase-1 ordering statistic is computed over token positions
selected by these ids, so missing a variant silently biases which occurrences of
"Monday" get counted, and a plural or abbreviation collapsed into the wrong day
would corrupt the Spearman directly.
"""
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = ["January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"]


# Lowercase forms that are ordinary English words far more often than calendar
# terms. Labelling these positions as calendar tokens would swamp the real
# occurrences: "may" and "march" in particular are common verbs/modals.
AMBIGUOUS_LOWER = {"may", "march", "august", "sun", "sat", "mon", "wed", "thus",
                   "fri", "tue", "thu", "dec", "jan", "sept"}


def variants(word):
    """Surface forms of a calendar word that are safe to use as labels.

    Three-letter abbreviations are excluded entirely. Generating them produced
    "Thus"/"thus" for Thursday, "Sun"/"sun" for Sunday, "Sat"/"sat" for Saturday
    and "Mons"/"mons" for Monday -- ordinary words that occur far more often in
    their non-calendar sense, which would have dominated the labelled positions.

    For words that are ambiguous in lowercase but not capitalised (May, March,
    August), only the capitalised forms are kept.
    """
    forms = {word, word + "s"}
    if word.lower() not in AMBIGUOUS_LOWER:
        forms |= {word.lower(), word.lower() + "s"}
    out = set()
    for f in forms:
        out |= {f, " " + f}
    return sorted(out)


def resolve(tokenizer, words, single_token_only=True):
    """Map each word to the token ids that represent it on its own.

    Only single-token forms are usable: a position in the cache is one token, so
    a name that tokenises into several pieces has no single position to label.
    Returns (mapping, report) so the caller can see what was dropped.
    """
    mapping, report = {}, {}
    for w in words:
        ids, kept, dropped = [], [], []
        for v in variants(w):
            enc = tokenizer.encode(v, add_special_tokens=False)
            if len(enc) == 1:
                ids.append(enc[0]); kept.append(v)
            elif not single_token_only:
                ids.extend(enc); kept.append(v)
            else:
                dropped.append(v)
        mapping[w] = sorted(set(ids))
        report[w] = dict(n_ids=len(set(ids)), kept=kept, dropped=dropped)
    return mapping, report


def audit(tokenizer):
    day_map, day_rep = resolve(tokenizer, DAYS)
    mon_map, mon_rep = resolve(tokenizer, MONTHS)
    collisions = {}
    for name, m in (("day", day_map), ("month", mon_map)):
        seen = {}
        for w, ids in m.items():
            for i in ids:
                if i in seen:
                    collisions.setdefault(name, []).append((seen[i], w, i))
                seen[i] = w
    return dict(days=day_map, months=mon_map,
                day_report=day_rep, month_report=mon_rep, collisions=collisions)
