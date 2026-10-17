# PREREG — Stage 4 v2, discovery tier: naming, behaviour and replication of the validated occurrence units (FROZEN 2026-10-15)

Frozen before any discovery-tier data existed. When this file's sha256 was appended to
`docs/HASHES.txt`:
- S2 and R2 had never been extracted;
- no battery association, label or context of any non-control D2 unit had been computed;
- no Stage-4 unit had been steered.

`scripts/e64_occurrence_v2_discovery.py` asserts at every data phase:
- that this file's sha256 is in `docs/HASHES.txt`;
- that its own sha256 appears in this text (§13);
- that every frozen v2 input is unchanged (§2).

**This file authorises no run.** Each phase needs the user's go-ahead. Changing any rule here
means writing a new file and hashing it before the first phase that the rule governs.

## 0. What was seen before freezing

**Everything that v2 §0 lists, plus the v2 D2 run:**
- the freeze: units, screens and per-screen pass counts;
- the positive control, including the specificity tables of the **six recovering units**: their
  battery associations, their top-1% composition, and 12 sampled contexts each;
- `e63r_d2_identity_descriptives.py` (label-free);
- REPORT's "Stage 4 v2" section and the v2 hand-over note.

**What this prereg is and is not blind to.**
- The battery extension (§4) is **motivated by the control units' contexts**:
  - on P2, the L7 paren unit's top association was bracket_inside, and its contexts included
    curly single quotes;
  - on D2, the L12 paren unit fired inside comma-delimited asides.
- It is therefore not blind to those six units.
- It is **blind to every other unit**. For the 137 screened non-control units at L7/L12, no
  association, label or context has been computed. Their label-free screen statistics exist in
  the frozen JSON; they were not inspected while drafting this file.
- The only D2 numbers read while drafting were:
  - screened-unit counts (L7 logcosh 35, skew 30; L12 logcosh 36, skew 42);
  - n_rows = 203,328 per layer;
  - frozen KIR (14.7 at L7, 9.3 at L12);
  - the control-unit IDs and |AUROC|s already in REPORT.

**Code run while drafting (no data):**
- `e64 --phase selftest`: the new battery's state machine on 17 hand-written strings (§4.3);
- a random-array smoke test of e64's numeric helpers, in the session scratchpad.
  - `fit_maps` reproduces `e63.type_whiten` exactly.
  - The naming rule, twin matching and identity diagnostics behave as specified.

No corpus or reserve token was processed by e64. No model, SAE or tokenizer was loaded.

**Earlier use of the reserve (not Stage 4):**
- Stage 2's ruler behaviour (`e47r`) used 8-token slices of reserve 2,000–7,999 as steering
  contexts.
- Stage 3 (`e53`) used 8-token slices of all 8,000 reserve sequences.
- Both tested type-mean objects. Neither computed an occurrence representation, and neither
  selected or informed any Stage-4 unit.
- R2 (reserve 2,000–2,999) and the behavioural contexts (reserve 3,000–3,999) are therefore new
  to Stage 4.
- v1's R (reserve 0–999) was extracted and deleted, never analysed. v1's behavioural block
  (reserve 1,000–1,999) was never used. Neither is used here.

## 1. Question and scope

**Question.** The question is v1 §1, now asked with an instrument that passed its positive
control: among identity-removed single occurrences, is there reproducible, behaviourally active
directional structure that no named feature accounts for?

**Scope: the layers and unit kinds that the v2 control validated.**

| layer | unit kinds | screened units (frozen) | validated by |
|---|---|---|---|
| L7 | FastICA-64 logcosh, FastICA-64 skew | 35 + 30 | paren (both kinds) |
| L12 | FastICA-64 logcosh, FastICA-64 skew | 36 + 42 | paren and the untuned quote control (both kinds) |

Total: **143 screened units**.

**Excluded:**
- **L24**: its control failed, and its floor sits above the probe ceiling.
- **Clusters**: 0 / 2 / 0 screened, none recovered.
- **Arrangements**: see §7.3.

**Unchanged from `PREREG_occurrence_v2.md`:**
- the representation (v2 §4: three first-moment cycles; PCA-64 whitened; per-type equalisation
  with **N_MIN 32, n0 16**; global re-whitening);
- the units and screens (v2 §5):
  - **η²(token) of y ≤ 0.30**;
  - **η²(token) of y² ≤ 0.30**;
  - **split-half |cos| ≥ 0.8**;
  - bimodality off;
  - ICA fits on every second row, with the seed retry rule.

No D2 unit is refitted. The units are the frozen ones in `results/occ_units_v2_L{7,12}.npz`.

## 2. Frozen inputs and their reconstruction

**Inputs, asserted by sha256 at every data phase:**

| file | sha256 |
|---|---|
| `PREREG_occurrence_v2.md` | edf7575075663cf4477ed3fdcec5e8eaa85ae89ac9e4c0e924bcc93a370035ba |
| `scripts/e63_occurrence_v2.py` | 6f822a439025e01b176ae446d96413dc736078d5dbe740d03f0f05052f6e4543 |
| `scripts/e61_occurrence_screen.py` | 43a445016e82735c70ca6f2c623917fa864fce6920317832c9f2da9035bf92e1 |
| `scripts/e60_extract.py` | a6d99f92f1e21d613e80cceb594724dee9faed719131e5fc372d7e4f113cdd20 |
| `results/e63_pilot_choice.json` | 4c71cd1b25166465b7a776a66e828923026c78c303b31dd996b521faf589b5fa |
| `results/e63_occurrence_v2.json` (final) | 84fe64f4fd5c294a8d769bb760bfaef599fc0ed60817db74c14f3f509f06aa22 |
| `results/occ_units_v2_L7.npz` | a8cf7ab2f874855840e760e1f5663e76032eabe1b1220929cf1dc2007f11e457 |
| `results/occ_units_v2_L12.npz` | b84a7b13d52bd54acdc665175c7972b4e9418659a3e6b56d47bdc3bb895b5d5b |

**Why a reconstruction step is needed.**
- The frozen npz holds the D2 rows, every unit's scores (Y) and the ICA mixing matrices. It does
  **not** hold the PCA basis or the per-type maps.
- The D2 occurrence cache was deleted.
- S2 presence, R2 matching and steering all need D2's map. Re-extraction on MPS is not
  guaranteed to be bit-exact, and the PCA or global-rotation bases could flip sign or rotate
  inside near-degenerate eigenspaces. Either would silently decouple the frozen mixing columns
  from a recomputed W.

**Rule (`--phase reconstruct`):**
1. Re-extract D2 with v2's command, restricted to layers 7 and 12 (§3).
2. Recompute D2's rows and representation with the frozen code: `e63.represent_base`, then
   `e63.pca_white`, then `fit_maps`. `fit_maps` is `e63.type_whiten` with its maps kept, the
   same operations in the same order.
3. The phase asserts all of the following:
   - the rows equal the frozen rows;
   - `fit_maps` matches `e63.type_whiten` within 1e-4;
   - applying the kept maps to D2 reproduces the same W.
4. **Acceptance.** Regress each frozen unit's scores on [W′, 1] by least squares. The frozen units
   are accepted iff **R² ≥ 0.999 for every screened unit, at both layers and both kinds**.
   - Below that: **STOP**. No later phase runs, and the failure is reported.
5. **Kept for later phases** (`results/e64_maps_L{L}.npz`):
   - the PCA mean, basis and scales;
   - the pooled inverse square root S_pool^(−½), and Σ̂_t^(−½) for every type with n_t ≥ 32,
     keyed by token id;
   - the global rotation and scales;
   - every unit's readout, its R², and its **pattern** in W′, Cov(W′, ỹ);
   - its heavy-tail sign (§6.1);
   - its steering direction (§8.2).

All later geometry is in W′ coordinates and uses the patterns. That makes it immune to any sign
flip or rotation that the reconstruction absorbed.

## 3. Data

1,000 sequences × 255 tokens per set.

| set | content | role | command (`HF_HOME=hf_home`, offline) |
|---|---|---|---|
| D2 | corpus 1,000–1,999 (regenerated) | frozen units; labelling; maps | `e60_extract.py --source corpus --seq-start 1000 --n-seq 1000 --layers 7,12 --tag occD2` |
| S2 | D2's sequences, tokens permuted within each sequence (e60's seeded `--shuffle`) | context-shuffled presence; shuffled-text yield | `e60_extract.py --source corpus --seq-start 1000 --n-seq 1000 --shuffle --layers 7,12 --tag occS2` |
| R2 | reserve 2,000–2,999 | replication | `e60_extract.py --source reserve --seq-start 2000 --n-seq 1000 --layers 7,12 --tag occR2` |
| B | reserve 3,000–3,999 (tokens only) | behavioural steering contexts | none (read from `results/reserve_tokens.i32`) |

**Fixed rules for S2 and R2:**
- Each set uses only its own rows for row selection (positions ≥ 4, types with ≥ 8 rows) and for
  every first moment: type, document and position means, 3 cycles; the per-type means in Z; the
  global mean.
- They are mapped with **D2's** PCA, per-type and global maps.
  - Types that D2 equalised (n_t ≥ 32 in D2) use D2's Σ̂_t.
  - Every other type, including types absent from D2, uses D2's S_pool.
- This is v1 §4's rule ("PCA fitted on D, applied to S and R"), extended to the per-type step.
  It puts every set in one coordinate system, so unit cosines are meaningful.

## 4. Extended named-feature battery (fixed before any discovery label exists)

### 4.1 Unchanged

The v1 battery, `e61.feature_battery`, is used as frozen:
- paren_inside, bracket_inside, quote_inside;
- line_pos and sent_pos (graded);
- after_newline, list_item, after_colon, in_url, after_digit, in_caps_run;
- prev_punct, prev_capitalised, word_continuation;
- next_punct, next_newline, next_continuation;
- code_markup.

### 4.2 New features (exact rules, as coded in `e64.battery_ext_seq`)

**How the states are computed.**
- One character-level pass runs over each sequence's text: the concatenation of its tokens'
  individual decodes, which is e61's representation.
- Each token is assigned **the state of the text strictly before its first character**, the v1
  convention.
- A one-character lookahead may read the token's own first character. The residual at that
  position sees that token.
- Every sequence starts in the state 0, and at a line start.
- **Newline**: every new state resets to 0, and the current line becomes empty.
- **Sentence end**: a character in `. ! ?` that is not between two digits and is followed by
  whitespace, end of text, or one of `" ” ’ ' ) ]`. It resets dash_aside and comma_aside.

| feature | rule |
|---|---|
| **squote_curly_inside** | Depth counter. `‘` (U+2018) adds 1. `’` (U+2019) subtracts 1 only if depth > 0 **and** the next character is not alphanumeric, so `don’t`, `it’s` and `’re` do not close. Resets at newline. State = depth > 0. |
| **squote_straight_inside** (heuristic) | Depth counter. `'` **opens** if the previous character is whitespace or one of `( [ { " “ ‘ — – -`, and the next character is neither whitespace nor `'`. Otherwise `'` **closes** if depth > 0, the previous character is not whitespace, and the next character is not alphanumeric. Resets at newline. |
| **dash_aside** | Parity of aside dashes since the last sentence end or newline. An aside dash is one of: an em dash `—` (a run counts once); an en dash `–` with whitespace on both sides; a single `-` with whitespace on both sides; or a run of exactly two hyphens `--`, spaced or not. A dash that is the first non-space character of its line (a list bullet) never counts, and neither does a hyphen run of 3 or more. State = odd parity. |
| **comma_aside** | Parity of commas since the last sentence end or newline. A comma between two digits (`1,000`) does not count. State = odd parity. |
| **enclosure_inside** | paren_inside ∨ bracket_inside ∨ quote_inside ∨ squote_curly_inside: the task's list ( ), [ ], " ", “ ”, ‘ ’. The first three are v1's definitions, unchanged; quote_inside already counts `“ ”` as toggles. |
| **aside_inside** | enclosure_inside ∨ squote_straight_inside ∨ dash_aside ∨ comma_aside: the generic enclosure/aside feature. |

**Why the rules are built this way.**
- comma_aside and dash_aside are **prefix-only parities**. A unit is computed from the prefix, so
  a feature it could track must be a function of the prefix. A span rule that looks ahead to the
  closing comma would label tokens the model cannot yet know are inside an aside.
- The cost is that comma parity conflates asides with list and clause commas. That is why
  comma_aside sits only in aside_inside and never in enclosure_inside.
- squote_straight_inside is outside enclosure_inside because a straight `'` is mostly an
  apostrophe. It is in the battery because straight-quoted spans would otherwise go unnamed.

**Known misses, stated before any data:**
1. A curly closing `’` followed by a letter ends no span (`‘The Times’s`).
2. A plural-possessive `’` or `'` (`dogs’ bones`) closes an open span early.
3. A stray `‘` or `'` before a digit (`‘90s`) opens a span that only a newline closes.
4. Abbreviations (`e.g.`, `Mr.`) end a sentence for the parity features.
5. Pieces that decode to a partial UTF-8 byte are invisible to every rule.

### 4.3 Self-test (run while drafting; synthetic strings only)

`e64 --phase selftest` asserts the expected state before each token for 17 cases, including:
- `‘I don’t know’ she`, tokenised both with `’` and `t` as separate tokens and with `’t` as one;
- `well-known fact` (no dash);
- `\n- item one` (a bullet);
- `1990–2000` (unspaced: no dash) against ` yes – no`;
- `a--b--c`;
- `1,000 people`;
- `3.5 x, y`;
- newline resets.

All 17 pass.

## 5. Twins and families

FastICA's two contrasts often find the same direction: on D2, L7 logcosh #12 and skew #62 are both
the paren unit.

**Rule.** At each layer, a screened logcosh unit and a screened skew unit are **twins** iff each is
the other's best match by |cos| between the frozen mixing columns (same D2 W) **and** that |cos|
is ≥ 0.8. A **family** is a unit plus its twin, if it has one.

**Uses:**
- claims are counted by family (§10);
- a unit's sibling null excludes its twin (§8);
- multiplicity is reported per unit tested, and the family count is given beside it.

## 6. Labelling on D2 (`--phase label`)

### 6.1 Battery route (mechanical)

For every screened unit u at L7 and L12, with y = its frozen D2 scores:

**Tails.**
- The tails are the top and bottom 1% of rows: 2,034 of 203,328.
- The **heavy tail** is the side with the larger |1% / 99% quantile|, v1's e61/e62 rule.

**NAMED** iff some extended-battery feature f meets one of these:

| feature type | criterion |
|---|---|
| binary, with ≥ 50 positive and ≥ 50 negative rows | **|AUROC(y, f)| ≥ 0.75**, **or** in either 1% tail, f's rate ≥ **3×** its base rate **and** ≥ **0.10** |
| graded (line_pos, sent_pos) | **|Spearman ρ| ≥ 0.5** |

Otherwise the unit is **battery-UNNAMED**. Binary features below 50 positives are reported but
not used.

**Why there is an enrichment clause.** v1 §8's directional criteria (|AUROC| ≥ 0.75 or |ρ| ≥ 0.5)
**would not name half of the control units**: L12 paren 0.736 and 0.732, L12 quote 0.747 and
0.745. Those units are known-variable directions; their top-1% tails are 50% paren and 90% quote.

**Why the 0.10 minimum.** A 3× enrichment of a 0.07–0.24% feature is a handful of rows, which
noise can produce. With the 0.10 floor, 3× binds only for features above a 3.3% base rate.

**Calibration from numbers already in REPORT (no new data).** This rule names all six recovering
units:
- L7 logcosh #12 and skew #62: via |AUROC| 0.779 and 0.777;
- L12 logcosh #11 and skew #23: via the top 1% being 50.7% and 49.2% paren, against a 3.1% base
  (about 16×);
- L12 logcosh #33 and skew #33: via 91.2% and 89.8% quote, against a 6.0% base.

The phase hashes its output and then **asserts that these six are NAMED**. If not, **STOP**: the
rule has failed its own anchor.

### 6.2 Context route (judgment; conservative only)

**Written by the label phase**, for every unit:
- 12 contexts from each 1% tail (fixed seed: 5 + 1000·L + id);
- its eight commonest tail token types.

**Procedure.**
1. The label output is hashed first.
2. For every battery-UNNAMED unit, the analyst writes into `results/e64_context_reading.json`
   either a short construction name and the tail it lives in, or "none".
3. `--phase seal-reading` asserts that every unit has an entry, and hashes the file.

**Rule.** A reading can only move a unit from UNNAMED to **NAMED_BY_CONTEXT**, and never the
reverse. It is sealed before any S2, R2 or behavioural number exists. NAMED_BY_CONTEXT units go
through tiers 2–3 and are reported, but they are not discovery claims. (A context-named unit is
"a variable outside the battery", not "a variable no one has named".)

### 6.3 Identity diagnostics (reported for every unit; not gates, since the v2 screens are frozen)

| diagnostic | definition |
|---|---|
| **tail-TDI** | For heavy-tail rows: the rate at which a random same-type occurrence (self excluded) is also in the heavy tail, divided by 1%. The directional analogue of cluster TDI. |
| tail composition | number of tail types; largest single-type share; top 8 types |
| **identity-suspect** flag | tail-TDI > 3 (the frozen cluster cap) or largest type share > 0.30 (the frozen cluster share cap) |

### 6.4 Extended features on the control units

The extended-battery table is reported for the six control units: enclosure_inside,
aside_inside and the components. This is **description only**. The v2 positive-control verdict
is final, and no second control gate exists.

## 7. S2: context-shuffled presence and shuffled-text yield (`--phase shuffled`)

**Representation.** S2 uses its own rows and first moments, then D2's maps (§3). On W_S2, both
ICA kinds are fitted **independently** with the frozen v2 unit code (`e63.ica_units`: seeds
0/1/2, the retry rule, the same screens).

### 7.1 Presence

**Definition.** presence(u) = the max |cos| between u's pattern and the S2 mixing columns of the
same kind.

**Rule.** presence ≥ 0.8 labels u **context-generic**. As in v1 §5, this is **a label, not a
disqualification**: counting variables such as paren depth remain computable from a shuffled
multiset of tokens, so a real in-context variable can be present in S2.

**Reported:**
- the control units' presence, as the calibration of what "present" means;
- the chance level: the p99.9 of the best |cos| for 10,000 random directions.

### 7.2 Shuffled-text yield

**Definition.** The number of S2 units that pass the frozen screens on S2 **and** are
battery-UNNAMED, with the extended battery computed on the shuffled tokens. It is counted per
layer and kind.

**Use.** It is reported beside D2's battery-UNNAMED count, as the instrument's yield on text
with no natural order: the empirical false-flag rate for tier 1. It is not a per-unit gate.

KIR on W_S2 is reported.

### 7.3 Arrangements: not run

**Why:**
- v1 §5's ring and ruler screens run over ≥ 8 screened **cluster centroids**. Clusters are
  excluded, with 0 / 2 / 0 screened.
- Directional units have no centroids. Their tail means are close to scaled mixing columns,
  which are near-orthogonal in whitened W by construction. Ring and ruler screens over a near-
  orthogonal set are degenerate.

**Consequences.** No arrangement claim is possible in this tier, and S2 has no arrangement role.

## 8. Behavioural tier (`--phase behaviour`, MPS)

### 8.1 Units

At each layer, the following are evaluated in full:
1. the **anchor**, first:
   - **L7 logcosh #12** (paren);
   - **L12 logcosh #33** (quote).
   Each is its layer's recovering unit with the highest |AUROC|, logcosh before skew.
2. the other recovering units, as secondary anchors (reported);
3. every battery-UNNAMED unit. NAMED_BY_CONTEXT units are run and reported separately.

**Anchor rule.** If the anchor fails, **no behavioural verdict is issued at that layer**, as in
v1 §9.2.

### 8.2 Direction

**The problem.** v2 has no single linear map from raw residual space to W, because the per-type
step is type-specific. e62's v1 rule, U·(m·sd), assumed one.

**Rule.** d_u = sign_u · normalise( U_pca · diag(sd) · Cov(Z, ỹ_u) ):
- Z = D2's whitened PCA-64 coordinates **before** the per-type step;
- ỹ_u = u's standardised frozen scores;
- sign_u points toward the heavy tail;
- d_u is embedded at the kept coordinates, with 0 at the dropped ones, in 1,152 dimensions.

**Why this rule.**
- It is the residual-space pattern of the unit: the average direction along which the residual
  moves as y_u rises.
- With a single linear map it reduces exactly to e62's rule.

### 8.3 Signature, sites and intervention

These are e62's, unchanged except where noted.

| item | rule |
|---|---|
| signature T_u | The 20 next tokens most over-represented after u's heavy-tail D2 rows, against token-matched expectation; each token's count must be ≥ 5. |
| random set R_u | 20 tokens frequency-matched (corpus count within ±10%), with a fixed seed. |
| **sites (changed)** | 200 random (sequence, position) sites from **reserve 3,000–3,999**, at position ≥ 5 of the BOS-included array; seed 6462 + L. |
| prompt | BOS, up to 64 preceding tokens, then the site token. |
| intervention | h ← h + ε‖h‖·d_u at the output of decoder layer L (the extraction hook point), at the site position only; ε ∈ {0.005, 0.01, 0.02}. |
| readout | Δ = [lse(logits[T_u]) − lse(logits[R_u])] steered, minus the same unsteered. |

### 8.4 Nulls, at the same 200 sites

1. **10 random directions** orthogonal to d_u, scored with u's own signature, at ε = 0.02.
2. **Sibling bank (changed).** Every other screened unit at the layer, of **both kinds**,
   excluding u's twin. Each is scored with its own direction and own signature, at ε = 0.02.
   - That is 63–77 siblings per unit, against v1's sample of up to 30.
   - The bank is computed once per layer and reused.

### 8.5 PASS rule (v1 §9.2)

PASS needs all three:
- **sign consistency ≥ 0.8** at some ε: the per-type mean Δ over the 20 commonest site token
  types is > 0 for at least 80% of them;
- mean Δ at ε = 0.02 **> the random maximum**;
- mean Δ at ε = 0.02 **> the sibling p99**, computed with `method="higher"`. With fewer than 100
  siblings this is the sibling maximum.

### 8.6 Expected false passes, and a correction to v1 §11

**The bound.** Under exchangeability with the siblings, a unit beats the sibling maximum with
probability ≤ 1/(n_sib + 1). So the expected number of false passes at a layer is
≤ Σ_u 1/(n_sib,u + 1). That is about 1.3–1.6% per candidate.

**The correction.** v1 §11's "units × 0.01" was too low for v1's own design. The p99 of ≤ 30
siblings is their maximum, so the bound there was about 3%.

**Layer stop rule.** If the number of PASSes among discovery candidates at a layer is ≤ this
bound, tier 2 shows no excess over chance at that layer, and **no discovery is claimed there**.

## 9. Replication on R2 (`--phase replicate`)

**Representation.** R2 uses its own rows and first moments, then D2's maps (§3). Both kinds are
fitted independently with the frozen v2 unit code (`e63.ica_units`, screens unchanged). This is
v1 §9.3's "independent fit on R": fitted on R2's rows, in D2's coordinates.

**For each D2 unit u**, let v be the best R2 unit of the same kind by |cos| between u's pattern and
v's mixing column. u replicates iff all three hold:
1. **|cos| ≥ 0.8**;
2. **the screens hold**: v passes the frozen screens on R2 (η²(y) and η²(y²) on R2's token types,
   and R2 split-half |cos| ≥ 0.8);
3. for discovery candidates, **v is battery-UNNAMED on R2** under §6.1. There is no context route
   on R2.

**Replication anchors.**
- A control unit replicates iff it matches at ≥ 0.8, its match passes the R2 screens, and the
  match reaches |AUROC| ≥ 0.70 for the control's label on R2. 0.70 is the v2 floor.
- If **no** control unit replicates at a layer, the replication tier is **uninformative** there,
  and no discovery is claimed at that layer.

**Reported:**
- the chance level: the p99.9 of the best |cos| for 10,000 random directions against R2's
  mixing columns (about 0.5 is expected in 64 dimensions);
- KIR on W_R2;
- the number of screened units on R2.

## 10. Tiers and the claim (`--phase verdict`)

| tier | requirement |
|---|---|
| 1. geometric candidate | Screened (frozen v2), battery-UNNAMED (§6.1), and not NAMED_BY_CONTEXT (§6.2). |
| 2. behavioural | PASS (§8.5), with the layer's anchor passing and the layer's PASS count above the §8.6 bound. |
| 3. replicated | Replicates on R2 (§9), at a layer where replication is informative. |

**DISCOVERY = all three.**

**How claims are counted and labelled.**
- Claims are counted **by family**: a family is a discovery if any member clears all three tiers.
- Every claim carries its labels:
  - context-generic (S2 presence ≥ 0.8);
  - identity-suspect (§6.3);
  - its S2 and R2 cosines;
  - its tail composition.

**Other outcomes** are reported under their own names and are not discoveries:
- "replicated geometric" (1 + 3);
- "behavioural + geometric" (1 + 2);
- "geometric candidate";
- NAMED_BY_CONTEXT.

**Wording if no unit clears all three.** "No unnamed occurrence variable found at L7/L12 by
this instrument." Not "none exists": the instrument's yield is bounded by its 0.73–0.78
recovery of known variables.

## 11. Multiplicity

**There are 143 screened units. The candidates are the battery-UNNAMED subset, counted by unit
and by family.**

| tier | false-pass control |
|---|---|
| 1 | No permutation null exists for "unnamed". S2's shuffled-text yield (§7.2) is the empirical comparator, reported beside D2's count. |
| 2 | Expected false passes ≤ Σ 1/(n_sib + 1) per layer (§8.6), with the layer stop rule. |
| 3 | A chance match at |cos| ≥ 0.8 is far above the random-direction p99.9 (reported). A false replication therefore needs real but non-contextual structure, which is what the S2 label and the identity diagnostics report. |

**The honest error rate of a discovery claim.** It is the tier-2 bound. A unit can be a real,
replicating direction whose behavioural pass is chance. The verdict reports that bound beside
every claim.

## 12. KIR and identity

Residual identity on the v2 representation is still **9–15× chance by KIR**: D2 14.7 at L7 and
9.3 at L12.

**What that means.**
- A kNN neighbourhood still shares token type far above chance.
- η²(y²) ≤ 0.30 bounds the share of a unit's y² variance that type explains. It does not bound
  type-specific higher-order structure.

**Reported:**
- KIR on D2 (frozen), S2 and R2, all in D2 coordinates;
- for every candidate, tail-TDI, tail type count, largest type share, top types and the
  identity-suspect flag (§6.3).

An identity-suspect discovery is reported as such, and it is never the headline.

## 13. Integrity

| file | sha256 |
|---|---|
| `scripts/e64_occurrence_v2_discovery.py` | 1d548ada0a1b5e765b2e8be43f5151b8a7b9f2c7d1087a878c600c44a9e94230 |

**What e64 asserts.**
- **At every data phase:**
  - this file's sha256 is in `docs/HASHES.txt`;
  - its own sha256 is in this text;
  - every §2 input matches its sha256;
  - the layers are restricted to {7, 12}.
- **Before each phase:** the phase's predecessors' outputs are in `docs/HASHES.txt`.

**What each phase appends to `docs/HASHES.txt`.**

| phase | appends |
|---|---|
| reconstruct | `e64_maps_L{7,12}.npz`, `e64_reconstruct.json` |
| label | `e64_labels.json`, before the reading template is written |
| seal-reading | `e64_context_reading.json` |
| shuffled | `e64_shuffled.json` |
| behaviour | `e64_behaviour.json` |
| replicate | `e64_replicate.json` |
| verdict | `e64_verdict.json` |

## 14. Order, commands and resources (each step needs the user's go-ahead)

All commands run from `~/Downloads/gf-discovery-ext` with `HF_HOME=hf_home`, offline.

**Precondition for every MPS step.** Swap used must be ≤ 8 GB (`sysctl vm.swapusage`). e64
asserts it at every data phase. At freeze time swap was 10.6 GB used, so free memory or reboot
first.

**Order:**
1. D2 re-extraction (§3), about 3 min MPS.
2. `e64 --phase reconstruct`. Accept or STOP.
3. `e64 --phase label`. Asserts that the control units are NAMED.
4. The analyst's context reading, then `e64 --phase seal-reading`.
5. S2 extraction, then `e64 --phase shuffled`.
6. `e64 --phase behaviour`, MPS: anchor first.
7. R2 extraction, then `e64 --phase replicate`.
8. `e64 --phase verdict`.
9. Delete the occD2, occS2 and occR2 caches.

**Resources:**

| item | estimate |
|---|---|
| extractions | three, about 3 min each |
| CPU phases | a few minutes each |
| behaviour | each screened unit once at ε = 0.02, plus 13 steered passes per evaluated unit, 200 prompts each: about 30–90 min on MPS |
| disk | caches about 1.2 GB per set at two layers (3.5 GB); maps about 0.1 GB; the folder peaks near 10.5 GB, under the 20 GB cap |
