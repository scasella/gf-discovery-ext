# PREREG — SAE-gap round: do the e69 pilot survivors hold on fresh text? (frozen 2026-10-15, before any fresh text is read)

**What this is.** The full preregistration licensed by the e69 pilot's GO
(`results/e69_pilot.json`, sha `05dcc30f…`).
- **Authority.** The user asked for it ("write the full fresh-text prereg for the e69 survivors,
  with the judges' grafts, review it, and freeze it. Don't extract any fresh data until I
  approve").
- **What freezing does and does not do.** Freezing this document extracts nothing. Every
  fresh-text phase below starts only on the user's explicit go-ahead.
- **Earlier verdicts.** No verdict of e64–e69 is amended.

**Sources.**
- **Design:** `sae_error`, chosen by 3 of 3 judges; record in
  `docs/sae_gap_calibration/design_panel_result.json` (manifest sha `6ed34758…`).
- **Pilot plan:** `NOTE_sae_gap_pilot.md` (sha `4a1bd995…`).
- **Pilot result:** P0–P2 in `REPORT.md` ("SAE-gap pilot (e69)").

**What has been seen (all on burned D2 and S2).**
- The three D2 slow units and every pilot number: the shuffle ratios, the predictability, bag and
  sub-threshold numbers, the labels.
- No fresh-text number of any kind exists.
- The thresholds below are the pilot's frozen bars wherever the pilot had one. New thresholds are
  marked **new**.

---

## 1. Objects (frozen now)

**Error representation.** Exactly the e69 instrument:
- e = h − (JumpReLU_fp32(h) @ W_dec + b_dec) with `resid_post_all/layer_12_width_16k_l0_big`.
- h is the raw L12 residual (output of decoder layer 12), stored as fp16 and read back, exactly as
  D2's `results/e67_raw_L12_rows.npy`. Coordinate 1038 is dropped after the error is formed.
- Kept rows: positions ≥ 4, token types with ≥ 8 rows in the set.
- Identity removal: type, then document, then position means, cycled 3 times (`demean3`).

**Primary units: the two e69 survivors, fixed.**
- These are D2 AMUSE units **#1 and #2** from `results/e69_p0_units.npz` (sha `21746c68…`).
- Each has a frozen D2 filter f_u (`filt[:, u]`) and pattern p_u (`pat[:, u]`), both 1,151-d
  residual-space vectors.
- On any new set, the unit's score is y_u = demean3(e_set) · f_u. Nothing is refitted to define
  the primary units.
- This makes the primary arm a **fixed-direction confirmation**. Rotation between the
  near-degenerate units cannot move it.

**Secondary (exploratory) arm.**
- The full instrument is refitted on D3: PCA-256 whitening, multi-lag AMUSE (τ ∈ {1, 2, 4, 8, 16}),
  m = 32, split-half against the top 64, and the e69 screens.
- Any screened D3 unit that matches neither p₁ nor p₂ (|cos| < 0.8) is **exploratory**.
- Exploratory units are reported. They are never steered under this prereg and can never yield a
  claim here. A later prereg would be needed.

## 2. Data (all from `results/corpus_tokens_pass1.i32`; none used before for occurrence work)

| set | corpus sequences | role | layers stored |
|---|---|---|---|
| X0 | D2 = 1,000–1,999, re-extracted | extraction identity check (burned, not fresh) | L12 |
| **D3** | 2,000–2,999 | primary tests, controls, naming, exploratory refit | L11, L12, L13, plus next-token entropy and log p |
| **S3** | D3's tokens shuffled within each sequence, using e60's rule (seed 600,000 + 2,000 + i) | context-shuffled null | L12 |
| **R3** | 3,000–3,999 | replication | L12 |
| B5 | 4,000–4,999 | steering test block | none stored; prompts only |
| B6 | 5,000–5,999 | held-out steering block | none stored; prompts only |
| C | 6,000–6,999 | confirmation; **only under a separate prereg frozen before it is touched** | — |

**Duplicate screen.** A sequence's key is its first 32 tokens.
- A new-set sequence is dropped if its key matches a sequence in any used set: corpus 0–1,999
  or 13,000–14,999, or reserve 0–7,999.
- It is also dropped if its key matches an earlier sequence in the new sets, taken in the order
  D3, R3, B5, B6, C.
- The designers' count was 0–1 per set. Dropped indices are recorded. Sets are not topped up.

**Storage.**
- Raw rows are fp16 at kept rows (about 0.47 GB per layer per set).
- **Coordinate 1038 is also stored in an fp32 sidecar** at every layer, because fp16 overflows
  above 65,504.
- L12 primary computations use the fp16 value, so that D3 matches the D2 instrument exactly. The
  sidecar is used at L11 and L13, and it is asserted to be < 65,504 at L12 (D2's maximum is about
  63k).
- If any L12 value overflows, that row's 1038 is taken from the sidecar, and the count is
  reported.

**Extraction.**
- A new script, `scripts/e70_sae_gap_geometric.py`, does the extraction with e67's capture path:
  model in bf16, MPS, batch 16, `model.layers.N.output`.
- **X0** re-extracts D2 L12 and must reproduce `e67_raw_L12_rows.npy` (≥ 99% identical values and
  per-coordinate corr ≥ 0.999). Otherwise **STOP**.
- Next-token entropy and log p for D3 come from the same model, with logits processed 2 sequences
  at a time (fp32 log-softmax).
- Swap is checked before **every batch**. The batch loop stops cleanly at > 12 GB of swap, and the
  phase is resumable.

## 3. Primary tests (per primary unit u ∈ {1, 2}; on fresh text)

Every bar here is fixed now. "Pass" means all of the listed parts.

**G1. Persistence (D3, and R3 for replication).**
- lag-1 acf of y_u ≥ 0.20 (the e69 screen bar).
- Reported: acf at lags 1, 2, 4, 8, 16 and 32, and the variance of y_u relative to D2.

**G2. Context-specificity (D3 against S3).** All three of:
- (a) **Transfer.** acf₁(S3)/acf₁(D3) < 0.50, and acov₁(S3)/acov₁(D3) < 0.50, both under the
  fixed filter. These are the pilot's GENERIC-b bars, as a pass condition.
- (b) **Refit match.** No S3-refitted AMUSE unit among the top 64 has pattern |cos| ≥ 0.80 with
  p_u.
- (c) **Inflation guard (new).** S3's mean log‖e‖ exceeds D3's by < 0.25 (natural log). Otherwise
  G2 is "undetermined" for every unit, because shuffled text would then have changed the error
  itself too much to compare.
- Reported: var(y_S3)/var(y_D3), FVU, L0 and mean log‖h‖ on D3 against S3.

**G3. Geometric reproducibility (D3, and R3 for replication).** Either of:
- **Unit match:** an AMUSE unit refitted on the set, among its top 64, has pattern |cos| ≥ 0.80
  with p_u.
- **Subspace capture (new; a guard against rotation among near-degenerate units):** c_u ≥ 0.80
  **and** c_u > the p99.9 of the null.
  - c_u = ‖Q₈ᵀ p_u‖², where Q₈ is an orthonormal basis of the set's top-8 AMUSE patterns.
  - Null: 1,000 random directions g ~ N(0, I₂₅₆) in the set's whitened error PCA-256, pulled back
    as U diag(sd) g and normalised (seed 7040).

**G4. Unnamed (D3).** u is NAMED if any of these holds:
- **Predictability.** |Spearman| ≥ 0.30 with next-token entropy, −log p, or their causal running
  means over 4 and 16 positions. Each is tested raw and identity-removed on non-final rows.
- **Position.** |Spearman| ≥ 0.30 with line position, sentence position, or tokens since the last
  punctuation. These are `e61.feature_battery`'s `line_pos` and `sent_pos`, plus the punctuation
  count.
- **Battery.** |AUROC| ≥ 0.65 for any binary feature of `e61.feature_battery` with ≥ 50 positives
  and ≥ 50 negatives.
- **Parse rules.** |AUROC| ≥ 0.65 for any of the 16 e64c rules on a D3 `en_core_web_trf` parse.
  These are e64c's `lexical` and `parse_rules` functions, unchanged and imported; D3 pieces are
  decoded one token at a time as in e64c.
- **Bag of previous tokens.** Held-out R² ≥ 0.30 (train on even sequences, test on odd), using
  e69's construction exactly:
  - windows W ∈ {8, 16, 32, 64};
  - a 128-d type embedding: the PCA of D3's own per-type mean raw L12 residual;
  - features passed through `demean3`;
  - two predictors: ridge (RidgeCV α ∈ 10^[−2..4]) and an MLP (hidden (256,), early stopping,
    seed 0).
- **SAE state.** |Spearman| ≥ 0.30 with the identity-removed log‖h‖, log‖e‖ or L0.
- **Tail diversity (new).** Either 1% tail of y_u covers < 20 token types or < 20 documents.

**G5. Error share (D3).**
- Split the fixed filter's raw-residual score exactly: y_raw = demean3(h_keep)·f_u =
  y_e + y_sae, where y_e = demean3(e)·f_u and y_sae = demean3(SAE(h)_keep)·f_u.
- ES = Cov(y_e, y_raw)/Var(y_raw).
- **Require ES ≥ 0.50 under both** the 16k l0_big SAE (the source) **and** the 16k l0_small SAE,
  whose own error e_small replaces e.
- Reported: ES for 32 RAND-E directions (seed 7020; random in D3's whitened error PCA-256, pulled
  back as above).

**Geometric verdict.**

| verdict | requires |
|---|---|
| **REPLICATED CONTEXT-SPECIFIC ERROR COMPONENT** | G1, G2, G3, G4 (unnamed) and G5 pass on D3, **and** G1 and G3 pass on R3 |
| PERSISTENT BUT GENERIC | G1 passes; G2 fails |
| NAMED | G4 names it |
| NOT REPRODUCED | G1 or G3 fails on D3, or on R3 |
| SAE-SPAN | G5 fails |

**Multiplicity.** There are 2 primary units, and each must pass every gate, so no correction is
needed at the geometric tier. The exploratory arm is excluded from all claims.

## 4. Controls (on D3)

**C1, knockout control.**
- e_KO = e + Σ_{k∈K} z_k W_dec[k], with K the 20 frozen latents from the design:
  - paren: 2682, 7780, 4001, 4493, 1559, 8951, 3503, 1442, 507, 1056;
  - quote: 5254, 1229, 1994, 1534, 564, 1882, 7302, 10306, 398, 1550.
- The exploratory instrument runs on e_KO.
- **Recovered** = a screened unit with |AUROC| ≥ 0.70 for `paren_inside` or `quote_inside`
  (`e61.feature_battery`), above the p99.9 of `e63.null_engine` (1,000 token-matched
  permutations).
- C1 gates the exploratory arm and any absence statement. It does not gate the primary
  fixed-direction tests.
- The recovered KO units are also the coverage tier's error-space positives (§6).

**C2′, planted in h and re-encoded (descriptive; quoted with any absence statement).**
- The plant is a synthetic two-state Markov variable, independent of the text:
  - per sequence, seed 7030 + i;
  - stationary rate 0.06, mean run 16 tokens.
- It is added to the raw h at κ · σ along a unit direction v, with κ ∈ {1, 2, 4}. σ is the SD of
  identity-removed raw h along v.
- v is either (i) random within D3's top-256 error PCs (as a 1,151-d vector), or (ii) random at
  full rank (seed 7031).
- The SAE is then **re-encoded** and e recomputed, and the instrument runs.
- Recovered = a screened unit with |AUROC| ≥ 0.70 against the plant, above 100 sequence-block
  permutations of the plant.
- Absorption is reported: the share of the planted variance along v that the SAE reconstructs.

**C3, extraction sanity.** X0, above.

## 5. Behavioural tier (B5, B6). Separate go-ahead; `scripts/e72_*` is frozen before it runs

The rules are fixed here. The script must implement them verbatim, and is reviewed and hashed
before B5 is touched. It runs only for primary units with a REPLICATED verdict.

**Rig.** e65's frozen code, imported unchanged:
- site drawing: 200 sites per block, one position uniform in [5, 255] of the BOS-included row,
  with prompt = BOS + up to 64 preceding tokens + the site token; site seeds 7512 (B5) and
  7612 (B6);
- natural σ;
- signature scoring, Δ, sign consistency.

**Two RAND banks.**
- **Anchor bank.** The anchor is e65 L12 logcosh #33 (quote). It is scored against e65's validated
  decoded 64-d pullback bank (`e65.rand_bank`, 99 directions; seeds 7522 for B5 and 7622 for B6).
- **Unit bank (RAND-E).** The primary units are scored against 99 directions g ~ N(0, I₂₅₆) in
  D3's whitened error PCA-256, pulled back and normalised (seeds 7523 for B5, 7623 for B6). Each
  is dosed by its own σ.

**Push.**
- Direction: d = p_u, with 1038 zeroed, unit norm.
- Dose: c·σ_d, where σ_d is the SD of the **raw** residual along d over block prompt positions
  ≥ 4; c ∈ {0.5, 1, 2, 4}, with the gate at c\* = 2.
- **Dose-void (new):** a realised dose below 0.9× the requested one (after the bf16 cast) gives no
  verdict.

**Orientations.**
- Both signs are tested: +d is scored against T⁺ (the upper 1% of D3) and −d against T⁻ (the lower
  1%). Signatures are built with `e64.signature`.
- Each orientation is its own test.
- An empty signature (no token with count ≥ 5) **never passes**. |T| is reported for every unit
  **before** any steering.

**SIG pool.**
- the signatures of the 32 exploratory D3 units in both tails, excluding u's own;
- plus the 78 frozen v2 L12 signatures in `e65_frozen.json`.

**PASS (per orientation, per block).** All of:
- Δ > 0;
- sign consistency ≥ 0.8 at some c ≤ 2;
- Δ > the unit-bank RAND maximum;
- Δ > the SIG maximum.

**Order.**
1. the unsteered pass;
2. the anchor; it must PASS on B5, or there is no behavioural verdict;
3. the units.
4. B6 then repeats the anchor, followed by each B5 pass, with fresh sites, σ and banks.

**Behavioural lead** = PASS on both B5 and B6. The expected false passes Σ 1/(n′+1) over the tests
run are reported.

**Reported, not gating:**
- σ_d(raw)/σ_d(e);
- the ES split of the pushed direction on D3; a push with ES < 0.5 is labelled "not
  error-attributable";
- a window push at c\* (positions site−8 to site) for the anchor and the units;
- the full next-token KL at the site;
- a supervised anchor: the C1 KO-quote unit's pattern, steered the same way.

## 6. Coverage tier (no fresh text). `scripts/e71_*` is frozen before it runs

It runs for every primary unit with a REPLICATED verdict, on D3 rows. It runs descriptively for
the others.

**Dictionaries.**
- L12 `resid_post_all`: 16k big and small (on disk); 262k big and small (each downloaded, used and
  deleted).
- L11 and L13 `resid_post_all`: 16k big and small.
- L13 `resid_post`: 16k medium, 65k medium and 65k big (on disk), and 262k medium (downloaded,
  used and deleted).
- Before each download, the folder plus the file must be ≤ 20 GB, or the script stops.

**Classes per (unit, dictionary).**

| class | requires |
|---|---|
| **SINGLE-COVERED** | any latent with activation \|r\| ≥ 0.70 or pre-activation \|r\| ≥ 0.70 with y_u; or decoder \|cos\| ≥ 0.70 with p_u (masked) plus e68's tail-overlap match ("an existing latent, mis-reconstructed") |
| **DISTRIBUTED-COVERED** | held-out R² ≥ 0.50 (train on even sequences, test on odd; RidgeCV) of y_u from the identity-removed features {a_j, clip(pre_j, 0, θ_j)} of the K latents with the largest even-half \|r\| (activation or pre-activation), for K ∈ {128, 512} |
| **WIDTH-COVERED** | ES < 0.50 under a 262k SAE's error (G5's split with that SAE) |
| **NOT COVERED** | none of the above in any dictionary |

**Calibration gates.** These are the e68 lesson: the tier must see what is known to be there.
- **Gate A (D2, decoded space, already-used data).**
  - The SINGLE and DISTRIBUTED classes, using every dictionary **except** the source L12 16k big,
    must flag ≥ 5 of 6 known positives as covered.
  - The 6 positives: L12 logcosh #13, skew #16, and the controls logcosh #11, skew #23,
    logcosh #33 and skew #33. Each uses its frozen D2 score.
  - Reported, not gating: the rate for the 40 e68 joint-null decoded directions.
- **Gate B (D3, error space).**
  - Every recovered C1 KO unit must be flagged covered by the source dictionary.
  - At most 10% of 32 RAND-E directions may be flagged covered, in any class.
- If either gate fails, coverage is **UNDETERMINED**, and no "not covered" statement is made.

## 7. Claims

| claim | requires |
|---|---|
| Replicated, context-specific, unnamed persistent component of the L12 SAE error | the REPLICATED verdict (§3) |
| Causal lead | that, plus a behavioural lead (§5) with the anchor passing |
| Candidate discovery | that, plus NOT COVERED (§6) with both gates valid |
| **Confirmed** | that, plus a separate prereg on corpus 6,000–6,999, frozen before C is touched |

Exploratory units yield no claim under this prereg.

## 8. Order, stops and approvals

| step | what runs | approval | stop |
|---|---|---|---|
| 1 | X0, then D3 (L11/L12/L13 + entropy) and S3 extraction | **user go-ahead 1** | X0 fails → STOP |
| 2 | D3 decode and trf parse (`.venv-parse`), then the geometric phase on D3/S3 (G1–G5, C1, C2′, exploratory arm) | — | no primary unit passes G1–G5 on D3 → STOP; R3 is not extracted |
| 3 | R3 extraction and the replication phase | — (covered by go-ahead 1) | no REPLICATED unit → STOP; B5/B6 are not touched |
| 4 | Coverage (`e71`, frozen before it runs) | — (no fresh text; downloads under the cap) | gates fail → UNDETERMINED |
| 5 | Steering B5, then B6 (`e72`, frozen before it runs) | **user go-ahead 2** | anchor fails → no behavioural verdict |
| 6 | Confirmation on C | **a new prereg and user go-ahead 3** | — |

**Rules for every phase.**
- Hashes are appended with the e64 helper only.
- Each output is hashed before the next phase reads it.
- Nothing is edited after hashing.
- Local MPS only. Check swap and disk before every phase, and confirm with `ps` that no other
  job of this project is running.
- Long jobs are launched detached (`start_new_session`) under `caffeinate`, with a 12 GB swap
  watchdog, because other sessions share this 24 GB Mac.

## 9. Resources

- **Disk.** The folder is 13 GB now.
  - D3 adds 3 layers and S3 and R3 add 1 layer each, at about 0.47 GB per layer: about 2.4 GB.
  - Each 262k download adds 2.42 GB at a time.
  - Peak is about 18 GB, under the 20 GB cap.
- **Time (estimate).**
  - Extraction: about 15 min of MPS.
  - D3 trf parse: 20–40 min of CPU.
  - Geometric phase: about 30 min of CPU (C1's 1,000 permutations dominate).
  - Coverage: about 30 min.
  - Steering: about 45 min of MPS.

## 10. Prior odds (stated before any fresh number)

The design's chain was about 0.02–0.03 before the pilot. The pilot passed its shuffle test
narrowly (46% kept against a 50% bar), on burned data.

| step | probability |
|---|---|
| G2 holds on fresh D3 | about 0.5 |
| G1, G3 and G5, and replication on R3 | about 0.6 |
| stays unnamed, with the D3 parse and full battery | about 0.8 |
| behavioural lead, given diffuse signatures | about 0.25 |
| NOT COVERED with valid gates | about 0.5 |

The product is **about 0.03** for a candidate discovery, and about 0.24 for the geometric-tier
claim alone.
