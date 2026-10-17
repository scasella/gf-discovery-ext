# Stage 4 v2 discovery tier, addendum D1: swap precondition raised (2026-10-15)

This addendum was written after `PREREG_occurrence_v2_discovery.md` was frozen (sha `1e24fcdf…`)
and before any of its phases ran. At that point no D2 re-extraction, S2, R2, label or steering
existed.

## Change

**What changed.** The swap precondition in prereg §14 rises from **≤ 8 GB** to **≤ 16 GB**, at
the user's request and with their approval. In `e64`, `SWAP_MAX_GB` goes from 8.0 to 16.0.

**Why it is safe.** This is a resource guard only and has no effect on the analysis:
- it changes no data, rule, threshold, seed or unit;
- v2's extractions ran with swap up to about 13 GB with no issue.

**Free disk at this point:** 45 GB.

## Code change needed to record it

`e64` now:
- checks that this addendum's sha256 is in `docs/HASHES.txt`;
- accepts its own sha256 if it appears in the prereg **or** in a hashed addendum.

Nothing else in e64 changed, and `--phase selftest` still passes all 17 cases.

| file | sha256 |
|---|---|
| `scripts/e64_occurrence_v2_discovery.py` (D1 version) | 1c97e42b4fc2fbf64410e8dda39fb8a9d6ae938e3fa960e6bf36d1a009fcf1f4 |
| superseded e64 (prereg §13) | 1d548ada0a1b5e765b2e8be43f5151b8a7b9f2c7d1087a878c600c44a9e94230 |
