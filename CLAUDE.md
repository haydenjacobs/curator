# CLAUDE.md — rules for working on Curator

Curator is a deterministic, content-based book and movie recommender built on
the Tag Genome Dataset for Books and the MovieLens Tag Genome Dataset 2021
(GroupLens). The owner is non-technical: explain in
plain language, give exact commands and say what each does. No UI yet; the
focus is data architecture and algorithm correctness.

## Non-negotiable rules

- **Deterministic.** Same input → identical output. No randomness, no AI/LLM
  judgment inside scoring. Ties are broken by medium, then id.
- **Loud errors over silent failures.** If data doesn't match expectations
  (row counts, missing items, scores outside 0–1, duplicate item-tag pairs,
  unknown tags, missing values), stop with a clear message. Never guess or
  fill gaps. Known defects in the published data are accepted only when
  declared exactly in `config.py` (the affected items are left out, never
  repaired); the build stops if a declared defect isn't there. Lookup never
  auto-picks among several matches.
- **Correctness over speed.**
- **One config file.** Every tunable number lives in `config.py` with a comment.
  No magic numbers elsewhere.
- **Raw vs processed.** Raw data stays untouched in `data/raw/` (movies in
  `data/raw/movies/`). `build.py` creates `data/processed/books/`,
  `data/processed/movies/` and `data/processed/shared_tags.csv`. The scoring code (`curator/data.py`,
  `curator/lookup.py`, `curator/engine.py`) reads ONLY processed files, never
  raw ones, so the data source can be swapped without changing the algorithm.
- **Engine separate from CLI.** `curator/engine.py` has no printing or argv
  handling; `recommend.py` is a thin wrapper. A web UI must be addable without
  touching the algorithm.
- **Never commit data.** `data/` is git-ignored (CC BY-NC 3.0 license + size).
- Use `scores/glmer.csv` (0–1 scores) from both datasets. Never use
  `scores/tagdl.csv` (not 0–1). Ignore reviews, ratings, survey answers,
  `processed/`, `predictions/`, `metadata_updated.json`.
- Keep the README attribution: Kotkov et al., 2022, "The Tag Genome Dataset
  for Books"; Kotkov et al., 2021 and Vig et al., 2012 for the movie genome.
  Both CC BY-NC 3.0, non-commercial.

## Algorithm (keep in sync with README)

Default `MULTI_BOOK_MODE = "common"`. **Owner requirement: books and movies in;
the owner picks books, movies or mixed out; the algorithm first finds what the
inputs have in common, then recommends on those commonalities, favouring
distinctive (rare) traits over generic ones.**

1. Lookup per medium by case-insensitive normalized title (full title, then
   other names: books without trailing "(Series, #n)"; movies without year,
   with a moved article put back in front, or by alternative title), or
   `id:<n>`. 1 match → use; many → list with creator/year/id and stop; none →
   say so (with "contains" suggestions) and stop.
2. Tag space: all of one medium's tags when inputs and output are all that
   medium; otherwise the shared tags (exact names + `SHARED_TAG_SPELLINGS`),
   with movie scores quantile-mapped onto the book scale (`CALIBRATE_MOVIES`).
3. Excess = max(0, score − median of the item's own catalog).
4. Common: a tag counts only if ≥ `COMMON_SHARE` of inputs score ≥
   `TAG_THRESHOLD`; common_t = the excess that many inputs reach (min for all).
5. Strength = common × rarity × multiplier; rarity `log(N / max(n_t, 1))`
   over the wanted output pool. Core = top `COMMON_TAGS` with strength ≥
   `COMMON_MIN_STRENGTH` × max. Empty core → error ("nothing distinctive in
   common").
6. Match = Σ strength × candidate excess / Σ strength, over the core only.
7. Exclude inputs; cap results per author/director per medium
   (`MAX_PER_AUTHOR`, default 1).
8. Top `TOP_N` by match (ties: medium, then id); mixed lists reserve
   `MIXED_MIN_PER_MEDIUM` per medium. Each result carries its top core tags.

Book-only cosine modes kept for comparison: `"per_book"` (one profile per liked
book, profiles take turns picking so every book is represented) and `"blend"`
(one averaged profile); rarity-weighted cosine; dislikes (negative weights)
via `Recommender.recommend_rows`. Their results must not change.

## Commands

- `pip install -r requirements.txt` — install dependencies
- `python build.py` — download (if needed), validate, write `data/processed/`
- `python recommend.py "Dune" --movie "Heat" --want mixed` — recommendations
  (positional = books, `--movie` = movies, `--want books|movies|mixed`)
- `python -m pytest` — tests (synthetic data, no download needed)

Any change to scoring needs tests; keep the determinism, rarity, author-cap,
overlap/common-core, cross-medium, lookup-error and build-validation tests
passing.

## Git

The owner is the only contributor; work goes on the default branch (`main`).
