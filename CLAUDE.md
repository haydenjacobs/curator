# CLAUDE.md — rules for working on Curator

Curator is a deterministic, content-based book recommender built on the Tag
Genome Dataset for Books (GroupLens). The owner is non-technical: explain in
plain language, give exact commands and say what each does. No UI yet; the
focus is data architecture and algorithm correctness.

## Non-negotiable rules

- **Deterministic.** Same input → identical output. No randomness, no AI/LLM
  judgment inside scoring. Ties are broken by book id.
- **Loud errors over silent failures.** If data doesn't match expectations
  (row counts, missing books, scores outside 0–1, duplicate book-tag pairs,
  unknown tags, missing values), stop with a clear message. Never guess or
  fill gaps. Lookup never auto-picks among several matches.
- **Correctness over speed.**
- **One config file.** Every tunable number lives in `config.py` with a comment.
  No magic numbers elsewhere.
- **Raw vs processed.** Raw data stays untouched in `data/raw/`. `build.py`
  creates `data/processed/`. The scoring code (`curator/data.py`,
  `curator/lookup.py`, `curator/engine.py`) reads ONLY processed files, never
  raw ones, so the data source can be swapped without changing the algorithm.
- **Engine separate from CLI.** `curator/engine.py` has no printing or argv
  handling; `recommend.py` is a thin wrapper. A web UI must be addable without
  touching the algorithm.
- **Never commit data.** `data/` is git-ignored (CC BY-NC 3.0 license + size).
- Use `scores/glmer.csv` (0–1 scores). Never use `scores/tagdl.csv` (not 0–1).
  Ignore reviews, ratings, survey answers, `processed/`, `predictions/`.
- Keep the README attribution: Kotkov et al., 2022, "The Tag Genome Dataset
  for Books", CC BY-NC 3.0, non-commercial.

## Algorithm (keep in sync with README)

1. Lookup by case-insensitive normalized title (full title, then title without
   trailing "(Series, #n)"), or `id:<n>`. 1 match → use; many → list with
   author/year/id and stop; none → say so (with "contains" suggestions) and stop.
2. Rarity `w_t = log(N / n_t)`, `n_t` = books with score ≥ `TAG_THRESHOLD`;
   `n_t = 0` → weight 0.
3. Profile = mean over inputs of (input weight × score). Likes = +1; the engine
   supports negative weights (dislikes) via `Recommender.recommend_rows`, not
   exposed in the CLI yet.
4. Per-tag multipliers from config (default 1.0).
5. Cosine similarity of weighted profile vs weighted book vectors.
6. Exclude input books; cap results per author (`MAX_PER_AUTHOR`, default 1).
7. Top N (default 3) with top 5 shared tags by contribution.

## Commands

- `pip install -r requirements.txt` — install dependencies
- `python build.py` — download (if needed), validate, write `data/processed/`
- `python recommend.py "Dune" "Neuromancer"` — recommendations
- `python -m pytest` — tests (synthetic data, no download needed)

Any change to scoring needs tests; keep the determinism, rarity, author-cap,
lookup-error and build-validation tests passing.

## Git

The owner is the only contributor; work goes on the default branch (`main`).
