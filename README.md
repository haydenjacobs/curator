# Curator — book recommendations from the Tag Genome

Tell Curator one or more books you liked. It returns the 3 most similar books,
each with the tags that explain why it matched.

It's **content-based**: every book has a score from 0 to 1 for each of 727
descriptive tags ("space opera", "unreliable narrator", "surf", ...). Curator
builds your taste profile from those scores and finds the books whose tag
profile is closest. It's fully deterministic: the same input always gives the
same output, with no randomness and no AI judgment in the scoring.

There's no user interface yet. You run it from the command line.

---

## 1. Setup (one time)

You need Python 3.10 or newer. In a terminal, inside this folder:

```bash
python3 -m venv .venv
```
Creates a private Python environment in a `.venv` folder so this project's
libraries don't clash with anything else on your computer.

```bash
source .venv/bin/activate
```
Switches this terminal to that environment. (Windows: `.venv\Scripts\activate`.)
Run this again every time you open a new terminal.

```bash
pip install -r requirements.txt
```
Installs the three libraries Curator uses: numpy (number crunching), pandas
(reading tables), and pytest (running the tests).

## 2. Build the data (one time, ~a few minutes)

```bash
python build.py
```
What it does:
1. Downloads the dataset zip (~1.4 GB) into `data/raw/` if it isn't there yet,
   and extracts the three files we use. Raw files are never changed afterwards.
2. Checks them thoroughly. It **stops with a clear error** if anything is off:
   wrong row counts, books with scores but no title, scores outside 0–1,
   duplicate book-tag pairs, unknown tags, missing values, and more.
3. Writes the processed files into `data/processed/`:
   - `books.csv` — id, title, author, year, plus normalized title for searching
   - `tags.csv` — tag id and name
   - `scores.npy` — the books × tags score matrix
   - `manifest.json` — counts, build date, source file and its checksum, and
     statistics about how the scores are distributed

The recommender reads **only** `data/processed/`. To switch to a different
data source later, you'd write a new build step that produces the same files —
the algorithm wouldn't change.

## 3. Get recommendations

```bash
python recommend.py "Dune" "Neuromancer"
```
Each argument is a book you liked. Titles are matched ignoring upper/lower case,
accents and punctuation, and a trailing series note in brackets is optional
(`"Dune"` finds `"Dune (Dune Chronicles, #1)"`).

- **One match** → used.
- **Several matches** (e.g. two different books called "Twins") → it stops and
  lists them with author, year and id. Re-run using the id:
  ```bash
  python recommend.py id:12345
  ```
- **No match** → it says the book isn't in the catalog, and lists titles that
  *contain* your text, if any, so you can pick one. It never guesses.

Ask for more than 3 results with `-n`, e.g. `python recommend.py -n 10 "Dune"`.

Example output shape:
```
Because you liked:
  • Dune (Dune Chronicles, #1) — Frank Herbert (1965)  [id:...]

1. <title> — <author> (<year>)  [id:...]
   similarity: 0.812
   why: desert (21%), politics (14%), ecology (9%), religion (7%), space opera (6%)
```
The % after each tag is how much of the similarity score that tag accounts for.

## 4. Run the tests

```bash
python -m pytest
```
Runs the automated tests on a tiny made-up dataset (no download needed). They
check that bad data is caught, results are deterministic, rare tags outweigh
common ones, the one-per-author rule works, and lookup errors are clear.

---

## How the algorithm works

1. **Lookup** — find each input book (rules above).
2. **Tag rarity** — `weight = log(N / n)`, where N is the number of books and n
   is how many books the tag "really applies" to (score ≥ the threshold). A tag
   on almost every book ("fiction") gets a weight near 0; a niche tag ("surf")
   gets a high weight. A tag that applies to *no* book gets weight 0 (it's never
   strong enough anywhere to count as evidence).
3. **Taste profile** — for each tag, the average across your books of
   (input weight × that book's score). Liked books have input weight +1. The
   engine already accepts −1 for disliked books; that just isn't exposed yet.
4. **Per-tag multipliers** — every tag is also multiplied by a number from the
   config (1.0 by default) so you can mute or boost tags.
5. **Similarity** — cosine similarity between your weighted profile and each
   book's weighted tags (1 = identical direction, 0 = nothing in common).
6. **Exclusions** — your input books are never recommended, and at most one
   book per author is returned (configurable).
7. **Output** — the top 3, each with the 5 *shared* tags (the recommended book
   and at least one of your books both score ≥ threshold) that contributed most.
   Exact ties are broken by book id, so ordering never varies.

## Settings (`config.py`)

Every tunable number is in `config.py`, each with a comment. The main ones:

| Setting | Default | What it does |
|---|---|---|
| `TAG_THRESHOLD` | 0.5 | Score at which a tag "really applies" to a book. Drives rarity weights and which tags are shown as shared. |
| `TOP_N` | 3 | Number of recommendations. |
| `EXPLAIN_TAGS` | 5 | Number of shared tags shown per recommendation. |
| `MAX_PER_AUTHOR` | 1 | Max results per author. `None` = no limit. |
| `EXCLUDE_INPUT_AUTHORS` | False | If True, never recommend books by authors of your input books (e.g. no Dune sequels). |
| `LIKE_WEIGHT` | 1.0 | Input weight for liked books. (`DISLIKE_WEIGHT` = −1 is reserved for later.) |
| `TAG_MULTIPLIERS` | `{}` | Per-tag boost/mute, e.g. `{"romance": 0.0}`. Unlisted tags use `DEFAULT_TAG_MULTIPLIER` (1.0). Misspelled tag names are an error. |
| `EXPECTED_BOOKS` / `EXPECTED_TAGS` | 9,374 / 727 | What the build insists on finding. |

If you change `TAG_THRESHOLD`, you don't need to rebuild — weights are
recomputed each run from the processed matrix.

## The tag threshold

`python build.py` prints a table of how many book-tag pairs score ≥ 0.1, 0.3,
0.5 and 0.7, and how many tags a typical book has at each level (also stored in
`data/processed/manifest.json`). The threshold should be the level at which a
typical book has a meaningful handful of tags — enough to describe it, few
enough that each one means something. The final choice and reasoning will be
recorded here once the real data has been examined.

## Project layout

```
config.py            all settings
build.py             command: download + validate + process
recommend.py         command: recommendations (command-line wrapper only)
curator/build.py     build logic and validation
curator/data.py      loads data/processed/ (the only thing the engine reads)
curator/lookup.py    title/id lookup
curator/engine.py    the algorithm (no printing, reusable from a web app)
curator/text.py      title normalization
tests/               automated tests (synthetic data)
data/                downloaded + processed data (NOT in git)
```

## Data, license and attribution

This project uses the **Tag Genome Dataset for Books** from GroupLens:

> Denis Kotkov, Alan Medlar, Alexandr Maslov, Umesh Raj Satyal, Mats Neovius,
> and Dorota Glowacka. 2022. *The Tag Genome Dataset for Books.* In ACM SIGIR
> Conference on Human Information Interaction and Retrieval (CHIIR '22).
> https://doi.org/10.1145/3498366.3505833

Dataset: https://grouplens.org/datasets/book-genome/ — licensed under
**CC BY-NC 3.0** (attribution, non-commercial). Curator is a free,
non-commercial project. The dataset is never committed to this repository;
`python build.py` downloads it.
