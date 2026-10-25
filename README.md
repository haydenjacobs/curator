# Curator: book and movie recommendations from the Tag Genome

Tell Curator some books and/or movies you liked. It first works out **what they
have in common**, shows you that, and then recommends books, movies, or a mixed
list that are strongest on exactly those shared traits. Every result comes with
the tags that explain why it matched.

It's **content-based**. Every book has a score from 0 to 1 for each of 727
descriptive tags ("space opera", "unreliable narrator", "surf", ...), and every
movie has one for each of 1,084 tags. 495 tags exist in both datasets, which is
how a book and a movie get compared. It's fully deterministic: the same input
always gives the same output, with no randomness and no AI judgment in the
scoring.

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

## 2. Build the data (one time, ~a few minutes plus downloads)

```bash
python build.py
```
What it does:
1. Downloads both datasets if they aren't there yet: books (~1.4 GB) into
   `data/raw/`, movies (~1.9 GB) into `data/raw/movies/`. It extracts the three
   files it uses from each. Raw files are never changed afterwards, and the zips
   can be deleted once extracted.
2. Checks them thoroughly. It **stops with a clear error** if anything is off:
   wrong row counts, items with scores but no title, scores outside 0–1,
   duplicate item-tag pairs, unknown tags, missing values, and more. (The movie
   data has a few known defects, which are accepted only exactly as listed in
   `config.py`. See "Things to know about the data".)
3. Writes the processed files:
   - `data/processed/books/` and `data/processed/movies/`, each with
     `items.csv` (id, title, author/director, year, search forms of the title),
     `tags.csv`, `scores.npy` (the items × tags score matrix) and
     `manifest.json` (counts, build date, source checksum, statistics)
   - `data/processed/shared_tags.csv`: the 495 tags books and movies share

If you already built an older version of Curator, just run `python build.py`
again. It reuses the downloaded book files and removes the old processed files.

The recommender reads **only** `data/processed/`. To switch to a different
data source later, you'd write a new build step that produces the same files.
The algorithm wouldn't change.

## 3. Get recommendations

```bash
python recommend.py "The Hunger Games" --movie "Heat" --want mixed
```
- Plain arguments are **books** you liked: `"Dune" "Neuromancer"`.
- Each `--movie` (or `-m`) is a **movie** you liked: `--movie "Heat" --movie "Alien"`.
- `--want` (or `-w`) picks what comes back: `books` (the default), `movies`, or
  `mixed` (books and movies ranked together, at least one of each).
- `-n 10` asks for 10 results instead of 3.

More examples:
```bash
python recommend.py "Dune"                                    # books like Dune
python recommend.py --movie "Alien" --movie "Fargo" --want books
python recommend.py "Pride and Prejudice" --movie "Blade Runner" --want movies
```

Titles are matched ignoring upper/lower case, accents and punctuation:
- Books: a trailing series note is optional (`"Dune"` finds
  `"Dune (Dune Chronicles, #1)"`).
- Movies: the year is optional (`"Heat"` finds `"Heat (1995)"`). MovieLens
  stores some titles as `"Three Musketeers, The"` or with a second title in
  brackets. `"The Three Musketeers"`, `"Il Postino"` and `"Se7en"` all work.

What happens with a title:
- **One match** → used.
- **Several matches** (e.g. the 1984 and 2000 "Dune" films) → it stops and
  lists them with author/director, year and id. Re-run using the id:
  ```bash
  python recommend.py --movie id:2021
  ```
- **No match** → it says so and lists titles that *contain* your text, if any,
  so you can pick one. It never guesses.

Example output:
```
Because you liked:
  • [book] The Hunger Games (The Hunger Games, #1) — Suzanne Collins (2008)  [id:2792775]
  • [movie] Heat (1995) — dir. Michael Mann  [id:6]

What these have in common (strongest first):
  action, thriller, suspense, destiny, obsession, love story, fighting, action packed, drama, romance
  (Books and movies are compared on the tags both datasets have.)

Books and movies for you:
1. [movie] Speed (1994) — dir. Jan de Bont  [id:377]
   match: 0.513
   why: action, thriller, suspense, love story, action packed
2. [movie] Matrix, The (1999) — dir. Andy Wachowski, Larry Wachowski  [id:2571]
   match: 0.512
   why: action, thriller, suspense, destiny, fighting
3. [book] The Hit (Will Robie, #2) — David Baldacci (2013)  [id:21512735]
   match: 0.460
   why: action, thriller, suspense, love story, fighting
```
"match" is how strongly the result has the shared traits (a weighted average,
see below). "why" lists the shared traits that added the most to it.

## 4. Run the tests

```bash
python -m pytest
```
Runs the automated tests on a tiny made-up dataset (no download needed). They
check that bad data is caught, results are deterministic, the overlap logic and
rarity weighting work, books and movies are compared only on shared tags,
mixed lists contain both, the one-per-author rule works, and lookup errors are
clear.

---

## How the algorithm works ("common" mode, the default)

1. **Lookup.** Find each input book or movie (rules above).
2. **Which tags.** If everything (your inputs and what you asked for) is
   books, all 727 book tags are used. If everything is movies, all 1,084 movie
   tags. As soon as books and movies meet, only the **495 shared tags**.
   In that case the movie scores are first **calibrated** to the book scale:
   the movie scoring model is about twice as generous (the top 1% of movies
   score ~0.71 on a typical shared tag, the top 1% of books ~0.41). Each movie
   score is therefore replaced by the book score at the same rank. A movie in
   the top 2% for "dystopia" counts like a book in the top 2% for "dystopia".
3. **What stands out about each input.** For every tag: how far the score is
   *above typical* (the median over that catalog). Every title has weak
   background scores on every tag, and these don't count.
4. **Overlap first.** A tag is **common** only if *every* input really has it
   (score ≥ 0.5, the tag threshold). How common it is = the smaller of the
   inputs' "above typical" amounts: what they *all* share.
5. **Uniqueness.** Each common tag is multiplied by its **rarity**,
   `log(N / n)`, where n is how many items in the pool you asked for have the
   tag. "Fiction" (on almost everything) counts ~0; a niche tag counts a lot.
   The 10 strongest form the **common core**, which is shown to you first.
   Tags less than 5% as strong as the strongest are dropped. If nothing
   is left, Curator says your inputs have nothing distinctive in common
   instead of guessing.
6. **Match.** Every candidate is scored **only on the core**: a weighted
   average of how far above typical it is on each core tag, weighted by the
   tag's strength. Traits that only one of your inputs has don't count at all,
   so you don't just get near-copies of one input.
7. **Exclusions.** Your inputs are never recommended, and at most one result
   per author (books) or director (movies) is returned (configurable).
8. **Output.** The top N, best match first. A mixed list holds at least one
   book and one movie. Exact ties are broken by medium, then id, so ordering
   never varies.

With a single input, the "common core" is simply that title's 10 most
distinctive traits.

### The older modes (books only)

`MULTI_BOOK_MODE` in `config.py` can switch back to the original cosine-similarity
modes, which only work with books in and books out:
- `"per_book"`: each liked book gets its own taste profile, and the profiles
  take turns picking, so every book you entered gets its own similar books.
- `"blend"`: all your books are averaged into one profile.

Both compare your weighted tag profile with each book's (cosine similarity).
They give exactly the same results as before movies were added.

## Settings (`config.py`)

Every tunable number is in `config.py`, each with a comment. The main ones:

| Setting | Default | What it does |
|---|---|---|
| `TAG_THRESHOLD` | 0.5 | Score at which a tag "really applies". An input must reach it for a tag to count as common; also drives rarity. |
| `TOP_N` | 3 | Number of recommendations. |
| `DEFAULT_WANT` | `"books"` | What comes back when you don't pass `--want`. |
| `MULTI_BOOK_MODE` | `"common"` | `"common"`: overlap-first (books and movies). `"per_book"` / `"blend"`: the older book-only modes. |
| `COMMON_TAGS` | 10 | Size of the common core. |
| `COMMON_SHARE` | 1.0 | Share of inputs that must have a tag. 1.0 = all; with 3+ inputs, 0.66 = "at least 2 of 3". |
| `COMMON_MIN_STRENGTH` | 0.05 | Core tags must be at least this fraction as strong as the strongest. |
| `MIXED_MIN_PER_MEDIUM` | 1 | In a mixed list, at least this many books and this many movies. |
| `CALIBRATE_MOVIES` | True | Put movie scores on the book scale when books and movies are compared. |
| `EXPLAIN_TAGS` | 5 | Number of "why" tags shown per recommendation. |
| `MAX_PER_AUTHOR` | 1 | Max results per author/director. `None` = no limit. |
| `EXCLUDE_INPUT_AUTHORS` | False | If True, never recommend books by your input books' authors (no Dune sequels), or movies by your input movies' directors. |
| `TAG_MULTIPLIERS` | `{}` | Per-tag boost/mute, e.g. `{"romance": 0.0}`. Misspelled tag names are an error. Shared tags go by their book spelling. |

If you change an algorithm setting, you don't need to rebuild. Everything is
recomputed each run from the processed files.

## The tag threshold

`python build.py` prints this table for both datasets (also stored in each
`manifest.json`). Books:

| Threshold | Book-tag pairs at or above | Tags per book (median) | Tags that apply to no book |
|---|---|---|---|
| 0.1 | 2,767,860 (41%) | 295 | 2 |
| 0.3 | 1,071,619 (16%) | 114 | 17 |
| **0.5** | **507,130 (7%)** | **54** | **51** |
| 0.7 | 254,848 (4%) | 27 | 88 |

Movies at 0.5: 42 tags per movie (median), every tag applies to some movie.

**Chosen: 0.5.**
- At 0.1 or 0.3, a "typical" book has 100–300 tags, so "applies" means
  almost nothing.
- At 0.7, 88 book tags (12%), including real topics like "deserts", "magic
  realism" and "boarding school", can never apply to any book.
- At 0.5, a typical book has 54 tags (34–85). Dune's tags at this level include
  space, ecology and politics, but also some noise like "submarines" (0.53).

## Downloading the data (the certificate pin)

GroupLens's download server has a misconfigured security certificate. Its chain
ends at a root certificate whose key is mathematically invalid, so no standard
tool can verify it. The build therefore does this, for both downloads (same
server):

1. It tries the normal certificate check first.
2. If that fails, it accepts the connection only if the server's certificate
   is exactly the one recorded in `config.py` (`DATASET_TLS_PIN_SHA256`).
3. Once a file has downloaded, its fingerprint must match `DATASET_ZIP_SHA256`
   (books) or `MOVIE_DATASET_ZIP_SHA256` (movies), so every download, on any
   computer, gets byte-for-byte the same file as the first one.

The site's current certificate expires on **13 Dec 2026**. After that, a fresh
download stops with a "doesn't match the pin" error. Then you, or Claude, check
the new certificate and update `DATASET_TLS_PIN_SHA256`. The zip fingerprint
check still guarantees the data itself hasn't changed. If `data/raw/` already
holds the files, nothing is downloaded and the pin doesn't matter.

## Things to know about the data

Books:
- **Years are edition years**, not first publication: Dune shows 2006, Pride
  and Prejudice 2000. 1,256 books (13%) have no year (stored as 0 in the data)
  and show "?".
- **Titles repeat**: some exact titles belong to two different books
  (e.g. two different "Dark Matter"s). Lookup lists them so you can pick by id.
- **Series appear as separate books** ("Catching Fire (The Hunger Games, #2)").
  The one-per-author rule stops a whole series filling your list, but one
  sequel can still appear. Set `EXCLUDE_INPUT_AUTHORS = True` to block that.
- One tag, "contempor​ary", contains an invisible character and is a near
  duplicate of "contemporary". It's kept as-is, since it's what the data says.

Movies (checked 25 Oct 2026, all declared in `config.py`; the build stops if
anything else is wrong, or if one of these is ever *not* wrong):
- **5 movies are left out.** 4 scored movies (ids 863, 3027, 94466, 106642)
  have no title or any other information. And **Toy Story (1995)** is missing
  its score for one tag ("airplane"): the published file is one row short. Its
  readme counts the header line as a row. Curator never fills in missing
  values, so Toy Story is not in the catalog. 9,729 movies remain.
- `tags.json` lists 1,094 tags, but only 1,084 have scores (the other 10, e.g.
  "magic realism", are left out).
- Years come from the title ("Heat (1995)"). 3 movies have no director.

Books ↔ movies:
- The book dataset's readme says 512 of its tags overlap with the movie tags.
  The actual files share **491** identical tag names. 4 more are the same tag
  spelled differently ("anti war"/"anti-war", "feel good"/"feel-good",
  "anti semitism"/"anti-semitism", "tear-jerker"/"tear jerker"). These are listed
  by hand in `config.py` (`SHARED_TAG_SPELLINGS`), never matched by guesswork.
  Total: **495** shared tags.
- **Mixed lists lean towards movies.** In the checks so far, movies took most
  mixed-list slots even after calibration. That is partly explained: for some
  tag groups (action, thriller, suspense, fighting) movie scores rise and fall
  together about twice as strongly as book scores, so one movie fits several
  core tags at once. For other groups (humor, or dark/mystery) there is no
  such difference, and that part isn't understood yet.
  `MIXED_MIN_PER_MEDIUM` guarantees each medium a place.

General:
- **Opinion tags** like "good", "recommended" and "5 stars" score high on nearly
  every item, so they're never above typical and never count as common.
- **What very different inputs share is often a mood or genre** ("dark,
  thriller, mystery" for Gone Girl + Spirited Away). The core shows exactly
  what was found, so you can judge it.

## Project layout

```
config.py            all settings
build.py             command: download + validate + process
recommend.py         command: recommendations (command-line wrapper only)
curator/build.py     build logic and validation
curator/data.py      loads data/processed/ (the only thing the engine reads)
curator/lookup.py    title/id lookup (books and movies)
curator/engine.py    the algorithm (no printing, reusable from a web app)
curator/text.py      title normalization
tests/               automated tests (synthetic data)
data/                downloaded + processed data (NOT in git)
```

## Data, license and attribution

This project uses two datasets from GroupLens, both licensed under
**CC BY-NC 3.0** (attribution, non-commercial). Curator is a free,
non-commercial project. The datasets are never committed to this repository;
`python build.py` downloads them.

**Tag Genome Dataset for Books** (https://grouplens.org/datasets/book-genome/):

> Denis Kotkov, Alan Medlar, Alexandr Maslov, Umesh Raj Satyal, Mats Neovius,
> and Dorota Glowacka. 2022. *The Tag Genome Dataset for Books.* In ACM SIGIR
> Conference on Human Information Interaction and Retrieval (CHIIR '22).
> https://doi.org/10.1145/3498366.3505833

**MovieLens Tag Genome Dataset 2021**
(https://grouplens.org/datasets/movielens/tag-genome-2021/):

> Denis Kotkov, Alexandr Maslov, and Mats Neovius. 2021. *Revisiting the Tag
> Relevance Prediction Problem.* In Proceedings of the 44th International ACM
> SIGIR Conference on Research and Development in Information Retrieval.
> https://doi.org/10.1145/3404835.3463019

> Jesse Vig, Shilad Sen, and John Riedl. 2012. *The Tag Genome: Encoding
> Community Knowledge to Support Novel Interaction.* ACM Transactions on
> Interactive Intelligent Systems 2(3):13:1–13:44.
> https://doi.org/10.1145/2362394.2362395
