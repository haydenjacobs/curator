"""
Every tunable number for Curator lives in this one file.

If you want to change how recommendations behave, change it here and nowhere
else. Each setting has a comment explaining what it does.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Where things live
# ---------------------------------------------------------------------------

# Project root (the folder this file is in). Everything else is relative to it.
PROJECT_ROOT = Path(__file__).resolve().parent

# Untouched downloaded data. Nothing in this folder is ever modified.
RAW_DIR = PROJECT_ROOT / "data" / "raw"

# Files created by the build step. The recommendation engine reads ONLY these.
# Inside: books/ and movies/ (one catalog each) and shared_tags.csv.
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# ---------------------------------------------------------------------------
# Data source (Tag Genome Dataset for Books, GroupLens)
# ---------------------------------------------------------------------------

# Where the build step downloads the dataset from (about 1.4 GB).
DATASET_URL = "https://files.grouplens.org/datasets/book-genome/book-genome.zip"

# Security lock #1 (first download). The site's certificate can't be verified
# the standard way (its chain ends at a root with a malformed key), so we accept
# only this exact certificate, identified by its SHA-256 fingerprint. It is only
# used when the standard check fails. The site's current certificate expires
# 13 Dec 2026; after that the build will stop and ask you to re-check the pin
# (see README, "Downloading the data").
DATASET_TLS_PIN_SHA256 = "66b1471bd5b84c08efb30f06b4bec8503f76d6a361ae276459d994b979a7762f"

# Security lock #2 (every download after the first). SHA-256 fingerprint of the
# zip we first downloaded. Any later download, on any computer, must match it
# byte for byte or the build stops. Recorded from the first download, 25 Oct 2026.
DATASET_ZIP_SHA256 = "cb8b23176783cc3cd1cb6754e6c92be36b52c5de12b90a2404a59e580c92afe7"

# Name of the downloaded zip inside RAW_DIR.
DATASET_ZIP_NAME = "book-genome.zip"

# The three files we use from inside the zip. The build finds each one by the
# end of its path inside the zip (so the zip's top folder name doesn't matter)
# and copies it, unchanged, to RAW_DIR / <this path>.
# We deliberately do NOT use scores/tagdl.csv: its scores are not on a 0-1 scale.
SCORES_FILE = "scores/glmer.csv"
METADATA_FILE = "raw/metadata.json"
TAGS_FILE = "raw/tags.json"

# What the dataset readme says we should find. If the files don't match these
# exactly, the build stops with an error instead of carrying on.
EXPECTED_BOOKS = 9_374
EXPECTED_TAGS = 727
EXPECTED_SCORE_ROWS = EXPECTED_BOOKS * EXPECTED_TAGS  # 6,814,898

# ---------------------------------------------------------------------------
# Data source (MovieLens Tag Genome Dataset 2021, GroupLens)
# ---------------------------------------------------------------------------

# Where the build step downloads the movie dataset from (about 1.9 GB). Same
# server as the books, so the same certificate pin (DATASET_TLS_PIN_SHA256) applies.
MOVIE_DATASET_URL = "https://files.grouplens.org/datasets/tag-genome-2021/genome_2021.zip"

# SHA-256 fingerprint of the movie zip, recorded from the first download,
# 25 Oct 2026. Every later download must match it byte for byte.
MOVIE_DATASET_ZIP_SHA256 = "20a48e952c0161126edb9c6988b4cc2b70869f872dbae4eac4dce455afc768d6"

# The movie files live in their own folder so they can't collide with the books'.
MOVIE_RAW_DIR = RAW_DIR / "movies"
MOVIE_DATASET_ZIP_NAME = "genome_2021.zip"

# The files we use from inside the zip (same layout as the book dataset).
# raw/metadata.json, not raw/metadata_updated.json: the readme documents the
# former, and the two have identical titles and directors for every movie.
MOVIE_SCORES_FILE = "scores/glmer.csv"
MOVIE_METADATA_FILE = "raw/metadata.json"
MOVIE_TAGS_FILE = "raw/tags.json"

# What the build must find. tags.json lists 1,094 tags but only 1,084 have
# scores (the other 10, e.g. "magic realism", are left out).
EXPECTED_MOVIES = 9_734
EXPECTED_MOVIE_TAG_RECORDS = 1_094
EXPECTED_MOVIE_TAGS = 1_084
# One row fewer than movies x tags: the published glmer.csv has no score for
# ("airplane", Toy Story). Its readme says 10,551,656 rows, but that count
# includes the header line. Checked 25 Oct 2026; tagdl.csv has the same gap.
EXPECTED_MOVIE_SCORE_ROWS = EXPECTED_MOVIES * EXPECTED_MOVIE_TAGS - 1  # 10,551,655

# Known defects in the published movie data. The build checks that EXACTLY
# these are wrong (anything else still stops it) and leaves these movies out,
# rather than guessing missing values. Found 25 Oct 2026.
#   Scored movies with no title or other metadata at all:
MOVIES_WITHOUT_METADATA = (863, 3027, 94466, 106642)
#   Movies missing some scores (movie id -> tags with no score):
MOVIES_WITH_MISSING_SCORES = {1: ("airplane",)}  # Toy Story (1995)

# Books and movies are compared on the tags both datasets have. Tags match
# when their names are identical; the build insists on finding exactly this
# many (the book readme says "512", but the actual files share 491 names).
EXPECTED_SHARED_TAGS_EXACT = 491
# Plus these book tags, which are the same tag spelled differently in the
# movie data (book name -> movie name). Total shared tags: 491 + 4 = 495.
SHARED_TAG_SPELLINGS = {
    "anti semitism": "anti-semitism",
    "anti war": "anti-war",
    "feel good": "feel-good",
    "tear-jerker": "tear jerker",
}

# ---------------------------------------------------------------------------
# When does a tag "really apply" to a book?
# ---------------------------------------------------------------------------

# A book "has" a tag when its score for that tag is at least this value.
# Used for: (1) tag rarity weights, (2) deciding which tags count as "shared"
# when explaining a match. Chosen from the real score distribution
# (`python build.py` prints it): at 0.5 a typical book has 54 tags (range
# 34-85), which reads as a real description; 0.3 gives ~114 tags/book including
# obvious noise, and 0.7 leaves 88 of 727 tags (e.g. "deserts", "magic realism")
# unable to apply to any book. See README section "The tag threshold".
TAG_THRESHOLD = 0.5

# The thresholds the build step reports statistics for (for choosing the above).
THRESHOLD_CANDIDATES = (0.1, 0.3, 0.5, 0.7)

# ---------------------------------------------------------------------------
# The algorithm
# ---------------------------------------------------------------------------

# How many recommendations to return.
TOP_N = 3

# What to recommend when the command doesn't say (--want): "books",
# "movies" or "mixed" (books and movies ranked together in one list).
DEFAULT_WANT = "books"

# How many tags to show as the explanation for each recommendation.
EXPLAIN_TAGS = 5

# Maximum number of recommendations from any single author (books) or
# director (movies). 1 means you'll never get two books by the same author in
# one list (stops a whole series filling your results). Set to None for no limit.
MAX_PER_AUTHOR = 1

# If True, never recommend books by an author of one of your input books
# (e.g. liking "Dune" won't get you "Dune Messiah"), or movies by a director
# of one of your input movies. Off by default.
EXCLUDE_INPUT_AUTHORS = False

# How your liked books/movies are combined.
#   "common" (default): first find what your inputs have IN COMMON (tags every
#       one of them has more than a typical item does, rare tags counting most),
#       then recommend what is strongest on exactly those shared traits. Works
#       for books and movies together. See README "How the algorithm works".
#   "per_book" (books only): each liked book gets its own taste profile, and
#       the profiles take turns picking results, so EVERY book you enter gets
#       its own similar books.
#   "blend" (books only): average all your books into one profile (the
#       original design). One input can crowd out the others.
MULTI_BOOK_MODE = "common"

# "common" mode: how many of the strongest shared tags make up the "common
# core" that candidates are scored on (and that is shown to you). Fewer =
# sharper focus on the very top commonalities; more = broader matching.
COMMON_TAGS = 10

# "common" mode: what share of your inputs must have a tag for it to count as
# common. 1.0 = all of them. With 3+ inputs you could relax this, e.g. 0.66 =
# "at least 2 of 3". Must be > 0 and <= 1.
COMMON_SHARE = 1.0

# "common" mode: a shared tag only joins the common core if it is at least
# this fraction as strong as the strongest shared tag (0.05 = 5%). Stops
# near-meaningless commonalities from padding the list. 0 = no cutoff.
COMMON_MIN_STRENGTH = 0.05

# "common" mode, mixed lists: each medium (books, movies) gets at least this
# many of the slots, so a mixed list is never all books or all movies.
MIXED_MIN_PER_MEDIUM = 1

# When books and movies are compared, calibrate movie scores to the book
# scale: each movie score is replaced by the book score at the same rank for
# that tag. Needed because the two datasets' scoring models differ: on the
# shared tags the top 1% of movies score ~0.71 vs ~0.41 for books (measured
# 25 Oct 2026), so uncalibrated mixed lists lean towards movies.
CALIBRATE_MOVIES = True

# Weight given to each input when building your taste profile.
# Liked items count +1. The engine supports negative weights (a disliked book
# would be -1) in the "per_book" and "blend" modes, but dislikes are not
# exposed in the command yet.
LIKE_WEIGHT = 1.0
DISLIKE_WEIGHT = -1.0  # not used yet; reserved for a future "dislike" option

# Per-tag multipliers, applied on top of the rarity weight. Every tag defaults
# to 1.0 (no change). Use 0 to mute a tag completely, or >1 to boost it.
# Keys are tag names exactly as they appear in data/processed/books/tags.csv
# or data/processed/movies/tags.csv. When books and movies are compared, the
# shared tags go by their book spelling (data/processed/shared_tags.csv).
# A name that doesn't exist is an error (so typos can't silently do nothing).
# Example:  TAG_MULTIPLIERS = {"romance": 0.0, "surf": 2.0}
DEFAULT_TAG_MULTIPLIER = 1.0
TAG_MULTIPLIERS: dict[str, float] = {}

# Number of decimal places shown for match/similarity scores in the output.
DISPLAY_DECIMALS = 3
