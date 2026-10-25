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
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# ---------------------------------------------------------------------------
# Data source (Tag Genome Dataset for Books, GroupLens)
# ---------------------------------------------------------------------------

# Where the build step downloads the dataset from (about 1.4 GB).
DATASET_URL = "https://files.grouplens.org/datasets/book-genome/book-genome.zip"

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
# When does a tag "really apply" to a book?
# ---------------------------------------------------------------------------

# A book "has" a tag when its score for that tag is at least this value.
# Used for: (1) tag rarity weights, (2) deciding which tags count as "shared"
# when explaining a match. To be chosen from the real score distribution --
# `python build.py` prints how many book-tag pairs clear 0.1 / 0.3 / 0.5 / 0.7
# and how many tags a typical book has at each (also saved in
# data/processed/manifest.json). See README section "The tag threshold".
# PROVISIONAL: 0.5 is a placeholder until the real data has been examined.
TAG_THRESHOLD = 0.5

# The thresholds the build step reports statistics for (for choosing the above).
THRESHOLD_CANDIDATES = (0.1, 0.3, 0.5, 0.7)

# ---------------------------------------------------------------------------
# The algorithm
# ---------------------------------------------------------------------------

# How many recommendations to return.
TOP_N = 3

# How many shared tags to show as the explanation for each recommendation.
EXPLAIN_TAGS = 5

# Maximum number of recommendations from any single author. 1 means you'll
# never get two books by the same author in one list (stops a whole series
# filling your results). Set to None for no limit.
MAX_PER_AUTHOR = 1

# If True, never recommend books by an author of one of your input books
# (e.g. liking "Dune" won't get you "Dune Messiah"). Off by default.
EXCLUDE_INPUT_AUTHORS = False

# Weight given to each input book when building your taste profile.
# Liked books count +1. The engine already supports negative weights (a
# disliked book would be -1), but dislikes are not exposed in the command yet.
LIKE_WEIGHT = 1.0
DISLIKE_WEIGHT = -1.0  # not used yet; reserved for a future "dislike" option

# Per-tag multipliers, applied on top of the rarity weight. Every tag defaults
# to 1.0 (no change). Use 0 to mute a tag completely, or >1 to boost it.
# Keys are tag names exactly as they appear in data/processed/tags.csv.
# A name that doesn't exist is an error (so typos can't silently do nothing).
# Example:  TAG_MULTIPLIERS = {"romance": 0.0, "surf": 2.0}
DEFAULT_TAG_MULTIPLIER = 1.0
TAG_MULTIPLIERS: dict[str, float] = {}

# Number of decimal places shown for similarity scores in the output.
DISPLAY_DECIMALS = 3
