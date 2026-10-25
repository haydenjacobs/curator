"""
Shared test fixtures: a tiny synthetic dataset in the same format as the real
one, so the tests never need the multi-GB downloads.
"""

import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from curator.build import BuildSettings, build, build_shared_tags  # noqa: E402
from curator.data import Library, load_catalog, load_library  # noqa: E402
from curator.engine import EngineSettings, Recommender  # noqa: E402

TAGS = ["fiction", "space", "desert", "cyberpunk", "surf", "romance"]

# id, title, authors, year, scores per tag (in TAGS order)
BOOKS = [
    (1, "Dune (Dune Chronicles, #1)", ["Frank Herbert"], 1965, [0.9, 0.9, 0.95, 0.1, 0.0, 0.1]),
    (2, "Dune Messiah (Dune Chronicles, #2)", ["Frank Herbert"], 1969, [0.9, 0.85, 0.9, 0.1, 0.0, 0.1]),
    (3, "Children of Dune (Dune Chronicles, #3)", ["Frank Herbert"], 1976, [0.9, 0.85, 0.85, 0.1, 0.0, 0.1]),
    (4, "Neuromancer", ["William Gibson"], 1984, [0.9, 0.6, 0.0, 0.95, 0.0, 0.05]),
    (5, "Arrakis Tales", ["Someone Else"], 2001, [0.9, 0.7, 0.8, 0.1, 0.0, 0.1]),
    (6, "Beach Days", ["Sandy Shore"], 2010, [0.9, 0.0, 0.1, 0.0, 0.95, 0.6]),
    (7, "Twins", ["Author One"], 1990, [0.8, 0.2, 0.1, 0.2, 0.1, 0.7]),
    (8, "Twins", ["Author Two"], 1995, [0.8, 0.1, 0.2, 0.1, 0.0, 0.8]),
    (9, "1984", ["George Orwell"], 1949, [0.9, 0.0, 0.0, 0.3, 0.0, 0.3]),
    (10, "No Author Book", [], 0, [0.9, 0.5, 0.5, 0.5, 0.0, 0.1]),  # 0 = unknown year
]

# Movie tags: "space", "desert", "cyberpunk", "romance" match book tags exactly;
# "surfing" is the movies' spelling of the books' "surf" (declared below);
# "heist" and "sci-fi" exist only for movies, "fiction" only for books.
MOVIE_TAGS = ["space", "desert", "cyberpunk", "heist", "romance", "surfing", "sci-fi"]
SHARED_SPELLINGS = {"surf": "surfing"}

# id, title (with year, MovieLens style), directors, scores per tag (MOVIE_TAGS order)
MOVIES = [
    (101, "Dune (1984)", "David Lynch", [0.9, 0.95, 0.1, 0.0, 0.2, 0.0, 0.9]),
    (102, "Dune (2021)", "Denis Villeneuve", [0.9, 0.9, 0.1, 0.0, 0.2, 0.0, 0.9]),
    (103, "Heat (1995)", "Michael Mann", [0.0, 0.0, 0.1, 0.95, 0.3, 0.0, 0.0]),
    (104, "Point Break (1991)", "Kathryn Bigelow", [0.0, 0.0, 0.0, 0.8, 0.6, 0.95, 0.0]),
    (105, "Blade Runner (1982)", "Ridley Scott", [0.5, 0.0, 0.95, 0.1, 0.6, 0.0, 0.95]),
    (106, "Alien (1979)", "Ridley Scott", [0.95, 0.1, 0.2, 0.0, 0.0, 0.0, 0.95]),
    (107, "Three Musketeers, The (1993)", "Stephen Herek", [0.0, 0.0, 0.0, 0.2, 0.5, 0.0, 0.0]),
    (108, "Seven (a.k.a. Se7en) (1995)", "David Fincher", [0.0, 0.0, 0.1, 0.3, 0.0, 0.0, 0.0]),
    (109, "Lawrence of Arabia (1962)", "David Lean", [0.0, 0.95, 0.0, 0.0, 0.3, 0.0, 0.0]),
    (110, "No Director Movie (2000)", "", [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
]

THRESHOLD = 0.5


def write_raw(raw_dir: Path, books=BOOKS, tags=TAGS, score_rows=None, tag_records=None, meta_records=None):
    """Write synthetic raw book files. Overrides let tests inject bad data."""
    (raw_dir / "scores").mkdir(parents=True, exist_ok=True)
    (raw_dir / "raw").mkdir(parents=True, exist_ok=True)
    if tag_records is None:
        tag_records = [{"tag": t, "id": i} for i, t in enumerate(tags)]
    with open(raw_dir / "raw" / "tags.json", "w") as f:
        for r in tag_records:
            f.write(json.dumps(r) + "\n")
    if meta_records is None:
        meta_records = [
            {"item_id": b[0], "url": "", "title": b[1], "authors": b[2], "lang": "eng",
             "img": "", "year": b[3], "description": ""}
            for b in books
        ]
    with open(raw_dir / "raw" / "metadata.json", "w") as f:
        for r in meta_records:
            f.write(json.dumps(r) + "\n")
    if score_rows is None:
        score_rows = [(tags[j], b[0], b[4][j]) for b in books for j in range(len(tags))]
    with open(raw_dir / "scores" / "glmer.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tag", "item_id", "score"])
        w.writerows(score_rows)


def write_movie_raw(raw_dir: Path, movies=MOVIES, tags=MOVIE_TAGS, score_rows=None, tag_records=None,
                    meta_records=None):
    """Write synthetic raw movie files (MovieLens layout: year in the title, directedBy)."""
    if tag_records is None:
        tag_records = [{"tag": t, "id": 500 + i} for i, t in enumerate(tags)]
    if meta_records is None:
        meta_records = [{"title": m[1], "directedBy": m[2], "starring": "", "item_id": m[0]} for m in movies]
    if score_rows is None:
        score_rows = [(tags[j], m[0], m[3][j]) for m in movies for j in range(len(tags))]
    write_raw(raw_dir, books=(), tags=tags, score_rows=score_rows, tag_records=tag_records, meta_records=meta_records)


def settings_for(tmp_path: Path, n_books=len(BOOKS), n_tags=len(TAGS)) -> BuildSettings:
    return BuildSettings(
        raw_dir=tmp_path / "raw",
        processed_dir=tmp_path / "processed",
        scores_file="scores/glmer.csv",
        metadata_file="raw/metadata.json",
        tags_file="raw/tags.json",
        expected_books=n_books,
        expected_tags=n_tags,
        expected_score_rows=n_books * n_tags,
        threshold_candidates=(0.1, 0.3, 0.5, 0.7),
        tag_threshold=THRESHOLD,
    )


def movie_settings_for(tmp_path: Path, n_movies=len(MOVIES), n_tags=len(MOVIE_TAGS), **overrides) -> BuildSettings:
    s = BuildSettings(
        raw_dir=tmp_path / "raw_movies",
        processed_dir=tmp_path / "processed" / "movies",
        scores_file="scores/glmer.csv",
        metadata_file="raw/metadata.json",
        tags_file="raw/tags.json",
        expected_books=n_movies,
        expected_tags=n_tags,
        expected_score_rows=n_movies * n_tags,
        threshold_candidates=(0.1, 0.3, 0.5, 0.7),
        tag_threshold=THRESHOLD,
        medium="movie",
        creator_field="directedBy",
        year_from_title=True,
    )
    return s.__class__(**{**s.__dict__, **overrides})


def build_library(tmp_path: Path) -> Path:
    """Build books + movies + shared tags into tmp_path/processed, like build.py does."""
    write_raw(tmp_path / "raw")
    book = settings_for(tmp_path)
    build(book.__class__(**{**book.__dict__, "processed_dir": tmp_path / "processed" / "books"}), log=lambda *_: None)
    write_movie_raw(tmp_path / "raw_movies")
    build(movie_settings_for(tmp_path), log=lambda *_: None)
    build_shared_tags(tmp_path / "processed", 4, SHARED_SPELLINGS, log=lambda *_: None)
    return tmp_path / "processed"


@pytest.fixture
def library(tmp_path):
    return load_library(build_library(tmp_path))


@pytest.fixture
def catalog(library):
    return library.catalogs["book"]


@pytest.fixture
def movies(library):
    return library.catalogs["movie"]


@pytest.fixture
def make_engine(library):
    """Engine in "common" mode (the default) unless told otherwise."""
    def _make(**overrides):
        return Recommender(library, EngineSettings(**{"tag_threshold": THRESHOLD, **overrides}))
    return _make


@pytest.fixture
def cosine_engine(library):
    """Engine in the book-only "per_book" cosine mode (overridable)."""
    def _make(**overrides):
        settings = {"tag_threshold": THRESHOLD, "multi_book_mode": "per_book", **overrides}
        return Recommender(library, EngineSettings(**settings))
    return _make


def books_only_library(tmp_path: Path, books=BOOKS) -> Library:
    """A library with just a book catalog (no movies, no shared tags)."""
    write_raw(tmp_path / "raw", books=books)
    s = settings_for(tmp_path, n_books=len(books))
    build(s, log=lambda *_: None)
    return Library(catalogs={"book": load_catalog(s.processed_dir)}, shared_tags=None)
