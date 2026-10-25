"""
Shared test fixtures: a tiny synthetic dataset in the same format as the real
one, so the tests never need the 1.4 GB download.
"""

import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from curator.build import BuildSettings, build  # noqa: E402
from curator.data import load_catalog  # noqa: E402
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

THRESHOLD = 0.5


def write_raw(raw_dir: Path, books=BOOKS, tags=TAGS, score_rows=None, tag_records=None, meta_records=None):
    """Write synthetic raw files. Overrides let tests inject bad data."""
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


@pytest.fixture
def catalog(tmp_path):
    write_raw(tmp_path / "raw")
    s = settings_for(tmp_path)
    build(s, log=lambda *_: None)
    return load_catalog(s.processed_dir)


@pytest.fixture
def make_engine(catalog):
    def _make(**overrides):
        return Recommender(catalog, EngineSettings(**{"tag_threshold": THRESHOLD, **overrides}))
    return _make
