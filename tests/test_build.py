"""Build validation: bad data must stop the build with a clear message."""

import json

import numpy as np
import pytest

from conftest import BOOKS, TAGS, settings_for, write_raw
from curator.build import build
from curator.data import load_catalog
from curator.errors import DataValidationError

QUIET = dict(log=lambda *_: None)


def good_rows():
    return [(TAGS[j], b[0], b[4][j]) for b in BOOKS for j in range(len(TAGS))]


def run_build(tmp_path, **raw_overrides):
    write_raw(tmp_path / "raw", **raw_overrides)
    return build(settings_for(tmp_path), **QUIET)


def test_good_data_builds(tmp_path):
    manifest = run_build(tmp_path)
    assert manifest["counts"]["books"] == len(BOOKS)
    assert manifest["counts"]["tags"] == len(TAGS)
    assert manifest["counts"]["books_missing_author"] == 1
    assert manifest["counts"]["books_missing_year"] == 1
    assert manifest["counts"]["titles_shared_by_multiple_books"] == 1  # "Twins"
    cat = load_catalog(tmp_path / "processed")
    assert cat.scores.shape == (len(BOOKS), len(TAGS))
    # Row/column order matches books.csv/tags.csv
    assert cat.scores[0, TAGS.index("desert")] == pytest.approx(0.95)
    assert list(cat.tags["name"]) == TAGS
    manifest_file = json.loads((tmp_path / "processed" / "manifest.json").read_text())
    assert "build_date_utc" in manifest_file and manifest_file["source"]["scores_file"] == "scores/glmer.csv"


def test_tag_column_as_ids_also_works(tmp_path):
    rows = [(j, b[0], b[4][j]) for b in BOOKS for j in range(len(TAGS))]
    run_build(tmp_path, score_rows=rows)
    assert load_catalog(tmp_path / "processed").scores.shape == (len(BOOKS), len(TAGS))


def test_wrong_row_count(tmp_path):
    with pytest.raises(DataValidationError, match="expected 60 rows, found 59"):
        run_build(tmp_path, score_rows=good_rows()[:-1])


def test_score_above_one(tmp_path):
    rows = good_rows()
    rows[3] = (rows[3][0], rows[3][1], 1.2)
    with pytest.raises(DataValidationError, match="outside 0–1"):
        run_build(tmp_path, score_rows=rows)


def test_score_below_zero(tmp_path):
    rows = good_rows()
    rows[7] = (rows[7][0], rows[7][1], -0.01)
    with pytest.raises(DataValidationError, match="outside 0–1"):
        run_build(tmp_path, score_rows=rows)


def test_missing_score_value(tmp_path):
    rows = good_rows()
    rows[5] = (rows[5][0], rows[5][1], "")
    with pytest.raises(DataValidationError, match="missing values"):
        run_build(tmp_path, score_rows=rows)


def test_duplicate_book_tag_pair(tmp_path):
    rows = good_rows()
    rows[1] = rows[0]  # same count, but one pair twice and one pair missing
    with pytest.raises(DataValidationError, match="duplicate book-tag pairs"):
        run_build(tmp_path, score_rows=rows)


def test_scored_book_missing_from_metadata(tmp_path):
    meta = [{"item_id": b[0], "title": b[1], "authors": b[2], "year": b[3]} for b in BOOKS[1:]]
    with pytest.raises(DataValidationError, match="no entry in raw/metadata.json"):
        run_build(tmp_path, meta_records=meta)


def test_duplicate_metadata_id(tmp_path):
    meta = [{"item_id": b[0], "title": b[1], "authors": b[2], "year": b[3]} for b in BOOKS]
    meta.append(dict(meta[0]))
    with pytest.raises(DataValidationError, match="duplicate item_id"):
        run_build(tmp_path, meta_records=meta)


def test_missing_title(tmp_path):
    meta = [{"item_id": b[0], "title": b[1], "authors": b[2], "year": b[3]} for b in BOOKS]
    meta[2]["title"] = "  "
    with pytest.raises(DataValidationError, match="missing title"):
        run_build(tmp_path, meta_records=meta)


def test_unknown_tag_in_scores(tmp_path):
    rows = good_rows()
    rows[0] = ("not-a-real-tag", rows[0][1], rows[0][2])
    with pytest.raises(DataValidationError, match="not in tags.json"):
        run_build(tmp_path, score_rows=rows)


def test_wrong_tag_count(tmp_path):
    tags = [{"tag": t, "id": i} for i, t in enumerate(TAGS)] + [{"tag": "extra", "id": 99}]
    with pytest.raises(DataValidationError, match="expected 6 tags, found 7"):
        run_build(tmp_path, tag_records=tags)


def test_wrong_columns(tmp_path):
    write_raw(tmp_path / "raw")
    path = tmp_path / "raw" / "scores" / "glmer.csv"
    path.write_text(path.read_text().replace("tag,item_id,score", "tag,book,score", 1))
    with pytest.raises(DataValidationError, match="expected columns"):
        build(settings_for(tmp_path), **QUIET)


def test_wrong_expected_book_count(tmp_path):
    write_raw(tmp_path / "raw")
    s = settings_for(tmp_path)
    s = s.__class__(**{**s.__dict__, "expected_books": 11, "expected_score_rows": 60})
    with pytest.raises(DataValidationError, match="expected 11 books, found 10"):
        build(s, **QUIET)


def test_failed_build_leaves_no_valid_manifest(tmp_path):
    run_build(tmp_path)
    rows = good_rows()
    rows[0] = (rows[0][0], rows[0][1], 7.0)
    with pytest.raises(DataValidationError):
        run_build(tmp_path, score_rows=rows)
    # The old good build is still there and loads; nothing half-written.
    assert load_catalog(tmp_path / "processed").scores.max() <= 1.0


def test_load_without_build_is_clear(tmp_path):
    with pytest.raises(DataValidationError, match="Run the build first"):
        load_catalog(tmp_path / "nothing-here")


def test_load_detects_tampered_matrix(tmp_path):
    run_build(tmp_path)
    m = np.load(tmp_path / "processed" / "scores.npy")
    m[0, 0] = 3.0
    np.save(tmp_path / "processed" / "scores.npy", m)
    with pytest.raises(DataValidationError, match="outside 0–1"):
        load_catalog(tmp_path / "processed")
