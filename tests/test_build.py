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
    assert manifest["medium"] == "book"
    assert manifest["counts"]["items"] == len(BOOKS)
    assert manifest["counts"]["tags"] == len(TAGS)
    assert manifest["counts"]["items_missing_creator"] == 1
    assert manifest["counts"]["items_missing_year"] == 1
    assert manifest["counts"]["titles_shared_by_multiple_items"] == 1  # "Twins"
    cat = load_catalog(tmp_path / "processed")
    assert cat.scores.shape == (len(BOOKS), len(TAGS))
    # Row/column order matches items.csv/tags.csv
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
    with pytest.raises(DataValidationError, match="duplicate item-tag pairs"):
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


def test_zip_with_wrong_fingerprint_is_rejected(tmp_path):
    import zipfile

    from curator.build import ensure_raw_files

    raw = tmp_path / "raw"
    raw.mkdir()
    with zipfile.ZipFile(raw / "book-genome.zip", "w") as zf:
        zf.writestr("book-genome/scores/glmer.csv", "tag,item_id,score\n")
    with pytest.raises(DataValidationError, match="doesn't match DATASET_ZIP_SHA256"):
        ensure_raw_files(raw, "https://example.invalid/x.zip", "book-genome.zip",
                         ["scores/glmer.csv"], zip_sha256="0" * 64, **QUIET)
    assert not (raw / "scores" / "glmer.csv").exists()  # nothing extracted


def test_zip_extracts_only_needed_files(tmp_path):
    import zipfile

    from curator.build import ensure_raw_files
    from curator.download import sha256_file

    raw = tmp_path / "raw"
    raw.mkdir()
    with zipfile.ZipFile(raw / "book-genome.zip", "w") as zf:
        zf.writestr("book-genome/scores/glmer.csv", "tag,item_id,score\n")
        zf.writestr("book-genome/scores/tagdl.csv", "ignored\n")
    ensure_raw_files(raw, "https://example.invalid/x.zip", "book-genome.zip", ["scores/glmer.csv"],
                     zip_sha256=sha256_file(raw / "book-genome.zip"), **QUIET)
    assert (raw / "scores" / "glmer.csv").is_file()
    assert not (raw / "scores" / "tagdl.csv").exists()


# --- movies -------------------------------------------------------------------

from conftest import MOVIE_TAGS, MOVIES, SHARED_SPELLINGS, build_library, movie_settings_for, write_movie_raw  # noqa: E402
from curator.build import build_shared_tags  # noqa: E402
from curator.data import load_library, load_shared_tags  # noqa: E402
from curator.text import movie_aliases  # noqa: E402


def movie_rows(skip=()):
    return [(MOVIE_TAGS[j], m[0], m[3][j]) for m in MOVIES for j in range(len(MOVIE_TAGS))
            if (MOVIE_TAGS[j], m[0]) not in skip]


def run_movie_build(tmp_path, settings=None, **raw_overrides):
    write_movie_raw(tmp_path / "raw_movies", **raw_overrides)
    return build(settings or movie_settings_for(tmp_path), **QUIET)


def test_good_movie_data_builds(tmp_path):
    manifest = run_movie_build(tmp_path)
    assert manifest["medium"] == "movie"
    assert manifest["counts"]["items"] == len(MOVIES)
    assert manifest["counts"]["items_missing_creator"] == 1
    assert manifest["counts"]["items_missing_year"] == 0
    cat = load_catalog(tmp_path / "processed" / "movies", "movie")
    heat = cat.item_info(2)
    assert heat == {"id": 103, "title": "Heat (1995)", "creator": "Michael Mann", "year": "1995", "medium": "movie"}


def test_movie_title_aliases():
    assert movie_aliases("Heat (1995)") == ["heat"]
    assert movie_aliases("Three Musketeers, The (1993)") == ["the three musketeers"]
    assert movie_aliases("Postman, The (Postino, Il) (1994)") == ["the postman", "il postino"]
    assert movie_aliases("Seven (a.k.a. Se7en) (1995)") == ["seven", "se7en"]
    assert movie_aliases("Paris, Texas (1984)") == ["paris texas"]  # "Texas" is not an article


def test_unscored_tags_in_tags_file_are_left_out(tmp_path):
    records = [{"tag": t, "id": 500 + i} for i, t in enumerate(MOVIE_TAGS)] + [{"tag": "too short", "id": 999}]
    s = movie_settings_for(tmp_path, expected_tag_records=len(MOVIE_TAGS) + 1)
    manifest = run_movie_build(tmp_path, s, tag_records=records)
    assert manifest["counts"]["tags_without_scores"] == 1
    assert "too short" not in set(load_catalog(tmp_path / "processed" / "movies", "movie").tags["name"])


def test_unscored_tag_needs_the_tag_record_count(tmp_path):
    records = [{"tag": t, "id": 500 + i} for i, t in enumerate(MOVIE_TAGS)] + [{"tag": "too short", "id": 999}]
    with pytest.raises(DataValidationError, match="expected 7 tags, found 8"):
        run_movie_build(tmp_path, tag_records=records)


def _meta_without(ids):
    return [{"title": m[1], "directedBy": m[2], "item_id": m[0]} for m in MOVIES if m[0] not in ids]


def test_declared_movie_without_metadata_is_left_out(tmp_path):
    s = movie_settings_for(tmp_path, known_without_metadata=(104,))
    manifest = run_movie_build(tmp_path, s, meta_records=_meta_without({104}))
    assert manifest["counts"]["items"] == len(MOVIES) - 1
    assert manifest["counts"]["left_out_no_metadata"] == [104]


def test_undeclared_movie_without_metadata_stops_the_build(tmp_path):
    with pytest.raises(DataValidationError, match="have no entry in raw/metadata.json"):
        run_movie_build(tmp_path, meta_records=_meta_without({104}))


def test_declared_defect_that_isnt_there_stops_the_build(tmp_path):
    with pytest.raises(DataValidationError, match="declares exactly these as known"):
        run_movie_build(tmp_path, movie_settings_for(tmp_path, known_without_metadata=(104,)))


def test_declared_missing_score_leaves_movie_out(tmp_path):
    s = movie_settings_for(tmp_path, expected_score_rows=len(MOVIES) * len(MOVIE_TAGS) - 1,
                           known_incomplete={103: ("heist",)})
    manifest = run_movie_build(tmp_path, s, score_rows=movie_rows(skip={("heist", 103)}))
    assert manifest["counts"]["left_out_missing_scores"] == {"103": ["heist"]}
    assert 103 not in set(load_catalog(tmp_path / "processed" / "movies", "movie").items["id"])


def test_undeclared_missing_score_stops_the_build(tmp_path):
    s = movie_settings_for(tmp_path, expected_score_rows=len(MOVIES) * len(MOVIE_TAGS) - 1)
    with pytest.raises(DataValidationError, match="missing movie-tag pairs"):
        run_movie_build(tmp_path, s, score_rows=movie_rows(skip={("heist", 103)}))


# --- shared tags ----------------------------------------------------------------

def test_shared_tags_exact_names_plus_declared_spellings(tmp_path):
    lib = load_library(build_library(tmp_path))
    shared = lib.shared_tags
    assert list(shared["name"]) == ["space", "desert", "cyberpunk", "surf", "romance"]  # book tag order
    assert dict(zip(shared["name"], shared["movie_name"]))["surf"] == "surfing"


def test_shared_tag_count_mismatch_lists_near_misses(tmp_path):
    build_library(tmp_path)
    with pytest.raises(DataValidationError, match="share 4 identically named tags, config.py expects 5"):
        build_shared_tags(tmp_path / "processed", 5, SHARED_SPELLINGS, **QUIET)


def test_bad_declared_spelling_is_an_error(tmp_path):
    build_library(tmp_path)
    with pytest.raises(DataValidationError, match="not a book tag and a movie tag"):
        build_shared_tags(tmp_path / "processed", 4, {"surf": "surf-board"}, **QUIET)
    with pytest.raises(DataValidationError, match="already matches exactly"):
        build_shared_tags(tmp_path / "processed", 4, {"surf": "space"}, **QUIET)


def test_tampered_shared_tags_are_detected(tmp_path):
    processed = build_library(tmp_path)
    path = processed / "shared_tags.csv"
    path.write_text(path.read_text().replace('"surfing"', '"heist"'))
    books = load_catalog(processed / "books", "book")
    movies = load_catalog(processed / "movies", "movie")
    with pytest.raises(DataValidationError, match="don't match the movie tags.csv"):
        load_shared_tags(path, books, movies)


def test_catalog_of_the_wrong_medium_is_rejected(tmp_path):
    processed = build_library(tmp_path)
    with pytest.raises(DataValidationError, match="holds 'movie' data, expected 'book'"):
        load_catalog(processed / "movies", "book")
