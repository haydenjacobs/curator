"""
Build step. Run:   python build.py

Downloads the book and movie datasets into data/raw/ if they aren't there yet,
checks them, and writes the processed files the recommender uses into
data/processed/. Stops with a clear error message if anything about the data
is wrong.
"""

import sys

import config
from curator.build import (
    ITEMS_FILE,
    MANIFEST_FILE,
    MATRIX_FILE,
    MEDIUM_DIRS,
    TAGS_FILE,
    BuildSettings,
    build,
    build_shared_tags,
    ensure_raw_files,
)
from curator.errors import CuratorError

# Files an older version of Curator wrote straight into data/processed/ (before
# books and movies got their own folders). Removed so they can't be mistaken
# for current data.
OLD_LAYOUT_FILES = ("books.csv", TAGS_FILE, MATRIX_FILE, MANIFEST_FILE)


def book_settings() -> BuildSettings:
    return BuildSettings(
        raw_dir=config.RAW_DIR,
        processed_dir=config.PROCESSED_DIR / MEDIUM_DIRS["book"],
        scores_file=config.SCORES_FILE,
        metadata_file=config.METADATA_FILE,
        tags_file=config.TAGS_FILE,
        expected_books=config.EXPECTED_BOOKS,
        expected_tags=config.EXPECTED_TAGS,
        expected_score_rows=config.EXPECTED_SCORE_ROWS,
        threshold_candidates=config.THRESHOLD_CANDIDATES,
        tag_threshold=config.TAG_THRESHOLD,
        medium="book",
        creator_field="authors",
    )


def movie_settings() -> BuildSettings:
    return BuildSettings(
        raw_dir=config.MOVIE_RAW_DIR,
        processed_dir=config.PROCESSED_DIR / MEDIUM_DIRS["movie"],
        scores_file=config.MOVIE_SCORES_FILE,
        metadata_file=config.MOVIE_METADATA_FILE,
        tags_file=config.MOVIE_TAGS_FILE,
        expected_books=config.EXPECTED_MOVIES,
        expected_tags=config.EXPECTED_MOVIE_TAGS,
        expected_score_rows=config.EXPECTED_MOVIE_SCORE_ROWS,
        threshold_candidates=config.THRESHOLD_CANDIDATES,
        tag_threshold=config.TAG_THRESHOLD,
        medium="movie",
        creator_field="directedBy",
        year_from_title=True,
        expected_tag_records=config.EXPECTED_MOVIE_TAG_RECORDS,
        known_without_metadata=config.MOVIES_WITHOUT_METADATA,
        known_incomplete=config.MOVIES_WITH_MISSING_SCORES,
    )


def print_summary(label: str, manifest: dict) -> None:
    c = manifest["counts"]
    print(f"\n{label}: {c['items']:,}   Tags: {c['tags']}   Score rows: {c['score_rows']:,}")
    print(f"Missing a creator: {c['items_missing_creator']}   missing a year: {c['items_missing_year']}")
    print(f"Titles shared by more than one item: {c['titles_shared_by_multiple_items']}")
    print("How many item-tag pairs clear each threshold:")
    print(f"  {'threshold':>9}  {'pairs':>10}  {'% pairs':>7}  {'tags/item (median)':>18}  {'items w/ 0 tags':>15}  {'tags on no item':>15}")
    for t, s in manifest["threshold_stats"].items():
        print(
            f"  {t:>9}  {s['pairs_at_or_above']:>10,}  {s['pct_of_all_pairs']:>7}  "
            f"{s['tags_per_item_median']:>18g}  {s['items_with_zero_tags']:>15}  {s['tags_applying_to_no_item']:>15}"
        )


def main() -> int:
    try:
        ensure_raw_files(
            config.RAW_DIR,
            config.DATASET_URL,
            config.DATASET_ZIP_NAME,
            [config.SCORES_FILE, config.METADATA_FILE, config.TAGS_FILE],
            tls_pin_sha256=config.DATASET_TLS_PIN_SHA256,
            zip_sha256=config.DATASET_ZIP_SHA256,
        )
        ensure_raw_files(
            config.MOVIE_RAW_DIR,
            config.MOVIE_DATASET_URL,
            config.MOVIE_DATASET_ZIP_NAME,
            [config.MOVIE_SCORES_FILE, config.MOVIE_METADATA_FILE, config.MOVIE_TAGS_FILE],
            tls_pin_sha256=config.DATASET_TLS_PIN_SHA256,
            zip_sha256=config.MOVIE_DATASET_ZIP_SHA256,
        )
        for name in OLD_LAYOUT_FILES:
            old = config.PROCESSED_DIR / name
            if old.is_file():
                print(f"Removing {old} (old layout; replaced by the books/ and movies/ folders)")
                old.unlink()
        books = build(book_settings())
        movies = build(movie_settings())
        shared = build_shared_tags(config.PROCESSED_DIR, config.EXPECTED_SHARED_TAGS_EXACT, config.SHARED_TAG_SPELLINGS)
    except CuratorError as exc:
        print(f"\nBUILD FAILED: {exc}", file=sys.stderr)
        return 1

    print_summary("Books", books)
    print_summary("Movies", movies)
    print(f"\nTags shared by books and movies: {len(shared)}")
    print(f"\nFull details: {config.PROCESSED_DIR}/books/{MANIFEST_FILE}, {config.PROCESSED_DIR}/movies/{MANIFEST_FILE}")
    print(f"(Catalog files: {ITEMS_FILE}, {TAGS_FILE}, {MATRIX_FILE} in each folder.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
