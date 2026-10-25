"""
Build step. Run:   python build.py

Downloads the dataset into data/raw/ if it isn't there yet, checks it, and
writes the processed files the recommender uses into data/processed/.
Stops with a clear error message if anything about the data is wrong.
"""

import sys

import config
from curator.build import BuildSettings, build, ensure_raw_files
from curator.errors import CuratorError


def main() -> int:
    try:
        ensure_raw_files(
            config.RAW_DIR,
            config.DATASET_URL,
            config.DATASET_ZIP_NAME,
            [config.SCORES_FILE, config.METADATA_FILE, config.TAGS_FILE],
        )
        manifest = build(
            BuildSettings(
                raw_dir=config.RAW_DIR,
                processed_dir=config.PROCESSED_DIR,
                scores_file=config.SCORES_FILE,
                metadata_file=config.METADATA_FILE,
                tags_file=config.TAGS_FILE,
                expected_books=config.EXPECTED_BOOKS,
                expected_tags=config.EXPECTED_TAGS,
                expected_score_rows=config.EXPECTED_SCORE_ROWS,
                threshold_candidates=config.THRESHOLD_CANDIDATES,
                tag_threshold=config.TAG_THRESHOLD,
            )
        )
    except CuratorError as exc:
        print(f"\nBUILD FAILED: {exc}", file=sys.stderr)
        return 1

    c = manifest["counts"]
    print(f"\nBooks: {c['books']:,}   Tags: {c['tags']}   Score rows: {c['score_rows']:,}")
    print(f"Books missing an author: {c['books_missing_author']}   missing a year: {c['books_missing_year']}")
    print(f"Titles shared by more than one book: {c['titles_shared_by_multiple_books']}")
    print("\nHow many book-tag pairs clear each threshold:")
    print(f"  {'threshold':>9}  {'pairs':>10}  {'% pairs':>7}  {'tags/book (median)':>18}  {'books w/ 0 tags':>15}  {'tags on no book':>15}")
    for t, s in manifest["threshold_stats"].items():
        print(
            f"  {t:>9}  {s['pairs_at_or_above']:>10,}  {s['pct_of_all_pairs']:>7}  "
            f"{s['tags_per_book_median']:>18g}  {s['books_with_zero_tags']:>15}  {s['tags_applying_to_no_book']:>15}"
        )
    print(f"\nFull details: {config.PROCESSED_DIR / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
