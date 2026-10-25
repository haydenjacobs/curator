"""
Build step: raw dataset files -> validated processed files.

Reads (never modifies) the raw files in data/raw/, checks them thoroughly,
and writes:
    data/processed/books.csv     one row per book (id, title, author, year, ...)
    data/processed/tags.csv      one row per tag (id, name)
    data/processed/scores.npy    books x tags score matrix (rows/cols in the
                                 same order as books.csv / tags.csv)
    data/processed/manifest.json counts, build date, source files, statistics

Any problem with the data stops the build with a DataValidationError.
"""

from __future__ import annotations

import csv
import datetime as _dt
import hashlib
import json
import shutil
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .errors import DataValidationError
from .text import base_title, normalize

BOOKS_FILE = "books.csv"
TAGS_FILE = "tags.csv"
MATRIX_FILE = "scores.npy"
MANIFEST_FILE = "manifest.json"
BOOK_COLUMNS = ["id", "title", "author", "year", "title_normalized", "title_base_normalized", "author_key"]
AUTHOR_KEY_SEP = "|"  # separates multiple normalized authors in author_key


@dataclass(frozen=True)
class BuildSettings:
    raw_dir: Path
    processed_dir: Path
    scores_file: str
    metadata_file: str
    tags_file: str
    expected_books: int
    expected_tags: int
    expected_score_rows: int
    threshold_candidates: tuple
    tag_threshold: float


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def ensure_raw_files(raw_dir: Path, url: str, zip_name: str, needed: list[str], log=print) -> None:
    """Download the zip (if needed) and extract just the files we use (if needed)."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    missing = [n for n in needed if not (raw_dir / n).is_file()]
    if not missing:
        log(f"Raw files already present in {raw_dir} — skipping download.")
        return

    zip_path = raw_dir / zip_name
    if not zip_path.is_file():
        part = zip_path.with_suffix(zip_path.suffix + ".part")
        log(f"Downloading {url} (about 1.4 GB, this can take a while)...")
        try:
            with urllib.request.urlopen(url) as resp, open(part, "wb") as out:
                shutil.copyfileobj(resp, out, length=1024 * 1024)
        except Exception as exc:  # network errors are fatal and explained
            part.unlink(missing_ok=True)
            raise DataValidationError(f"Download failed from {url}: {exc}") from exc
        part.rename(zip_path)
        log(f"Saved {zip_path}")

    if not zipfile.is_zipfile(zip_path):
        raise DataValidationError(f"{zip_path} is not a valid zip file. Delete it and run the build again.")

    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if not n.endswith("/")]
        for wanted in missing:
            matches = [n for n in names if n == wanted or n.endswith("/" + wanted)]
            if len(matches) != 1:
                raise DataValidationError(
                    f"Expected exactly one '{wanted}' inside {zip_path.name}, found {len(matches)}: {matches[:5]}"
                )
            target = raw_dir / wanted  # our own path, never the archive's (no path tricks)
            target.parent.mkdir(parents=True, exist_ok=True)
            log(f"Extracting {matches[0]} -> {target}")
            with zf.open(matches[0]) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst, length=1024 * 1024)


# ---------------------------------------------------------------------------
# Readers + validation
# ---------------------------------------------------------------------------

def _read_json_records(path: Path) -> list[dict]:
    """Read a file that is either one JSON array or one JSON object per line."""
    if not path.is_file():
        raise DataValidationError(f"Missing raw file: {path}")
    text = path.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if stripped.startswith("["):
        try:
            records = json.loads(text)
        except json.JSONDecodeError as exc:
            raise DataValidationError(f"{path.name}: invalid JSON: {exc}") from exc
    else:
        records = []
        for lineno, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise DataValidationError(f"{path.name} line {lineno}: invalid JSON: {exc}") from exc
    for i, r in enumerate(records):
        if not isinstance(r, dict):
            raise DataValidationError(f"{path.name}: record {i + 1} is not a JSON object")
    return records


def _as_int(value, what: str) -> int:
    if isinstance(value, bool):
        raise DataValidationError(f"{what}: expected an integer, got {value!r}")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value.strip())
    raise DataValidationError(f"{what}: expected an integer, got {value!r}")


def read_tags(path: Path, expected_tags: int) -> pd.DataFrame:
    records = _read_json_records(path)
    rows = []
    for i, r in enumerate(records, start=1):
        if "id" not in r or "tag" not in r:
            raise DataValidationError(
                f"{path.name} record {i}: expected keys 'tag' and 'id', found {sorted(r.keys())}"
            )
        name = r["tag"]
        if not isinstance(name, str) or not name.strip():
            raise DataValidationError(f"{path.name} record {i}: empty or non-text tag name {name!r}")
        rows.append((_as_int(r["id"], f"{path.name} record {i} id"), name))
    tags = pd.DataFrame(rows, columns=["id", "name"])
    if len(tags) != expected_tags:
        raise DataValidationError(f"{path.name}: expected {expected_tags} tags, found {len(tags)}")
    dup_ids = tags[tags["id"].duplicated(keep=False)]
    if len(dup_ids):
        raise DataValidationError(f"{path.name}: duplicate tag ids: {sorted(set(dup_ids['id']))[:10]}")
    dup_names = tags[tags["name"].duplicated(keep=False)]
    if len(dup_names):
        raise DataValidationError(f"{path.name}: duplicate tag names: {sorted(set(dup_names['name']))[:10]}")
    return tags.sort_values("id", kind="stable").reset_index(drop=True)


def _author_fields(value, where: str) -> tuple[str, str]:
    """Return (display author, author_key). Missing author -> ("", "")."""
    if value is None:
        names = []
    elif isinstance(value, str):
        names = [value] if value.strip() else []
    elif isinstance(value, list):
        names = []
        for a in value:
            if isinstance(a, dict) and isinstance(a.get("name"), str):
                a = a["name"]
            if not isinstance(a, str):
                raise DataValidationError(f"{where}: unexpected author entry {a!r}")
            if a.strip():
                names.append(a.strip())
    else:
        raise DataValidationError(f"{where}: unexpected authors value {value!r}")
    names = [" ".join(n.split()) for n in names]
    keys = []
    for n in names:
        k = normalize(n)
        if k and k not in keys:
            keys.append(k)
    return ", ".join(names), AUTHOR_KEY_SEP.join(keys)


def _year_field(value, where: str) -> str:
    """Year as text ("" when missing). Non-year values are an error, not a guess."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return ""
    year = _as_int(value, f"{where} year")
    return str(year)


def read_metadata(path: Path) -> pd.DataFrame:
    records = _read_json_records(path)
    rows = []
    for i, r in enumerate(records, start=1):
        where = f"{path.name} record {i}"
        for key in ("item_id", "title"):
            if key not in r:
                raise DataValidationError(f"{where}: missing '{key}' (keys found: {sorted(r.keys())})")
        item_id = _as_int(r["item_id"], f"{where} item_id")
        title = r["title"]
        if not isinstance(title, str) or not title.strip():
            raise DataValidationError(f"{where} (item_id {item_id}): missing title")
        title = " ".join(title.split())
        author, author_key = _author_fields(r.get("authors"), f"{where} (item_id {item_id})")
        year = _year_field(r.get("year"), f"{where} (item_id {item_id})")
        rows.append((item_id, title, author, year, normalize(title), normalize(base_title(title)), author_key))
    meta = pd.DataFrame(rows, columns=BOOK_COLUMNS)
    dups = meta[meta["id"].duplicated(keep=False)]
    if len(dups):
        raise DataValidationError(f"{path.name}: duplicate item_id values: {sorted(set(dups['id']))[:10]}")
    return meta


def read_scores(path: Path, tags: pd.DataFrame, expected_rows: int) -> pd.DataFrame:
    """Read glmer.csv and return columns item_id, tag_id, score (fully validated)."""
    if not path.is_file():
        raise DataValidationError(f"Missing raw file: {path}")
    header = pd.read_csv(path, nrows=0).columns.tolist()
    if sorted(header) != ["item_id", "score", "tag"]:
        raise DataValidationError(f"{path.name}: expected columns tag, item_id, score; found {header}")
    try:
        # Tags are read as text so a tag literally named e.g. "NA" isn't turned
        # into a blank; blank item_id/score fields become NaN and are caught below.
        df = pd.read_csv(
            path,
            dtype={"tag": "str", "item_id": "float64", "score": "float64"},
            keep_default_na=False,
            na_values={"item_id": [""], "score": [""]},
        )
    except ValueError as exc:
        raise DataValidationError(f"{path.name}: could not parse values: {exc}") from exc

    if len(df) != expected_rows:
        raise DataValidationError(f"{path.name}: expected {expected_rows:,} rows, found {len(df):,}")
    if df.isna().any().any():
        bad = df[df.isna().any(axis=1)].head(5)
        raise DataValidationError(f"{path.name}: {int(df.isna().any(axis=1).sum())} rows with missing values, e.g.\n{bad}")

    if not (df["item_id"] == np.floor(df["item_id"])).all():
        raise DataValidationError(f"{path.name}: some item_id values are not whole numbers")
    df["item_id"] = df["item_id"].astype("int64")

    out_of_range = df[(df["score"] < 0.0) | (df["score"] > 1.0)]
    if len(out_of_range):
        raise DataValidationError(
            f"{path.name}: {len(out_of_range):,} scores outside 0–1, e.g.\n{out_of_range.head(5)}"
        )

    # The 'tag' column may hold tag ids or tag names; map either to tag ids.
    # If it matches neither exactly, stop.
    tag_col = df["tag"]
    distinct = tag_col.unique()
    if all(v.isdigit() for v in distinct):
        id_set = set(tags["id"])
        unknown = {v for v in distinct if int(v) not in id_set}
        if unknown:
            raise DataValidationError(f"{path.name}: tag ids not in tags.json: {sorted(unknown)[:10]}")
        df["tag_id"] = tag_col.astype("int64")
    else:
        name_to_id = dict(zip(tags["name"], tags["id"]))
        unknown = set(distinct) - set(name_to_id)
        if unknown:
            raise DataValidationError(f"{path.name}: tag names not in tags.json: {sorted(unknown)[:10]}")
        df["tag_id"] = tag_col.map(name_to_id).astype("int64")

    dup_mask = df.duplicated(["item_id", "tag_id"], keep=False)
    if dup_mask.any():
        raise DataValidationError(
            f"{path.name}: {int(dup_mask.sum()):,} rows are duplicate book-tag pairs, e.g.\n{df[dup_mask].head(6)}"
        )
    return df[["item_id", "tag_id", "score"]]


# ---------------------------------------------------------------------------
# Statistics for the manifest
# ---------------------------------------------------------------------------

def threshold_stats(matrix: np.ndarray, thresholds) -> dict:
    n_books, n_tags = matrix.shape
    out = {}
    for t in thresholds:
        above = matrix >= t
        per_book = above.sum(axis=1)
        per_tag = above.sum(axis=0)
        out[f"{t:g}"] = {
            "pairs_at_or_above": int(above.sum()),
            "pct_of_all_pairs": round(100.0 * float(above.mean()), 2),
            "tags_per_book_median": float(np.median(per_book)),
            "tags_per_book_mean": round(float(per_book.mean()), 1),
            "tags_per_book_p10": float(np.percentile(per_book, 10)),
            "tags_per_book_p90": float(np.percentile(per_book, 90)),
            "tags_per_book_min": int(per_book.min()),
            "tags_per_book_max": int(per_book.max()),
            "books_with_zero_tags": int((per_book == 0).sum()),
            "tags_applying_to_no_book": int((per_tag == 0).sum()),
            "tags_applying_to_over_half_of_books": int((per_tag > n_books / 2).sum()),
        }
    return out


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Main build
# ---------------------------------------------------------------------------

def build(s: BuildSettings, log=print) -> dict:
    scores_path = s.raw_dir / s.scores_file
    meta_path = s.raw_dir / s.metadata_file
    tags_path = s.raw_dir / s.tags_file

    log(f"Reading tags from {tags_path} ...")
    tags = read_tags(tags_path, s.expected_tags)
    log(f"Reading book metadata from {meta_path} ...")
    meta = read_metadata(meta_path)
    log(f"Reading scores from {scores_path} (millions of rows, please wait) ...")
    scores = read_scores(scores_path, tags, s.expected_score_rows)

    # --- books: exactly the scored books, every one must have metadata ---
    scored_ids = np.sort(scores["item_id"].unique())
    if len(scored_ids) != s.expected_books:
        raise DataValidationError(f"{s.scores_file}: expected {s.expected_books:,} books, found {len(scored_ids):,}")
    tag_ids_in_scores = scores["tag_id"].nunique()
    if tag_ids_in_scores != s.expected_tags:
        raise DataValidationError(f"{s.scores_file}: expected {s.expected_tags} tags, found {tag_ids_in_scores}")

    meta_ids = set(meta["id"])
    missing_meta = [int(i) for i in scored_ids if int(i) not in meta_ids]
    if missing_meta:
        raise DataValidationError(
            f"{len(missing_meta)} scored books have no entry in {s.metadata_file}, e.g. item_ids {missing_meta[:10]}"
        )
    extra_meta = sorted(meta_ids - set(int(i) for i in scored_ids))
    if extra_meta:
        log(f"Note: {len(extra_meta)} books in metadata have no scores and are left out of the catalog.")

    books = (
        meta[meta["id"].isin(scored_ids)].sort_values("id", kind="stable").reset_index(drop=True)
    )

    # --- matrix: every (book, tag) cell must be filled exactly once ---
    row_of = pd.Series(np.arange(len(books)), index=books["id"].to_numpy())
    col_of = pd.Series(np.arange(len(tags)), index=tags["id"].to_numpy())
    matrix = np.full((len(books), len(tags)), np.nan, dtype=np.float64)
    matrix[row_of[scores["item_id"]].to_numpy(), col_of[scores["tag_id"]].to_numpy()] = scores["score"].to_numpy()
    if np.isnan(matrix).any():
        missing = int(np.isnan(matrix).sum())
        raise DataValidationError(f"Score matrix has {missing:,} missing book-tag pairs (every book must have every tag)")

    # --- reporting (no effect on the data) ---
    dup_titles = books.groupby("title_base_normalized")["id"].count()
    dup_titles = dup_titles[dup_titles > 1].sort_values(ascending=False, kind="stable")
    author_counts = books[books["author_key"] != ""].groupby("author_key")["id"].count()
    author_counts = author_counts.sort_values(ascending=False, kind="stable")
    stats = threshold_stats(matrix, s.threshold_candidates)

    manifest = {
        "build_date_utc": _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat(),
        "source": {
            "dataset": "Tag Genome Dataset for Books (Kotkov et al., 2022), GroupLens",
            "scores_file": s.scores_file,
            "scores_sha256": _sha256(scores_path),
            "metadata_file": s.metadata_file,
            "tags_file": s.tags_file,
        },
        "counts": {
            "books": int(len(books)),
            "tags": int(len(tags)),
            "score_rows": int(len(scores)),
            "metadata_records": int(len(meta)),
            "metadata_records_without_scores": len(extra_meta),
            "books_missing_author": int((books["author"] == "").sum()),
            "books_missing_year": int((books["year"] == "").sum()),
            "titles_shared_by_multiple_books": int(len(dup_titles)),
            "books_with_a_shared_title": int(dup_titles.sum()),
        },
        "score_quantiles": {
            f"p{q}": round(float(np.percentile(matrix, q)), 4) for q in (1, 10, 25, 50, 75, 90, 95, 99)
        },
        "threshold_stats": stats,
        "tag_threshold_at_build": s.tag_threshold,
        "examples": {
            "most_common_shared_titles": {k: int(v) for k, v in dup_titles.head(10).items()},
            "authors_with_most_books": {k: int(v) for k, v in author_counts.head(10).items()},
        },
        "files": {"books": BOOKS_FILE, "tags": TAGS_FILE, "matrix": MATRIX_FILE},
    }

    # --- write (manifest last, so a half-finished build is never "valid") ---
    s.processed_dir.mkdir(parents=True, exist_ok=True)
    (s.processed_dir / MANIFEST_FILE).unlink(missing_ok=True)
    books.to_csv(s.processed_dir / BOOKS_FILE, index=False, columns=BOOK_COLUMNS, quoting=csv.QUOTE_NONNUMERIC)
    tags.to_csv(s.processed_dir / TAGS_FILE, index=False, quoting=csv.QUOTE_NONNUMERIC)
    np.save(s.processed_dir / MATRIX_FILE, matrix, allow_pickle=False)
    with open(s.processed_dir / MANIFEST_FILE, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")
    log(f"Build complete: {len(books):,} books × {len(tags)} tags written to {s.processed_dir}")
    return manifest
