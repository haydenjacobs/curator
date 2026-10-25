"""
Build step: raw dataset files -> validated processed files.

Reads (never modifies) the raw files in data/raw/, checks them thoroughly,
and writes one folder per medium (data/processed/books, data/processed/movies):
    items.csv      one row per book/movie (id, title, creator, year, ...)
    tags.csv       one row per tag (id, name)
    scores.npy     items x tags score matrix (rows/cols in the same order as
                   items.csv / tags.csv)
    manifest.json  counts, build date, source files, statistics
plus data/processed/shared_tags.csv: the tags books and movies both have.

Any problem with the data stops the build with a DataValidationError. Known
defects in the published data are only accepted when declared exactly in
config.py; anything else, or a declared defect that isn't there, stops the build.
"""

from __future__ import annotations

import csv
import datetime as _dt
import json
import shutil
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .download import download, sha256_file
from .errors import DataValidationError
from .text import ALIAS_SEP, book_aliases, movie_aliases, normalize, title_year

MEDIA = ("book", "movie")
MEDIUM_DIRS = {"book": "books", "movie": "movies"}   # folder names in data/processed
ITEMS_FILE = "items.csv"
TAGS_FILE = "tags.csv"
MATRIX_FILE = "scores.npy"
MANIFEST_FILE = "manifest.json"
SHARED_TAGS_FILE = "shared_tags.csv"
SHARED_TAGS_COLUMNS = ["name", "book_tag_id", "movie_name", "movie_tag_id"]
ITEM_COLUMNS = ["id", "title", "creator", "year", "title_normalized", "title_aliases", "creator_key"]
CREATOR_KEY_SEP = "|"  # separates multiple normalized authors/directors in creator_key


@dataclass(frozen=True)
class BuildSettings:
    raw_dir: Path
    processed_dir: Path
    scores_file: str
    metadata_file: str
    tags_file: str
    expected_books: int            # items with scores (books or movies)
    expected_tags: int             # tags with scores
    expected_score_rows: int
    threshold_candidates: tuple
    tag_threshold: float
    medium: str = "book"
    creator_field: str = "authors"         # "directedBy" for movies
    year_from_title: bool = False          # movies: year is the "(1995)" at the end of the title
    expected_tag_records: int | None = None  # entries in tags.json, if more than expected_tags
    known_without_metadata: tuple = ()     # scored item ids that have no metadata (left out)
    known_incomplete: dict = field(default_factory=dict)  # item id -> tag names it has no score for (left out)


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def ensure_raw_files(
    raw_dir: Path,
    url: str,
    zip_name: str,
    needed: list[str],
    tls_pin_sha256: str | None = None,
    zip_sha256: str | None = None,
    log=print,
) -> None:
    """Download the zip (if needed) and extract just the files we use (if needed)."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    missing = [n for n in needed if not (raw_dir / n).is_file()]
    if not missing:
        log(f"Raw files already present in {raw_dir} — skipping download.")
        return

    zip_path = raw_dir / zip_name
    if not zip_path.is_file():
        log(f"Downloading {url} (about 1.4 GB, this can take a while)...")
        download(url, zip_path, tls_pin_sha256, zip_sha256, log)
    elif zip_sha256:
        actual = sha256_file(zip_path)
        if actual != zip_sha256.lower():
            raise DataValidationError(
                f"{zip_path} doesn't match DATASET_ZIP_SHA256 in config.py (got {actual}). "
                "Delete it and run the build again."
            )

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


def _creator_fields(value, where: str) -> tuple[str, str]:
    """Return (display creator, creator_key) for a book's authors or a movie's
    directors. Missing -> ("", "")."""
    if value is None:
        names = []
    elif isinstance(value, str):
        # Co-authors / co-directors are stored as one comma-separated string,
        # e.g. "Neil Gaiman, Michael Reaves". (Checked on the real data: no
        # name contains a comma, such as "King, Jr.".)
        names = [n for n in value.split(",") if n.strip()]
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
    display = ", ".join(names)
    keys = []
    for n in names:
        k = normalize(n)
        if k and k not in keys:
            keys.append(k)
    return display, CREATOR_KEY_SEP.join(keys)


def _year_field(value, where: str) -> str:
    """Year as text ("" when missing). Non-year values are an error, not a guess.
    The book dataset uses 0 for an unknown year (1,256 books), so 0 counts as missing."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return ""
    year = _as_int(value, f"{where} year")
    if year == 0:
        return ""
    if year < 0:
        raise DataValidationError(f"{where}: negative year {year}")
    return str(year)


def read_metadata(path: Path, medium: str = "book", creator_field: str = "authors",
                  year_from_title: bool = False) -> pd.DataFrame:
    records = _read_json_records(path)
    aliases = movie_aliases if medium == "movie" else book_aliases
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
        creator, creator_key = _creator_fields(r.get(creator_field), f"{where} (item_id {item_id})")
        if year_from_title:
            year = title_year(title)
        else:
            year = _year_field(r.get("year"), f"{where} (item_id {item_id})")
        rows.append((item_id, title, creator, year, normalize(title), ALIAS_SEP.join(aliases(title)), creator_key))
    meta = pd.DataFrame(rows, columns=ITEM_COLUMNS)
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
            f"{path.name}: {int(dup_mask.sum()):,} rows are duplicate item-tag pairs, e.g.\n{df[dup_mask].head(6)}"
        )
    return df[["item_id", "tag_id", "score"]]


# ---------------------------------------------------------------------------
# Statistics for the manifest
# ---------------------------------------------------------------------------

def threshold_stats(matrix: np.ndarray, thresholds) -> dict:
    n_items, n_tags = matrix.shape
    out = {}
    for t in thresholds:
        above = matrix >= t
        per_item = above.sum(axis=1)
        per_tag = above.sum(axis=0)
        out[f"{t:g}"] = {
            "pairs_at_or_above": int(above.sum()),
            "pct_of_all_pairs": round(100.0 * float(above.mean()), 2),
            "tags_per_item_median": float(np.median(per_item)),
            "tags_per_item_mean": round(float(per_item.mean()), 1),
            "tags_per_item_p10": float(np.percentile(per_item, 10)),
            "tags_per_item_p90": float(np.percentile(per_item, 90)),
            "tags_per_item_min": int(per_item.min()),
            "tags_per_item_max": int(per_item.max()),
            "items_with_zero_tags": int((per_item == 0).sum()),
            "tags_applying_to_no_item": int((per_tag == 0).sum()),
            "tags_applying_to_over_half_of_items": int((per_tag > n_items / 2).sum()),
        }
    return out


# ---------------------------------------------------------------------------
# Main build
# ---------------------------------------------------------------------------

DATASET_NAMES = {
    "book": "Tag Genome Dataset for Books (Kotkov et al., 2022), GroupLens",
    "movie": "MovieLens Tag Genome Dataset 2021 (Kotkov et al., 2021; Vig et al., 2012), GroupLens",
}


def build(s: BuildSettings, log=print) -> dict:
    noun = s.medium
    scores_path = s.raw_dir / s.scores_file
    meta_path = s.raw_dir / s.metadata_file
    tags_path = s.raw_dir / s.tags_file

    log(f"Reading {noun} tags from {tags_path} ...")
    all_tags = read_tags(tags_path, s.expected_tag_records or s.expected_tags)
    log(f"Reading {noun} metadata from {meta_path} ...")
    meta = read_metadata(meta_path, s.medium, s.creator_field, s.year_from_title)
    log(f"Reading {noun} scores from {scores_path} (millions of rows, please wait) ...")
    scores = read_scores(scores_path, all_tags, s.expected_score_rows)
    n_score_rows = len(scores)

    # --- tags: exactly the scored ones ---
    scored_tag_ids = set(scores["tag_id"].unique())
    if len(scored_tag_ids) != s.expected_tags:
        raise DataValidationError(f"{s.scores_file}: expected {s.expected_tags} tags, found {len(scored_tag_ids)}")
    tags = all_tags[all_tags["id"].isin(scored_tag_ids)].reset_index(drop=True)
    unscored_tags = sorted(set(all_tags["name"]) - set(tags["name"]))
    if unscored_tags:
        log(f"Note: {len(unscored_tags)} tags in {s.tags_file} have no scores and are left out: {unscored_tags}")

    # --- items: exactly the scored ones, every one must have metadata ---
    scored_ids = np.sort(scores["item_id"].unique())
    if len(scored_ids) != s.expected_books:
        raise DataValidationError(f"{s.scores_file}: expected {s.expected_books:,} {noun}s, found {len(scored_ids):,}")

    meta_ids = set(meta["id"])
    missing_meta = sorted(int(i) for i in scored_ids if int(i) not in meta_ids)
    if missing_meta != sorted(s.known_without_metadata):
        raise DataValidationError(
            f"{len(missing_meta)} scored {noun}s have no entry in {s.metadata_file}, e.g. item_ids {missing_meta[:10]}"
            + (f" (config.py declares exactly these as known: {sorted(s.known_without_metadata)})"
               if s.known_without_metadata else "")
        )
    if missing_meta:
        log(f"Note: {len(missing_meta)} scored {noun}s have no title or other metadata and are left out "
            f"(declared in config.py): item_ids {missing_meta}")
    extra_meta = sorted(meta_ids - set(int(i) for i in scored_ids))
    if extra_meta:
        log(f"Note: {len(extra_meta):,} {noun}s in metadata have no scores and are left out of the catalog.")

    items = meta[meta["id"].isin(scored_ids)].sort_values("id", kind="stable").reset_index(drop=True)
    scores = scores[scores["item_id"].isin(items["id"])]

    # --- matrix: every (item, tag) cell must be filled exactly once ---
    row_of = pd.Series(np.arange(len(items)), index=items["id"].to_numpy())
    col_of = pd.Series(np.arange(len(tags)), index=tags["id"].to_numpy())
    matrix = np.full((len(items), len(tags)), np.nan, dtype=np.float64)
    matrix[row_of[scores["item_id"]].to_numpy(), col_of[scores["tag_id"]].to_numpy()] = scores["score"].to_numpy()
    gaps = {
        int(items.at[r, "id"]): sorted(tags["name"].iloc[np.flatnonzero(np.isnan(matrix[r]))])
        for r in np.flatnonzero(np.isnan(matrix).any(axis=1))
    }
    declared = {int(k): sorted(v) for k, v in s.known_incomplete.items()}
    if gaps != declared:
        raise DataValidationError(
            f"Score matrix has {int(np.isnan(matrix).sum()):,} missing {noun}-tag pairs (every {noun} must have every "
            f"tag). {noun.capitalize()}s with gaps: {dict(list(gaps.items())[:5])}"
            + (f" (config.py declares exactly these as known: {declared})" if declared else "")
        )
    if gaps:
        keep = ~items["id"].isin(list(gaps))
        log(f"Note: {len(gaps)} {noun}(s) with missing scores are left out (declared in config.py): "
            + "; ".join(f'{items.loc[items["id"] == i, "title"].iat[0]} (id {i}) has no score for {t}'
                        for i, t in gaps.items()))
        matrix = matrix[keep.to_numpy()]
        items = items[keep].reset_index(drop=True)

    # --- reporting (no effect on the data) ---
    first_alias = items["title_aliases"].str.split(ALIAS_SEP).str[0]
    dup_titles = items.groupby(first_alias)["id"].count()
    dup_titles = dup_titles[dup_titles > 1].sort_values(ascending=False, kind="stable")
    creator_counts = items[items["creator_key"] != ""].groupby("creator_key")["id"].count()
    creator_counts = creator_counts.sort_values(ascending=False, kind="stable")
    stats = threshold_stats(matrix, s.threshold_candidates)

    manifest = {
        "medium": s.medium,
        "build_date_utc": _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat(),
        "source": {
            "dataset": DATASET_NAMES[s.medium],
            "scores_file": s.scores_file,
            "scores_sha256": sha256_file(scores_path),
            "metadata_file": s.metadata_file,
            "tags_file": s.tags_file,
        },
        "counts": {
            "items": int(len(items)),
            "tags": int(len(tags)),
            "score_rows": int(n_score_rows),
            "metadata_records": int(len(meta)),
            "metadata_records_without_scores": len(extra_meta),
            "tags_without_scores": len(unscored_tags),
            "left_out_no_metadata": missing_meta,
            "left_out_missing_scores": {str(k): v for k, v in gaps.items()},
            "items_missing_creator": int((items["creator"] == "").sum()),
            "items_missing_year": int((items["year"] == "").sum()),
            "titles_shared_by_multiple_items": int(len(dup_titles)),
            "items_with_a_shared_title": int(dup_titles.sum()),
        },
        "score_quantiles": {
            f"p{q}": round(float(np.percentile(matrix, q)), 4) for q in (1, 10, 25, 50, 75, 90, 95, 99)
        },
        "threshold_stats": stats,
        "tag_threshold_at_build": s.tag_threshold,
        "examples": {
            "most_common_shared_titles": {k: int(v) for k, v in dup_titles.head(10).items()},
            "creators_with_most_items": {k: int(v) for k, v in creator_counts.head(10).items()},
        },
        "files": {"items": ITEMS_FILE, "tags": TAGS_FILE, "matrix": MATRIX_FILE},
    }

    # --- write (manifest last, so a half-finished build is never "valid") ---
    s.processed_dir.mkdir(parents=True, exist_ok=True)
    (s.processed_dir / MANIFEST_FILE).unlink(missing_ok=True)
    items.to_csv(s.processed_dir / ITEMS_FILE, index=False, columns=ITEM_COLUMNS, quoting=csv.QUOTE_NONNUMERIC)
    tags.to_csv(s.processed_dir / TAGS_FILE, index=False, quoting=csv.QUOTE_NONNUMERIC)
    np.save(s.processed_dir / MATRIX_FILE, matrix, allow_pickle=False)
    with open(s.processed_dir / MANIFEST_FILE, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")
    log(f"Build complete: {len(items):,} {noun}s × {len(tags)} tags written to {s.processed_dir}")
    return manifest


# ---------------------------------------------------------------------------
# Shared tags (books <-> movies)
# ---------------------------------------------------------------------------

def build_shared_tags(processed_dir: Path, expected_exact: int, renames: dict, log=print) -> pd.DataFrame:
    """Match book tags to movie tags and write shared_tags.csv.

    Tags match only when their names are identical, plus the book->movie
    spelling pairs declared in config.py (e.g. "anti war" -> "anti-war").
    Exact matches must number exactly `expected_exact`, and every declared
    pair must exist and not already be matched; anything else stops the build
    and lists the near misses so they can be checked by a person."""
    tags = {}
    for m in MEDIA:
        path = processed_dir / MEDIUM_DIRS[m] / TAGS_FILE
        if not path.is_file():
            raise DataValidationError(f"Missing {path}: build the {m} catalog first.")
        tags[m] = pd.read_csv(path, dtype={"name": str}, keep_default_na=False)
    book_id = dict(zip(tags["book"]["name"], tags["book"]["id"]))
    movie_id = dict(zip(tags["movie"]["name"], tags["movie"]["id"]))

    exact = [n for n in tags["book"]["name"] if n in movie_id]
    if len(exact) != expected_exact:
        by_norm: dict[str, list] = {}
        for n in movie_id:
            by_norm.setdefault(normalize(n), []).append(n)
        near = {b: by_norm[normalize(b)] for b in book_id if b not in movie_id and normalize(b) in by_norm}
        raise DataValidationError(
            f"Books and movies share {len(exact)} identically named tags, config.py expects {expected_exact}. "
            f"Book tags that differ from a movie tag only in punctuation/case: {near}"
        )
    for b, mv in renames.items():
        if b not in book_id or mv not in movie_id:
            raise DataValidationError(f"Declared shared tag {b!r} -> {mv!r}: not a book tag and a movie tag.")
        if b in movie_id or mv in book_id:
            raise DataValidationError(f"Declared shared tag {b!r} -> {mv!r}: one of them already matches exactly.")
    if len(set(renames.values())) != len(renames):
        raise DataValidationError("Declared shared tags map two book tags to the same movie tag.")

    pairs = {n: n for n in exact} | dict(renames)
    shared = pd.DataFrame(
        [(b, int(book_id[b]), pairs[b], int(movie_id[pairs[b]])) for b in tags["book"]["name"] if b in pairs],
        columns=SHARED_TAGS_COLUMNS,
    )
    shared.to_csv(processed_dir / SHARED_TAGS_FILE, index=False, quoting=csv.QUOTE_NONNUMERIC)
    log(f"Shared tags: {len(exact)} identical names + {len(renames)} declared spelling pairs = {len(shared)} "
        f"written to {processed_dir / SHARED_TAGS_FILE}")
    return shared
