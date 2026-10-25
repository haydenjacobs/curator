"""
Loads the processed files (and ONLY the processed files) into memory.

This is the single boundary between the data source and the algorithm: if the
data source changes later, a new build step just needs to produce these same
processed files.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .build import (
    CREATOR_KEY_SEP,
    ITEM_COLUMNS,
    ITEMS_FILE,
    MANIFEST_FILE,
    MATRIX_FILE,
    MEDIA,
    MEDIUM_DIRS,
    SHARED_TAGS_COLUMNS,
    SHARED_TAGS_FILE,
    TAGS_FILE,
)
from .errors import DataValidationError


@dataclass(frozen=True)
class Catalog:
    medium: str                # "book" or "movie"
    items: pd.DataFrame        # columns: ITEM_COLUMNS; row i <-> matrix row i
    tags: pd.DataFrame         # columns: id, name;      row j <-> matrix column j
    scores: np.ndarray         # shape (n_items, n_tags), values in [0, 1]
    manifest: dict

    @property
    def n_items(self) -> int:
        return self.scores.shape[0]

    def creator_keys(self, row: int) -> frozenset[str]:
        key = self.items.at[row, "creator_key"]
        return frozenset(key.split(CREATOR_KEY_SEP)) if key else frozenset()

    def item_info(self, row: int) -> dict:
        b = self.items.iloc[row]
        return {"id": int(b["id"]), "title": b["title"], "creator": b["creator"], "year": b["year"],
                "medium": self.medium}


@dataclass(frozen=True)
class Library:
    """Every catalog that was built, plus the tags books and movies share."""
    catalogs: dict             # medium -> Catalog
    shared_tags: pd.DataFrame | None  # columns: SHARED_TAGS_COLUMNS; None if only one medium


def load_catalog(processed_dir: Path, medium: str = "book") -> Catalog:
    """Load one medium's catalog from its own folder (e.g. data/processed/books)."""
    manifest_path = processed_dir / MANIFEST_FILE
    if not manifest_path.is_file():
        raise DataValidationError(
            f"No processed data found in {processed_dir}. Run the build first:  python build.py"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("medium") != medium:
        raise DataValidationError(
            f"{processed_dir} holds {manifest.get('medium')!r} data, expected {medium!r}. Re-run: python build.py"
        )

    items = pd.read_csv(processed_dir / ITEMS_FILE, dtype=str, keep_default_na=False)
    if list(items.columns) != ITEM_COLUMNS:
        raise DataValidationError(f"{ITEMS_FILE}: unexpected columns {list(items.columns)}")
    items["id"] = items["id"].astype("int64")
    tags = pd.read_csv(processed_dir / TAGS_FILE, dtype=str, keep_default_na=False)
    if list(tags.columns) != ["id", "name"]:
        raise DataValidationError(f"{TAGS_FILE}: unexpected columns {list(tags.columns)}")
    tags["id"] = tags["id"].astype("int64")
    scores = np.load(processed_dir / MATRIX_FILE, allow_pickle=False)

    counts = manifest.get("counts", {})
    if scores.shape != (len(items), len(tags)):
        raise DataValidationError(
            f"Processed files disagree: matrix is {scores.shape}, {ITEMS_FILE} has {len(items)} rows, tags.csv {len(tags)}"
        )
    if (counts.get("items"), counts.get("tags")) != (len(items), len(tags)):
        raise DataValidationError("Processed files don't match manifest.json counts. Re-run: python build.py")
    if not np.isfinite(scores).all() or scores.min() < 0 or scores.max() > 1:
        raise DataValidationError(f"{MATRIX_FILE}: contains values outside 0–1. Re-run: python build.py")
    if items["id"].duplicated().any() or tags["name"].duplicated().any():
        raise DataValidationError("Processed files contain duplicate ids or tag names. Re-run: python build.py")
    return Catalog(medium=medium, items=items, tags=tags, scores=scores, manifest=manifest)


def load_shared_tags(path: Path, books: Catalog, movies: Catalog) -> pd.DataFrame:
    """The tags that exist in both catalogs, checked against both tags.csv files."""
    if not path.is_file():
        raise DataValidationError(f"Missing {path}. Run the build first:  python build.py")
    shared = pd.read_csv(path, dtype=str, keep_default_na=False)
    if list(shared.columns) != SHARED_TAGS_COLUMNS:
        raise DataValidationError(f"{SHARED_TAGS_FILE}: unexpected columns {list(shared.columns)}")
    for col in ("book_tag_id", "movie_tag_id"):
        shared[col] = shared[col].astype("int64")
    if shared["name"].duplicated().any() or shared["book_tag_id"].duplicated().any() or shared["movie_tag_id"].duplicated().any():
        raise DataValidationError(f"{SHARED_TAGS_FILE}: duplicate entries. Re-run: python build.py")
    for medium, cat, name_col, id_col in (("book", books, "name", "book_tag_id"),
                                          ("movie", movies, "movie_name", "movie_tag_id")):
        id_of = dict(zip(cat.tags["name"], cat.tags["id"]))
        bad = [n for n, i in zip(shared[name_col], shared[id_col]) if id_of.get(n) != i]
        if bad:
            raise DataValidationError(
                f"{SHARED_TAGS_FILE}: tags that don't match the {medium} tags.csv: {bad[:10]}. Re-run: python build.py"
            )
    return shared


def load_library(processed_dir: Path) -> Library:
    """Load data/processed/books, data/processed/movies and the shared tag list."""
    catalogs = {m: load_catalog(processed_dir / MEDIUM_DIRS[m], m) for m in MEDIA}
    shared = load_shared_tags(processed_dir / SHARED_TAGS_FILE, catalogs["book"], catalogs["movie"])
    return Library(catalogs=catalogs, shared_tags=shared)
