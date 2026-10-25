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

from .build import AUTHOR_KEY_SEP, BOOK_COLUMNS, BOOKS_FILE, MANIFEST_FILE, MATRIX_FILE, TAGS_FILE
from .errors import DataValidationError


@dataclass(frozen=True)
class Catalog:
    books: pd.DataFrame        # columns: BOOK_COLUMNS; row i <-> matrix row i
    tags: pd.DataFrame         # columns: id, name;     row j <-> matrix column j
    scores: np.ndarray         # shape (n_books, n_tags), values in [0, 1]
    manifest: dict

    @property
    def n_books(self) -> int:
        return self.scores.shape[0]

    def author_keys(self, row: int) -> frozenset[str]:
        key = self.books.at[row, "author_key"]
        return frozenset(key.split(AUTHOR_KEY_SEP)) if key else frozenset()

    def book_info(self, row: int) -> dict:
        b = self.books.iloc[row]
        return {"id": int(b["id"]), "title": b["title"], "author": b["author"], "year": b["year"]}


def load_catalog(processed_dir: Path) -> Catalog:
    manifest_path = processed_dir / MANIFEST_FILE
    if not manifest_path.is_file():
        raise DataValidationError(
            f"No processed data found in {processed_dir}. Run the build first:  python build.py"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    books = pd.read_csv(processed_dir / BOOKS_FILE, dtype=str, keep_default_na=False)
    if list(books.columns) != BOOK_COLUMNS:
        raise DataValidationError(f"{BOOKS_FILE}: unexpected columns {list(books.columns)}")
    books["id"] = books["id"].astype("int64")
    tags = pd.read_csv(processed_dir / TAGS_FILE, dtype=str, keep_default_na=False)
    if list(tags.columns) != ["id", "name"]:
        raise DataValidationError(f"{TAGS_FILE}: unexpected columns {list(tags.columns)}")
    tags["id"] = tags["id"].astype("int64")
    scores = np.load(processed_dir / MATRIX_FILE, allow_pickle=False)

    counts = manifest.get("counts", {})
    if scores.shape != (len(books), len(tags)):
        raise DataValidationError(
            f"Processed files disagree: matrix is {scores.shape}, books.csv has {len(books)} rows, tags.csv {len(tags)}"
        )
    if (counts.get("books"), counts.get("tags")) != (len(books), len(tags)):
        raise DataValidationError("Processed files don't match manifest.json counts. Re-run: python build.py")
    if not np.isfinite(scores).all() or scores.min() < 0 or scores.max() > 1:
        raise DataValidationError(f"{MATRIX_FILE}: contains values outside 0–1. Re-run: python build.py")
    if books["id"].duplicated().any() or tags["name"].duplicated().any():
        raise DataValidationError("Processed files contain duplicate book ids or tag names. Re-run: python build.py")
    return Catalog(books=books, tags=tags, scores=scores, manifest=manifest)
