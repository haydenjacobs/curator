"""
Find input books in the catalog.

Rules (deterministic, never guesses):
  * "id:12345" looks a book up by its id.
  * Otherwise the text is a title, compared case-insensitively (accents and
    punctuation ignored). Tried in this order, stopping at the first step that
    finds anything:
      1. exact full title           ("Dune (Dune Chronicles, #1)")
      2. exact title without its trailing series note in brackets ("Dune")
    One match -> that book. Several -> AmbiguousBookError listing them.
  * If neither finds anything, titles *containing* the text are listed as
    suggestions in a BookNotFoundError -- they are never picked automatically.
"""

from __future__ import annotations

from .data import Catalog
from .errors import AmbiguousBookError, BookNotFoundError
from .text import normalize

ID_PREFIX = "id:"
MAX_SUGGESTIONS = 10


def _candidates(catalog: Catalog, rows) -> list[dict]:
    rows = sorted(rows, key=lambda r: int(catalog.books.at[r, "id"]))
    return [catalog.book_info(r) for r in rows]


def find_book(catalog: Catalog, query: str) -> int:
    """Return the catalog row index for one query, or raise a clear error."""
    q = query.strip()
    if not q:
        raise BookNotFoundError("Empty book title given.")
    books = catalog.books

    if q.lower().startswith(ID_PREFIX):
        raw = q[len(ID_PREFIX):].strip()
        if not raw.isdigit():
            raise BookNotFoundError(f'"{query}" is not a valid id (expected e.g. id:12345).')
        rows = books.index[books["id"] == int(raw)].tolist()
        if not rows:
            raise BookNotFoundError(f"No book with id {raw} in the catalog.")
        return rows[0]

    norm = normalize(q)
    if not norm:
        raise BookNotFoundError(f'"{query}" has no letters or numbers to search for.')
    for column in ("title_normalized", "title_base_normalized"):
        rows = books.index[books[column] == norm].tolist()
        if len(rows) == 1:
            return rows[0]
        if len(rows) > 1:
            raise AmbiguousBookError(query, _candidates(catalog, rows))

    contains = books.index[books["title_normalized"].str.contains(norm, regex=False)].tolist()
    msg = f'"{query}" is not in the catalog.'
    if contains:
        shown = _candidates(catalog, contains)[:MAX_SUGGESTIONS]
        msg += f" {len(contains)} title(s) contain that text"
        msg += " (showing the first 10)" if len(contains) > MAX_SUGGESTIONS else ""
        msg += ":\n" + "\n".join(
            f"  id:{c['id']}  {c['title']} — {c['author'] or 'unknown author'} ({c['year'] or 'unknown year'})"
            for c in shown
        )
    raise BookNotFoundError(msg)
