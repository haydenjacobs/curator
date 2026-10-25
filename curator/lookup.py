"""
Find input books and movies in their catalogs.

Rules (deterministic, never guesses), the same for books and movies:
  * "id:12345" looks an item up by its id.
  * Otherwise the text is a title, compared case-insensitively (accents and
    punctuation ignored). Tried in this order, stopping at the first step that
    finds anything:
      1. exact full title           ("Dune (Dune Chronicles, #1)", "Heat (1995)")
      2. one of its other names: books without the trailing series note
         ("Dune"); movies without the year, with a moved article put back in
         front, or by their alternative title ("Heat", "The Three
         Musketeers", "Se7en")
    One match -> that item. Several -> AmbiguousItemError listing them.
  * If neither finds anything, titles *containing* the text are listed as
    suggestions in an ItemNotFoundError -- they are never picked automatically.
"""

from __future__ import annotations

from .data import Catalog
from .errors import AmbiguousItemError, ItemNotFoundError, describe_candidate
from .text import ALIAS_SEP, normalize

ID_PREFIX = "id:"
MAX_SUGGESTIONS = 10


def _candidates(catalog: Catalog, rows) -> list[dict]:
    rows = sorted(rows, key=lambda r: int(catalog.items.at[r, "id"]))
    return [catalog.item_info(r) for r in rows]


def find_item(catalog: Catalog, query: str) -> int:
    """Return the catalog row index for one query, or raise a clear error."""
    noun = catalog.medium
    q = query.strip()
    if not q:
        raise ItemNotFoundError(f"Empty {noun} title given.")
    items = catalog.items

    if q.lower().startswith(ID_PREFIX):
        raw = q[len(ID_PREFIX):].strip()
        if not raw.isdigit():
            raise ItemNotFoundError(f'"{query}" is not a valid id (expected e.g. id:12345).')
        rows = items.index[items["id"] == int(raw)].tolist()
        if not rows:
            raise ItemNotFoundError(f"No {noun} with id {raw} in the {noun} catalog.")
        return rows[0]

    norm = normalize(q)
    if not norm:
        raise ItemNotFoundError(f'"{query}" has no letters or numbers to search for.')
    exact = items["title_normalized"] == norm
    alias = items["title_aliases"].map(lambda a: norm in a.split(ALIAS_SEP))
    for mask in (exact, alias):
        rows = items.index[mask].tolist()
        if len(rows) == 1:
            return rows[0]
        if len(rows) > 1:
            raise AmbiguousItemError(query, _candidates(catalog, rows), noun)

    contains = items.index[items["title_normalized"].str.contains(norm, regex=False)].tolist()
    msg = f'"{query}" is not in the {noun} catalog.'
    if contains:
        shown = _candidates(catalog, contains)[:MAX_SUGGESTIONS]
        msg += f" {len(contains)} title(s) contain that text"
        msg += " (showing the first 10)" if len(contains) > MAX_SUGGESTIONS else ""
        msg += ":\n" + "\n".join(f"  {describe_candidate(c)}" for c in shown)
    raise ItemNotFoundError(msg)
