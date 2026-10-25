"""Title/author normalization used by both the build step and lookup."""

import re
import unicodedata

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_SPACES = re.compile(r"\s+")
# A trailing series note, e.g. "Dune (Dune Chronicles, #1)" -> "Dune"
_TRAILING_PAREN = re.compile(r"\s*\([^()]*\)\s*$")


def normalize(text: str) -> str:
    """Lowercase, strip accents and punctuation, collapse spaces."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.casefold()
    text = _PUNCT.sub(" ", text)
    return _SPACES.sub(" ", text).strip()


def base_title(title: str) -> str:
    """Title with a trailing parenthetical (usually the series) removed."""
    stripped = _TRAILING_PAREN.sub("", title).strip()
    return stripped if stripped else title
