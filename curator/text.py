"""Title/author normalization used by both the build step and lookup."""

import re
import unicodedata

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_SPACES = re.compile(r"\s+")
# A trailing series note, e.g. "Dune (Dune Chronicles, #1)" -> "Dune"
_TRAILING_PAREN = re.compile(r"\s*\([^()]*\)\s*$")
# A movie title's trailing year, e.g. "Heat (1995)"
_TRAILING_YEAR = re.compile(r"\s*\((\d{4})\)\s*$")
# MovieLens moves leading articles to the end: "Three Musketeers, The",
# "Postino, Il", "Haine, La". These are the articles it uses that way.
_TRAILING_ARTICLE = re.compile(
    r"^(.*\S),\s*(The|A|An|La|Le|Les|L'|Il|Lo|Gli|I|Die|Der|Das|Den|Det|De|Het|En|El|Los|Las|Une|Un|Ein|Eine|O|Os|As)$"
)
# Separates the alternative normalized titles stored for one item.
ALIAS_SEP = "|"


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


def title_year(title: str) -> str:
    """The year at the end of a movie title ("Heat (1995)" -> "1995"), or ""."""
    m = _TRAILING_YEAR.search(title)
    return m.group(1) if m else ""


def _front_article(title: str) -> str:
    """"Three Musketeers, The" -> "The Three Musketeers"."""
    m = _TRAILING_ARTICLE.match(title.strip())
    return f"{m.group(2)} {m.group(1)}" if m else title.strip()


def book_aliases(title: str) -> list[str]:
    """Normalized titles a book can be looked up by (besides its full title)."""
    return [normalize(base_title(title))]


def movie_aliases(title: str) -> list[str]:
    """Normalized titles a movie can be looked up by (besides its full title).

    "Postman, The (Postino, Il) (1994)" -> ["the postman", "il postino"]
    "Seven (a.k.a. Se7en) (1995)"       -> ["seven", "se7en"]
    """
    t = _TRAILING_YEAR.sub("", title).strip() or title
    names = [t]
    m = re.match(r"^(.*\S)\s*\(([^()]*)\)$", t)
    if m:
        alt = re.sub(r"^(a\.k\.a\.|aka)\s+", "", m.group(2).strip(), flags=re.IGNORECASE)
        names = [m.group(1), alt]
    out = []
    for n in names:
        a = normalize(_front_article(n))
        if a and a not in out:
            out.append(a)
    return out
