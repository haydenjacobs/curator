"""
The recommendation algorithm (content-based filtering). No I/O, no printing:
a command line or a web server can both call this the same way.

Steps:
  1. Look up the input books (see lookup.py).
  2. Tag rarity weight  w_t = log(N / n_t), where n_t = books whose score for
     tag t is >= TAG_THRESHOLD. Tags that apply to no book get weight 0.
  3. Taste profile      profile_t = mean over inputs of (input weight x score).
                        Likes are +1; negative weights (dislikes) work too.
  4. Tag weight         weight_t = w_t x multiplier_t   (multipliers default 1.0)
  5. Similarity         cosine( profile x weight,  candidate scores x weight )
  6. Exclusions         input books are never recommended; at most
                        MAX_PER_AUTHOR results per author.
  7. Output             top N, each with the shared tags that contributed most.
Ties are broken by book id, so the same input always gives the same output.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .data import Catalog
from .errors import ConfigError, CuratorError
from .lookup import find_book

# Similarities are rounded to this many decimals before ranking, so tiny
# floating-point noise can never reorder results; exact ties go to the lower id.
_RANK_DECIMALS = 10


@dataclass(frozen=True)
class EngineSettings:
    tag_threshold: float
    top_n: int = 3
    explain_tags: int = 5
    max_per_author: int | None = 1
    exclude_input_authors: bool = False
    like_weight: float = 1.0
    default_tag_multiplier: float = 1.0
    tag_multipliers: dict = field(default_factory=dict)

    @classmethod
    def from_config(cls, cfg) -> "EngineSettings":
        return cls(
            tag_threshold=cfg.TAG_THRESHOLD,
            top_n=cfg.TOP_N,
            explain_tags=cfg.EXPLAIN_TAGS,
            max_per_author=cfg.MAX_PER_AUTHOR,
            exclude_input_authors=cfg.EXCLUDE_INPUT_AUTHORS,
            like_weight=cfg.LIKE_WEIGHT,
            default_tag_multiplier=cfg.DEFAULT_TAG_MULTIPLIER,
            tag_multipliers=dict(cfg.TAG_MULTIPLIERS),
        )


@dataclass(frozen=True)
class SharedTag:
    name: str
    contribution: float     # this tag's share of the similarity score (they add up to it)
    candidate_score: float  # the recommended book's score for the tag
    profile_score: float    # your taste profile's score for the tag


@dataclass(frozen=True)
class Recommendation:
    rank: int
    book: dict              # id, title, author, year
    similarity: float       # cosine similarity, -1..1 (0..1 when there are only likes)
    shared_tags: list[SharedTag]


@dataclass(frozen=True)
class RecommendationResult:
    inputs: list[dict]      # the resolved input books (id, title, author, year, weight)
    recommendations: list[Recommendation]


def _validate(settings: EngineSettings, catalog: Catalog) -> None:
    if not (0.0 < settings.tag_threshold <= 1.0):
        raise ConfigError(f"TAG_THRESHOLD must be > 0 and <= 1, got {settings.tag_threshold}")
    if not (isinstance(settings.top_n, int) and settings.top_n >= 1):
        raise ConfigError(f"TOP_N must be a whole number >= 1, got {settings.top_n}")
    if not (isinstance(settings.explain_tags, int) and settings.explain_tags >= 0):
        raise ConfigError(f"EXPLAIN_TAGS must be a whole number >= 0, got {settings.explain_tags}")
    if settings.max_per_author is not None and not (
        isinstance(settings.max_per_author, int) and settings.max_per_author >= 1
    ):
        raise ConfigError(f"MAX_PER_AUTHOR must be None or a whole number >= 1, got {settings.max_per_author}")
    if settings.like_weight <= 0:
        raise ConfigError(f"LIKE_WEIGHT must be positive, got {settings.like_weight}")
    known = set(catalog.tags["name"])
    unknown = sorted(set(settings.tag_multipliers) - known)
    if unknown:
        raise ConfigError(f"TAG_MULTIPLIERS names tags that don't exist: {unknown}. See data/processed/tags.csv.")
    for name, m in [("DEFAULT_TAG_MULTIPLIER", settings.default_tag_multiplier), *settings.tag_multipliers.items()]:
        if not (isinstance(m, (int, float)) and math.isfinite(m) and m >= 0):
            raise ConfigError(f"Tag multiplier for {name!r} must be a number >= 0, got {m!r}")


def rarity_weights(scores: np.ndarray, threshold: float) -> np.ndarray:
    """w_t = log(N / n_t). A tag that applies to no book (n_t = 0) gets 0:
    it never 'really applies', so it can't be evidence of a match."""
    n_books = scores.shape[0]
    n_t = (scores >= threshold).sum(axis=0)
    w = np.zeros(scores.shape[1], dtype=np.float64)
    has = n_t > 0
    w[has] = np.log(n_books / n_t[has])
    return w


class Recommender:
    def __init__(self, catalog: Catalog, settings: EngineSettings):
        _validate(settings, catalog)
        self.catalog = catalog
        self.settings = settings
        self.rarity = rarity_weights(catalog.scores, settings.tag_threshold)
        multipliers = np.full(len(catalog.tags), float(settings.default_tag_multiplier))
        name_to_col = {n: j for j, n in enumerate(catalog.tags["name"])}
        for name, m in settings.tag_multipliers.items():
            multipliers[name_to_col[name]] = float(m)
        self.multipliers = multipliers
        self.tag_weight = self.rarity * multipliers
        self.weighted = catalog.scores * self.tag_weight          # candidate vectors
        self.norms = np.linalg.norm(self.weighted, axis=1)

    # -- public API -------------------------------------------------------

    def recommend(self, liked: list[str]) -> RecommendationResult:
        """Recommend from titles (or "id:123") of books the user liked."""
        if not liked:
            raise CuratorError("Give at least one book you liked.")
        rows = [find_book(self.catalog, q) for q in liked]
        return self.recommend_rows([(r, self.settings.like_weight) for r in rows])

    def recommend_rows(self, inputs: list[tuple[int, float]]) -> RecommendationResult:
        """Core algorithm on (catalog row, weight) pairs. Positive weight =
        liked, negative = disliked. The hook for dislikes lives here."""
        if not inputs:
            raise CuratorError("Give at least one book.")
        seen = {}
        for row, _ in inputs:
            if row in seen:
                info = self.catalog.book_info(row)
                raise CuratorError(f'"{info["title"]}" (id:{info["id"]}) was given more than once.')
            seen[row] = True
        if not any(w > 0 for _, w in inputs):
            raise CuratorError("At least one input book must be a liked book.")

        rows = np.array([r for r, _ in inputs])
        weights = np.array([w for _, w in inputs], dtype=np.float64)
        scores = self.catalog.scores

        profile = (weights[:, None] * scores[rows]).mean(axis=0)            # step 3
        weighted_profile = profile * self.tag_weight                        # step 4
        p_norm = float(np.linalg.norm(weighted_profile))
        if p_norm == 0.0:
            raise CuratorError("Your taste profile is empty (all its tags are muted or weightless). Check TAG_MULTIPLIERS.")

        dots = self.weighted @ weighted_profile                             # step 5
        denom = self.norms * p_norm
        sims = np.zeros_like(dots)
        ok = denom > 0
        sims[ok] = dots[ok] / denom[ok]

        # Step 6: exclusions. Candidate order: highest similarity, then lowest id.
        ids = self.catalog.books["id"].to_numpy()
        order = np.lexsort((ids, -np.round(sims, _RANK_DECIMALS)))
        excluded = set(int(r) for r in rows)
        input_authors = set()
        if self.settings.exclude_input_authors:
            for r in rows:
                input_authors |= self.catalog.author_keys(int(r))
        liked_rows = rows[weights > 0]
        liked_has_tag = (scores[liked_rows] >= self.settings.tag_threshold).any(axis=0)

        author_counts: dict[str, int] = {}
        cap = self.settings.max_per_author
        recs: list[Recommendation] = []
        for r in order:
            r = int(r)
            if len(recs) >= self.settings.top_n:
                break
            if r in excluded or not ok[r]:
                continue
            authors = self.catalog.author_keys(r)
            if authors & input_authors:
                continue
            if cap is not None and any(author_counts.get(a, 0) >= cap for a in authors):
                continue
            for a in authors:
                author_counts[a] = author_counts.get(a, 0) + 1
            recs.append(
                Recommendation(
                    rank=len(recs) + 1,
                    book=self.catalog.book_info(r),
                    similarity=float(sims[r]),
                    shared_tags=self._explain(r, weighted_profile, profile, denom[r], liked_has_tag),
                )
            )

        resolved = [dict(self.catalog.book_info(int(r)), weight=float(w)) for r, w in inputs]
        return RecommendationResult(inputs=resolved, recommendations=recs)

    # -- helpers ----------------------------------------------------------

    def _explain(self, row, weighted_profile, profile, denom, liked_has_tag) -> list[SharedTag]:
        """Top shared tags by contribution to the cosine similarity. A tag is
        'shared' when the candidate AND at least one liked input book score it
        >= TAG_THRESHOLD. Contributions add up to the similarity score."""
        contrib = self.weighted[row] * weighted_profile / denom
        cand = self.catalog.scores[row]
        shared = np.flatnonzero(liked_has_tag & (cand >= self.settings.tag_threshold) & (contrib > 0))
        shared = sorted(shared, key=lambda j: (-round(float(contrib[j]), _RANK_DECIMALS), int(j)))
        names = self.catalog.tags["name"]
        return [
            SharedTag(
                name=names.iat[j],
                contribution=float(contrib[j]),
                candidate_score=float(cand[j]),
                profile_score=float(profile[j]),
            )
            for j in shared[: self.settings.explain_tags]
        ]
