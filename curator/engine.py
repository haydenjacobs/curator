"""
The recommendation algorithm (content-based filtering). No I/O, no printing:
a command line or a web server can both call this the same way.

Steps:
  1. Look up the input books (see lookup.py).
  2. Tag rarity weight  w_t = log(N / max(n_t, 1)), where n_t = books whose
     score for tag t is >= TAG_THRESHOLD. A tag that clears the threshold on no
     book is treated as maximally rare (as if n_t = 1).
  3. Taste profile      profile_t = mean over inputs of (input weight x score).
                        Likes are +1; negative weights (dislikes) work too.
                        In "per_book" mode (default) each liked book gets its
                        own profile (plus any dislikes); in "blend" mode all
                        inputs share one profile.
  4. Tag weight         weight_t = w_t x multiplier_t   (multipliers default 1.0)
  5. Similarity         cosine( profile x weight,  candidate scores x weight )
  6. Exclusions         input books are never recommended; at most
                        MAX_PER_AUTHOR results per author.
  7. Output             top N, each with the shared tags that contributed most.
                        Profiles take turns picking, so every liked book is
                        represented (N grows to the number of liked books).
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
    multi_book_mode: str = "per_book"

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
            multi_book_mode=cfg.MULTI_BOOK_MODE,
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
    because_of: list[dict]  # the liked book(s) this pick was matched to


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
    if settings.multi_book_mode not in ("per_book", "blend"):
        raise ConfigError(f'MULTI_BOOK_MODE must be "per_book" or "blend", got {settings.multi_book_mode!r}')
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
    """w_t = log(N / max(n_t, 1)).

    n_t = 0 happens for real content tags the scoring model rates low across the
    board (in the real data at 0.5: "cooking", "chess", "haunted house", whose
    top books are exactly the right ones but score ~0.4-0.49). They are as rare
    as a tag can be, so they get the maximum weight log(N), never infinity."""
    n_books = scores.shape[0]
    n_t = (scores >= threshold).sum(axis=0)
    return np.log(n_books / np.maximum(n_t, 1)).astype(np.float64)


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
        seen = set()
        for row, _ in inputs:
            if row in seen:
                info = self.catalog.book_info(row)
                raise CuratorError(f'"{info["title"]}" (id:{info["id"]}) was given more than once.')
            seen.add(row)
        liked = [(r, w) for r, w in inputs if w > 0]
        disliked = [(r, w) for r, w in inputs if w < 0]
        if not liked:
            raise CuratorError("At least one input book must be a liked book.")

        # One taste profile for all inputs ("blend"), or one per liked book
        # ("per_book"), each including every disliked book.
        if self.settings.multi_book_mode == "blend":
            groups = [self._score_group(inputs)]
        else:
            groups = [self._score_group([like] + disliked) for like in liked]

        # Step 6: exclusions shared by every group.
        excluded = {r for r, _ in inputs}
        input_authors: set[str] = set()
        if self.settings.exclude_input_authors:
            for r, _ in inputs:
                input_authors |= self.catalog.author_keys(r)
        author_counts: dict[str, int] = {}
        cap = self.settings.max_per_author
        picked: set[int] = set()

        def eligible(r: int, g: "_Group") -> bool:
            if r in excluded or r in picked or not g.ok[r]:
                return False
            authors = self.catalog.author_keys(r)
            if authors & input_authors:
                return False
            return cap is None or all(author_counts.get(a, 0) < cap for a in authors)

        def next_candidate(g: "_Group") -> int | None:
            # Skipped candidates never become eligible again (picks and author
            # counts only grow), so the pointer only moves forward.
            while g.pos < len(g.order) and not eligible(int(g.order[g.pos]), g):
                g.pos += 1
            return int(g.order[g.pos]) if g.pos < len(g.order) else None

        # Step 7: take turns, one pick per group per round, so every liked
        # book is represented. Within a round, the group whose best remaining
        # match is strongest goes first (ties: lowest input book id), which
        # makes the result independent of the order the books were typed in.
        n_results = max(self.settings.top_n, len(groups))
        recs: list[Recommendation] = []
        while len(recs) < n_results:
            ready = [(g, r) for g in groups if (r := next_candidate(g)) is not None]
            if not ready:
                break
            ready.sort(key=lambda gr: (-round(float(gr[0].sims[gr[1]]), _RANK_DECIMALS), gr[0].key))
            for g, _ in ready:
                if len(recs) >= n_results:
                    break
                r = next_candidate(g)  # an earlier pick this round may have taken it
                if r is None:
                    continue
                picked.add(r)
                for a in self.catalog.author_keys(r):
                    author_counts[a] = author_counts.get(a, 0) + 1
                recs.append(
                    Recommendation(
                        rank=len(recs) + 1,
                        book=self.catalog.book_info(r),
                        similarity=float(g.sims[r]),
                        shared_tags=self._explain(r, g),
                        because_of=[self.catalog.book_info(x) for x in g.liked_rows],
                    )
                )

        resolved = [dict(self.catalog.book_info(int(r)), weight=float(w)) for r, w in inputs]
        return RecommendationResult(inputs=resolved, recommendations=recs)

    # -- helpers ----------------------------------------------------------

    def _score_group(self, group: list[tuple[int, float]]) -> "_Group":
        """Steps 3-5 for one group of (row, weight) inputs."""
        rows = np.array([r for r, _ in group])
        weights = np.array([w for _, w in group], dtype=np.float64)
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

        ids = self.catalog.books["id"].to_numpy()
        liked_rows = sorted((int(r) for r, w in group if w > 0), key=lambda r: int(ids[r]))
        return _Group(
            key=int(ids[liked_rows[0]]),
            liked_rows=liked_rows,
            profile=profile,
            weighted_profile=weighted_profile,
            denom=denom,
            sims=sims,
            ok=ok,
            order=np.lexsort((ids, -np.round(sims, _RANK_DECIMALS))),
            liked_has_tag=(scores[liked_rows] >= self.settings.tag_threshold).any(axis=0),
        )

    def _explain(self, row: int, g: "_Group") -> list[SharedTag]:
        """Top shared tags by contribution to the cosine similarity. A tag is
        'shared' when the candidate AND at least one liked book of the group
        score it >= TAG_THRESHOLD. Contributions add up to the similarity score."""
        contrib = self.weighted[row] * g.weighted_profile / g.denom[row]
        cand = self.catalog.scores[row]
        shared = np.flatnonzero(g.liked_has_tag & (cand >= self.settings.tag_threshold) & (contrib > 0))
        shared = sorted(shared, key=lambda j: (-round(float(contrib[j]), _RANK_DECIMALS), int(j)))
        names = self.catalog.tags["name"]
        return [
            SharedTag(
                name=names.iat[j],
                contribution=float(contrib[j]),
                candidate_score=float(cand[j]),
                profile_score=float(g.profile[j]),
            )
            for j in shared[: self.settings.explain_tags]
        ]


@dataclass
class _Group:
    """One taste profile and its ranking of every catalog book."""
    key: int                    # lowest liked-book id in the group (tie-breaker)
    liked_rows: list[int]
    profile: np.ndarray
    weighted_profile: np.ndarray
    denom: np.ndarray
    sims: np.ndarray
    ok: np.ndarray
    order: np.ndarray           # catalog rows, best match first
    liked_has_tag: np.ndarray
    pos: int = 0                # next position in `order` to consider
