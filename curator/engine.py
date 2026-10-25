"""
The recommendation algorithm (content-based filtering). No I/O, no printing:
a command line or a web server can both call this the same way.

Default: "common" mode (MULTI_BOOK_MODE = "common"). Books and movies in,
books, movies or a mixed list out:
  1. Look up the input books and movies (see lookup.py).
  2. Tag space        if every input and the wanted output are books, all book
                      tags; same for movies; otherwise only the tags both
                      datasets share. In that shared space movie scores are
                      calibrated to the book scale (CALIBRATE_MOVIES): each is
                      replaced by the book score at the same rank, because the
                      movie scores run about twice as high as the book scores.
  3. Excess           excess_t = max(0, score_t - median_t), the median taken
                      over the item's own catalog. Removes the weak background
                      score every item has.
  4. Overlap          a tag is common when at least COMMON_SHARE of the inputs
                      (default: all of them) really have it (score >=
                      TAG_THRESHOLD); common_t = the excess those inputs all
                      reach (with every input: the smallest).
  5. Uniqueness       strength_t = common_t x rarity_t x multiplier_t, where
                      rarity_t = log(N / max(n_t, 1)) over the wanted output
                      pool. The COMMON_TAGS strongest tags form the "common
                      core" (ignoring tags under COMMON_MIN_STRENGTH x the
                      strongest); nothing left = nothing in common = error.
  6. Match            match = sum(strength_t x candidate excess_t) / sum(strength_t)
                      over the core only: how strongly a candidate has the
                      traits the inputs share.
  7. Exclusions       inputs are never recommended; at most MAX_PER_AUTHOR
                      results per author (books) / director (movies).
  8. Output           top N; a mixed list holds at least MIXED_MIN_PER_MEDIUM
                      of each medium. Each result lists the core tags that
                      contributed most.

"per_book" and "blend" modes (books only) use cosine similarity instead:
  rarity w_t = log(N / max(n_t, 1)); profile_t = mean over inputs of
  (input weight x score), one profile per liked book ("per_book") or one for
  all ("blend"); similarity = cosine(profile x w x multiplier, book x w x
  multiplier); per_book profiles take turns picking so every liked book is
  represented.

Ties are broken by medium then id, so the same input always gives the same output.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .data import Catalog, Library
from .errors import ConfigError, CuratorError
from .lookup import find_item

# Scores are rounded to this many decimals before ranking, so tiny
# floating-point noise can never reorder results; exact ties go to the lower id.
_RANK_DECIMALS = 10

MODES = ("common", "per_book", "blend")
# What the user can ask for -> the media that go in the result list.
WANT_MEDIA = {"books": ("book",), "movies": ("movie",), "mixed": ("book", "movie")}
_MEDIUM_ORDER = {"book": 0, "movie": 1}


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
    multi_book_mode: str = "common"
    common_tags: int = 10
    common_share: float = 1.0
    common_min_strength: float = 0.05
    mixed_min_per_medium: int = 1
    calibrate_movies: bool = True

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
            common_tags=cfg.COMMON_TAGS,
            common_share=cfg.COMMON_SHARE,
            common_min_strength=cfg.COMMON_MIN_STRENGTH,
            mixed_min_per_medium=cfg.MIXED_MIN_PER_MEDIUM,
            calibrate_movies=cfg.CALIBRATE_MOVIES,
        )


@dataclass(frozen=True)
class SharedTag:
    name: str
    contribution: float     # this tag's share of the score (they add up to it)
    candidate_score: float  # the recommended item's score for the tag (as compared)
    profile_score: float    # common mode: the inputs' shared excess; else your taste profile's score


@dataclass(frozen=True)
class CommonTag:
    """One tag of the common core: something all (or COMMON_SHARE of) the inputs have."""
    name: str
    strength: float         # common excess x rarity x multiplier (what ranks the core)
    common: float           # the excess over typical that the inputs share
    input_scores: tuple     # each input's score for the tag (as compared), in input order


@dataclass(frozen=True)
class Recommendation:
    rank: int
    item: dict              # id, title, creator, year, medium
    score: float            # common mode: core match (0..1); other modes: cosine similarity
    shared_tags: list[SharedTag]
    because_of: list[dict]  # the liked item(s) this pick was matched to


@dataclass(frozen=True)
class RecommendationResult:
    inputs: list[dict]      # the resolved inputs (id, title, creator, year, medium, weight)
    want: str               # "books", "movies" or "mixed"
    mode: str               # MULTI_BOOK_MODE used
    shared_space: bool      # True when books and movies were compared on their shared tags only
    common_tags: list[CommonTag]  # the common core (common mode only; empty otherwise)
    recommendations: list[Recommendation]


@dataclass(frozen=True)
class _TagSpace:
    names: list            # tag names, in space order
    cols: dict             # medium -> column of each space tag in that medium's matrix


def _validate(settings: EngineSettings, library: Library) -> None:
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
    if settings.multi_book_mode not in MODES:
        raise ConfigError(f"MULTI_BOOK_MODE must be one of {', '.join(map(repr, MODES))}, got {settings.multi_book_mode!r}")
    if not (isinstance(settings.common_tags, int) and settings.common_tags >= 1):
        raise ConfigError(f"COMMON_TAGS must be a whole number >= 1, got {settings.common_tags}")
    if not (isinstance(settings.common_share, (int, float)) and 0.0 < settings.common_share <= 1.0):
        raise ConfigError(f"COMMON_SHARE must be > 0 and <= 1, got {settings.common_share}")
    if not (isinstance(settings.common_min_strength, (int, float)) and 0.0 <= settings.common_min_strength <= 1.0):
        raise ConfigError(f"COMMON_MIN_STRENGTH must be between 0 and 1, got {settings.common_min_strength}")
    if not (isinstance(settings.mixed_min_per_medium, int) and settings.mixed_min_per_medium >= 0):
        raise ConfigError(f"MIXED_MIN_PER_MEDIUM must be a whole number >= 0, got {settings.mixed_min_per_medium}")
    if settings.like_weight <= 0:
        raise ConfigError(f"LIKE_WEIGHT must be positive, got {settings.like_weight}")
    known = set().union(*(set(c.tags["name"]) for c in library.catalogs.values()))
    unknown = sorted(set(settings.tag_multipliers) - known)
    if unknown:
        raise ConfigError(f"TAG_MULTIPLIERS names tags that don't exist: {unknown}. See data/processed/*/tags.csv.")
    for name, m in [("DEFAULT_TAG_MULTIPLIER", settings.default_tag_multiplier), *settings.tag_multipliers.items()]:
        if not (isinstance(m, (int, float)) and math.isfinite(m) and m >= 0):
            raise ConfigError(f"Tag multiplier for {name!r} must be a number >= 0, got {m!r}")


def rarity_weights(scores: np.ndarray, threshold: float) -> np.ndarray:
    """w_t = log(N / max(n_t, 1)).

    n_t = 0 happens for real content tags the scoring model rates low across the
    board (in the real book data at 0.5: "cooking", "chess", "haunted house",
    whose top books are exactly the right ones but score ~0.4-0.49). They are as
    rare as a tag can be, so they get the maximum weight log(N), never infinity."""
    n_items = scores.shape[0]
    n_t = (scores >= threshold).sum(axis=0)
    return np.log(n_items / np.maximum(n_t, 1)).astype(np.float64)


def excess_scores(scores: np.ndarray, medians: np.ndarray) -> np.ndarray:
    """How far each score is above the tag's typical (median) score; never below 0."""
    return np.maximum(scores - medians, 0.0)


def common_core(excess: np.ndarray, applies: np.ndarray, rarity: np.ndarray, multipliers: np.ndarray,
                share: float, size: int, min_strength: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """What the inputs have in common.

    excess, applies: (n_inputs, n_tags); applies = the input really has the
    tag (score >= TAG_THRESHOLD). A tag counts only for the inputs that have
    it; `common` is the excess that at least ceil(share x n_inputs) of them
    reach (share 1.0 -> every input must have it, and common is the smallest
    excess). strength = common x rarity x multiplier.
    Returns (core columns strongest first, common, strength); the core holds at
    most `size` tags, all with strength > 0 and at least `min_strength` x the
    strongest tag's strength (ties: lower column first)."""
    n = excess.shape[0]
    k = min(n, max(1, math.ceil(share * n - 1e-9)))
    counted = np.where(applies, excess, 0.0)
    common = -np.sort(-counted, axis=0)[k - 1]
    strength = common * rarity * multipliers
    rounded = np.round(strength, _RANK_DECIMALS)
    order = np.lexsort((np.arange(len(strength)), -rounded))
    floor = min_strength * float(rounded[order[0]]) if len(order) else 0.0
    core = np.array([j for j in order[:size] if rounded[j] > 0 and rounded[j] >= floor], dtype=np.int64)
    return core, common, strength


def quantile_map(x: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Calibrate each column of x to the distribution of the same column of ref:
    every value is replaced by the ref value at the same rank (percentile).
    Ties get the same (average) rank, so the mapping is order-preserving and
    deterministic."""
    out = np.empty_like(x, dtype=np.float64)
    n = x.shape[0]
    for j in range(x.shape[1]):
        col = x[:, j]
        srt = np.sort(col)
        if n == 1:
            pct = np.array([0.5])
        else:
            rank = (np.searchsorted(srt, col, "left") + np.searchsorted(srt, col, "right") - 1) / 2.0
            pct = rank / (n - 1)
        ref_sorted = np.sort(ref[:, j])
        # Same as np.quantile(ref[:, j], pct) (linear interpolation), but fast.
        out[:, j] = np.interp(pct * (len(ref_sorted) - 1), np.arange(len(ref_sorted)), ref_sorted)
    return out


def core_match(candidate_excess: np.ndarray, core_strength: np.ndarray) -> np.ndarray:
    """Weighted average of each candidate's excess on the core tags.
    candidate_excess: (n_candidates, n_core); core_strength: (n_core,), all > 0."""
    return candidate_excess @ core_strength / core_strength.sum()


class Recommender:
    def __init__(self, library: Library, settings: EngineSettings):
        if not library.catalogs:
            raise CuratorError("No catalogs loaded. Run the build first:  python build.py")
        _validate(settings, library)
        self.library = library
        self.catalogs = library.catalogs
        self.settings = settings
        self.medians = {m: np.median(c.scores, axis=0) for m, c in self.catalogs.items()}

        self.spaces: dict[str, _TagSpace] = {
            m: _TagSpace(list(c.tags["name"]), {m: np.arange(len(c.tags))}) for m, c in self.catalogs.items()
        }
        if library.shared_tags is not None and {"book", "movie"} <= set(self.catalogs):
            cols = {}
            for m in ("book", "movie"):
                col_of = {int(i): j for j, i in enumerate(self.catalogs[m].tags["id"])}
                cols[m] = np.array([col_of[int(i)] for i in library.shared_tags[f"{m}_tag_id"]], dtype=np.int64)
            self.spaces["shared"] = _TagSpace(list(library.shared_tags["name"]), cols)

        # The scores each space compares, per medium: (scores, medians).
        self.space_scores: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {
            (m, m): (c.scores, self.medians[m]) for m, c in self.catalogs.items()
        }
        if "shared" in self.spaces:
            cols = self.spaces["shared"].cols
            book_s = self.catalogs["book"].scores[:, cols["book"]]
            movie_s = self.catalogs["movie"].scores[:, cols["movie"]]
            if settings.calibrate_movies:
                movie_s = quantile_map(movie_s, book_s)
            for m, sc in (("book", book_s), ("movie", movie_s)):
                self.space_scores[("shared", m)] = (sc, np.median(sc, axis=0))

        # Book-only cosine modes ("per_book", "blend").
        if "book" in self.catalogs:
            book = self.catalogs["book"]
            self.rarity = rarity_weights(book.scores, settings.tag_threshold)
            self.multipliers = self._multipliers(list(book.tags["name"]))
            self.tag_weight = self.rarity * self.multipliers
            self.weighted = book.scores * self.tag_weight          # candidate vectors
            self.norms = np.linalg.norm(self.weighted, axis=1)

    # -- public API -------------------------------------------------------

    def recommend(self, books=(), movies=(), want: str = "books") -> RecommendationResult:
        """Recommend from titles (or "id:123") of books and movies the user liked."""
        queries = [("book", q) for q in books] + [("movie", q) for q in movies]
        if not queries:
            raise CuratorError("Give at least one book or movie you liked.")
        rows = [(m, find_item(self._catalog(m), q), self.settings.like_weight) for m, q in queries]
        return self.recommend_rows(rows, want)

    def recommend_rows(self, inputs: list[tuple[str, int, float]], want: str = "books") -> RecommendationResult:
        """Core algorithm on (medium, catalog row, weight) triples. Positive
        weight = liked, negative = disliked (dislikes: per_book/blend only)."""
        if want not in WANT_MEDIA:
            raise CuratorError(f"Unknown output {want!r}: choose one of {', '.join(WANT_MEDIA)}.")
        if not inputs:
            raise CuratorError("Give at least one book or movie.")
        seen = set()
        for medium, row, _ in inputs:
            cat = self._catalog(medium)
            if (medium, row) in seen:
                info = cat.item_info(row)
                raise CuratorError(f'"{info["title"]}" ({medium} id:{info["id"]}) was given more than once.')
            seen.add((medium, row))
        for medium in WANT_MEDIA[want]:
            self._catalog(medium)
        if not any(w > 0 for _, _, w in inputs):
            raise CuratorError("At least one input must be something you liked.")

        mode = self.settings.multi_book_mode
        if mode == "common":
            return self._recommend_common(inputs, want)
        if want != "books" or any(m != "book" for m, _, _ in inputs):
            raise CuratorError(
                f'MULTI_BOOK_MODE = "{mode}" only works with books in and books out. '
                'Set MULTI_BOOK_MODE = "common" in config.py to mix books and movies.'
            )
        return self._recommend_similarity([(r, w) for _, r, w in inputs])

    # -- common mode ------------------------------------------------------

    def _recommend_common(self, inputs: list[tuple[str, int, float]], want: str) -> RecommendationResult:
        s = self.settings
        if any(w < 0 for _, _, w in inputs):
            raise CuratorError('Dislikes are not supported in "common" mode yet.')
        out_media = WANT_MEDIA[want]
        if len(out_media) > 1 and s.mixed_min_per_medium * len(out_media) > s.top_n:
            raise ConfigError(
                f"A mixed list of {s.top_n} can't hold {s.mixed_min_per_medium} of each medium "
                "(MIXED_MIN_PER_MEDIUM). Ask for more results or lower the setting."
            )
        media = {m for m, _, _ in inputs} | set(out_media)
        key = next(iter(media)) if len(media) == 1 else "shared"
        if key not in self.spaces:
            raise CuratorError("Comparing books with movies needs both catalogs and their shared tags. Run: python build.py")
        space = self.spaces[key]

        # Steps 3-5: what the inputs have in common.
        in_scores = np.vstack([self.space_scores[(key, m)][0][r] for m, r, _ in inputs])
        in_medians = np.vstack([self.space_scores[(key, m)][1] for m, _, _ in inputs])
        pool = np.vstack([self.space_scores[(key, m)][0] for m in out_media])
        rarity = rarity_weights(pool, s.tag_threshold)
        core, common, strength = common_core(
            excess_scores(in_scores, in_medians), in_scores >= s.tag_threshold,
            rarity, self._multipliers(space.names), s.common_share, s.common_tags, s.common_min_strength,
        )
        if len(core) == 0:
            who = "all of them" if s.common_share >= 1 else f"at least {s.common_share:.0%} of them"
            raise CuratorError(
                f"These have nothing distinctive in common: no tag applies strongly (score >= {s.tag_threshold}) "
                f"to {who}, apart from generic ones like \"fiction\". Try other inputs, or lower COMMON_SHARE "
                "in config.py (with 3 or more inputs)."
            )
        core_strength = strength[core]
        common_tags = [
            CommonTag(
                name=space.names[j],
                strength=float(strength[j]),
                common=float(common[j]),
                input_scores=tuple(float(x) for x in in_scores[:, j]),
            )
            for j in core
        ]

        # Step 6: score every candidate on the core only.
        cand_scores, cand_excess, cand_match = {}, {}, {}
        for m in out_media:
            sc, med = self.space_scores[(key, m)]
            cand_scores[m] = sc[:, core]
            cand_excess[m] = excess_scores(cand_scores[m], med[core])
            cand_match[m] = core_match(cand_excess[m], core_strength)

        # Global ranking: best match first; ties by medium, then id.
        all_m = np.concatenate([np.full(self.catalogs[m].n_items, _MEDIUM_ORDER[m]) for m in out_media])
        all_r = np.concatenate([np.arange(self.catalogs[m].n_items) for m in out_media])
        all_id = np.concatenate([self.catalogs[m].items["id"].to_numpy() for m in out_media])
        all_s = np.concatenate([cand_match[m] for m in out_media])
        order = np.lexsort((all_id, all_m, -np.round(all_s, _RANK_DECIMALS)))
        medium_of = {v: k for k, v in _MEDIUM_ORDER.items()}
        ranked = [(medium_of[int(all_m[i])], int(all_r[i])) for i in order]

        # Step 7: exclusions.
        excluded = {(m, r) for m, r, _ in inputs}
        input_creators: set[tuple[str, str]] = set()
        if s.exclude_input_authors:
            for m, r, _ in inputs:
                input_creators |= {(m, a) for a in self.catalogs[m].creator_keys(r)}
        creator_counts: dict[tuple[str, str], int] = {}
        chosen: list[tuple[int, str, int]] = []   # (position in ranked, medium, row)
        taken: set[tuple[str, int]] = set()

        def try_take(pos: int, m: str, r: int) -> bool:
            if (m, r) in excluded or (m, r) in taken:
                return False
            creators = {(m, a) for a in self.catalogs[m].creator_keys(r)}
            if creators & input_creators:
                return False
            if s.max_per_author is not None and any(creator_counts.get(c, 0) >= s.max_per_author for c in creators):
                return False
            taken.add((m, r))
            for c in creators:
                creator_counts[c] = creator_counts.get(c, 0) + 1
            chosen.append((pos, m, r))
            return True

        # Step 8: a mixed list first reserves its minimum of each medium, then
        # fills the remaining slots with the best matches of either medium.
        if len(out_media) > 1:
            for medium in out_media:
                got = 0
                for pos, (m, r) in enumerate(ranked):
                    if got >= s.mixed_min_per_medium:
                        break
                    if m == medium and try_take(pos, m, r):
                        got += 1
        for pos, (m, r) in enumerate(ranked):
            if len(chosen) >= s.top_n:
                break
            try_take(pos, m, r)
        chosen.sort()

        because = sorted(
            (self.catalogs[m].item_info(r) for m, r, w in inputs if w > 0),
            key=lambda b: (_MEDIUM_ORDER[b["medium"]], b["id"]),
        )
        recs = []
        for rank, (_, m, r) in enumerate(chosen, start=1):
            contrib = core_strength * cand_excess[m][r] / core_strength.sum()
            shown = sorted(
                (i for i in range(len(core)) if contrib[i] > 0),
                key=lambda i: (-round(float(contrib[i]), _RANK_DECIMALS), i),
            )[: s.explain_tags]
            recs.append(
                Recommendation(
                    rank=rank,
                    item=self.catalogs[m].item_info(r),
                    score=float(cand_match[m][r]),
                    shared_tags=[
                        SharedTag(
                            name=space.names[core[i]],
                            contribution=float(contrib[i]),
                            candidate_score=float(cand_scores[m][r, i]),
                            profile_score=float(common[core[i]]),
                        )
                        for i in shown
                    ],
                    because_of=because,
                )
            )
        return RecommendationResult(
            inputs=self._resolved(inputs), want=want, mode="common", shared_space=key == "shared",
            common_tags=common_tags, recommendations=recs,
        )

    # -- per_book / blend modes (books only) ------------------------------

    def _recommend_similarity(self, inputs: list[tuple[int, float]]) -> RecommendationResult:
        book = self.catalogs["book"]
        liked = [(r, w) for r, w in inputs if w > 0]
        disliked = [(r, w) for r, w in inputs if w < 0]

        # One taste profile for all inputs ("blend"), or one per liked book
        # ("per_book"), each including every disliked book.
        if self.settings.multi_book_mode == "blend":
            groups = [self._score_group(inputs)]
        else:
            groups = [self._score_group([like] + disliked) for like in liked]

        # Exclusions shared by every group.
        excluded = {r for r, _ in inputs}
        input_authors: set[str] = set()
        if self.settings.exclude_input_authors:
            for r, _ in inputs:
                input_authors |= book.creator_keys(r)
        author_counts: dict[str, int] = {}
        cap = self.settings.max_per_author
        picked: set[int] = set()

        def eligible(r: int, g: "_Group") -> bool:
            if r in excluded or r in picked or not g.ok[r]:
                return False
            authors = book.creator_keys(r)
            if authors & input_authors:
                return False
            return cap is None or all(author_counts.get(a, 0) < cap for a in authors)

        def next_candidate(g: "_Group") -> int | None:
            # Skipped candidates never become eligible again (picks and author
            # counts only grow), so the pointer only moves forward.
            while g.pos < len(g.order) and not eligible(int(g.order[g.pos]), g):
                g.pos += 1
            return int(g.order[g.pos]) if g.pos < len(g.order) else None

        # Take turns, one pick per group per round, so every liked book is
        # represented. Within a round, the group whose best remaining match is
        # strongest goes first (ties: lowest input book id), which makes the
        # result independent of the order the books were typed in.
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
                for a in book.creator_keys(r):
                    author_counts[a] = author_counts.get(a, 0) + 1
                recs.append(
                    Recommendation(
                        rank=len(recs) + 1,
                        item=book.item_info(r),
                        score=float(g.sims[r]),
                        shared_tags=self._explain(r, g),
                        because_of=[book.item_info(x) for x in g.liked_rows],
                    )
                )

        return RecommendationResult(
            inputs=self._resolved([("book", r, w) for r, w in inputs]), want="books",
            mode=self.settings.multi_book_mode, shared_space=False, common_tags=[], recommendations=recs,
        )

    # -- helpers ----------------------------------------------------------

    def _catalog(self, medium: str) -> Catalog:
        if medium not in self.catalogs:
            raise CuratorError(f"No {medium} data loaded. Run the build first:  python build.py")
        return self.catalogs[medium]

    def _multipliers(self, names: list) -> np.ndarray:
        s = self.settings
        return np.array([float(s.tag_multipliers.get(n, s.default_tag_multiplier)) for n in names])

    def _resolved(self, inputs) -> list[dict]:
        return [dict(self.catalogs[m].item_info(int(r)), weight=float(w)) for m, r, w in inputs]

    def _score_group(self, group: list[tuple[int, float]]) -> "_Group":
        """Profile, weighting and cosine similarity for one group of (row, weight) books."""
        book = self.catalogs["book"]
        rows = np.array([r for r, _ in group])
        weights = np.array([w for _, w in group], dtype=np.float64)
        scores = book.scores

        profile = (weights[:, None] * scores[rows]).mean(axis=0)
        weighted_profile = profile * self.tag_weight
        p_norm = float(np.linalg.norm(weighted_profile))
        if p_norm == 0.0:
            raise CuratorError("Your taste profile is empty (all its tags are muted or weightless). Check TAG_MULTIPLIERS.")

        dots = self.weighted @ weighted_profile
        denom = self.norms * p_norm
        sims = np.zeros_like(dots)
        ok = denom > 0
        sims[ok] = dots[ok] / denom[ok]

        ids = book.items["id"].to_numpy()
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
        book = self.catalogs["book"]
        contrib = self.weighted[row] * g.weighted_profile / g.denom[row]
        cand = book.scores[row]
        shared = np.flatnonzero(g.liked_has_tag & (cand >= self.settings.tag_threshold) & (contrib > 0))
        shared = sorted(shared, key=lambda j: (-round(float(contrib[j]), _RANK_DECIMALS), int(j)))
        names = book.tags["name"]
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
