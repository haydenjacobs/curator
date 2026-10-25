"""The algorithm: lookup, rarity weights, determinism, author cap, explanations."""

import numpy as np
import pytest

from conftest import TAGS, THRESHOLD
from curator.engine import rarity_weights
from curator.errors import AmbiguousBookError, BookNotFoundError, ConfigError, CuratorError
from curator.lookup import find_book


def ids(result):
    return [r.book["id"] for r in result.recommendations]


# --- lookup -----------------------------------------------------------------

def test_lookup_exact_title_case_insensitive(catalog):
    assert catalog.book_info(find_book(catalog, "neuromancer"))["id"] == 4
    assert catalog.book_info(find_book(catalog, "  NEUROMANCER ")) ["id"] == 4


def test_lookup_title_without_series_note(catalog):
    # "Dune (Dune Chronicles, #1)" is found by just "Dune", not confused with Dune Messiah.
    assert catalog.book_info(find_book(catalog, "Dune"))["id"] == 1
    assert catalog.book_info(find_book(catalog, "dune (dune chronicles #1)"))["id"] == 1


def test_lookup_by_id_and_numeric_title(catalog):
    assert catalog.book_info(find_book(catalog, "id:7"))["id"] == 7
    assert catalog.book_info(find_book(catalog, "1984"))["id"] == 9  # a title, not an id


def test_lookup_no_match_is_clear(catalog):
    with pytest.raises(BookNotFoundError, match='"Moby Dick" is not in the catalog'):
        find_book(catalog, "Moby Dick")


def test_lookup_no_exact_match_lists_suggestions_but_never_picks(catalog):
    with pytest.raises(BookNotFoundError) as exc:
        find_book(catalog, "Messiah")
    assert "id:2" in str(exc.value) and "Dune Messiah" in str(exc.value)


def test_lookup_multiple_matches_lists_author_and_year(catalog):
    with pytest.raises(AmbiguousBookError) as exc:
        find_book(catalog, "twins")
    msg = str(exc.value)
    assert "matches 2 books" in msg
    assert "id:7  Twins — Author One (1990)" in msg
    assert "id:8  Twins — Author Two (1995)" in msg
    assert [c["id"] for c in exc.value.candidates] == [7, 8]


def test_lookup_unknown_id(catalog):
    with pytest.raises(BookNotFoundError, match="No book with id 999"):
        find_book(catalog, "id:999")


# --- rarity -----------------------------------------------------------------

def test_broad_tags_weigh_less_than_niche_tags(catalog):
    w = rarity_weights(catalog.scores, THRESHOLD)
    col = {t: i for i, t in enumerate(TAGS)}
    # "fiction" applies to every book -> log(N/N) = 0; "surf" applies to one -> log(10).
    assert w[col["fiction"]] == pytest.approx(0.0)
    assert w[col["surf"]] == pytest.approx(np.log(10 / 1))
    assert w[col["fiction"]] < w[col["space"]] < w[col["surf"]]


def test_tag_on_no_book_gets_max_weight_not_infinity():
    scores = np.array([[0.9, 0.1], [0.8, 0.2], [0.1, 0.3]])
    w = rarity_weights(scores, 0.5)
    assert np.isfinite(w).all()
    assert w[1] == pytest.approx(np.log(3))  # treated as if on exactly 1 book
    assert w[1] >= w.max()


# --- determinism ------------------------------------------------------------

def test_same_input_gives_identical_output(make_engine):
    a = make_engine().recommend(["Dune", "Neuromancer"])
    b = make_engine().recommend(["Dune", "Neuromancer"])
    assert a == b


def test_input_order_does_not_change_results(make_engine):
    e = make_engine()
    a = e.recommend(["Dune", "Neuromancer"])
    b = e.recommend(["Neuromancer", "Dune"])
    assert a.recommendations == b.recommendations


# --- exclusions / author cap -----------------------------------------------

def test_input_books_never_recommended(make_engine):
    res = make_engine(top_n=9).recommend(["Dune", "Neuromancer"])
    assert 1 not in ids(res) and 4 not in ids(res)


def test_author_cap_default_one_per_author(make_engine):
    # Liking Dune: Dune Messiah and Children of Dune are both very similar,
    # but only one Frank Herbert book may appear.
    res = make_engine(top_n=5).recommend(["Dune"])
    herbert = [r for r in res.recommendations if r.book["author"] == "Frank Herbert"]
    assert len(herbert) == 1
    assert len(res.recommendations) == 5


def test_author_cap_can_be_raised_or_removed(make_engine):
    res2 = make_engine(top_n=5, max_per_author=2).recommend(["Dune"])
    assert sum(r.book["author"] == "Frank Herbert" for r in res2.recommendations) == 2
    res_none = make_engine(top_n=5, max_per_author=None).recommend(["Dune"])
    assert ids(res_none)[:2] == [2, 3]


def test_exclude_input_authors_option(make_engine):
    res = make_engine(top_n=5, exclude_input_authors=True).recommend(["Dune"])
    assert all(r.book["author"] != "Frank Herbert" for r in res.recommendations)


def test_books_without_author_are_not_capped_together(make_engine):
    res = make_engine(top_n=9).recommend(["Dune"])
    assert 10 in ids(res)  # the authorless book is still eligible


# --- profile, multipliers, explanations ------------------------------------

def test_explanations_are_shared_tags_and_sum_within_similarity(make_engine):
    res = make_engine().recommend(["Dune"])
    top = res.recommendations[0]
    names = [t.name for t in top.shared_tags]
    assert "desert" in names and "fiction" not in names  # fiction has 0 rarity weight
    assert sum(t.contribution for t in top.shared_tags) <= top.similarity + 1e-12
    for t in top.shared_tags:
        assert t.candidate_score >= THRESHOLD


def test_muting_a_tag_removes_its_influence(make_engine):
    normal = make_engine().recommend(["Beach Days"])
    muted = make_engine(tag_multipliers={"surf": 0.0}).recommend(["Beach Days"])
    for r in muted.recommendations:
        assert "surf" not in [t.name for t in r.shared_tags]
    assert normal != muted


def test_unknown_multiplier_tag_is_an_error(catalog, make_engine):
    with pytest.raises(ConfigError, match="don't exist"):
        make_engine(tag_multipliers={"romanse": 0.0})


def test_dislike_hook_works_without_structural_change(catalog, make_engine):
    e = make_engine(top_n=9)
    dune, beach = find_book(catalog, "Dune"), find_book(catalog, "Beach Days")
    liked_only = e.recommend_rows([(dune, 1.0)])
    with_dislike = e.recommend_rows([(dune, 1.0), (beach, -1.0)])
    assert beach not in [find_book(catalog, f"id:{i}") for i in ids(with_dislike)]
    assert liked_only != with_dislike


def test_same_book_twice_is_an_error(make_engine):
    with pytest.raises(CuratorError, match="more than once"):
        make_engine().recommend(["Dune", "id:1"])


def test_similarity_is_cosine(catalog, make_engine):
    e = make_engine(top_n=1, max_per_author=None)
    res = e.recommend(["Dune"])
    w = rarity_weights(catalog.scores, THRESHOLD)
    a, b = catalog.scores[0] * w, catalog.scores[1] * w
    assert res.recommendations[0].similarity == pytest.approx(a @ b / np.linalg.norm(a) / np.linalg.norm(b))


def test_author_cap_applies_to_co_authors(tmp_path):
    from conftest import BOOKS, settings_for, write_raw
    from curator.build import build
    from curator.data import load_catalog
    from curator.engine import EngineSettings, Recommender

    # A Dune-like book co-written by Frank Herbert, stored the way the real
    # data stores co-authors: one comma-separated string.
    extra = (11, "Dune Prequel", "Brian Herbert, Frank Herbert", 2000, [0.9, 0.9, 0.95, 0.1, 0.0, 0.1])
    write_raw(tmp_path / "raw", books=BOOKS + [extra])
    s = settings_for(tmp_path, n_books=len(BOOKS) + 1)
    build(s, log=lambda *_: None)
    cat = load_catalog(s.processed_dir)
    assert cat.author_keys(10) == {"brian herbert", "frank herbert"}
    res = Recommender(cat, EngineSettings(tag_threshold=THRESHOLD, top_n=6)).recommend(["Dune"])
    herbert = [r for r in res.recommendations if "Frank Herbert" in r.book["author"]]
    assert len(herbert) == 1
