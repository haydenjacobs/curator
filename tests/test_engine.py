"""Lookup, rarity weights, and the book-only cosine modes ("per_book", "blend")."""

import numpy as np
import pytest

from conftest import TAGS, THRESHOLD
from curator.engine import rarity_weights
from curator.errors import AmbiguousItemError, ConfigError, CuratorError, ItemNotFoundError
from curator.lookup import find_item


def ids(result):
    return [r.item["id"] for r in result.recommendations]


# --- lookup -----------------------------------------------------------------

def test_lookup_exact_title_case_insensitive(catalog):
    assert catalog.item_info(find_item(catalog, "neuromancer"))["id"] == 4
    assert catalog.item_info(find_item(catalog, "  NEUROMANCER ")) ["id"] == 4


def test_lookup_title_without_series_note(catalog):
    # "Dune (Dune Chronicles, #1)" is found by just "Dune", not confused with Dune Messiah.
    assert catalog.item_info(find_item(catalog, "Dune"))["id"] == 1
    assert catalog.item_info(find_item(catalog, "dune (dune chronicles #1)"))["id"] == 1


def test_lookup_by_id_and_numeric_title(catalog):
    assert catalog.item_info(find_item(catalog, "id:7"))["id"] == 7
    assert catalog.item_info(find_item(catalog, "1984"))["id"] == 9  # a title, not an id


def test_lookup_no_match_is_clear(catalog):
    with pytest.raises(ItemNotFoundError, match='"Moby Dick" is not in the book catalog'):
        find_item(catalog, "Moby Dick")


def test_lookup_no_exact_match_lists_suggestions_but_never_picks(catalog):
    with pytest.raises(ItemNotFoundError) as exc:
        find_item(catalog, "Messiah")
    assert "id:2" in str(exc.value) and "Dune Messiah" in str(exc.value)


def test_lookup_multiple_matches_lists_author_and_year(catalog):
    with pytest.raises(AmbiguousItemError) as exc:
        find_item(catalog, "twins")
    msg = str(exc.value)
    assert "matches 2 books" in msg
    assert "id:7  Twins — Author One (1990)" in msg
    assert "id:8  Twins — Author Two (1995)" in msg
    assert [c["id"] for c in exc.value.candidates] == [7, 8]


def test_lookup_unknown_id(catalog):
    with pytest.raises(ItemNotFoundError, match="No book with id 999"):
        find_item(catalog, "id:999")


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

def test_same_input_gives_identical_output(cosine_engine):
    a = cosine_engine().recommend(["Dune", "Neuromancer"])
    b = cosine_engine().recommend(["Dune", "Neuromancer"])
    assert a == b


def test_input_order_does_not_change_results(cosine_engine):
    e = cosine_engine()
    a = e.recommend(["Dune", "Neuromancer"])
    b = e.recommend(["Neuromancer", "Dune"])
    assert a.recommendations == b.recommendations


# --- exclusions / author cap -----------------------------------------------

def test_input_books_never_recommended(cosine_engine):
    res = cosine_engine(top_n=9).recommend(["Dune", "Neuromancer"])
    assert 1 not in ids(res) and 4 not in ids(res)


def test_author_cap_default_one_per_author(cosine_engine):
    # Liking Dune: Dune Messiah and Children of Dune are both very similar,
    # but only one Frank Herbert book may appear.
    res = cosine_engine(top_n=5).recommend(["Dune"])
    herbert = [r for r in res.recommendations if r.item["creator"] == "Frank Herbert"]
    assert len(herbert) == 1
    assert len(res.recommendations) == 5


def test_author_cap_can_be_raised_or_removed(cosine_engine):
    res2 = cosine_engine(top_n=5, max_per_author=2).recommend(["Dune"])
    assert sum(r.item["creator"] == "Frank Herbert" for r in res2.recommendations) == 2
    res_none = cosine_engine(top_n=5, max_per_author=None).recommend(["Dune"])
    assert ids(res_none)[:2] == [2, 3]


def test_exclude_input_authors_option(cosine_engine):
    res = cosine_engine(top_n=5, exclude_input_authors=True).recommend(["Dune"])
    assert all(r.item["creator"] != "Frank Herbert" for r in res.recommendations)


def test_books_without_author_are_not_capped_together(cosine_engine):
    res = cosine_engine(top_n=9).recommend(["Dune"])
    assert 10 in ids(res)  # the authorless book is still eligible


# --- profile, multipliers, explanations ------------------------------------

def test_explanations_are_shared_tags_and_sum_within_similarity(cosine_engine):
    res = cosine_engine().recommend(["Dune"])
    top = res.recommendations[0]
    names = [t.name for t in top.shared_tags]
    assert "desert" in names and "fiction" not in names  # fiction has 0 rarity weight
    assert sum(t.contribution for t in top.shared_tags) <= top.score + 1e-12
    for t in top.shared_tags:
        assert t.candidate_score >= THRESHOLD


def test_muting_a_tag_removes_its_influence(cosine_engine):
    normal = cosine_engine().recommend(["Beach Days"])
    muted = cosine_engine(tag_multipliers={"surf": 0.0}).recommend(["Beach Days"])
    for r in muted.recommendations:
        assert "surf" not in [t.name for t in r.shared_tags]
    assert normal != muted


def test_unknown_multiplier_tag_is_an_error(catalog, cosine_engine):
    with pytest.raises(ConfigError, match="don't exist"):
        cosine_engine(tag_multipliers={"romanse": 0.0})


def test_dislike_hook_works_without_structural_change(catalog, cosine_engine):
    e = cosine_engine(top_n=9)
    dune, beach = find_item(catalog, "Dune"), find_item(catalog, "Beach Days")
    liked_only = e.recommend_rows([("book", dune, 1.0)])
    with_dislike = e.recommend_rows([("book", dune, 1.0), ("book", beach, -1.0)])
    assert beach not in [find_item(catalog, f"id:{i}") for i in ids(with_dislike)]
    assert liked_only != with_dislike


def test_same_book_twice_is_an_error(cosine_engine):
    with pytest.raises(CuratorError, match="more than once"):
        cosine_engine().recommend(["Dune", "id:1"])


def test_similarity_is_cosine(catalog, cosine_engine):
    e = cosine_engine(top_n=1, max_per_author=None)
    res = e.recommend(["Dune"])
    w = rarity_weights(catalog.scores, THRESHOLD)
    a, b = catalog.scores[0] * w, catalog.scores[1] * w
    assert res.recommendations[0].score == pytest.approx(a @ b / np.linalg.norm(a) / np.linalg.norm(b))


def test_author_cap_applies_to_co_authors(tmp_path):
    from conftest import BOOKS, settings_for, write_raw
    from curator.build import build
    from curator.data import Library, load_catalog
    from curator.engine import EngineSettings, Recommender

    # A Dune-like book co-written by Frank Herbert, stored the way the real
    # data stores co-authors: one comma-separated string.
    extra = (11, "Dune Prequel", "Brian Herbert, Frank Herbert", 2000, [0.9, 0.9, 0.95, 0.1, 0.0, 0.1])
    write_raw(tmp_path / "raw", books=BOOKS + [extra])
    s = settings_for(tmp_path, n_books=len(BOOKS) + 1)
    build(s, log=lambda *_: None)
    cat = load_catalog(s.processed_dir)
    assert cat.creator_keys(10) == {"brian herbert", "frank herbert"}
    settings = EngineSettings(tag_threshold=THRESHOLD, top_n=6, multi_book_mode="per_book")
    res = Recommender(Library({"book": cat}, None), settings).recommend(["Dune"])
    herbert = [r for r in res.recommendations if "Frank Herbert" in r.item["creator"]]
    assert len(herbert) == 1


# --- several liked books: every one must be represented ---------------------

def _because(rec):
    return [b["id"] for b in rec.because_of]


def test_every_liked_book_is_represented(cosine_engine):
    # Dune (desert/space) and Beach Days (surf/romance) are very different.
    res = cosine_engine().recommend(["Dune", "Beach Days"])
    sources = {i for r in res.recommendations for i in _because(r)}
    assert sources == {1, 6}
    assert len(res.recommendations) == 3


def test_more_liked_books_than_slots_gives_one_each(cosine_engine):
    res = cosine_engine(top_n=3).recommend(["Dune", "Beach Days", "Neuromancer", "1984"])
    assert len(res.recommendations) == 4
    assert sorted(i for r in res.recommendations for i in _because(r)) == [1, 4, 6, 9]


def test_per_book_results_independent_of_input_order(cosine_engine):
    e = cosine_engine(top_n=5)
    a = e.recommend(["Dune", "Beach Days", "Neuromancer"])
    b = e.recommend(["Neuromancer", "Dune", "Beach Days"])
    assert a.recommendations == b.recommendations


def test_no_book_recommended_twice(cosine_engine):
    res = cosine_engine(top_n=9, max_per_author=None).recommend(["Dune", "Neuromancer"])
    ids_ = [r.item["id"] for r in res.recommendations]
    assert len(ids_) == len(set(ids_))


def test_blend_mode_still_available(cosine_engine):
    res = cosine_engine(multi_book_mode="blend").recommend(["Dune", "Beach Days"])
    assert all(_because(r) == [1, 6] for r in res.recommendations)


def test_unknown_mode_is_an_error(cosine_engine):
    with pytest.raises(ConfigError, match="MULTI_BOOK_MODE"):
        cosine_engine(multi_book_mode="average")
