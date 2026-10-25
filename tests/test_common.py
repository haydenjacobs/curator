"""The "common" algorithm: overlap first, uniqueness, books + movies, mixed lists."""

import numpy as np
import pytest

from curator.engine import common_core, core_match, excess_scores, quantile_map
from curator.errors import AmbiguousItemError, ConfigError, CuratorError, ItemNotFoundError
from curator.lookup import find_item


def ids(result):
    return [r.item["id"] for r in result.recommendations]


def core_names(result):
    return [t.name for t in result.common_tags]


# --- the building blocks ------------------------------------------------------

def test_excess_is_distance_above_typical_never_negative():
    scores = np.array([[0.9, 0.1], [0.2, 0.6]])
    medians = np.array([0.3, 0.3])
    assert excess_scores(scores, medians) == pytest.approx(np.array([[0.6, 0.0], [0.0, 0.3]]))


def test_a_tag_only_one_input_has_is_not_common():
    excess = np.array([[0.5, 0.8], [0.4, 0.0]])
    applies = np.array([[True, True], [True, False]])
    core, common, _ = common_core(excess, applies, np.ones(2), np.ones(2), share=1.0, size=10)
    assert list(core) == [0]
    assert common[1] == 0.0
    assert common[0] == pytest.approx(0.4)  # what BOTH have: the weaker of the two


def test_a_tag_must_really_apply_to_every_input():
    # Input 2 is a little above typical on tag 0 but its score is below the threshold.
    excess = np.array([[0.5, 0.3], [0.1, 0.3]])
    applies = np.array([[True, True], [False, True]])
    core, _, _ = common_core(excess, applies, np.ones(2), np.ones(2), share=1.0, size=10)
    assert list(core) == [1]


def test_rare_commonalities_outrank_generic_ones():
    excess = np.array([[0.4, 0.4], [0.4, 0.4]])
    applies = np.ones((2, 2), dtype=bool)
    rarity = np.array([0.2, 2.0])  # tag 1 is rare
    core, _, strength = common_core(excess, applies, rarity, np.ones(2), share=1.0, size=10)
    assert list(core) == [1, 0]
    assert strength[1] > strength[0]


def test_weak_commonalities_are_cut_off():
    excess = np.array([[0.5, 0.01], [0.5, 0.01]])
    applies = np.ones((2, 2), dtype=bool)
    core, _, _ = common_core(excess, applies, np.ones(2), np.ones(2), share=1.0, size=10, min_strength=0.05)
    assert list(core) == [0]


def test_common_share_relaxes_how_many_inputs_need_a_tag():
    excess = np.array([[0.5], [0.4], [0.0]])
    applies = np.array([[True], [True], [False]])
    all_core, _, _ = common_core(excess, applies, np.ones(1), np.ones(1), share=1.0, size=10)
    two_of_three, common, _ = common_core(excess, applies, np.ones(1), np.ones(1), share=0.66, size=10)
    assert list(all_core) == [] and list(two_of_three) == [0]
    assert common[0] == pytest.approx(0.4)


def test_core_match_counts_only_core_tags_weighted_by_strength():
    cand = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert core_match(cand, np.array([3.0, 1.0])) == pytest.approx([0.75, 0.25])


def test_quantile_map_puts_scores_on_the_reference_scale():
    generous = np.array([[0.2], [0.6], [0.8], [1.0]])     # e.g. movie scores
    strict = np.array([[0.0], [0.1], [0.2], [0.5]])       # e.g. book scores
    mapped = quantile_map(generous, strict)
    assert list(mapped[:, 0]) == pytest.approx([0.0, 0.1, 0.2, 0.5])
    tied = quantile_map(np.array([[0.3], [0.3], [0.9]]), np.array([[0.0], [0.4], [1.0]]))
    assert tied[0, 0] == tied[1, 0] < tied[2, 0]


# --- books in, books out ----------------------------------------------------------

def test_overlap_of_two_books_drives_the_results(make_engine):
    # Dune (space, desert) + Neuromancer (space, cyberpunk): only "space" is shared.
    res = make_engine().recommend(["Dune", "Neuromancer"])
    assert res.mode == "common" and not res.shared_space
    assert core_names(res) == ["space"]
    assert ids(res)[0] == 2  # Dune Messiah: strongest on space (ties with Children of Dune -> lower id)
    for r in res.recommendations:
        assert [t.name for t in r.shared_tags] in ([], ["space"])
        assert sum(t.contribution for t in r.shared_tags) == pytest.approx(r.score)


def test_generic_tags_never_count_as_common(make_engine):
    res = make_engine().recommend(["Dune"])
    assert "fiction" not in core_names(res)  # every book has it
    assert {"space", "desert"} <= set(core_names(res))


def test_nothing_in_common_is_a_clear_error(make_engine):
    with pytest.raises(CuratorError, match="nothing distinctive in common"):
        make_engine().recommend(["Beach Days", "Neuromancer"])


def test_common_share_setting(make_engine):
    with pytest.raises(CuratorError, match="nothing distinctive in common"):
        make_engine().recommend(["Dune", "Arrakis Tales", "Beach Days"])
    res = make_engine(common_share=0.66).recommend(["Dune", "Arrakis Tales", "Beach Days"])
    assert "desert" in core_names(res)


def test_inputs_never_recommended_and_author_cap(make_engine):
    res = make_engine(top_n=5).recommend(["Arrakis Tales"])
    assert 5 not in ids(res)
    assert sum(r.item["creator"] == "Frank Herbert" for r in res.recommendations) == 1
    uncapped = make_engine(top_n=5, max_per_author=None).recommend(["Arrakis Tales"])
    assert sum(r.item["creator"] == "Frank Herbert" for r in uncapped.recommendations) == 3


def test_common_mode_is_deterministic_and_order_independent(make_engine):
    e = make_engine(top_n=5)
    a = e.recommend(["Dune", "Neuromancer"])
    assert a == e.recommend(["Dune", "Neuromancer"])
    b = e.recommend(["Neuromancer", "Dune"])
    assert a.recommendations == b.recommendations and core_names(a) == core_names(b)


def test_dislikes_are_refused_in_common_mode(make_engine, catalog):
    e = make_engine()
    with pytest.raises(CuratorError, match="Dislikes are not supported"):
        e.recommend_rows([("book", find_item(catalog, "Dune"), 1.0), ("book", find_item(catalog, "1984"), -1.0)])


# --- movies -------------------------------------------------------------------

def test_movie_lookup(movies):
    assert movies.item_info(find_item(movies, "heat"))["id"] == 103
    assert movies.item_info(find_item(movies, "Heat (1995)"))["id"] == 103
    assert movies.item_info(find_item(movies, "The Three Musketeers"))["id"] == 107
    assert movies.item_info(find_item(movies, "Se7en"))["id"] == 108
    with pytest.raises(AmbiguousItemError) as exc:
        find_item(movies, "Dune")
    assert "matches 2 movies" in str(exc.value)
    assert [c["id"] for c in exc.value.candidates] == [101, 102]
    with pytest.raises(ItemNotFoundError, match="No movie with id 999"):
        find_item(movies, "id:999")
    with pytest.raises(ItemNotFoundError, match="not in the movie catalog"):
        find_item(movies, "Moby Dick")


def test_book_and_movie_in_movies_out(make_engine):
    # Dune (book: space, desert) + Lawrence of Arabia (movie: desert).
    res = make_engine().recommend(["Dune"], ["Lawrence of Arabia"], want="movies")
    assert res.shared_space
    assert core_names(res) == ["desert"]
    assert all(r.item["medium"] == "movie" for r in res.recommendations)
    assert 109 not in ids(res)                       # the input movie
    assert set(ids(res)[:2]) == {101, 102}           # the two Dune films: strongest on desert


def test_only_shared_tags_are_used_across_media(make_engine, library):
    res = make_engine().recommend(["Neuromancer"], [], want="movies")
    shared = set(library.shared_tags["name"])
    assert set(core_names(res)) <= shared
    # Muting a book-only tag can't change a book->movie comparison.
    muted = make_engine(tag_multipliers={"fiction": 0.0}).recommend(["Neuromancer"], [], want="movies")
    assert muted.recommendations == res.recommendations


def test_movie_director_cap(make_engine):
    # Neuromancer -> cyberpunk/space movies: Blade Runner and Alien are both Ridley Scott.
    res = make_engine(top_n=5).recommend(["Neuromancer"], [], want="movies")
    assert sum(r.item["creator"] == "Ridley Scott" for r in res.recommendations) == 1
    res2 = make_engine(top_n=5, max_per_author=None).recommend(["Neuromancer"], [], want="movies")
    assert {105, 106} <= set(ids(res2))


def test_mixed_list_holds_both_media(make_engine):
    res = make_engine(top_n=3).recommend(["Dune"], ["Lawrence of Arabia"], want="mixed")
    media = [r.item["medium"] for r in res.recommendations]
    assert "book" in media and "movie" in media and len(media) == 3
    scores = [round(r.score, 10) for r in res.recommendations]
    assert len(res.recommendations) == 3 and [r.rank for r in res.recommendations] == [1, 2, 3]
    assert scores == sorted(scores, reverse=True)


def test_mixed_list_too_small_for_both_media_is_an_error(make_engine):
    with pytest.raises(ConfigError, match="MIXED_MIN_PER_MEDIUM"):
        make_engine(top_n=1).recommend(["Dune"], [], want="mixed")


def test_movies_only_in_and_out_use_all_movie_tags(make_engine):
    # Heat + Point Break share "heist", a movie-only tag.
    res = make_engine().recommend([], ["Heat", "Point Break"], want="movies")
    assert not res.shared_space
    assert core_names(res)[0] == "heist"


def test_cross_media_results_are_deterministic_and_order_independent(make_engine):
    e = make_engine(top_n=4)
    a = e.recommend(["Dune"], ["Lawrence of Arabia", "Alien"], want="mixed")
    assert a == e.recommend(["Dune"], ["Lawrence of Arabia", "Alien"], want="mixed")
    b = e.recommend(["Dune"], ["Alien", "Lawrence of Arabia"], want="mixed")
    assert a.recommendations == b.recommendations


def test_same_title_as_book_and_movie_is_not_a_duplicate(make_engine):
    res = make_engine().recommend(["Dune"], ["id:101"], want="books")
    assert {(b["medium"], b["id"]) for b in res.inputs} == {("book", 1), ("movie", 101)}


def test_cosine_modes_refuse_movies(cosine_engine):
    with pytest.raises(CuratorError, match='only works with books in and books out'):
        cosine_engine().recommend(["Dune"], ["Heat"], want="books")
    with pytest.raises(CuratorError, match='only works with books in and books out'):
        cosine_engine().recommend(["Dune"], [], want="movies")


def test_unknown_want_is_an_error(make_engine):
    with pytest.raises(CuratorError, match="Unknown output"):
        make_engine().recommend(["Dune"], [], want="games")
