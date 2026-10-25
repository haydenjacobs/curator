"""The command-line wrapper: output format and clear errors."""

import pytest

import config
import recommend


@pytest.fixture
def cli(library, monkeypatch, tmp_path):
    monkeypatch.setattr(config, "PROCESSED_DIR", tmp_path / "processed")
    monkeypatch.setattr(config, "TAG_THRESHOLD", 0.5)
    return recommend.main


def test_cli_prints_recommendations(cli, capsys):
    assert cli(["Dune", "Arrakis Tales"]) == 0
    out = capsys.readouterr().out
    assert "Because you liked:" in out and "What these have in common" in out
    assert "Books for you:" in out and "1. " in out and "why:" in out and "match:" in out
    assert cli(["Dune", "Arrakis Tales"]) == 0
    assert capsys.readouterr().out == out  # deterministic


def test_cli_books_and_movies_mixed(cli, capsys):
    assert cli(["Dune", "--movie", "Lawrence of Arabia", "--want", "mixed"]) == 0
    out = capsys.readouterr().out
    assert "[book] Dune (Dune Chronicles, #1) — Frank Herbert (1965)" in out
    assert "[movie] Lawrence of Arabia (1962) — dir. David Lean  [id:109]" in out
    assert "Books and movies for you:" in out
    assert "tags both datasets have" in out
    assert "[book]" in out.split("for you:")[1] and "[movie]" in out.split("for you:")[1]


def test_cli_movies_only(cli, capsys):
    assert cli(["-m", "Heat", "-m", "Point Break", "-w", "movies"]) == 0
    out = capsys.readouterr().out
    assert "Movies for you:" in out and "heist" in out


def test_cli_needs_an_input(cli, capsys):
    with pytest.raises(SystemExit):
        cli(["--want", "movies"])
    assert "give at least one book" in capsys.readouterr().err


def test_cli_cosine_mode_still_prints(cli, capsys, monkeypatch):
    monkeypatch.setattr(config, "MULTI_BOOK_MODE", "per_book")
    assert cli(["Dune", "Neuromancer"]) == 0
    out = capsys.readouterr().out
    assert "because you liked:" in out and "similarity:" in out and "in common" not in out


def test_cli_ambiguous_title_exits_with_list(cli, capsys):
    assert cli(["Twins"]) == 1
    err = capsys.readouterr().err
    assert "matches 2 books" in err and "id:7" in err and "id:8" in err


def test_cli_ambiguous_movie(cli, capsys):
    assert cli(["--movie", "Dune"]) == 1
    err = capsys.readouterr().err
    assert "matches 2 movies" in err and "id:101" in err and "id:102" in err


def test_cli_missing_book(cli, capsys):
    assert cli(["Moby Dick"]) == 1
    assert "not in the book catalog" in capsys.readouterr().err


def test_cli_nothing_in_common(cli, capsys):
    assert cli(["Beach Days", "Neuromancer"]) == 1
    assert "nothing distinctive in common" in capsys.readouterr().err
