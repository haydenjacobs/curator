"""The command-line wrapper: output format and clear errors."""

import config
import recommend


def test_cli_prints_recommendations(catalog, monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(config, "PROCESSED_DIR", tmp_path / "processed")
    monkeypatch.setattr(config, "TAG_THRESHOLD", 0.5)
    assert recommend.main(["Dune", "Neuromancer"]) == 0
    out = capsys.readouterr().out
    assert "Because you liked:" in out and "1. " in out and "why:" in out
    assert recommend.main(["Dune", "Neuromancer"]) == 0
    assert capsys.readouterr().out == out  # deterministic


def test_cli_ambiguous_title_exits_with_list(catalog, monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(config, "PROCESSED_DIR", tmp_path / "processed")
    assert recommend.main(["Twins"]) == 1
    err = capsys.readouterr().err
    assert "matches 2 books" in err and "id:7" in err and "id:8" in err


def test_cli_missing_book(catalog, monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(config, "PROCESSED_DIR", tmp_path / "processed")
    assert recommend.main(["Moby Dick"]) == 1
    assert "not in the catalog" in capsys.readouterr().err
