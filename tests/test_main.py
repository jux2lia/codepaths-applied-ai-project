"""End-to-end tests of the command line, on the real catalog and the example playlists."""

import json

import pytest

from src import main as cli
from src.data import CATALOG_PATH, ROOT

pytestmark = pytest.mark.skipif(not CATALOG_PATH.exists(), reason="run `python data/download.py` first")

EXAMPLE = str(ROOT / "examples" / "neo_soul.csv")


@pytest.fixture(autouse=True)
def log_to_tmp(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "LOG_FILE", tmp_path / "test.log")


def run_cli(tmp_path, *args):
    return cli.main(["--playlist", EXAMPLE, "--trace-dir", str(tmp_path / "traces"), *args])


def load_traces(tmp_path):
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted((tmp_path / "traces").glob("*.json"))]


def test_prints_route_and_saves_trace(tmp_path, capsys):
    assert run_cli(tmp_path, "--to", "hard rock") == 0
    out = capsys.readouterr().out
    assert "15/15 songs found" in out
    assert "Route (" in out and "Full trace saved" in out

    [trace] = load_traces(tmp_path)
    assert trace["playlist"]["match_rate"] == 1.0
    assert trace["destination"] == {"name": "hard rock", "level": "subgenre"}
    assert trace["rounds"] and "result" in trace
    assert "Round 0" in (tmp_path / "test.log").read_text(encoding="utf-8")


def test_same_input_gives_same_trace(tmp_path):
    run_cli(tmp_path, "--to", "hard rock")
    run_cli(tmp_path, "--to", "hard rock")
    a, b = load_traces(tmp_path)  # two files, even within the same second
    a.pop("run"), b.pop("run")  # only the timestamp differs
    assert a == b


def test_unknown_destination_exits_with_suggestion(tmp_path, capsys):
    assert run_cli(tmp_path, "--to", "hard rok") == 1
    assert "Did you mean: hard rock?" in capsys.readouterr().err
    assert not (tmp_path / "traces").exists()


def test_missing_playlist_exits_with_message(tmp_path, capsys):
    code = cli.main(["--playlist", str(tmp_path / "nope.csv"), "--to", "rock", "--no-save"])
    assert code == 1
    assert "Playlist not found" in capsys.readouterr().err
