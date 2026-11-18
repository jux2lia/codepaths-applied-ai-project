import pandas as pd
import pytest

from src import eval_harness
from src.data import CATALOG_PATH, PlaylistMatch, prepare_catalog
from src.guardrails import (
    MAX_MAX_STEP,
    MAX_STEPS,
    GuardrailError,
    check_playlist,
    clamp_settings,
    run_with_guardrails,
)
from tests.test_agent import synthetic_raw


@pytest.fixture(scope="module")
def catalog():
    return prepare_catalog(synthetic_raw())


def as_playlist(rows: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {"track_id": rows["track_id"], "track_name": rows["track_name"], "artists": rows["artist"]}
    )


def match(total, matched):
    unmatched = pd.DataFrame({"track_id": [None] * (total - matched)})
    return PlaylistMatch(matched=pd.DataFrame(), unmatched=unmatched, total=total)


# --------------------------------------------------------------------------- settings


def test_settings_in_range_are_untouched():
    assert clamp_settings(4, 0.12) == (4, 0.12, [])


@pytest.mark.parametrize(
    "steps, max_step, expected",
    [(0, 0.12, (1, 0.12)), (99, 0.12, (MAX_STEPS, 0.12)), (4, 5.0, (4, MAX_MAX_STEP)), (4, -1, (4, 0.03))],
)
def test_out_of_range_settings_are_clamped_with_a_warning(steps, max_step, expected):
    n, s, warnings = clamp_settings(steps, max_step)
    assert (n, s) == expected
    assert len(warnings) == 1


# --------------------------------------------------------------------------- playlist


def test_empty_playlist_stops():
    with pytest.raises(GuardrailError, match="empty"):
        check_playlist(match(0, 0))


def test_zero_match_stops():
    with pytest.raises(GuardrailError, match="0% match"):
        check_playlist(match(5, 0))


def test_low_match_rate_warns():
    [warning] = check_playlist(match(10, 3))
    assert "30%" in warning


def test_good_match_rate_has_no_warning():
    assert check_playlist(match(10, 8)) == []


def test_single_song_warns():
    assert any("one song" in w for w in check_playlist(match(1, 1)))


# --------------------------------------------------------------------------- full run


def test_guarded_run_on_good_input_has_no_warnings(catalog):
    run = run_with_guardrails(catalog, as_playlist(catalog.iloc[:20]), "hard rock")
    assert run.warnings == []
    assert run.result.passed
    assert run.result.trace["guardrail_warnings"] == []


def test_destination_already_in_taste_warns(catalog):
    run = run_with_guardrails(catalog, as_playlist(catalog.iloc[:20]), "neo soul")
    assert any("already neo soul" in w for w in run.warnings)


def test_clamped_settings_are_used(catalog):
    run = run_with_guardrails(catalog, as_playlist(catalog.iloc[:20]), "hard rock", n_steps=0, max_step=9)
    assert run.settings == {"n_steps": 1, "max_step": MAX_MAX_STEP, "lineage": True}
    assert len(run.result.trace["rounds"][0]["path"]) == 2  # start + 1 step


def test_unknown_destination_stops(catalog):
    with pytest.raises(ValueError, match="Did you mean"):
        run_with_guardrails(catalog, as_playlist(catalog.iloc[:20]), "hard rok")


@pytest.mark.skipif(not CATALOG_PATH.exists(), reason="run `python data/download.py` first")
def test_eval_harness_all_scenarios_pass(capsys, tmp_path, monkeypatch):
    monkeypatch.setattr(eval_harness.cli, "LOG_FILE", tmp_path / "test.log")
    assert eval_harness.main([]) == 0
    assert "FAIL" not in capsys.readouterr().out
