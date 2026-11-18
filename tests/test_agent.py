import json

import numpy as np
import pandas as pd
import pytest

from src import agent
from src.checks import Failure, check_arrival, check_artists, check_progress, check_step_sizes
from src.data import FEATURES, prepare_catalog
from src.planner import Waypoint, plan_waypoints, resolve_destination
from src.scorer import distance, score_songs

# --------------------------------------------------------------------------- fixtures


def synthetic_raw(n=2000, seed=0) -> pd.DataFrame:
    """Songs spread along a line in feature space: calm neo soul at t=0, loud hard rock at t=1."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, n)

    def f(start, end):
        return np.clip(start + (end - start) * t + rng.normal(0, 0.03, n), 0, 1)

    rock = t > 0.6
    return pd.DataFrame(
        {
            "track_id": [f"{i:022d}" for i in range(n)],
            "track_name": [f"Song {i}" for i in range(n)],
            "track_artist": [f"Artist {i}" for i in range(n)],
            "track_popularity": 50,
            "playlist_genre": np.where(rock, "rock", "r&b"),
            "playlist_subgenre": np.where(rock, "hard rock", "neo soul"),
            "danceability": f(0.7, 0.4),
            "energy": f(0.3, 0.95),
            "speechiness": f(0.10, 0.06),
            "acousticness": f(0.6, 0.05),
            "instrumentalness": f(0.0, 0.1),
            "liveness": f(0.1, 0.2),
            "valence": f(0.6, 0.3),
            "tempo": 90 + 70 * t + rng.normal(0, 3, n),
            "loudness": -12 + 9 * t + rng.normal(0, 0.5, n),
        }
    )


@pytest.fixture(scope="module")
def catalog():
    return prepare_catalog(synthetic_raw())


@pytest.fixture(scope="module")
def playlist(catalog):
    return catalog.iloc[:20]  # the calmest neo soul songs


def path_frame(**cols) -> pd.DataFrame:
    n = len(next(iter(cols.values())))
    base = {
        "artist": [f"A{i}" for i in range(n)],
        "progress": np.linspace(0.1, 0.9, n),
        "is_destination": [False] * (n - 1) + [True],
        "step_distance": [0.0] + [0.05] * (n - 1),
    }
    return pd.DataFrame(base | cols)


# --------------------------------------------------------------------------- planner


def test_waypoints_are_evenly_spaced_and_end_at_target():
    wps = plan_waypoints(np.zeros(9), np.ones(9), 4)
    assert [wp.vector[0] for wp in wps] == [0.25, 0.5, 0.75, 1.0]


def test_route_needs_at_least_one_step():
    with pytest.raises(ValueError):
        plan_waypoints(np.zeros(9), np.ones(9), 0)


def test_resolve_destination(catalog):
    assert resolve_destination(catalog, "Rock").level == "genre"
    assert resolve_destination(catalog, " hard rock ").level == "subgenre"


def test_unknown_destination_suggests_close_match(catalog):
    with pytest.raises(ValueError, match="Did you mean: hard rock"):
        resolve_destination(catalog, "hard rok")


# --------------------------------------------------------------------------- scorer


def test_scorer_prefers_the_song_closer_to_the_waypoint():
    target = np.full(9, 0.5)
    songs = np.array([np.full(9, 0.9), np.full(9, 0.55)])
    scores = score_songs(songs, ["happy", "happy"], [0.5, 0.5], target, 0.5, "happy")
    assert scores.argmax() == 1


def test_scorer_mood_and_genre_bonuses():
    target = np.full(9, 0.5)
    songs = np.array([np.full(9, 0.5)] * 3)
    scores = score_songs(songs, ["sad", "happy", "happy"], [0.5, 0.5, 0.9], target, 0.5, "happy")
    assert scores[1] - scores[0] == pytest.approx(1.0)  # mood match is worth 1
    assert scores[1] - scores[2] == pytest.approx(2 * 0.4)  # compass off by 0.4, weight 2


# --------------------------------------------------------------------------- checks


def test_check_step_sizes():
    path = path_frame(step_distance=[0.0, 0.05, 0.2, 0.05])
    assert [f.step for f in check_step_sizes(path, 0.1)] == [2]


def test_check_progress_allows_small_dips():
    path = path_frame(progress=[0.1, 0.3, 0.28, 0.1])
    assert [f.step for f in check_progress(path, tolerance=0.05)] == [3]


def test_check_arrival():
    assert check_arrival(path_frame(progress=[0.1, 0.6]), 0.5) == []
    assert check_arrival(path_frame(progress=[0.1, 0.4]), 0.5)[0].check == "no_arrival"
    not_member = path_frame(progress=[0.1, 0.9], is_destination=[False, False])
    assert check_arrival(not_member, 0.5)[0].check == "no_arrival"


def test_check_artists():
    path = path_frame(artist=["A", "B", "A", "C"])
    assert [f.step for f in check_artists(path)] == [2]


# --------------------------------------------------------------------------- revise


def revise_path(n):
    vectors = np.arange(n, dtype=float)[:, None] * np.ones(len(FEATURES))
    path = pd.DataFrame(vectors, columns=FEATURES)
    path["artist"] = [f"A{i}" for i in range(n)]
    path["progress"] = np.linspace(0.1, 0.9, n)
    return path


def test_revise_inserts_bridge_at_midpoint():
    wps = [Waypoint(np.full(9, float(i))) for i in (1, 2, 3)]
    new, fixes = agent.revise(wps, revise_path(4), [Failure("step_too_big", 2, "")])
    assert len(new) == 4
    assert new[1].kind == "bridge"
    assert new[1].vector[0] == pytest.approx(1.5)  # halfway between path rows 1 and 2
    assert len(wps) == 3  # original plan untouched


def test_revise_bans_artist_floors_progress_and_requires_arrival():
    wps = [Waypoint(np.full(9, float(i))) for i in (1, 2, 3)]
    failures = [
        Failure("artist_repeat", 1, ""),
        Failure("progress_dropped", 2, ""),
        Failure("no_arrival", 3, ""),
    ]
    new, fixes = agent.revise(wps, revise_path(4), failures)
    assert new[0].banned_artists == {"A1"}
    assert new[1].min_progress == pytest.approx(revise_path(4).loc[1, "progress"])
    assert new[2].require_destination
    assert len(fixes) == 3


# --------------------------------------------------------------------------- agent


def test_route_arrives_and_passes_all_checks(catalog, playlist):
    result = agent.run(catalog, playlist, "hard rock")
    assert result.passed
    last = result.path.iloc[-1]
    assert "hard rock" in last["subgenres"] and last["progress"] >= agent.ARRIVAL_PROGRESS
    assert result.path["artist"].is_unique
    assert (result.path["step_distance"] <= agent.MAX_STEP).all()


def test_plan_requires_the_last_song_to_arrive(catalog, playlist):
    result = agent.run(catalog, playlist, "hard rock")
    first = result.trace["rounds"][0]
    assert first["waypoints"][-1].get("require_destination")
    assert not any(f["check"] == "no_arrival" for f in first["failed_checks"])


def test_revise_changes_the_output(catalog, playlist):
    # One giant step can't pass the step-size check, so Revise must add bridges.
    result = agent.run(catalog, playlist, "hard rock", n_steps=1)
    first = result.trace["rounds"][0]
    assert any(f["check"] == "step_too_big" for f in first["failed_checks"])
    assert result.revisions > 0
    assert len(result.path) > len(first["path"])
    assert "bridge" in set(result.path["kind"])


def test_same_input_gives_same_route(catalog, playlist):
    a = agent.run(catalog, playlist, "hard rock")
    b = agent.run(catalog, playlist, "hard rock")
    assert a.path["track_id"].tolist() == b.path["track_id"].tolist()
    assert a.trace == b.trace


def test_never_recommends_playlist_songs_or_other_releases_of_them(catalog, playlist):
    # Add another release of a playlist song (new ID, same title + artist) near the route start.
    copy = catalog.iloc[[21]].copy()
    copy["track_id"] = "X" * 22
    copy["track_name"] = playlist.iloc[0]["track_name"] + " - Remastered 2011"
    copy["artist"] = playlist.iloc[0]["artist"]
    result = agent.run(pd.concat([catalog, copy], ignore_index=True), playlist, "hard rock")
    picks = result.path.iloc[1:]
    assert not picks["track_id"].isin(set(playlist["track_id"]) | {"X" * 22}).any()


def test_revision_cap_returns_best_path_with_warning(catalog, playlist):
    result = agent.run(catalog, playlist, "hard rock", n_steps=1, max_step=0.001, max_revisions=2)
    assert not result.passed
    assert len(result.trace["rounds"]) == 3
    assert "revision cap" in result.warnings[0]


def test_empty_playlist_is_rejected(catalog):
    with pytest.raises(ValueError, match="no taste"):
        agent.run(catalog, catalog.iloc[:0], "hard rock")


def test_trace_is_json_serialisable(catalog, playlist):
    json.dumps(agent.run(catalog, playlist, "hard rock", n_steps=1).trace)


def test_waypoint_distances_shrink_with_more_steps():
    a, b = np.zeros(9), np.ones(9)
    gaps = [distance(a, plan_waypoints(a, b, n)[0].vector) for n in (1, 2, 4)]
    assert gaps == sorted(gaps, reverse=True)
