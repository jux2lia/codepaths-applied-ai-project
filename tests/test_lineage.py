import numpy as np
import pandas as pd
import pytest

from src import agent
from src.checks import check_step_sizes
from src.data import CATALOG_PATH, FEATURES, load_catalog, prepare_catalog, song_key
from src.lineage import LineageGraph, describe, load_edges
from src.scorer import distance
from tests.test_agent import synthetic_raw


def edge(frm, to, kind="sample"):
    return {
        "from": {"title": frm[0], "artist": frm[1]},
        "to": {"title": to[0], "artist": to[1]},
        "type": kind,
        "source_url": "https://en.wikipedia.org/wiki/Example",
    }


@pytest.fixture(scope="module")
def catalog():
    return prepare_catalog(synthetic_raw())


@pytest.fixture(scope="module")
def playlist(catalog):
    return catalog.iloc[:20]


def test_describe():
    e = edge(("Under Pressure", "Queen"), ("Ice Ice Baby", "Vanilla Ice"))
    assert describe(e) == '"Ice Ice Baby" (Vanilla Ice) samples "Under Pressure" (Queen)'


def test_graph_links_both_directions_and_every_release():
    keys = np.array([
        song_key("Under Pressure", "Queen"),
        song_key("Under Pressure - Remastered 2011", "Queen"),  # another release
        song_key("Ice Ice Baby", "Vanilla Ice"),
        song_key("Something Else", "Nobody"),
    ], dtype=object)
    g = LineageGraph([edge(("Under Pressure", "Queen"), ("Ice Ice Baby", "Vanilla Ice"))], keys)
    assert g.matched_edges == 1
    assert set(g.neighbours([2])) == {0, 1}
    assert set(g.neighbours([1])) == {2}
    assert g.neighbours([3]) == {}


def test_edges_outside_the_catalog_are_ignored():
    keys = np.array([song_key("Ice Ice Baby", "Vanilla Ice")], dtype=object)
    g = LineageGraph([edge(("Under Pressure", "Queen"), ("Ice Ice Baby", "Vanilla Ice"))], keys)
    assert g.matched_edges == 0


def test_sample_linked_step_may_be_bigger():
    path = pd.DataFrame({
        "step_distance": [0.0, 0.2, 0.2],
        "lineage_with_previous": [False, True, False],
    })
    failures = check_step_sizes(path, max_step=0.12, lineage_max_step=0.3)
    assert [f.step for f in failures] == [2]  # step 1 is justified by the link, step 2 isn't


def lineage_target(catalog, playlist):
    """A song a bit too far from the route's start for a normal step (0.13-0.28 away)
    but further toward hard rock: only a sample link can justify going straight there."""
    start = agent.run(catalog, playlist, "hard rock").path.iloc[0]
    d = distance(catalog[FEATURES].to_numpy(float), start[FEATURES].to_numpy(float))
    ok = (d > 0.13) & (d < 0.28) & (catalog.index > start.name + 100)
    target = catalog[ok].iloc[0]
    return start, target


def test_lineage_link_changes_the_route(catalog, playlist):
    start, target = lineage_target(catalog, playlist)
    edges = [edge((start["track_name"], start["artist"]), (target["track_name"], target["artist"]))]

    without = agent.run(catalog, playlist, "hard rock")
    with_lineage = agent.run(catalog, playlist, "hard rock", lineage_edges=edges)

    assert target["track_id"] not in set(without.path["track_id"])
    step1 = with_lineage.path.iloc[1]
    assert step1["track_id"] == target["track_id"]
    assert step1["lineage_with_previous"] and "samples" in step1["lineage"]
    assert step1["step_distance"] > agent.MAX_STEP  # a jump only the link justifies
    assert with_lineage.passed
    assert with_lineage.trace["settings"]["lineage"] == {"enabled": True, "edges_in_catalog": 1}
    assert "lineage" in with_lineage.trace["rounds"][-1]["path"][1]


def test_without_edges_lineage_columns_are_empty(catalog, playlist):
    result = agent.run(catalog, playlist, "hard rock")
    assert result.path["lineage"].isna().all()
    assert result.trace["settings"]["lineage"]["enabled"] is False


@pytest.mark.skipif(not CATALOG_PATH.exists(), reason="run `python data/download.py` first")
def test_every_lineage_edge_is_in_the_catalog_and_sourced():
    edges = load_edges()
    keys = np.array([song_key(t, a) for t, a in zip(*[load_catalog()[c] for c in ("track_name", "artist")])],
                    dtype=object)
    assert LineageGraph(edges, keys).matched_edges == len(edges)
    assert all(e["source_url"].startswith("https://en.wikipedia.org/wiki/") for e in edges)
    assert all(e.get("evidence") for e in edges)
