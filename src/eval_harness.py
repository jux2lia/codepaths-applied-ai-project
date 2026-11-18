"""Evaluation harness: run fixed scenarios through the guarded agent and score them.

    python -m src.eval_harness              # PASS/FAIL per scenario + overall score
    python -m src.eval_harness --markdown   # same, as a table for the README

Uses only the public catalog and the example playlists (no private data), so
anyone can reproduce the results. Exits with code 1 if any scenario fails.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Callable

import pandas as pd

from src import main as cli
from src.data import ROOT, load_catalog, load_playlist
from src.guardrails import GuardedRun, run_with_guardrails

EXAMPLES = ROOT / "examples"


@dataclass
class Scenario:
    name: str
    playlist: Callable[[pd.DataFrame], pd.DataFrame]  # catalog -> playlist table
    destination: str
    expect: str  # what should happen, in words
    check: Callable[[GuardedRun | None, Exception | None], tuple[bool, str]]
    settings: dict | None = None


# --------------------------------------------------------------------------- playlists


def example(name: str):
    return lambda catalog: load_playlist(EXAMPLES / f"{name}.csv")


def from_catalog(rows: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {"track_id": rows["track_id"], "track_name": rows["track_name"], "artists": rows["artist"]}
    ).reset_index(drop=True)


def fake_songs(n: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "track_id": [f"{'Z' * 18}{i:04d}" for i in range(n)],
            "track_name": [f"Unreleased demo {i}" for i in range(n)],
            "artists": ["Bedroom Band"] * n,
        }
    )


def empty(catalog):
    return pd.DataFrame(columns=["track_id", "track_name", "artists"])


def one_song(catalog):
    return load_playlist(EXAMPLES / "neo_soul.csv").head(1)


def half_matched(catalog):
    return pd.concat([load_playlist(EXAMPLES / "neo_soul.csv").head(4), fake_songs(6)], ignore_index=True)


def one_artist(catalog):
    queen = catalog[catalog["artist"] == "Queen"].sort_values("track_id").head(10)
    return from_catalog(queen)


# --------------------------------------------------------------------------- expectations


def warned(run, text):
    return any(text in w for w in run.warnings)


def describe(run: GuardedRun) -> str:
    r = run.result
    kinds = r.path["kind"].tolist()
    bridges = kinds.count("bridge")
    status = "passed" if r.passed else "failed"
    return (
        f"{len(r.path) - 1} steps, {r.revisions} revision(s), {bridges} bridge(s), checks {status}, "
        f"ends on {r.path.iloc[-1]['track_name']} — {r.path.iloc[-1]['artist']} ({r.path.iloc[-1]['subgenre']})"
    )


def arrives(run, err):
    if err:
        return False, f"error: {err}"
    r = run.result
    ok = r.passed and r.trace["result"]["remaining_failures"] == []
    return ok, describe(run)


def revised_with_bridge(run, err):
    if err:
        return False, f"error: {err}"
    r = run.result
    first = r.trace["rounds"][0]["failed_checks"]
    ok = r.passed and any(f["check"] == "step_too_big" for f in first) and "bridge" in set(r.path["kind"])
    return ok, describe(run)


def stopped_with(text):
    def check(run, err):
        if err is None:
            return False, "ran instead of stopping"
        message = str(err).splitlines()[0]
        return text in str(err), f"stopped: {message}"
    return check


def runs_with_warning(text, also=None):
    def check(run, err):
        if err:
            return False, f"error: {err}"
        ok = warned(run, text) and (also is None or also(run))
        w = next((w for w in run.warnings if text in w), "no matching warning")
        return ok, f"warning: {w} | {describe(run)}"
    return check


def no_repeated_artists(run, err):
    if err:
        return False, f"error: {err}"
    artists = run.result.path["artist"]
    ok = artists.is_unique and run.result.passed
    return ok, f"artists on route: {artists.nunique()}/{len(artists)} unique | {describe(run)}"


def cap_hit(run, err):
    if err:
        return False, f"error: {err}"
    r = run.result
    ok = (not r.passed) and warned(run, "revision cap") and len(r.path) > 1
    return ok, f"warning: {r.warnings[0] if r.warnings else 'none'} | {describe(run)}"


SCENARIOS = [
    Scenario("Demo: neo soul -> hard rock", example("neo_soul"), "hard rock",
             "route arrives in hard rock, all checks pass", arrives),
    Scenario("Demo: classic rock -> big room", example("classic_rock"), "big room",
             "step too big in round 0 -> Revise inserts a bridge song", revised_with_bridge),
    Scenario("Demo: classic rock -> reggaeton", example("classic_rock"), "reggaeton",
             "long trip across genres arrives, all checks pass", arrives),
    Scenario("Typo in destination", example("neo_soul"), "hard rok",
             "stop and suggest 'hard rock'", stopped_with("Did you mean: hard rock?")),
    Scenario("Genre not in dataset", example("neo_soul"), "jazz",
             "stop and list the valid genres", stopped_with("Genres: edm")),
    Scenario("Empty playlist", empty, "rock",
             "stop: playlist is empty", stopped_with("playlist is empty")),
    Scenario("No songs in catalog (0% match)", lambda c: fake_songs(5), "rock",
             "stop: 0% match, explain catalog coverage", stopped_with("0% match")),
    Scenario("Low match rate (4 of 10)", half_matched, "hard rock",
             "warn about 40% match, still build a route", runs_with_warning("40%")),
    Scenario("One-song playlist", one_song, "hard rock",
             "warn that the profile is one song, still build a route",
             runs_with_warning("Only one song matched")),
    Scenario("Destination = current taste", example("neo_soul"), "neo soul",
             "warn the route stays close to home, still arrive",
             runs_with_warning("already neo soul", also=lambda run: run.result.passed)),
    Scenario("Whole playlist is one artist", one_artist, "trap",
             "no artist repeats anywhere on the route", no_repeated_artists),
    Scenario("Out-of-range settings (steps=0, max_step=5)", example("neo_soul"), "rock",
             "clamp to steps=1, max_step=0.5 with warnings, still build a route",
             runs_with_warning("Steps must be", also=lambda run: warned(run, "Max step must be")),
             settings={"n_steps": 0, "max_step": 5}),
    Scenario("Impossible step limit (max_step=0.03)", example("classic_rock"), "big room",
             "hit the revision cap, return best route with a warning", cap_hit,
             settings={"max_step": 0.03}),
]


# --------------------------------------------------------------------------- runner


def run_scenario(s: Scenario, catalog: pd.DataFrame) -> tuple[bool, str]:
    run, err = None, None
    try:
        run = run_with_guardrails(catalog, s.playlist(catalog), s.destination, **(s.settings or {}))
    except ValueError as e:  # GuardrailError and unknown destinations
        err = e
    return s.check(run, err)


def check_determinism(catalog: pd.DataFrame) -> tuple[bool, str]:
    routes = [
        run_with_guardrails(catalog, load_playlist(EXAMPLES / "neo_soul.csv"), "hard rock")
        .result.path["track_id"].tolist()
        for _ in range(2)
    ]
    return routes[0] == routes[1], "two runs, identical route" if routes[0] == routes[1] else "routes differ"


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(prog="python -m src.eval_harness", description=__doc__.split("\n")[0])
    parser.add_argument("--markdown", action="store_true", help="print a Markdown table for the README")
    args = parser.parse_args(argv)
    cli.setup_logging(cli.LOG_FILE)  # warnings go to the log file, not the terminal

    catalog = load_catalog()
    rows = [(s.name, s.expect, *run_scenario(s, catalog)) for s in SCENARIOS]
    rows.append(("Same input twice", "identical route both times", *check_determinism(catalog)))
    passed = sum(ok for _, _, ok, _ in rows)

    if args.markdown:
        print("| Scenario | Expected | Result | What happened |")
        print("|---|---|---|---|")
        for name, expect, ok, detail in rows:
            print(f"| {name} | {expect} | {'✅ PASS' if ok else '❌ FAIL'} | {detail.replace('|', '/')} |")
    else:
        for name, expect, ok, detail in rows:
            print(f"{'PASS' if ok else 'FAIL'}  {name}")
            print(f"      expected: {expect}")
            print(f"      got:      {detail}")
    print(f"\nScore: {passed}/{len(rows)} scenarios passed ({passed / len(rows):.0%})")
    return 0 if passed == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
