"""The Bubble Breaker agent: Plan -> Act -> Check -> Revise.

Plan:   start at the user's most typical song, draw a straight line to the centre of
        the destination genre, and split it into n_steps waypoints. The last
        waypoint must be filled by a song from the destination.
Act:    at each waypoint, take the unused songs nearest to it and let the scorer
        pick the best of them.
Check:  steps small enough? compass reading rising? last song really in the
        destination? no artist twice?
Revise: fix whatever failed (insert a bridge waypoint, ban an artist, raise the
        minimum compass reading, require arrival) and act again, up to max_revisions.

Every round is recorded in a JSON-serialisable trace and logged.
Nothing here is random, so the same input always gives the same route.

Run it from the command line with src/main.py.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.checks import Failure, check_path
from src.compass import GenreCompass
from src.data import FEATURES, derive_mood, song_key
from src.lineage import LineageGraph
from src.planner import Destination, Waypoint, plan_waypoints, resolve_destination
from src.profile import TasteProfile, build_profile
from src.scorer import distance, score_parts

logger = logging.getLogger(__name__)

N_STEPS = 4  # deliberately few, so far routes need bridges
# Max feature distance between consecutive songs. For scale: a song's nearest
# neighbour is ~0.03 away, two random songs ~0.25. Tuned by running all 30
# destinations: at 0.10, 7-9 routes still fail after 5 revisions; at 0.12 all pass.
MAX_STEP = 0.12
MAX_REVISIONS = 5
MAX_WAYPOINTS = 12
ARRIVAL_PROGRESS = 0.5  # the last song must clearly sound like the destination
PROGRESS_TOLERANCE = 0.05  # small compass dips are noise, not backtracking
# Act only considers this many songs nearest each waypoint. Without it the genre term
# (weight 2) outweighs closeness and the agent jumps to far-away songs with the right
# compass reading.
CANDIDATES_PER_WAYPOINT = 50
# Lineage (src/lineage.py): songs linked by a sample to the previous song or to the
# playlist join the candidates and get this bonus (the scale of a mood match).
LINEAGE_BONUS = 1.0
# A step along a sample link may be this big: the shared sample connects the two
# songs even when their audio features don't (Under Pressure -> Ice Ice Baby: 0.22).
LINEAGE_MAX_STEP = 0.30

_VALENCE = FEATURES.index("valence")
_ENERGY = FEATURES.index("energy")


@dataclass
class AgentResult:
    path: pd.DataFrame  # one row per song; row 0 is the user's own song
    trace: dict
    passed: bool  # True if the returned path passes every check
    warnings: list[str] = field(default_factory=list)

    @property
    def revisions(self) -> int:
        return len(self.trace["rounds"]) - 1


class _Pool:
    """The catalog as arrays, plus each song's compass reading."""

    def __init__(self, catalog: pd.DataFrame, destination: Destination, lineage_edges=None):
        self.catalog = catalog.reset_index(drop=True)
        self.X = self.catalog[FEATURES].to_numpy(float)
        self.artists = self.catalog["artist"].to_numpy(object)
        self.moods = self.catalog["mood"].to_numpy(object)
        self.keys = np.array(
            [song_key(t, a) for t, a in zip(self.catalog["track_name"], self.catalog["artist"])],
            dtype=object,
        )
        self.is_destination = destination.members(self.catalog)
        self.compass = GenreCompass(self.X, self.is_destination)
        self.progress = self.compass.progress(self.X)
        self.lineage = LineageGraph(lineage_edges, self.keys) if lineage_edges else None

    def index_of(self, track_ids) -> np.ndarray:
        return np.flatnonzero(self.catalog["track_id"].isin(set(track_ids)).to_numpy())


def run(
    catalog: pd.DataFrame,
    playlist_songs: pd.DataFrame,
    destination: str,
    n_steps: int = N_STEPS,
    max_step: float = MAX_STEP,
    max_revisions: int = MAX_REVISIONS,
    lineage_edges: list[dict] | None = None,
) -> AgentResult:
    """Build a route from the playlist's taste to `destination` (a genre or subgenre).

    lineage_edges (from src.lineage.load_edges) switch on the second retrieval
    source; None leaves the agent on audio features alone.
    """
    dest = resolve_destination(catalog, destination)
    pool = _Pool(catalog, dest, lineage_edges)
    profile = build_profile(playlist_songs)

    # Plan
    start = int(pool.index_of([profile.anchor["track_id"]])[0])
    # Aim for the heart of the destination: songs that are listed under it AND clearly
    # sound like it. (The average of every listed song is too generic.)
    arrived = pool.is_destination & (pool.progress >= ARRIVAL_PROGRESS)
    target = pool.X[arrived if arrived.any() else pool.is_destination].mean(axis=0)
    waypoints = plan_waypoints(pool.X[start], target, n_steps)
    # The route must end in the destination; that's known up front, so it's part of
    # the plan. (The arrival check stays as a safety net, e.g. if constraints get relaxed.)
    waypoints[-1].require_destination = True
    # Don't recommend songs they already have, including other releases of them.
    owned = {song_key(t, a) for t, a in zip(playlist_songs["track_name"], playlist_songs["artist"])}
    exclude = np.isin(pool.keys, list(owned))
    exclude[pool.index_of(playlist_songs["track_id"])] = True

    trace = {
        "destination": {"name": dest.name, "level": dest.level},
        "settings": {
            "n_steps": n_steps,
            "max_step": max_step,
            "max_revisions": max_revisions,
            "arrival_progress": ARRIVAL_PROGRESS,
            "lineage": {
                "enabled": pool.lineage is not None,
                "edges_in_catalog": pool.lineage.matched_edges if pool.lineage else 0,
            },
        },
        "profile": _profile_summary(profile, pool, start),
        "rounds": [],
    }
    warnings: list[str] = []
    best = None  # (failures, path) of the round with the fewest failures
    logger.info(
        "Route to %s (%s) from %s, %d steps, max_step=%.3f",
        dest.name, dest.level, trace["profile"]["start_song"], n_steps, max_step,
    )

    for round_no in range(max_revisions + 1):
        # Act
        path, notes = _act(pool, waypoints, start, exclude, max_step)
        # Check
        failures = check_path(path, max_step, ARRIVAL_PROGRESS, PROGRESS_TOLERANCE, LINEAGE_MAX_STEP)
        entry = {
            "round": round_no,
            "waypoints": [_waypoint_summary(wp, pool) for wp in waypoints],
            "path": _path_summary(path),
            "failed_checks": [f.to_dict() for f in failures],
            "notes": notes,
            "fixes": [],
        }
        trace["rounds"].append(entry)
        logger.info(
            "Round %d: %d steps, failed: %s",
            round_no, len(path) - 1, ", ".join(f"{f.check}@{f.step}" for f in failures) or "none",
        )
        for note in notes:
            logger.warning("Round %d: %s", round_no, note)
        if best is None or len(failures) <= len(best[0]):
            best = (failures, path)
        if not failures:
            break
        if round_no == max_revisions:
            warnings.append(
                f"Hit the revision cap ({max_revisions}); returning the path with the fewest "
                f"failed checks ({len(best[0])})."
            )
            break
        # Revise
        waypoints, fixes = revise(waypoints, path, failures)
        entry["fixes"] = fixes
        for fix in fixes:
            logger.info("Round %d fix: %s", round_no, fix)
        if not fixes:
            warnings.append("Checks failed but no fix was possible; stopping early.")
            break

    for w in warnings:
        logger.warning(w)
    failures, path = best
    trace["result"] = {
        "passed": not failures,
        "revisions": len(trace["rounds"]) - 1,
        "remaining_failures": [f.to_dict() for f in failures],
        "warnings": warnings,
    }
    return AgentResult(path=path, trace=trace, passed=not failures, warnings=warnings)


def _act(pool: _Pool, waypoints: list[Waypoint], start: int, exclude: np.ndarray, max_step: float):
    """Pick the best unused song for each waypoint, in order."""
    used = exclude.copy()
    used[start] = True
    playlist_rows = np.flatnonzero(exclude)  # the playlist's songs (and other releases of them)
    picks, parts, notes = [start], [None], []
    lineage_cols = [(None, None, False)]  # (link text, source url, linked to previous song)
    for i, wp in enumerate(waypoints, start=1):
        allowed = ~used
        if wp.banned_artists:
            allowed &= ~np.isin(pool.artists, list(wp.banned_artists))
        if wp.min_progress is not None:
            allowed &= pool.progress >= wp.min_progress
        if wp.require_destination:
            allowed &= pool.is_destination
        if not allowed.any():
            notes.append(f"step {i}: no song met the constraints {wp.constraints()}; relaxed them")
            allowed = ~used

        # Nearest allowed songs to the waypoint, then the scorer ranks them.
        idx = np.flatnonzero(allowed)
        near = distance(pool.X[idx], wp.vector)
        k = min(CANDIDATES_PER_WAYPOINT, len(idx))
        cand = idx[np.argsort(near, kind="stable")[:k]]

        # Second retrieval source: songs linked by lineage to the previous song or the playlist.
        links = _lineage_candidates(pool, picks[-1], playlist_rows, allowed, max_step)
        if links:
            cand = np.union1d(cand, list(links))

        target_progress = float(pool.compass.progress(wp.vector)[0])
        target_mood = derive_mood(wp.vector[_VALENCE], wp.vector[_ENERGY])
        scored = score_parts(
            pool.X[cand], pool.moods[cand], pool.progress[cand], wp.vector, target_progress, target_mood
        )
        bonus = np.array([LINEAGE_BONUS if int(c) in links else 0.0 for c in cand])
        scored["lineage_bonus"] = bonus
        scored["score"] = scored["score"] + bonus
        j = int(np.argmax(scored["score"]))
        best = int(cand[j])
        used[best] = True
        picks.append(best)
        parts.append({k: float(v[j]) for k, v in scored.items()} | {
            "target_progress": target_progress,
            "target_mood": target_mood,
        })
        lineage_cols.append(links.get(best, (None, None, False)))

    path = pool.catalog.iloc[picks].reset_index(drop=True)
    path.insert(0, "step", range(len(picks)))
    path.insert(1, "kind", ["start"] + [wp.kind for wp in waypoints])
    path["progress"] = pool.progress[picks]
    path["is_destination"] = pool.is_destination[picks]
    X = pool.X[picks]
    path["step_distance"] = np.r_[0.0, distance(X[1:], X[:-1])]
    path["score_parts"] = parts
    # dtype=object keeps None as None (pandas would otherwise turn it into a truthy NaN).
    path["lineage"] = pd.Series([text for text, _, _ in lineage_cols], dtype=object)
    path["lineage_source"] = pd.Series([url for _, url, _ in lineage_cols], dtype=object)
    path["lineage_with_previous"] = [with_prev for _, _, with_prev in lineage_cols]
    return path, notes


def _lineage_candidates(pool: _Pool, prev: int, playlist_rows, allowed: np.ndarray, max_step: float) -> dict:
    """Lineage-linked songs that may fill the next step: row -> (text, source, linked to prev).

    - Linked to the previous song: the shared sample justifies a jump up to LINEAGE_MAX_STEP.
    - Linked to a song in the playlist: a normal-size step (max_step).
    Either way the song must not move backwards on the compass.
    """
    if pool.lineage is None:
        return {}
    found = {}
    sources = [(pool.lineage.neighbours([prev]), True, LINEAGE_MAX_STEP),
               (pool.lineage.neighbours(playlist_rows), False, max_step)]
    for linked, with_prev, limit in sources:
        for row, link in linked.items():
            if row in found or not allowed[row]:
                continue
            if distance(pool.X[row], pool.X[prev]) > limit:
                continue
            if pool.progress[row] < pool.progress[prev] - PROGRESS_TOLERANCE:
                continue
            where = "the previous song" if with_prev else "a song in your playlist"
            found[row] = (f"{link.text}, linked to {where}", link.source_url, with_prev)
    return found


def revise(waypoints: list[Waypoint], path: pd.DataFrame, failures: list[Failure]):
    """Return a new waypoint list with a fix for every failure, plus descriptions of the fixes.

    Path row k (k >= 1) was picked for waypoints[k - 1].
    """
    waypoints = [
        Waypoint(wp.vector, wp.kind, set(wp.banned_artists), wp.min_progress, wp.require_destination)
        for wp in waypoints
    ]
    fixes = []
    bridges = []
    for f in failures:
        wp = waypoints[f.step - 1]
        if f.check == "artist_repeat":
            artist = path.loc[f.step, "artist"]
            wp.banned_artists.add(artist)
            fixes.append(f"step {f.step}: ban {artist} and pick again")
        elif f.check == "progress_dropped":
            floor = float(path.loc[f.step - 1, "progress"])
            wp.min_progress = max(wp.min_progress or 0.0, floor)
            fixes.append(f"step {f.step}: require compass >= {floor:.2f}")
        elif f.check == "no_arrival":
            wp.require_destination = True
            fixes.append(f"step {f.step}: last pick must be a destination song")
        elif f.check == "step_too_big":
            bridges.append(f.step)

    # Insert bridges last, from the back, so earlier indices stay valid.
    X = path[FEATURES].to_numpy(float)
    for step in sorted(bridges, reverse=True):
        if len(waypoints) >= MAX_WAYPOINTS:
            fixes.append(f"step {step}: too big, but already at {MAX_WAYPOINTS} waypoints")
            continue
        midpoint = (X[step - 1] + X[step]) / 2
        waypoints.insert(step - 1, Waypoint(midpoint, kind="bridge"))
        fixes.append(f"step {step}: too big, insert a bridge halfway from step {step - 1}")
    return waypoints, fixes


# --------------------------------------------------------------------------- trace helpers


def _label(row) -> str:
    return f"{row['track_name']} — {row['artist']}"


def _profile_summary(profile: TasteProfile, pool: _Pool, start: int) -> dict:
    return {
        "n_songs": profile.n_songs,
        "top_genres": profile.top_genres,
        "moods": profile.moods,
        "start_song": _label(profile.anchor),
        "start_progress": round(float(pool.progress[start]), 3),
    }


def _waypoint_summary(wp: Waypoint, pool: _Pool) -> dict:
    return {
        "kind": wp.kind,
        "target_progress": round(float(pool.compass.progress(wp.vector)[0]), 3),
        "target_mood": derive_mood(wp.vector[_VALENCE], wp.vector[_ENERGY]),
        **wp.constraints(),
    }


def _path_summary(path: pd.DataFrame) -> list[dict]:
    return [
        {
            "step": int(row["step"]),
            "kind": row["kind"],
            "song": _label(row),
            "genre": row["genre"],
            "subgenre": row["subgenre"],
            "mood": row["mood"],
            "progress": round(float(row["progress"]), 3),
            "step_distance": round(float(row["step_distance"]), 3),
            **({"lineage": row["lineage"], "lineage_source": row["lineage_source"]}
               if row["lineage"] else {}),
        }
        for _, row in path.iterrows()
    ]
