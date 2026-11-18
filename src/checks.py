"""Check step: test a finished path and report everything wrong with it.

The path is a table with one row per song: row 0 is the start (the user's own
song), rows 1..N are the agent's picks. Columns used: artist, progress,
is_destination, step_distance.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd


@dataclass
class Failure:
    check: str  # step_too_big | progress_dropped | no_arrival | artist_repeat
    step: int  # path row of the offending song
    detail: str

    def to_dict(self) -> dict:
        return asdict(self)


def check_step_sizes(
    path: pd.DataFrame, max_step: float, lineage_max_step: float | None = None
) -> list[Failure]:
    """A step may be bigger than max_step only if it follows a sample link from the
    previous song (column lineage_with_previous), up to lineage_max_step."""
    linked = path["lineage_with_previous"] if "lineage_with_previous" in path else [False] * len(path)
    failures = []
    for step, (d, with_prev) in enumerate(zip(path["step_distance"], linked)):
        limit = lineage_max_step if (with_prev and lineage_max_step is not None) else max_step
        if step > 0 and d > limit:
            kind = "sample-linked jump" if limit != max_step else "jump"
            failures.append(Failure("step_too_big", step, f"{kind} of {d:.3f} > max {limit:.3f}"))
    return failures


def check_progress(path: pd.DataFrame, tolerance: float) -> list[Failure]:
    p = path["progress"].to_numpy()
    return [
        Failure("progress_dropped", step, f"compass fell {p[step - 1]:.2f} -> {p[step]:.2f}")
        for step in range(1, len(p))
        if p[step] < p[step - 1] - tolerance
    ]


def check_arrival(path: pd.DataFrame, min_progress: float) -> list[Failure]:
    last = path.iloc[-1]
    if last["is_destination"] and last["progress"] >= min_progress:
        return []
    why = "not listed in the destination" if not last["is_destination"] else (
        f"compass {last['progress']:.2f} < {min_progress:.2f}"
    )
    return [Failure("no_arrival", len(path) - 1, f"last song doesn't arrive: {why}")]


def check_artists(path: pd.DataFrame) -> list[Failure]:
    seen: dict[str, int] = {}
    failures = []
    for step, artist in enumerate(path["artist"]):
        if artist in seen:
            failures.append(Failure("artist_repeat", step, f"{artist} already at step {seen[artist]}"))
        else:
            seen[artist] = step
    return failures


def check_path(
    path: pd.DataFrame,
    max_step: float,
    arrival_progress: float,
    progress_tolerance: float,
    lineage_max_step: float | None = None,
) -> list[Failure]:
    return (
        check_artists(path)
        + check_step_sizes(path, max_step, lineage_max_step)
        + check_progress(path, progress_tolerance)
        + check_arrival(path, arrival_progress)
    )
