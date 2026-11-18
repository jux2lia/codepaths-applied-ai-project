"""Plan step: resolve the destination and lay out straight-line waypoints to it."""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Destination:
    name: str
    level: str  # "genre" or "subgenre"

    def members(self, catalog: pd.DataFrame) -> np.ndarray:
        """True for every catalog song listed under this genre/subgenre."""
        column = "genres" if self.level == "genre" else "subgenres"
        return catalog[column].map(lambda names: self.name in names).to_numpy(bool)


def resolve_destination(catalog: pd.DataFrame, name: str) -> Destination:
    """Match a genre or subgenre name (case-insensitive); suggest close matches if unknown."""
    key = name.strip().lower()
    genres = sorted({g for names in catalog["genres"] for g in names})
    subgenres = sorted({s for names in catalog["subgenres"] for s in names})
    if key in genres:
        return Destination(key, "genre")
    if key in subgenres:
        return Destination(key, "subgenre")
    close = difflib.get_close_matches(key, genres + subgenres, n=3, cutoff=0.6)
    hint = f" Did you mean: {', '.join(close)}?" if close else ""
    raise ValueError(
        f"Unknown destination '{name}'.{hint}\n"
        f"Genres: {', '.join(genres)}\nSubgenres: {', '.join(subgenres)}"
    )


@dataclass
class Waypoint:
    """A point in audio-feature space the route should pass through, plus any
    constraints on which song may fill it (set by the plan or added by Revise)."""

    vector: np.ndarray
    kind: str = "planned"  # "planned" or "bridge"
    banned_artists: set[str] = field(default_factory=set)
    min_progress: float | None = None
    require_destination: bool = False

    def constraints(self) -> dict:
        out = {}
        if self.banned_artists:
            out["banned_artists"] = sorted(self.banned_artists)
        if self.min_progress is not None:
            out["min_progress"] = round(float(self.min_progress), 3)
        if self.require_destination:
            out["require_destination"] = True
        return out


def plan_waypoints(start: np.ndarray, end: np.ndarray, n_steps: int) -> list[Waypoint]:
    """n_steps evenly spaced points from start (exclusive) to end (inclusive)."""
    if n_steps < 1:
        raise ValueError("A route needs at least one step.")
    start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    return [Waypoint(start + (end - start) * i / n_steps) for i in range(1, n_steps + 1)]
