"""Act step: score every candidate song against one waypoint on the route.

The Module 3 scorer ranked songs against one fixed profile:

    score = genre (0 or 2) + mood (0 or 1) + (1 - |energy - target energy|)

Here the target moves along the route, so each term is generalised:

    score = 2.0 * genre_fit + 1.0 * mood_fit + feature_closeness

- genre_fit: 1 - |song's compass reading - waypoint's compass reading|. The compass
  (see compass.py) says how much a song sounds like the destination, from 0 to 1.
  Early waypoints want songs that only lean a little toward the destination; the
  last ones want songs that clearly belong to it.
- mood_fit: 1 if the song's mood matches the mood at the waypoint, else 0.
- feature_closeness: 1 - distance over all audio features (was: energy only).
"""

from __future__ import annotations

import numpy as np

GENRE_WEIGHT = 2.0
MOOD_WEIGHT = 1.0


def distance(a, b) -> np.ndarray:
    """Root-mean-square difference across features (0 = identical, 1 = opposite)."""
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return np.sqrt(np.mean(diff**2, axis=-1))


def score_parts(features, moods, progress, target_vector, target_progress, target_mood) -> dict:
    """Each term of the score, for every candidate (arrays of length n)."""
    genre_fit = 1.0 - np.abs(np.asarray(progress, dtype=float) - target_progress)
    mood_fit = (np.asarray(moods) == target_mood).astype(float)
    closeness = 1.0 - distance(features, target_vector)
    return {
        "genre_fit": genre_fit,
        "mood_fit": mood_fit,
        "closeness": closeness,
        "score": GENRE_WEIGHT * genre_fit + MOOD_WEIGHT * mood_fit + closeness,
    }


def score_songs(features, moods, progress, target_vector, target_progress, target_mood) -> np.ndarray:
    return score_parts(features, moods, progress, target_vector, target_progress, target_mood)["score"]
