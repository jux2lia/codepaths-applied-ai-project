"""Taste profile: summarise the playlist songs that matched the catalog."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data import FEATURES
from src.scorer import distance


@dataclass
class TasteProfile:
    centroid: np.ndarray  # average audio features of the playlist
    anchor: pd.Series  # the playlist song closest to the centroid; routes start here
    top_genres: dict[str, int]
    moods: dict[str, int]
    n_songs: int


def build_profile(matched: pd.DataFrame) -> TasteProfile:
    """Build a profile from the catalog rows of the user's matched playlist songs."""
    if matched.empty:
        raise ValueError("None of the playlist's songs are in the catalog, so there's no taste to start from.")
    X = matched[FEATURES].to_numpy(float)
    centroid = X.mean(axis=0)
    anchor = matched.iloc[int(np.argmin(distance(X, centroid)))]
    return TasteProfile(
        centroid=centroid,
        anchor=anchor,
        top_genres=matched["genre"].value_counts().to_dict(),
        moods=matched["mood"].value_counts().to_dict(),
        n_songs=len(matched),
    )
