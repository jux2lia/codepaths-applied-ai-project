"""Genre compass: how much does a song sound like the destination? (0 to 1)

A logistic regression on the audio features, trained to tell destination songs
apart from everything else. Balanced class weights mean 0.5 is "could go either way"
no matter how rare the destination is. The agent uses it as its progress meter.

Because the model is linear in the features, its reading rises smoothly along a
straight line toward the destination, which is what the waypoints follow.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


class GenreCompass:
    def __init__(self, features: np.ndarray, is_destination: np.ndarray):
        is_destination = np.asarray(is_destination, dtype=bool)
        if is_destination.all() or not is_destination.any():
            raise ValueError("The compass needs both destination and non-destination songs.")
        self.model = LogisticRegression(class_weight="balanced", max_iter=1000)
        self.model.fit(features, is_destination)

    def progress(self, features) -> np.ndarray:
        """Probability that each song (row) belongs to the destination."""
        return self.model.predict_proba(np.atleast_2d(features))[:, 1]
