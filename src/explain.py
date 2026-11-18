"""'Why this song?' explanations for each step of a route.

This is the template version: it only states facts the agent actually used
(compass reading, the waypoint's target, mood, why a bridge was added), so it
works without an API key. An LLM version can be added later with this as the
fallback.
"""

from __future__ import annotations

import pandas as pd


def explain_step(row: pd.Series, destination: str, is_last: bool) -> str:
    progress = float(row["progress"])
    if row["kind"] == "start":
        return (
            f"Where you are now: the song in your playlist closest to your overall taste. "
            f"It sounds {progress:.0%} like {destination}."
        )

    parts = row["score_parts"] or {}
    sentences = []
    if row["kind"] == "bridge":
        sentences.append("Bridge song: the agent added this stop because the jump here was too big.")
    if row.get("lineage"):
        sentences.append(f"Lineage: {row['lineage']}.")
    target = parts.get("target_progress")
    if target is not None:
        sentences.append(
            f"Sounds {progress:.0%} like {destination}; this stretch of the route aimed for about {target:.0%}."
        )
    target_mood = parts.get("target_mood")
    if target_mood == row["mood"]:
        sentences.append(f"Keeps the {row['mood']} mood the route calls for here.")
    elif target_mood:
        sentences.append(f"Mood: {row['mood']} (this stretch was heading for {target_mood}).")
    if is_last and row["is_destination"]:
        sentences.append(f"Arrival: a song listed as {destination}.")
    return " ".join(sentences)
