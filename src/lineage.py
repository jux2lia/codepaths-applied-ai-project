"""Lineage: a second knowledge source next to the audio features.

data/lineage.json lists real sample / interpolation relationships between songs
in the catalog, each with a source (a Wikipedia article quoting the relationship).
The agent retrieves candidates from two places:

1. audio features: the songs nearest each waypoint (agent.py), and
2. lineage: songs that sample, or are sampled by, the previous song on the route
   or a song in the user's playlist (this module).

A lineage link is a musical connection the audio features can't see ("Ice Ice
Baby" and "Under Pressure" share a bassline but are far apart in audio terms), so
the agent allows a bigger jump along one and gives it a scoring bonus.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.data import ROOT, song_key

LINEAGE_PATH = ROOT / "data" / "lineage.json"

VERBS = {
    "sample": "samples",
    "interpolation": "interpolates",
    "remake": "remakes",
    "credited_similarity": "credits the writers of, for melodic similarity,",
}


@dataclass(frozen=True)
class Link:
    other: int  # row of the connected song in the agent's catalog
    text: str  # e.g. '"Ice Ice Baby" (Vanilla Ice) samples "Under Pressure" (Queen)'
    source_url: str


def load_edges(path: Path = LINEAGE_PATH) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))["edges"]


def describe(edge: dict) -> str:
    to, frm = edge["to"], edge["from"]
    verb = VERBS.get(edge["type"], "draws on")
    return f'"{to["title"]}" ({to["artist"]}) {verb} "{frm["title"]}" ({frm["artist"]})'


class LineageGraph:
    """Lineage edges mapped onto catalog rows. A song with several releases
    (single, album, remaster) gets the links on every release."""

    def __init__(self, edges: list[dict], song_keys: np.ndarray):
        rows_by_key: dict[str, list[int]] = defaultdict(list)
        for row, key in enumerate(song_keys):
            rows_by_key[key].append(row)

        self.links: dict[int, list[Link]] = defaultdict(list)
        self.matched_edges = 0
        for edge in edges:
            a = rows_by_key.get(song_key(edge["from"]["title"], edge["from"]["artist"]), [])
            b = rows_by_key.get(song_key(edge["to"]["title"], edge["to"]["artist"]), [])
            if not a or not b:
                continue
            self.matched_edges += 1
            link_text = describe(edge)
            for x in a:  # links work in both directions
                for y in b:
                    self.links[x].append(Link(y, link_text, edge["source_url"]))
                    self.links[y].append(Link(x, link_text, edge["source_url"]))

    def neighbours(self, rows) -> dict[int, Link]:
        """Songs linked to any of `rows`, each with the (first) link that connects it."""
        found: dict[int, Link] = {}
        for row in sorted(set(int(r) for r in rows)):
            for link in self.links.get(row, []):
                found.setdefault(link.other, link)
        return found
