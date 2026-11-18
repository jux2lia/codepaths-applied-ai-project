"""Phase 1 data layer: load the song catalog and match a Spotify playlist to it.

Run `python -m src.data` (or `python -m src.data path/to/playlist.csv`) to print a
catalog summary and the playlist's match rate.
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "data" / "raw" / "spotify_songs.csv"
PLAYLIST_PATH = ROOT / "data" / "my_playlist.csv"

# Source column -> the name the rest of the project uses.
COLUMN_ALIASES = {
    "track_artist": "artist",
    "playlist_genre": "genre",
    "playlist_subgenre": "subgenre",
}

# Audio features that Spotify already reports on a 0-1 scale.
UNIT_FEATURES = [
    "danceability",
    "energy",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
]
# Features on other scales (BPM, decibels) that get rescaled to 0-1.
SCALED_FEATURES = ["tempo", "loudness"]
FEATURES = UNIT_FEATURES + SCALED_FEATURES

# Clip to these quantiles before min-max scaling, so a single outlier
# (e.g. one track at -46 dB) doesn't squash everyone else into a narrow band.
CLIP_QUANTILES = (0.01, 0.99)

MOOD_THRESHOLD = 0.5

# Exportify column -> our column name.
EXPORTIFY_COLUMNS = {
    "Track URI": "uri",
    "Track Name": "track_name",
    "Artist Name(s)": "artists",
}


# --------------------------------------------------------------------------- catalog


def load_catalog(path: Path = CATALOG_PATH) -> pd.DataFrame:
    """Load the raw catalog CSV and return it cleaned, deduped, scaled and mood-labelled."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Catalog not found at {path}. Run `python data/download.py` to fetch it."
        )
    return prepare_catalog(pd.read_csv(path))


def prepare_catalog(raw: pd.DataFrame) -> pd.DataFrame:
    """Turn the raw catalog table into one clean row per track."""
    df = raw.rename(columns=COLUMN_ALIASES)
    df = df.dropna(subset=["track_id", "track_name", "artist", *FEATURES])
    df = dedupe(df)
    # Mood uses the raw 0-1 valence/energy; keep human-readable tempo/loudness too.
    df["mood"] = [derive_mood(v, e) for v, e in zip(df["valence"], df["energy"])]
    df["tempo_bpm"] = df["tempo"]
    df["loudness_db"] = df["loudness"]
    return scale_features(df)


def dedupe(df: pd.DataFrame) -> pd.DataFrame:
    """Keep one row per track_id.

    The dataset lists a track once per playlist it appeared in, so the same song can
    show up under several genres. We keep the row for the genre it appears under most
    often (ties go to whichever was listed first) and store every genre and subgenre
    in the `genres` / `subgenres` list columns.
    """
    grouped = df.groupby("track_id", sort=False)
    primary = grouped["genre"].agg(_most_common).rename("_primary")
    genres = grouped["genre"].agg(_unique_in_order)
    subgenres = grouped["subgenre"].agg(_unique_in_order)

    out = df.join(primary, on="track_id")
    out = out[out["genre"] == out["_primary"]].drop_duplicates("track_id")
    out = out.drop(columns="_primary").reset_index(drop=True)
    out["genres"] = out["track_id"].map(genres)
    out["subgenres"] = out["track_id"].map(subgenres)
    return out


def scale_features(df: pd.DataFrame, columns: list[str] = SCALED_FEATURES) -> pd.DataFrame:
    """Rescale `columns` to 0-1: clip to CLIP_QUANTILES, then min-max."""
    out = df.copy()
    for col in columns:
        lo, hi = out[col].quantile(list(CLIP_QUANTILES))
        if hi <= lo:  # constant column: nothing to spread out
            out[col] = 0.0
            continue
        out[col] = (out[col].clip(lo, hi) - lo) / (hi - lo)
    return out


def derive_mood(valence: float, energy: float) -> str:
    """Rough mood label from Spotify's valence (positivity) and energy.

    Four quadrants: happy (positive, energetic), calm (positive, low energy),
    angry (negative, energetic), sad (negative, low energy). These are coarse —
    a low-valence, high-energy song isn't always literally "angry".
    """
    if valence >= MOOD_THRESHOLD:
        return "happy" if energy >= MOOD_THRESHOLD else "calm"
    return "angry" if energy >= MOOD_THRESHOLD else "sad"


def _most_common(values: pd.Series):
    # Counter keeps first-seen order for ties, so this is deterministic.
    return Counter(values).most_common(1)[0][0]


def _unique_in_order(values: pd.Series) -> list:
    return list(dict.fromkeys(values))


# --------------------------------------------------------------------------- playlist

_TRACK_ID = re.compile(
    r"(?:spotify:track:|open\.spotify\.com/track/|^)([A-Za-z0-9]{22})(?![A-Za-z0-9])"
)


def parse_track_id(uri) -> str | None:
    """'spotify:track:XXXX', an open.spotify.com URL, or a bare ID -> 'XXXX'.

    Returns None for anything else (e.g. local files: 'spotify:local:...').
    """
    if not isinstance(uri, str):
        return None
    match = _TRACK_ID.search(uri.strip())
    return match.group(1) if match else None


def load_playlist(source=PLAYLIST_PATH) -> pd.DataFrame:
    """Load an Exportify CSV as columns: track_id, track_name, artists.

    `source` is a path, or a file object such as a Streamlit upload.
    """
    if hasattr(source, "read"):
        name = Path(getattr(source, "name", "uploaded file")).name
    else:
        source = Path(source)
        name = source.name
        if not source.exists():
            raise FileNotFoundError(
                f"Playlist not found at {source}. Export one with Exportify and save it there."
            )
    try:
        raw = pd.read_csv(source)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as err:
        raise ValueError(f"Couldn't read {name} as a CSV file ({type(err).__name__}).") from err
    missing = [c for c in EXPORTIFY_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(
            f"{name} doesn't look like an Exportify export; missing columns: {missing}"
        )
    df = raw[list(EXPORTIFY_COLUMNS)].rename(columns=EXPORTIFY_COLUMNS)
    df["track_id"] = df["uri"].map(parse_track_id)
    return df[["track_id", "track_name", "artists"]]


@dataclass
class PlaylistMatch:
    matched: pd.DataFrame  # catalog rows for the playlist's songs, plus a `matched_by` column
    unmatched: pd.DataFrame  # playlist rows we couldn't find in the catalog
    total: int  # distinct songs in the playlist

    @property
    def matched_count(self) -> int:
        return self.total - len(self.unmatched)

    @property
    def match_rate(self) -> float:
        return self.matched_count / self.total if self.total else 0.0


def match_playlist(playlist: pd.DataFrame, catalog: pd.DataFrame) -> PlaylistMatch:
    """Find each playlist song in the catalog.

    1. By Spotify track ID.
    2. Otherwise by normalised title + artist. The same song often has several IDs
       (single vs. album vs. remaster), so the playlist may hold a different ID
       than the catalog. If several catalog rows fit, the most popular one wins.
    """
    pl = playlist.drop_duplicates(["track_id", "track_name", "artists"]).reset_index(drop=True)
    pl["catalog_id"] = pl["track_id"].where(pl["track_id"].isin(set(catalog["track_id"])))
    pl["matched_by"] = pl["catalog_id"].notna().map({True: "id", False: None})

    rest = pl[pl["catalog_id"].isna()]
    if not rest.empty:
        left = rest.assign(
            _title=rest["track_name"].map(_normalize_title),
            _artists=rest["artists"].map(_normalize_text),
        ).reset_index(names="_row")
        right = catalog[["track_id", "track_name", "artist", "track_popularity"]].assign(
            _title=catalog["track_name"].map(_normalize_title),
            _artist=catalog["artist"].map(_normalize_text),
        )
        cand = left[left["_title"] != ""].merge(right, on="_title", suffixes=("", "_cat"))
        # Exportify joins multiple artists into one string; the catalog lists the main one.
        same_artist = [a.startswith(b) for a, b in zip(cand["_artists"], cand["_artist"])]
        cand = cand[pd.Series(same_artist, index=cand.index, dtype=bool)]
        cand = cand.sort_values("track_popularity", ascending=False).drop_duplicates("_row")
        pl.loc[cand["_row"], "catalog_id"] = cand["track_id_cat"].to_numpy()
        pl.loc[cand["_row"], "matched_by"] = "name"

    found = pl[pl["catalog_id"].notna()]
    matched = (
        found[["catalog_id", "matched_by"]]
        .drop_duplicates("catalog_id")
        .merge(catalog, left_on="catalog_id", right_on="track_id")
        .drop(columns="catalog_id")
    )
    unmatched = pl[pl["catalog_id"].isna()][["track_id", "track_name", "artists"]]
    return PlaylistMatch(matched=matched, unmatched=unmatched.reset_index(drop=True), total=len(pl))


def song_key(title, artist) -> str:
    """Same song, different release (single/album/remaster) -> same key."""
    return f"{_normalize_title(title)}|{_normalize_text(artist)}"


def _normalize_text(text) -> str:
    """Lowercase, punctuation -> spaces. Keeps accented letters ('canción')."""
    return re.sub(r"[\W_]+", " ", str(text).lower()).strip()


def _normalize_title(title) -> str:
    """Drop '(feat. X)', '[Remastered]' and ' - Radio Edit' style suffixes."""
    title = re.sub(r"\s*[\(\[].*?[\)\]]", "", str(title))
    title = title.split(" - ")[0]
    return _normalize_text(title)


# --------------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # song titles on Windows
    argv = sys.argv[1:] if argv is None else argv
    playlist_path = Path(argv[0]) if argv else PLAYLIST_PATH

    catalog = load_catalog()
    print(
        f"Catalog: {len(catalog):,} unique tracks, "
        f"{catalog['genre'].nunique()} genres, {catalog['subgenre'].nunique()} subgenres"
    )
    print("Genres:", ", ".join(sorted(catalog["genre"].unique())))
    print("Moods: ", catalog["mood"].value_counts().to_dict())

    if not playlist_path.exists():
        print(f"\nNo playlist at {playlist_path} — export one with Exportify and save it there.")
        return

    result = match_playlist(load_playlist(playlist_path), catalog)
    by = result.matched["matched_by"].value_counts().to_dict()
    print(
        f"\nPlaylist match rate: {result.match_rate:.0%} "
        f"({result.matched_count}/{result.total} songs; "
        f"{by.get('id', 0)} by ID, {by.get('name', 0)} by title+artist)"
    )
    if not result.unmatched.empty:
        print(f"Unmatched ({len(result.unmatched)}), first 10:")
        for row in result.unmatched.head(10).itertuples():
            print(f"  - {row.track_name} — {row.artists}")


if __name__ == "__main__":
    main()
