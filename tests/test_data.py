import pandas as pd
import pytest

from src.data import (
    CATALOG_PATH,
    FEATURES,
    derive_mood,
    load_catalog,
    load_playlist,
    match_playlist,
    parse_track_id,
    prepare_catalog,
)


def _id(n: int) -> str:
    """A fake 22-character Spotify track ID."""
    return f"{n:022d}"


def raw_row(track_id, name, artist, genre, subgenre="sub", popularity=50, **features):
    row = {
        "track_id": track_id,
        "track_name": name,
        "track_artist": artist,
        "track_popularity": popularity,
        "playlist_genre": genre,
        "playlist_subgenre": subgenre,
        "danceability": 0.5,
        "energy": 0.5,
        "speechiness": 0.1,
        "acousticness": 0.2,
        "instrumentalness": 0.0,
        "liveness": 0.1,
        "valence": 0.5,
        "tempo": 120.0,
        "loudness": -6.0,
    }
    row.update(features)
    return row


@pytest.fixture
def catalog():
    rows = [
        # Same track listed under three playlists: rock twice, pop once.
        raw_row(_id(1), "Song A", "Band", "pop", "dance pop", tempo=90.0, loudness=-40.0),
        raw_row(_id(1), "Song A", "Band", "rock", "hard rock", tempo=90.0, loudness=-40.0),
        raw_row(_id(1), "Song A", "Band", "rock", "album rock", tempo=90.0, loudness=-40.0),
        raw_row(_id(2), "Yonkers", "Tyler, The Creator", "rap", tempo=170.0, loudness=0.5),
        raw_row(_id(3), "Hey Jude - Remastered 2015", "The Beatles", "rock", popularity=80),
        raw_row(_id(4), "Hey Jude", "The Beatles", "rock", popularity=40),
        raw_row(_id(5), None, "Nameless", "pop"),  # dropped: missing title
    ]
    return prepare_catalog(pd.DataFrame(rows))


# --------------------------------------------------------------------------- mood


@pytest.mark.parametrize(
    "valence, energy, mood",
    [
        (0.9, 0.9, "happy"),
        (0.9, 0.1, "calm"),
        (0.1, 0.9, "angry"),
        (0.1, 0.1, "sad"),
        (0.5, 0.5, "happy"),  # the threshold counts as "high"
    ],
)
def test_derive_mood_quadrants(valence, energy, mood):
    assert derive_mood(valence, energy) == mood


# --------------------------------------------------------------------------- catalog


def test_no_duplicate_track_ids(catalog):
    assert catalog["track_id"].is_unique


def test_rows_missing_a_title_are_dropped(catalog):
    assert _id(5) not in set(catalog["track_id"])


def test_dedupe_keeps_most_common_genre_and_lists_all(catalog):
    song = catalog.set_index("track_id").loc[_id(1)]
    assert song["genre"] == "rock"
    assert song["subgenre"] == "hard rock"  # first rock row, not the pop row's subgenre
    assert song["genres"] == ["pop", "rock"]
    assert song["subgenres"] == ["dance pop", "hard rock", "album rock"]


def test_all_features_between_0_and_1(catalog):
    values = catalog[FEATURES]
    assert ((values >= 0) & (values <= 1)).all().all()


def test_raw_tempo_and_loudness_are_kept(catalog):
    song = catalog.set_index("track_id").loc[_id(2)]
    assert song["tempo_bpm"] == 170.0
    assert song["loudness_db"] == 0.5


def test_every_track_has_a_mood(catalog):
    assert catalog["mood"].isin({"happy", "calm", "angry", "sad"}).all()


# --------------------------------------------------------------------------- playlist


@pytest.mark.parametrize(
    "uri, expected",
    [
        (f"spotify:track:{_id(7)}", _id(7)),
        (f"https://open.spotify.com/track/{_id(7)}?si=abc123", _id(7)),
        (_id(7), _id(7)),
        ("spotify:local:Artist:Album:Title:200", None),
        (None, None),
        (float("nan"), None),
    ],
)
def test_parse_track_id(uri, expected):
    assert parse_track_id(uri) == expected


def playlist(*rows):
    return pd.DataFrame(rows, columns=["track_id", "track_name", "artists"])


def test_match_by_id(catalog):
    result = match_playlist(playlist((_id(1), "Song A", "Band")), catalog)
    assert result.match_rate == 1.0
    assert result.matched["matched_by"].tolist() == ["id"]


def test_match_by_title_and_artist_when_id_differs(catalog):
    # Album version in the playlist, different IDs in the catalog: take the most popular.
    result = match_playlist(playlist((_id(99), "Hey Jude - 2018 Mix", "The Beatles")), catalog)
    assert result.matched["track_id"].tolist() == [_id(3)]
    assert result.matched["matched_by"].tolist() == ["name"]


def test_artist_name_with_comma_and_featured_artist(catalog):
    # Exportify separates multiple artists with ";".
    pl = playlist((_id(98), "Yonkers", "Tyler, The Creator;Frank Ocean"))
    assert match_playlist(pl, catalog).matched["track_id"].tolist() == [_id(2)]


def test_same_title_by_another_artist_does_not_match(catalog):
    result = match_playlist(playlist((_id(97), "Hey Jude", "Some Cover Band")), catalog)
    assert result.match_rate == 0.0
    assert result.unmatched["track_name"].tolist() == ["Hey Jude"]


def test_match_rate_counts_duplicates_and_local_files_once(catalog):
    pl = playlist(
        (_id(1), "Song A", "Band"),
        (_id(1), "Song A", "Band"),  # added twice
        (None, "Bootleg", "Unknown"),  # local file, no ID
        (_id(96), "Not In Catalog", "Nobody"),
    )
    result = match_playlist(pl, catalog)
    assert result.total == 3
    assert result.matched_count == 1
    assert result.match_rate == pytest.approx(1 / 3)


def test_empty_playlist(catalog):
    result = match_playlist(playlist(), catalog)
    assert result.total == 0 and result.match_rate == 0.0


def test_load_playlist_reads_exportify(tmp_path):
    path = tmp_path / "pl.csv"
    pd.DataFrame(
        {
            "Track URI": [f"spotify:track:{_id(1)}"],
            "Track Name": ["Song A"],
            "Album Name": ["Album"],
            "Artist Name(s)": ["Band"],
        }
    ).to_csv(path, index=False)
    assert load_playlist(path).to_dict("records") == [
        {"track_id": _id(1), "track_name": "Song A", "artists": "Band"}
    ]


def test_load_playlist_rejects_other_csvs(tmp_path):
    path = tmp_path / "pl.csv"
    pd.DataFrame({"title": ["x"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Exportify"):
        load_playlist(path)


# --------------------------------------------------------------------------- real data


@pytest.mark.skipif(not CATALOG_PATH.exists(), reason="run `python data/download.py` first")
def test_real_catalog_loads():
    df = load_catalog()
    assert df["track_id"].is_unique
    assert len(df) > 28_000
    assert df["genre"].nunique() == 6
    values = df[FEATURES]
    assert ((values >= 0) & (values <= 1)).all().all()
