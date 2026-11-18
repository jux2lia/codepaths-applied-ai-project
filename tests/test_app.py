"""Drive the Streamlit app headlessly: pick inputs, press the button, check the output."""

import io

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from src import main as cli
from src.data import CATALOG_PATH, ROOT, load_catalog, load_playlist
from src.viz import build_map, bubble_chart

needs_catalog = pytest.mark.skipif(not CATALOG_PATH.exists(), reason="run `python data/download.py` first")


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "TRACE_DIR", tmp_path / "traces")
    monkeypatch.setattr(cli, "LOG_FILE", tmp_path / "test.log")
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120)
    at.run()
    assert not at.exception
    return at


def build(at, playlist, destination, **sliders):
    at.sidebar.radio[0].set_value(playlist)
    at.sidebar.selectbox[0].set_value(destination)
    for i, (key, value) in enumerate(sliders.items()):
        at.sidebar.slider[{"steps": 0, "max_step": 1}[key]].set_value(value)
    at.sidebar.button[0].click().run()
    assert not at.exception
    return at


@needs_catalog
def test_start_screen_explains_what_to_do(app):
    assert "Build route" in app.info[0].value


@needs_catalog
def test_build_route_shows_metrics_cards_rounds_and_saves_trace(app, tmp_path):
    build(app, "Example: classic rock", "big room")
    metrics = {m.label: m.value for m in app.metric}
    assert metrics["Playlist songs found"] == "15/15"
    assert metrics["Revisions"] == "1"
    assert metrics["Checks"] == "All passed"
    text = " ".join(md.value for md in app.markdown)
    assert "Round 0" in text and "insert a bridge" in text
    assert len(list((tmp_path / "traces").glob("*.json"))) == 1


@needs_catalog
def test_guardrail_warning_is_shown(app):
    build(app, "Example: neo soul", "neo soul")
    assert any("already neo soul" in w.value for w in app.warning)


@needs_catalog
def test_upload_without_file_shows_error(app):
    app.sidebar.radio[0].set_value("Upload an Exportify CSV").run()
    app.sidebar.button[0].click().run()
    assert "Upload an Exportify CSV first" in app.error[0].value


def test_uploaded_file_object_is_accepted():
    csv = "Track URI,Track Name,Artist Name(s)\nspotify:track:" + "A" * 22 + ",Song,Band\n"
    f = io.BytesIO(csv.encode())
    f.name = "upload.csv"
    assert load_playlist(f)["track_id"].tolist() == ["A" * 22]


def test_unreadable_upload_gives_clear_error():
    f = io.BytesIO(b"\xff\xfe\x00garbage")
    f.name = "photo.csv"
    with pytest.raises(ValueError, match="photo.csv"):
        load_playlist(f)


@needs_catalog
def test_bubble_chart_layers_and_deterministic_map():
    catalog = load_catalog()
    a, b = build_map(catalog), build_map(catalog)
    assert np.array_equal(a.coords, b.coords)
    assert "energetic" in a.x_title.split("·")[1]  # more energetic is to the right
    route = catalog.iloc[:3].assign(step=[0, 1, 2], kind=["start", "planned", "planned"],
                                    progress=[0.1, 0.4, 0.8], step_distance=[0, 0.05, 0.05])
    mask = (catalog["genre"] == "rock").to_numpy()
    for mode in ("light", "dark"):
        fig = bubble_chart(a, catalog, set(catalog["track_id"][:10]), route, "rock", mask, mode=mode)
        assert [t.name for t in fig.data] == ["Other songs", "rock songs", "Your playlist", "Route"]
        assert list(fig.data[3].text) == ["0", "1", "2"]
