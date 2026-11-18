"""The bubble chart: every catalog song on a 2-D map, with the playlist and route on top.

The map is a PCA projection of the 9 audio features. It's deterministic (full SVD,
signs fixed so "more energetic" is right and "happier" is up), so the same song
always lands in the same place.

Colours are the reference data-viz palette's first three categorical slots, which
are validated for all-pairs use (scatter) in both light and dark mode. The route
colour (aqua) is below 3:1 contrast on the light surface, so route bubbles carry
direct step-number labels and the app shows a table view of the route.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.decomposition import PCA

from src.data import FEATURES

SEED = 42  # PCA with a full SVD is deterministic anyway; fixed for the record

PALETTES = {
    "light": {
        "surface": "#fcfcfb", "ink": "#0b0b0b", "ink_2": "#52514e", "muted": "#898781",
        "grid": "#e1e0d9", "background_song": "#c3c2b7",
        "playlist": "#2a78d6", "destination": "#eb6834", "route": "#1baf7a",
    },
    "dark": {
        "surface": "#1a1a19", "ink": "#f0efec", "ink_2": "#c3c2b7", "muted": "#898781",
        "grid": "#2c2c2a", "background_song": "#52514e",
        "playlist": "#3987e5", "destination": "#d95926", "route": "#199e70",
    },
}

# Plain-language names for axis titles.
FEATURE_WORDS = {
    "danceability": "danceable", "energy": "energetic", "speechiness": "spoken",
    "acousticness": "acoustic", "instrumentalness": "instrumental", "liveness": "live",
    "valence": "happy", "tempo": "fast", "loudness": "loud",
}


@dataclass
class SongMap:
    pca: PCA
    signs: np.ndarray  # +1/-1 per axis, so the axes point the intuitive way
    coords: np.ndarray  # (n_catalog, 2), aligned with the catalog's row order
    x_title: str
    y_title: str

    def project(self, features) -> np.ndarray:
        return self.pca.transform(np.atleast_2d(features)) * self.signs


def build_map(catalog: pd.DataFrame) -> SongMap:
    X = catalog[FEATURES].to_numpy(float)
    pca = PCA(n_components=2, svd_solver="full", random_state=SEED).fit(X)
    loadings = pca.components_.copy()
    # Orient: energy increases to the right, valence (happiness) increases upward.
    signs = np.array([
        np.sign(loadings[0, FEATURES.index("energy")]) or 1.0,
        np.sign(loadings[1, FEATURES.index("valence")]) or 1.0,
    ])
    loadings *= signs[:, None]
    return SongMap(
        pca=pca,
        signs=signs,
        coords=pca.transform(X) * signs,
        x_title=_axis_title(loadings[0], "←", "→"),
        y_title=_axis_title(loadings[1], "↓", "↑"),
    )


def _axis_title(loading: np.ndarray, low_arrow: str, high_arrow: str) -> str:
    def words(order):
        # The strongest feature, plus the runner-up only if it's nearly as strong.
        first, second = order[0], order[1]
        top = [first] + ([second] if abs(loading[second]) >= 0.8 * abs(loading[first]) else [])
        return ", ".join(FEATURE_WORDS[FEATURES[i]] for i in top)
    high = words(np.argsort(-loading))
    low = words(np.argsort(loading))
    return f"{low_arrow} more {low}   ·   more {high} {high_arrow}"


def bubble_chart(
    song_map: SongMap,
    catalog: pd.DataFrame,
    playlist_ids: set[str],
    route: pd.DataFrame,
    destination_name: str,
    destination_mask: np.ndarray,
    mode: str = "light",
) -> go.Figure:
    c = PALETTES.get(mode, PALETTES["light"])
    xy = song_map.coords
    in_playlist = catalog["track_id"].isin(playlist_ids).to_numpy()
    other = ~in_playlist & ~destination_mask

    fig = go.Figure()
    # 1. Everything else: quiet context, no hover.
    fig.add_trace(go.Scattergl(
        x=xy[other, 0], y=xy[other, 1], mode="markers", name="Other songs",
        marker={"size": 4, "color": c["background_song"], "opacity": 0.35},
        hoverinfo="skip",
    ))
    # 2. The destination genre.
    dest = catalog[destination_mask]
    fig.add_trace(go.Scattergl(
        x=xy[destination_mask, 0], y=xy[destination_mask, 1], mode="markers",
        name=f"{destination_name} songs",
        marker={"size": 6, "color": c["destination"], "opacity": 0.45},
        customdata=np.c_[dest["track_name"], dest["artist"], dest["subgenre"]],
        hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}<br>%{customdata[2]}<extra></extra>",
    ))
    # 3. The user's playlist. (All layers are WebGL: WebGL draws above SVG, so mixing
    # the two would bury the route under the background songs.)
    mine = catalog[in_playlist]
    fig.add_trace(go.Scattergl(
        x=xy[in_playlist, 0], y=xy[in_playlist, 1], mode="markers", name="Your playlist",
        marker={"size": 10, "color": c["playlist"], "line": {"width": 2, "color": c["surface"]}},
        customdata=np.c_[mine["track_name"], mine["artist"], mine["subgenre"]],
        hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}<br>%{customdata[2]}"
                      "<extra>your playlist</extra>",
    ))
    # 4. The route, numbered.
    rxy = song_map.project(route[FEATURES].to_numpy(float))
    fig.add_trace(go.Scattergl(
        x=rxy[:, 0], y=rxy[:, 1], mode="lines+markers+text", name="Route",
        line={"width": 2, "color": c["route"]},
        marker={"size": 16, "color": c["route"], "line": {"width": 2, "color": c["surface"]}},
        text=[str(s) for s in route["step"]], textposition="top center",
        textfont={"color": c["ink"], "size": 13},
        customdata=np.c_[
            route["step"], route["kind"], route["track_name"], route["artist"], route["subgenre"],
            route["mood"], route["progress"].round(2), route["step_distance"].round(3),
        ],
        hovertemplate=(
            "<b>Step %{customdata[0]} · %{customdata[1]}</b><br>%{customdata[2]} — %{customdata[3]}"
            "<br>%{customdata[4]}, %{customdata[5]}<br>compass %{customdata[6]} · jump %{customdata[7]}"
            "<extra></extra>"
        ),
    ))

    # Start zoomed in on the route (its 9-D steps look short in 2-D); double-click
    # the chart to see the whole catalog.
    x_range, y_range = _zoom(rxy, xy)
    axis = {
        "showgrid": True, "gridcolor": c["grid"], "gridwidth": 1, "zeroline": False,
        "showticklabels": False, "linecolor": c["grid"],
        "title": {"font": {"color": c["ink_2"], "size": 13}},
    }
    fig.update_layout(
        height=600,
        margin={"l": 40, "r": 16, "t": 48, "b": 48},
        paper_bgcolor=c["surface"], plot_bgcolor=c["surface"],
        font={"family": 'system-ui, -apple-system, "Segoe UI", sans-serif', "color": c["ink_2"]},
        legend={"orientation": "h", "y": 1.02, "yanchor": "bottom", "x": 0, "font": {"color": c["ink"]},
                "itemsizing": "constant"},
        hoverlabel={"bgcolor": c["surface"], "font": {"color": c["ink"]}, "bordercolor": c["grid"]},
        xaxis={**axis, "title": {**axis["title"], "text": song_map.x_title}, "range": x_range},
        yaxis={**axis, "title": {**axis["title"], "text": song_map.y_title}, "range": y_range,
               "scaleanchor": "x", "scaleratio": 1},
    )
    return fig


def _zoom(route_xy: np.ndarray, all_xy: np.ndarray, pad: float = 0.6, min_share: float = 0.3):
    """Axis ranges around the route: its bounding box plus padding, never smaller
    than min_share of the full map, so there's always some context around it."""
    ranges = []
    for axis in (0, 1):
        full = all_xy[:, axis].max() - all_xy[:, axis].min()
        lo, hi = route_xy[:, axis].min(), route_xy[:, axis].max()
        span = max((hi - lo) * (1 + 2 * pad), full * min_share)
        mid = (lo + hi) / 2
        ranges.append([float(mid - span / 2), float(mid + span / 2)])
    return ranges
