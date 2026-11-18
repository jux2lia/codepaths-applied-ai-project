"""Bubble Breaker — Streamlit app.

    streamlit run app.py

Pick a playlist and a destination genre; the agent builds a gradual route of songs
from your taste to that genre, checks it, and repairs it. This calls the same
guarded agent as the CLI (src/main.py), and saves the same trace files.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from src import agent
from src import main as cli
from src.data import PLAYLIST_PATH, ROOT, load_catalog, load_playlist
from src.explain import explain_step
from src.guardrails import MAX_MAX_STEP, MAX_STEPS, MIN_MAX_STEP, MIN_STEPS, run_with_guardrails
from src.planner import resolve_destination
from src.viz import build_map, bubble_chart

st.set_page_config(page_title="Bubble Breaker", page_icon="🫧", layout="wide")

UPLOAD = "Upload an Exportify CSV"


@st.cache_resource(show_spinner="Loading the song catalog…")
def load_data():
    catalog = load_catalog()
    return catalog, build_map(catalog)


@st.cache_resource
def start_logging():
    cli.setup_logging(cli.LOG_FILE)


def playlist_sources() -> dict:
    sources = {}
    if PLAYLIST_PATH.exists():
        sources["My playlist"] = PLAYLIST_PATH
    sources["Example: neo soul"] = ROOT / "examples" / "neo_soul.csv"
    sources["Example: classic rock"] = ROOT / "examples" / "classic_rock.csv"
    sources[UPLOAD] = None
    return sources


def build_route(source, destination: str, steps: int, max_step: float, use_lineage: bool, catalog) -> None:
    """Run the guarded agent and keep the result in session state."""
    name = Path(source.name).name
    started = datetime.now()
    try:
        run = run_with_guardrails(catalog, load_playlist(source), destination, steps, max_step,
                                  use_lineage=use_lineage)
    except (ValueError, FileNotFoundError) as err:
        st.session_state.update(run=None, error=str(err))
        return
    trace = cli.build_trace(run, started, name, {"to": destination, "steps": steps,
                                                 "max_step": max_step, "lineage": use_lineage})
    path = cli.save_trace(trace, cli.TRACE_DIR, started, Path(name).stem, destination)
    st.session_state.update(run=run, trace=trace, trace_path=path, error=None)


# --------------------------------------------------------------------------- inputs

start_logging()
catalog, song_map = load_data()
genres = sorted({g for names in catalog["genres"] for g in names})
subgenres = sorted({s for names in catalog["subgenres"] for s in names})

with st.sidebar:
    st.header("Plan a route")
    sources = playlist_sources()
    choice = st.radio("Start from", list(sources))
    upload = st.file_uploader("Exportify CSV", type="csv") if choice == UPLOAD else None
    options = genres + subgenres
    destination = st.selectbox(
        "Destination", options, index=options.index("hard rock"),
        format_func=lambda o: f"{o} (genre)" if o in genres else o,
        help="Any of the 6 genres or 24 subgenres in the catalog.",
    )
    steps = st.slider(
        "Planned steps", MIN_STEPS, MAX_STEPS, agent.N_STEPS,
        help="Steps in the first plan. The agent adds bridge songs if a jump is too big.",
    )
    max_step = st.slider(
        "Max jump between songs", MIN_MAX_STEP, MAX_MAX_STEP, agent.MAX_STEP, step=0.01,
        help="How different two songs in a row may be (0 = identical). Smaller = smoother but harder.",
    )
    use_lineage = st.toggle(
        "Use song lineage", value=True,
        help="Also consider songs that sample (or are sampled by) the previous song or a song in "
             "your playlist, from data/lineage.json. Off = audio features only.",
    )
    if st.button("Build route", type="primary", width="stretch"):
        source = sources[choice] if choice != UPLOAD else upload
        if source is None:
            st.session_state.update(run=None, error="Upload an Exportify CSV first.")
        else:
            with st.spinner("Planning, checking and revising…"):
                build_route(source, destination, steps, max_step, use_lineage, catalog)

# --------------------------------------------------------------------------- output

st.title("🫧 Bubble Breaker")
st.write(
    "A gradual route from the music you know to a genre you don't: each song is only a small "
    "step from the last. The agent plans the route, checks it for jarring jumps, backtracking and "
    "repeated artists, and repairs what fails."
)

if st.session_state.get("error"):
    st.error(st.session_state["error"])
run = st.session_state.get("run")
if run is None:
    if not st.session_state.get("error"):
        st.info("Pick a playlist and a destination in the sidebar, then press **Build route**.")
    st.stop()

result, trace = run.result, st.session_state["trace"]
route = result.path
dest_name = trace["destination"]["name"]

for w in run.warnings:
    st.warning(w)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Playlist songs found", f"{run.match.matched_count}/{run.match.total}",
          help=f"{run.match.match_rate:.0%} of the playlist is in the catalog")
m2.metric("Route length", f"{len(route) - 1} steps")
m3.metric("Revisions", result.revisions)
failing = len(trace["result"]["remaining_failures"])
m4.metric("Checks", "All passed" if result.passed else f"{failing} failing")

st.subheader(f"Your route to {dest_name}")
mode = getattr(st.context.theme, "type", None) or "light"
fig = bubble_chart(
    song_map, catalog, set(run.match.matched["track_id"]), route, dest_name,
    resolve_destination(catalog, dest_name).members(catalog), mode=mode,
)
st.plotly_chart(fig, width="stretch", theme=None, config={"displaylogo": False})
st.caption(
    "Each dot is a song, placed by its audio features (9 features squeezed into 2, so distances on "
    "the map are approximate; the agent measures jumps in all 9). Numbers are the route's steps."
)

st.subheader("Step by step")
rows = list(route.iterrows())
for i in range(0, len(rows), 3):
    for col, (_, row) in zip(st.columns(3), rows[i:i + 3]):
        with col, st.container(border=True):
            st.markdown(f"**{row['step']}. {row['track_name']}**  \n{row['artist']}")
            st.caption(f"{row['kind']} · {row['subgenre']} · {row['mood']}"
                       + (f" · jump {row['step_distance']:.3f}" if row["step"] else ""))
            st.progress(min(float(row["progress"]), 1.0), text=f"sounds {row['progress']:.0%} like {dest_name}")
            st.write(explain_step(row, dest_name, is_last=row["step"] == len(route) - 1))
            if row["lineage_source"]:
                st.caption(f"[Source]({row['lineage_source']})")

st.subheader("How the agent got there")
for r in trace["rounds"]:
    failed = ", ".join(f"{f['check'].replace('_', ' ')} at step {f['step']}" for f in r["failed_checks"])
    st.markdown(f"**Round {r['round']}** — {len(r['path']) - 1} steps; "
                + (f"failed checks: {failed}" if failed else "all checks passed"))
    for fix in r["fixes"]:
        st.markdown(f"- fix: {fix}")

with st.expander("Route as a table"):
    st.dataframe(
        pd.DataFrame({
            "step": route["step"], "kind": route["kind"], "song": route["track_name"],
            "artist": route["artist"], "subgenre": route["subgenre"], "mood": route["mood"],
            "compass": route["progress"].round(2), "jump": route["step_distance"].round(3),
            "lineage": route["lineage"],
        }),
        hide_index=True,
    )

with st.expander("Full trace (JSON)"):
    path = st.session_state["trace_path"]
    st.caption(f"Saved to {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}")
    st.download_button("Download trace", path.read_bytes(), file_name=path.name, mime="application/json")
    st.json(trace, expanded=False)
