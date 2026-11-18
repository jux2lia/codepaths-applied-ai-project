"""Bubble Breaker command line: build a route of songs from a playlist to a new genre.

    python -m src.main --to "hard rock"
    python -m src.main --playlist examples/neo_soul.csv --to "hard rock" --steps 4

Prints the route and a summary of every Plan -> Act -> Check -> Revise round,
saves the full trace to logs/traces/, and appends to logs/bubble_breaker.log.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path

from src import agent
from src.data import PLAYLIST_PATH, ROOT, load_catalog, load_playlist
from src.guardrails import GuardedRun, run_with_guardrails

LOG_FILE = ROOT / "logs" / "bubble_breaker.log"
TRACE_DIR = ROOT / "logs" / "traces"

logger = logging.getLogger("src.main")


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m src.main",
        description="Build a route of songs from your playlist's taste to a new genre.",
    )
    p.add_argument("--to", dest="destination", required=True,
                   help="destination genre or subgenre, e.g. rock or 'hard rock'")
    p.add_argument("--playlist", type=Path, default=PLAYLIST_PATH,
                   help="Exportify CSV (default: data/my_playlist.csv)")
    p.add_argument("--steps", type=int, default=agent.N_STEPS,
                   help=f"planned steps before any revisions (default: {agent.N_STEPS})")
    p.add_argument("--max-step", type=float, default=agent.MAX_STEP,
                   help=f"largest allowed jump between songs (default: {agent.MAX_STEP})")
    p.add_argument("--no-lineage", action="store_true",
                   help="use audio features only, without sample links from data/lineage.json")
    p.add_argument("--trace-dir", type=Path, default=TRACE_DIR,
                   help="where to save the JSON trace (default: logs/traces)")
    p.add_argument("--no-save", action="store_true", help="don't save a trace file")
    return p.parse_args(argv)


def setup_logging(log_file: Path) -> None:
    """Send the agent's log messages (loggers under 'src') to log_file."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    src_logger = logging.getLogger("src")
    src_logger.setLevel(logging.INFO)
    for h in [h for h in src_logger.handlers if isinstance(h, logging.FileHandler)]:
        src_logger.removeHandler(h)
        h.close()
    handler = logging.FileHandler(log_file, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    src_logger.addHandler(handler)


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # song titles on Windows
    args = parse_args(argv)
    setup_logging(LOG_FILE)
    started = datetime.now()
    logger.info("Run: playlist=%s to=%s steps=%d max_step=%.3f",
                args.playlist.name, args.destination, args.steps, args.max_step)

    try:
        catalog = load_catalog()
        run = run_with_guardrails(catalog, load_playlist(args.playlist), args.destination,
                                  n_steps=args.steps, max_step=args.max_step,
                                  use_lineage=not args.no_lineage)
    except (ValueError, FileNotFoundError) as err:  # includes GuardrailError
        logger.error("Stopped: %s", err)
        print(f"Error: {err}", file=sys.stderr)
        return 1

    match, result = run.match, run.result
    logger.info("Matched %d/%d playlist songs (%.0f%%)",
                match.matched_count, match.total, 100 * match.match_rate)
    print_match(args.playlist, match)
    trace = build_trace(run, started, args.playlist.name,
                        {"to": args.destination, "steps": args.steps, "max_step": args.max_step,
                         "lineage": not args.no_lineage})
    print_route(result)
    for w in run.warnings:
        print(f"\nWarning: {w}")
    if not args.no_save:
        path = save_trace(trace, args.trace_dir, started, args.playlist.stem, result.trace["destination"]["name"])
        logger.info("Trace saved to %s", path)
        print(f"\nFull trace saved to {_display(path)}")
    return 0


def build_trace(run: GuardedRun, started: datetime, playlist_name: str, args: dict) -> dict:
    """The full JSON trace of one run: inputs, match, and every agent round."""
    match = run.match
    return {
        "run": {
            "started": started.isoformat(timespec="seconds"),
            "playlist_file": playlist_name,
            "args": args,
            "settings_used": run.settings,
        },
        "playlist": {
            "songs": match.total,
            "matched": match.matched_count,
            "match_rate": round(match.match_rate, 3),
            "unmatched": [f"{r.track_name} — {r.artists}" for r in match.unmatched.itertuples()],
        },
        **run.result.trace,
    }


def print_match(playlist_path: Path, match) -> None:
    print(f"Playlist: {playlist_path.name} — {match.matched_count}/{match.total} songs "
          f"found in the catalog ({match.match_rate:.0%})")
    if not match.unmatched.empty:
        shown = ", ".join(match.unmatched["track_name"].head(5))
        more = f" and {len(match.unmatched) - 5} more" if len(match.unmatched) > 5 else ""
        print(f"          not found: {shown}{more}")


def print_route(result: agent.AgentResult) -> None:
    t = result.trace
    profile = t["profile"]
    genres = ", ".join(f"{g} {n}" for g, n in profile["top_genres"].items())
    moods = ", ".join(f"{m} {n}" for m, n in profile["moods"].items())
    print(f"Taste:    {genres}  |  moods: {moods}")
    print(f"From:     {profile['start_song']}  (compass {profile['start_progress']:.2f})")
    print(f"To:       {t['destination']['name']} ({t['destination']['level']})")

    print("\nPlan -> Act -> Check -> Revise")
    for r in t["rounds"]:
        failed = ", ".join(f"{f['check']} at step {f['step']}" for f in r["failed_checks"]) or "none"
        print(f"  Round {r['round']}: {len(r['path']) - 1} steps, failed checks: {failed}")
        for fix in r["fixes"]:
            print(f"      fix: {fix}")

    n = result.revisions
    status = "all checks passed" if result.passed else "some checks still failing"
    print(f"\nRoute ({status}, {n} revision{'s' if n != 1 else ''}):")
    for row in result.path.itertuples():
        print(f"  {row.step:>2}. [{row.kind:<7}] {row.track_name} — {row.artist}")
        print(f"      {row.subgenre}, {row.mood}, compass {row.progress:.2f}, jump {row.step_distance:.3f}")
        if row.lineage:
            print(f"      lineage: {row.lineage}")


def save_trace(trace: dict, trace_dir: Path, started: datetime, playlist: str, destination: str) -> Path:
    trace_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{started:%Y%m%d-%H%M%S}_{_slug(playlist)}_to_{_slug(destination)}"
    path = trace_dir / f"{stem}.json"
    n = 2
    while path.exists():  # two runs in the same second
        path = trace_dir / f"{stem}_{n}.json"
        n += 1
    path.write_text(json.dumps(trace, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _display(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    sys.exit(main())
