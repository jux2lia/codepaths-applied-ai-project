"""Input guardrails: check what the user gives the agent before it runs.

run_with_guardrails() is the single entry point the CLI, the eval harness and the
Streamlit app all use:

- settings out of range (steps, max step)  -> clamped, with a warning
- empty playlist / no songs in the catalog -> stop with a clear message
- low match rate or a one-song profile     -> continue, with a warning
- unknown destination                      -> stop, suggesting close matches
- destination is already the user's taste  -> continue, with a warning
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from src import agent
from src.data import PlaylistMatch, match_playlist
from src.lineage import load_edges
from src.planner import Destination, resolve_destination

MIN_STEPS, MAX_STEPS = 1, agent.MAX_WAYPOINTS
MIN_MAX_STEP, MAX_MAX_STEP = 0.03, 0.5  # below ~0.03 even neighbouring songs are "too far"
LOW_MATCH_RATE = 0.5
ALREADY_THERE_SHARE = 0.5

logger = logging.getLogger(__name__)


class GuardrailError(ValueError):
    """The input can't produce a meaningful route; the message says why."""


@dataclass
class GuardedRun:
    match: PlaylistMatch
    result: agent.AgentResult
    warnings: list[str]  # guardrail warnings first, then the agent's own
    settings: dict  # the settings actually used, after clamping


def clamp_settings(n_steps: int, max_step: float) -> tuple[int, float, list[str]]:
    warnings = []
    steps = min(max(int(n_steps), MIN_STEPS), MAX_STEPS)
    if steps != n_steps:
        warnings.append(f"Steps must be {MIN_STEPS}-{MAX_STEPS}; using {steps} instead of {n_steps}.")
    step = min(max(float(max_step), MIN_MAX_STEP), MAX_MAX_STEP)
    if step != max_step:
        warnings.append(
            f"Max step must be {MIN_MAX_STEP}-{MAX_MAX_STEP}; using {step} instead of {max_step}."
        )
    return steps, step, warnings


def check_playlist(match: PlaylistMatch) -> list[str]:
    if match.total == 0:
        raise GuardrailError("The playlist is empty. Export a playlist with at least one song.")
    if match.matched_count == 0:
        raise GuardrailError(
            f"None of the playlist's {match.total} songs are in the catalog (0% match), so there's "
            "no taste to start from. The catalog has ~28k mainstream songs released up to early 2020."
        )
    warnings = []
    if match.match_rate < LOW_MATCH_RATE:
        warnings.append(
            f"Only {match.matched_count} of {match.total} songs ({match.match_rate:.0%}) are in the "
            "catalog; your taste profile is based on just those."
        )
    if match.matched_count == 1:
        warnings.append("Only one song matched, so the route starts from that single song.")
    return warnings


def check_destination(matched: pd.DataFrame, dest: Destination) -> list[str]:
    share = float(dest.members(matched).mean())
    if share >= ALREADY_THERE_SHARE:
        return [
            f"{share:.0%} of your matched songs are already {dest.name}; "
            "the route will stay close to home."
        ]
    return []


def run_with_guardrails(
    catalog: pd.DataFrame,
    playlist: pd.DataFrame,
    destination: str,
    n_steps: int = agent.N_STEPS,
    max_step: float = agent.MAX_STEP,
    use_lineage: bool = True,
) -> GuardedRun:
    """Validate the input, run the agent, and collect every warning.

    use_lineage adds the second retrieval source (data/lineage.json).
    Raises GuardrailError (or ValueError for an unknown destination) when no
    meaningful route is possible.
    """
    n_steps, max_step, warnings = clamp_settings(n_steps, max_step)
    match = match_playlist(playlist, catalog)
    warnings += check_playlist(match)
    dest = resolve_destination(catalog, destination)
    warnings += check_destination(match.matched, dest)
    for w in warnings:
        logger.warning(w)

    edges = load_edges() if use_lineage else None
    result = agent.run(catalog, match.matched, dest.name, n_steps=n_steps, max_step=max_step,
                       lineage_edges=edges)
    result.trace["guardrail_warnings"] = list(warnings)
    return GuardedRun(
        match=match,
        result=result,
        warnings=warnings + result.warnings,
        settings={"n_steps": n_steps, "max_step": max_step, "lineage": use_lineage},
    )
