"""Prompt-context helpers that describe the current transit model and task rules to the language models.
"""
from coop_navigation_sds.TransportNetwork.network import LINES, STATIONS
from coop_navigation_sds.TransportNetwork.constraints import (
    available_agent_a_constraints,
    constraint_request_text,
    stage_route_options,
)
from coop_navigation_sds.TransportNetwork.routes import route_text_from_steps


def network_vocabulary_text():
    """Return the shared station and line vocabulary known by both agents."""
    station_names = ", ".join(STATIONS)
    line_names = ", ".join(sorted(LINES))
    return f"Known station names: {station_names}. Known line names: {line_names}."


def compact_prompt_context(scenario, persona=None, stated_constraint_keys=()):
    """Return Agent B's verified candidates using only constraints heard so far."""
    destinations = scenario.get("destination_stations") or [scenario["destination_station"]]
    destination_text = " to ".join(destinations) if len(destinations) > 1 else scenario["destination_station"]
    return (
        f"Time: {scenario['start_time_min']} minutes after midnight. "
        f"Start: {scenario['start_station']}. Destination: {destination_text}. "
        f"{network_vocabulary_text()} "
        "Use only these verified candidates and their stated facts. "
        "Candidate ordering uses only constraints recovered from the caller transcript. "
        f"Candidates: {compact_route_candidate_text(scenario, persona, stated_constraint_keys)}"
    )


def caller_prompt_context(scenario, persona=None):
    """Return only the information a hotline caller plausibly knows."""
    destinations = scenario.get("destination_stations") or [scenario["destination_station"]]
    destination_text = " to ".join(destinations) if len(destinations) > 1 else scenario["destination_station"]
    constraints = [
        constraint_request_text(key, persona, scenario)
        for key in available_agent_a_constraints(persona or {}, scenario or {})[:2]
    ]
    constraint_text = "; ".join(constraints) if constraints else "none"
    return (
        f"Time: {scenario['start_time_min']} minutes after midnight. "
        f"Start: {scenario['start_station']}. Destination: {destination_text}. "
        f"{network_vocabulary_text()} "
        "You know these names but not which stations a line serves, network connectivity, "
        "route candidates, schedules, delays, capacity, or transfer times. "
        f"Private constraints to reveal only after the route and duration are acceptable: {constraint_text}."
    )


def compact_route_candidate_text(
    scenario,
    persona=None,
    stated_constraint_keys=(),
    limit=3,
):
    """Return concise valid route candidates instead of the full network."""
    if scenario.get("start_station") not in STATIONS or scenario.get("destination_station") not in STATIONS:
        return "use the listed station names and constraints."
    routes = stage_route_options(
        scenario,
        persona or {},
        stated_keys=stated_constraint_keys,
        limit=max(limit, 20),
    )[:limit]
    if not routes:
        return "no valid candidate found."
    parts = []
    for index, route in enumerate(routes, start=1):
        parts.append(
            f"{index}) {route_text_from_steps(route['steps'])} "
            f"transfer risk {route['transfer_miss_risk_class']}; "
            f"delay risk {route['delay_risk_class']}."
        )
    return " ".join(parts)
