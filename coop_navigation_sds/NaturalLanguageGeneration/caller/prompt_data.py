"""Prompt-context helpers that describe the current transit model and task rules to the language models.
"""
from coop_navigation_sds.TransportNetwork.network import STATIONS
from coop_navigation_sds.TransportNetwork.constraints import (
    available_agent_a_constraints,
    constraint_request_text,
    probability_class,
    ranked_constraint_routes,
)
from coop_navigation_sds.TransportNetwork.routes import route_text_from_steps


def compact_prompt_context(scenario, persona=None):
    """Compact prompt context function for this module's MVC responsibility.

    Args:
        scenario: Input value used by `compact_prompt_context`; see the function signature and caller context for the expected type.

    Returns:
        The computed value or side effect documented by the implementation.
    """
    destinations = scenario.get("destination_stations") or [scenario["destination_station"]]
    destination_text = " to ".join(destinations) if len(destinations) > 1 else scenario["destination_station"]
    return (
        f"Time: {scenario['start_time_min']} minutes after midnight. "
        f"Start: {scenario['start_station']}. Destination: {destination_text}. "
        "Use only these verified candidates and their stated facts. "
        f"Candidates: {compact_route_candidate_text(scenario, persona)}"
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
        "You do not know the transit network, lines, route candidates, delays, or transfer times. "
        f"Private constraints to reveal only after the route and duration are acceptable: {constraint_text}."
    )


def compact_route_candidate_text(scenario, persona=None, limit=3):
    """Return concise valid route candidates instead of the full network."""
    if scenario.get("start_station") not in STATIONS or scenario.get("destination_station") not in STATIONS:
        return "use the listed station names and constraints."
    routes = ranked_constraint_routes(scenario, persona or {}, limit=limit)
    if not routes:
        return "no valid candidate found."
    parts = []
    for index, route in enumerate(routes, start=1):
        parts.append(
            f"{index}) {route_text_from_steps(route.steps)} "
            f"transfer risk {probability_class(route.transfer_miss_probability)}; "
            f"delay risk {probability_class(route.delay_probability)}."
        )
    return " ".join(parts)
