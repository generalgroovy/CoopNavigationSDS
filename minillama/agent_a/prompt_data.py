"""Prompt-context helpers that describe the current transit model and task rules to the language models.
"""
from minillama.agent_a.config import AGENT_RULES, ROUTE_TASK
from minillama.model.metro_data import (
    compact_line_fullness_text,
    compact_station_class_text,
    compact_station_crowding_text,
    compact_station_transfer_text,
    compact_transport_mode_text,
    STATIONS,
)
from minillama.model.route_constraints import (
    available_agent_a_constraints,
    constraint_request_text,
    nearby_walking_links,
    probability_class,
    ranked_constraint_routes,
)
from minillama.model.route_planner import route_text_from_steps


def compact_prompt_context(scenario, persona=None):
    """Compact prompt context function for this module's MVC responsibility.

    Args:
        scenario: Input value used by `compact_prompt_context`; see the function signature and caller context for the expected type.

    Returns:
        The computed value or side effect documented by the implementation.
    """
    destinations = scenario.get("destination_stations") or [scenario["destination_station"]]
    destination_text = " to ".join(destinations) if len(destinations) > 1 else scenario["destination_station"]
    allowed_modes = ", ".join(scenario.get("allowed_modes", ())) or "all modes"
    return (
        f"Time: {scenario['start_time_min']} minutes after midnight. "
        f"Start: {scenario['start_station']}. Destination: {destination_text}. "
        f"Transfer cost: {scenario['transfer_time_min']} minutes base; station transfer times may be longer. "
        f"Ticket modes allowed: {allowed_modes}. "
        f"Maximum transfer-miss risk: {probability_class(scenario.get('max_transfer_miss_probability', 0.3))}. "
        f"Maximum delay risk: {probability_class(scenario.get('max_delay_probability', 0.45))}. "
        f"Walking range: {persona_walk_text(persona)}. "
        f"{ROUTE_TASK} "
        f"Route candidates: {compact_route_candidate_text(scenario, persona)} "
        f"Station classes: {compact_station_class_text()} "
        f"Modes: {compact_transport_mode_text()} "
        f"Long transfers: {compact_station_transfer_text()} "
        f"Walking links: {compact_walking_link_text()} "
        f"Lines: {compact_line_fullness_text(scenario['start_time_min'])} "
        f"Hubs: {compact_station_crowding_text(scenario['start_time_min'])}"
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
        "You do not know the transit network, lines, station classes, route candidates, delays, or transfer times. "
        f"Private constraints to reveal only after the route and duration are acceptable: {constraint_text}."
    )


def compact_route_candidate_text(scenario, persona=None, limit=3):
    """Return concise valid route candidates instead of the full network."""
    if scenario.get("start_station") not in STATIONS or scenario.get("destination_station") not in STATIONS:
        return "use the listed station names and constraints."
    routes = ranked_constraint_routes(scenario, persona or {}, limit=limit)
    if not routes:
        return "no valid candidate found under the current ticket modes."
    parts = []
    for index, route in enumerate(routes, start=1):
        parts.append(
            f"{index}) {route_text_from_steps(route.steps)} "
            f"Modes {' to '.join(route.mode_sequence)}; "
            f"transfer risk {probability_class(route.transfer_miss_probability)}; "
            f"delay risk {probability_class(route.delay_probability)}."
        )
    return " ".join(parts)


def persona_walk_text(persona=None):
    preferences = (persona or {}).get("preferences", {})
    minutes = preferences.get("max_walking_min")
    return f"up to {minutes} minutes" if minutes is not None else "not specified"


def compact_walking_link_text(limit=6):
    parts = []
    for walk, station_a, station_b, transit in nearby_walking_links(limit=limit):
        transit_text = f", direct transit {transit} minutes" if transit is not None else ""
        parts.append(f"{station_a}-{station_b}: walk {walk} minutes{transit_text}")
    return "; ".join(parts) if parts else "none"
