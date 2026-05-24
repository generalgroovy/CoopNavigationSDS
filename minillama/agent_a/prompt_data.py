"""Prompt-context helpers that describe the current transit model and task rules to the language models.
"""
from minillama.agent_a.config import AGENT_RULES, ROUTE_TASK
from minillama.model.metro_data import (
    compact_line_fullness_text,
    compact_network_text,
    compact_station_crowding_text,
    compact_travel_time_text,
)


def compact_prompt_context(scenario):
    """Compact prompt context function for this module's MVC responsibility.

    Args:
        scenario: Input value used by `compact_prompt_context`; see the function signature and caller context for the expected type.

    Returns:
        The computed value or side effect documented by the implementation.
    """
    destinations = scenario.get("destination_stations") or [scenario["destination_station"]]
    destination_text = " -> ".join(destinations) if len(destinations) > 1 else scenario["destination_station"]
    return (
        f"Time: {scenario['start_time_min']} min after midnight. "
        f"Start: {scenario['start_station']}. Destination: {destination_text}. "
        f"Transfer cost: {scenario['transfer_time_min']} min. "
        f"{ROUTE_TASK} "
        f"Network: {compact_network_text()} "
        f"Lines: {compact_line_fullness_text(scenario['start_time_min'])} "
        f"Hubs: {compact_station_crowding_text(scenario['start_time_min'])} "
        f"Segments: {compact_travel_time_text()}"
    )
