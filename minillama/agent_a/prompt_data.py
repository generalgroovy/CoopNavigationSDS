"""Prompt-context helpers that describe the current transit model and task rules to the language models.
"""
from minillama.agent_a.config import AGENT_RULES, ROUTE_TASK
from minillama.model.metro_data import (
    compact_line_fullness_text,
    compact_network_text,
    compact_station_crowding_text,
    compact_station_transfer_text,
    compact_transport_mode_text,
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
    destination_text = " to ".join(destinations) if len(destinations) > 1 else scenario["destination_station"]
    allowed_modes = ", ".join(scenario.get("allowed_modes", ())) or "all modes"
    return (
        f"Time: {scenario['start_time_min']} minutes after midnight. "
        f"Start: {scenario['start_station']}. Destination: {destination_text}. "
        f"Transfer cost: {scenario['transfer_time_min']} minutes base; station transfer times may be longer. "
        f"Ticket modes allowed: {allowed_modes}. "
        f"Maximum transfer-miss risk: {round(scenario.get('max_transfer_miss_probability', 0.3) * 100)} percent. "
        f"{ROUTE_TASK} "
        f"Network: {compact_network_text()} "
        f"Modes: {compact_transport_mode_text()} "
        f"Long transfers: {compact_station_transfer_text()} "
        f"Lines: {compact_line_fullness_text(scenario['start_time_min'])} "
        f"Hubs: {compact_station_crowding_text(scenario['start_time_min'])} "
        f"Segments: {compact_travel_time_text()}"
    )
