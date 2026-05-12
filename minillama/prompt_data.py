"""Prompt-context helpers that describe the current transit model and task rules to the language models.
"""
from minillama.metro_data import compact_line_fullness_text, compact_network_text, compact_travel_time_text


AGENT_RULES = (
    "Speak as if this is a live phone conversation. "
    "Use natural conversational sentences and react to the latest message. "
    "Keep each turn brief, usually one to three short sentences. "
    "Be specific, not robotic or overly formal. "
    "No code, JSON, tables, or bullets. "
    "Do not answer empty. "
)


ROUTE_TASK = (
    "Build one current candidate route together. "
    "Do not repeat the same station sequence as a new proposal. "
    "Extend it by connected station segments; revise it only for a faster connected alternative. "
    "A successful route starts at Agent A's start station and reaches the destination. "
    "Best means lowest riding plus waiting plus transfer time. "
    "Apply transfer time only at a station where the line changes. "
    "Also consider Agent A's preference profile and current line fullness. "
    "All listed segments work both ways. "
    "Arrival time at a station is also departure time there. "
    "Mention stations in travel order."
)


def compact_prompt_context(scenario):
    """Compact prompt context function for this module's MVC responsibility.
    
    Args:
        scenario: Input value used by `compact_prompt_context`; see the function signature and caller context for the expected type.
    
    Returns:
        The computed value or side effect documented by the implementation.
    """
    return (
        f"Current time is {scenario['start_time_min']} minutes after midnight. "
        f"The traveler starts at {scenario['start_station']} and wants to reach {scenario['destination_station']}. "
        f"Changing lines costs {scenario['transfer_time_min']} minutes. "
        f"{ROUTE_TASK} "
        f"Network: {compact_network_text()} "
        f"Current fullness: {compact_line_fullness_text()} "
        f"Segment travel times: {compact_travel_time_text()}"
    )
