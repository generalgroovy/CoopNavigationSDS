from metro_data import compact_network_text, compact_travel_time_text


AGENT_RULES = (
    "Speak in full natural sentences. "
    "No code. "
    "No JSON. "
    "No tables. "
    "No bullets. "
    "Do not answer empty. "
    "Use two to four short spoken sentences."
)


ROUTE_TASK = (
    "Build the route step by step across the conversation. "
    "Reason together about connected station segments before accepting a route. "
    "A route is successful only if it goes from the start station to the destination station. "
    "Its duration is riding time plus waiting time plus transfer time. "
    "Mention station names in travel order when proposing or refining a route."
)


def compact_prompt_context(scenario):
    return (
        f"Current time is {scenario['start_time_min']} minutes after midnight. "
        f"The traveler starts at {scenario['start_station']} and wants to reach {scenario['destination_station']}. "
        f"Changing lines costs {scenario['transfer_time_min']} minutes. "
        f"{ROUTE_TASK} "
        f"Network: {compact_network_text()} "
        f"Segment travel times: {compact_travel_time_text()}"
    )
