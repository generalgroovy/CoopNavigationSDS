"""Shared prompt policy for dialog agents.

This module keeps system-prompt assembly, phase instructions, and template
selection in one place so the rest of the pipeline can stay focused on routing
and model execution.
"""

from minillama.agent_a.config import AGENT_RULES
from minillama.agent_a.personas import preference_text
from minillama.agent_a.prompt_data import compact_prompt_context


AGENT_A_ROUTE_TEMPLATES = {
    "focused_commuter": [
        (
            "What's the first set of lines to take to {destination}? "
            "If there's a faster or less crowded option, compare it."
        ),
        (
            "That may work, but is it really the quickest line sequence? "
            "Compare it with another connected option and include riding, waiting, transfer time, and crowding."
        ),
        (
            "Okay, confirm the best line sequence. "
            "Say the stations in order and the total time."
        ),
    ],
    "distracted_multitasker": [
        (
            "Sorry, I'm moving around. Start with one line sequence to {destination}, "
            "and mention another only if it might be faster or much less crowded."
        ),
        (
            "Repeat the current best line sequence and why it wins. "
            "Also tell me about any transfer, and keep ride, wait, and transfer minutes separate."
        ),
        (
            "Give me the final best line-by-line route station by station. "
            "Mention any change and the total time."
        ),
    ],
    "verbose_planner": [
        (
            "Let's build a line sequence to {destination} and compare one transfer option if useful. "
            "Check connectivity, timing, and crowding before deciding."
        ),
        (
            "Add waiting and transfer time to riding time. "
            "Weigh the current line sequence against a different option and mention crowding."
        ),
        (
            "Summarize the best connected line sequence. "
            "Give me the station order, any line change, and the total time."
        ),
    ],
    "hesitant_speaker": [
        (
            "I'm not sure which lines to take to {destination}. "
            "Can we build one route and check it piece by piece?"
        ),
        (
            "I'm still not sure that's the best route. "
            "Compare the time before changing anything."
        ),
        (
            "I think I follow, but confirm the final best line sequence. "
            "Say each station in order and the total time."
        ),
    ],
    "adversarial_tester": [
        (
            "Don't assume the first line sequence to {destination} is best. "
            "Build a candidate, then show a different viable alternative."
        ),
        (
            "That sounds plausible, but I want to test it. "
            "Which line links, timing, and crowding details make it better than the other route?"
        ),
        (
            "Give me the final route once you've checked every segment. "
            "Then give the line order, station order, and total time."
        ),
    ],
    "non_native_speaker": [
        (
            "Please explain one good way to {destination} in simple words. "
            "Say the line names clearly and mention another way only if it is faster."
        ),
        (
            "How long is the current best route with waiting and changes? "
            "Tell me if I need to change trains and whether it gets crowded."
        ),
        (
            "Repeat the best connected route with the line names. "
            "Say where I change trains and the total time."
        ),
    ],
    "frustrated_user": [
        (
            "I need the best connected route to {destination}. "
            "Start with your strongest option and compare another only if it saves time or avoids a packed train."
        ),
        (
            "Check whether changing routes is actually worth it. "
            "Include waiting and transfer time."
        ),
        (
            "Give me the final best line sequence clearly. "
            "Mention any transfer and the total time."
        ),
    ],
}


def build_agent_a_system(persona, scenario):
    return (
        "You are Agent A, the traveler. "
        f"Persona: {persona['name']}. {persona['description']} "
        f"{preference_text(persona)} "
        f"{AGENT_RULES} "
        "First make sure the route reaches the destination; then ask about time, changes, or crowding. "
        f"{compact_prompt_context(scenario)}"
    )


def build_agent_b_system(scenario, persona=None):
    return (
        "You are Agent B, the transit assistant. "
        "Be natural and concise. "
        "Answer the caller's latest concern, then advance the route. "
        "Validity comes first: connected start-to-destination route, correct lines, waits, and transfer time. "
        "Constraints are second: fullness and number of changes. "
        f"{preference_text(persona or {})} "
        "Use a simple spoken form: take LINE from STATION at TIME to STATION. "
        "Mention a change only when the line changes; keep station order and total time clear. "
        f"{AGENT_RULES} "
        f"{compact_prompt_context(scenario)}"
    )


def build_agent_b_phase_instruction(turn, destination):
    if turn == 0:
        return (
            f"Give one valid candidate route to {destination}. "
            "Use: take LINE from STATION at TIME to STATION; then mention station order and total time."
        )

    if turn == 1:
        return (
            "If there is a valid alternative, compare it against the current route. "
            "Only prefer it if validity and total time or constraints improve."
        )

    if turn == 2:
        return (
            "Check route validity before discussing preferences. "
            "Repair any missing station, line, wait, or transfer detail."
        )

    phase = (turn - 3) % 3
    if phase == 0:
        return (
            "Resolve uncertainty by naming the next connected segment. "
            "If complete, only compare an alternative if it is likely better."
        )
    if phase == 1:
        return (
            "Update total time as riding + waiting + transfer. "
            "Change route only for a faster valid option or better constraint fit."
        )
    return (
        "Confirm the best route as lines to take, station by station. "
        "State the total time as riding + waiting + transfer time."
    )


def generate_agent_a_template(turn, persona, scenario, conversation=None):
    reaction = agent_a_route_reaction(turn, persona, scenario, conversation or [])
    if reaction:
        return reaction

    destination = scenario["destination_station"]
    persona_key = persona.get("key", "focused_commuter")
    templates = AGENT_A_ROUTE_TEMPLATES.get(persona_key, AGENT_A_ROUTE_TEMPLATES["focused_commuter"])
    return templates[turn % len(templates)].format(destination=destination)


def agent_a_route_reaction(turn, persona, scenario, conversation):
    """Return a concise caller reaction based on Agent B's latest route reply."""
    last_agent_b = next((text for speaker, text in reversed(conversation) if speaker == "Agent B"), "")
    if not last_agent_b:
        return ""

    from minillama.evaluation.route_interpreter import NaturalRouteInterpreter
    from minillama.model.route_planner import (
        estimate_route_time,
        route_line_change_count,
        route_station_sequence,
    )

    interpreter = NaturalRouteInterpreter()
    if not interpreter.has_station_mentions(last_agent_b):
        return (
            "I need the actual route, not just a general answer. "
            f"Can you give me a connected station order to {scenario['destination_station']}?"
        )

    route = interpreter.interpret_reply(last_agent_b, scenario)
    if not route:
        return (
            "I may be missing the connection there. "
            "Can you restate it as connected stations with the line change points?"
        )

    reaches_destination = (
        route[0] == scenario["start_station"]
        and route[-1] == scenario["destination_station"]
    )
    if not reaches_destination:
        return (
            f"That gives me part of the trip, but I still need to reach {scenario['destination_station']}. "
            "What is the next connected segment?"
        )

    estimate = estimate_route_time(
        route,
        scenario["start_time_min"],
        scenario["transfer_time_min"],
    )
    if estimate:
        arrival, steps = estimate
        duration = arrival - scenario["start_time_min"]
        changes = route_line_change_count(steps)
        fullness_values = [step.get("fullness", 0) for step in steps]
        average_fullness = round(sum(fullness_values) / len(fullness_values)) if fullness_values else 0
        route_summary = f"That gets me there in {duration} minutes with {changes} line change(s)"
        if average_fullness:
            route_summary += f" and about {average_fullness}% fullness"
    else:
        steps = []
        route_summary = "That sounds connected"

    if turn >= 2:
        station_order = " -> ".join(route_station_sequence(steps)) if steps else " -> ".join(route)
        return (
            f"{route_summary}. "
            f"Please confirm this as the final route: {station_order}, including line changes, total time, and any crowding issue."
        )

    request = agent_a_alternative_request(persona)
    return (
        f"{route_summary}. "
        f"Before we settle, can you compare another valid route that is {request}?"
    )


def agent_a_alternative_request(persona):
    """Choose the next route-improvement request from persona constraints."""
    preferences = persona.get("preferences", {})
    priority = preferences.get("priority", "").lower()
    switching = preferences.get("switching", "").lower()
    fullness = preferences.get("fullness", "").lower()

    wants_less_full = any(term in fullness for term in ("less crowded", "dislikes", "crowded", "full", "packed"))
    accepts_full = any(term in fullness for term in ("does not mind", "secondary"))
    wants_fewer_changes = any(term in switching for term in ("fewer", "avoid", "avoiding", "unnecessary", "only for meaningful"))
    wants_fastest = any(term in priority for term in ("fast", "quick", "time"))

    constraints = []
    if wants_fastest:
        constraints.append("faster")
    if wants_less_full and not accepts_full:
        constraints.append("less full")
    if wants_fewer_changes:
        constraints.append("fewer line changes")

    if not constraints:
        constraints.append("a better fit for my preferences")
    if len(constraints) == 1:
        return constraints[0]
    return ", ".join(constraints[:-1]) + f", or {constraints[-1]}"
