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
        "You are Agent A, a traveler on a transit hotline in a speech-dialog evaluation study. "
        f"Persona: {persona['name']}. {persona['description']} "
        f"{preference_text(persona)} "
        f"{AGENT_RULES} "
        "Stay in character as the caller. "
        "Ask practical follow-up questions, react to Agent B, and briefly confirm what you understood. "
        "Push back when a route seems disconnected, slower, crowded, or misaligned with your preferences. "
        f"{compact_prompt_context(scenario)}"
    )


def build_agent_b_system(scenario, persona=None):
    return (
        "You are Agent B, a transit hotline assistant speaking with a caller in real time. "
        "Sound helpful, direct, and conversational. "
        "Respond to the caller's latest concern before moving the route forward. "
        "Build one line-by-line route, compare a distinct alternative when useful, and keep the faster connected option. "
        f"{preference_text(persona or {})} "
        "Say which lines to take, where to transfer, and the station order that supports that line sequence. "
        "Explain timing and crowding plainly as riding, waiting, transfer time, and crowding tradeoffs. "
        f"{AGENT_RULES} "
        f"{compact_prompt_context(scenario)}"
    )


def build_agent_b_phase_instruction(turn, destination):
    if turn == 0:
        return (
            f"Start with one candidate line sequence toward {destination}. "
            "Mention the lines to take in order and say what to compare next."
        )

    if turn == 1:
        return (
            "If another plausible line sequence exists, bring it up naturally. "
            "Compare connectivity, riding time, waiting time, transfer time, and line changes."
        )

    if turn == 2:
        return (
            "Check that the line sequence really connects start to destination. "
            "If anything is missing, add the next connected station and line change."
        )

    phase = (turn - 3) % 3
    if phase == 0:
        return (
            "Resolve uncertainty by naming the next line segment. "
            "If the line sequence is already complete, mention a shorter-looking alternative only if it seems worth checking."
        )
    if phase == 1:
        return (
            "Update timing using riding, waiting, and transfer minutes. "
            "Refine the line sequence only if that finds a faster valid option."
        )
    return (
        "Confirm the best route as lines to take, station by station. "
        "State the total time as riding + waiting + transfer time."
    )


def generate_agent_a_template(turn, persona, scenario):
    destination = scenario["destination_station"]
    persona_key = persona.get("key", "focused_commuter")
    templates = AGENT_A_ROUTE_TEMPLATES.get(persona_key, AGENT_A_ROUTE_TEMPLATES["focused_commuter"])
    return templates[turn % len(templates)].format(destination=destination)
