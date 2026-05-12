"""Prompt and utterance helpers for Agent A and Agent B, including cleanup and template fallback behavior.
"""
import re

from minillama.config import HISTORY_MESSAGES
from minillama.model_adapters import ChatMessage, messages_to_prompt
from minillama.prompt_data import AGENT_RULES, compact_prompt_context
from minillama.personas import preference_text
from minillama.metro_data import STATION_POS
from minillama.route_planner import candidate_time_routes, fmt_time, optimal_time_route, route_station_sequence


STATION_NAMES = list(STATION_POS)
STATION_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(station) for station in STATION_NAMES) + r")\b",
    re.IGNORECASE,
)
STATION_LOOKUP = {station.lower(): station for station in STATION_NAMES}


def build_agent_a_system(persona, scenario):
    """Build agent a system function for this module's MVC responsibility.
    
    Args:
        persona: Input value used by `build_agent_a_system`; see the function signature and caller context for the expected type.
        scenario: Input value used by `build_agent_a_system`; see the function signature and caller context for the expected type.
    
    Returns:
        The computed value or side effect documented by the implementation.
    """
    return (
        f"You are Agent A, a traveler calling a transit hotline. "
        f"Persona: {persona['name']}. {persona['description']} "
        f"{preference_text(persona)} "
        f"{AGENT_RULES} "
        "Stay in character as the caller. "
        "Ask practical follow-up questions, react to what Agent B just said, and briefly confirm what you understood. "
        "Push back naturally when a route seems disconnected, slower than expected, or misaligned with your preferences. "
        f"{compact_prompt_context(scenario)}"
    )


def build_agent_b_system(scenario, persona=None):
    """Build agent b system function for this module's MVC responsibility.
    
    Args:
        scenario: Input value used by `build_agent_b_system`; see the function signature and caller context for the expected type.
    
    Returns:
        The computed value or side effect documented by the implementation.
    """
    return (
        "You are Agent B, a transit hotline assistant speaking with a caller in real time. "
        "Sound helpful, direct, and conversational rather than scripted. "
        "Respond to the caller's latest concern before moving the route discussion forward. "
        "Build one current route, compare it with a distinct alternative when useful, and keep the faster connected option. "
        f"{preference_text(persona or {})} "
        "Say station names in travel order and explain route timing in plain language as riding, waiting, and transfer time. "
        f"{AGENT_RULES} "
        f"{compact_prompt_context(scenario)}"
    )


def initial_conversation(scenario):
    """Initial conversation function for this module's MVC responsibility.
    
    Args:
        scenario: Input value used by `initial_conversation`; see the function signature and caller context for the expected type.
    
    Returns:
        The computed value or side effect documented by the implementation.
    """
    return [
        (
            "Agent A",
            (
                f"Hi, I'm at {scenario['start_station']} at {fmt_time(scenario['start_time_min'])}, "
                f"and I need to get to {scenario['destination_station']}. "
                "Can you help me figure out the best connected route?"
            ),
        )
    ]


def build_prompt(active_agent_name, active_agent_system, history):
    """Build prompt function for this module's MVC responsibility.
    
    Args:
        active_agent_name: Input value used by `build_prompt`; see the function signature and caller context for the expected type.
        active_agent_system: Input value used by `build_prompt`; see the function signature and caller context for the expected type.
        history: Input value used by `build_prompt`; see the function signature and caller context for the expected type.
    
    Returns:
        The computed value or side effect documented by the implementation.
    """
    return messages_to_prompt(build_messages(active_agent_name, active_agent_system, history))


def build_messages(active_agent_name, active_agent_system, history):
    """Build provider-neutral chat messages for any common chat model."""
    messages = [ChatMessage("system", active_agent_system)]

    # Keep history content free of explicit speaker labels. The role token already
    # encodes whose turn it was, and labels tend to be echoed by smaller models.
    for speaker, text in history[-HISTORY_MESSAGES:]:
        role = "assistant" if speaker == active_agent_name else "user"
        messages.append(ChatMessage(role, text))

    return messages


def clean_reply(text):
    """Clean reply function for this module's MVC responsibility.
    
    Args:
        text: Input value used by `clean_reply`; see the function signature and caller context for the expected type.
    
    Returns:
        The computed value or side effect documented by the implementation.
    """
    text = text.strip()

    stop_markers = [
        "<|user|>",
        "<|assistant|>",
        "<|system|>",
        "</s>",
        "```",
        "{",
        "[",
    ]

    for marker in stop_markers:
        if marker in text:
            text = text.split(marker)[0].strip()

    # Some generations start with speaker tags like "Agent B:" or "Assistant -".
    # Strip only leading labels while preserving the actual message body.
    while True:
        stripped = re.sub(
            r"^\s*(?:agent\s*[ab]|assistant|user)\s*(?::|-)?\s*",
            "",
            text,
            flags=re.IGNORECASE,
        )
        if stripped == text:
            break
        text = stripped

    text = " ".join(line.strip() for line in text.splitlines() if line.strip())

    banned_fragments = [
        "def ",
        "import ",
        "print(",
        "return ",
        "json",
        "{",
        "}",
        "[",
        "]",
        "|",
    ]

    if any(fragment in text.lower() for fragment in banned_fragments):
        return ""

    # Reject placeholder outputs that are only speaker markers or are too short
    # to be useful for route-building.
    normalized = re.sub(r"\s+", " ", text.strip().lower())
    if normalized in {"agent", "agent a", "agent b", "assistant", "user"}:
        return ""

    words = re.findall(r"[A-Za-z0-9]+", text)
    if len(words) < 4:
        return ""

    return text


def fallback_reply(active_agent_name, scenario, route_index=0):
    """Fallback reply function for this module's MVC responsibility.
    
    Args:
        active_agent_name: Input value used by `fallback_reply`; see the function signature and caller context for the expected type.
        scenario: Input value used by `fallback_reply`; see the function signature and caller context for the expected type.
    
    Returns:
        The computed value or side effect documented by the implementation.
    """
    start = scenario["start_station"]
    destination = scenario["destination_station"]
    transfer = scenario["transfer_time_min"]

    if active_agent_name == "Agent B":
        alternatives = candidate_time_routes(
            start,
            destination,
            scenario["start_time_min"],
            scenario["transfer_time_min"],
            limit=3,
        )
        if alternatives:
            duration_min, stations, _ = alternatives[route_index % len(alternatives)]
            snippet = ", then ".join(stations[:6])
            duration = f"{duration_min} minutes"
        else:
            arrival, steps = optimal_time_route(
                start,
                destination,
                scenario["start_time_min"],
                scenario["transfer_time_min"],
            )
            stations = route_station_sequence(steps) if steps else []
            snippet = ", then ".join(stations[:5]) if len(stations) >= 2 else f"{start}, then {destination}"
            duration = f"{arrival - scenario['start_time_min']} minutes" if arrival is not None else "an unknown duration"
        return (
            f"One connected option is {snippet}. "
            f"With {transfer} minute(s) for each line change, plus waiting, that comes to {duration} overall. "
            "I can compare it with another connected option and keep the faster one."
        )

    return (
        "Could you compare the connected options and keep the faster one once riding, waiting, and transfer time are all counted?"
    )


def generate_agent_a_template(turn, persona, scenario):
    """Generate agent a template function for this module's MVC responsibility.
    
    Args:
        turn: Input value used by `generate_agent_a_template`; see the function signature and caller context for the expected type.
        persona: Input value used by `generate_agent_a_template`; see the function signature and caller context for the expected type.
        scenario: Input value used by `generate_agent_a_template`; see the function signature and caller context for the expected type.
    
    Returns:
        The computed value or side effect documented by the implementation.
    """
    destination = scenario["destination_station"]
    persona_key = persona.get("key", "focused_commuter")

    templates = {
        "focused_commuter": [
            (
                f"What's the first connected route you'd try to get me to {destination}? "
                "If there's a faster-looking option, compare that too."
            ),
            (
                "That may work, but is it really the quickest? "
                "Please compare it with another connected option and include riding, waiting, and transfer time."
            ),
            (
                "Okay, please confirm the best connected route. "
                "Say the stations in order and the total time."
            ),
        ],
        "distracted_multitasker": [
            (
                f"Sorry, I'm moving around. Start with one connected route to {destination}, "
                "and mention one other option only if it might be faster."
            ),
            (
                "Can you repeat the current best route and why it beats the other one? "
                "Also tell me about any transfer, and keep ride, wait, and transfer minutes separate."
            ),
            (
                "Please give me the final best route station by station. "
                "Mention any change and the total time."
            ),
        ],
        "verbose_planner": [
            (
                f"Let's build a route to {destination} and compare one transfer option if it looks useful. "
                "I'd like to check connectivity and timing before deciding."
            ),
            (
                "Now add waiting and transfer time to the riding time. "
                "I'd also like you to weigh the current route against a different option."
            ),
            (
                "Please summarize the best connected route. "
                "Give me the station order, any line change, and the total time."
            ),
        ],
        "hesitant_speaker": [
            (
                f"I'm not sure which way to {destination} actually connects. "
                "Can we build one route and check it piece by piece?"
            ),
            (
                "I'm still not sure that's the best route. "
                "Could you compare the time before changing anything?"
            ),
            (
                "I think I follow, but please confirm the final best route. "
                "Say each station in order and the total time."
            ),
        ],
        "adversarial_tester": [
            (
                f"Don't assume the first route to {destination} is the best one. "
                "Build a candidate, then show me a different viable alternative."
            ),
            (
                "That sounds plausible, but I want to test it. "
                "Which links and timing details make it better than the other route?"
            ),
            (
                "Give me the final route once you've checked that every segment connects. "
                "Then give the station order and total time."
            ),
        ],
        "non_native_speaker": [
            (
                f"Please explain one good way to {destination} in simple words. "
                "Say the station names clearly, and mention another way only if it is faster."
            ),
            (
                "How long is the current best route with waiting and changes? "
                "Please tell me if I need to change trains."
            ),
            (
                "Please repeat the best connected route with the station names. "
                "Say where I change trains and the total time."
            ),
        ],
        "frustrated_user": [
            (
                f"I need the best connected route to {destination}. "
                "Start with your strongest option and compare another one only if it might save time."
            ),
            (
                "Just check whether changing routes is actually worth it. "
                "Include waiting and transfer time."
            ),
            (
                "Now give me the final best route clearly. "
                "Mention any transfer and the total time."
            ),
        ],
    }

    options = templates.get(persona_key, templates["focused_commuter"])
    return options[turn % len(options)]

