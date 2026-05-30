"""Prompt and utterance helpers for Agent A and Agent B."""
import re

from minillama.agent_a.config import HISTORY_MESSAGES
from minillama.agent_a.prompting import (
    build_agent_a_system,
    build_agent_b_system,
    generate_agent_a_template,
)
from minillama.model.metro_data import STATION_POS
from minillama.model.model_adapters import ChatMessage, messages_to_prompt
from minillama.model.route_planner import (
    optimal_time_route,
    route_line_change_count,
    route_text_from_steps,
)
from minillama.model.route_constraints import (
    optimal_constraint_route,
    ranked_constraint_routes,
    route_constraint_status,
    stated_constraint_keys,
    unsatisfied_constraint_keys,
)


STATION_NAMES = list(STATION_POS)
STATION_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(station) for station in STATION_NAMES) + r")\b",
    re.IGNORECASE,
)
STATION_LOOKUP = {station.lower(): station for station in STATION_NAMES}


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
                "Can you help me figure out which lines to take?"
            ),
        )
    ]


def build_prompt(active_agent_name, active_agent_system, history):
    return messages_to_prompt(build_messages(active_agent_name, active_agent_system, history))


def build_messages(active_agent_name, active_agent_system, history):
    messages = [ChatMessage("system", active_agent_system)]

    # Keep history content free of explicit speaker labels. The role token already
    # encodes whose turn it was, and labels tend to be echoed by smaller models.
    for speaker, text in history[-HISTORY_MESSAGES:]:
        role = "assistant" if speaker == active_agent_name else "user"
        messages.append(ChatMessage(role, text))

    return messages


def clean_reply(text):
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


def fallback_reply(active_agent_name, scenario, route_index=0, persona=None, conversation=None):
    start = scenario["start_station"]
    destination = scenario["destination_station"]

    if active_agent_name == "Agent B":
        ranked_routes = ranked_constraint_routes(scenario, persona or {}, limit=8)
        stated_keys = stated_constraint_keys(conversation or [])
        constraint_route = optimal_constraint_route(scenario, persona or {})
        prior_routes = set()
        if conversation:
            from minillama.evaluation.route_interpreter import NaturalRouteInterpreter

            interpreter = NaturalRouteInterpreter()
            for speaker, text in conversation:
                if speaker != "Agent B":
                    continue
                route = interpreter.interpret_reply(text, scenario)
                if route:
                    prior_routes.add(tuple(route))
        if stated_keys:
            satisfying = [
                route
                for route in ranked_routes
                if not unsatisfied_constraint_keys(
                    route_constraint_status(
                        route.steps,
                        persona or {},
                        scenario,
                        stated_keys,
                        constraint_route=constraint_route,
                    )
                )
            ]
            if satisfying:
                ranked_routes = satisfying
        fresh_routes = [route for route in ranked_routes if tuple(route.route) not in prior_routes]
        if fresh_routes:
            selected = fresh_routes[0]
        elif ranked_routes:
            selected = ranked_routes[route_index % len(ranked_routes)]
        else:
            selected = None
        if selected is not None:
            steps = selected.steps
            snippet = route_text_from_steps(steps)
        else:
            arrival, steps = optimal_time_route(
                start,
                destination,
                scenario["start_time_min"],
                scenario["transfer_time_min"],
                allowed_modes=scenario.get("allowed_modes"),
            )
            snippet = route_text_from_steps(steps) if steps else f"take a line from {start} to {destination}"
        if steps:
            from minillama.model.route_constraints import probability_class, route_delay_probability, route_has_near_capacity, route_transfer_miss_probability

            if agent_a_requested_secondary_constraints(conversation or []):
                delay_risk = probability_class(route_delay_probability(steps))
                transfer_risk = probability_class(route_transfer_miss_probability(steps))
                capacity = "near capacity" if route_has_near_capacity(steps) else "not near capacity"
                return f"{snippet} {capacity.capitalize()}; delay risk {delay_risk}; transfer miss risk {transfer_risk}."
            return snippet
        return (
            f"One connected option: {snippet} "
            f"Transfer time applies only when changing lines."
        )

    return (
        "Valid route. Now compare one shorter or faster option; mention transfers or near-capacity trains only if they change the choice."
    )


def agent_a_requested_secondary_constraints(conversation):
    """Return whether Agent A has asked about non-time route constraints."""
    direct_terms = (
        "near-capacity",
        "capacity",
        "crowded",
        "packed",
        "full",
        "delay",
        "risk",
        "transfer miss",
        "safer transfer",
        "ticket",
        "mode",
        "walk",
        "walking",
    )
    change_preference_terms = (
        "fewer changes",
        "fewer line changes",
        "avoid changes",
        "avoid line changes",
        "less switching",
        "avoid switching",
    )
    for speaker, text in conversation:
        if speaker != "Agent A":
            continue
        lower = text.lower()
        if any(term in lower for term in direct_terms):
            return True
        if any(term in lower for term in change_preference_terms):
            return True
    return False
