"""Shared prompt policy for dialog agents.

This module keeps system-prompt assembly, phase instructions, and template
selection in one place so the rest of the pipeline can stay focused on routing
and model execution.
"""

from coop_navigation_sds.NaturalLanguageGeneration.caller.config import AGENT_RULES
from coop_navigation_sds.NaturalLanguageGeneration.caller.personas import preference_text
from coop_navigation_sds.NaturalLanguageGeneration.caller.prompt_data import caller_prompt_context, compact_prompt_context
from coop_navigation_sds.NaturalLanguageUnderstanding.clarification import (
    clarification_attempt_count,
    clarification_question,
    misunderstood_word_options,
)
from coop_navigation_sds.TransportNetwork.constraints import (
    OBJECTIVE_MODE_LABELS,
    OBJECTIVE_SHORTEST_ROUTE,
    OBJECTIVE_SHORTEST_WITH_CONSTRAINTS,
    OBJECTIVE_VALID_ROUTE,
    acceptable_duration_limit,
    available_agent_a_constraints,
    constraint_request_text,
    normalize_objective_mode,
    optimal_constraint_route,
    route_constraint_status,
    stated_constraint_keys,
    unsatisfied_constraint_keys,
)


def agent_a_repeat_trip_facts(scenario, conversation=None):
    """Return a concise caller repetition after Agent B did not understand."""
    from coop_navigation_sds.TransportNetwork.routes import fmt_time

    if clarification_attempt_count(conversation or [], "Agent B") > 1:
        return (
            f"Starting station: {scenario['start_station']}. Destination station: "
            f"{scenario['destination_station']}. Departure time: {fmt_time(scenario['start_time_min'])}."
        )
    return (
        f"Sure: I said {scenario['start_station']} at {fmt_time(scenario['start_time_min'])}, "
        f"going to {scenario['destination_station']}. Please use those."
    )


def agent_a_unclear_route_reply(text, destination):
    """Return a compact repair request for a nonsensical Agent B answer."""
    fallback = f"I need the actual route to {destination}. Give connected boarding stations."
    word, options = misunderstood_word_options(text)
    if not word:
        return fallback
    if len(options) >= 2:
        return (
            f"When you said '{word}', did you mean {options[0]} or {options[1]}? "
            f"{fallback}"
        )
    return f"When you said '{word}', did you mean {options[0]}? {fallback}"


AGENT_A_ROUTE_TEMPLATES = {
    "focused_commuter": [
        (
            "I'm at {start} at {time}, going to {destination}. "
            "What route should I take?"
        ),
        (
            "Can you compare that with a shorter or faster valid route?"
        ),
        (
            "Thanks, I'll take the best one. Can you confirm the stations and total time?"
        ),
    ],
    "distracted_multitasker": [
        (
            "Sorry, I'm moving around. Start with one line sequence to {destination}, "
            "and mention another only if it might be faster or avoids near-capacity trains."
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
            "Check connectivity, timing, and capacity status before deciding."
        ),
        (
            "Add waiting and transfer time to riding time. "
            "Weigh the current line sequence against a different option and mention capacity status."
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
            "Which line links, timing, and capacity details make it better than the other route?"
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
    objective_mode = normalize_objective_mode(scenario.get("agent_a_objective_mode"))
    objective_text = OBJECTIVE_MODE_LABELS[objective_mode].lower()
    if objective_mode == OBJECTIVE_VALID_ROUTE:
        objective_instruction = "Get one valid connected route, then choose it and end naturally. "
    elif objective_mode == OBJECTIVE_SHORTEST_ROUTE:
        objective_instruction = "Get a valid route first, then ask for one faster route before choosing. "
    else:
        objective_instruction = (
            "Priority 1: establish a valid route from start to destination. "
            "Priority 2: accept it only if arrival time is within the configured duration limit. "
            "Priority 3: reveal exactly one private constraint; if it is missed, ask for a better route "
            "without revealing another constraint. Reveal constraint 2 only after constraint 1 is met. "
            "Never reveal more than one new constraint in one turn. "
        )
    return (
        "You are Agent A, a caller on a transit hotline. "
        "You have no transit-network knowledge. "
        f"Persona: {persona['name']}. {persona['description']} "
        f"{preference_text(persona)} "
        f"{AGENT_RULES} "
        f"Objective: {objective_text}. "
        "First state start time, start station, and destination. "
        "Remember Agent B's previous viable routes and corrections. "
        "If Agent B did not understand, repeat start station, start time, and destination. "
        "If Agent B's answer makes no sense, ask whether a likely word was meant before continuing. "
        f"{objective_instruction}End naturally when you choose a route. "
        f"{caller_prompt_context(scenario, persona)}"
    )


def build_agent_b_system(scenario, persona=None):
    return (
        "You are Agent B on a transit hotline. Reply naturally in one short sentence. "
        "Use only the verified route candidates below. Give one valid route first; later offer a distinct candidate that retains accepted constraints. "
        "React only to what the caller was heard saying. If a key fact is unclear, ask only for that fact. "
        "Do not repeat a route or mention fullness, changes, or risk before the caller asks. "
        f"{preference_text(persona or {})} "
        "Say line legs and total time once. No reasoning, lists, or formatting. "
        f"{compact_prompt_context(scenario, persona)}"
    )


def build_agent_b_phase_instruction(turn, destination):
    """Compatibility wrapper for callers that still provide a turn index."""
    from coop_navigation_sds.DialogManagement.stages import ConversationStage

    stages = (
        ConversationStage.PROPOSAL,
        ConversationStage.COMPARISON,
        ConversationStage.REFINEMENT,
        ConversationStage.CONFIRMATION,
    )
    return build_agent_b_stage_instruction(stages[min(max(int(turn), 0), 3)], destination)


def build_agent_b_stage_instruction(stage, destination):
    """Return concise Agent B guidance for an explicit conversation stage."""
    from coop_navigation_sds.DialogManagement.stages import ConversationStage

    try:
        normalized = stage if isinstance(stage, ConversationStage) else ConversationStage(stage)
    except ValueError:
        normalized = ConversationStage.REFINEMENT

    instructions = {
        ConversationStage.DISCOVERY: (
            "Ask for the missing start station, destination, or departure time."
        ),
        ConversationStage.PROPOSAL: (
            f"Give one valid route to {destination}. Say line legs and total time once."
        ),
        ConversationStage.COMPARISON: (
            "Respond directly to the caller's latest comparison request. "
            "Offer one distinct valid alternative and state the practical tradeoff."
        ),
        ConversationStage.REFINEMENT: (
            "Respond directly to the latest constraint or correction. "
            "Keep remembered accepted details and change only what the caller asked to improve."
        ),
        ConversationStage.CONFIRMATION: (
            "Confirm the best route as lines and stations in order. "
            "State changes and total time once, without introducing a new option."
        ),
        ConversationStage.CLOSED: (
            "Acknowledge the selected route briefly and close the call naturally."
        ),
    }
    return instructions[normalized]


def generate_agent_a_template(turn, persona, scenario, conversation=None):
    reaction = agent_a_route_reaction(turn, persona, scenario, conversation or [])
    if reaction:
        return reaction

    destination = scenario["destination_station"]
    persona_key = persona.get("key", "focused_commuter")
    templates = AGENT_A_ROUTE_TEMPLATES.get(persona_key, AGENT_A_ROUTE_TEMPLATES["focused_commuter"])
    from coop_navigation_sds.TransportNetwork.routes import fmt_time

    return templates[turn % len(templates)].format(
        destination=destination,
        start=scenario["start_station"],
        time=fmt_time(scenario["start_time_min"]),
    )


def progressive_constraint_limit(persona, scenario):
    """Return the configured number of private constraints Agent A may reveal."""
    available = len(available_agent_a_constraints(persona, scenario))
    configured = int((scenario or {}).get("maximum_progressive_constraints", 2) or 0)
    return available if configured <= 0 else min(available, configured)


def constraint_sequence_complete(persona, scenario, stated_keys):
    """Return whether every configured progressive constraint has been stated."""
    return len(stated_keys) >= progressive_constraint_limit(persona, scenario)


def route_comparison_complete(conversation, scenario, allowed_modes):
    """Return whether enough distinct valid routes have been discussed."""
    required = max(1, int((scenario or {}).get("minimum_compared_routes", 2) or 1))
    return valid_agent_b_route_count(conversation, scenario, allowed_modes) >= required


def final_agent_a_turn(scenario, conversation):
    """Return whether the next Agent A reply consumes the configured turn budget."""
    maximum = int((scenario or {}).get("maximum_dialog_turns", 0) or 0)
    return maximum > 0 and len(conversation) + 1 >= maximum


def agent_a_route_reaction(turn, persona, scenario, conversation):
    """Return a concise caller reaction based on Agent B's latest route reply."""
    objective_mode = normalize_objective_mode(scenario.get("agent_a_objective_mode"))
    last_agent_b = next((text for speaker, text in reversed(conversation) if speaker == "Agent B"), "")
    if not last_agent_b:
        return ""

    from coop_navigation_sds.NaturalLanguageUnderstanding.interpreter import NaturalRouteInterpreter
    from coop_navigation_sds.TransportNetwork.routes import (
        estimate_route_time,
        route_line_change_count,
        route_station_sequence,
    )
    from coop_navigation_sds.TransportNetwork.constraints import (
        dialog_route_allowed_modes,
    )

    interpreter = NaturalRouteInterpreter()
    stated_keys = stated_constraint_keys(conversation)
    allowed_modes = dialog_route_allowed_modes(scenario, persona, stated_keys)
    lower_reply = last_agent_b.lower()
    if (
        any(term in lower_reply for term in ("did not catch", "heard", "unclearly", "repeat", "reset", "say only"))
        and any(term in lower_reply for term in ("start", "destination", "time", "mean"))
    ):
        return agent_a_repeat_trip_facts(scenario, conversation)
    if not interpreter.has_station_mentions(last_agent_b):
        if "earlier" in lower_reply and any(
            term in lower_reply for term in ("fits", "meets", "satisf", "best valid choice")
        ):
            constraint_route = optimal_constraint_route(scenario, persona, objective_mode=objective_mode)
            duration_limit = acceptable_duration_limit(
                scenario,
                persona,
                constraint_route=constraint_route,
            )
            prior_match = best_prior_route_matching_constraints(
                conversation[:-1],
                scenario,
                persona,
                stated_keys,
                duration_limit,
                constraint_route,
            )
            if prior_match:
                prior_duration, prior_route, prior_steps = prior_match
                available_constraints = available_agent_a_constraints(persona, scenario)
                next_constraint = next(
                    (key for key in available_constraints if key not in stated_keys),
                    None,
                )
                sequence_complete = constraint_sequence_complete(
                    persona, scenario, stated_keys
                )
                comparison_complete = route_comparison_complete(
                    conversation, scenario, allowed_modes
                )
                if (
                    (sequence_complete or next_constraint is None)
                    and (comparison_complete or final_agent_a_turn(scenario, conversation))
                ):
                    station_order = (
                        " to ".join(route_station_sequence(prior_steps))
                        if prior_steps
                        else " to ".join(prior_route)
                    )
                    return f"Thanks, that fits. I'll take {station_order}."
                if sequence_complete or next_constraint is None:
                    return "That route still fits. Compare one distinct viable alternative before I choose."
                return (
                    f"That fits. Can you make it "
                    f"{constraint_request_text(next_constraint, persona, scenario)}?"
                )
        return agent_a_unclear_route_reply(last_agent_b, scenario["destination_station"])

    route = interpreter.interpret_reply(last_agent_b, scenario)
    if not route:
        repair = clarification_question(
            last_agent_b,
            "I missed the connection. Restate the boarding stations and change points.",
        )
        return repair

    reaches_destination = (
        route[0] == scenario["start_station"]
        and route[-1] == scenario["destination_station"]
    )
    if not reaches_destination:
        return (
            f"That does not reach {scenario['destination_station']} yet. "
            "What is the next connected segment?"
        )

    estimate = estimate_route_time(
        route,
        scenario["start_time_min"],
        scenario["transfer_time_min"],
        allowed_modes=allowed_modes,
    )
    if estimate:
        arrival, steps = estimate
        duration = arrival - scenario["start_time_min"]
        changes = route_line_change_count(steps)
        change_text = "no changes" if changes == 0 else "1 change" if changes == 1 else f"{changes} changes"
        route_summary = f"Okay, that is {duration} minutes with {change_text}"
    else:
        steps = []
        duration = None
        route_summary = "That sounds connected"

    if objective_mode == OBJECTIVE_VALID_ROUTE:
        station_order = " to ".join(route_station_sequence(steps)) if steps else " to ".join(route)
        return f"{route_summary}. Thanks, that is valid. I'll take {station_order}. Please confirm the total time."

    if objective_mode == OBJECTIVE_SHORTEST_ROUTE:
        prior_best_duration = best_prior_route_duration(conversation[:-1], scenario)
        if prior_best_duration is None:
            return f"{route_summary}. Compare one faster valid route."
        if duration is not None and duration > prior_best_duration:
            return (
                f"That is slower than the earlier {prior_best_duration}-minute route. "
                "Keep the earlier option unless there is a faster valid one."
            )
        station_order = " to ".join(route_station_sequence(steps)) if steps else " to ".join(route)
        return f"{route_summary}. I would choose the shortest valid option, {station_order}. Confirm total time."

    constraint_route = optimal_constraint_route(scenario, persona, objective_mode=objective_mode)
    duration_limit = acceptable_duration_limit(scenario, persona, constraint_route=constraint_route)
    stated_keys = stated_constraint_keys(conversation)
    if duration_limit is not None:
        if duration is None:
            return "That reaches the destination, but I need the total time before checking preferences."
        if duration > duration_limit:
            prior_match = best_prior_route_matching_constraints(
                conversation[:-1],
                scenario,
                persona,
                stated_keys,
                duration_limit,
                constraint_route,
            )
            if prior_match and stated_keys:
                prior_duration, prior_route, prior_steps = prior_match
                available_constraints = available_agent_a_constraints(persona, scenario)
                next_constraint = next((key for key in available_constraints if key not in stated_keys), None)
                sequence_complete = constraint_sequence_complete(
                    persona, scenario, stated_keys
                )
                comparison_complete = route_comparison_complete(
                    conversation, scenario, allowed_modes
                )
                if (
                    (sequence_complete or next_constraint is None)
                    and (comparison_complete or final_agent_a_turn(scenario, conversation))
                ):
                    station_order = " to ".join(route_station_sequence(prior_steps)) if prior_steps else " to ".join(prior_route)
                    return f"That earlier route fits. Thanks, I'll take {station_order}."
                if sequence_complete or next_constraint is None:
                    return "The earlier route fits. Compare one distinct viable alternative before I choose."
                return (
                    f"The earlier route fits. Can you make it "
                    f"{constraint_request_text(next_constraint, persona, scenario)}?"
                )
            return (
                f"That reaches {scenario['destination_station']}, but {duration} minutes is too long. "
                f"Find {duration_limit} minutes or less?"
            )

    statuses = route_constraint_status(steps, persona, scenario, stated_keys, constraint_route=constraint_route)
    unsatisfied = unsatisfied_constraint_keys(statuses)
    if unsatisfied:
        key = unsatisfied[0]
        detail = statuses[key]
        if key == "delay":
            return f"Delay risk is {detail['actual']}; I need {detail['limit']} or lower. Another route?"
        if key == "transfer_miss":
            return f"Transfer risk is {detail['actual']}; I need {detail['limit']} or lower. Another option?"
        if key == "transfers":
            return f"That has {detail['actual']} changes; I can handle {detail['limit']}. Another route?"
        if key == "fullness":
            return "That uses a near-capacity train. Any less full option?"

    available_constraints = available_agent_a_constraints(persona, scenario)
    next_constraint = next((key for key in available_constraints if key not in stated_keys), None)
    if final_agent_a_turn(scenario, conversation):
        best_match = best_prior_route_matching_constraints(
            conversation,
            scenario,
            persona,
            stated_keys,
            duration_limit,
            constraint_route,
        )
        if best_match:
            _best_duration, best_route, best_steps = best_match
            station_order = (
                " to ".join(route_station_sequence(best_steps))
                if best_steps
                else " to ".join(best_route)
            )
            return f"Thanks, I am out of time. I'll take {station_order}."
    if constraint_sequence_complete(persona, scenario, stated_keys) or next_constraint is None:
        if not route_comparison_complete(conversation, scenario, allowed_modes):
            return "That fits so far. Compare one distinct viable alternative before I choose."
        prior_best_duration = best_prior_route_duration(conversation[:-1], scenario)
        if prior_best_duration is not None and duration is not None and prior_best_duration < duration:
            return "The earlier route also fits. Thanks, I'll take that."
        station_order = " to ".join(route_station_sequence(steps)) if steps else " to ".join(route)
        return f"Thanks, that meets my requirements. I'll take {station_order}."
    if next_constraint:
        acknowledgement = (
            "That route and timing work"
            if not stated_keys
            else "That meets the current requirement"
        )
        return f"{acknowledgement}. Can you make it {constraint_request_text(next_constraint, persona, scenario)}?"

    station_order = " to ".join(route_station_sequence(steps)) if steps else " to ".join(route)
    return f"Thanks, that meets my requirements. I'll take {station_order}."


def agent_a_alternative_request(persona):
    """Choose the next route-improvement request from persona constraints."""
    preferences = persona.get("preferences", {})
    priority = preferences.get("priority", "").lower()
    switching = preferences.get("switching", "").lower()
    fullness = preferences.get("fullness", "").lower()
    reliability = preferences.get("reliability", "").lower()

    wants_less_full = any(term in fullness for term in ("less crowded", "dislikes", "crowded", "full", "packed"))
    accepts_full = any(term in fullness for term in ("does not mind", "secondary"))
    wants_fewer_changes = any(term in switching for term in ("fewer", "avoid", "avoiding", "unnecessary", "only for meaningful"))
    wants_fastest = any(term in priority for term in ("fast", "quick", "travel time"))
    wants_low_delay = any(term in f"{priority} {reliability}" for term in ("delay", "reliable", "on time", "low risk"))
    wants_safe_transfers = preferences.get("max_transfer_miss_probability") is not None

    constraints = []
    if wants_fastest:
        constraints.append("faster")
    if wants_less_full and not accepts_full:
        constraints.append("avoids near-capacity trains")
    if wants_fewer_changes:
        constraints.append("fewer line changes")
    if wants_low_delay:
        constraints.append("lower delay risk")
    if wants_safe_transfers:
        constraints.append("safer transfers")

    if not constraints:
        constraints.append("a better fit for my preferences")
    if len(constraints) == 1:
        return constraints[0]
    return ", ".join(constraints[:-1]) + f", or {constraints[-1]}"


def agent_a_secondary_constraint_request(persona):
    """Return non-time constraints Agent A should mention after valid routes exist."""
    preferences = persona.get("preferences", {})
    priority = preferences.get("priority", "").lower()
    switching = preferences.get("switching", "").lower()
    fullness = preferences.get("fullness", "").lower()
    reliability = preferences.get("reliability", "").lower()

    wants_less_full = any(term in fullness for term in ("less crowded", "dislikes", "crowded", "full", "packed"))
    accepts_full = any(term in fullness for term in ("does not mind", "secondary"))
    wants_fewer_changes = any(term in switching for term in ("fewer", "avoid", "avoiding", "unnecessary", "only for meaningful"))
    wants_low_delay = any(term in f"{priority} {reliability}" for term in ("delay", "reliable", "on time", "low risk"))
    wants_safe_transfers = preferences.get("max_transfer_miss_probability") is not None

    constraints = []
    if wants_less_full and not accepts_full:
        constraints.append("near-capacity trains")
    if wants_fewer_changes:
        constraints.append("line changes")
    if wants_low_delay:
        constraints.append("delay risk")
    if wants_safe_transfers:
        constraints.append("transfer-miss risk")
    if not constraints:
        return ""
    if len(constraints) == 1:
        return constraints[0]
    if len(constraints) == 2:
        return f"{constraints[0]} and {constraints[1]}"
    return ", ".join(constraints[:-1]) + f", and {constraints[-1]}"


def secondary_constraints_already_asked(conversation):
    asked = agent_a_secondary_constraint_request({"preferences": {"fullness": "crowded", "switching": "fewer", "reliability": "delay", "max_transfer_miss_probability": 0.2}})
    terms = [term.strip() for term in asked.replace("and", ",").split(",") if term.strip()]
    agent_a_text = " ".join(text.lower() for speaker, text in conversation if speaker == "Agent A")
    return any(term in agent_a_text for term in terms)


def valid_agent_b_route_count(conversation, scenario, allowed_modes):
    """Count distinct valid Agent B route proposals in the conversation."""
    from coop_navigation_sds.NaturalLanguageUnderstanding.interpreter import NaturalRouteInterpreter
    from coop_navigation_sds.TransportNetwork.routes import estimate_route_time

    interpreter = NaturalRouteInterpreter()
    routes = set()
    for speaker, text in conversation:
        if speaker != "Agent B":
            continue
        route = interpreter.interpret_reply(text, scenario)
        if not route:
            continue
        if estimate_route_time(
            route,
            scenario["start_time_min"],
            scenario["transfer_time_min"],
            allowed_modes=allowed_modes,
        ):
            routes.add(tuple(route))
    return len(routes)


def best_prior_route_duration(conversation, scenario):
    """Return the best earlier Agent B route duration, if any."""
    from coop_navigation_sds.NaturalLanguageUnderstanding.interpreter import NaturalRouteInterpreter
    from coop_navigation_sds.TransportNetwork.constraints import route_allowed_modes
    from coop_navigation_sds.TransportNetwork.routes import estimate_route_time

    interpreter = NaturalRouteInterpreter()
    allowed_modes = route_allowed_modes(scenario)
    durations = []
    for speaker, text in conversation:
        if speaker != "Agent B":
            continue
        route = interpreter.interpret_reply(text, scenario)
        if not route:
            continue
        estimate = estimate_route_time(
            route,
            scenario["start_time_min"],
            scenario["transfer_time_min"],
            allowed_modes=allowed_modes,
        )
        if estimate:
            arrival, _ = estimate
            durations.append(arrival - scenario["start_time_min"])
    return min(durations) if durations else None


def best_prior_route_matching_constraints(conversation, scenario, persona, stated_keys, duration_limit, constraint_route=None):
    """Return the best prior Agent B route satisfying the current revealed goals."""
    from coop_navigation_sds.NaturalLanguageUnderstanding.interpreter import NaturalRouteInterpreter
    from coop_navigation_sds.TransportNetwork.constraints import route_allowed_modes
    from coop_navigation_sds.TransportNetwork.routes import estimate_route_time

    interpreter = NaturalRouteInterpreter()
    allowed_modes = route_allowed_modes(scenario, persona)
    matches = []
    for speaker, text in conversation:
        if speaker != "Agent B":
            continue
        route = interpreter.interpret_reply(text, scenario)
        if not route:
            continue
        estimate = estimate_route_time(
            route,
            scenario["start_time_min"],
            scenario["transfer_time_min"],
            allowed_modes=allowed_modes,
        )
        if not estimate:
            continue
        arrival, steps = estimate
        duration = arrival - scenario["start_time_min"]
        if duration_limit is not None and duration > duration_limit:
            continue
        statuses = route_constraint_status(steps, persona, scenario, stated_keys, constraint_route=constraint_route)
        if unsatisfied_constraint_keys(statuses):
            continue
        matches.append((duration, route, steps))
    return min(matches, key=lambda item: item[0]) if matches else None
