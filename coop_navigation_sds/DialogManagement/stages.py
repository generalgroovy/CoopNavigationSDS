"""Conversation-stage and context inference shared by both dialog agents."""
from dataclasses import dataclass
from enum import Enum


class ConversationStage(str, Enum):
    """Observable stages of one route-planning conversation."""

    DISCOVERY = "discovery"
    PROPOSAL = "proposal"
    COMPARISON = "comparison"
    REFINEMENT = "refinement"
    CONFIRMATION = "confirmation"
    CLOSED = "closed"

    @property
    def label(self):
        return self.value.replace("_", " ").title()


@dataclass(frozen=True)
class DialogContext:
    """Compact conversational memory derived from the heard transcript."""

    stage: ConversationStage
    latest_agent_a: str
    latest_agent_b: str
    agent_a_turn_count: int
    agent_b_turn_count: int
    response_focus: str
    conversation: tuple


@dataclass(frozen=True)
class AgentMemoryView:
    """One agent's persistent view of intended speech and heard transcripts."""

    owner: str
    latest_spoken: str
    latest_heard: str
    pending_focus: str
    task_variables: dict
    missing_task_variables: tuple
    active_constraints: tuple
    active_constraint_details: dict
    current_route: tuple
    current_route_summary: str
    current_route_duration_min: int | None
    clarification_requests: tuple
    conversation: tuple

    def prompt_summary(self):
        """Return a compact grounding reminder for a language-model prompt."""
        spoken = self.latest_spoken or "none yet"
        heard = self.latest_heard or "none yet"
        route = self.current_route_summary or "no accepted route candidate yet"
        constraints = ", ".join(self.active_constraints) if self.active_constraints else "none stated yet"
        task = _task_variable_summary(self.task_variables, self.missing_task_variables)
        return (
            f"Your memory: last thing you intended to say: {spoken!r}; "
            f"last thing you heard through speech recognition: {heard!r}; "
            f"recognized task variables: {task}; "
            f"current request focus: {self.pending_focus}; "
            f"current route candidate: {route}; "
            f"currently active caller constraints: {constraints}. "
            "Continue from that state, preserve resolved facts, and do not restart a resolved clarification."
        )


def _latest(conversation, speaker):
    return next(
        (str(text) for current_speaker, text in reversed(conversation) if current_speaker == speaker),
        "",
    )


def _format_clock(minutes):
    if minutes is None:
        return None
    try:
        value = int(minutes)
    except (TypeError, ValueError):
        return None
    return f"{value // 60:02d}:{value % 60:02d}"


def _task_variable_summary(task_variables, missing):
    labels = {
        "start_station": "start",
        "destination_station": "destination",
        "start_time_min": "time",
    }
    parts = []
    for key, label in labels.items():
        value = (task_variables or {}).get(key)
        if key == "start_time_min":
            value = _format_clock(value)
        parts.append(f"{label}={value or 'unknown'}")
    if missing:
        parts.append("missing=" + ", ".join(str(item) for item in missing))
    return "; ".join(parts)


def _task_variables_for(owner, conversation, scenario=None):
    """Return critical task variables from the selected agent's own perspective."""
    if owner == "Agent A":
        values = {
            "start_station": (scenario or {}).get("start_station"),
            "destination_station": (scenario or {}).get("destination_station"),
            "start_time_min": (scenario or {}).get("start_time_min"),
        }
        missing = tuple(key for key, value in values.items() if value is None)
        return values, missing
    try:
        from coop_navigation_sds.NaturalLanguageGeneration.assistant.pipeline import heard_trip_report
        report = heard_trip_report(conversation or ())
        return dict(report.get("facts") or {}), tuple(report.get("missing_slots") or ())
    except Exception:
        values = {"start_station": None, "destination_station": None, "start_time_min": None}
        return values, tuple(values)


def _constraint_detail(key, scenario=None, persona=None):
    preferences = (persona or {}).get("preferences", {})
    if key == "tickets":
        modes = (scenario or {}).get("ticket_modes")
        return ",".join(modes) if modes else "not recognized"
    if key == "walking":
        limit = (scenario or {}).get("max_walking_min") or preferences.get("max_walking_min")
        return f"max {limit} min" if limit is not None else "mentioned"
    if key == "delay":
        risk = (scenario or {}).get("max_delay_risk") or preferences.get("max_delay_risk")
        return f"max {risk} risk" if risk else "mentioned"
    if key == "transfer_miss":
        risk = (scenario or {}).get("max_transfer_risk") or preferences.get("max_transfer_risk")
        return f"max {risk} risk" if risk else "mentioned"
    if key == "transfers":
        return "fewer line changes"
    if key == "fullness":
        return "avoid near-capacity trains"
    return "mentioned"


def _response_focus(text):
    lower = text.lower()
    focus_terms = (
        ("confirmation", ("confirm", "final", "that works", "i'll take", "i will take")),
        ("reliability", ("reliable", "delay", "risk", "miss")),
        ("capacity", ("crowd", "capacity", "packed", "busy")),
        ("transfers", ("transfer", "change", "switch")),
        ("alternative", ("compare", "alternative", "another", "different", "else")),
        ("time", ("fast", "short", "time", "minutes", "quicker")),
        ("clarification", ("repeat", "missed", "unclear", "again", "did you mean", "heard")),
    )
    return next(
        (focus for focus, terms in focus_terms if any(term in lower for term in terms)),
        "route",
    )


def dialog_context(conversation):
    """Infer the current stage and latest conversational focus."""
    turns = list(conversation or [])
    latest_agent_a = _latest(turns, "Agent A")
    latest_agent_b = _latest(turns, "Agent B")
    agent_a_turn_count = sum(1 for speaker, _text in turns if speaker == "Agent A")
    agent_b_turn_count = sum(1 for speaker, _text in turns if speaker == "Agent B")
    focus = _response_focus(latest_agent_a)
    lower = latest_agent_a.lower()

    if not turns or not latest_agent_a:
        stage = ConversationStage.DISCOVERY
    elif any(term in lower for term in ("thanks, i'll take", "thanks, i will take", "goodbye")):
        stage = ConversationStage.CLOSED
    elif focus == "confirmation":
        stage = ConversationStage.CONFIRMATION
    elif agent_b_turn_count == 0:
        stage = ConversationStage.PROPOSAL
    elif focus == "alternative":
        stage = ConversationStage.COMPARISON
    elif focus in {"reliability", "capacity", "transfers"}:
        stage = ConversationStage.REFINEMENT
    elif agent_b_turn_count == 1:
        stage = ConversationStage.COMPARISON
    else:
        stage = ConversationStage.REFINEMENT

    return DialogContext(
        stage=stage,
        latest_agent_a=latest_agent_a,
        latest_agent_b=latest_agent_b,
        agent_a_turn_count=agent_a_turn_count,
        agent_b_turn_count=agent_b_turn_count,
        response_focus=focus,
        conversation=tuple(turns),
    )


def _current_route_memory(conversation, scenario=None, persona=None):
    """Return the latest route candidate visible in this transcript."""
    if not scenario:
        return (), "", None

    try:
        from coop_navigation_sds.NaturalLanguageUnderstanding.interpreter import NaturalRouteInterpreter
        from coop_navigation_sds.TransportNetwork.constraints import (
            dialog_route_allowed_modes,
            stated_constraint_keys,
        )
        from coop_navigation_sds.TransportNetwork.routes import (
            estimate_route_time,
            route_line_change_count,
            route_station_sequence,
        )
    except Exception:
        return (), "", None

    interpreter = NaturalRouteInterpreter()
    stated_keys = stated_constraint_keys(conversation or [])
    allowed_modes = dialog_route_allowed_modes(scenario, persona or {}, stated_keys)
    for speaker, text in reversed(tuple(conversation or ())):
        if speaker != "Agent B":
            continue
        route = interpreter.interpret_reply(str(text or ""), scenario)
        if not route:
            continue
        if route[0] != scenario.get("start_station") or route[-1] != scenario.get("destination_station"):
            continue
        estimate = estimate_route_time(
            route,
            scenario.get("start_time_min", 0),
            scenario.get("transfer_time_min", 0),
            allowed_modes=allowed_modes,
        )
        if estimate:
            arrival, steps = estimate
            duration = arrival - scenario.get("start_time_min", 0)
            changes = route_line_change_count(steps)
            change_text = "no changes" if changes == 0 else "1 change" if changes == 1 else f"{changes} changes"
            station_text = " to ".join(route_station_sequence(steps))
            return tuple(route), f"{station_text}, {duration} minutes, {change_text}", duration
        return tuple(route), " to ".join(route), None
    return (), "", None


def agent_memory_view(owner, conversation, scenario=None, persona=None):
    """Build a memory view from one agent's perspective-specific history."""
    turns = tuple(conversation or ())
    other = "Agent B" if owner == "Agent A" else "Agent A"
    latest_spoken = _latest(turns, owner)
    latest_heard = _latest(turns, other)
    try:
        from coop_navigation_sds.TransportNetwork.constraints import stated_constraint_keys
        active_constraints = tuple(stated_constraint_keys(turns))
    except Exception:
        active_constraints = ()
    task_variables, missing_task_variables = _task_variables_for(
        owner,
        turns,
        scenario=scenario,
    )
    active_constraint_details = {
        key: _constraint_detail(key, scenario=scenario, persona=persona)
        for key in active_constraints
    }
    current_route, route_summary, route_duration = _current_route_memory(
        turns,
        scenario=scenario,
        persona=persona,
    )
    requests = tuple(
        text
        for speaker, text in turns
        if speaker == owner and _response_focus(text) == "clarification"
    )
    return AgentMemoryView(
        owner=owner,
        latest_spoken=latest_spoken,
        latest_heard=latest_heard,
        pending_focus=_response_focus(latest_heard),
        task_variables=task_variables,
        missing_task_variables=missing_task_variables,
        active_constraints=active_constraints,
        active_constraint_details=active_constraint_details,
        current_route=current_route,
        current_route_summary=route_summary,
        current_route_duration_min=route_duration,
        clarification_requests=requests,
        conversation=turns,
    )
