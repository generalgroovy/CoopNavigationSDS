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


def _latest(conversation, speaker):
    return next(
        (str(text) for current_speaker, text in reversed(conversation) if current_speaker == speaker),
        "",
    )


def _response_focus(text):
    lower = text.lower()
    focus_terms = (
        ("confirmation", ("confirm", "final", "that works", "i'll take", "i will take")),
        ("reliability", ("reliable", "delay", "risk", "miss")),
        ("capacity", ("crowd", "capacity", "packed", "busy")),
        ("transfers", ("transfer", "change", "switch")),
        ("alternative", ("compare", "alternative", "another", "different", "else")),
        ("time", ("fast", "short", "time", "minutes", "quicker")),
        ("clarification", ("repeat", "missed", "unclear", "again")),
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
