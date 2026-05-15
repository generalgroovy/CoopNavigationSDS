"""Structured event models used by session monitoring and logging."""
from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class StructuredEvent:
    """Base structured event for a monitored session."""

    kind: str
    session_id: str
    timestamp: float
    name: str
    payload: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class ProgramSegmentEvent(StructuredEvent):
    """Structured event for a monitored program segment."""

    def __init__(self, session_id, timestamp, name, payload=None):
        super().__init__(
            kind="program.segment",
            session_id=session_id,
            timestamp=timestamp,
            name=name,
            payload=payload or {},
        )


@dataclass(frozen=True)
class ConversationStepEvent(StructuredEvent):
    """Structured event for a single conversation step."""

    def __init__(self, session_id, timestamp, name, payload=None):
        super().__init__(
            kind="conversation.step",
            session_id=session_id,
            timestamp=timestamp,
            name=name,
            payload=payload or {},
        )


@dataclass(frozen=True)
class SystemEvent(StructuredEvent):
    """Structured event for generic session events."""

    def __init__(self, session_id, timestamp, name, payload=None):
        super().__init__(
            kind="system",
            session_id=session_id,
            timestamp=timestamp,
            name=name,
            payload=payload or {},
        )
