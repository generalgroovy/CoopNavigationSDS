"""Exact, serializable provenance for language-model prompt calls."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone


PROMPT_POLICY_VERSION = "2026-06-28.1"


def _message_document(message):
    return {
        "role": str(getattr(message, "role", "")),
        "content": str(getattr(message, "content", "")),
    }


def begin_prompt_audit(*, agent, purpose, stage, turn, messages):
    """Create immutable prompt-call evidence before model generation starts."""
    message_documents = [_message_document(message) for message in messages]
    serialized = json.dumps(
        message_documents,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return {
        "prompt_policy_version": PROMPT_POLICY_VERSION,
        "agent": str(agent),
        "purpose": str(purpose),
        "stage": str(stage),
        "turn": int(turn),
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "messages": message_documents,
        "message_count": len(message_documents),
        "prompt_sha256": hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
        "raw_output": None,
        "cleaned_output": None,
        "accepted": False,
        "decision": "generation_pending",
        "error": None,
        "delivered_output": None,
        "delivery_source": None,
    }


def finish_prompt_audit(
    audit,
    *,
    raw_output=None,
    cleaned_output=None,
    accepted=False,
    decision,
    error=None,
):
    """Finalize one prompt audit with the model output and verifier decision."""
    audit.update({
        "raw_output": None if raw_output is None else str(raw_output),
        "cleaned_output": None if cleaned_output is None else str(cleaned_output),
        "accepted": bool(accepted),
        "decision": str(decision),
        "error": None if error is None else str(error),
    })
    return audit


def record_prompt_delivery(audit, output, source):
    """Record which guarded output was ultimately sent to text-to-speech."""
    audit.update({
        "delivered_output": str(output),
        "delivery_source": str(source),
    })
    return audit


def consume_prompt_audits(owner):
    """Return and clear an object's buffered audits without sharing mutable state."""
    audits = list(getattr(owner, "prompt_audits", ()) or ())
    if hasattr(owner, "prompt_audits"):
        owner.prompt_audits.clear()
    return audits
