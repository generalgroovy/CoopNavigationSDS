"""Clarification helpers for compact spoken repair turns."""
from difflib import SequenceMatcher
import re

from coop_navigation_sds.TransportNetwork.network import LINES, STATIONS


COMMON_WORDS = {
    "about",
    "again",
    "answer",
    "can",
    "catch",
    "change",
    "could",
    "destination",
    "does",
    "event",
    "from",
    "going",
    "help",
    "heard",
    "line",
    "keep",
    "minutes",
    "please",
    "repeat",
    "route",
    "said",
    "simple",
    "start",
    "station",
    "take",
    "that",
    "although",
    "time",
    "what",
    "which",
    "with",
}


def route_vocabulary():
    """Return words that can be clarified in the current route domain."""
    return tuple(STATIONS) + tuple(LINES) + (
        "route",
        "start",
        "destination",
        "transfer",
        "change",
        "minutes",
    )


def misunderstood_word_options(text, vocabulary=None, limit=2, threshold=0.66):
    """Return one unclear word and likely intended vocabulary alternatives."""
    words = re.findall(r"[A-Za-z][A-Za-z-]{2,}", text or "")
    candidates = vocabulary or route_vocabulary()
    candidate_lookup = {candidate.casefold(): candidate for candidate in candidates}
    for word in words:
        folded = word.casefold()
        if folded in COMMON_WORDS or folded in candidate_lookup:
            continue
        scored = sorted(
            (
                (SequenceMatcher(None, folded, candidate.casefold()).ratio(), candidate)
                for candidate in candidates
            ),
            reverse=True,
        )
        options = []
        for score, candidate in scored:
            if score < threshold:
                break
            if candidate not in options:
                options.append(candidate)
            if len(options) >= limit:
                break
        if options:
            return word, options
    return None, []


def clarification_question(text, fallback, vocabulary=None):
    """Build a short natural clarification question from likely word confusion."""
    word, options = misunderstood_word_options(text, vocabulary=vocabulary)
    if not word:
        return fallback
    if len(options) >= 2:
        return f"Did you mean {options[0]} or {options[1]} by '{word}'?"
    return f"Did you mean {options[0]} by '{word}'?"


def is_clarification_request(text):
    """Return whether an utterance asks the other speaker to repair hearing."""
    folded = str(text or "").casefold()
    return any(
        phrase in folded
        for phrase in (
            "did not catch",
            "didn't catch",
            "heard ",
            "missed",
            "unclearly",
            "please repeat",
            "did you mean",
            "say that again",
        )
    )


def clarification_target(text):
    """Return the slot or term targeted by a clarification request."""
    folded = str(text or "").casefold()
    if not is_clarification_request(folded):
        return None
    if "start" in folded and "destination" in folded and "time" in folded:
        return "trip_details"
    if "starting station" in folded or re.search(r"\bstart(?:ing)?\b", folded):
        return "start_station"
    if "destination station" in folded or "destination" in folded or "where to" in folded or "going to" in folded:
        return "destination_station"
    if "departure time" in folded or "what time" in folded or "leaving" in folded:
        return "start_time_min"
    match = re.search(r"\bheard\s+['\"]?([^?'\"]+)", str(text or ""), flags=re.IGNORECASE)
    if match:
        heard = " ".join(re.findall(r"[A-Za-z0-9]+", match.group(1).casefold()))
        return f"heard:{heard}" if heard else "word"
    match = re.search(r"\bdid you mean\s+(.+?)(?:\?| by\b|$)", str(text or ""), flags=re.IGNORECASE)
    if match:
        options = " ".join(re.findall(r"[A-Za-z0-9]+", match.group(1).casefold()))
        return f"term:{options}" if options else "word"
    return "clarification"


def clarification_attempt_count(conversation, speaker=None, target=None):
    """Count prior repair requests, optionally for one speaker and target."""
    return sum(
        1
        for turn_speaker, text in conversation
        if (
            (speaker is None or turn_speaker == speaker)
            and is_clarification_request(text)
            and (target is None or clarification_target(text) == target)
        )
    )


def clarification_signature(text):
    """Return a stable signature for one spoken repair request."""
    if not is_clarification_request(text):
        return None
    target = clarification_target(text)
    words = re.findall(r"[a-z0-9]+", str(text or "").casefold())
    return f"{target or 'clarification'}:" + " ".join(words)


def clarification_was_asked(conversation, speaker, question):
    """Return whether this speaker already issued the same repair request."""
    signature = clarification_signature(question)
    if not signature:
        return False
    return any(
        turn_speaker == speaker and clarification_signature(text) == signature
        for turn_speaker, text in conversation or ()
    )


def transcript_repair_question(corrections):
    """Turn a domain-critical transcript normalization into an explicit repair turn."""
    vocabulary = {word.casefold() for word in route_vocabulary()}
    for correction in corrections or ():
        source = " ".join(correction.get("source_tokens") or ())
        target = " ".join(correction.get("target_tokens") or ())
        target_words = re.findall(r"[A-Za-z0-9]+", target.casefold())
        if not source or not target or not any(word in vocabulary for word in target_words):
            continue
        return f"I heard '{source}'. Did you mean '{target}'?"
    return None


def is_clarification_confirmation(text):
    """Return whether text is a concise answer to a hearing-repair question."""
    return bool(re.match(r"^(?:yes,?\s+|no,?\s+)?i meant\b", str(text or "").strip(), re.IGNORECASE))


def last_substantive_agent_b_utterance(conversation):
    """Return the latest Agent B turn that is not a repair confirmation."""
    return next(
        (
            text
            for speaker, text in reversed(conversation or ())
            if speaker == "Agent B" and not is_clarification_confirmation(text)
        ),
        "",
    )


def _term_from_prior_speech(heard_option, prior_spoken_text):
    prior = str(prior_spoken_text or "")
    prior_terms = [
        term
        for term in route_vocabulary()
        if re.search(rf"(?<![A-Za-z0-9]){re.escape(term)}(?![A-Za-z0-9])", prior, re.IGNORECASE)
    ]
    heard_words = re.findall(r"[A-Za-z0-9]+", str(heard_option or ""))
    scored = [
        (SequenceMatcher(None, word.casefold(), term.casefold()).ratio(), term)
        for word in heard_words
        for term in prior_terms
        if word.casefold() not in COMMON_WORDS
    ]
    if not scored:
        return None
    score, term = max(scored)
    return term if score >= 0.6 else None


def clarification_confirmation(conversation, prior_spoken_text=None):
    """Answer one explicit hearing repair using Agent B's own prior wording."""
    if not conversation or conversation[-1][0] != "Agent A":
        return None
    latest = conversation[-1][1]
    if not is_clarification_request(latest):
        return None
    match = re.search(r"did you mean\s+['\"]?([^?'\"]+)", latest, flags=re.IGNORECASE)
    heard_option = match.group(1).strip() if match else "that wording"
    prior = prior_spoken_text or last_substantive_agent_b_utterance(conversation[:-1])
    intended = _term_from_prior_speech(heard_option, prior) or heard_option
    if intended.casefold() == heard_option.casefold():
        return f"Yes, I meant {intended}."
    return f"I meant {intended}."
