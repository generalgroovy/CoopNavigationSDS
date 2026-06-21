"""Clarification helpers for compact spoken repair turns."""
from difflib import SequenceMatcher
import re

from coop_navigation_sds.TransportNetwork.network import LINES, STATIONS


COMMON_WORDS = {
    "about",
    "again",
    "answer",
    "catch",
    "change",
    "could",
    "destination",
    "does",
    "event",
    "from",
    "going",
    "heard",
    "line",
    "minutes",
    "please",
    "repeat",
    "route",
    "said",
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
        return f"I heard '{word}' unclearly. Did you mean {options[0]} or {options[1]}? {fallback}"
    return f"I heard '{word}' unclearly. Did you mean {options[0]}? {fallback}"


def is_clarification_request(text):
    """Return whether an utterance asks the other speaker to repair hearing."""
    folded = str(text or "").casefold()
    return any(
        phrase in folded
        for phrase in (
            "did not catch",
            "didn't catch",
            "heard ",
            "unclearly",
            "please repeat",
            "did you mean",
            "say that again",
        )
    )


def clarification_attempt_count(conversation, speaker=None):
    """Count prior repair requests, optionally for one speaker."""
    return sum(
        1
        for turn_speaker, text in conversation
        if (speaker is None or turn_speaker == speaker) and is_clarification_request(text)
    )
