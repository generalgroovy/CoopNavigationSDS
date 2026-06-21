"""Conservative post-ASR normalization for the navigation domain."""
from difflib import SequenceMatcher
import re

from coop_navigation_sds.TransportNetwork.network import LINES, STATIONS


TRANSIT_ALIASES = {
    "harbour": "Harbor",
    "hamper": "Harbor",
    "harder": "Harbor",
    "harper": "Harbor",
    "rude": "route",
    "root": "route",
    "rout": "route",
    "rowed": "route",
    "roots": "routes",
    "wrote": "route",
}

NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "twenty-one": 21, "twenty-two": 22,
    "twenty-three": 23, "twenty-four": 24, "twenty-five": 25,
    "twenty-six": 26, "twenty-seven": 27, "twenty-eight": 28,
    "twenty-nine": 29, "thirty": 30,
}


def normalize_spoken_line_codes(text):
    """Repair a line code only when its transport mode supplies context."""
    result = str(text or "")
    specifications = (
        ("metro", "M", ("m", "em", "and")),
        ("tram", "T", ("t", "tee", "tea")),
        ("bus", "B", ("b", "bee", "be")),
    )
    number_pattern = "|".join(
        sorted((re.escape(word) for word in NUMBER_WORDS), key=len, reverse=True)
    ) + r"|\d{1,2}"
    for mode, prefix, heard_prefixes in specifications:
        prefix_pattern = "|".join(re.escape(value) for value in heard_prefixes)
        pattern = re.compile(
            rf"\b({mode}\s+(?:line\s+)?)(?:{prefix_pattern})\s*({number_pattern})\b",
            flags=re.IGNORECASE,
        )

        def replace(match):
            number_text = match.group(2).casefold()
            number = NUMBER_WORDS.get(number_text, int(number_text) if number_text.isdigit() else 0)
            line_name = f"{prefix}{number}"
            return f"{match.group(1)}{line_name}" if line_name in LINES else match.group(0)

        result = pattern.sub(replace, result)
    return result


def transit_vocabulary():
    """Return public names and task words that recognizers may confuse."""
    return tuple(STATIONS) + tuple(LINES) + (
        "route",
        "routes",
        "station",
        "transfer",
        "destination",
    )


def normalize_transit_transcript(text, threshold=0.86, vocabulary=None):
    """Repair only close domain terms; never substitute the source utterance."""
    terms = tuple(vocabulary or transit_vocabulary())
    single_terms = tuple(term for term in terms if " " not in term and len(term) >= 4)
    tokens = re.findall(
        r"\b[\w'-]+\b|[^\w\s]",
        normalize_spoken_line_codes(text),
    )
    normalized = []
    for token in tokens:
        folded = token.casefold()
        if folded in TRANSIT_ALIASES:
            normalized.append(TRANSIT_ALIASES[folded])
            continue
        if any(folded == term.casefold() for term in terms):
            normalized.append(token)
            continue
        if len(token) < 4 or not token.isalpha():
            normalized.append(token)
            continue
        best = max(
            single_terms,
            key=lambda term: SequenceMatcher(None, folded, term.casefold()).ratio(),
            default=None,
        )
        ratio = SequenceMatcher(None, folded, best.casefold()).ratio() if best else 0.0
        normalized.append(best if best and ratio >= float(threshold) else token)
    result = re.sub(r"\s+([,.;:!?])", r"\1", " ".join(normalized)).strip()
    return re.sub(r"(?<=\d):\s+(?=\d)", ":", result)


def transcript_token_changes(source, target):
    """Return case-insensitive word-level replacements, deletions, and insertions."""
    source_tokens = re.findall(r"[\w'-]+", str(source or ""))
    target_tokens = re.findall(r"[\w'-]+", str(target or ""))
    matcher = SequenceMatcher(
        None,
        [token.casefold() for token in source_tokens],
        [token.casefold() for token in target_tokens],
        autojunk=False,
    )
    changes = []
    for operation, source_start, source_end, target_start, target_end in matcher.get_opcodes():
        if operation == "equal":
            continue
        changes.append({
            "operation": operation,
            "source_tokens": source_tokens[source_start:source_end],
            "target_tokens": target_tokens[target_start:target_end],
        })
    return changes
