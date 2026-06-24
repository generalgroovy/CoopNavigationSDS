"""Agent B controller pipeline."""
from dataclasses import dataclass
from difflib import SequenceMatcher
import re

from coop_navigation_sds.NaturalLanguageGeneration.caller.agents import STATION_LOOKUP, STATION_PATTERN, build_messages, clean_reply, fallback_reply
from coop_navigation_sds.NaturalLanguageGeneration.caller.prompting import build_agent_b_stage_instruction, build_agent_b_system
from coop_navigation_sds.Configuration.speech import MAX_REPAIR_ATTEMPTS, REPAIR_SIMILARITY_THRESHOLD
from coop_navigation_sds.NaturalLanguageUnderstanding.clarification import (
    clarification_attempt_count,
    clarification_question,
    clarification_target,
)
from coop_navigation_sds.NaturalLanguageUnderstanding.interpreter import NaturalRouteInterpreter
from coop_navigation_sds.DialogManagement.stages import agent_memory_view, dialog_context


_CLOCK_WORD_VALUES = {
    "zero": 0, "oh": 0, "o": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30,
    "forty": 40, "fifty": 50, "seventy": 70,
}

TRIP_SLOT_LABELS = {
    "start_station": "starting station",
    "destination_station": "destination station",
    "start_time_min": "departure time",
}


def _clock_token_value(token):
    token = str(token or "").casefold()
    if token.isdigit():
        return int(token)
    return _CLOCK_WORD_VALUES.get(token)


def _time_context(text, latest_agent_b_question=""):
    combined = f"{latest_agent_b_question} {text}".casefold()
    return bool(
        re.search(
            r"\b(at|time|departure|depart(?:ing)?|start(?:ing)? time|"
            r"leav(?:e|ing)|clock|morning|evening|said|use those|"
            r"trip details|when)\b",
            combined,
        )
    )


def _short_clock_answer(text):
    tokens = re.findall(r"[a-z]+|\d+", str(text or "").casefold())
    return 1 <= len(tokens) <= 4 and all(
        token in _CLOCK_WORD_VALUES or token.isdigit()
        for token in tokens
    )


def parse_heard_clock(text, allow_contextless=True):
    """Parse numeric or naturally spoken 24-hour clock expressions."""
    numeric = re.search(r"\b([01]?\d|2[0-3]):([0-5]\d)\b", str(text or ""))
    if numeric:
        return int(numeric.group(1)) * 60 + int(numeric.group(2))
    if not allow_contextless and not _time_context(text) and not _short_clock_answer(text):
        return None
    compact_numeric = re.search(
        r"(?:\bat\s+|\btime\s+|\bdeparture\s+time\s+|^)([01]?\d|2[0-3])\s*(?:[-,]|\s)\s*([0-5]?\d)\b",
        str(text or "").casefold(),
    )
    if compact_numeric:
        return int(compact_numeric.group(1)) * 60 + int(compact_numeric.group(2))
    compact_digits = re.search(r"\b([01]?\d|2[0-3])([0-5]\d)\b", str(text or "").casefold())
    if compact_digits:
        return int(compact_digits.group(1)) * 60 + int(compact_digits.group(2))
    tokens = re.findall(r"[a-z]+|\d+", str(text or "").lower())
    for index, token in enumerate(tokens):
        hour = _clock_token_value(token)
        if hour is None or hour > 23 or index + 1 >= len(tokens):
            continue
        first_minute = tokens[index + 1]
        if first_minute in {"oh", "o", "zero", "0", "to", "too", "two"} and index + 2 < len(tokens):
            minute = _clock_token_value(tokens[index + 2])
            if minute == 70:
                minute = 7
        elif first_minute in {"hundred"} and index + 2 < len(tokens):
            minute = _clock_token_value(tokens[index + 2])
        else:
            minute = _clock_token_value(first_minute)
            if minute in {20, 30, 40, 50} and index + 2 < len(tokens):
                minute += _clock_token_value(tokens[index + 2]) or 0
            elif minute == 70 and index + 1 < len(tokens):
                minute = 7
        if minute is not None and 0 <= minute <= 59:
            return hour * 60 + minute
    return None


def heard_trip_report(conversation):
    """Recover trip slots and evidence from Agent B's accumulated heard memory."""
    facts = {key: None for key in TRIP_SLOT_LABELS}
    evidence = {key: None for key in TRIP_SLOT_LABELS}
    latest_agent_b_question = ""
    for turn_index, (speaker, raw_text) in enumerate(conversation or (), start=1):
        text = str(raw_text or "")
        if speaker == "Agent B":
            latest_agent_b_question = text
            continue
        if speaker != "Agent A":
            continue

        mentions = [
            (match, STATION_LOOKUP[match.group(0).lower()])
            for match in STATION_PATTERN.finditer(text)
        ]
        for match, station in mentions:
            prefix = text[max(0, match.start() - 48):match.start()].casefold()
            if facts["start_station"] is None and re.search(
                r"(?:\bfrom|\bstart(?:ing)?(?: station)?(?: is| at)?|\bi(?:'m| am) at)\s*$",
                prefix,
            ):
                facts["start_station"] = station
                evidence["start_station"] = {
                    "turn": turn_index,
                    "text": text,
                    "method": "station_prefix",
                }
            strong_destination_prefix = re.search(
                r"(?:\bgoing to|\bheaded to|\bdestination(?: station)?(?: is)?|\breach)\s*$",
                prefix,
            )
            weak_destination_prefix = re.search(r"\bto\s*$", prefix)
            if (
                facts["destination_station"] is None
                and (strong_destination_prefix or (weak_destination_prefix and len(mentions) == 1))
            ):
                facts["destination_station"] = station
                evidence["destination_station"] = {
                    "turn": turn_index,
                    "text": text,
                    "method": "station_prefix",
                }

        if len(mentions) >= 2:
            if facts["start_station"] is None:
                facts["start_station"] = mentions[0][1]
                evidence["start_station"] = {
                    "turn": turn_index,
                    "text": text,
                    "method": "first_station_mention",
                }
            if facts["destination_station"] is None:
                facts["destination_station"] = mentions[-1][1]
                evidence["destination_station"] = {
                    "turn": turn_index,
                    "text": text,
                    "method": "last_station_mention",
                }
        elif len(mentions) == 1:
            station = mentions[0][1]
            question = latest_agent_b_question.casefold()
            if facts["start_station"] is None and "start" in question:
                facts["start_station"] = station
                evidence["start_station"] = {
                    "turn": turn_index,
                    "text": text,
                    "method": "targeted_station_repair",
                }
            elif facts["destination_station"] is None and any(
                term in question for term in ("destination", "going to", "where to")
            ):
                facts["destination_station"] = station
                evidence["destination_station"] = {
                    "turn": turn_index,
                    "text": text,
                    "method": "targeted_station_repair",
                }

        if facts["start_time_min"] is None:
            allow_contextless = (
                "departure time" in latest_agent_b_question.casefold()
                or _time_context(text, latest_agent_b_question)
                or _short_clock_answer(text)
                or len(mentions) >= 2
            )
            parsed_time = parse_heard_clock(text, allow_contextless=allow_contextless)
            if parsed_time is not None:
                facts["start_time_min"] = parsed_time
                evidence["start_time_min"] = {
                    "turn": turn_index,
                    "text": text,
                    "method": "clock_expression",
                    "contextual": bool(allow_contextless),
                }
    missing = tuple(key for key, value in facts.items() if value is None)
    return {
        "facts": facts,
        "evidence": evidence,
        "missing_slots": missing,
        "missing_labels": tuple(TRIP_SLOT_LABELS[key] for key in missing),
        "completeness": (
            sum(value is not None for value in facts.values()) / max(len(facts), 1)
        ),
    }


def heard_trip_facts(conversation):
    """Recover trip slots from Agent B's accumulated ASR-grounded memory."""
    return heard_trip_report(conversation)["facts"]


@dataclass
class DialogState:
    """Small state model passed between dialog controller stages for a single turn.
    """
    test_case: object
    conversation: list
    turn: int = 0
    scenario_override: dict | None = None
    persona_override: dict | None = None
    stage_override: object | None = None

    @property
    def scenario(self):
        """Scenario method for this module's MVC responsibility.

        Returns:
            The computed value or side effect documented by the implementation.
        """
        return self.scenario_override or self.test_case.scenario

    @property
    def persona(self):
        """Persona method for this module's MVC responsibility.

        Returns:
            The computed value or side effect documented by the implementation.
        """
        return self.persona_override or self.test_case.persona

    @property
    def context(self):
        """Return conversational memory derived only from heard utterances."""
        return dialog_context(self.conversation)

    @property
    def memory(self):
        """Return Agent B's persistent intended/heard memory view."""
        return agent_memory_view(
            "Agent B",
            self.conversation,
            scenario=self.assistant_scenario,
            persona=self.persona,
        )

    @property
    def stage(self):
        """Return the current explicit conversation stage."""
        return self.stage_override or self.context.stage

    @property
    def assistant_scenario(self):
        """Return journey facts Agent B actually received through ASR."""
        agent_a_text = " ".join(text for speaker, text in self.conversation if speaker == "Agent A")
        facts = self.heard_trip_state["facts"]
        if any(value is None for value in facts.values()):
            return None

        heard = dict(self.scenario)
        heard.update(facts)
        heard["destination_stations"] = [facts["destination_station"]]
        heard["max_transfer_miss_probability"] = 1.0
        heard["max_delay_probability"] = 1.0
        lower = agent_a_text.lower()
        if not any(keyword in lower for keyword in ("ticket", "cannot take", "no bus", "no tram", "no metro")):
            heard.pop("ticket_modes", None)
        if not any(keyword in lower for keyword in ("walking", "walk", "on foot")):
            heard.pop("max_walking_min", None)
        return heard

    @property
    def missing_trip_slots(self):
        """Return trip facts still absent from Agent B's heard memory."""
        return tuple(self.heard_trip_state["missing_labels"])

    @property
    def heard_trip_state(self):
        """Return fact recovery state used by Agent B before planning."""
        return heard_trip_report(self.conversation)

    def trip_clarification_prompt(self):
        """Return the shared repair prompt for every Agent B backend.

        Agent B never terminates the call. If repeated targeted repair fails,
        it asks for a structured reset while Agent A/controller policy decides
        whether the conversation should continue or close.
        """
        maximum = max(1, int(self.scenario.get("clarification_max_attempts", 2)))
        missing = self.missing_trip_slots
        if missing:
            prompt = f"I missed the {missing[0]}. Please say only that."
            target = clarification_target(prompt)
            attempts = clarification_attempt_count(
                self.conversation, "Agent B", target=target
            )
            generic_attempts = clarification_attempt_count(
                self.conversation, "Agent B", target="trip_details"
            )
            if attempts >= maximum or generic_attempts >= maximum:
                if len(missing) > 1:
                    return (
                        "Let's reset the trip details. Say only: starting station, "
                        "destination station, and departure time."
                    )
                return (
                    f"I still need the {missing[0]}. "
                    f"Please say only the {missing[0]}."
                )
            return prompt
        attempts = clarification_attempt_count(self.conversation, "Agent B")
        if attempts >= maximum:
            return (
                "Let's reset the trip details. Say only: starting station, "
                "destination station, and departure time."
            )
        return clarification_question(
            self.latest_agent_a_text,
            "Please repeat the start station, destination, and time.",
        )

    @property
    def latest_agent_a_text(self):
        """Return the latest heard caller utterance."""
        return next((text for speaker, text in reversed(self.conversation) if speaker == "Agent A"), "")


class VerbalTransformationPipeline:
    """Transforms dialog state into a verbal Agent B response."""

    def __init__(self, model_adapter):
        """  init   method for this module's MVC responsibility.

        Args:
            model_adapter: Input value used by `__init__`; see the function signature and caller context for the expected type.

        Returns:
            The computed value or side effect documented by the implementation.
        """
        self.model_adapter = model_adapter
        self.prompt_builder = AgentBPromptStage()
        self.generator = ModelGenerationStage(model_adapter)
        self.cleaner = VerbalCleanupStage()
        self.route_interpreter = NaturalRouteInterpreter()

    def run_agent_b(self, state: DialogState) -> str:
        """Run agent b method for this module's MVC responsibility.

        Args:
            state: Input value used by `run_agent_b`; see the function signature and caller context for the expected type.

        Returns:
            The computed value or side effect documented by the implementation.
        """
        scenario = state.assistant_scenario
        if scenario is None:
            return state.trip_clarification_prompt()

        messages = self.prompt_builder.run(state)
        raw_reply = self.generator.run(messages)

        for attempt in range(MAX_REPAIR_ATTEMPTS + 1):
            reply = self.cleaner.run(raw_reply)
            if reply and not STATION_PATTERN.search(reply):
                reply = ""
            if reply and not self.reply_reaches_goal(reply, scenario):
                reply = ""
            if reply and not self.cleaner.repeats_prior_agent_b(reply, state.conversation):
                return reply

            if attempt >= MAX_REPAIR_ATTEMPTS:
                break

            messages = self.prompt_builder.run_repair(state, reply)
            raw_reply = self.generator.run(messages)

        phase_fallback = self.prompt_builder.phase_fallback(state)
        if not self.cleaner.repeats_prior_agent_b(phase_fallback, state.conversation):
            return phase_fallback

        return (
            "I have no new distinct viable route. The best earlier route "
            "still fits the current request."
        )

    def reply_reaches_goal(self, reply, scenario):
        route = self.route_interpreter.interpret_reply(reply, scenario)
        return (
            bool(route)
            and route[0] == scenario["start_station"]
            and route[-1] == scenario["destination_station"]
        )


class AgentBPromptStage:
    def run(self, state: DialogState) -> str:
        scenario = state.assistant_scenario
        system_prompt = (
            build_agent_b_system(scenario, None)
            + " "
            + build_agent_b_stage_instruction(state.stage, scenario["destination_station"])
            + f" Current stage: {state.stage.label}. "
            + " Resolve only the latest pending request. Do not repeat a prior route or clarification. "
            + state.memory.prompt_summary()
        )
        return build_messages("Agent B", system_prompt, state.conversation)

    def run_repair(self, state: DialogState, repeated_reply: str) -> str:
        scenario = state.assistant_scenario
        system_prompt = (
            build_agent_b_system(scenario, None)
            + " "
            + build_agent_b_stage_instruction(state.stage, scenario["destination_station"])
            + f" Current stage: {state.stage.label}. "
            + " The previous draft repeated information or lacked a complete route. "
            + "Give one fresh connected candidate; if none remains, recommend the best earlier route."
            + " "
            + state.memory.prompt_summary()
        )
        repair_history = list(state.conversation)
        if repeated_reply:
            repair_history.append(("Agent A", "That sounds like a repeat. Can you say it more clearly or update the route?"))
        return build_messages("Agent B", system_prompt, repair_history)

    def phase_fallback(self, state: DialogState) -> str:
        return fallback_reply(
            "Agent B",
            state.assistant_scenario,
            route_index=state.turn,
            persona={},
            conversation=state.conversation,
        )


class ModelGenerationStage:
    def __init__(self, model_adapter):
        self.model_adapter = model_adapter

    def run(self, messages) -> str:
        if hasattr(self.model_adapter, "generate_messages"):
            return self.model_adapter.generate_messages(messages)
        return self.model_adapter.generate(build_prompt_from_messages(messages))


def build_prompt_from_messages(messages):
    """Fallback serializer for legacy deterministic adapters."""
    from coop_navigation_sds.NaturalLanguageGeneration.models import messages_to_prompt

    return messages_to_prompt(messages)


class VerbalCleanupStage:
    def run(self, raw_reply: str) -> str:
        return clean_reply(raw_reply)

    def repeats_prior_agent_b(self, reply: str, conversation) -> bool:
        normalized_reply = self._normalize(reply)
        if not normalized_reply:
            return False

        for speaker, text in conversation:
            if speaker != "Agent B":
                continue

            prior = self._normalize(text)
            if prior == normalized_reply:
                return True

            if SequenceMatcher(None, prior, normalized_reply).ratio() >= REPAIR_SIMILARITY_THRESHOLD:
                return True

        return False

    def _normalize(self, text: str) -> str:
        return re.sub(r"\s+", " ", text.strip().lower())
