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
)
from coop_navigation_sds.NaturalLanguageUnderstanding.interpreter import NaturalRouteInterpreter
from coop_navigation_sds.DialogManagement.stages import dialog_context


_CLOCK_WORD_VALUES = {
    "zero": 0, "oh": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30,
    "forty": 40, "fifty": 50,
}


def parse_heard_clock(text):
    """Parse numeric or naturally spoken 24-hour clock expressions."""
    numeric = re.search(r"\b([01]?\d|2[0-3]):([0-5]\d)\b", text)
    if numeric:
        return int(numeric.group(1)) * 60 + int(numeric.group(2))
    tokens = re.findall(r"[a-z]+", text.lower())
    for index, token in enumerate(tokens):
        hour = _CLOCK_WORD_VALUES.get(token)
        if hour is None or hour > 23 or index + 1 >= len(tokens):
            continue
        first_minute = tokens[index + 1]
        if first_minute in {"oh", "zero"} and index + 2 < len(tokens):
            minute = _CLOCK_WORD_VALUES.get(tokens[index + 2])
        else:
            minute = _CLOCK_WORD_VALUES.get(first_minute)
            if minute in {20, 30, 40, 50} and index + 2 < len(tokens):
                minute += _CLOCK_WORD_VALUES.get(tokens[index + 2], 0)
        if minute is not None and 0 <= minute <= 59:
            return hour * 60 + minute
    return None


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
    def stage(self):
        """Return the current explicit conversation stage."""
        return self.stage_override or self.context.stage

    @property
    def assistant_scenario(self):
        """Return journey facts Agent B actually received through ASR."""
        agent_a_text = " ".join(text for speaker, text in self.conversation if speaker == "Agent A")
        stations = []
        for match in STATION_PATTERN.finditer(agent_a_text):
            station = STATION_LOOKUP[match.group(0).lower()]
            if station not in stations:
                stations.append(station)
        start_time_min = parse_heard_clock(agent_a_text)
        if len(stations) < 2 or start_time_min is None:
            return None

        heard = dict(self.scenario)
        heard["start_station"] = stations[0]
        heard["destination_station"] = stations[1]
        heard["destination_stations"] = [stations[1]]
        heard["start_time_min"] = start_time_min
        heard["max_transfer_miss_probability"] = 1.0
        heard["max_delay_probability"] = 1.0
        lower = agent_a_text.lower()
        if not any(keyword in lower for keyword in ("ticket", "cannot take", "no bus", "no tram", "no metro")):
            heard.pop("ticket_modes", None)
        if not any(keyword in lower for keyword in ("walking", "walk", "on foot")):
            heard.pop("max_walking_min", None)
        return heard

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
            attempts = clarification_attempt_count(state.conversation, "Agent B")
            maximum = max(1, int(state.scenario.get("clarification_max_attempts", 2)))
            if attempts >= maximum:
                return (
                    "Let's reset the trip details. Say only: starting station, "
                    "destination station, and departure time."
                )
            if attempts:
                return (
                    "I still need the trip details. Please say each separately: "
                    "starting station, destination station, then departure time."
                )
            return clarification_question(
                state.latest_agent_a_text,
                "Please repeat the start station, destination, and time.",
            )

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

        return reply if reply else fallback_reply(
            "Agent B",
            scenario,
            route_index=state.turn,
            persona={},
            conversation=state.conversation,
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
            + " Do not repeat a line sequence word for word; move forward with a distinct candidate, comparison, correction, or confirmation."
        )
        return build_messages("Agent B", system_prompt, state.conversation)

    def run_repair(self, state: DialogState, repeated_reply: str) -> str:
        scenario = state.assistant_scenario
        system_prompt = (
            build_agent_b_system(scenario, None)
            + " "
            + build_agent_b_stage_instruction(state.stage, scenario["destination_station"])
            + f" Current stage: {state.stage.label}. "
            + " The previous draft was repetitive or did not give a complete connected route. Give a fresh valid route reply."
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
