"""Agent B controller pipeline."""
from dataclasses import dataclass
from difflib import SequenceMatcher
import re

from minillama.agent_a.agents import STATION_PATTERN, build_messages, clean_reply, fallback_reply
from minillama.agent_a.prompting import build_agent_b_phase_instruction, build_agent_b_system
from minillama.agent_b.config import MAX_REPAIR_ATTEMPTS, REPAIR_SIMILARITY_THRESHOLD
from minillama.evaluation.route_interpreter import NaturalRouteInterpreter


@dataclass
class DialogState:
    """Small state model passed between dialog controller stages for a single turn.
    """
    test_case: object
    conversation: list
    turn: int = 0
    scenario_override: dict | None = None
    persona_override: dict | None = None

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
        messages = self.prompt_builder.run(state)
        raw_reply = self.generator.run(messages)

        for attempt in range(MAX_REPAIR_ATTEMPTS + 1):
            reply = self.cleaner.run(raw_reply)
            if reply and not STATION_PATTERN.search(reply):
                reply = ""
            if reply and not self.reply_reaches_goal(reply, state.scenario):
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

        return reply if reply else fallback_reply("Agent B", state.scenario, route_index=state.turn, persona=state.persona, conversation=state.conversation)

    def reply_reaches_goal(self, reply, scenario):
        route = self.route_interpreter.interpret_reply(reply, scenario)
        return (
            bool(route)
            and route[0] == scenario["start_station"]
            and route[-1] == scenario["destination_station"]
        )


class AgentBPromptStage:
    def run(self, state: DialogState) -> str:
        system_prompt = (
            build_agent_b_system(state.scenario, state.persona)
            + " "
            + build_agent_b_phase_instruction(state.turn, state.scenario["destination_station"])
            + " Do not repeat a line sequence word for word; move forward with a distinct candidate, comparison, correction, or confirmation."
        )
        return build_messages("Agent B", system_prompt, state.conversation)

    def run_repair(self, state: DialogState, repeated_reply: str) -> str:
        system_prompt = (
            build_agent_b_system(state.scenario, state.persona)
            + " "
            + build_agent_b_phase_instruction(state.turn, state.scenario["destination_station"])
            + " The previous draft was repetitive or did not give a complete connected route. Give a fresh valid route reply."
        )
        repair_history = list(state.conversation)
        if repeated_reply:
            repair_history.append(("Agent A", "That sounds like a repeat. Can you say it more clearly or update the route?"))
        return build_messages("Agent B", system_prompt, repair_history)

    def phase_fallback(self, state: DialogState) -> str:
        return fallback_reply("Agent B", state.scenario, route_index=state.turn, persona=state.persona, conversation=state.conversation)


class ModelGenerationStage:
    def __init__(self, model_adapter):
        self.model_adapter = model_adapter

    def run(self, messages) -> str:
        if hasattr(self.model_adapter, "generate_messages"):
            return self.model_adapter.generate_messages(messages)
        return self.model_adapter.generate(build_prompt_from_messages(messages))


def build_prompt_from_messages(messages):
    """Fallback serializer for legacy deterministic adapters."""
    from minillama.model.model_adapters import messages_to_prompt

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
