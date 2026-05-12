"""Agent B controller pipeline. It builds prompts, calls the model adapter, repairs repeated replies, and cleans generated text.
"""
from dataclasses import dataclass
from difflib import SequenceMatcher
import re

from minillama.agents import STATION_PATTERN, build_agent_b_system, build_messages, clean_reply, fallback_reply


@dataclass
class DialogState:
    """Small state model passed between dialog controller stages for a single turn.
    """
    test_case: object
    conversation: list
    turn: int = 0

    @property
    def scenario(self):
        """Scenario method for this module's MVC responsibility.
        
        Returns:
            The computed value or side effect documented by the implementation.
        """
        return self.test_case.scenario

    @property
    def persona(self):
        """Persona method for this module's MVC responsibility.
        
        Returns:
            The computed value or side effect documented by the implementation.
        """
        return self.test_case.persona


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

    def run_agent_b(self, state: DialogState) -> str:
        """Run agent b method for this module's MVC responsibility.
        
        Args:
            state: Input value used by `run_agent_b`; see the function signature and caller context for the expected type.
        
        Returns:
            The computed value or side effect documented by the implementation.
        """
        messages = self.prompt_builder.run(state)
        raw_reply = self.generator.run(messages)
        reply = self.cleaner.run(raw_reply)
        if reply and not STATION_PATTERN.search(reply):
            reply = ""
        if reply and not self.cleaner.repeats_prior_agent_b(reply, state.conversation):
            return reply

        repair_messages = self.prompt_builder.run_repair(state, reply)
        raw_reply = self.generator.run(repair_messages)
        reply = self.cleaner.run(raw_reply)
        if reply and not STATION_PATTERN.search(reply):
            reply = ""
        if reply and not self.cleaner.repeats_prior_agent_b(reply, state.conversation):
            return reply

        phase_fallback = self.prompt_builder.phase_fallback(state)
        if not self.cleaner.repeats_prior_agent_b(phase_fallback, state.conversation):
            return phase_fallback

        return reply if reply else fallback_reply("Agent B", state.scenario, route_index=state.turn)


class AgentBPromptStage:
    """Prompt-stage controller for Agent B phase instructions and repair prompts.
    """
    def run(self, state: DialogState) -> str:
        """Run method for this module's MVC responsibility.
        
        Args:
            state: Input value used by `run`; see the function signature and caller context for the expected type.
        
        Returns:
            The computed value or side effect documented by the implementation.
        """
        system_prompt = (
            build_agent_b_system(state.scenario, state.persona)
            + " "
            + self._phase_instruction(state)
            + " Do not repeat a station sequence already proposed word for word; move the conversation forward with a distinct candidate, a comparison, a correction, or a clear confirmation."
        )
        return build_messages("Agent B", system_prompt, state.conversation)

    def run_repair(self, state: DialogState, repeated_reply: str) -> str:
        """Run repair method for this module's MVC responsibility.
        
        Args:
            state: Input value used by `run_repair`; see the function signature and caller context for the expected type.
            repeated_reply: Input value used by `run_repair`; see the function signature and caller context for the expected type.
        
        Returns:
            The computed value or side effect documented by the implementation.
        """
        system_prompt = (
            build_agent_b_system(state.scenario, state.persona)
            + " "
            + self._phase_instruction(state)
            + " The previous draft sounded repetitive. Continue the conversation with a fresh, useful route-building reply."
        )
        repair_history = list(state.conversation)
        if repeated_reply:
            repair_history.append(("Agent A", "That sounds like a repeat. Can you say it more clearly or update the route?"))
        return build_messages("Agent B", system_prompt, repair_history)

    def _phase_instruction(self, state: DialogState) -> str:
        """ phase instruction method for this module's MVC responsibility.
        
        Args:
            state: Input value used by `_phase_instruction`; see the function signature and caller context for the expected type.
        
        Returns:
            The computed value or side effect documented by the implementation.
        """
        scenario = state.scenario
        destination = scenario["destination_station"]

        if state.turn == 0:
            return (
                f"Start with one candidate route toward {destination}. "
                "Mention the stations in order and say what you would compare next."
            )

        if state.turn == 1:
            return (
                "If another plausible route exists, bring it up naturally. "
                "Compare connectivity, riding time, waiting time, and transfer time."
            )

        if state.turn == 2:
            return (
                "Check that the route really connects start to destination. "
                "If anything is missing, add the next connected station."
            )

        phase = (state.turn - 3) % 3
        if phase == 0:
            return (
                "Resolve any uncertainty by naming the next connected segment. "
                "If the route is already complete, mention a different shorter-looking alternative only if it seems worth checking."
            )
        if phase == 1:
            return (
                "Update the timing using riding, waiting, and transfer minutes. "
                "Refine the route only if that calculation finds a faster valid option."
            )
        return (
            "Confirm the best route station by station. "
            "State the total time as riding plus waiting plus transfer time."
        )

    def phase_fallback(self, state: DialogState) -> str:
        """Phase fallback method for this module's MVC responsibility.
        
        Args:
            state: Input value used by `phase_fallback`; see the function signature and caller context for the expected type.
        
        Returns:
            The computed value or side effect documented by the implementation.
        """
        # Planner-backed fallback that always includes station names so the dialog
        # can keep building a connected route even when the model degenerates.
        return fallback_reply("Agent B", state.scenario, route_index=state.turn)


class ModelGenerationStage:
    """Thin controller stage that delegates prompt text to the configured model adapter.
    """
    def __init__(self, model_adapter):
        """  init   method for this module's MVC responsibility.
        
        Args:
            model_adapter: Input value used by `__init__`; see the function signature and caller context for the expected type.
        
        Returns:
            The computed value or side effect documented by the implementation.
        """
        self.model_adapter = model_adapter

    def run(self, messages) -> str:
        """Run method for this module's MVC responsibility.
        
        Args:
            prompt: Input value used by `run`; see the function signature and caller context for the expected type.
        
        Returns:
            The computed value or side effect documented by the implementation.
        """
        if hasattr(self.model_adapter, "generate_messages"):
            return self.model_adapter.generate_messages(messages)
        return self.model_adapter.generate(build_prompt_from_messages(messages))


def build_prompt_from_messages(messages):
    """Fallback serializer for legacy deterministic adapters."""
    from minillama.model_adapters import messages_to_prompt

    return messages_to_prompt(messages)


class VerbalCleanupStage:
    """Cleanup stage for generated dialog text and repeated-response detection.
    """
    def run(self, raw_reply: str) -> str:
        """Run method for this module's MVC responsibility.
        
        Args:
            raw_reply: Input value used by `run`; see the function signature and caller context for the expected type.
        
        Returns:
            The computed value or side effect documented by the implementation.
        """
        return clean_reply(raw_reply)

    def repeats_prior_agent_b(self, reply: str, conversation) -> bool:
        """Repeats prior agent b method for this module's MVC responsibility.
        
        Args:
            reply: Input value used by `repeats_prior_agent_b`; see the function signature and caller context for the expected type.
            conversation: Input value used by `repeats_prior_agent_b`; see the function signature and caller context for the expected type.
        
        Returns:
            The computed value or side effect documented by the implementation.
        """
        normalized_reply = self._normalize(reply)
        if not normalized_reply:
            return False

        for speaker, text in conversation:
            if speaker != "Agent B":
                continue

            prior = self._normalize(text)
            if prior == normalized_reply:
                return True

            if SequenceMatcher(None, prior, normalized_reply).ratio() >= 0.92:
                return True

        return False

    def _normalize(self, text: str) -> str:
        """ normalize method for this module's MVC responsibility.
        
        Args:
            text: Input value used by `_normalize`; see the function signature and caller context for the expected type.
        
        Returns:
            The computed value or side effect documented by the implementation.
        """
        return re.sub(r"\s+", " ", text.strip().lower())
