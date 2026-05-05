from dataclasses import dataclass
from difflib import SequenceMatcher
import re

from agents import build_agent_b_system, build_prompt, clean_reply, fallback_reply


@dataclass
class DialogState:
    test_case: object
    conversation: list
    turn: int = 0

    @property
    def scenario(self):
        return self.test_case.scenario

    @property
    def persona(self):
        return self.test_case.persona


class VerbalTransformationPipeline:
    """Transforms dialog state into a verbal Agent B response."""

    def __init__(self, model_adapter):
        self.model_adapter = model_adapter
        self.prompt_builder = AgentBPromptStage()
        self.generator = ModelGenerationStage(model_adapter)
        self.cleaner = VerbalCleanupStage()

    def run_agent_b(self, state: DialogState) -> str:
        prompt = self.prompt_builder.run(state)
        raw_reply = self.generator.run(prompt)
        reply = self.cleaner.run(raw_reply)
        if reply and not self.cleaner.repeats_prior_agent_b(reply, state.conversation):
            return reply

        repair_prompt = self.prompt_builder.run_repair(state, reply)
        raw_reply = self.generator.run(repair_prompt)
        reply = self.cleaner.run(raw_reply)
        if reply and not self.cleaner.repeats_prior_agent_b(reply, state.conversation):
            return reply

        phase_fallback = self.prompt_builder.phase_fallback(state)
        if not self.cleaner.repeats_prior_agent_b(phase_fallback, state.conversation):
            return phase_fallback

        return reply if reply else fallback_reply("Agent B", state.scenario)


class AgentBPromptStage:
    def run(self, state: DialogState) -> str:
        system_prompt = (
            build_agent_b_system(state.scenario)
            + " "
            + self._phase_instruction(state)
            + " Do not repeat an earlier answer; each reply must advance the spoken route discussion."
        )
        return build_prompt("Agent B", system_prompt, state.conversation)

    def run_repair(self, state: DialogState, repeated_reply: str) -> str:
        system_prompt = (
            build_agent_b_system(state.scenario)
            + " "
            + self._phase_instruction(state)
            + " Your previous draft repeated earlier wording. Give a different natural spoken reply that advances the route discussion."
        )
        repair_history = list(state.conversation)
        if repeated_reply:
            repair_history.append(("Agent A", "That repeats what you already said. Please continue with the next useful detail."))
        return build_prompt("Agent B", system_prompt, repair_history)

    def _phase_instruction(self, state: DialogState) -> str:
        scenario = state.scenario
        destination = scenario["destination_station"]

        if state.turn == 0:
            return (
                f"For this reply, open the route discussion by proposing connected station segments toward {destination}. "
                "Name concrete stations and point out what still needs to be checked."
            )

        if state.turn == 1:
            return (
                "For this reply, build on the previous route by adding riding time, waiting time, and transfer cost. "
                "Explain the partial duration calculation in natural speech."
            )

        if state.turn == 2:
            return (
                "For this reply, check whether the spoken route is connected from the start toward the destination. "
                "If a segment is missing, add the next station needed to complete it."
            )

        return (
            "For this reply, confirm the complete route conversationally, station by station, "
            "and state the total duration as travel plus waiting plus transfer time."
        )

    def phase_fallback(self, state: DialogState) -> str:
        scenario = state.scenario
        start = scenario["start_station"]
        destination = scenario["destination_station"]
        transfer = scenario["transfer_time_min"]

        if state.turn == 0:
            return (
                f"I would compare a simple route out of {start} with a route that changes lines toward {destination}. "
                "For each option, we need to verify the station links and then add riding, waiting, and transfer time."
            )

        if state.turn == 1:
            return (
                f"At this point I would include the {transfer} minute change penalty before choosing. "
                "The duration should be the riding minutes plus waiting minutes plus any transfer minutes."
            )

        if state.turn == 2:
            return (
                f"The route still needs to be checked station by station until it reaches {destination}. "
                "Once every segment connects, we can calculate the full duration."
            )

        return (
            f"I would now confirm the selected path from {start} toward {destination} in station order. "
            "The important checks are that it reaches the destination and that the duration is calculated from riding, waiting, and transfer time."
        )


class ModelGenerationStage:
    def __init__(self, model_adapter):
        self.model_adapter = model_adapter

    def run(self, prompt: str) -> str:
        return self.model_adapter.generate(prompt)


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

            if SequenceMatcher(None, prior, normalized_reply).ratio() >= 0.92:
                return True

        return False

    def _normalize(self, text: str) -> str:
        return re.sub(r"\s+", " ", text.strip().lower())
