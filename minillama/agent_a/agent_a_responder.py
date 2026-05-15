"""Agent A controller strategies. Agent A can be template-driven for speed or LLM-driven for the final UserLM setup.
"""
from minillama.agent_a.agents import (
    build_agent_a_system,
    build_messages,
    build_prompt,
    clean_reply,
    fallback_reply,
    generate_agent_a_template,
)


class TemplateAgentAResponder:
    """Fast deterministic Agent A behavior for repeatable experiments."""

    name = "template-agent-a"

    def reply(self, turn: int, persona: dict, scenario: dict, conversation: list) -> str:
        """Reply method for this module's MVC responsibility.
        
        Args:
            turn: Input value used by `reply`; see the function signature and caller context for the expected type.
            persona: Input value used by `reply`; see the function signature and caller context for the expected type.
            scenario: Input value used by `reply`; see the function signature and caller context for the expected type.
            conversation: Input value used by `reply`; see the function signature and caller context for the expected type.
        
        Returns:
            The computed value or side effect documented by the implementation.
        """
        return generate_agent_a_template(turn, persona, scenario)


class LLMAgentAResponder:
    """Agent A behavior backed by the same ModelAdapter interface as Agent B."""

    name = "llm-agent-a"

    def __init__(self, model_adapter):
        """  init   method for this module's MVC responsibility.
        
        Args:
            model_adapter: Input value used by `__init__`; see the function signature and caller context for the expected type.
        
        Returns:
            The computed value or side effect documented by the implementation.
        """
        self.model_adapter = model_adapter

    def reply(self, turn: int, persona: dict, scenario: dict, conversation: list) -> str:
        """Reply method for this module's MVC responsibility.
        
        Args:
            turn: Input value used by `reply`; see the function signature and caller context for the expected type.
            persona: Input value used by `reply`; see the function signature and caller context for the expected type.
            scenario: Input value used by `reply`; see the function signature and caller context for the expected type.
            conversation: Input value used by `reply`; see the function signature and caller context for the expected type.
        
        Returns:
            The computed value or side effect documented by the implementation.
        """
        system_prompt = build_agent_a_system(persona, scenario)
        messages = build_messages("Agent A", system_prompt, conversation)
        if hasattr(self.model_adapter, "generate_messages"):
            raw_reply = self.model_adapter.generate_messages(messages)
        else:
            raw_reply = self.model_adapter.generate(build_prompt("Agent A", system_prompt, conversation))
        reply = clean_reply(raw_reply)
        return reply if reply else fallback_reply("Agent A", scenario, route_index=turn)
