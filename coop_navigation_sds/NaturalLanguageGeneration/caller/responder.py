"""Agent A controller strategies. Agent A can be template-driven for speed or LLM-driven for the final UserLM setup.
"""
from coop_navigation_sds.NaturalLanguageGeneration.caller.agents import (
    build_agent_a_system,
    build_messages,
    build_prompt,
    clean_reply,
    fallback_reply,
    generate_agent_a_template,
)
from coop_navigation_sds.TransportNetwork.constraints import stated_constraint_keys
from coop_navigation_sds.NaturalLanguageUnderstanding.clarification import (
    last_substantive_agent_b_utterance,
)
from coop_navigation_sds.DialogManagement.stages import agent_memory_view
from coop_navigation_sds.NaturalLanguageGeneration.prompt_audit import (
    begin_prompt_audit,
    consume_prompt_audits as drain_prompt_audits,
    finish_prompt_audit,
    record_prompt_delivery,
)


AGENT_A_MINILLAMA = "staged"
AGENT_A_TINYLLAMA = "tinyllama"
AGENT_A_USERLM = "userlm"
MODEL_BACKED_AGENT_A_TYPES = (AGENT_A_TINYLLAMA, AGENT_A_USERLM)
AGENT_A_TYPES = (AGENT_A_MINILLAMA, AGENT_A_TINYLLAMA, AGENT_A_USERLM)


def normalize_agent_a_type(value=None, legacy_llm_agent_a=False):
    """Normalize the caller implementation while preserving the legacy boolean."""
    if isinstance(value, bool):
        return AGENT_A_USERLM if value else AGENT_A_MINILLAMA
    if value is None or str(value).strip() == "":
        return AGENT_A_USERLM if legacy_llm_agent_a else AGENT_A_MINILLAMA
    normalized = str(value).strip().lower().replace("-", "").replace("_", "")
    aliases = {
        "minillama": AGENT_A_MINILLAMA,
        "tinyllama": AGENT_A_TINYLLAMA,
        "staged": AGENT_A_MINILLAMA,
        "template": AGENT_A_MINILLAMA,
        "deterministic": AGENT_A_MINILLAMA,
        "userlm": AGENT_A_USERLM,
        "llm": AGENT_A_USERLM,
        "model": AGENT_A_USERLM,
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported Agent A implementation '{value}'. Use one of {AGENT_A_TYPES}.")
    return aliases[normalized]


def available_agent_a_types():
    return AGENT_A_TYPES


def agent_a_uses_model(agent_a_type):
    """Return whether the Agent A implementation requires a language model."""
    return normalize_agent_a_type(agent_a_type) in MODEL_BACKED_AGENT_A_TYPES


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
        return generate_agent_a_template(turn, persona, scenario, conversation)


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
        self.prompt_audits = []

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
        template_reply = generate_agent_a_template(
            turn,
            persona,
            scenario,
            conversation,
        )
        latest_agent_b = last_substantive_agent_b_utterance(conversation)
        if latest_agent_b:
            from coop_navigation_sds.NaturalLanguageUnderstanding.interpreter import NaturalRouteInterpreter

            heard_route = NaturalRouteInterpreter().interpret_reply(
                latest_agent_b,
                scenario,
            )
            if (
                not heard_route
                or heard_route[0] != scenario["start_station"]
                or heard_route[-1] != scenario["destination_station"]
            ):
                return template_reply

        memory = agent_memory_view(
            "Agent A",
            conversation,
            scenario=scenario,
            persona=persona,
        )
        system_prompt = f"{build_agent_a_system(persona, scenario)} {memory.prompt_summary()}"
        messages = build_messages("Agent A", system_prompt, conversation)
        audit = begin_prompt_audit(
            agent="Agent A",
            purpose="response",
            stage=memory.pending_focus,
            turn=len(conversation) + 1,
            messages=messages,
        )
        self.prompt_audits.append(audit)
        try:
            if hasattr(self.model_adapter, "generate_messages"):
                raw_reply = self.model_adapter.generate_messages(messages)
            else:
                raw_reply = self.model_adapter.generate(
                    build_prompt("Agent A", system_prompt, conversation)
                )
        except Exception as exc:
            finish_prompt_audit(
                audit,
                accepted=False,
                decision="generation_error",
                error=repr(exc),
            )
            raise
        reply = clean_reply(raw_reply)
        if not reply:
            finish_prompt_audit(
                audit,
                raw_output=raw_reply,
                cleaned_output=reply,
                accepted=False,
                decision="template_fallback_invalid_output",
            )
            record_prompt_delivery(audit, template_reply, "deterministic_policy_fallback")
            return template_reply

        current_keys = set(stated_constraint_keys(conversation))
        permitted_keys = set(stated_constraint_keys([*conversation, ("Agent A", template_reply)]))
        generated_keys = set(stated_constraint_keys([*conversation, ("Agent A", reply)]))
        generated_new_keys = generated_keys - current_keys
        permitted_new_keys = permitted_keys - current_keys
        if len(generated_new_keys) > 1 or not generated_new_keys <= permitted_new_keys:
            finish_prompt_audit(
                audit,
                raw_output=raw_reply,
                cleaned_output=reply,
                accepted=False,
                decision="template_fallback_constraint_policy",
            )
            record_prompt_delivery(audit, template_reply, "deterministic_policy_fallback")
            return template_reply

        template_closes = _closes_call(template_reply)
        if _closes_call(reply) and not template_closes:
            finish_prompt_audit(
                audit,
                raw_output=raw_reply,
                cleaned_output=reply,
                accepted=False,
                decision="template_fallback_premature_closure",
            )
            record_prompt_delivery(audit, template_reply, "deterministic_policy_fallback")
            return template_reply
        finish_prompt_audit(
            audit,
            raw_output=raw_reply,
            cleaned_output=reply,
            accepted=True,
            decision="accepted",
        )
        record_prompt_delivery(audit, reply, "model")
        return reply

    def consume_prompt_audits(self):
        """Return exact prompt calls made since the previous controller read."""
        return drain_prompt_audits(self)


def _closes_call(text):
    lower = str(text or "").lower()
    return "thanks" in lower and any(
        term in lower
        for term in ("i'll take", "i will take", "choose", "that works")
    )
