"""Agent B plugin configuration and loading."""
from dataclasses import dataclass
from importlib import import_module
import os

from coop_navigation_sds.NaturalLanguageGeneration.caller.agents import fallback_reply
from coop_navigation_sds.NaturalLanguageGeneration.caller.agents import agent_a_requested_secondary_constraints
from coop_navigation_sds.NaturalLanguageGeneration.assistant.pipeline import VerbalTransformationPipeline
from coop_navigation_sds.NaturalLanguageUnderstanding.interpreter import NaturalRouteInterpreter
from coop_navigation_sds.TransportNetwork.constraints import stage_route_options, stated_constraint_keys
from coop_navigation_sds.TransportNetwork.routes import route_text_from_steps


DEFAULT_AGENT_B_PLUGIN = "llm"
MODEL_BACKED_AGENT_B_KEYS = {"minillama", "llm", ""}

@dataclass(frozen=True)
class AgentBModelSpec:
    """Public metadata for a selectable Agent B implementation."""

    key: str
    label: str
    description: str
    style: str
    model_backed: bool = False


AGENT_B_MODEL_SPECS = {
    "llm": AgentBModelSpec(
        "llm",
        "Configurable language model",
        "Model-backed spoken route assistant using Transformers, OpenAI-compatible, or Ollama.",
        "adaptive",
        model_backed=True,
    ),
    "simple": AgentBModelSpec(
        "simple",
        "Simple deterministic assistant",
        "Fast deterministic baseline for smoke tests and reproducible comparisons.",
        "direct",
    ),
    "pareto": AgentBModelSpec(
        "pareto",
        "Balanced spoken advisor",
        "Balances journey time, transfers, capacity, and reliability on the Pareto frontier.",
        "balanced",
    ),
    "robust": AgentBModelSpec(
        "robust",
        "Reliability-first spoken guide",
        "Prioritizes reliability and low-risk routes, explaining risk when the caller asks.",
        "reassuring",
    ),
    "diverse": AgentBModelSpec(
        "diverse",
        "Alternative route explorer",
        "Actively proposes a different viable route instead of repeating prior suggestions.",
        "exploratory",
    ),
}


def agent_b_plugin_description(plugin_key):
    """Return concise UI help for a built-in or custom Agent B implementation."""
    config = AgentBPluginConfig(plugin_key or DEFAULT_AGENT_B_PLUGIN)
    spec = AGENT_B_MODEL_SPECS.get(config.normalized_key)
    return spec.description if spec else "Custom Agent B plugin loaded from a Python module."


@dataclass(frozen=True)
class AgentBPluginConfig:
    """Runtime selection for Agent B."""
    key: str = DEFAULT_AGENT_B_PLUGIN

    @classmethod
    def from_env(cls):
        value = os.environ.get(
            "COOP_NAVIGATION_SDS_AGENT_B_PLUGIN",
            os.environ.get("MINILLAMA_AGENT_B_PLUGIN", DEFAULT_AGENT_B_PLUGIN),
        )
        return cls(value.strip() or DEFAULT_AGENT_B_PLUGIN)

    @property
    def normalized_key(self):
        return DEFAULT_AGENT_B_PLUGIN if self.key in MODEL_BACKED_AGENT_B_KEYS else self.key

    @property
    def needs_model(self):
        return self.normalized_key == DEFAULT_AGENT_B_PLUGIN

    @property
    def label(self):
        spec = AGENT_B_MODEL_SPECS.get(self.normalized_key)
        return spec.label if spec else self.key


class LlmAgentBPlugin:
    """Default CoopNavigationSDS Agent B backed by a configured model adapter."""
    name = "llm-agent-b"

    def __init__(self, model_adapter):
        if model_adapter is None:
            raise ValueError("The llm Agent B plugin requires a model adapter.")
        self.model_adapter = model_adapter
        self.pipeline = VerbalTransformationPipeline(model_adapter)

    def run_agent_b(self, state):
        return self.pipeline.run_agent_b(state)

    def consume_prompt_audits(self):
        """Expose prompt evidence without coupling orchestration to pipeline internals."""
        return self.pipeline.consume_prompt_audits()


class SimplePlannerAgentBPlugin:
    """Small deterministic Agent B plugin for demos and tests."""
    name = "simple-planner-agent-b"

    def run_agent_b(self, state):
        if state.assistant_scenario is None:
            return state.trip_clarification_prompt()
        return fallback_reply(
            "Agent B",
            state.assistant_scenario,
            route_index=state.turn,
            persona={},
            conversation=state.conversation,
        )


class ResearchPlannerAgentBPlugin:
    """Deterministic research policy with an explicit route-selection strategy."""

    strategy = "pareto"
    name = "research-planner-agent-b"
    speaking_style = "balanced"

    def run_agent_b(self, state):
        scenario = state.assistant_scenario
        if scenario is None:
            return state.trip_clarification_prompt()
        stated_keys = stated_constraint_keys(state.conversation)
        options = stage_route_options(
            scenario,
            state.persona,
            stated_keys=stated_keys,
            limit=120,
        )
        if not options:
            return fallback_reply(
                "Agent B",
                scenario,
                route_index=state.turn,
                persona=state.persona,
                conversation=state.conversation,
            )
        prior_routes = self._prior_routes(state.conversation, scenario)
        fresh = [option for option in options if tuple(option["route"]) not in prior_routes]
        if not fresh and prior_routes:
            selected = self.select_option(options, prior_routes)
            return self._format_existing_option(selected, state.context)
        selected = self.select_option(fresh or options, prior_routes)
        return self._format_option(selected, state.context)

    def select_option(self, options, prior_routes):
        if self.strategy == "robust":
            return min(options, key=self._robust_key)
        if self.strategy == "diverse":
            return min(options, key=lambda option: self._diverse_key(option, prior_routes))
        frontier = [
            option
            for option in options
            if not any(self._dominates(other, option) for other in options if other is not option)
        ]
        return min(frontier or options, key=self._pareto_tiebreak)

    @staticmethod
    def _risk_value(value):
        return {"low": 0, "medium": 1, "high": 2}.get(str(value), 3)

    @classmethod
    def _vector(cls, option):
        return (
            option["duration_min"],
            option["line_change_count"],
            option["near_capacity_count"],
            cls._risk_value(option["delay_risk_class"]),
            cls._risk_value(option["transfer_miss_risk_class"]),
        )

    @classmethod
    def _dominates(cls, left, right):
        left_values = cls._vector(left)
        right_values = cls._vector(right)
        return all(a <= b for a, b in zip(left_values, right_values)) and any(
            a < b for a, b in zip(left_values, right_values)
        )

    @classmethod
    def _pareto_tiebreak(cls, option):
        vector = cls._vector(option)
        return (sum(vector[1:]), vector[0], len(option["route"]), tuple(option["route"]))

    @classmethod
    def _robust_key(cls, option):
        delay = cls._risk_value(option["delay_risk_class"])
        transfer = cls._risk_value(option["transfer_miss_risk_class"])
        return (
            max(delay, transfer),
            delay + transfer,
            option["near_capacity_count"],
            option["line_change_count"],
            option["duration_min"],
            tuple(option["route"]),
        )

    @staticmethod
    def _route_edges(route):
        return set(zip(route, route[1:]))

    @classmethod
    def _diverse_key(cls, option, prior_routes):
        if not prior_routes:
            return (option["duration_min"], tuple(option["route"]))
        edges = cls._route_edges(option["route"])
        similarities = []
        for prior in prior_routes:
            prior_edges = cls._route_edges(prior)
            union = edges | prior_edges
            similarities.append(len(edges & prior_edges) / len(union) if union else 1.0)
        return (
            max(similarities),
            sum(similarities),
            option["duration_min"],
            tuple(option["route"]),
        )

    @staticmethod
    def _prior_routes(conversation, scenario):
        interpreter = NaturalRouteInterpreter()
        return {
            tuple(route)
            for speaker, text in conversation
            if speaker == "Agent B"
            for route in [interpreter.interpret_reply(text, scenario)]
            if route
        }

    def _format_option(self, option, context):
        reply = route_text_from_steps(option["steps"])
        prefix = self._style_prefix(context)
        if not agent_a_requested_secondary_constraints(context.conversation):
            return f"{prefix} {reply}"
        return f"{prefix} {reply} {self._requested_detail(option, context)}".strip()

    def _format_existing_option(self, option, context):
        line = option["line_sequence"][0] if option["line_sequence"] else "route"
        if not agent_a_requested_secondary_constraints(context.conversation):
            return (
                f"The earlier {line} option is still the best valid choice "
                f"at {option['duration_min']} minutes."
            )
        detail = self._requested_detail(option, context)
        return f"The earlier {line} option still fits. {detail}".strip()

    @staticmethod
    def _requested_detail(option, context):
        latest = context.latest_agent_a.lower()
        if any(term in latest for term in ("crowd", "capacity", "packed", "full")):
            return "It is near capacity." if option["near_capacity_count"] else "It is not near capacity."
        if any(term in latest for term in ("transfer miss", "safer transfer")):
            return f"Transfer miss risk is {option['transfer_miss_risk_class']}."
        if any(term in latest for term in ("delay", "reliable", "risk")):
            return f"Delay risk is {option['delay_risk_class']}."
        return ""

    def _style_prefix(self, context, existing=False):
        if self.speaking_style == "reassuring":
            if context.response_focus == "reliability":
                return "For reliability,"
            return "Best reliable choice:"
        if self.speaking_style == "exploratory":
            if context.agent_b_turn_count == 0:
                return "Option:"
            return (
                "Earlier option:"
                if existing
                else "Different option:"
            )
        if context.response_focus == "time":
            return "Balanced fast option:"
        return "Balanced option:"


class ParetoPlannerAgentBPlugin(ResearchPlannerAgentBPlugin):
    strategy = "pareto"
    name = "pareto-planner-agent-b"


class RobustPlannerAgentBPlugin(ResearchPlannerAgentBPlugin):
    strategy = "robust"
    name = "robust-planner-agent-b"
    speaking_style = "reassuring"


class DiversePlannerAgentBPlugin(ResearchPlannerAgentBPlugin):
    strategy = "diverse"
    name = "diverse-planner-agent-b"
    speaking_style = "exploratory"


class FunctionAgentBPlugin:
    """Adapter for function plugins."""
    name = "function-agent-b"

    def __init__(self, fn):
        self.fn = fn

    def run_agent_b(self, state):
        return self.fn(state)


def available_agent_b_plugin_keys(extra_key=None):
    keys = list(AGENT_B_MODEL_SPECS)
    if extra_key and extra_key not in keys:
        keys.append(extra_key)
    return keys


def create_agent_b_plugin(plugin_key, model_adapter):
    """Create an Agent B plugin from a built-in key or `module:factory` path."""
    config = AgentBPluginConfig(plugin_key or DEFAULT_AGENT_B_PLUGIN)
    if config.normalized_key == DEFAULT_AGENT_B_PLUGIN:
        return LlmAgentBPlugin(model_adapter)
    if config.normalized_key == "simple":
        return SimplePlannerAgentBPlugin()
    if config.normalized_key == "pareto":
        return ParetoPlannerAgentBPlugin()
    if config.normalized_key == "robust":
        return RobustPlannerAgentBPlugin()
    if config.normalized_key == "diverse":
        return DiversePlannerAgentBPlugin()
    return load_custom_agent_b_plugin(config.normalized_key, model_adapter)


def load_custom_agent_b_plugin(plugin_path, model_adapter):
    """Load `module:factory_or_class` and return a run_agent_b-compatible object."""
    if ":" not in plugin_path:
        raise ValueError("Custom Agent B plugin must use 'package.module:factory_or_class'.")

    module_name, attr_name = plugin_path.split(":", 1)
    plugin_obj = getattr(import_module(module_name), attr_name)

    try:
        plugin = plugin_obj(model_adapter)
    except TypeError:
        plugin = plugin_obj()

    if callable(plugin) and not hasattr(plugin, "run_agent_b"):
        return FunctionAgentBPlugin(plugin)
    if not hasattr(plugin, "run_agent_b"):
        raise TypeError("Agent B plugin must provide run_agent_b(state) or be callable.")
    return plugin
