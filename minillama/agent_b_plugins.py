"""Agent B plugin loading and built-in plugin implementations."""
from importlib import import_module

from minillama.agents import fallback_reply
from minillama.pipeline import VerbalTransformationPipeline


class LlmAgentBPlugin:
    """Default Agent B plugin backed by the configured model adapter."""

    name = "llm-agent-b"

    def __init__(self, model_adapter):
        self.model_adapter = model_adapter
        self.pipeline = VerbalTransformationPipeline(model_adapter)

    def run_agent_b(self, state):
        return self.pipeline.run_agent_b(state)


class SimplePlannerAgentBPlugin:
    """Small deterministic Agent B plugin for demos and plugin testing."""

    name = "simple-planner-agent-b"

    def run_agent_b(self, state):
        return fallback_reply("Agent B", state.scenario, route_index=state.turn)


def create_agent_b_plugin(plugin_key, model_adapter):
    """Create an Agent B plugin from a built-in name or dotted import path."""
    if plugin_key in (None, "", "llm"):
        return LlmAgentBPlugin(model_adapter)
    if plugin_key == "simple":
        return SimplePlannerAgentBPlugin()
    return load_custom_agent_b_plugin(plugin_key, model_adapter)


def load_custom_agent_b_plugin(plugin_path, model_adapter):
    """Load `module:factory_or_class` and return a run_agent_b-compatible object."""
    if ":" not in plugin_path:
        raise ValueError(
            "Custom Agent B plugin must use 'package.module:factory_or_class'."
        )

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


class FunctionAgentBPlugin:
    """Adapter for simple function plugins."""

    name = "function-agent-b"

    def __init__(self, fn):
        self.fn = fn

    def run_agent_b(self, state):
        return self.fn(state)
