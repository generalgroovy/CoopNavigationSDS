"""Agent B plugin configuration and loading."""
from dataclasses import dataclass
from importlib import import_module
import os

from minillama.agent_a.agents import fallback_reply
from minillama.agent_b.pipeline import VerbalTransformationPipeline


DEFAULT_AGENT_B_PLUGIN = "minillama"
MODEL_BACKED_AGENT_B_KEYS = {"minillama", "llm", ""}

BUILTIN_AGENT_B_PLUGINS = {
    "minillama": "MiniLlama route assistant",
    "simple": "Deterministic route planner",
    "llm": "LLM route assistant alias",
}


@dataclass(frozen=True)
class AgentBPluginConfig:
    """Runtime selection for Agent B."""
    key: str = DEFAULT_AGENT_B_PLUGIN

    @classmethod
    def from_env(cls):
        return cls(os.environ.get("MINILLAMA_AGENT_B_PLUGIN", DEFAULT_AGENT_B_PLUGIN).strip() or DEFAULT_AGENT_B_PLUGIN)

    @property
    def normalized_key(self):
        return DEFAULT_AGENT_B_PLUGIN if self.key in MODEL_BACKED_AGENT_B_KEYS else self.key

    @property
    def needs_model(self):
        return self.normalized_key == DEFAULT_AGENT_B_PLUGIN

    @property
    def label(self):
        return BUILTIN_AGENT_B_PLUGINS.get(self.key, BUILTIN_AGENT_B_PLUGINS.get(self.normalized_key, self.key))


class LlmAgentBPlugin:
    """Default MiniLlama Agent B backed by any configured model adapter."""
    name = "minillama-agent-b"

    def __init__(self, model_adapter):
        if model_adapter is None:
            raise ValueError("The llm Agent B plugin requires a model adapter.")
        self.model_adapter = model_adapter
        self.pipeline = VerbalTransformationPipeline(model_adapter)

    def run_agent_b(self, state):
        return self.pipeline.run_agent_b(state)


class SimplePlannerAgentBPlugin:
    """Small deterministic Agent B plugin for demos and tests."""
    name = "simple-planner-agent-b"

    def run_agent_b(self, state):
        return fallback_reply("Agent B", state.scenario, route_index=state.turn)


class FunctionAgentBPlugin:
    """Adapter for function plugins."""
    name = "function-agent-b"

    def __init__(self, fn):
        self.fn = fn

    def run_agent_b(self, state):
        return self.fn(state)


def available_agent_b_plugin_keys(extra_key=None):
    keys = ["minillama", "simple"]
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
