"""Interactive GUI entry point and startup controller."""
import logging
import queue
import threading

from huggingface_hub.utils import logging as hf_logging

from minillama.config import (
    DEVICE,
    LLM_AGENT_A,
    MAX_INPUT_TOKENS,
    MAX_NEW_TOKENS,
    AGENT_B_PLUGIN,
    MODEL,
    MODEL_PROVIDER,
    NUM_TURNS,
)
from minillama.agent_a_responder import LLMAgentAResponder, TemplateAgentAResponder
from minillama.agent_b_plugins import create_agent_b_plugin
from minillama.dialog_manager import DialogManager
from minillama.test_cases import DEFAULT_TEST_CASE, get_test_case
from minillama.model_runtime import create_model_adapter
from minillama.gui import DialogWindow


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
hf_logging.set_verbosity_warning()


def build_agent_a_responder(model_adapter):
    """Create the configured Agent A responder implementation."""
    if LLM_AGENT_A:
        return LLMAgentAResponder(model_adapter)
    return TemplateAgentAResponder()


def conversation_worker(event_queue, model_adapter, test_case_key):
    """Run one dialog in a background worker and stream UI events."""
    test_case = get_test_case(test_case_key)
    agent_b_plugin = create_agent_b_plugin(AGENT_B_PLUGIN, model_adapter)
    manager = DialogManager(
        test_case,
        agent_b_plugin,
        NUM_TURNS,
        agent_a_responder=build_agent_a_responder(model_adapter),
    )

    event_queue.put(("system", f"Model: {getattr(model_adapter, 'name', MODEL)}"))
    event_queue.put(("system", f"Provider: {MODEL_PROVIDER}"))
    event_queue.put(("system", f"Device: {getattr(model_adapter, 'device', DEVICE)}"))
    event_queue.put(("system", f"Turns={NUM_TURNS}, max_new_tokens={MAX_NEW_TOKENS}, max_length={MAX_INPUT_TOKENS}"))
    event_queue.put(("system", f"Agent A: {manager.agent_a_responder.name}"))
    event_queue.put(("system", f"Agent B: {getattr(agent_b_plugin, 'name', type(agent_b_plugin).__name__)}"))
    try:
        manager.run(event_queue)
    except Exception as exc:
        logging.exception("Conversation worker failed")
        event_queue.put(("warning", f"Conversation stopped: {exc}"))
        event_queue.put(("done",))


def main():
    """Start one interactive dialog run."""
    selected_test_case = DEFAULT_TEST_CASE
    scenario = get_test_case(selected_test_case).scenario
    model_adapter = create_model_adapter()
    event_queue = queue.Queue()

    worker = threading.Thread(
        target=conversation_worker,
        args=(event_queue, model_adapter, selected_test_case),
        daemon=True,
    )
    worker.start()

    dialog = DialogWindow(event_queue, scenario)
    dialog.run()


if __name__ == "__main__":
    main()
