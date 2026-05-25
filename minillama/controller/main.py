"""Interactive GUI entry point and startup controller."""
import logging
import queue
import threading

try:
    from huggingface_hub.utils import logging as hf_logging
except ModuleNotFoundError:
    hf_logging = None

from minillama.agent_a.agent_a_responder import LLMAgentAResponder, TemplateAgentAResponder
from minillama.agent_a.config import DEFAULT_PERSONA, LLM_AGENT_A, PERSONAS
from minillama.agent_b.config import (
    AGENT_B_PLUGIN,
    DEFAULT_SPEECH_PATTERN,
    SPEECH_ASR_ENGINE,
    SPEECH_AUDIO_DIR,
    SPEECH_ENGINE,
    SPEECH_INCOMING_ENABLED,
    SPEECH_OUTGOING_ENABLED,
    SPEECH_PLAYBACK_ENABLED,
    SPEECH_REALTIME_ENABLED,
    SPEECH_SCOPE,
    SPEECH_TTS_ENGINE,
)
from minillama.agent_b.plugin_registry import AgentBPluginConfig, available_agent_b_plugin_keys, create_agent_b_plugin
from minillama.agent_b.speech_io import SpeechPipelineConfig, SpeechTransport
from minillama.controller.config import (
    CONSTRAINT_MISS_LIMIT,
    GUI_ENABLED,
    GUI_MODE,
    INVALID_ROUTE_LIMIT,
    METRIC_SNAPSHOT_INTERVAL,
    NETWORK_PICTURE_DIR,
    NUM_TURNS,
    PROTOCOL_LOG_DIR,
    RESEARCH_LOG_DIR,
    SESSION_LOG_DIR,
    SESSION_LOG_PROFILE,
    SESSION_NAME,
)
from minillama.controller.dialog_manager import DialogManager
from minillama.controller.session_logging import MonitoringEventQueue, SessionLogger
from minillama.evaluation.research_artifacts import write_single_run_research_outputs, write_network_research_artifacts
from minillama.model.config import MAX_INPUT_TOKENS, MAX_NEW_TOKENS, MODEL, MODEL_PROVIDER
from minillama.test_cases.config import DEFAULT_TEST_CASE
from minillama.test_cases.test_cases import TEST_CASES, get_test_case


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
if hf_logging is not None:
    hf_logging.set_verbosity_warning()


def build_agent_a_responder(model_adapter, llm_agent_a=LLM_AGENT_A):
    """Create the configured Agent A responder implementation."""
    if llm_agent_a and model_adapter is not None:
        return LLMAgentAResponder(model_adapter)
    return TemplateAgentAResponder()


def conversation_worker(event_queue, model_adapter, run_config):
    """Run one dialog and stream optional UI events."""
    test_case = get_test_case(run_config["test_case_key"]).with_persona(run_config.get("persona_key", DEFAULT_PERSONA))
    agent_b_plugin = create_agent_b_plugin(run_config["agent_b_plugin"], model_adapter)
    speech_transport = SpeechTransport(
        config=SpeechPipelineConfig(
            incoming_enabled=run_config["speech_incoming_enabled"],
            outgoing_enabled=run_config["speech_outgoing_enabled"],
            scope=run_config["speech_scope"],
            pattern_key=run_config["speech_pattern_key"],
            engine=run_config["speech_engine"],
            tts_engine=run_config["tts_engine"],
            asr_engine=run_config["asr_engine"],
            audio_dir=run_config["speech_audio_dir"],
            playback_enabled=run_config["speech_playback_enabled"],
            realtime_enabled=run_config["speech_realtime_enabled"],
            agent_a_words_per_minute=int(run_config["agent_a_words_per_minute"]),
            agent_b_words_per_minute=int(run_config["agent_b_words_per_minute"]),
            min_utterance_sec=float(run_config["min_utterance_sec"]),
            max_utterance_sec=float(run_config["max_utterance_sec"]),
        )
    )
    num_turns = int(run_config["num_turns"])
    manager = DialogManager(
        test_case,
        agent_b_plugin,
        num_turns,
        speech_transport=speech_transport,
        agent_a_responder=build_agent_a_responder(model_adapter, bool(run_config.get("llm_agent_a", LLM_AGENT_A))),
        monitor=event_queue,
        invalid_route_limit=int(run_config["invalid_route_limit"]),
        constraint_miss_limit=int(run_config["constraint_miss_limit"]),
        metric_snapshot_interval=int(run_config["metric_snapshot_interval"]),
    )
    model_name = getattr(model_adapter, "name", "no-model")
    model_provider = MODEL_PROVIDER if model_adapter is not None else "none"
    device = getattr(model_adapter, "device", "not available")

    try:
        with event_queue.segment(
            "dialog.run",
            model=model_name,
            provider=model_provider,
            device=device,
            turns=num_turns,
            max_new_tokens=MAX_NEW_TOKENS,
            max_input_tokens=MAX_INPUT_TOKENS,
            agent_a=manager.agent_a_responder.name,
            agent_b=getattr(agent_b_plugin, "name", type(agent_b_plugin).__name__),
            speech_pipeline=speech_transport.description,
        ):
            event_queue.put(("system", f"Model: {model_name}"))
            event_queue.put(("system", f"Provider: {model_provider}"))
            event_queue.put(("system", f"Device: {device}"))
            event_queue.put(("system", f"Turns={num_turns}, max_new_tokens={MAX_NEW_TOKENS}, max_length={MAX_INPUT_TOKENS}"))
            event_queue.put(("system", f"Agent A: {manager.agent_a_responder.name}"))
            event_queue.put(("system", f"Agent B: {getattr(agent_b_plugin, 'name', type(agent_b_plugin).__name__)}"))
            result = manager.run(event_queue)
            protocol_dir = run_config.get("protocol_log_dir")
            if protocol_dir:
                research_paths = write_single_run_research_outputs(result, test_case.scenario, protocol_dir)
                event_queue.put(("system", f"Conversation protocol: {research_paths['protocol']['summary']}"))
                event_queue.put(("system", f"Compiled metrics: {research_paths['metrics_file']}"))
                event_queue.put(("system", f"Metric phase logs: {research_paths['phase_log_dir']}"))
    except Exception as exc:
        logging.exception("Conversation worker failed")
        event_queue.put(("warning", f"Conversation stopped: {exc}"))
        event_queue.put(("done",))
    finally:
        event_queue.close()


def default_run_config():
    """Return the default interactive run configuration."""
    return {
        "test_case_key": DEFAULT_TEST_CASE,
        "persona_key": DEFAULT_PERSONA,
        "agent_b_plugin": AGENT_B_PLUGIN,
        "num_turns": NUM_TURNS,
        "invalid_route_limit": INVALID_ROUTE_LIMIT,
        "constraint_miss_limit": CONSTRAINT_MISS_LIMIT,
        "metric_snapshot_interval": METRIC_SNAPSHOT_INTERVAL,
        "llm_agent_a": LLM_AGENT_A,
        "speech_pattern_key": DEFAULT_SPEECH_PATTERN,
        "speech_engine": SPEECH_ENGINE if SPEECH_ENGINE != "patterned" else "file",
        "tts_engine": SPEECH_TTS_ENGINE or ("file" if SPEECH_ENGINE == "patterned" else SPEECH_ENGINE),
        "asr_engine": SPEECH_ASR_ENGINE or ("file" if SPEECH_ENGINE == "patterned" else SPEECH_ENGINE),
        "speech_audio_dir": SPEECH_AUDIO_DIR,
        "speech_incoming_enabled": True if SPEECH_SCOPE == "none" and not SPEECH_INCOMING_ENABLED else SPEECH_INCOMING_ENABLED,
        "speech_outgoing_enabled": True if SPEECH_SCOPE == "none" and not SPEECH_OUTGOING_ENABLED else SPEECH_OUTGOING_ENABLED,
        "speech_playback_enabled": True if SPEECH_SCOPE == "none" and not SPEECH_PLAYBACK_ENABLED else SPEECH_PLAYBACK_ENABLED,
        "speech_realtime_enabled": True if SPEECH_SCOPE == "none" and not SPEECH_REALTIME_ENABLED else SPEECH_REALTIME_ENABLED,
        "speech_scope": "both" if SPEECH_SCOPE == "none" else SPEECH_SCOPE,
        "agent_a_words_per_minute": 165,
        "agent_b_words_per_minute": 175,
        "min_utterance_sec": 0.6,
        "max_utterance_sec": 3.5,
        "gui_enabled": GUI_ENABLED,
        "gui_mode": GUI_MODE,
        "protocol_log_dir": PROTOCOL_LOG_DIR,
    }


def select_run_config():
    """Show the startup configuration form, falling back to defaults when unavailable."""
    defaults = default_run_config()
    if not GUI_ENABLED:
        return defaults
    choices = {
        "test_case_keys": list(TEST_CASES),
        "persona_keys": list(PERSONAS),
        "agent_b_plugins": available_agent_b_plugin_keys(AGENT_B_PLUGIN),
        "speech_patterns": ["clean", "hesitant", "compressed", "noisy_station"],
        "speech_engines": ["patterned", "file"],
        "tts_engines": ["patterned", "file", "loopback"],
        "asr_engines": ["patterned", "file", "loopback"],
        "speech_scopes": ["both", "agent_a", "agent_b", "none"],
        "gui_modes": ["conversation", "full"],
    }
    try:
        from minillama.view.gui import StartupConfigDialog

        selected = StartupConfigDialog(choices, defaults).show()
    except Exception as exc:
        logging.warning("Startup configuration UI unavailable; using defaults: %s", exc)
        selected = defaults
    return selected


def run_gui_loop(ui_queue, scenario, gui_mode):
    """Run the Tk GUI in the current thread."""
    from minillama.view.gui import DialogWindow

    dialog = DialogWindow(ui_queue, scenario, minimal=gui_mode == "conversation")
    dialog.run()


def start_gui_thread(ui_queue, scenario, gui_mode=GUI_MODE, dialog_runner=run_gui_loop):
    """Start the optional GUI in an isolated thread and return the thread handle."""
    def _target():
        try:
            dialog_runner(ui_queue, scenario, gui_mode)
        except Exception:
            logging.exception("GUI thread stopped")

    thread = threading.Thread(
        target=_target,
        name="minillama-gui",
        daemon=False,
    )
    thread.start()
    return thread


def main():
    """Start one interactive dialog run."""
    run_config = select_run_config()
    if run_config is None:
        return

    scenario = get_test_case(run_config["test_case_key"]).with_persona(run_config.get("persona_key", DEFAULT_PERSONA)).scenario
    write_network_research_artifacts(
        scenario["start_time_min"],
        RESEARCH_LOG_DIR,
        picture_dir=NETWORK_PICTURE_DIR,
    )
    ui_queue = queue.Queue()
    session_logger = None if SESSION_LOG_PROFILE == "off" else SessionLogger(
        SESSION_NAME,
        SESSION_LOG_DIR,
        profile=SESSION_LOG_PROFILE,
    )
    event_queue = MonitoringEventQueue(ui_queue, session_logger)

    agent_b_config = AgentBPluginConfig(run_config["agent_b_plugin"])
    if not agent_b_config.needs_model and not run_config.get("llm_agent_a", LLM_AGENT_A):
        model_adapter = None
    else:
        from minillama.model.model_runtime import create_model_adapter

        with event_queue.segment("model.load", model_provider=MODEL_PROVIDER, model_name=MODEL):
            model_adapter = create_model_adapter()

    gui_thread = None
    if run_config.get("gui_enabled", True):
        gui_thread = start_gui_thread(ui_queue, scenario, run_config.get("gui_mode", GUI_MODE))

    conversation_worker(event_queue, model_adapter, run_config)
    if gui_thread is not None and gui_thread.is_alive():
        gui_thread.join()


if __name__ == "__main__":
    main()
