# MiniLlama

MiniLlama is a transit-route dialog sandbox with a fullscreen `tkinter`/`ttk` GUI, batch evaluation, and session logging.

## Layout

### `minillama/`
Package root and module entrypoint.

- `__init__.py`: package marker.
- `__main__.py`: `python -m minillama` entrypoint.

### `minillama/agent_a/`
Agent A prompt logic and persona handling.

- `__init__.py`: package marker.
- `config.py`: Agent A defaults, personas, and prompt text constants.
- `agents.py`: Agent A prompt builders, cleanup, and fallback replies.
- `agent_a_responder.py`: template and LLM-backed Agent A responders.
- `personas.py`: persona lookup and preference formatting helpers.
- `prompt_data.py`: shared prompt context for Agent A and Agent B.

### `minillama/agent_b/`
Agent B prompting, plugins, and simulated speech transport.

- `__init__.py`: package marker.
- `config.py`: Agent B defaults and speech-pattern settings.
- `plugin_registry.py`: built-in Agent B plugins, custom plugin loading, and plugin config.
- `agent_b_plugins.py`: compatibility imports for plugin code.
- `pipeline.py`: Agent B verbal transformation pipeline.
- `speech_io.py`: loopback and patterned speech transport.

### `minillama/controller/`
Runtime orchestration, GUI startup, batch execution, and logging.

- `__init__.py`: package marker.
- `config.py`: controller defaults and logging profile.
- `dialog_manager.py`: one-dialog controller and route tracking.
- `dialog_result.py`: dialog result data model and null queue.
- `events.py`: structured logging event models.
- `main.py`: interactive startup flow.
- `route_memory.py`: route deduplication memory.
- `run_experiments.py`: batch CLI entrypoint.
- `runner.py`: batch grid execution and CSV export.
- `session_logging.py`: configurable structured logging.

### `minillama/evaluation/`
Metrics and route interpretation.

- `__init__.py`: package marker.
- `config.py`: evaluation weights and scoring constants.
- `metrics.py`: automatic metric computation.
- `route_interpreter.py`: spoken-route parsing and scoring.

### `minillama/model/`
Transit network model and model-runtime helpers.

- `__init__.py`: package marker.
- `config.py`: model, network, and shared runtime settings.
- `metro_data.py`: generated network, crowding, and prompt text helpers.
- `model_adapters.py`: Hugging Face and OpenAI-compatible adapters.
- `model_runtime.py`: model/tokenizer loading and adapter creation.
- `network_overview.py`: complete line and station data rows for the GUI.
- `route_planner.py`: route validation, timing, and schedule helpers.
- `station_names.py`: station name generation.

### `minillama/test_cases/`
Scenarios and standardized evaluation cases.

- `__init__.py`: package marker.
- `config.py`: scenario and test-case defaults.
- `scenarios.py`: scenario construction and lookup.
- `test_cases.py`: standardized test-case binding and opening utterances.

### `minillama/view/`
GUI rendering and view-layer layout.

- `__init__.py`: package marker.
- `config.py`: GUI layout, theme, and map constants.
- `gui.py`: interactive dashboard and transit map view.

### `tests/`
Regression tests for config facades, dialog monitoring, and session logging.

- `test_config_facades.py`: import and facade consistency checks.
- `test_dialog_manager_monitoring.py`: dialog telemetry coverage.
- `test_session_logging.py`: logging behavior checks.

## Run

Interactive GUI:

```powershell
.venv\Scripts\python.exe -m minillama
```

The GUI opens with a compact run-configuration form. Agent B defaults to the MiniLlama/model-backed assistant and can optionally switch to the built-in deterministic planner or a custom `package.module:factory` plugin. The default run is text-only: incoming ASR and outgoing TTS stages are off and the speech scope is `none`. The form can still enable speech simulation for Agent A, Agent B, both agents, or neither agent.

Batch metrics:

```powershell
.venv\Scripts\python.exe -m minillama.controller.run_experiments
```

Batch runs use the configured Agent B model adapter and the `--model-params` sweep maps to real generation presets. `--agent-b-plugin` is optional and defaults to `minillama`; use `simple` for the deterministic planner, `llm` as a compatibility alias, or `package.module:factory` for a custom plugin. `MINILLAMA_AGENT_B_PLUGIN` sets the same default for GUI and batch runs. Batch speech defaults are also text-only: `--speech-patterns clean --speech-incoming false --speech-outgoing false --speech-scope none`.

Useful speech-pipeline batch controls:

```powershell
.venv\Scripts\python.exe -m minillama.controller.run_experiments --agent-b-plugin simple --speech-patterns clean,hesitant --speech-incoming true --speech-outgoing true --speech-scope both
```

- `--speech-incoming true|false`: include the incoming ASR/transcript stage.
- `--speech-outgoing true|false`: include the outgoing TTS/verbalization stage.
- `--speech-scope both|agent_a|agent_b|none`: choose whose turns pass through speech stages.
- `MINILLAMA_SPEECH_INCOMING`, `MINILLAMA_SPEECH_OUTGOING`, and `MINILLAMA_SPEECH_SCOPE` set the same defaults for GUI and batch runs.
- `--log-profile off|startup|runtime|full`: optional structured JSONL logging for batch audits. The default is `off` for low runtime overhead.
- `--log-dir PATH`: destination for optional batch logs.
- `--progress`: print each completed condition id during long batch runs.

The evaluation report now exports a staged metric stack aligned with speech-dialog analysis. Audio ingress, VAD, and diarization fields are present but `None` in this text-only setup; ASR, SLU/DST proxies, policy/tool metrics, NLG, runtime, end-to-end, and post-hoc aggregates are computed from the dialog trace.

Speech turns now keep separate generated, outgoing, and incoming text traces. The GUI and CSV metrics report ASR WER, TTS text-change rate, station precision/recall, and incoming/outgoing speech-stage enabled rates without duplicating those details in the conversation window.

Network data is displayed in its own GUI card with complete line rows, station rows, headways, current fullness, neighbors, route sequences, and segment travel times. The map remains separate so the data table can be scanned without depending on the drawing.

Logging defaults to `off`. Set `MINILLAMA_SESSION_LOG_PROFILE` to `startup`, `runtime`, or `full` to compare overhead.
