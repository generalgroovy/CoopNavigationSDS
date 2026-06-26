import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

from coop_navigation_sds.Configuration.schema import (
    CONFIG_SCHEMA_VERSION,
    TRACE_SCHEMA_VERSION,
    RunArtifactPaths,
    sanitized_config,
)
from coop_navigation_sds.Configuration.component_catalog import startup_choices
from coop_navigation_sds.Configuration.pipeline import ComponentStatus
from coop_navigation_sds.Configuration.run_identity import naming_scheme_document, single_run_label
from coop_navigation_sds.Configuration.settings import load_run_settings, save_run_settings
from coop_navigation_sds.app import default_run_config, normalize_run_config, prepare_execution_run_config
from coop_navigation_sds.DialogManagement.result import DialogResult
from coop_navigation_sds.EvaluationMetrics.metrics import (
    DEFAULT_METRIC_CONFIG,
    failure_indicator_analysis,
)
from coop_navigation_sds.ResultsAndArtifacts.artifacts import (
    calculate_batch_metrics_from_inputs,
    calculate_metrics_from_inputs,
    write_batch_metric_inputs,
    write_metric_inputs,
)
from coop_navigation_sds.ResultsAndArtifacts.logging import StructuredEvent
from coop_navigation_sds.TransportNetwork.test_cases import get_test_case
from coop_navigation_sds.smoke import smoke_run_config


def test_readme_documents_complete_experiment_network_contract():
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")

    for heading in (
        "### Network Parameters",
        "### Structural Invariants",
        "### Route Representation",
        "### Fullness, Demand, and Delay",
        "### Access and Constraint Evaluation",
        "### Default Seed 42 Network",
        "### Standard Scenarios",
    ):
        assert heading in readme
    for required_value in (
        "36",
        "M1-M20",
        "Near-capacity threshold",
        "Default line-specific segment travel times",
        "Default station coordinates, public modes, transfer times, and demand districts",
        "`network_overview.json`",
        "`network_graph.svg`",
    ):
        assert required_value in readme


def test_settings_round_trip_excludes_credentials_and_transient_paths():
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "settings.json"
        save_run_settings({
            "model_profile": "custom",
            "model_api_key": "secret",
            "model_name": "local-model",
            "execution_run_dir": "private-run",
        }, path)
        document = json.loads(path.read_text(encoding="utf-8"))
        loaded = load_run_settings(path=path)

    assert document["schema_version"] == CONFIG_SCHEMA_VERSION
    assert "model_api_key" not in document["config"]
    assert "execution_run_dir" not in document["config"]
    assert loaded["model_name"] == "local-model"


def test_shared_artifact_paths_are_flat_and_stable():
    paths = RunArtifactPaths(Path("run"), "condition").as_dict()

    assert paths["metric_inputs"] == Path("run/metric_inputs.json")
    assert all(path.parent == Path("run") for path in paths.values())
    assert len(paths.values()) == len(set(paths.values()))


def test_structured_events_and_sanitized_config_have_versioned_contracts():
    event = StructuredEvent("system", "session", 1.0, "start").to_dict()
    config = sanitized_config({"model_api_key": "secret", "network_seed": 42})

    assert event["schema_version"] == TRACE_SCHEMA_VERSION
    assert config == {"model_api_key": "<redacted>", "network_seed": 42}


def test_metric_record_can_be_recalculated_from_persisted_raw_evidence():
    scenario = get_test_case("morning_peak_cross_city").scenario
    result = DialogResult(
        condition_id="replay",
        test_case_key="morning_peak_cross_city",
        persona_key="focused_commuter",
        scenario_key="morning_peak_cross_city",
        speech_pattern_key="clean",
        model_name="deterministic",
        conversation=[("Agent A", "I need a route."), ("Agent B", "Please clarify the stations.")],
        route=[],
        route_steps=[],
        route_valid=False,
        route_reaches_goal=False,
        route_correct=False,
        route_duration_min=None,
        runtime_sec=0.1,
        extra={
            "messages": 2,
            "agent_memories": {
                "Agent A": [{"speaker": "Agent A", "text": "I need a route."}],
                "Agent B": [{"speaker": "Agent A", "text": "I need a route."}],
            },
            "speech_turns": [],
            "timing_turns": [],
            "phase_timings": [],
            "nlu_turns": [],
            "runtime_events": [],
            "candidate_events": [],
            "resolved_scenario": scenario,
        },
    )
    with tempfile.TemporaryDirectory() as temporary:
        evidence_path = write_metric_inputs(result, scenario, Path(temporary) / "metric_inputs.json")
        record = calculate_metrics_from_inputs(evidence_path)
        batch_path = write_batch_metric_inputs([result], Path(temporary) / "batch_inputs.json")
        batch_records = calculate_batch_metrics_from_inputs(batch_path)
        document = json.loads(evidence_path.read_text(encoding="utf-8"))

    assert document["captured_before_metric_calculation"] is True
    assert document["schema_version"] == TRACE_SCHEMA_VERSION
    assert set(document["trace_collections"]["agent_memories"]) == {"Agent A", "Agent B"}
    assert record.condition_id == "replay"
    assert record.metric_families["whole_dialogue"]["trace_completeness_rate"] == 1.0
    assert record.metric_families["whole_dialogue"]["first_deviation_turn"] == 2
    assert record.metric_families["whole_dialogue"]["first_deviation_phase"] == "task_outcome"
    assert len(batch_records) == 1
    assert batch_records[0].condition_id == "replay"


def test_smoke_configuration_has_no_heavy_backend_requirement():
    config = smoke_run_config("temporary-results")

    assert config["agent_b_plugin"] == "simple"
    assert config["tts_engine"] == "file"
    assert config["asr_engine"] == "file"
    assert config["speech_playback_enabled"] is False
    assert "metric_config" not in config


def test_single_results_root_contains_the_run_and_all_runtime_audio():
    with tempfile.TemporaryDirectory() as temporary:
        config = default_run_config()
        config.update({"results_root": temporary, "agent_b_plugin": "simple"})
        prepared = prepare_execution_run_config(config)
        run_dir = Path(prepared["execution_run_dir"])

        assert run_dir.parent == Path(temporary)
        assert Path(prepared["speech_audio_dir"]) == run_dir
        assert "protocol_log_dir" not in prepared
        scheme = json.loads((Path(temporary) / "naming_scheme.json").read_text(encoding="utf-8"))
        assert scheme["codes"]["TL1"] == "TinyLlama 1.1B Chat language model"


def test_tinyllama_agent_a_selects_tinyllama_model_profile():
    config = normalize_run_config({
        **default_run_config(),
        "agent_a_type": "tinyllama",
        "agent_b_plugin": "simple",
        "model_profile": "custom",
        "model_name": "",
    })

    assert config["agent_a_type"] == "tinyllama"
    assert config["model_profile"] == "tinyllama_1b_transformers"
    assert config["model_name"] == "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def test_interactive_catalog_hides_file_controls_and_unavailable_components():
    def status(_kind, key, _config=None):
        return ComponentStatus(key, key not in {"coqui", "qwen3_asr"}, "test status")

    with patch(
        "coop_navigation_sds.Configuration.component_catalog.component_status",
        side_effect=status,
    ):
        choices = startup_choices(config=default_run_config(), operational_only=True)

    assert "file" not in choices["tts_engines"]
    assert "file" not in choices["asr_engines"]
    assert "coqui" not in choices["tts_engines"]
    assert "qwen3_asr" not in choices["asr_engines"]


def test_interactive_catalog_preserves_current_filtered_combobox_values():
    def status(_kind, _key, _config=None):
        return ComponentStatus("blocked", False, "test status")

    config = {
        **default_run_config(),
        "agent_b_plugin": "custom.module:factory",
        "model_profile": "custom",
        "model_provider": "ollama",
        "tts_engine": "chattts",
        "asr_engine": "vosk",
    }

    with patch(
        "coop_navigation_sds.Configuration.component_catalog.component_status",
        side_effect=status,
    ):
        choices = startup_choices(config=config, operational_only=True)

    assert "custom.module:factory" in choices["agent_b_plugins"]
    assert "custom" in choices["model_profiles"]
    assert "ollama" in choices["model_providers"]
    assert "chattts" in choices["tts_engines"]
    assert "vosk" in choices["asr_engines"]


def test_failure_thresholds_exclude_outcome_metrics_and_require_both_classes():
    records = []
    for index in range(6):
        success = index < 3
        records.append(SimpleNamespace(
            success=success,
            metric_families={
                "asr": {"wer": 0.1 if success else 0.8},
                "task_outcome": {"completion": float(success)},
            },
        ))

    report = failure_indicator_analysis(records)

    assert report["status"] == "available"
    assert report["indicators"][0]["metric"] == "asr_wer"
    assert all(not item["metric"].startswith("task_outcome_") for item in report["indicators"])


def test_run_code_is_compact_but_encodes_primary_condition_variables():
    config = default_run_config()
    config.update({
        "test_case_key": "morning_peak_cross_city",
        "persona_key": "focused_commuter",
        "agent_a_type": "userlm",
        "model_profile": "tinyllama_1b_transformers",
        "tts_engine": "piper",
        "asr_engine": "faster_whisper",
        "network_seed": 42,
    })

    label = single_run_label(config)

    assert label.startswith("R-")
    assert "ULM" in label and "TL1" in label and "PIP" in label and "FWH" in label
    assert len(label) < 50
    assert naming_scheme_document()["codes"]["TL1"] == "TinyLlama 1.1B Chat language model"
