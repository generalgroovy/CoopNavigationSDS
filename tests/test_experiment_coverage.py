import csv
import json
from pathlib import Path
import tempfile

from coop_navigation_sds.ResultsAndArtifacts.coverage import _agent_model_combination_rows, _agent_model_html, _coverage_key, update_experiment_coverage
from coop_navigation_sds.Configuration.jobs import load_experiment_job
from coop_navigation_sds.ResultsAndArtifacts.coverage import _planned_rows


def _write_job(path):
    path.write_text(json.dumps({
        "schema_version": 1,
        "name": "coverage_test",
        "coverage_strategy": "full_factorial",
        "iterations": 1,
        "config": {
            "agent_a_type": "userlm",
            "paired_audio_text_runs": True,
        },
        "grid": {
            "test_cases": ["morning_peak_cross_city"],
            "personas": ["focused_commuter"],
            "agent_a_audio_personas": ["high_clarity_caller"],
            "agent_b_audio_personas": ["clear_operator"],
            "speech_patterns": ["clean"],
            "model_params": ["greedy"],
            "objective_modes": ["shortest_valid_route_with_constraints"],
            "tts_engines": ["piper"],
            "asr_engines": ["vosk"],
            "agent_b_models": ["TinyLlama/TinyLlama-1.1B-Chat-v1.0"],
        },
        "parameter_values": {
            "matrix_family": ["agent_b_llm_comparison_v1"],
            "experiment_platform": ["windows"],
            "asr_beam_size": [1],
            "agent_b_llm_size": ["small"],
            "agent_b_model_slot": ["small1"],
            "agent_b_model_role": ["primary"],
        },
    }, indent=2), encoding="utf-8")


def test_coverage_registry_indexes_only_finalized_standard_runs():
    project_root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(dir=project_root) as temporary:
        root = Path(temporary)
        jobs = root / "jobs"
        results = root / "results"
        jobs.mkdir()
        results.mkdir()
        job_path = jobs / "nested" / "coverage.job"
        job_path.parent.mkdir()
        _write_job(job_path)
        planned = list(_planned_rows(job_path))
        assert len(planned) == 2

        incomplete = results / "incomplete"
        incomplete.mkdir()
        (incomplete / "conditions.jsonl").write_text("{}\n", encoding="utf-8")

        completed = results / "agent_b" / "primary" / "small" / "completed"
        completed.mkdir(parents=True)
        audio_row = next(row for row in planned if row["run_type"] == "audio_variant")
        (completed / "conditions.jsonl").write_text(
            json.dumps({**audio_row, "task_success": True}) + "\n",
            encoding="utf-8",
        )
        (completed / "run_summary.json").write_text(json.dumps({
            "result_run_id": "completed",
            "result_scope": "batch",
            "condition_count": 1,
            "successful_condition_count": 1,
        }), encoding="utf-8")

        paths = update_experiment_coverage(results, job_roots=[jobs])
        summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
        rows = list(csv.DictReader(paths["conditions"].open(encoding="utf-8")))

        assert summary["planned_configuration_count"] == 2
        assert summary["completed_planned_configuration_count"] == 1
        assert summary["completed_run_count"] == 1
        assert {row["status"] for row in rows} == {"completed", "planned"}
        completed_row = next(row for row in rows if row["status"] == "completed")
        assert completed_row["configured_tts_engine"] == "piper"
        assert completed_row["configured_asr_engine"] == "vosk"
        assert completed_row["run_ids"] == "completed"
        assert paths["matrix"].is_file()
        assert paths["agent_model_matrix"].is_file()
        assert paths["agent_model_report"].is_file()
        assert paths["case_coverage"].is_file()
        case_rows = list(csv.DictReader(paths["case_coverage"].open(encoding="utf-8")))
        assert case_rows
        assert case_rows[0]["test_case_key"] == "morning_peak_cross_city"
        assert summary["case_coverage"]["treatment_count"] == 1
        agent_matrix = list(csv.DictReader(paths["agent_model_matrix"].open(encoding="utf-8")))
        assert len(agent_matrix) == 5
        assert {row["model_slot"] for row in agent_matrix} == {
            "small1", "small2", "medium1", "medium2", "large1"
        }
        assert "completed/planned" in paths["report"].read_text(encoding="utf-8")


def test_agent_model_matrix_separates_canonical_slots_and_active_control():
    coverage = [{
        "coverage_key": "one",
        "matrix_family": "agent_b_llm_comparison_v1",
        "agent_a_type": "userlm",
        "agent_b_model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "agent_b_llm_size": "small",
        "agent_b_model_slot": "small1",
        "agent_b_model_role": "primary",
        "planned": True,
        "completed_count": 1,
        "successful_count": 1,
    }]
    active = [{
        "result_run_id": "baseline",
        "agent_a_type": "tinyllama",
        "agent_b_model": "TinyLlama",
        "agent_b_llm_size": "small",
        "agent_b_model_role": "support_baseline",
        "matrix_family": "small_agent_b_speech_grid_v1",
        "planned_condition_count": 52,
        "observed_condition_count": 10,
        "failed_condition_count": 0,
    }]

    rows, controls = _agent_model_combination_rows(coverage, active)

    small_primary = next(row for row in rows if row["agent_a_type"] == "userlm" and row["model_slot"] == "small1")
    assert len(rows) == 5
    assert small_primary["status"] == "complete"
    assert small_primary["agent_b_models"] == "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    assert controls[0]["active_observed_condition_count"] == 10


def test_agent_model_matrix_reports_combined_memory_in_ascending_order():
    coverage = [{
        "matrix_family": "agent_b_llm_comparison_v1",
        "agent_a_type": "userlm",
        "agent_b_model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "agent_b_llm_size": "small",
        "agent_b_model_slot": "small1",
        "agent_b_model_role": "primary",
        "planned": True,
        "completed_count": 0,
        "successful_count": 0,
    }]

    rows, _controls = _agent_model_combination_rows(coverage, [])
    row = next(item for item in rows if item["agent_a_type"] == "userlm" and item["model_slot"] == "small1")

    assert row["agent_a_model"] == "microsoft/UserLM-8b"
    assert row["agent_a_approximate_memory_gb"] == 34.0
    assert row["agent_b_approximate_memory_gb"] == 4.5
    assert row["combined_approximate_memory_gb"] == 38.5

    combined = [
        item["combined_approximate_memory_gb"] for item in rows
        if item["combined_approximate_memory_gb"] is not None
    ]
    assert combined == sorted(combined)


def test_agent_model_html_sorts_slots_by_numeric_memory_and_hides_unavailable_viability():
    rows, controls = _agent_model_combination_rows([], [])
    for index, row in enumerate(rows):
        row["agent_b_models"] = {
            "small1": "small-one",
            "small2": "small-two",
            "medium1": "medium-one",
            "medium2": "medium-two",
            "large1": "large-twelve",
        }[row["model_slot"]]
        row["agent_b_approximate_memory_gb"] = {
            "small1": 3.0,
            "small2": 4.0,
            "medium1": 6.0,
            "medium2": 7.0,
            "large1": 12.0,
        }[row["model_slot"]]
        row["combined_approximate_memory_gb"] = row["agent_a_approximate_memory_gb"] + row["agent_b_approximate_memory_gb"]
        row.pop("system_viability", None)

    report = _agent_model_html(rows, controls)

    assert "large2" not in report
    assert "large-ten" not in report
    assert "large-twelve" in report
    assert "System viability" not in report


def test_coverage_key_normalizes_missing_platform_consistently():
    assert _coverage_key({"experiment_platform": None}) == _coverage_key({"experiment_platform": "unspecified"})


def test_coverage_key_distinguishes_case_seed_and_condition_identity():
    base = {
        "condition_id": "condition-a",
        "test_case_key": "morning_peak_cross_city",
        "network_seed": 11,
    }

    assert _coverage_key(base) != _coverage_key({**base, "condition_id": "condition-b"})
    assert _coverage_key(base) != _coverage_key({**base, "test_case_key": "midday_transfer"})
    assert _coverage_key(base) != _coverage_key({**base, "network_seed": 29})


def test_focused_agent_b_job_is_bounded_and_covers_declared_levels():
    root = Path(__file__).resolve().parents[1]
    path = root / "jobs" / "agent_b_llm" / "tinyllama_comparison" / "primary" / "01-small-llama3.2-1b.job"
    job = load_experiment_job(path)
    rows = list(_planned_rows(path))
    audio_rows = [row for row in rows if row["run_type"] == "audio_variant"]
    text_rows = [row for row in rows if row["run_type"] == "text_only"]

    assert job["config"]["agent_a_type"] == "tinyllama"
    assert job["config"]["model_name"] == "llama3.2:1b"
    assert job["config"]["num_turns"] == 20
    assert job["config"]["log_profile"] == "full"
    assert len(audio_rows) == 13
    assert len(audio_rows) == len(text_rows)
    assert {row["scenario_key"] for row in audio_rows} == {
        "morning_peak_cross_city", "midday_transfer", "late_event", "airport_connection"
    }
    assert {row["persona_key"] for row in audio_rows} == {
        "focused_commuter", "distracted_multitasker", "verbose_planner", "delay_sensitive_traveler"
    }
    assert {row["agent_a_audio_persona"] for row in audio_rows} == {
        "high_clarity_caller", "hesitant_caller"
    }
    assert {row["agent_b_audio_persona"] for row in audio_rows} == {
        "clear_operator", "degraded_operator"
    }
    assert {row["configured_asr_engine"] for row in audio_rows} == {
        "faster_whisper"
    }
    assert {row["asr_search_width"] for row in audio_rows} == {1, 6}
    assert {row["speech_pattern_key"] for row in audio_rows} == {"clean", "hesitant", "noisy_station"}
    assert {row["agent_b_llm_size"] for row in rows} == {"small"}
